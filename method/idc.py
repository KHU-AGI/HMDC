import sys
import time
import torch
import torch.nn.functional as F
from method.dc_base import DC_BASE, IDCTransform
from utils.thread_with_return_value import Thread_With_Return_Value
from model import get_model

class IDC(DC_BASE):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.pt_from = args.pt_from
        self.fix_iter = args.fix_iter
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.pt_from = args.pt_from
        self.early = args.early
        self.match = args.match
        self.metric = args.metric
        self.n_data = args.n_data
        self.bias = args.bias
        self.fc = args.fc


    def condense(self):
        self.load_img(self.path)
        if self.match == 'grad':
            if 'ft' not in self.model.lower() and self.model.lower() != 'convnet':
                self.model = self.model + '_ft'
                self.net, self.inp_size = get_model(self.model, self.num_classes, self.channel)
                self.net = self.net.to(self.device)
                params = []
                for n, p in self.net.named_parameters():
                    if 'head' in n or '.fc.' in n:
                        params.append(p)
                self.optimizer = torch.optim.SGD(params=params, lr=self.lr_net)
        self.visualize(self.path, 0)

        if self.pt_from >= 0:
            self.test_model_accuracy()
            pt_dict = self.net.state_dict()
        time_start = time.time()
        for it in range(self.start_iteration, self.iteration):
            if self.pt_from >= 0:
                # pretrain the network with real data
                self.net.load_state_dict(pt_dict)
            if it % self.fix_iter == 0:
                self.net.reset_parameters()
                self.net.train()
                self.optimizer = torch.optim.SGD(self.net.parameters(),
                                                 lr=self.lr_net, 
                                                 momentum=self.momentum,
                                                 weight_decay=self.weight_decay)
                if self.early > 0:
                    for _ in range(self.early):
                        for step, (images, labels) in self.get_train_epoch(torch.arange(self.num_classes)):
                            images, labels = self.get_train_images(torch.arange(self.num_classes), self.batchsize, seed=self.dist_seed())
                            images, labels = images.to(self.device), labels.to(self.device)
                            if self.dsa:
                                images, labels = self.diff_augment(images, labels, self.dsa_strategy_net)
                            if self.mixup_net == 'cut':
                                images, targets = self.cutmix(images, labels)
                            else:
                                targets = F.one_hot(labels, self.num_classes).float()
                            images = F.interpolate(images, size=self.inp_size, mode='bilinear')
                            outputs, _ = self.net(images)
                            loss = self.net.loss_fn(outputs, targets)
                            acc = (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().float() / len(labels)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.step(self.optimizer)

            for o in range(self.o_iter):
                loss_total = 0
                # Update synset
                for c in range(self.global_rank, self.num_classes, self.world_size):
                    real_images, real_labels = self.get_train_images([c], self.batchsize, seed=self.dist_seed())
                    syn_images, syn_labels = self.get_synthetic_images([c], self.batchsize, seed=self.dist_seed())
                    real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
                    syn_images, syn_labels = syn_images.to(self.device), syn_labels.to(self.device)
                    if self.dsa:
                        seed = self.syn_seed()
                        syn_images, syn_labels = IDCTransform(self)((syn_images, syn_labels), seed=seed)
                        real_images, real_labels = IDCTransform(self)((real_images, real_labels), seed=seed)
                    _t1 = Thread_With_Return_Value(target=lambda : self.net(F.interpolate(real_images, size=self.inp_size, mode='bilinear', align_corners=True)))
                    _t2 = Thread_With_Return_Value(target=lambda : self.net(F.interpolate(syn_images, size=self.inp_size, mode='bilinear', align_corners=True)))
                    _t1.start(); _t2.start();
                    real_outputs, real_features = _t1.join();
                    syn_outputs, syn_features = _t2.join();
                    loss = 0
                    if self.match == 'feat':
                        for f1, f2 in zip(real_features, syn_features):
                            loss += self.match_fn(f1.mean(0), f2.mean(0))
                    if self.match == 'grad':
                        _t1 = Thread_With_Return_Value(target=lambda : [g.detach() if g is not None else None for g in torch.autograd.grad(self.net.loss_fn(real_outputs, real_labels), self.net.parameters(), allow_unused=True, retain_graph=True)])
                        _t2 = Thread_With_Return_Value(target=lambda : torch.autograd.grad(self.net.loss_fn(syn_outputs, syn_labels), self.net.parameters(), allow_unused=True, retain_graph=True, create_graph=True))
                        _t1.start(); _t2.start();
                        g_real = _t1.join()
                        g_syn = _t2.join()
                        for i in range(len(g_real)):
                            if g_real[i] is None or g_syn[i] is None: continue
                            if len(g_real[i].shape) == 1 and not self.bias: continue
                            if len(g_real[i].shape) == 2 and not self.fc: continue
                            loss += self.match_fn(g_real[i], g_syn[i])
                    loss_total += loss.item()
                    loss.backward()
                    self.logger.debug(f'  Iteration {it}, class {c}, loss {loss.item():.4f}')
                self.step(self.image_optimizer)
                self.image_optimizer.zero_grad()
                self.logger.info(f'  Iteration {it}, Outter{o}, loss {loss_total/self.num_classes:.4f}')
                if self.n_data > 0:
                    for _ in range(self.i_iter):
                        n_data = 0
                        train_epoch = self.get_train_epoch(torch.arange(self.num_classes), transform=IDCTransform(self))
                        while n_data < self.n_data:
                            images, labels = next(train_epoch)
                            images, labels = images.to(self.device), labels.to(self.device)
                            outputs, _ = self.net._compiled_forward(images)
                            loss = self.net.loss_fn(outputs, labels)
                            acc = (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().float() / len(labels)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.step(self.optimizer)
                            n_data += len(labels)
                past_iter = (it - self.start_iteration) * self.o_iter + o + 1
                left_iter = self.iteration * self.o_iter - past_iter
                eta = (time.time() - time_start) / past_iter * left_iter
                eta = f"{int(eta) // 3600}:{int(eta) % 3600 // 60}:{int(eta) % 60}s.{int((eta - int(eta)) * 1000)}"
                self.logger.info(f'Iteration {it}, Outter{o}, ETA {eta}')
            self.start_iteration = it+1
            if self._is_main_process():
                self.save_img(self.path)
                self.visualize(self.path, it+1)
            sys.stdout.flush()

    def match_fn(self, x1, x2):
        if self.metric == 'l1':
            return F.l1_loss(x1, x2, reduction='sum')
        elif self.metric == 'mse':
            return F.mse_loss(x1, x2, reduction='sum')
        elif self.metric == 'l1_mean':
            return F.l1_loss(x1, x2, reduction='mean')
        elif self.metric == 'cos':
            return 1 - F.cosine_similarity(x1, x2, dim=0).sum()
        else:
            raise NotImplementedError

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(torch.tensor(1. - lam))
        cut_w = torch.tensor(W * cut_rat)
        cut_h = torch.tensor(H * cut_rat)

        # uniform
        cx = torch.randint(W, (1,))
        cy = torch.randint(H, (1,))

        bbx1 = torch.clip(cx - cut_w // 2, 0, W).long()
        bby1 = torch.clip(cy - cut_h // 2, 0, H).long()
        bbx2 = torch.clip(cx + cut_w // 2, 0, W).long()
        bby2 = torch.clip(cy + cut_h // 2, 0, H).long()

        return bbx1, bby1, bbx2, bby2