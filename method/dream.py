import gc
import sys
import time
import asyncio
import torch
import torch.nn.functional as F
from torch.cuda import mem_get_info
from model import get_model
from method.dc_base import DC_BASE, IDCTransform, TestTransform
from utils.kmeans import KMeans
from utils.diff_augment import Transform
from utils.thread_with_return_value import Thread_With_Return_Value
from utils.clustering_sample_selection import Clustering_Sample_Selection as CSS

class DREAM(DC_BASE):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.pt_from = args.pt_from
        self.mixup = args.mixup
        self.mixup_net = args.mixup_net
        self.fix_iter = args.fix_iter
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.pt_from = args.pt_from
        self.early = args.early
        self.match = args.match
        self.metric = args.metric
        self.n_data = args.n_data
        self.beta = args.beta

        self.bias = args.bias
        self.fc = args.fc
        self.interval = args.interval
            
        if self.mixup == 'cut':
            self.dsa_strategy = 'color_crop' if self.dsa else None
        else:
            self.dsa_strategy = 'color_crop_cutout' if self.dsa else None
        if self.mixup_net == 'cut':
            self.dsa_strategy_net = 'color_crop' if self.dsa else None
        else:
            self.dsa_strategy_net = 'color_crop_cutout' if self.dsa else None
        # self.dsa_strategy_net = "color_crop_cutout_flip_scale_rotate"

    def init_synthetic_images(self):
        self.synthetic_images = torch.randn(self.num_classes, self.n_ipc, self.channel, *(self.im_size), device=self.device)
        self.synthetic_labels = torch.arange(self.num_classes, device=self.device).unsqueeze(1).repeat(1,self.n_ipc)
        if 'aug' in self.init:
            if 'real' in self.init:
                for c in range(self.num_classes):
                    image, label = self.get_train_images(torch.tensor([c]), self.n_ipc * self.factor ** 2)
                    image, label = self.pack(image, label, self.factor)
                    self.synthetic_images[c] = image.to(self.device)
                    self.synthetic_labels[c] = label.to(self.device)
            if 'kmeans' in self.init:
                for c in range(self.num_classes):
                    imgs = self.train_images[c]
                    imgs = F.interpolate(imgs, size=self.inp_size, mode='bilinear', align_corners=True)
                    imgs = imgs.to(self.device)
                    q_idxs, _ = CSS(imgs, self.net._compiled_embed).query(self.n_ipc * self.factor ** 2)
                    imgs, _ = self.pack(imgs[q_idxs], torch.tensor([c]*len(q_idxs)), self.factor)
                    self.synthetic_images[c] = imgs
        else:
            if 'real' in self.init:
                for c in range(self.num_classes):
                    image, label = self.get_train_images(torch.tensor([c]), self.n_ipc)
                    self.synthetic_images[c] = image.to(self.device)
                    self.synthetic_labels[c] = label.to(self.device)
            if 'kmeans' in self.init:
                for c in range(self.num_classes):
                    imgs = self.train_images[c]
                    imgs = F.interpolate(imgs, size=self.inp_size, mode='bilinear', align_corners=True)
                    imgs = imgs.to(self.device)
                    q_idxs, _ = CSS(imgs, self.net._compiled_embed).query(self.n_ipc)
                    self.synthetic_images[c] = imgs[q_idxs].detach().clone()
        self.synthetic_images.requires_grad = True
        self.logger.info(f'Initialization Done with {self.init}!')

    def condense(self):
        asyncio.run(self._condense())

    async def _condense(self):
        if self.num_classes % self.world_size != 0:
            self.logger.critical(f'Number of classes should be divisible by world size! {self.num_classes} % {self.world_size} != 0');exit(1);
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
        query_list = torch.zeros(self.num_classes, self.batchsize, dtype=torch.long)

        async def _net_forward(net, images):
            return net(images)
        
        async def _net_compiled_embed(net, images):
            return net._compiled_embed(images)
            
        async def _net_loss(net, outputs, targets):
            return net.loss_fn(outputs, targets)
        
        async def _grad(parameters, loss):
            return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)

        async def _grad_with_graph(parameters, loss):
            return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True, create_graph=True)

        async def _match_fn(x1, x2):
            return self.match_fn(x1, x2)

        async def _gradient_mse(grad_list_1, grad_list_2):
            return sum(await asyncio.gather(*[_match_fn(g1, g2) for g1, g2 in zip(grad_list_1, grad_list_2) \
                                              if g1 is not None and g2 is not None and not (len(g1.shape) == 1 and self.bias or len(g1.shape) == 2 and self.fc)]))

        if self.pt_from >= 0:
            self.test_model_accuracy()
            pt_dict = self.net.state_dict()
        time_start = time.time()
        where_i_start = self.start_iteration
        for it in range(self.start_iteration, self.iteration):
            if self.pt_from >= 0:
                # pretrain the network with real data
                self.net.load_state_dict(pt_dict)
            if it % self.fix_iter == 0:
                self.logger.info(f'start iteration {it}')
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
                            if self.dsa:
                                seed = self.syn_seed()
                                images, targets = IDCTransform(self)(images, labels, self.dsa_strategy_net, seed)
                            outputs, _ = self.net._compiled_forward(images)
                            loss = self.net.loss_fn(outputs, targets)
                            acc = (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().float() / len(labels)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.step(self.optimizer)
            loss_total = 0
            for o in range(self.o_iter):
                # Update synset
                for c in range(self.global_rank, self.num_classes, self.world_size):
                    self.logger.info(f'Class {c} / {self.num_classes}')
                    if o % self.interval == 0:
                        with torch.no_grad():
                            while True:
                                embeddings = []
                                try:
                                    length = 0
                                    for i, (image, label) in enumerate(self.get_train_epoch([c], batchsize=self.batchsize, transform=TestTransform(self), random=False, distributed=False)):
                                        length += len(image)
                                        image, label = image.to(self.device), label.to(self.device)
                                        image = F.interpolate(image, size=self.inp_size, mode='bilinear')
                                        embeddings.append(_net_compiled_embed(self.net, image))
                                    embeddings = await asyncio.gather(*embeddings)
                                    embeddings = torch.cat(embeddings, dim=0)
                                    break;
                                except RuntimeError as e: self.logger.error(f'RuntimeError: {e}, retrying...'); continue;
                            embeddings = embeddings.view(embeddings.shape[0], -1)
                            kmeans = KMeans(n_clusters=self.batchsize, max_iter=100, batchsize=embeddings.shape[0], mode='euclidean', init='kmeans++', seed=self.syn_seed())
                            pred = kmeans.fit_predict(embeddings)
                            centroids = kmeans.get_centroids()
                            dist = kmeans.compute_distance_matrix(embeddings, centroids)
                            q_idxs = torch.argmin(dist, dim=0)#[:self.batchsize]
                            query_list[c] = q_idxs.detach().cpu()
                    real_images = self.train_images[c][query_list[c]].to(self.device)
                    real_labels = self.train_labels[c][query_list[c]].to(self.device)
                    syn_images, syn_labels = self.get_synthetic_images([c])

                    real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
                    syn_images, syn_labels = syn_images.to(self.device), syn_labels.to(self.device)

                    if self.dsa:
                        seed = self.syn_seed()
                        real_images, real_labels = IDCTransform(self)((real_images, real_labels), seed=seed)
                        syn_images, syn_labels = IDCTransform(self)((syn_images, syn_labels), seed=seed)
                        # (syn_images, syn_labels), (real_images, real_labels) = \
                        #     await asyncio.gather(_apply_transform((syn_images, syn_labels), IDCTransform(self), seed=seed),
                        #                          _apply_transform((real_images, real_labels), IDCTransform(self), seed=seed))
                    real = _net_forward(self.net, real_images)
                    syn = _net_forward(self.net, syn_images)
                    (real_outputs, real_features), (syn_outputs, syn_features) = await asyncio.gather(real, syn)

                    loss = 0
                    if self.match == 'feat':
                        for f1, f2 in zip(real_features, syn_features):
                            loss += self.match_fn(f1.sum(0), f2.sum(0))
                    if self.match == 'grad':
                        real_loss, syn_loss = await asyncio.gather(
                            _net_loss(self.net, real_outputs, real_labels),
                            _net_loss(self.net, syn_outputs, syn_labels))
                        g_real, g_syn = await asyncio.gather(
                            _grad(self.net.parameters(), real_loss),
                            _grad_with_graph(self.net.parameters(), syn_loss))
                        loss = await _gradient_mse(g_real, g_syn)
                    loss_total += loss.item()
                    loss.backward()
                    self.logger.debug(f'  Iteration {it}, class {c}, loss {loss.item():.4f}')
                self.step(self.image_optimizer)
                self.image_optimizer.zero_grad()
                self.logger.info(f'  Iteration {it}, Outter{o}, loss {loss_total/self.num_classes:.4f}')

                if self.n_data > 0:
                    train_epoch = iter(self.get_train_epoch(torch.arange(self.num_classes), transform=IDCTransform(self)))
                    for _ in range(self.i_iter):
                        n_data = 0
                        for i, (images, targets) in enumerate(train_epoch):
                            images, targets = images.to(self.device), targets.to(self.device)
                            images = F.interpolate(images, size=self.inp_size, mode='bilinear')
                            outputs, _ = self.net._compiled_forward(images)
                            loss = self.net.loss_fn(outputs, targets)
                            acc = (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().float() / len(targets)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.step(self.optimizer)
                            n_data += self.batchsize
                            if n_data >= self.n_data: break;
                past_iter = (it - where_i_start) * self.o_iter + o + 1
                left_iter = self.iteration * self.o_iter - past_iter
                eta = (time.time() - time_start) / past_iter * left_iter
                eta = f"{int(eta) // 3600}:{int(eta) % 3600 // 60:02d}:{int(eta) % 60:02d}s.{int((eta - int(eta)) * 1000):03d}"
                self.logger.info(f'Iteration {it}, Outter{o}, loss {loss_total/self.num_classes*self.world_size:.4f},  ETA {eta}, Time/iter {(time.time() - time_start) / past_iter:.2f}s')
            self.start_iteration = it+1
            gc.collect()
            if self._is_main_process():
                self.save_img(self.path)
                self.visualize(self.path, it+1)

    def match_fn(self, x1, x2):
        if self.metric == 'l1':
            return F.l1_loss(x1, x2, reduction='sum')
        elif self.metric == 'mse':
            return F.mse_loss(x1, x2, reduction='sum')
        elif self.metric == 'l1_mean':
            return F.l1_loss(x1, x2, reduction='mean')
        elif self.metric == 'cos':
            return F.cosine_similarity(x1, x2, dim=0).sum()
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
    
    def diff_augment(self, images, labels, strategy='color_crop_cutout_flip_scale_rotate', seed=None):
        strategy = strategy.split('_')
        is_flip = 'flip' in strategy
        is_color = 'color' in strategy
        is_cutout = 'cutout' in strategy
        if is_flip : strategy.remove('flip')
        if is_color : strategy.remove('color')
        if is_cutout : strategy.remove('cutout')
        strategy = '_'.join(strategy)
        if is_flip:
            images = Transform(images, 'flip', seed=self.dist_seed() if seed is None else seed, param=self.dsa_param)
        if is_color:
            images = Transform(images, 'color', seed=self.dist_seed() if seed is None else seed, param=self.dsa_param)
        auges_images = []
        for aug in range(self.dsa_iter):
            auges_images.append(
                Transform(
                    images, strategy,
                    seed=self.dist_seed() if seed is None else seed,
                    param=self.dsa_param
                ))
        images = torch.cat(auges_images)
        labels = labels.repeat(self.dsa_iter)
        if is_cutout:
            images = Transform(images, 'cutout', seed=self.dist_seed() if seed is None else seed, param=self.dsa_param)
        return images, labels