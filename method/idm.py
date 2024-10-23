import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from method.dc_base import DC_BASE, TrainTransform
from model import get_model

class IDM(DC_BASE):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.net_generate_interval = args.net_generate_interval
        self.net_push_num = args.net_push_num
        self.net_num = args.net_num
        self.net_begin = args.net_begin
        self.net_end = args.net_end
        self.ij_selection = args.ij_selection
        self.train_net_num = args.train_net_num
        self.aug_num = args.aug_num
        self.fetch_net_num = args.fetch_net_num
        self.ce_weight = args.ce_weight

    def condense(self):
        # del self.net
        # del self.optimizer

        self.net_list = []
        self.optimizer_list = []
        self.acc_meters = []

        for net_index in range(3):
            net, _ = get_model(self.model, self.num_classes, self.channel)
            net = net.to(self.device)

            optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)
            optimizer_net.zero_grad()

            self.net_list.append(net)
            self.optimizer_list.append(optimizer_net)
            self.acc_meters.append({'sum':0, 'num':0})

        self.image_optimizer = torch.optim.SGD((self.synthetic_images,), lr=self.lr_img, momentum=0.5)

        set_timer = time.time()
        for it in range(self.iteration):
            self.logger.info(f"Iteration {it}")
            if it % 100 == 0:
                # initialize the accraucy meters
                for acc_index, _ in enumerate(self.acc_meters):
                    self.acc_meters[acc_index]['sum'] = 0
                    self.acc_meters[acc_index]['num'] = 0
            if it % 100 == 0 or it == self.iteration - 1:
                # save the images
                if self._is_main_process():
                    self.visualize(self.path, it)
            if it % self.net_generate_interval == 0:
                for _ in range(self.net_push_num):
                    if len(self.net_list) == self.net_num:
                        self.net_list.pop(0)
                        self.optimizer_list.pop(0)
                        self.acc_meters.pop(0)
                    net, _ = get_model(self.model, self.num_classes, self.channel)
                    net = net.cuda()
                    net.train()
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)
                    optimizer_net.zero_grad()

                    self.net_list.append(net)
                    self.optimizer_list.append(optimizer_net)
                    self.acc_meters.append({'sum':0, 'num':0})

            net_index_list = torch.randperm(len(self.net_list))[:self.train_net_num]
            for o in range(self.o_iter):
                train_acc = 0
                for net_ind in net_index_list:
                    net = self.net_list[net_ind]
                    net.eval()
                    optimizer = self.optimizer_list[net_ind]
                    net_acc = self.acc_meters[net_ind]

                self.image_optimizer.zero_grad()
                for c in range(self.global_rank, self.num_classes, self.world_size):
                    loss_c = 0
                    real_images, real_labels = self.get_train_images([c], self.batchsize, seed=self.dist_seed())
                    real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
                    syn_images, syn_labels = self.get_synthetic_images([c], self.n_ipc, seed=self.dist_seed())
                    syn_images, syn_labels = syn_images.to(self.device), syn_labels.to(self.device)

                    if self.dsa:
                        # augment images
                        seed = self.syn_seed()
                        real_images, real_labels = TrainTransform(self)((real_images, real_labels), seed=seed)
                        syn_images, syn_labels = TrainTransform(self)((syn_images, syn_labels), seed=seed)
                    real_images = F.interpolate(real_images, size=self.inp_size, mode='bilinear')
                    syn_images = F.interpolate(syn_images, size=self.inp_size, mode='bilinear')

                    real_outputs, real_features = net._compiled_forward(real_images)
                    syn_outputs, syn_features = net._compiled_forward(syn_images)

                    loss_c = ((real_features[-1].mean(0) - syn_features[-1].mean(0))**2).sum()

                    syn_ce_loss = 0
                    weight_i = net_acc['sum'] / net_acc['num'] if net_acc['num'] != 0 else 0
                    syn_ce_loss += (F.cross_entropy(syn_outputs, syn_labels.repeat(self.aug_num, 1)) * weight_i * 100.0)
                    loss_c += (syn_ce_loss * self.ce_weight)
                    loss_c.backward()
                    self.logger.debug(f"weight_i: {weight_i}, syn_ce_loss: {syn_ce_loss.item()}, loss_c: {loss_c.item()}")
                self.step(self.image_optimizer)
                self.logger.info(f"train_acc: {train_acc}")
                    
                shuffled_net_index = torch.randperm(len(self.net_list))
                for j in range(min(self.fetch_net_num, len(shuffled_net_index))):
                    train_net_ind = shuffled_net_index[j]
                    net = self.net_list[train_net_ind]
                    optimizer = self.optimizer_list[train_net_ind]
                    net.train()
                    for i in range(self.i_iter):
                        images, labels = self.get_train_images(range(self.num_classes), self.batchsize)
                        images, labels = images.to(self.device), labels.to(self.device)
                        if self.dsa:
                            images, labels = TrainTransform(self)((images, labels), seed=seed)
                        images = F.interpolate(images, size=self.inp_size, mode='bilinear')
                        outputs, _ = net._compiled_forward(images)
                        self.acc_meters[train_net_ind]['sum'] += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().float().item()
                        self.acc_meters[train_net_ind]['num'] += len(labels)
                        loss = F.cross_entropy(outputs, labels)
                        optimizer.zero_grad()
                        loss.backward()
                        self.step(optimizer)
                    if self.distributed:
                        tmp_tensor = torch.tensor(self.acc_meters[train_net_ind]['sum']).to(self.device)
                        dist.all_reduce(tmp_tensor, op=dist.ReduceOp.SUM)
                        self.acc_meters[train_net_ind]['sum'] = tmp_tensor.item()
                        tmp_tensor = torch.tensor(self.acc_meters[train_net_ind]['num']).to(self.device)
                        dist.all_reduce(tmp_tensor, op=dist.ReduceOp.SUM)
                        self.acc_meters[train_net_ind]['num'] = tmp_tensor.item()
                    self.logger.debug(f"train_net_ind: {train_net_ind} train_acc: {self.acc_meters[train_net_ind]['sum'] / self.acc_meters[train_net_ind]['num']}")
                past_iter = it * self.o_iter + o + 1
                left_iter = self.o_iter * self.iteration - past_iter
                eta = (time.time() - set_timer) / past_iter * left_iter
                eta = f"{int(eta) // 3600}:{int(eta) % 3600 // 60}:{int(eta) % 60}s.{int((eta - int(eta)) * 1000)}"
                self.logger.info(f"train_net_ind: {train_net_ind} train_acc: {self.acc_meters[train_net_ind]['sum'] / self.acc_meters[train_net_ind]['num']}, ETA: {eta}")
            self.start_iteration = it + 1
            if self._is_main_process():
                self.save_img(self.path)
                self.visualize(self.path, it)