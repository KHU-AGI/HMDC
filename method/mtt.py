
import os
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
from model import get_model
from method.dc_base import DC_BASE

class MTT(DC_BASE):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.max_start_iteration = args.max_start_iteration
        self.num_experts = args.num_experts
        self.experts_epoch = args.experts_epoch

    def condense(self):
        for i in range(self.num_experts):
            if not os.path.exists(os.path.join(self.path, "timestamp_{}.pt".format(i))):
                self.build_teacher(i)
                break
        self.condense_images()
        return
    
    def build_teacher(self, num_exsists=0):
        self.teacher, _ = get_model(self.model, self.num_classes, self.channel)
        self.teacher = self.teacher.to(self.device)
        # self.teacher.load_state_dict(torch.load(self.path))
        self.teacher.eval()
        self.teacher.to(self.device)

        ''' set augmentation for whole-dataset training '''
        dc_aug_param = 'crop_scale_rotate'  # for whole-dataset training
        self.logger.info('DC augmentation parameters: \n', dc_aug_param)

        # self.train_dataset = self.TensorDataset(self.train_images, self.train_labels)
        # self.trainloader = DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4, pin_memory=True)

        for it in range(self.global_rank + num_exsists, self.num_experts, self.world_size):
            ''' Train synthetic data '''
            self.net.train()
            self.dist_seed()
            self.net.reset_parameters()
            teacher_optim = torch.optim.SGD(self.teacher.parameters(), lr=self.lr_net, momentum=self.momentum, weight_decay=self.weight_decay)
            lr_schedule = [self.epochs // 2 + 1]
            timestamp = []
            for e in range(self.epochs):
                if e in lr_schedule:
                    for param_group in teacher_optim.param_groups:
                        param_group['lr'] *= 0.1
                for i, (images, labels) in enumerate(self.get_train_epoch(torch.arange(self.num_classes), self.batchsize, seed=self.dist_seed(), distributed=False)):
                    images, labels = images.to(self.device), labels.to(self.device)
                    if self.dsa:
                        # Augment images
                        images, labels = self.diff_augment(images, labels, self.dsa_strategy, seed=self.dist_seed())
                    # Forward
                    teacher_optim.zero_grad()
                    outputs, features = self.teacher(images)
                    # Backward
                    loss = self.net.loss_fn(outputs, labels)
                    loss.backward()
                    teacher_optim.step()
                self.logger.debug("Itr: {}\tEpoch: {}\tLoss: {}".format(it, e, loss.item()))
                sys.stdout.flush()
                timestamp.append(self.net.state_dict().copy())
            torch.save(timestamp, os.path.join(self.path, "timestamp_{}.pt".format(it)))


    def condense_images(self):
        self.fix_seed()
        self.batchsize = self.batchsize // self.world_size
        ''' set augmentation for whole-dataset training '''
        dc_aug_param = 'crop_scale_rotate'
        self.logger.info(f'DC augmentation parameters: {dc_aug_param}')

        buffer = []
        for i in range(self.num_experts):
            try:
                buffer.append(torch.load(os.path.join(self.path, "timestamp_{}.pt".format(i)), map_location=torch.device('cpu')))
            except Exception as e:
                self.logger.warning(f"timestamp_{i}.pt not found : {e}")
                continue
        self.logger.info(f"teachers : {len(buffer)}")
        self.logger.info(f"iters : {len(buffer[0])}")
        max_iter = len(buffer[0])
        sys.stdout.flush()

        num_params = sum(p.numel() for p in self.net.parameters())
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.epochs // 2 + 1], gamma=0.1)

        syn_lr = torch.tensor([self.lr_net], requires_grad=True)
        syn_lr = syn_lr.detach().to(self.device).requires_grad_(True)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=1e-5, momentum=0.5)
        self.image_optimizer.zero_grad()
        optimizer_lr.zero_grad()

        for it in range(self.start_iteration, self.iteration):
            self.net.reset_parameters()
            expert_trajectory = buffer[torch.randint(0, len(buffer), (1,)).item()]
            start_epoch = torch.randint(0, self.max_start_iteration, (1,)).item()
            target_params = expert_trajectory[(start_epoch + self.experts_epoch) % max_iter]

            # target_params = torch.cat([param.data.view(-1) for name, param in target_params.items()]).to(self.device, non_blocking=True)
            net_params = {k:v.clone().detach() for k, v in self.net.named_parameters()}
            inital_net = {k:v.clone().detach() for k, v in net_params.items()}

            for o in range(self.o_iter):
                with torch.no_grad():
                    for n, p in self.net.named_parameters():
                        self.logger.debug(f"n : {n} p : {p.isnan().sum()}")
                        p.data = net_params[n].clone().detach()
                syn_images, syn_labels = self.get_synthetic_images(torch.arange(self.num_classes), self.batchsize, seed=self.dist_seed())
                if self.dsa:
                    # augment images
                    syn_images, syn_labels = self.diff_augment(syn_images, syn_labels, self.dsa_strategy, seed=self.dist_seed())
                syn_images = F.interpolate(syn_images, size=self.inp_size, mode='bilinear')
                # self.net.load_state_dict(net_params[-1])
                syn_outputs, syn_features = self.net(syn_images)
                loss = self.net.loss_fn(syn_outputs, syn_labels)
                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True, allow_unused=True)
                if self.distributed:
                    for g in grad:
                        if g is None:
                            g = torch.zeros_like(g)
                        dist.all_reduce(g, op=dist.ReduceOp.SUM)
                net_params = {k: v - syn_lr * g for k, v, g in zip(net_params, net_params.values(), grad)}
                # net_params.append({k:v.clone() for k, v in self.net.named_parameters()})
            
            param_loss = 0
            param_dist = 0
            # for k, np in net_params[-1].items():
            for k, np in net_params.items():
                tp = target_params[k].to(self.device)
                ip = inital_net[k]
                self.logger.debug(f"np : {np.isnan().sum()} tp : {tp.isnan().sum()} ip : {ip.isnan().sum()}")
                param_loss += F.mse_loss(np, tp, reduction='sum')
                param_dist += F.mse_loss(ip, tp, reduction='sum')
            param_loss /= num_params
            param_dist /= num_params
            param_loss /= (param_dist + 1e-12)

            self.image_optimizer.zero_grad()
            optimizer_lr.zero_grad()
            param_loss.backward()
            self.step(self.image_optimizer)
            self.step(optimizer_lr)
            self.scheduler.step()

            for _ in net_params:
                del _
            if self._is_main_process():
                self.save_img(self.path)
                self.visualize(self.path, it)
            self.logger.info("Itr: {}\tLoss: {}\tDist: {}".format(it, param_loss.item(), param_dist.item()))
            sys.stdout.flush()


