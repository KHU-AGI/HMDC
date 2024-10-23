import gc
import sys
import time
import asyncio
import torch
import torch.nn.functional as F
import torch.distributed as dist
from method.dc_base import DC_BASE, TrainTransform, TestTransform
from utils.log import Log
from utils.thread_with_return_value import Thread_With_Return_Value

class CAFE(DC_BASE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.discrimination_loss_weight = args.discrimination_loss_weight
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2

    def condense(self):
        self.load_img(self.path)
        self.logger.info(f"Start CAFE training")
        self.batchsize = self.batchsize

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr_net)

        def _load_img(c, bs):
            try:
                real_images, real_labels = self.get_train_images([c], bs, seed=self.dist_seed())
                syn_images, syn_labels = self.get_synthetic_images([c], bs, seed=self.dist_seed())
                real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
                syn_images, syn_labels = syn_images.to(self.device), syn_labels.to(self.device)
                if self.dsa:
                    # augment images
                    seed = self.syn_seed()
                    real_images, real_labels = TrainTransform(self)((real_images, real_labels), seed=seed)
                    syn_images, syn_labels = TrainTransform(self)((syn_images, syn_labels), seed=seed)
                real_images = F.interpolate(real_images, size=self.inp_size, mode='bilinear', align_corners=True)
                syn_images = F.interpolate(syn_images, size=self.inp_size, mode='bilinear', align_corners=True)
                return real_images, real_labels, syn_images, syn_labels
            except Exception as e:
                self.logger.error(f"Error occured in image process : {e}")
                return None, None, None, None
    
        @torch.no_grad()
        def _eval_img(c, bs):
            test_acc = 0
            count = 0
            with torch.no_grad():
                images, labels = self.get_train_images([c], bs, seed=self.dist_seed())
                real_images, real_labels = images.to(self.device), labels.to(self.device)
                real_images = F.interpolate(real_images, size=self.inp_size, mode='bilinear', align_corners=True)
                outputs, _ = self.net(real_images)
                test_acc += (outputs.argmax(dim=1) == real_labels).sum().float()
                count += len(real_labels)
            return test_acc, count
        
        def feature_matching_loss(real_features, syn_features):
            real_features = [f.view(self.num_classes, -1, *f.shape[1:]) for f in real_features]
            syn_features = [f.view(self.num_classes, -1, *f.shape[1:]) for f in syn_features]
            feature_matching_loss = 0
            for l, (fr, fs) in enumerate(zip(real_features, syn_features)):
                feature_matching_loss += F.mse_loss(fr.mean(), fs.mean(), reduction='sum')
            return feature_matching_loss

        def discrimination_loss(real_features, syn_features):
            FR = real_features[-1].view(self.num_classes, -1, *real_features[-1].shape[1:]).flatten(0, 1).flatten(1, -1)
            FS = syn_features[-1].view(self.num_classes, -1, *syn_features[-1].shape[1:]).mean(1).flatten(1, -1)
            O = torch.mm(FR, FS.T)
            discrimination_loss = F.cross_entropy(O, real_labels)
            return discrimination_loss

        for it in range(self.start_iteration, self.iteration):
            self.logger.info(f"Iteration {it} / {self.iteration}")
            self.net.reset_parameters()
            outter_watcher = []
            time_start = time.time()
            for o_iter in range(self.o_iter):
                train_acc = 0

                # Asymmetrical data loading
                if self.batchsize % self.num_classes == self.batchsize:
                    bs = self.batchsize
                else:
                    bs = self.batchsize - (self.batchsize % self.num_classes)
                _th_list = []
                for c in range(self.num_classes):
                    _th = Thread_With_Return_Value(target=_load_img, args=(c, bs))
                    _th.start()
                    _th_list.append(_th)
                ret = [th.join() for th in _th_list]
                real_images, real_labels, syn_images, syn_labels = zip(*ret)
                _t1 = Thread_With_Return_Value(target=torch.cat, args=(real_images, 0))
                _t2 = Thread_With_Return_Value(target=torch.cat, args=(real_labels, 0))
                _t3 = Thread_With_Return_Value(target=torch.cat, args=(syn_images, 0))
                _t4 = Thread_With_Return_Value(target=torch.cat, args=(syn_labels, 0))
                _t1.start();_t2.start();_t3.start();_t4.start();
                real_images = _t1.join().to(self.device)
                real_labels = _t2.join().to(self.device)
                syn_images = _t3.join().to(self.device)
                syn_labels = _t4.join().to(self.device)

                _t1 = Thread_With_Return_Value(target=self.net, args=(real_images,))
                _t2 = Thread_With_Return_Value(target=self.net, args=(syn_images,))
                _t1.start();_t2.start();
                real_outputs, real_features = _t1.join()
                syn_outputs, syn_features = _t2.join()
                train_acc = (syn_outputs.argmax(dim=1) == syn_labels.argmax(dim=1)).sum()

                _t1 = Thread_With_Return_Value(target=feature_matching_loss, args=(real_features, syn_features))
                _t2 = Thread_With_Return_Value(target=discrimination_loss, args=(real_features, syn_features))
                _t1.start();_t2.start();
                
                loss = _t1.join() + _t2.join() + F.cross_entropy(real_outputs, real_labels)
                self.image_optimizer.zero_grad()
                loss.backward()
                train_acc = train_acc / self.batchsize                
                self.step(self.image_optimizer)
                if self.distributed:
                    dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
                    train_acc = train_acc / self.world_size
                test_acc = 0
                count = 0
                self.logger.debug(f"Start test")
                _th_list = []
                for c in range(self.global_rank, self.num_classes, self.world_size):
                    _th = Thread_With_Return_Value(target=_eval_img, args=(c, self.batchsize))
                    _th.start()
                    _th_list.append(_th)
                for _th in _th_list:
                    t_a, c = _th.join(); test_acc += t_a; count += c;
                test_acc = test_acc / count
                if self.distributed:
                    dist.all_reduce(test_acc, op=dist.ReduceOp.SUM)
                    test_acc = test_acc / self.world_size
                outter_watcher.append(test_acc)
                self.logger.info(f"Outter: {o_iter} Loss: {loss.item():4f}  Test Acc: {test_acc*100:.2f}")
                if len(outter_watcher) == 10:
                    if max(outter_watcher) - min(outter_watcher) < self.lambda_1:
                        break
                    else:
                        outter_watcher.pop(0)
                inner_watcher = []
                for i_iter in range(self.i_iter):
                    if self.distributed:
                        dist.barrier()
                    train_acc = torch.zeros(1, device=self.device)
                    count = torch.zeros(1, device=self.device)
                    syn_epoch = self.get_synthetic_epoch(torch.arange(self.num_classes), transform=TrainTransform(self))
                    for i, (images, labels) in enumerate(syn_epoch):
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs, _ = self.net(images)
                        loss = F.cross_entropy(outputs, labels)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.step(self.optimizer)
                        train_acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().float()
                        count += len(labels)
                    train_acc = train_acc / count
                    if self.distributed:
                        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
                        train_acc = train_acc / self.world_size
                    test_acc = torch.zeros(1, device=self.device)
                    count = torch.zeros(1, device=self.device)
                    with torch.no_grad():
                        _th_list = []
                        for c in range(self.global_rank, self.num_classes, self.world_size):
                            _th = Thread_With_Return_Value(target=_eval_img, args=(c, self.batchsize))
                            _th.start()
                            _th_list.append(_th)
                        for _th in _th_list:
                            t_a, c = _th.join(); test_acc += t_a; count += c;
                        test_acc = test_acc / count
                        if self.distributed:
                            dist.all_reduce(test_acc, op=dist.ReduceOp.SUM)
                            test_acc = test_acc / self.world_size
                        inner_watcher.append(test_acc)
                    if len(inner_watcher) == 10:
                        if max(inner_watcher) - min(inner_watcher) > self.lambda_2: break;
                        else: inner_watcher.pop(0);
                    self.logger.info(f"  Inner: {i_iter} Loss: {loss.item():4f}  Train Acc: {train_acc.item()*100:.2f}  Test Acc: {test_acc.item()*100:.2f}")
                    gc.collect()
                past_iter = it * self.o_iter + o_iter + 1
                left_iter = self.o_iter * self.iteration - past_iter
                eta = (time.time() - time_start) / past_iter * left_iter
                eta = f"{int(eta) // 3600}:{int(eta) % 3600 // 60}:{int(eta) % 60}s.{int((eta - int(eta)) * 1000)}"
                self.logger.info(f"Outter Loop {o_iter} done, ETA: {eta}m")
                gc.collect()
            self.start_iteration += 1
            if self._is_main_process():
                self.save_img(self.path)
                self.visualize(self.path, it)
            self.logger.info(f"Iteration {it} done")
            
