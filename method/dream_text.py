import sys
import time
import asyncio
import torch
import torch.nn.functional as F
from method.text_dc_base import TEXT_DC_BASE
from utils.kmeans import KMeans
from utils.diff_augment import Transform

class DREAM_Text(TEXT_DC_BASE):
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

    def condense(self):
        asyncio.run(self._condense())

    async def _condense(self):
        self.visualize(self.path, 0)
        query_list = torch.zeros(self.num_classes, self.batchsize, dtype=torch.long)

        if 'lstm' in self.model:
            self.net._compiled_forward = self.net.forward
            self.net._compiled_embed = self.net.embed

        async def _net_compiled_forward(net, images):
            return net._compiled_forward(images)

        async def _net_forward(net, images):
            return net(images)
        
        async def _net_compiled_embed(net, images):
            return net._compiled_embed(images)
        
        async def _net_embed(net, images):
            return net.embed(images)
            
        async def _net_loss(net, outputs, targets):
            return net.loss_fn(outputs, targets)
        
        async def _grad(parameters, loss):
            return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)

        async def _apply_transform(data, transform, seed=None):
            return transform(data, seed=seed)

        async def _grad_with_graph(parameters, loss):
            return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True, create_graph=True)

        async def _interpolate(images, size):
            return F.interpolate(images, size=size, mode='bilinear', align_corners=True)
        
        async def _match_fn(x1, x2):
            return self.match_fn(x1, x2)

        async def _gradient_mse(grad_list_1, grad_list_2):
            return sum(await asyncio.gather(*[_match_fn(g1, g2) for g1, g2 in zip(grad_list_1, grad_list_2) \
                                              if g1 is not None and g2 is not None and not (len(g1.shape) == 1 and self.bias or len(g1.shape) == 2 and self.fc)]))

        time_start = time.time()
        for it in range(self.start_iteration, self.iteration):
            for o in range(self.o_iter):
                loss_total = 0
                for c in range(self.global_rank, self.num_classes, self.world_size):
                    self.logger.info(f'Class {c} / {self.num_classes}')
                    if o % self.interval == 0:
                        with torch.no_grad():
                            img_epoch = self.get_train_epoch([c], batchsize=self.batchsize, random=False, distributed=False)
                            try:
                                length = 0
                                embeddings = []
                                _embeddings = []
                                for i, (image, label) in enumerate(img_epoch):
                                    length += len(image)
                                    image, label = image.to(self.device), label.to(self.device)
                                    _embeddings.append(_net_compiled_embed(self.net, image))
                                    if (i + 1) % 64 == 0:
                                        embeddings_batch = await asyncio.gather(*_embeddings)
                                        embeddings_batch = torch.cat(embeddings_batch, dim=0)
                                        embeddings.append(embeddings_batch)
                                        # Process the embeddings_batch here
                                        _embeddings = []  # Reset the embeddings list
                                if _embeddings:  # Process the remaining embeddings
                                    embeddings_batch = await asyncio.gather(*_embeddings)
                                    embeddings_batch = torch.cat(embeddings_batch, dim=0)
                                    embeddings.append(embeddings_batch)
                                    # Process the embeddings_batch here
                                embeddings = torch.cat(embeddings, dim=0)
                            except RuntimeError as e:
                                self.logger.error(f'RuntimeError: {e}, retrying...'); continue;
                        embeddings = embeddings.view(embeddings.shape[0], -1)
                        kmeans = KMeans(n_clusters=self.batchsize, max_iter=100, batchsize=self.batchsize*4, mode='euclidean', init='kmeans++', seed=self.syn_seed())
                        pred = kmeans.fit_predict(embeddings)
                        centroids = kmeans.get_centroids()
                        dist = kmeans.compute_distance_matrix(embeddings, centroids)
                        q_idxs = torch.argmin(dist, dim=0)
                        query_list[c] = q_idxs.detach().cpu()
                    real_images = self.train_images[c][query_list[c]].to(self.device)
                    real_labels = self.train_labels[c][query_list[c]].to(self.device)
                    syn_images, syn_labels = self.get_synthetic_images([c])

                    real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
                    syn_images, syn_labels = syn_images.to(self.device), syn_labels.to(self.device)
                    
                    with torch.backends.cudnn.flags(enabled=False):
                        real = _net_forward(self.net, real_images)
                        syn = _net_forward(self.net, syn_images)
                        (real_outputs, real_features), (syn_outputs, syn_features) = await asyncio.gather(real, syn)

                    loss = 0
                    if self.match == 'feat':
                        for f1, f2 in zip(real_features, syn_features):
                            loss += self.match_fn(f1.mean(0), f2.mean(0))
                    if self.match == 'grad':
                        g_real, g_syn = await asyncio.gather(
                            _grad(self.net.parameters(), await _net_loss(self.net, real_outputs, real_labels)),
                            _grad_with_graph(self.net.parameters(), await _net_loss(self.net, syn_outputs, syn_labels)))

                        loss = await _gradient_mse(g_real, g_syn)
                    loss_total += loss.item()
                    loss.backward()
                    self.logger.debug(f'  Iteration {it}, class {c}, loss {loss.item():.4f}')
                self.step(self.image_optimizer)
                self.image_optimizer.zero_grad()
                self.logger.info(f'  Iteration {it}, Outter{o}, loss {loss_total/self.num_classes:.4f}')

                if self.n_data > 0:
                    for _ in range(self.i_iter):
                        n_data = 0
                        for i, (images, targets) in enumerate(self.get_train_epoch(torch.arange(self.num_classes))):
                            images, targets = images.to(self.device), targets.to(self.device)
                            outputs, _ = self.net._compiled_forward(images)
                            loss = self.net.loss_fn(outputs, targets)
                            acc = (outputs.argmax(dim=1) == targets).sum().float() / len(targets)
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.step(self.optimizer)
                            n_data += self.batchsize
                            if n_data >= self.n_data: break;
                past_iter = (it - self.start_iteration) * self.o_iter + o + 1
                left_iter = self.iteration * self.o_iter - past_iter
                eta = (time.time() - time_start) / past_iter * left_iter
                eta = f"{int(eta) // 3600}:{int(eta) % 3600 // 60:02d}:{int(eta) % 60:02d}s.{int((eta - int(eta)) * 1000):03d}"
                self.logger.info(f'Iteration {it}, Outter{o}, ETA {eta}, Time/iter {(time.time() - time_start) / past_iter:.2f}s')
            self.start_iteration = it+1
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