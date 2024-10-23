import gc
import sys
import time
import asyncio
from utils.kmeans import KMeans
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from method.dc_base import DC_BASE, TrainTransform, IDCTransform
from model import get_model
from utils.diff_augment import Transform
from utils.clustering_sample_selection import Clustering_Sample_Selection as CSS

class DualTransform(TrainTransform):
    def __init__(self, method : DC_BASE) -> None:
        super().__init__(method)
        self.inp_size_2 = method.inp_size_2

    @torch.no_grad
    def __call__(self, data, seed=None):
        images, labels = data
        images, labels = images.to(self.device), labels.to(self.device)
        if self.dsa: images, labels = Transform.diff_augment_idc(images, labels, self.dsa_strategy_net, seed=seed)
        if self.mixup_net == 'cut': images, targets = Transform.cutmix(images, labels, self.num_classes)
        else: targets = F.one_hot(labels, self.num_classes).float()
        images_1 = F.interpolate(images, size=self.inp_size, mode='bilinear', align_corners=True)
        images_2 = F.interpolate(images, size=self.inp_size_2, mode='bilinear', align_corners=True)
        return images_1, images_2, targets

class DualTestTransform(TrainTransform):
    def __init__(self, method : DC_BASE) -> None:
        super().__init__(method)
        self.inp_size_2 = method.inp_size_2

    @torch.no_grad
    def __call__(self, data, seed=None):
        images, labels = data
        images, labels = images.to(self.device), labels.to(self.device)
        images_1 = F.interpolate(images, size=self.inp_size, mode='bilinear', align_corners=True)
        images_2 = F.interpolate(images, size=self.inp_size_2, mode='bilinear', align_corners=True)
        return images_1, images_2, labels

class DualCondensation(DC_BASE):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.lr_net_1 = args.lr_net
        self.model_1 = args.model

        self.model_2 = args.model_2
        self.lr_net_2 = args.lr_net_2

        self.use_gradient_balance = True
        self.use_mutual_distillation = True

    async def vit_spatial(self, x):
        # cls = x[:,:1,:]
        x = x[:,1:,:] # + cls
        B, N, C = x.shape
        N = int(sqrt(N))
        x = x.reshape(B, N, N, C)
        x = x.permute(0, 3, 1, 2)
        return x                    # B x C x N x N
    
    async def vit_semantic(self, x):
        cls = x[:,0,:]
        return cls                  # B x C
    
    async def cnn_spatial(self, x):
        return x                    # B x C x H x W

    async def cnn_semantic(self, x):
        return x.flatten(2).mean(2) # B x C
    
    def init_synthetic_images(self):
        self.synthetic_images = torch.randn(self.num_classes, self.n_ipc, self.channel, *(self.im_size), device=self.device)
        self.synthetic_labels = torch.arange(self.num_classes, device=self.device).unsqueeze(1).repeat(1,self.n_ipc)
        if torch.__version__ < '2.0.0':
            self.logger.warning(f'Please use torch version 2.0.0 or higher for better performance!')
            self.net._compiled_forward = types.MethodType(torch.jit.script(self.net.forward), self.net)
            self.net._compiled_embed = types.MethodType(torch.jit.script(self.net.embed), self.net)
        elif torch.__version__ < '2.2.0' and sys.version_info > '3.11.0':
            self.logger.warning(f'Torch version under 2.2.0 is not supported compile with Python 3.11.0 or higher!')
            self.net._compiled_forward = torch.jit.script(self.net.forward)
            self.net._compiled_embed = torch.jit.script(self.net.embed)
        else:
            self.net._compiled_forward = torch.compile(self.net.forward)
            self.net._compiled_embed = torch.compile(self.net.embed)
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
        del self.net
        del self.optimizer
        self.interval = 10

        self.net_1, self.inp_size = get_model(self.model, self.num_classes, self.channel)
        self.net_1 = self.net_1.to(self.device)
        self.optimizer_1 = torch.optim.SGD(self.net_1.parameters(), lr=self.lr_net_1)
        self.optimizer_1.zero_grad()

        self.net_2, self.inp_size_2 = get_model(self.model_2, self.num_classes, self.channel)
        self.net_2 = self.net_2.to(self.device)
        self.optimizer_2 = torch.optim.SGD(self.net_2.parameters(), lr=self.lr_net_2)
        self.optimizer_2.zero_grad()

        if torch.__version__ < '2.0.0':
            self.logger.warning(f'Please use torch version 2.0.0 or higher for better performance!')
            self.net_1._compiled_forward = types.MethodType(torch.jit.script(self.net_1.forward), self.net)
            self.net_2._compiled_forward = types.MethodType(torch.jit.script(self.net_2.forward), self.net)
            self.net_1._compiled_embed = types.MethodType(torch.jit.script(self.net_1.embed), self.net)
            self.net_2._compiled_embed = types.MethodType(torch.jit.script(self.net_2.embed), self.net)
        elif torch.__version__ < '2.2.0' and sys.version_info > '3.11.0':
            self.logger.warning(f'Torch version under 2.2.0 is not supported compile with Python 3.11.0 or higher!')
            self.net_1._compiled_forward = torch.jit.script(self.net_1.forward)
            self.net_2._compiled_forward = torch.jit.script(self.net_2.forward)
            self.net_1._compiled_embed = torch.jit.script(self.net_1.embed)
            self.net_2._compiled_embed = torch.jit.script(self.net_2.embed)
        else:
            self.net_1._compiled_forward = torch.compile(self.net_1.forward)
            self.net_2._compiled_forward = torch.compile(self.net_2.forward)
            self.net_1._compiled_embed = torch.compile(self.net_1.embed)
            self.net_2._compiled_embed = torch.compile(self.net_2.embed)

        with torch.no_grad():
            self.length_1 = self.net_1.length()
            self.length_2 = self.net_2.length()

            dummy_input = torch.zeros((1, self.channel, self.inp_size[0], self.inp_size[1])).to(self.device)
            dummy_input_2 = torch.zeros((1, self.channel, self.inp_size_2[0], self.inp_size_2[1])).to(self.device)

            # Pre-compiling to prevent possible error
            _, feature_1 = self.net_1(dummy_input)
            _, feature_2 = self.net_2(dummy_input_2)
            
            if 'vit' in self.model.lower():
                self.spatial_1 = self.vit_spatial
                self.sementic_1 = self.vit_semantic
            else:
                self.spatial_1 = self.cnn_spatial
                self.sementic_1 = self.cnn_semantic
            if 'vit' in self.model_2.lower():
                self.spatial_2 = self.vit_spatial
                self.sementic_2 = self.vit_semantic
            else:
                self.spatial_2 = self.cnn_spatial
                self.sementic_2 = self.cnn_semantic

            feature_1_shapes = torch.stack([torch.tensor((await self.spatial_1(f)).shape) for f in feature_1]) # L x B x C x H x W
            feature_2_shapes = torch.stack([torch.tensor((await self.spatial_2(f)).shape) for f in feature_2]) # L x B x C x N x N

            MINIMUM_SIZE = True
            MINIMUM_CHANNEL = True

            if MINIMUM_SIZE:
                feature_1_minimum_size = feature_1_shapes.min(dim=0).values[2].item() # minimum size of each layer
                feature_2_minimum_size = feature_2_shapes.min(dim=0).values[2].item() # minimum size of each layer
                feature_minimum_size = feature_1_minimum_size if feature_1_minimum_size < feature_2_minimum_size else feature_2_minimum_size
            else:
                feature_1_minimum_size = feature_1_shapes.max(dim=0).values[2].item() # maximum size of each layer
                feature_2_minimum_size = feature_2_shapes.max(dim=0).values[2].item() # maximum size of each layer
                feature_minimum_size = feature_1_minimum_size if feature_1_minimum_size > feature_2_minimum_size else feature_2_minimum_size # maximum size of each layer

            if MINIMUM_CHANNEL:
                feature_1_minimum_channel = feature_1_shapes.min(dim=0).values[1].item() # minimum channel of each layer
                feature_2_minimum_channel = feature_2_shapes.min(dim=0).values[1].item() # minimum channel of each layer
                feature_minimum_channel = feature_1_minimum_channel if feature_1_minimum_channel < feature_2_minimum_channel else feature_2_minimum_channel
            else:
                feature_1_minimum_channel = feature_1_shapes.max(dim=0).values[1].item()
                feature_2_minimum_channel = feature_2_shapes.max(dim=0).values[1].item()
                feature_minimum_channel = feature_1_minimum_channel if feature_1_minimum_channel > feature_2_minimum_channel else feature_2_minimum_channel

        match_layers_1 = nn.ModuleList([
                nn.Linear(feature_1_shapes[i, 1].item(), feature_minimum_channel).to(self.device)
            for i in range(self.length_1)])
        match_layers_2 = nn.ModuleList([
                nn.Linear(feature_2_shapes[i, 1].item(), feature_minimum_channel).to(self.device)
            for i in range(self.length_2)])
        match_scale = nn.Parameter(torch.zeros(self.length_1, self.length_2).to(self.device))

        self.match_optimizer = torch.optim.SGD([
            {'params': match_layers_1.parameters(), 'lr': 0.001},
            {'params': match_layers_2.parameters(), 'lr': 0.001},
            {'params': match_scale, 'lr': 0.001}
        ], lr=0.1)
        
        async def _net_compiled_forward(net, images):
            return net._compiled_forward(images)

        async def _net_forward(layer, feature):
            return layer(feature)
        
        async def _net_embed(net, images):
            return net._compiled_embed(images)
        
        async def _net_loss(net, outputs, targets):
            return net.loss_fn(outputs, targets)
        
        async def _grad(parameters, loss):
            return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        
        async def _grad_with_graph(parameters, loss):
            return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True, create_graph=True)

        async def _interpolate(images, size):
            return F.interpolate(images, size=size, mode='bilinear', align_corners=True)
        
        async def _match_fn(feature_1, feature_2):
            return (feature_1 - feature_2).pow(2).sum()

        async def _gradient_mse(grad_list_1, grad_list_2):
            return sum(await asyncio.gather(*[_match_fn(g1, g2) for g1, g2 in zip(grad_list_1, grad_list_2) if g1 is not None and g2 is not None]))

        async def _match_layer_semantic(layers, features):
            return torch.stack(await asyncio.gather(*[_net_forward(layers[i], f) for i, f in enumerate(features)]), dim=1).unsqueeze(2)

        async def _match_layer_spatial(layers, features, size):
            return torch.stack(await asyncio.gather(*[_net_forward(layers[i],(await _interpolate(f, size=size)).permute(0, 2, 3, 1).flatten(1, 2)) for i, f in enumerate(features)]), dim=1)

        async def _match_scale(scale, feature_1, feature_2):
            if feature_1.size(1) < feature_2.size(1):
                feature_2 = feature_2.permute(0, 2, 3, 1) @ scale.t().softmax(dim=1)
                feature_2 = feature_2.permute(0, 3, 1, 2)
            else:
                feature_1 = feature_1.permute(0, 2, 3, 1) @ scale.softmax(dim=1)
                feature_1 = feature_1.permute(0, 3, 1, 2)
            return feature_1, feature_2
        
        async def _match_feature(feature_1, feature_2):
            sementic_feature_1 = asyncio.gather(*[self.sementic_1(f) for f in feature_1])
            sementic_feature_2 = asyncio.gather(*[self.sementic_2(f) for f in feature_2])
            spatial_feature_1 = asyncio.gather(*[self.spatial_1(f) for f in feature_1])
            spatial_feature_2 = asyncio.gather(*[self.spatial_2(f) for f in feature_2])
            feature_1 = torch.cat(await asyncio.gather(_match_layer_semantic(match_layers_1, await sementic_feature_1), 
                                                       _match_layer_spatial(match_layers_1, await spatial_feature_1, feature_minimum_size)), dim=2)
            feature_2 = torch.cat(await asyncio.gather(_match_layer_semantic(match_layers_2, await sementic_feature_2),
                                                       _match_layer_spatial(match_layers_2, await spatial_feature_2, feature_minimum_size)), dim=2)
            return await _match_scale(match_scale, feature_1, feature_2)

        query_list = torch.zeros(self.num_classes, self.batchsize, dtype=torch.long)

        num_parameters_1 = sum([p.numel() for p in self.net_1.parameters()])
        num_parameters_2 = sum([p.numel() for p in self.net_2.parameters()])

        param_ratio = (num_parameters_2 / num_parameters_1) ** 0.5
        # num_sample_1 = 2000 if param_ratio < 1 else int(2000 * param_ratio)
        # num_sample_2 = 2000 if param_ratio > 1 else int(2000 / param_ratio)

        self.logger.info(f'Number of Parameters: {num_parameters_1} / {num_parameters_2}')
        if self._is_main_process():
            self.visualize(self.path, 0)
        set_timer = time.time()
        start_it = self.start_iteration

        for it in range(self.start_iteration, self.iteration):
            with torch.no_grad():
                match_layers_1 = nn.ModuleList([
                        nn.Linear(feature_1_shapes[i, 1].item(), feature_minimum_channel).to(self.device)
                    for i in range(self.length_1)])
                match_layers_2 = nn.ModuleList([
                        nn.Linear(feature_2_shapes[i, 1].item(), feature_minimum_channel).to(self.device)
                    for i in range(self.length_2)])
                match_scale = nn.Parameter(torch.zeros(self.length_1, self.length_2).to(self.device))
            self.match_optimizer = torch.optim.Adam([
                {'params': match_layers_1.parameters(), 'lr': 5e-4},
                {'params': match_layers_2.parameters(), 'lr': 5e-4},
                {'params': match_scale, 'lr': 5e-2}
            ], lr=0.1)
            self.net_1.reset_parameters()
            self.net_2.reset_parameters()
            self.pre_epochs = 0
            # self.logger.info(f'Iteration {it} / {self.iteration} Warmup for {self.pre_epochs} epochs!')
            # for e in range(self.pre_epochs):
            #     for step, (images, labels) in enumerate(self.get_train_epoch([i for i in range(self.num_classes)])):
            #         if self.dsa:
            #             images, labels = DiffAugment.diff_augment_idc(images, labels, self.dsa_strategy_net)
            #         if self.mixup_net == 'cut':
            #             images, targets = DiffAugment.cutmix(images, labels)
            #         else:
            #             targets = F.one_hot(labels, self.num_classes).float()
            #         images_1 = F.interpolate(images, size=self.inp_size, mode='bilinear')
            #         images_2 = F.interpolate(images, size=self.inp_size_2, mode='bilinear')

            #         outputs_1, feature_1 = self.net_1(images_1)
            #         loss_1 = self.net_1.loss_fn(outputs_1, targets)

            #         outputs_2, feature_2 = self.net_2(images_2)
            #         loss_2 = self.net_2.loss_fn(outputs_2, targets)

            #         self.optimizer_1.zero_grad()
            #         loss_1.backward(retain_graph=True)
            #         self.step(self.optimizer_1)

            #         self.optimizer_2.zero_grad()
            #         loss_2.backward()
            #         self.step(self.optimizer_2)

            model1_dist = torch.ones(self.num_classes, device=self.device)
            model2_dist = torch.ones(self.num_classes, device=self.device)
            if self.use_mutual_distillation:
                # grad_accumulator = torch.ones(3, device=self.device)
                grad_accumulator = torch.zeros(self.num_classes, 3, device=self.device) + 1e-6
            else:
                # grad_accumulator = torch.ones(2, device=self.device)
                grad_accumulator = torch.zeros(self.num_classes, 2, device=self.device) + 1e-6
            self.logger.info(f"Start Iteration {it} / {self.iteration}")
            for o in range(self.o_iter):
                L1 = 0; L2 = 0; LF = 0; LT = 0;
                for c in range(self.global_rank, self.num_classes, self.world_size):
                    self.logger.info(f'Class {c} / {self.num_classes} | distance : {model1_dist[c]} / {model2_dist[c]}')
                    real_image_chunk = []
                    real_label_chunk = []
                    syn_image_chunk = []
                    syn_label_chunk = []

                    if o % self.interval == 0:
                        with torch.no_grad():
                            while True:
                                embeddings_1 = []
                                embeddings_2 = []
                                try:
                                    for i, (image1, image2, label) in enumerate(self.get_train_epoch([c], batchsize=self.batchsize, transform=DualTestTransform(self), random=False, distributed=False)):
                                        embeddings_1.append(_net_embed(self.net_1, image1))
                                        embeddings_2.append(_net_embed(self.net_2, image2))
                                    embeddings_1 = await asyncio.gather(*embeddings_1)
                                    embeddings_2 = await asyncio.gather(*embeddings_2)
                                    embeddings_1 = torch.cat(embeddings_1, dim=0)
                                    embeddings_2 = torch.cat(embeddings_2, dim=0)
                                    break;
                                except Exception as e: self.logger.error(f'RuntimeError: {e}, retrying...'); continue;
                            embeddings_1 = embeddings_1.view(embeddings_1.shape[0], -1)
                            embeddings_2 = embeddings_2.view(embeddings_2.shape[0], -1)
                            
                            kmeans = KMeans(self.batchsize, max_iter=100, batchsize=self.batchsize, mode='euclidean', init='kmeans++', seed=None)
                            pred1 = kmeans.fit_predict(embeddings_1)
                            centroids_1 = kmeans.get_centroids().clone()
                            dist = kmeans.compute_distance_matrix(embeddings_1, centroids_1)
                            q_idxs1 = torch.argmin(dist, dim=0)#[:self.batchsize]
                            dist = dist[torch.arange(len(dist)), pred1]
                            mean_dist1 = (dist / dist.max()).mean()

                            kmeans = KMeans(self.batchsize, max_iter=100, batchsize=self.batchsize, mode='euclidean', init='kmeans++', seed=None)
                            pred2 = kmeans.fit_predict(embeddings_2) 
                            centroids_2 = kmeans.get_centroids().clone()
                            dist = kmeans.compute_distance_matrix(embeddings_2, centroids_2)
                            q_idxs2 = torch.argmin(dist, dim=0)#[:self.batchsize]
                            dist = dist[torch.arange(len(dist)), pred2]
                            mean_dist2 = (dist / dist.max()).mean()
                            
                            model1_dist[c] = mean_dist1
                            model2_dist[c] = mean_dist2
                            if (o // self.interval) % 2 == 0: query_list[c] = q_idxs1.detach().cpu()
                            if (o // self.interval) % 2 == 1: query_list[c] = q_idxs2.detach().cpu()
                    self.dist_barrier()
                    real_images = self.train_images[c][query_list[c]].to(self.device)
                    real_labels = self.train_labels[c][query_list[c]].to(self.device)
                    syn_images, syn_labels = self.get_synthetic_images([c], self.batchsize, seed=self.dist_seed())
                    if self.dsa:
                        seed = self.syn_seed()
                        syn_images, syn_labels = IDCTransform(self)((syn_images, syn_labels), seed=seed)
                        real_images, real_labels = IDCTransform(self)((real_images, real_labels), seed=seed)
                    real_images, real_labels = real_images.to(self.device), real_labels.to(self.device)
                    syn_images, syn_labels = syn_images.to(self.device), syn_labels.to(self.device)

                    (real_outputs_1, real_features_1),(syn_outputs_1, syn_features_1),\
                    (real_outputs_2, real_features_2),(syn_outputs_2, syn_features_2) = \
                        await asyncio.gather(_net_forward(self.net_1, F.interpolate(real_images, size=self.inp_size, mode='bilinear', align_corners=True)),
                                                _net_forward(self.net_1, F.interpolate(syn_images, size=self.inp_size, mode='bilinear', align_corners=True)),
                                                _net_forward(self.net_2, F.interpolate(real_images, size=self.inp_size_2, mode='bilinear', align_corners=True)),
                                                _net_forward(self.net_2, F.interpolate(syn_images, size=self.inp_size_2, mode='bilinear', align_corners=True)))
                    real_loss_1, real_loss_2, syn_loss_1, syn_loss_2 = \
                        await asyncio.gather(_net_loss(self.net_1, real_outputs_1, real_labels),
                                                _net_loss(self.net_2, real_outputs_2, real_labels),
                                                _net_loss(self.net_1, syn_outputs_1, syn_labels),
                                                _net_loss(self.net_2, syn_outputs_2, syn_labels))
                    real_grad_1, real_grad_2, syn_grad_1, syn_grad_2 = \
                        await asyncio.gather(_grad(self.net_1.parameters(), real_loss_1),
                                                _grad(self.net_2.parameters(), real_loss_2),
                                                _grad_with_graph(self.net_1.parameters(), syn_loss_1),
                                                _grad_with_graph(self.net_2.parameters(), syn_loss_2))
                    if self.use_mutual_distillation:
                        (real_feature_1, real_feature_2), (syn_feature_1, syn_feature_2) = \
                            await asyncio.gather(_match_feature(real_features_1, real_features_2),
                                                    _match_feature(syn_features_1, syn_features_2))
                    
                        real_feature_loss = ((real_feature_1 - real_feature_2).pow(2).mean())
                        syn_feature_loss = ((syn_feature_1 - syn_feature_2).pow(2).mean())

                        real_feature_grad_1, real_feature_grad_2, syn_feature_grad_1, syn_feature_grad_2 = \
                            await asyncio.gather(_grad(self.net_1.parameters(), real_feature_loss),
                                                _grad(self.net_2.parameters(), real_feature_loss),
                                                _grad_with_graph(self.net_1.parameters(), syn_feature_loss),
                                                _grad_with_graph(self.net_2.parameters(), syn_feature_loss))

                        _feature_loss_1 = _gradient_mse(real_feature_grad_1, syn_feature_grad_1)
                        _feature_loss_2 = _gradient_mse(real_feature_grad_2, syn_feature_grad_2)

                        _loss_1 = _gradient_mse(real_grad_1, syn_grad_1)
                        _loss_2 = _gradient_mse(real_grad_2, syn_grad_2)

                        _feature_loss_1, _feature_loss_2, _loss_1, _loss_2 = \
                            await asyncio.gather(_feature_loss_1, _feature_loss_2, _loss_1, _loss_2)

                    else:
                        _feature_loss_1 = 0
                        _feature_loss_2 = 0
                        _loss_1 = _gradient_mse(real_grad_1, syn_grad_1)
                        _loss_2 = _gradient_mse(real_grad_2, syn_grad_2)

                        _loss_1, _loss_2 = await asyncio.gather(_loss_1, _loss_2)

                    feature_loss = _feature_loss_1 + _feature_loss_2
                    loss_1 = _loss_1; loss_2 = _loss_2;

                    if self.use_gradient_balance:
                        if o % 10 == 0 :
                            with torch.no_grad():
                                if self.use_mutual_distillation:
                                    (grad1,), (grad2,), (grad3,) = await asyncio.gather(_grad(self.synthetic_images, loss_1),
                                                                                        _grad(self.synthetic_images, loss_2),
                                                                                        _grad(self.synthetic_images, feature_loss))
                                else:
                                    (grad1,), (grad2,) = await asyncio.gather(_grad(self.synthetic_images, loss_1),
                                                                              _grad(self.synthetic_images, loss_2))

                                if grad1 is None or grad1.numel() == 0:
                                    grad1 = torch.zeros_like(self.synthetic_images)
                                if grad2 is None or grad2.numel() == 0:
                                    grad2 = torch.zeros_like(self.synthetic_images)
                                if self.use_mutual_distillation:
                                    if grad3 is None or grad3.numel() == 0:
                                        grad3 = torch.zeros_like(self.synthetic_images)
                                    grad_max = torch.zeros(self.num_classes, 3, device=self.device)
                                else:
                                    grad_max = torch.zeros(self.num_classes, 2, device=self.device)
                                grad_max[c, 0] = grad1.abs().max(); grad_max[c, 1] = grad2.abs().max();
                                if self.use_mutual_distillation: grad_max[c, 2] = grad3.abs().max();
                                # if self.distributed:
                                #     torch.distributed.all_reduce(grad_max, op=torch.distributed.ReduceOp.MAX)
                                grad_accumulator += grad_max.to(self.device)
                        with torch.no_grad():
                            grad_scaler = 1 / grad_accumulator[c]
                            grad_scaler = grad_scaler / grad_scaler.max()
                    else:
                        grad_scaler = [1, 1, 1, 1]

                    L1 += loss_1.item() 
                    L2 += loss_2.item()
                    if self.use_mutual_distillation:
                        LF += feature_loss.item()

                    loss = loss_1 * grad_scaler[0] + loss_2 * grad_scaler[1]
                    if self.use_mutual_distillation:
                        loss += feature_loss * grad_scaler[2]
                    LT += loss.item()
                    loss.backward()

                if self.use_gradient_balance:
                    with torch.no_grad():
                        grad = self.synthetic_images.grad.view(self.num_classes, self.n_ipc, self.channel, self.factor ** 2, -1)
                        grad = grad / (grad.abs().max(dim=3, keepdim=True)[0] + 1e-6)
                        grad = grad.view(self.num_classes, self.n_ipc, self.channel, *self.im_size)
                        self.synthetic_images.grad.data = grad
                        if grad.isnan().any():self.logger.critical('Nan Detected!'); exit();
                self.step(self.image_optimizer)
                self.image_optimizer.zero_grad()
                with torch.no_grad():
                    for ch in range(self.channel):
                        self.synthetic_images.data[:,:,ch] = self.synthetic_images.data[:,:,ch] * self.std[ch] + self.mean[ch]
                    self.synthetic_images.data = self.synthetic_images.data.clamp(0, 1)
                    for ch in range(self.channel):
                        self.synthetic_images.data[:,:,ch] = (self.synthetic_images.data[:,:,ch] - self.mean[ch]) / self.std[ch]
                    self.synthetic_images.requires_grad = True
                
                transform = DualTransform(self)
                for i in range(self.i_iter):
                    self.logger.debug(f'Iteration {it}, Inner {i}')
                    for step, (images_1, images_2, targets) in enumerate(self.get_train_epoch([i for i in range(self.num_classes)], transform=transform, batchsize=self.batchsize)):
                        images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(self.device)
                        (output_1, feature_1), (output_2, feature_2) = await asyncio.gather(_net_compiled_forward(self.net_1, images_1),
                                                                                            _net_compiled_forward(self.net_2, images_2))
                        loss_1, loss_2, (feature_1, feature_2) = await asyncio.gather(_net_loss(self.net_1, output_1, targets),
                                                                                        _net_loss(self.net_2, output_2, targets),
                                                                                        _match_feature(feature_1, feature_2))
                        if self.use_mutual_distillation:
                            feature_loss = ((feature_1 - feature_2).pow(2).mean())
                        else:
                            feature_loss = torch.zeros(1, device=self.device)
                        self.optimizer_1.zero_grad()
                        self.optimizer_2.zero_grad()
                        loss = loss_1 + loss_2 + feature_loss
                        loss.backward()
                        self.step(self.optimizer_1)
                        self.step(self.optimizer_2)
                        if step*self.batchsize >= 2000:
                            break
                    if self.use_mutual_distillation:
                        for step, (images_1, images_2, targets) in enumerate(self.get_train_epoch([i for i in range(self.num_classes)], transform=transform, batchsize=self.batchsize)):
                            with torch.no_grad():
                                images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(self.device)
                                (outputs_1, feature_1), (outputs_2, feature_2) = await asyncio.gather(_net_compiled_forward(self.net_1, images_1),
                                                                                                        _net_compiled_forward(self.net_2, images_2)) 
                            feature_1, feature_2 = await _match_feature(feature_1, feature_2)
                            loss = ((feature_1 - feature_2).pow(2).mean())
                            self.match_optimizer.zero_grad()
                            loss.backward()
                            self.step(self.match_optimizer)
                            if step*self.batchsize >= 2000:
                                break
                past_iter = (it - start_it) * self.o_iter + o + 1
                left_iter = self.o_iter * self.iteration - past_iter
                eta = (time.time() - set_timer) / past_iter * left_iter
                eta = f"{int(eta) // 3600}:{int(eta) % 3600 // 60:02d}:{int(eta) % 60:02d}s.{int((eta - int(eta)) * 1000):03d}"
                self.logger.info(f"Iteration {it}, Outter {o}, loss {LT/self.num_classes*self.world_size:.4f}, "\
                                                             f"loss_1 {L1/self.num_classes*self.world_size:.4f}, "\
                                                             f"loss_2 {L2/self.num_classes*self.world_size:.4f}, "\
                                                             f"feature_loss {LF/self.num_classes*self.world_size:.4f} eta {eta}")

            self.start_iteration = it + 1
            gc.collect()
            if self._is_main_process():
                self.save_img(self.path)
                self.visualize(self.path, it)