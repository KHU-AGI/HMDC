import os
import sys
import types
import shutil
import random
import traceback
import threading
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.distributed as dist
from math import ceil, sqrt
from torch.profiler import profile, ProfilerActivity
from torchvision.utils import save_image
from model import get_model
from utils.utils import get_dataset
from sklearn.manifold import TSNE
from utils.log import Log
from utils.diff_augment import Transform

class DC_BASE(object):
    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, index, transform=None):
            self.dataset = dataset
            self.index = index
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.dataset[self.index[idx]]
            x = x.unsqueeze(0)
            if self.transform:
                x, y = self.transform((x, y))
            return x.squeeze(), y
        def __len__(self):
            return len(self.index)

    def __init__(self, args) -> None:
        self.method = args.method
        self.seed = args.seed
        self.num_task = args.num_task
        self.dataset = args.dataset
        self.model = args.model

        self.n_ipc = args.n_ipc
        self.init = args.init
        self.factor = args.factor

        self.epochs = args.epochs
        self.eval_mode = args.eval_mode
        self.epoch_eval_train = args.epoch_eval_train

        self.iteration = args.iteration
        self.batchsize = args.batchsize
        self.start_iteration = 0

        self.data_path = args.data_path
        self.path = args.path
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        self.use_real_img = args.use_real_img
        self.memory_size = args.memory_size

        self.o_iter = args.o_iter
        self.i_iter = args.i_iter

        self.lr_img = args.lr_img
        self.lr_net = args.lr_net

        self.momentum = args.momentum
        self.weight_decay = args.weight_decay

        self.dsa = args.dsa
        self.dsa_iter = args.dsa_iter
        self.dsa_strategy = 'color_crop_cutout_flip_scale_rotate' if self.dsa else None
        self.dsa_strategy_net = 'color_crop_cutout_flip_scale_rotate' if self.dsa else None
        self.mixup = args.mixup
        self.mixup_net = args.mixup_net
        # Cutmix has negative effect on ViT
        if 'vit' in self.model.lower():
            self.mixup  = False
            self.mixup_net = False
        if self.mixup == 'cut' and self.dsa_strategy is not None:
            self.dsa_strategy.replace('cutout_', '')
        if self.mixup_net == 'cut' and self.dsa_strategy_net is not None:
            self.dsa_strategy_net.replace('cutout_', '')
        self.beta = args.beta
        self.logging_level = args.log_level

        self.dist_url = args.dist_url
        self.dist_backend = args.dist_backend
        try: self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        except: self.world_size = 0
        if self.world_size == 1 and self.dist_url != 'env://':
            self.dist_url = 'env://'
        try: self.global_rank = int(os.environ.get('RANK', 0))
        except: self.global_rank = 0
        self.world_size = self.world_size * torch.cuda.device_count()
        if 'Computation' in self.method:
            self.world_size = 1
            self.global_rank = 0
        print(f"World size: {self.world_size}, global rank: {self.global_rank}")
        self.distributed = self.world_size > 1 and torch.cuda.is_available()

        self.verbose = args.verbose
        self.num_workers = 6

    def run(self):
        self.log_queue = mp.Queue()
        if self.distributed:
            proc_list = []
            for i in range(torch.cuda.device_count()-1):
                p = mp.Process(target=self.worker, args=(i,))
                p.start()
                proc_list.append(p)
            self.worker(torch.cuda.device_count()-1)
            for p in proc_list:
                p.join()
        else: self.worker(0)

    def worker(self, gpu):
        try: 
            self.logger, self.listener = Log.init_logger(self.log_queue, self.method + "_" + str(gpu + self.global_rank * torch.cuda.device_count()), level=Log.get_level(self.logging_level), path=self.path)
            if torch.cuda.is_available():
                if self.distributed:
                    self.local_rank = gpu
                    self.global_rank = self.local_rank + self.global_rank * torch.cuda.device_count()
                    self.device = torch.device(f'cuda:{self.local_rank}')
                    torch.cuda.set_device(self.device)
                    dist_url = f"{self.dist_url}{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
                    self.logger.info(f"Initializing process group with {dist_url}...")
                    self.logger.debug(f"World size {self.world_size}, global rank {self.global_rank}, local rank {self.local_rank}...")
                    sys.stdout.flush()
                    torch.distributed.init_process_group(backend=self.dist_backend, init_method=dist_url, world_size=self.world_size, rank=self.global_rank)
                    self.logger.debug(f"Rank {self.global_rank} initialized.")
                    # self._setup_for_distributed(self._is_main_process())
                    torch.cuda.synchronize()
                else:
                    self.local_rank = gpu
                    self.device = torch.device(f'cuda:{self.local_rank}')
                    torch.cuda.set_device(self.device)
            else:
                self.device = torch.device('cpu')
                self.logger.warn("No GPU available!"
                                    " Using CPU for training. This will be extremely slow.")
            # if not self._is_main_process():
            #     self.logger.setLevel(Log.get_level("ERROR"))
            self.fix_seed()
            # Make the download occurs only in one process per node
            if self.distributed and self.local_rank != 0: self.dist_barrier()
            self.load_dataset()
            if self.distributed and self.local_rank == 0: self.dist_barrier()
            
            self.net, self.inp_size = get_model(self.model, self.num_classes, self.channel)
            self.net = self.net.to(self.device)
            # self.init_synthetic_images()
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr_net, momentum=0.9, weight_decay=5e-4)
            # self.image_optimizer = torch.optim.SGD((self.synthetic_images,), lr=self.lr_img, momentum=0.5)
            self.load_img(self.path)

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

            if 'Test' in self.method:
                self.test_condensed_images()
            elif self.method == 'Continual':
                self.train_continual()
            elif self.method == 'Visualize':
                self.visualize_features()
            elif self.method == 'Upperbound':
                self.test_model_accuracy()
            elif 'Computation' in self.method:
                self.computation_cost()
            else:
                self.condense()
            Log.stop_log_queue(self.listener)

        except Exception as e:
            self.logger.critical(f"Error: {e}")
            self.logger.critical(traceback.format_exc())
            self.logger.critical(sys.exc_info()[2])
            Log.stop_log_queue(self.listener)
            raise e

    def condense(self):
        raise NotImplementedError
    
    def fix_seed(self):
        if not self.seed is None:
            # Seed for reproducibility
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            self.logger.info(f"Seed fixed to {self.seed}")
            self.logger.warn("The seed is fixed. This will disable the cudnn benchmark and may slow down the training.")
            
    def computation_cost(self):
        torch.compile = lambda x: x # Disable torch dynamo. This makes include the compile time.
        self.iteration = 1
        self.o_iter = 1
        # with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            # with record_function("model_inference"):
            self.condense()
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10,))
        prof.export_chrome_trace("trace.json")
        shutil.rmtree(self.path)

    def test_model_accuracy(self):
        self.net.train()
        max_acc = 0
        trainaugment = IDCTransform(self)
        testaugment = TestTransform(self)
        for epoch in range(self.epochs):
            train_epoch = self.get_train_epoch(torch.arange(self.num_classes), transform=trainaugment)
            for images, targets in train_epoch:
                images, targets = images.detach().to(self.device), targets.detach().to(self.device)
                logits, _ = self.net(images)
                loss = self.net.loss_fn(logits, targets)
                acc = (logits.argmax(1) == targets.argmax(1)).float().sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.step(self.optimizer)
            if epoch % 10 == 0:
                with torch.no_grad():
                    test_acc = torch.zeros(1).to(self.device)
                    count = torch.zeros(1).to(self.device)
                    self.net.eval()
                    test_epoch = self.get_test_epoch(torch.arange(self.num_classes), transform=testaugment)
                    for images, labels in test_epoch:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits, _ = self.net(images)
                        acc = (logits.argmax(1) == labels).float().sum()
                        test_acc += acc.item()
                        count += images.shape[0]
                    if self.distributed:
                        dist.all_reduce(test_acc, op=dist.ReduceOp.SUM)
                        dist.all_reduce(count, op=dist.ReduceOp.SUM)
                    test_acc /= count
                    self.logger.debug(f"Test accuracy: {test_acc}")
                    if test_acc > best_acc:
                        best_acc = test_acc
                sys.stdout.flush()
        self.logger.info(f"Best accuracy: {max_acc}")

    def test_condensed_images(self):
        self.net.reset_parameters()
        self.load_img(self.path)
        self.visualize(self.path, 'test')
        self.net.eval()

        num_params = sum(p.numel() for p in self.net.parameters())
        num_learnable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.synthetic_images.requires_grad = False

        self.logger.info(f"Number of parameters: {num_params}")
        self.logger.info(f"Number of learnable parameters: {num_learnable_params}")

        best_acc = 0
        trainaugment = IDCTransform(self)
        testaugment = TestTransform(self)
        for epoch in range(self.epochs):
            syn_epoch = self.get_synthetic_epoch(torch.arange(self.num_classes), transform=trainaugment)
            i = 0
            for images, labels in syn_epoch:
                # self.logger.info(f"Epoch {epoch}, batch {i}")
                i += 1
                images, labels = images.detach().to(self.device), labels.detach().to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.net(images)
                loss = F.cross_entropy(logits, labels)
                acc = (logits.argmax(1) == labels.argmax(1)).float().sum()
                loss.backward()
                self.step(self.optimizer)
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    test_acc = torch.zeros(1).to(self.device)
                    count = torch.zeros(1).to(self.device)
                    self.net.eval()
                    test_epoch = self.get_test_epoch(torch.arange(self.num_classes), transform=testaugment)
                    for images, labels in test_epoch:
                        images, labels = images.to(self.device), labels.to(self.device)
                        logits, _ = self.net(images)
                        acc = (logits.argmax(1) == labels).float().sum()
                        test_acc += acc
                        count += images.shape[0]
                    if self.distributed:
                        dist.all_reduce(test_acc, op=dist.ReduceOp.SUM)
                        dist.all_reduce(count, op=dist.ReduceOp.SUM)
                    test_acc /= count
                    self.logger.info(f"Test accuracy: {test_acc}")
                    if test_acc > best_acc:
                        best_acc = test_acc
                sys.stdout.flush()
        self.logger.info(f"Best accuracy: {best_acc}")

    def train_continual(self):
        self.tasks = torch.arange(self.num_classes).split(self.num_classes // self.num_task)
        exposed_classes = torch.zeros(self.num_classes) - torch.inf
        for task in range(self.num_task):
            self.logger.info(f"\nTask {task} exposed classes: {exposed_classes}")
            self.net.train()
            exposed_classes[self.tasks[task]] = 0
            for epoch in range(self.epochs):
                train_acc = 0
                train_loss = 0
                count = 0
                for images, labels in self.get_train_epoch(self.tasks[task]):
                    images, labels = images.to(self.device), labels.to(self.device)
                    images = F.interpolate(images, size=self.inp_size, align_corners=False) 
                    if self.dsa:
                        images, labels = self.diff_augment(images, labels, self.dsa_strategy_net)
                    if self.mixup_net == 'cut':
                        images, targets = self.cutmix(images, labels)
                    else:
                        targets = F.one_hot(labels, self.num_classes).float()
                    self.optimizer.zero_grad()
                    logits, _ = self.net(images)
                    logits = logits * exposed_classes.unsqueeze(0).to(self.device)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    train_acc += (logits.argmax(1) == labels).float().sum()
                    train_loss += loss.item() * images.shape[0]
                    count += images.shape[0]
                    self.step(self.optimizer)
                train_acc /= count
                train_loss /= count
            self.net.eval()
            for test_task in range(task+1):
                test_acc = 0
                count = 0
                for images, labels in self.get_test_epoch(self.tasks[test_task]):
                    images, labels = images.to(self.device), labels.to(self.device)
                    images = F.interpolate(images, size=self.inp_size, mode='bilinear', align_corners=True)
                    logits, _ = self.net(images)
                    logits = logits * exposed_classes.unsqueeze(0).to(self.device)
                    acc = (logits.argmax(1) == labels).float().sum()
                    test_acc += acc.item()
                    count += images.shape[0]
                test_acc /= count
                self.logger.info({f"Test accuracy for task {test_task}: {test_acc}"})

    def visualize_features(self):
        # pretrain the network
        self.test_model_accuracy()
        self.net.eval()
        # get the features of real images
        real_features = [[] for _ in range(self.num_classes)]
        for images, labels in self.get_train_epoch(torch.arange(self.num_classes)):
            images, labels = images.to(self.device), labels.to(self.device)
            images = F.interpolate(images, size=self.inp_size, align_corners=True) 
            _, features = self.net(images)
            for i in range(self.num_classes):
                real_features[i].append(features[:,0][labels == i])
        real_features = [torch.stack(real_features[i]) for i in range(self.num_classes)]
        real_features = torch.stack(real_features)
        real_features = real_features[(real_features[...,:,:]==0).all(dim=(-2,-1)).logical_not()]
        # get the features of synthetic images
        syn_features = [[] for _ in range(self.num_classes)]
        for images, labels in self.get_synthetic_epoch(torch.arange(self.num_classes)):
            images, labels = images.to(self.device), labels.to(self.device)
            images = F.interpolate(images, size=self.inp_size, align_corners=False) 
            _, features = self.net(images)
            for i in range(self.num_classes):
                syn_features[i].append(features[:,0][labels == i])
        syn_features = [torch.stack(syn_features[i]) for i in range(self.num_classes)]
        syn_features = torch.stack(syn_features)
        syn_features = syn_features[(syn_features[...,:,:]==0).all(dim=-1).all(dim=-1).logical_not()]

        # t-SNE visualization
        real_features = real_features.reshape(-1, real_features.shape[-1])
        syn_features = syn_features.reshape(-1, syn_features.shape[-1])
        len_real = len(real_features)
        features = torch.cat([real_features, syn_features])
        labels = torch.cat([torch.zeros(real_features.shape[0]), torch.ones(syn_features.shape[0])])
        features = TSNE(n_components=2).fit_transform(features)
        real_features = features[:len_real]
        syn_features = features[len_real:]
        plt.scatter(real_features[:,0], real_features[:,1], c=labels[:len_real], marker='o', size=1, alpha=0.3)
        plt.scatter(syn_features[:,0], syn_features[:,1], c=labels[len_real:], marker='x', size=10)
        plt.savefig(f"{self.path}/tsne.png")

    def _is_main_process(self):
        return self.local_rank == 0
    
    def _setup_for_distributed(self, is_main_process):
        """
        This function disables self.logger.infoing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_main_process or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def load_dataset(self):
        self.channel, self.im_size, self.num_classes, self.class_names, self.mean, self.std, self.trainset, self.testset\
              = get_dataset(self.dataset, self.data_path)
        self.train_images = [[] for _ in range(self.num_classes)]
        self.train_labels = [[] for _ in range(self.num_classes)]
        for idx, (image, label) in enumerate(self.trainset):
            self.train_images[label].append(image)
            self.train_labels[label].append(label)
        self.train_images = [torch.stack(self.train_images[i]) for i in range(self.num_classes)]
        self.train_labels = [torch.tensor(self.train_labels[i]) for i in range(self.num_classes)]
        self.train_images = torch.stack(self.train_images)
        self.train_labels = torch.stack(self.train_labels)

        self.test_images = [[] for _ in range(self.num_classes)]
        self.test_labels = [[] for _ in range(self.num_classes)]
        for image, label in self.testset:
            self.test_images[label].append(image)
            self.test_labels[label].append(label)
        self.test_images = [torch.stack(self.test_images[i]) for i in range(self.num_classes)]
        self.test_labels = [torch.tensor(self.test_labels[i]) for i in range(self.num_classes)]
        self.test_images = torch.stack(self.test_images)
        self.test_labels = torch.stack(self.test_labels)
        self.logger.info(f"Dataset {self.dataset} loaded.")
        
    @torch.no_grad
    def init_synthetic_images(self):
        self.synthetic_images = torch.randn(self.num_classes, self.n_ipc, self.channel, *(self.im_size))
        self.synthetic_labels = torch.arange(self.num_classes).unsqueeze(1).repeat(1,self.n_ipc)
        if 'aug' in self.init:
            if 'real' in self.init:
                for c in range(self.num_classes):
                    image, label = self.get_train_images(torch.tensor([c]), self.n_ipc * self.factor ** 2)
                    image, label = self.pack(image, label, self.factor)
                    self.synthetic_images[c] = image.to(self.device)
                    self.synthetic_labels[c] = label.to(self.device)
        else:
            if 'real' in self.init:
                for c in range(self.num_classes):
                    image, label = self.get_train_images(torch.tensor([c]), self.n_ipc)
                    self.synthetic_images[c] = image.to(self.device)
                    self.synthetic_labels[c] = label.to(self.device)
        self.synthetic_images.requires_grad = True
        self.logger.info(f"Synthetic images initialized with {self.init} method.")

    def get_train_images(self, classes, batchsize=None, transform=None, seed=None):
        if batchsize is None:
            batchsize = self.batchsize
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        images = self.train_images[classes].flatten(0, 1)
        labels = self.train_labels[classes].flatten()
        if seed is not None: g = torch.Generator().manual_seed(seed)
        else: g = None
        indices = torch.randperm(images.shape[0], generator=g)[:batchsize]
        images = images[indices]
        labels = labels[indices]
        return images.to(self.device), labels.to(self.device)
    
    def get_test_images(self, classes, batchsize=None, transform=None, seed=None):
        if batchsize is None:
            batchsize = self.batchsize
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        images = self.test_images[classes].flatten(0, 1)
        labels = self.test_labels[classes].flatten()
        if seed is not None: g = torch.Generator().manual_seed(seed)
        else: g = None
        indices = torch.randperm(images.shape[0], generator=g)[:batchsize]
        images = images[indices]
        labels = labels[indices]
        return images.to(self.device), labels.to(self.device)
    
    def get_synthetic_images(self, classes, batchsize=None, transform=None, seed=None):
        if batchsize is None:
            batchsize = self.batchsize
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        images = self.synthetic_images[classes].flatten(0, 1)
        labels = self.synthetic_labels[classes].flatten()
        if seed is not None: g = torch.Generator().manual_seed(seed)
        else: g = None
        if 'aug' in self.init: images, labels = self.unpack(images, labels, self.factor)
        indices = torch.randperm(images.shape[0], generator=g)[:batchsize]
        images = images[indices]
        labels = labels[indices]
        return images, labels

    @torch.no_grad
    def get_train_epoch(self, classes, batchsize=None, random=True, seed=None, transform=None, distributed=None):
        if batchsize is None:
            batchsize = self.batchsize
        if distributed is None:
            distributed = self.distributed
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        images = self.train_images[classes].flatten(0, 1)
        labels = self.train_labels[classes].flatten(0, 1)
        if random:
            if seed is not None: g = torch.Generator().manual_seed(seed)
            else: g = None
            indice = torch.randperm(images.shape[0], generator=g)
        else:
            indice = torch.arange(images.shape[0])
        if distributed:
            batchsize = batchsize // self.world_size
            indice = indice[:len(indice)-(len(indice) % self.world_size)] # make sure the number of images is divisible by world_size
            indice = indice[self.global_rank::self.world_size]            # only use the images for this process
        images = images[indice]
        labels = labels[indice]
        if transform is not None:
            for i in range(0, images.shape[0], batchsize):
                yield transform((images[i:i+batchsize].to(self.device), labels[i:i+batchsize].to(self.device)))
        else:
            for i in range(0, images.shape[0], batchsize):
                yield images[i:i+batchsize].to(self.device), labels[i:i+batchsize].to(self.device)
            
    @torch.no_grad
    def get_synthetic_epoch(self, classes, batchsize=None, random=True, seed=None, transform=None, distributed=None):
        if batchsize is None:
            batchsize = self.batchsize
        if distributed is None:
            distributed = self.distributed
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        images = self.synthetic_images[classes].flatten(0, 1)
        labels = self.synthetic_labels[classes].flatten(0, 1)
        if 'aug' in self.init: images, labels = self.unpack(images, labels, self.factor)
        if random:
            if seed is not None: g = torch.Generator().manual_seed(seed)
            else: g = None
            indice = torch.randperm(images.shape[0], generator=g)
        else:
            indice = torch.arange(images.shape[0])
        if distributed:
            batchsize = batchsize // self.world_size
            indice = indice[:len(indice)-(len(indice) % self.world_size)] # make sure the number of images is divisible by world_size
            indice = indice[self.global_rank::self.world_size] # only use the images for this process
        images = images[indice]
        labels = labels[indice]
        if transform is not None:
            for i in range(0, images.shape[0], batchsize):
                yield transform((images[i:i+batchsize].to(self.device), labels[i:i+batchsize].to(self.device)))
        else:
            for i in range(0, images.shape[0], batchsize):
                yield images[i:i+batchsize].to(self.device), labels[i:i+batchsize].to(self.device)

    
    @torch.no_grad
    def get_test_epoch(self, classes, batchsize=None, random=True, seed=None, transform=None, distributed=None):
        if batchsize is None:
            batchsize = self.batchsize
        if distributed is None:
            distributed = self.distributed
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        images = self.test_images[classes].flatten(0, 1)
        labels = self.test_labels[classes].flatten(0, 1)
        if random:
            if seed is not None: g = torch.Generator().manual_seed(seed)
            else: g = None
            indice = torch.randperm(images.shape[0], generator=g)
        else:
            indice = torch.arange(images.shape[0])
        if distributed:
            batchsize = batchsize // self.world_size
            indice = indice[:len(indice)-(len(indice) % self.world_size)] # make sure the number of images is divisible by world_size
            indice = indice[self.global_rank::self.world_size] # only use the images for this process
        images = images[indice]
        labels = labels[indice]
        if transform is not None:
            for i in range(0, images.shape[0], batchsize):
                    yield transform((images[i:i+batchsize].to(self.device), labels[i:i+batchsize].to(self.device)))
        else:
            for i in range(0, images.shape[0], batchsize):
                yield images[i:i+batchsize].to(self.device), labels[i:i+batchsize].to(self.device)

    def unpack(self, image, label, factor=2):
        im_size = self.im_size[0] // factor, self.im_size[1] // factor
        image = torch.cat(torch.cat(image.split(im_size[0], dim=2),dim=0).split(im_size[1], dim=3), dim=0)
        image = F.interpolate(image, size=self.im_size, mode='bilinear', align_corners=True)
        label = label.repeat(factor**2)
        return image, label
    
    def pack(self, image, label, factor=2):
        num_images = int(image.shape[0] // factor ** 2)
        image = image[:num_images * factor ** 2]
        im_size = self.im_size[0] // factor, self.im_size[1] // factor
        image = F.interpolate(image, size=im_size, mode='bilinear', align_corners=True)
        image = torch.cat(image.split(num_images * factor, dim=0), dim=2)
        image = torch.cat(image.split(num_images, dim=0), dim=3)
        label = label[:num_images]
        return image, label

    def step(self, optimizer, reduce='sum'):
        '''
        Update the distributed network parameters.
        '''
        self.dist_barrier()
        with torch.no_grad():   
            if self.distributed:
                _device = optimizer.param_groups[0]['params'][0].device
                for p in optimizer.param_groups[0]['params']:
                    if p.grad is None or p.grad.numel() == 0:
                        p.grad = torch.zeros_like(p)
                    grad = p.grad; grad = grad.to(self.device);
                    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                    grad = grad.to(_device); p.grad = grad;
                    if reduce == 'mean':
                        optimizer.param_groups[0]['params'].grad /= self.world_size
        optimizer.step()
    
    def _visualize(self, path, it):
        img = self.synthetic_images.clone().detach()
        img = img.flatten(0, 1)
        size = img.shape[0]
        for ch in range(self.channel):
            img[:, ch] = img[:, ch] * self.std[ch] + self.mean[ch]
        img = img.clip(0, 1)
        save_image(img, f"{path}/synthetic_images_{it}.png", nrow=ceil(sqrt(size)))
    
    def visualize(self, path, it):
        threading.Thread(target=self._visualize, args=(path, it)).start()
        self.logger.info(f"Synthetic images saved to {path}/synthetic_images_{it}.png")

    def _save_img(self, path):
        payload = {}
        payload['synthetic_images'] = self.synthetic_images.cpu()
        payload['init'] = self.init
        payload['factor'] = self.factor
        payload['n_ipc'] = self.n_ipc
        payload['iteration'] = self.start_iteration
        # payload['method'] = self.method
        self.logger.debug(f"payload:")
        self.logger.debug(f"{self.synthetic_images.shape}")
        self.logger.debug(f"{self.init}")
        self.logger.debug(f"{self.factor}")
        self.logger.debug(f"{self.n_ipc}")
        self.logger.debug(f"{self.start_iteration}")
        # self.logger.debug(f"{self.method}")
        torch.save(payload, os.path.join(path, "synthetic_images.pth"))

    def save_img(self, path):
        threading.Thread(target=self._save_img, args=(path,)).start()
        self.logger.info(f"Synthetic images saved to {path}/synthetic_images.pth")

    def load_img(self, path):
        try:
            payload = torch.load(os.path.join(path, "synthetic_images.pth"))
            if isinstance(payload, dict):
                self.init = payload['init']
                self.factor = payload['factor']
                self.n_ipc = payload['n_ipc']
                self.start_iteration = payload['iteration']
                # self.method = payload['method']
                self.synthetic_images = payload['synthetic_images']
                # self.synthetic_images.to(self.device)
                self.synthetic_images.requires_grad = True
                self.logger.debug(f"payload:")
                self.logger.debug(f"{self.synthetic_images.shape}")
                self.logger.debug(f"{self.init}")
                self.logger.debug(f"{self.factor}")
                self.logger.debug(f"{self.n_ipc}")
                self.logger.debug(f"{self.start_iteration}")
                self.synthetic_labels = torch.arange(self.num_classes).unsqueeze(1).repeat(1,self.n_ipc)
                # self.logger.debug(f"{self.method}")
            else:
                self.logger.warn(f"Synthetic images has wrong format. This may cause error with wrong parameters. Forced loading...")
                self.synthetic_images = payload.reshape(self.num_classes, self.n_ipc, self.channel, *self.im_size)
                self.synthetic_images.requires_grad = True
            # self.synthetic_images = self.synthetic_images.to(self.device)
            self.image_optimizer = torch.optim.SGD((self.synthetic_images,), lr=self.lr_img, momentum=0.5)
            self.logger.info(f"Images loaded from {path}")
        except:
            self.logger.info(f"Synthetic images not found in {path}. Initializing...")
            self.init_synthetic_images()
            self.save_img(path)
            self.image_optimizer = torch.optim.SGD((self.synthetic_images,), lr=self.lr_img, momentum=0.5)
            self.logger.info(f"Images initialized and saved to {path}")

    def dist_seed(self):
        seed = int(torch.randn(1).item() * 0xFFFFFFFF) + self.global_rank
        return seed
    
    def syn_seed(self):
        seed = int(torch.randn(1).item() * 0xFFFFFFFF)
        return seed

    def dist_barrier(self):
        if self.distributed: dist.barrier()
    
    def __del__(self):
        if self.distributed:
            dist.destroy_process_group()
        torch.cuda.empty_cache()

class TrainTransform():
    def __init__(self, method : DC_BASE):
        self.device = method.device
        self.inp_size = method.inp_size
        self.num_classes = method.num_classes
        self.dsa = method.dsa
        self.dsa_strategy_net = method.dsa_strategy_net
        self.mixup_net = method.mixup_net
        
    def __call__(self, data, seed=None):
        images, labels = data
        if self.dsa: images, labels = Transform.diff_augment(images, labels, self.dsa_strategy_net, seed=seed)
        if self.mixup_net == 'cut': images, targets = Transform.cutmix(images, labels, self.num_classes)
        else: targets = F.one_hot(labels, self.num_classes).float()
        images = F.interpolate(images, size=self.inp_size, mode='bilinear', align_corners=True)
        return images, targets
    
class IDCTransform(TrainTransform):
    def __init__(self, method : DC_BASE):
        super().__init__(method)
    
    def __call__(self, data, seed=None):
        images, labels = data
        if self.dsa: images, labels = Transform.diff_augment_idc(images, labels, self.dsa_strategy_net, seed=seed)
        if self.mixup_net == 'cut': images, targets = Transform.cutmix(images, labels, self.num_classes)
        else: targets = F.one_hot(labels, self.num_classes).float()
        images = F.interpolate(images, size=self.inp_size, mode='bilinear', align_corners=True)
        return images, targets
    
class TestTransform(TrainTransform):
    def __init__(self, method : DC_BASE):
        super().__init__(method)

    def __call__(self, data, seed=None):
        images, labels = data
        images = F.interpolate(images, size=self.inp_size, mode='bilinear', align_corners=True)
        return images, labels