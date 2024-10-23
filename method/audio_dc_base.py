import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchaudio
from utils.utils import get_dataset
from utils.clustering_sample_selection import Clustering_Sample_Selection as CSS
from method.dc_base import DC_BASE

class AUDIO_DC_BASE(DC_BASE):
    def __init__(self, args) -> None:
        super().__init__(args)

    def test_model_accuracy(self):
        self.net.train()
        max_acc = 0
        patient = 0
        patience = 50
        for epoch in range(self.epochs):
            train_epoch = self.get_train_epoch(torch.arange(self.num_classes))
            for text, targets in train_epoch:
                text, targets = text.detach().to(self.device), targets.detach().to(self.device)
                logits, _ = self.net(text)
                loss = self.net.loss_fn(logits, targets)
                acc = (logits.argmax(1) == targets).float().sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.step(self.optimizer)
            if epoch % 10 == 0:
                with torch.no_grad():
                    test_acc = torch.zeros(1).to(self.device)
                    count = torch.zeros(1).to(self.device)
                    self.net.eval()
                    test_epoch = self.get_test_epoch(torch.arange(self.num_classes))
                    for text, labels in test_epoch:
                        text, labels = text.to(self.device), labels.to(self.device)
                        logits, _ = self.net(text)
                        acc = (logits.argmax(1) == labels).float().sum()
                        test_acc += acc.item()
                        count += text.shape[0]
                    if self.distributed:
                        dist.all_reduce(test_acc, op=dist.ReduceOp.SUM)
                        dist.all_reduce(count, op=dist.ReduceOp.SUM)
                    test_acc /= count
                    self.logger.debug(f"Test accuracy: {test_acc}")
                    if test_acc > max_acc:
                        max_acc = test_acc
                        patient = 0
                    else:
                        patient += 1
                        if patient >= patience:
                            break
                sys.stdout.flush()
        self.logger.info(f"Best accuracy: {max_acc}")

    def test_condensed_images(self):
        self.net.reset_parameters()
        self.load_img(self.path)
        self.visualize(self.path, 'test')
        # self.net.train()
        self.net.eval()

        num_params = sum(p.numel() for p in self.net.parameters())
        num_learnable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        self.logger.info(f"Number of parameters: {num_params}")
        self.logger.info(f"Number of learnable parameters: {num_learnable_params}")

        best_acc = 0
        patient = 0
        patience = 50
        for epoch in range(self.epochs):
            syn_epoch = self.get_synthetic_epoch(torch.arange(self.num_classes))
            i = 0
            for images, labels in syn_epoch:
                self.logger.info(f"Epoch {epoch}, batch {i}")
                i += 1
                images, labels = images.detach().to(self.device), labels.detach().to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.net(images)
                loss = F.cross_entropy(logits, labels)
                acc = (logits.argmax(1) == labels).float().sum()
                loss.backward()
                self.step(self.optimizer)
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    test_acc = torch.zeros(1).to(self.device)
                    count = torch.zeros(1).to(self.device)
                    self.net.eval()
                    test_epoch = self.get_test_epoch(torch.arange(self.num_classes))
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
                        patient = 0
                    else:
                        patient += 1
                        if patient >= patience:
                            break
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
                    logits, _ = self.net(images)
                    logits = logits * exposed_classes.unsqueeze(0).to(self.device)
                    acc = (logits.argmax(1) == labels).float().sum()
                    test_acc += acc.item()
                    count += images.shape[0]
                test_acc /= count
                self.logger.info({f"Test accuracy for task {test_task}: {test_acc}"})

    def load_dataset(self):
        self.channel, self.max_length, self.num_classes, self.class_names, self.mean, self.std, trainset, testset\
              = get_dataset(self.dataset, self.data_path)

        self.train_images = [[] for _ in range(self.num_classes)]
        self.train_labels = [[] for _ in range(self.num_classes)]
        for audio, label in trainset:
            self.train_images[label].append(audio)
            self.train_labels[label].append(label)
        self.train_images = [torch.stack(self.train_images[i]) for i in range(self.num_classes)]
        self.train_labels = [torch.tensor(self.train_labels[i]) for i in range(self.num_classes)]
        self.train_images = torch.stack(self.train_images)
        self.train_labels = torch.stack(self.train_labels)

        self.test_images = [[] for _ in range(self.num_classes)]
        self.test_labels = [[] for _ in range(self.num_classes)]
        for audio, label in testset:
            self.test_images[label].append(audio)
            self.test_labels[label].append(label)
        self.test_images = [torch.stack(self.test_images[i]) for i in range(self.num_classes)]
        self.test_labels = [torch.tensor(self.test_labels[i]) for i in range(self.num_classes)]
        self.test_images = torch.stack(self.test_images)
        self.test_labels = torch.stack(self.test_labels)
        self.logger.info(f"Dataset {self.dataset} loaded.")
        self.logger.info(f"Parameters: {self.channel} channels, {self.max_length} image size, {self.num_classes} classes.")
        self.logger.info(f"Trainset shape: {self.train_images.shape}, Testset shape: {self.test_images.shape}")
        
        
    @torch.no_grad
    def init_synthetic_images(self):
        self.synthetic_images = torch.randn(self.num_classes, self.n_ipc, self.channel, self.max_length, device=self.device)
        self.synthetic_labels = torch.arange(self.num_classes, device=self.device).unsqueeze(1).repeat(1,self.n_ipc)
        print(self.train_images.shape)
        if 'real' in self.init:
            for c in range(self.num_classes):
                image, label = self.get_train_images(torch.tensor([c]), self.n_ipc)
                self.synthetic_images[c] = image.to(self.device)
                self.synthetic_labels[c] = label.to(self.device)
        elif 'kmeans' in self.init:
            for c in range(self.num_classes):
                print(self.synthetic_images[c].shape)
                imgs = self.train_images[c]
                imgs = imgs.to(self.device)
                q_idxs, _ = CSS(imgs, self.net.embed).query(self.n_ipc)
                self.synthetic_images[c] = imgs[q_idxs].detach().clone()
        self.synthetic_images.requires_grad = True
        self.logger.info(f'Initialization Done with {self.init}!')

    def _visualize(self, path, it):
        # save synthetic images as audio files
        for c in range(self.num_classes):
            for i in range(self.n_ipc):
                audio = self.synthetic_images[c, i].detach().cpu()
                torchaudio.save(f'{path}/{it}_class_{c}_sample_{i}.wav', audio, 16000)

    def get_synthetic_images(self, classes, batchsize=None, transform=None, seed=None):
        if batchsize is None:
            batchsize = self.batchsize
        if not isinstance(classes, torch.Tensor):
            classes = torch.tensor(classes)
        images = self.synthetic_images[classes].flatten(0, 1)
        labels = self.synthetic_labels[classes].flatten()
        if seed is not None: g = torch.Generator().manual_seed(seed)
        else: g = None
        indices = torch.randperm(images.shape[0], generator=g)[:batchsize]
        images = images[indices]
        labels = labels[indices]
        return images, labels

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
