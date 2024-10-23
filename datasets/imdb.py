import os
import torch
import tarfile
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

class IMDB(Dataset):
    def __init__(self, path, tokenizer, max_length, download=True, train=True, transform=None, target_transform=None):
        self.url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        self.filename = 'aclImdb_v1.tar.gz'
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        if not os.path.exists(os.path.join(path, 'aclImdb')):
            if download:
                print('Downloading IMDB dataset...')
                download_url(self.url, path, self.filename)
                with tarfile.open(os.path.join(path, self.filename), 'r:gz') as tar:
                    tar.extractall(path)
            else: raise ValueError(f'Dataset not found at {path}')
        self.samples = []
        self.targets = []
        if train:
            for label in ['pos', 'neg']:
                folder = os.path.join(path, 'aclImdb', 'train', label)
                for file in os.listdir(folder):
                    with open(os.path.join(folder, file), 'r') as f:
                        self.samples.append(f.read())
                    self.targets.append(1 if label == 'pos' else 0)
        else:
            for label in ['pos', 'neg']:
                folder = os.path.join(path, 'aclImdb', 'test', label)
                for file in os.listdir(folder):
                    with open(os.path.join(folder, file), 'r') as f:
                        self.samples.append(f.read())
                    self.targets.append(1 if label == 'pos' else 0)
        
        self.classes = ['neg', 'pos']
            
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        text, label = self.samples[idx], self.targets[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt').input_ids.squeeze()
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            label = self.target_transform(label)
        return inputs, label