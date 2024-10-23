import os
import torch
import zipfile
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchaudio import load, transforms

class ESC10(Dataset):
    def __init__(self, path, download=True, train=True, transform=None, target_transform=None):
        super(ESC10, self).__init__()
        self.url = 'https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master'
        self.filename = 'esc50.zip'
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        if not os.path.exists(os.path.join(path, 'ESC-50-master')):
            os.makedirs(os.path.join(path, 'ESC-50-master'))
            if download:
                print('Downloading ESC-50 dataset...')
                download_url(self.url, path, self.filename)
                with zipfile.ZipFile(os.path.join(path, self.filename), 'r') as zip_ref:
                    zip_ref.extractall(path)
            else: raise ValueError(f'Dataset not found at {path}')
        

        data_list = pd.read_csv(os.path.join(path, 'ESC-50-master', 'meta', 'esc50.csv'))
        data_list = data_list[data_list['esc10'] == True]
        self.classes = data_list['category'].unique().tolist()
        self.samples = []
        self.targets = []
        for i, table in data_list.iterrows():
            self.samples.append(os.path.join(path, 'ESC-50-master', 'audio', table['filename']))
            self.targets.append(self.classes.index(table['category']))
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        samples = []
        targets = []
        generator = torch.Generator().manual_seed(42)
        for c in self.targets.unique():
            _in_class_idxcs = (self.targets == c).nonzero(as_tuple=False).view(-1)
            random_idxcs = torch.randperm(len(_in_class_idxcs), generator=generator)
            if train:
                _in_class_idxcs = _in_class_idxcs[random_idxcs][:int(0.8*len(random_idxcs))]
            else:
                _in_class_idxcs = _in_class_idxcs[random_idxcs][int(0.8*len(random_idxcs)):]
            for idx in _in_class_idxcs:
                samples.append(self.samples[idx])
                targets.append(self.targets[idx])
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        path, label = self.samples[idx], self.targets[idx]
        waveform, sample_rate = load(path)
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        if self.transform:
            waveform = self.transform(waveform)
        if self.target_transform:
            label = self.target_transform(label)
        return waveform, label
    