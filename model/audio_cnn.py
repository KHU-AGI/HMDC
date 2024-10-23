import torch
import torch.nn as nn
import torch.optim as optim
from model._model_base import _MODEL

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=21, stride=10, padding=0)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=15, stride=7, padding=0)
        self.pool2 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(64, 128, kernel_size=11, stride=5, padding=0)
        self.pool3 = nn.MaxPool1d(2)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, inputs):
        # inputs = (batch, Channel, Length)
        # Normalization into [0, 1]
        with torch.no_grad():
            min_ = inputs.min(dim=0, keepdim=True)[0]
            max_ = inputs.max(dim=0, keepdim=True)[0]
        inputs = (inputs - min_) / (max_ - min_ + 1e-6)
        features = []
        x = nn.functional.relu(self.conv1(inputs))
        x = self.pool1(x)
        features.append(x)
        x = nn.functional.relu(x + self.conv2(x))
        features.append(x)
        x = nn.functional.relu(self.conv3(x))
        x = self.pool2(x)
        features.append(x)
        x = nn.functional.relu(x + self.conv4(x))
        features.append(x)
        x = nn.functional.relu(self.conv5(x))
        x = self.pool3(x)
        features.append(x)
        x = nn.functional.relu(x + self.conv6(x))
        features.append(x)
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x, features
    
    def embed(self, inputs):
        # inputs = inputs.permute(0, 2, 1)
        with torch.no_grad():
            min_ = inputs.min(dim=0, keepdim=True)[0]
            max_ = inputs.max(dim=0, keepdim=True)[0]
        inputs = (inputs - min_) / (max_ - min_ + 1e-6)
        x = nn.functional.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = nn.functional.relu(x + self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool2(x)
        x = nn.functional.relu(x + self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = self.pool3(x)
        x = nn.functional.relu(x + self.conv6(x))
        x = x.mean(dim=-1)
        return x
    
    def length(self):
        return 6
    
    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
    
    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()