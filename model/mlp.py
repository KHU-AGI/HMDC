import torch
import torch.nn as nn
import torch.nn.functional as F
from model._model_base import _MODEL

''' MLP '''
class MLP(_MODEL):
    def __init__(self, channel, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28*28*1 if channel==1 else 32*32*3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

    def forward(self, x):
        layer_features = []
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        layer_features.append(out.clone())
        out = F.relu(self.fc_2(out))
        layer_features.append(out.clone())
        out = self.fc_3(out)
        layer_features = torch.stack(layer_features, dim=1)
        return out, layer_features
    
    def gradient(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        img = x.clone().detach().requires_grad_(True)
        layer_features = []
        out = img.view(img.size(0), -1)
        out = F.relu(self.fc_1(out))
        out.retain_grad()
        layer_features.append(out)
        out = F.relu(self.fc_2(out))
        out.retain_grad()
        layer_features.append(out)
        out = self.fc_3(out)
        out.retain_grad()
        layer_features = torch.stack(layer_features, dim=1)
        loss = self.loss_fn(out, targets)
        loss.backward()
        return img.grad, [f.grad for f in layer_features]
    
    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)

    def reset_parameters(self):
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()
        self.fc_3.reset_parameters()
        
    def forward_layer(self, x, layer):
        if layer == 0:
            out = x.view(x.size(0), -1)
            out = F.relu(self.fc_1(out))
        elif layer == 1:
            out = F.relu(self.fc_2(x))
        elif layer == 2:
            out = self.fc_3(x)
        else:
            raise NotImplementedError
        return out
    
    def embed(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        return out

    def length(self):
        return 3