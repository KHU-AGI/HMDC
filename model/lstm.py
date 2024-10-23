import torch
import torch.nn as nn
from model._model_base import _MODEL

class LSTM(_MODEL):
    def __init__(self, num_classes, num_layers, hidden_size, max_length):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(30522, 128)
        self.lstm = nn.ModuleList()
        for _ in range(num_layers):
            self.lstm.append(nn.LSTM(128, hidden_size, 1, batch_first=True,))
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_length = max_length

    def forward(self, inputs):
        if inputs.dim() == 2:
            x = self.embedding(inputs)
        elif inputs.dim() == 3:
            x = inputs @ self.embedding.weight
        features = []
        for i in range(self.num_layers):
            x, (hn, cn) = self.lstm[i](x)
            features.append(x)
        x = self.fc(x.mean(1))
        return x, features
    
    def embed(self, inputs):
        if inputs.dim() == 2:
            x = self.embedding(inputs)
        elif inputs.dim() == 3:
            x = inputs @ self.embedding.weight
        for i in range(self.num_layers):
            x, (hn, cn) = self.lstm[i](x)
        return x
    
    def reset_parameters(self):
        self.lstm = nn.ModuleList()
        device = self.parameters().__next__().device
        for _ in range(self.num_layers):
            self.lstm.append(nn.LSTM(128, self.hidden_size, 1, batch_first=True,)).to(device)
        self.fc = nn.Linear(self.hidden_size, self.num_classes).to(device)

    def length(self):
        return self.num_layers
    
    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)