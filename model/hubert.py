import torch
import torch.nn as nn
from model._model_base import _MODEL
from transformers import AutoModel, AutoTokenizer, HubertModel

class Hubert_FT(_MODEL):
    def __init__(self, num_classes, max_length, pretrained='facebook/hubert-base-ls960'):
        super(Hubert_FT, self).__init__()
        self.hubert = HubertModel.from_pretrained(pretrained)
        self.classifier = torch.nn.Linear(self.hubert.config.hidden_size, num_classes)
        self.pretrained = pretrained
        self.max_length = max_length

    def forward(self, inputs):
        with torch.no_grad():
            min_ = inputs.min(dim=0, keepdim=True)[0]
            max_ = inputs.max(dim=0, keepdim=True)[0]
        inputs = (inputs - min_) / (max_ - min_ + 1e-6)
        inputs = inputs[:,0,:] # Only use the first channel
        embeddings = self.hubert.feature_extractor(inputs)
        outputs = self.hubert.feature_projection(embeddings.transpose(1, 2))
        features = []
        for l in self.hubert.encoder.layers:
            outputs = l(outputs)[0]
            features.append(outputs)
        outputs = self.classifier(outputs.mean(dim=1))
        return outputs, features
    
    def embed(self, inputs):
        with torch.no_grad():
            min_ = inputs.min(dim=0, keepdim=True)[0]
            max_ = inputs.max(dim=0, keepdim=True)[0]
        inputs = (inputs - min_) / (max_ - min_ + 1e-6)
        inputs = inputs[:,0,:] # Only use the first channel
        embeddings = self.hubert.feature_extractor(inputs)
        outputs = self.hubert.feature_projection(embeddings.transpose(1, 2))
        features = []
        for l in self.hubert.encoder.layers:
            outputs = l(outputs)[0]
            features.append(outputs)
        return outputs

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
        
    def reset_parameters(self):
        device = self.parameters().__next__().device
        self.hubert = AutoModel.from_pretrained(self.pretrained).to(device)

    def length(self):
        return 12