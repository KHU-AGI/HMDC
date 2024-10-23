import torch
import torch.nn as nn
from model._model_base import _MODEL
from transformers import BertModel, BertTokenizer

class Bert_FT(_MODEL):
    def __init__(self, num_classes, max_length, pretrained='bert-base-uncased'):
        super(Bert_FT, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        self.max_length = max_length
        self.pretrained = pretrained

    def forward(self, inputs):
        if inputs.dim() == 2:
            outputs = self.bert.embeddings.word_embeddings(inputs)
        elif inputs.dim() == 3:
            outputs = inputs @ self.bert.embeddings.word_embeddings.weight
        features = []
        for layer in self.bert.encoder.layer:
            outputs = layer(outputs)[0]
            features.append(outputs)
        return self.classifier(outputs[:, 0]), features

    def embed(self, inputs):
        if inputs.dim() == 2:
            outputs = self.bert.embeddings.word_embeddings(inputs)
        elif inputs.dim() == 3:
            outputs = inputs @ self.bert.embeddings.word_embeddings.weight
        for layer in self.bert.encoder.layer:
            outputs = layer(outputs)[0]
        return outputs[:, 0]

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets)
    
    def reset_parameters(self):
        device = next(self.parameters()).device
        self.bert = BertModel.from_pretrained(self.pretrained).to(device)

    def length(self):
        return 12