import copy
import timm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model._model_base import _MODEL

class ResNet(_MODEL):
    def __init__(self, model_name, num_classes):
        super(ResNet, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def reset_parameters(self):
        self.model.fc.reset_parameters()

    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
    
    def forward(self, x):
        layer_features = []

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)

        layer_features.append(x)
        x = self.model.layer1(x)
        layer_features.append(x)
        x = self.model.layer2(x)
        layer_features.append(x)
        x = self.model.layer3(x)
        layer_features.append(x)
        x = self.model.layer4(x)
        layer_features.append(x)

        x = self.model.global_pool(x)
        x = self.model.fc(x)
        
        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        for param in self.parameters():
            param.requires_grad = True

        img = x.clone().detach().requires_grad_(True)
        layer_features = []
        required_grad = x.requires_grad

        x.retain_grad()
        layer_features.append(img)
        x = self.model.layer1(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer2(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer3(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer4(x)
        x.retain_grad()
        layer_features.append(x)

        x = self.model.global_pool(x)
        x = self.model.fc(x)

        loss = self.loss_fn(x, targets)
        loss.backward()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        return img.grad, [f.grad for f in layer_features]
    
    def forward_layer(self, x, layer):
        if layer == 0:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)
        elif layer == 1:
            x = self.model.layer1(x)
        elif layer == 2:
            x = self.model.layer2(x)
        elif layer == 3:
            x = self.model.layer3(x)
        elif layer == 4:
            x = self.model.layer4(x)
        else:
            raise NotImplementedError
        return x
    
    # @torch.compile
    # def _compiled_forward(self, x):
    #     layer_features = []
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.act1(x)
    #     x = self.model.maxpool(x)
    #     layer_features.append(x)
    #     x = self.model.layer1(x)
    #     layer_features.append(x)
    #     x = self.model.layer2(x)
    #     layer_features.append(x)
    #     x = self.model.layer3(x)
    #     layer_features.append(x)
    #     x = self.model.layer4(x)
    #     layer_features.append(x)
    #     x = self.model.global_pool(x)
    #     x = self.model.fc(x)
    #     return x, layer_features
    
    # @torch.compile
    # def _compiled_embed(self, x):
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.act1(x)
    #     x = self.model.maxpool(x)
    #     x = self.model.layer1(x)
    #     x = self.model.layer2(x)
    #     x = self.model.layer3(x)
    #     x = self.model.layer4(x)
    #     x = self.model.global_pool(x)
        # return x
    
    def length(self):
        return 5
    
class ResNet_FT(_MODEL):
    def __init__(self, model_name, num_classes):
        super(ResNet_FT, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.initial_model = copy.deepcopy(self.model.state_dict())

    def reset_parameters(self):
        '''
        Reset the model into pretrained state.
        '''
        self.model.load_state_dict(self.initial_model)
        self.model.fc.reset_parameters()

    def hard_reset_parameters(self):
        '''
        Reset the whole parameters of the model.
        '''
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param.data, 0.0)
            else:
                nn.init.kaiming_normal_(param.data)

    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
    
    def forward(self, x):
        layer_features = []
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        layer_features.append(x)
        x = self.model.layer1(x)
        layer_features.append(x)
        x = self.model.layer2(x)
        layer_features.append(x)
        x = self.model.layer3(x)
        layer_features.append(x)
        x = self.model.layer4(x)
        layer_features.append(x)

        x = self.model.global_pool(x)
        x = self.model.fc(x)
        
        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        img = x.clone().detach().requires_grad_(True)
        layer_features = []

        x = self.model.conv1(img)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer1(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer2(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer3(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer4(x)
        x.retain_grad()
        layer_features.append(x)

        x = self.model.global_pool(x)
        x = self.model.fc(x)

        loss = self.loss_fn(x, targets)
        loss.backward()

        return img.grad, [f.grad for f in layer_features]
    
    def forward_layer(self, x, layer):
        if layer == 0:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)
        elif layer == 1:
            x = self.model.layer1(x)
        elif layer == 2:
            x = self.model.layer2(x)
        elif layer == 3:
            x = self.model.layer3(x)
        elif layer == 4:
            x = self.model.layer4(x)
        else:
            raise NotImplementedError
        return x
    
    def embed(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.global_pool(x)
        return x
    
    # @torch.compile
    # def _compiled_forward(self, x):
    #     layer_features = []
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.act1(x)
    #     x = self.model.maxpool(x)
    #     layer_features.append(x)
    #     x = self.model.layer1(x)
    #     layer_features.append(x)
    #     x = self.model.layer2(x)
    #     layer_features.append(x)
    #     x = self.model.layer3(x)
    #     layer_features.append(x)
    #     x = self.model.layer4(x)
    #     layer_features.append(x)
    #     x = self.model.global_pool(x)
    #     x = self.model.fc(x)
    #     return x, layer_features
    
    # @torch.compile
    # def _compiled_embed(self, x):
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.act1(x)
    #     x = self.model.maxpool(x)
    #     x = self.model.layer1(x)
    #     x = self.model.layer2(x)
    #     x = self.model.layer3(x)
    #     x = self.model.layer4(x)
    #     x = self.model.global_pool(x)
    #     return x
    
    def length(self):
        return 5
    

class ResNet_Scratch(_MODEL):
    def __init__(self, model_name, num_classes):
        super(ResNet_Scratch, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def reset_parameters(self):
        '''
        Reset the model into pretrained state.
        '''
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param.data)
            elif param.dim() == 1:
                # nn.init.constant_(param.data, 0.0)
                nn.init.normal_(param.data)
            else:
                nn.init.kaiming_normal_(param.data)

    def hard_reset_parameters(self):
        '''
        Reset the whole parameters of the model.
        '''
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                nn.init.normal_(param.data)
            elif param.dim() == 1:
                # nn.init.constant_(param.data, 0.0)
                nn.init.normal_(param.data)
            else:
                nn.init.kaiming_normal_(param.data)

    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
    
    def forward(self, x):
        layer_features = []
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        layer_features.append(x)
        x = self.model.layer1(x)
        layer_features.append(x)
        x = self.model.layer2(x)
        layer_features.append(x)
        x = self.model.layer3(x)
        layer_features.append(x)
        x = self.model.layer4(x)
        layer_features.append(x)

        x = self.model.global_pool(x)
        x = self.model.fc(x)
        
        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        img = x.clone().detach().requires_grad_(True)
        layer_features = []

        x = self.model.conv1(img)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer1(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer2(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer3(x)
        x.retain_grad()
        layer_features.append(x)
        x = self.model.layer4(x)
        x.retain_grad()
        layer_features.append(x)

        x = self.model.global_pool(x)
        x = self.model.fc(x)

        loss = self.loss_fn(x, targets)
        loss.backward()

        return img.grad, [f.grad for f in layer_features]
    
    def forward_layer(self, x, layer):
        if layer == 0:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.act1(x)
            x = self.model.maxpool(x)
        elif layer == 1:
            x = self.model.layer1(x)
        elif layer == 2:
            x = self.model.layer2(x)
        elif layer == 3:
            x = self.model.layer3(x)
        elif layer == 4:
            x = self.model.layer4(x)
        else:
            raise NotImplementedError
        return x
    
    def embed(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.global_pool(x)
        return x
    
    # @torch.compile
    # def _compiled_forward(self, x):
    #     layer_features = []
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.act1(x)
    #     x = self.model.maxpool(x)
    #     layer_features.append(x)
    #     x = self.model.layer1(x)
    #     layer_features.append(x)
    #     x = self.model.layer2(x)
    #     layer_features.append(x)
    #     x = self.model.layer3(x)
    #     layer_features.append(x)
    #     x = self.model.layer4(x)
    #     layer_features.append(x)
    #     x = self.model.global_pool(x)
    #     x = self.model.fc(x)
    #     return x, layer_features
    
    # @torch.compile
    # def _compiled_embed(self, x):
    #     x = self.model.conv1(x)
    #     x = self.model.bn1(x)
    #     x = self.model.act1(x)
    #     x = self.model.maxpool(x)
    #     x = self.model.layer1(x)
    #     x = self.model.layer2(x)
    #     x = self.model.layer3(x)
    #     x = self.model.layer4(x)
    #     x = self.model.global_pool(x)
    #     return x
    
    def length(self):
        return 5