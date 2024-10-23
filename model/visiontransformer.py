import copy
import timm
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model._model_base import _MODEL

timm.layers.set_fused_attn(False, False)

class VisionTransformer(_MODEL):
    def __init__(self, model_name, num_classes):
        super(VisionTransformer, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def reset_parameters(self):
        self.model.head.reset_parameters()

    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
    
    def forward(self, x):
        layer_features = []
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        for blk in self.model.blocks:
            x = blk(x)
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0].clone())

        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        for param in self.parameters():
            param.requires_grad = True

        img = x.clone().detach().requires_grad_(True)
        layer_features = []
        
        x = self.model.patch_embed(img)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
            x.retain_grad()
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0])

        loss = self.loss_fn(x, targets)
        loss.backward()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
            
        return img.grad, [f.grad for f in layer_features]
    
    # @torch.compile
    # def _compiled_forward(self, x):
    #     layer_features = []
    #     x = self.model.patch_embed(x)
    #     x = self.model._pos_embed(x)
    #     x = self.model.pos_drop(x)
    #     for blk in self.model.blocks:
    #         x = blk(x)
    #         layer_features.append(x)
    #     x = self.model.norm(x)
    #     x = self.model.head(x[:, 0].clone())
    #     return x, layer_features
    
    # @torch.compile
    # def _compiled_embed(self, x):
    #     x = self.model.patch_embed(x)
    #     x = self.model._pos_embed(x)
    #     x = self.model.pos_drop(x)
    #     for blk in self.model.blocks:
    #         x = blk(x)
    #     x = self.model.norm(x)
    #     return x[:, 0].clone()

    def forward_layer(self, x, layer):
        return self.model.blocks[layer](x)
    
    def length(self) -> int:
        return len(self.model.blocks)

class VisionTransformer_FT(_MODEL):
    def __init__(self, model_name, num_classes):
        super(VisionTransformer_FT, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.initial_model = copy.deepcopy(self.model.state_dict())

    def reset_parameters(self):
        '''
        Reset the model into pretrained state.
        '''
        self.model.load_state_dict(self.initial_model)
        self.model.head.reset_parameters()

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
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0].clone())
        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        img = x.clone().detach().requires_grad_(True)
        layer_features = []
        
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
            x.retain_grad()
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0].clone())

        loss = self.loss_fn(x, targets)
        loss.backward()

        return img.grad, [f.grad for f in layer_features]
    
    def forward_layer(self, x, layer):
        return self.model.blocks[layer](x)
    
    def embed(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        return x[:, 0].clone()

    def length(self) -> int:
        return len(self.model.blocks)
    
    @torch.compile
    def _compiled_forward(self, x):
        layer_features = []
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0].clone())
        return x, layer_features
    
    @torch.compile
    def _compiled_embed(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        return x[:, 0].clone()
    

class VisionTransformer_Scratch(_MODEL):
    def __init__(self, model_name, num_classes):
        super(VisionTransformer_Scratch, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        for param in self.model.parameters():
            param.requires_grad = True

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

    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
    
    def forward(self, x):
        layer_features = []

        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0])

        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        img = x.clone().detach().requires_grad_(True)
        layer_features = []
        
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
            x.retain_grad()
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0].clone())

        loss = self.loss_fn(x, targets)
        loss.backward()

        return img.grad, [f.grad for f in layer_features]
    
    def forward_layer(self, x, layer):
        return self.model.blocks[layer](x)
    
    def embed(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        return x[:, 0].clone()

    @torch.compile
    def _compiled_forward(self, x):
        layer_features = []
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
            layer_features.append(x)
        x = self.model.norm(x)
        x = self.model.head(x[:, 0].clone())
        return x, layer_features
    
    @torch.compile
    def _compiled_embed(self, x):
        x = self.model.patch_embed(x)
        x = self.model._pos_embed(x)
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        x = self.model.norm(x)
        return x[:, 0].clone()
    
    def length(self) -> int:
        return len(self.model.blocks)