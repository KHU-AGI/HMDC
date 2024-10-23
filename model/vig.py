# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import copy
import timm
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model._model_base import _MODEL


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'gnn_patch16_224': _cfg(
        crop_pct=0.9, input_size=(3, 224, 224),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//8),
            act_layer(act),
            nn.Conv2d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//4),
            act_layer(act),
            nn.Conv2d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        
        self.stem = Stem(out_dim=channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 14, 14))

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv2d(channels, 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs) + self.pos_embed
        B, C, H, W = x.shape
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

@register_model
def vig_ti_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 12 # number of basic blocks in the backbone
            self.n_filters = 192 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_s_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 320 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model


@register_model
def vig_b_224_gelu(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=1000, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn # neighbor num (default:9)
            self.conv = 'mr' # graph conv layer {edge, mr}
            self.act = 'gelu' # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch' # batch or instance normalization {batch, instance}
            self.bias = True # bias of conv layer True or False
            self.n_blocks = 16 # number of basic blocks in the backbone
            self.n_filters = 640 # number of channels of deep features
            self.n_classes = num_classes # Dimension of out_channels
            self.dropout = drop_rate # dropout rate
            self.use_dilation = True # use dilated knn or not
            self.epsilon = 0.2 # stochastic epsilon for gcn
            self.use_stochastic = False # stochastic for gcn, True or False
            self.drop_path = drop_path_rate

    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['gnn_patch16_224']
    return model

class ViG(_MODEL):
    def __init__(self, model_name, num_classes):
        super(ViG, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        if model_name == 'vig_ti_224_gelu':
            self.model.load_state_dict(torch.load('./model/pretrained/vig_ti_74.5.pth'), strict=False)
        elif model_name == 'vig_b_224_gelu':
            self.model.load_state_dict(torch.load('./model/pretrained/vig_b_82.6.pth'), strict=False)
        else :
            raise NotImplementedError
        self.model.prediction = nn.Sequential(
            nn.Conv2d(self.model.prediction[0].in_channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(1024, num_classes, 1, bias=True)
        )
        for param in self.model.parameters():
            if 'prediction' not in param.name:
                param.requires_grad = False

    def reset_parameters(self):
        for module in self.model.prediction.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.data.zero_()
                    module.bias.requires_grad = True

    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
    
    def forward(self, x):
        layer_features = []
        x = self.model.stem(x) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
            layer_features.append(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.model.prediction(x).flatten(1)
        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        for param in self.parameters():
            param.requires_grad = True
        img = x.clone().detach().requires_grad_(True)
        layer_features = []
        x = self.model.stem(img) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
            x.retain_grad()
            layer_features.append(x)
        x = F.model.adaptive_avg_pool2d(x, 1)
        x = self.model.prediction(x).flatten(1)
        loss = self.loss_fn(x, targets)
        loss.backward()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
        return img.grad, [f.grad for f in layer_features]
    
    @torch.compile
    def _compiled_forward(self, x):
        layer_features = []
        x = self.model.stem(x) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
            layer_features.append(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.model.prediction(x).flatten(1)
        return x, layer_features
    
    @torch.compile
    def _compiled_embed(self, x):
        x = self.model.stem(x) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x
    
    def forward_layer(self, x, layer):
        return self.model.backbone[layer](x)
    
class ViG_FT(_MODEL):
    def __init__(self, model_name, num_classes):
        super(ViG_FT, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.model = timm.create_model(model_name, pretrained=True)
        if model_name == 'vig_ti_224_gelu':
            self.model.load_state_dict(torch.load('./model/pretrained/vig_ti_74.5.pth'), strict=False)
        elif model_name == 'vig_b_224_gelu':
            self.model.load_state_dict(torch.load('./model/pretrained/vig_b_82.6.pth'), strict=False)
        else :
            raise NotImplementedError
        self.model.prediction = nn.Sequential(
            nn.Conv2d(self.model.prediction[0].in_channels, 1024, 1, bias=True),
            nn.BatchNorm2d(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(1024, num_classes, 1, bias=True)
        )
        self.initial_model = copy.deepcopy(self.model.state_dict())

    def reset_parameters(self):
        self.model.load_state_dict(self.initial_model)
        for module in self.model.prediction.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.data.zero_()
                    module.bias.requires_grad = True

    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)
    
    def forward(self, x):
        layer_features = []
        x = self.model.stem(x) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
            layer_features.append(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.model.prediction(x).flatten(1)
        return x, layer_features
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        for param in self.parameters():
            param.requires_grad = True

        img = x.clone().detach().requires_grad_(True)
        layer_features = []

        x = self.model.stem(img) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
            x.retain_grad()
            layer_features.append(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.model.prediction(x).flatten(1)
        
        loss = self.loss_fn(x, targets)
        loss.backward()

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.head.parameters():
            param.requires_grad = True
            
        return img.grad, [f.grad for f in layer_features]
    
    @torch.compile
    def _compiled_forward(self, x):
        layer_features = []
        x = self.model.stem(x) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
            layer_features.append(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.model.prediction(x).flatten(1)
        return x, layer_features
    
    @torch.compile
    def _compiled_embed(self, x):
        x = self.model.stem(x) + self.model.pos_embed
        B, C, H, W = x.shape
        for i in range(self.model.n_blocks):
            x = self.model.backbone[i](x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x
    
    def forward_layer(self, x, layer):
        return self.model.backbone[layer](x)