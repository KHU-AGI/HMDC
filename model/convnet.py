import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from model._model_base import _MODEL

class ConvNet(_MODEL):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()
        self.channel = channel
        self.num_classes = num_classes
        self.net_width = net_width
        self.net_depth = net_depth
        self.net_act = net_act
        self.net_norm = net_norm
        self.net_pooling = net_pooling
        self.im_size = im_size

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)
        
    def forward(self, x):
        # fourth_middle = self.features[:-9](x)
        # third_middle = self.features[-9: -5](fourth_middle)
        # second_middle = self.features[-5: -1](third_middle)
        # out = self.features[-1:](second_middle)
        for i in range(len(self.features)):
            x = self.get_submodule(f"layer_{i}")(x)
            if len(self.features) - i == 9:
                fourth_middle = x
            elif len(self.features) - i == 5:
                third_middle = x
            elif len(self.features) - i == 1:
                second_middle = x
        out_middle = x.clone()
        out = x
        out_final = self.classifier(out.view(out.size(0), -1))
        return out_final, [fourth_middle, third_middle, second_middle, out]
    
    def gradient(self, x: Tensor, targets: Tensor) -> Tensor:
        # for param in self.parameters():
        #     param.requires_grad = True
        x = x.clone().detach().requires_grad_(True)
        fourth_middle = self.features[:-9](x)
        fourth_middle.retain_grad()
        third_middle = self.features[-9: -5](fourth_middle)
        third_middle.retain_grad()
        second_middle = self.features[-5: -1](third_middle)
        second_middle.retain_grad()
        out = self.features[-1:](second_middle)
        # second_middle = self.features[:-3](x)
        out_middle = out.clone()
        out.retain_grad()
        out_final = self.classifier(out.view(out.size(0), -1))

        loss = self.loss_fn(out_final, targets)
        loss.backward()

        return x.grad, [fourth_middle.grad, third_middle.grad, second_middle.grad, out.grad]
    
    def loss_fn(self, outputs, targets):
        return F.cross_entropy(outputs, targets)

    def reset_parameters(self):
        device = next(self.parameters()).device
        features, _ = self._make_layers(self.channel, self.net_width, self.net_depth, self.net_norm, self.net_act, self.net_pooling, self.im_size)
        with torch.no_grad():
            for n in range(len(self.features)):
                for param1, param2 in zip(self.get_submodule(f"layer_{n}").parameters(), features[n].to(device).parameters()):
                    param1.data = param2.data
            # for f1, f2 in zip(self.features.parameters(), features.parameters()):
            #     f1.data = f2.to(device).data
        del features
        
    def forward_layer(self, x, layer):
        if layer == 0:
            x = self.features[:-9](x)
        elif layer == 1:
            x = self.features[-9: -5](x)
        elif layer == 2:
            x = self.features[-5: -1](x)
        elif layer == 3:
            x = self.features[-1:](x)
        else:
            raise NotImplementedError
        return x

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2
        for n, l in enumerate(layers):
            self.register_module(str(f"layer_{n}"), l)
        return nn.Sequential(*layers), shape_feat
    
    def embed(self, x):
        # fourth_middle = self.features[:-9](x)
        # third_middle = self.features[-9: -5](fourth_middle)
        # second_middle = self.features[-5: -1](third_middle)
        # out = self.features[-1:](second_middle)
        for i in range(len(self.features)):
            x = self.get_submodule(f"layer_{i}")(x)
            if len(self.features) - i == 9:
                fourth_middle = x
            elif len(self.features) - i == 5:
                third_middle = x
            elif len(self.features) - i == 1:
                second_middle = x
        out_middle = x.clone()
        out = x
        return out.view(out.size(0), -1)
    
    @torch.compile
    def _compiled_forward(self, x):
        fourth_middle = self.features[:-9](x)
        third_middle = self.features[-9: -5](fourth_middle)
        second_middle = self.features[-5: -1](third_middle)
        out = self.features[-1:](second_middle)
        out_final = self.classifier(out.view(out.size(0), -1))
        return out_final, [fourth_middle, third_middle, second_middle, out]
    
    @torch.compile
    def _compiled_embed(self, x):
        fourth_middle = self.features[:-9](x)
        third_middle = self.features[-9: -5](fourth_middle)
        second_middle = self.features[-5: -1](third_middle)
        out = self.features[-1:](second_middle)
        return out.view(out.size(0), -1)

    def length(self):
        return 4
    