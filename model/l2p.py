import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2P(nn.Module):
    def __init__(self, model_name, num_classes, pool_size, selection_size, prompt_length, pretrained=True):
        super(L2P, self).__init__()
        
        self.pool_size = pool_size
        self.selection_size = selection_size
        self.prompt_length = prompt_length

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.head.weight.requires_grad = True
        self.model.head.bias.requires_grad = True

        self.prompts = nn.Parameter(torch.randn(self.pool_size, self.prompt_length, self.model.embed_dim), requires_grad=True)
        self.keys = nn.Parameter(torch.randn(self.pool_size, self.model.embed_dim), requires_grad=True)
    
    def forward(self, x):
        
        layer_features = []
        x = self.model.patch_embed(x)
        cls = self.model.cls_token.expand(x.shape[0], -1, -1) + self.model.pos_embed[:,:1]
        x = x + self.model.pos_embed[:,1:]

        token_append = torch.cat((cls, x), dim=1)
        token_append = token_append + self.model.pos_embed
        token_append = self.model.pos_drop(token_append)

        feature = self.model.blocks(token_append)[:,:1]
        similarity = F.cosine_similarity(feature, self.keys, dim=-1)
        topk_similarity, topk_indices = torch.topk(similarity, self.selection_size, dim=-1)
        topk_prompts = self.prompts[topk_indices].flatten(1, 2)
        self.similarity_loss = (1 - topk_similarity).mean()
        topk_prompts = topk_prompts + self.model.pos_embed[:,:1].expand(-1, self.selection_size * self.prompt_length, -1)
        x = torch.cat((cls, topk_prompts, x), dim=1)
        topk_prompts = self.model.pos_drop(topk_prompts)
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            layer_features.append(x.clone())
        x = self.model.norm(x)
        x = self.model.head(x[:, 0])

        layer_features = torch.stack(layer_features, dim=1)
        return x, layer_features

    def loss_fn(self, outputs, targets):
        return nn.CrossEntropyLoss()(outputs, targets) + self.similarity_loss

    def reset_parameters(self):
        self.model.head.reset_parameters()
        nn.init.normal_(self.prompts, std=0.02)
        nn.init.normal_(self.keys, std=0.02)
    
    def length(self):
        return 12