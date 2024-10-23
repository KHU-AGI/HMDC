import torch.nn as nn
from torch import Tensor
from typing import Tuple, Iterable

class _MODEL(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        raise NotImplementedError
    
    def gradient(self, x : Tensor, targets : Tensor) -> Tensor:
        raise NotImplementedError
    
    def loss_fn(self, outputs : Tensor, targets : Tensor) -> Tensor:
        raise NotImplementedError
    
    def forward(self, x : Tensor) -> Tuple[Tensor, Iterable[Tensor]]:
        raise NotImplementedError
    
    def forward_layer(self, x : Tensor, layer : int) -> Tensor:
        raise NotImplementedError
    
    def length(self) -> int:
        raise NotImplementedError
    