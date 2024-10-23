from typing import Tuple
from ._model_base import _MODEL
from .mlp import MLP
from .convnet import ConvNet
from .resnet import ResNet, ResNet_FT, ResNet_Scratch
from .visiontransformer import VisionTransformer, VisionTransformer_FT, VisionTransformer_Scratch
from .vig import ViG, ViG_FT
from .lstm import LSTM
from .bert import Bert_FT
from .hubert import Hubert_FT
from .audio_cnn import AudioCNN

def get_model(name, num_classes, channel) -> Tuple[_MODEL, int]:
    '''
    Get the model by name.
    ViT and ResNet based on timm naming.
    '''
    name = name.lower()
    if 'vit' in name:
        if '_ft' in name:
            name = name.replace('_ft', '')
            name = name + '_patch16_224'
            return VisionTransformer_FT(name, num_classes), (224, 224)
        elif '_scratch' in name:
            name = name.replace('_scratch', '')
            name = name + '_patch16_224'
            return VisionTransformer_Scratch(name, num_classes), (224, 224)
        else:
            name = name + '_patch16_224'
            return VisionTransformer(name, num_classes), (224, 224)
        
    elif 'resnet' in name:
        if '_ft' in name:
            name = name.replace('_ft', '')
            return ResNet_FT(name, num_classes), (224, 224)
        elif '_scratch' in name:
            name = name.replace('_scratch', '')
            return ResNet_Scratch(name, num_classes), (224, 224)
        else:
            return ResNet(name, num_classes), (224, 224)
        
    elif 'convnet' in name:
        num_layer = name.replace('convnet', '')
        num_layer = int(num_layer) if num_layer != '' else 3
        return ConvNet(channel, num_classes, 128, num_layer, 'relu', 'instancenorm', 'avgpooling'), (32, 32)
    
    elif 'mlp' in name:
        return MLP(channel, num_classes), (32, 32)
    
    elif 'vig' in name:
        if '_ft' in name:
            name = name.replace('_ft', '')
            name= name + '_224_gelu'
            return ViG_FT(name, num_classes), (224, 224)
        else:
            name= name + '_224_gelu'
            return ViG(name, num_classes), (224, 224)
        
    elif 'audio_cnn' in name:
        return AudioCNN(num_classes), 0
    
    elif 'hubert' in name:
        return Hubert_FT(num_classes, 128), 128

    elif 'lstm' in name:
        return LSTM(num_classes, 6, 128, 64), 64
    
    elif 'bert' in name:
        return Bert_FT(num_classes, 64), 64
    
    else:
        raise NotImplementedError(f"Model {name} not implemented.")
            
    