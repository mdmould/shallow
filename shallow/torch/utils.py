import torch
from torch import nn
from torch.nn import functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_tensor(data, dtype=torch.get_default_dtype(), device=device):
    
    return torch.as_tensor(data, dtype=dtype, device=device)


def get_activation(activation, functional=False):
    
    if callable(activation):
        return activation
    
    lib = 'F' if functional else 'nn'
    funcs = dir(eval(lib))
    idx = list(map(lambda _: _.lower(), funcs)).index(activation.lower())
    
    return eval(f'{lib}.{funcs[idx]}')


def shift_scale_Normalize(inputs):
    
    mean = torch.mean(inputs, dim=0)
    std = torch.std(inputs, dim=0)
    shift = - mean / std
    scale = 1.0 / std
    
    return shift, scale


def shift_scale_MinMax(inputs):
    
    minimum = torch.min(inputs, dim=0)
    maximum = torch.max(inputs, dim=0)
    shift = - minimum / (maximum - minimum)
    scale = 1.0 / (maximum - minimum)
    
    return shift, scale

