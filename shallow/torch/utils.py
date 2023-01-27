import torch

from ..utils import get_func
from . import train


cpu = torch.device('cpu')
gpu = torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu

dtype = torch.get_default_dtype()


def get_tensor(data, dtype=dtype, device=device):
    
    return torch.as_tensor(data, dtype=dtype, device=device)


def count_parameters(model, requires_grad=True):
    
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_activation(activation, functional=False):
        
    if functional:
        try:
            return get_func(activation, torch)
        except:
            return get_func(activation, torch.nn.functional)
    
    return get_func(activation, torch.nn)


def get_loss(loss):
    
    try:
        return get_func(loss + 'Loss', torch.nn)
    except:
        try:
            return get_func(loss + '_loss', torch.nn.functional)
        except:
            return get_func(loss, train)


def get_optimizer(optimizer):
    
    return get_func(optimizer, torch.optim)


def shift_and_scale(inputs):
    
    inputs = torch.as_tensor(inputs)
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    mean = torch.mean(inputs, dim=0)
    std = torch.std(inputs, dim=0)
    shift = -mean / std
    scale = 1.0 / std
    
    return shift, scale

