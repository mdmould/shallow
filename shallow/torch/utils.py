import torch

from ..utils import get_func
from . import training


cpu = torch.device('cpu')
gpu = torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu

dtype = torch.get_default_dtype()


def get_tensor(data, dtype=dtype, device=device):
    
    return torch.as_tensor(data, dtype=dtype, device=device)

 
def load_model(model, file):

    model.load_state_dict(torch.load(file))
    model.eval()
    
    return model


def count_parameters(model, requires_grad=True):
    
    if requires_grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_children(model):
    
    def recursive(module, children):
        
        modules = list(module.children())
        if len(modules) > 0:
            for child in modules:
                recursive(child, children)
                
        else:
            children.append(module)
            
        return children
    
    return recursive(model, [])


def eval_with_dropout(model):
    
    for module in get_children(model):
        if type(module) is torch.nn.modules.dropout.Dropout:
            module.train()
        else:
            module.eval()
    
    return model


def get_activation(activation, functional=False):

    if functional:
        try:
            func = get_func(activation, torch)
        except:
            func = get_func(activation, torch.nn.functional)
    
    else:
        func = get_func(activation, torch.nn)
        
    if type(func) is type:
        func = func()
        
    return func


def get_loss(loss, functional=False):
    
    if functional:
        try:
            func = get_func(loss + '_loss', torch.nn.functional)
        except:
            func = get_func(loss, training)
            
    else:
        func = get_func(loss + 'Loss', torch.nn)
        
    if type(func) is type:
        func = func()

    return func


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

