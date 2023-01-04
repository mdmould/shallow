import torch


cpu = torch.device('cpu')
gpu = torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu


def get_tensor(data, dtype=torch.get_default_dtype(), device=device):
    
    return torch.as_tensor(data, dtype=dtype, device=device)


def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters())


def get_func(func, lib):
    
    if callable(func):
        return func
    
    assert type(func) is str
    assert type(lib) is str
    
    funcs = dir(eval(lib))
    idx = list(map(lambda _: _.lower(), funcs)).index(func.lower())
    
    return eval(f'{lib}.{funcs[idx]}')


def get_activation(activation, functional=False):
        
    if functional:
        try:
            return get_func(activation, 'torch')
        except:
            return get_func(activation, 'torch.nn.functional')
    
    return get_func(activation, 'torch.nn')


def get_optimizer(optimizer):
    
    return get_func(optimizer, 'torch.optim')


def shift_and_scale(inputs):
    
    inputs = torch.as_tensor(inputs)
    mean = torch.mean(inputs, dim=0)
    std = torch.std(inputs, dim=0)
    shift = - mean / std
    scale = 1.0 / std
    
    return shift, scale

