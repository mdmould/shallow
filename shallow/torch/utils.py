import numpy as np
import torch


cpu = torch.device('cpu')
gpu = torch.device('cuda')
device = gpu if torch.cuda.is_available() else cpu


def training_split(n, f_train, f_valid=None, seed=None):
    
    assert 0 < f_train <= 1
    n_train = int(n * f_train)
    
    if f_valid is not None:
        assert 0 <= f_valid <= 1 - f_train
    else:
        f_valid = 1 - f_train
    n_valid = int(n * f_valid)
    
    idxs = np.arange(n)
    np.random.seed(n)
    np.random.shuffle(idxs)
    np.random.seed()
    
    train = np.sort(idxs[:n_train])
    valid = np.sort(idxs[n_train:n_train+n_valid])
    test = np.sort(idxs[n_train+n_valid:])
    
    return train, valid, test
    
#     assert 0 < f_train <= 1
#     n_train = int(n * f_train)
    
#     n_left = n - n_train
#     if n_left == 0:
#         return np.arange(n), [], []
    
#     if f_valid is not None:
#         assert 0 <= f_valid <= 1 - f_train
#     else:
#         f_valid = 1 - f_train
#     # n_valid = int(n * f_valid)
#     n_valid = int(n_left * f_valid / (1 - f_train))
    
#     idxs = np.arange(n)
#     np.random.seed(seed)
#     np.random.shuffle(idxs)
#     np.random.seed()
    
#     train = np.sort(idxs[:n_train])
#     valid = np.sort(idxs[n_train:n_train+n_valid])
    
#     n_left = n - n_train - n_valid
#     if n_left == 0:
#         return train, valid, []
    
#     n_test = n_left
#     test = np.sort(idxs[-n_test:])
    
    return train, valid, test


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

