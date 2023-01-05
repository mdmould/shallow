import numpy as np


def cartesian_product(axes):

    return np.array(np.meshgrid(*axes, indexing='ij')).reshape(len(axes), -1)


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


def get_func(func, lib):
    
    if callable(func):
        return func
    
    assert type(func) is str
    
    funcs = dir(lib)
    idx = list(map(lambda _: _.lower(), funcs)).index(func.lower())
    
    # return eval(f'{lib}.{funcs[idx]}')
    return getattr(lib, funcs[idx])
    
