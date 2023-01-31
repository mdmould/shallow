import numpy as np
from time import time
from itertools import product


def timer(func, *args, **kwargs):
    
    t0 = time()
    result = func(*args, **kwargs)
    print(time() - t0)
    
    return result


def seeder(seed, func, *args, **kwargs):

    state = np.random.get_state()
    np.random.seed(seed)
    result = func(*args, **kwargs)
    np.random.set_state(state)
    
    return result


# def cartesian_product(axes):
#     # Return all combinations of the items in each axis in axes
#     # axes is a list of lists - one for each parameter
#     # Can be mixed types, which are preserved in the output array
#     # First axis changes first in output
#     # Output array has shape (no. combinations, no. parameters = len(axes))

#     return np.array(
#         np.meshgrid(*axes, indexing='ij'), dtype=object,
#         ).reshape(len(axes), -1).T

# For backwards compat
def cartesian_product(axes):
    
    return list(product(*axes))


# def training_split(n, f_train, f_valid=None, seed=None):
    
#     assert 0 < f_train <= 1
#     n_train = int(n * f_train)
    
#     if f_valid is not None:
#         assert 0 <= f_valid <= 1 - f_train
#     else:
#         f_valid = 1 - f_train
#     n_valid = int(n * f_valid)
    
#     idxs = np.arange(n)
#     rng = np.random.default_rng(seed)
#     rng.shuffle(idxs)
    
#     train = np.sort(idxs[:n_train])
#     valid = np.sort(idxs[n_train:n_train+n_valid])
#     test = np.sort(idxs[n_train+n_valid:])
    
#     return train, valid, test

def training_split(n, f_train, seed=None):
    
    assert 0 < f_train <= 1
    n_train = int(n * f_train)
    
    idxs = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)
    
    train = np.sort(idxs[:n_train])
    valid = np.sort(idxs[n_train:])
    
    return train, valid


def get_func(func, lib):
    
    if callable(func):
        return func
    
    assert type(func) is str
    
    funcs = dir(lib)
    idx = list(map(lambda _: _.lower(), funcs)).index(func.lower())
    
    return getattr(lib, funcs[idx])
    
