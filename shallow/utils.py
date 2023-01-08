import numpy as np


def timer(func, *args, **kwargs):
    
    t0 = time.time()
    result = func(*args, **kwargs)
    print(time.time() - t0)
    
    return result


def seeder(seed, func, *args, **kwargs):
    
    state = np.random.get_state()
    np.random.seed(seed)
    result = func(*args, **kwargs)
    np.random.seed()
    np.random.set_state(state)
    
    return result


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
    seeder(seed, np.random.shuffle, idxs)
    
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
    
    return getattr(lib, funcs[idx])
    
