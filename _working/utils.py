import time
import numpy as np


def timer(func, *args, **kwargs):
    
    t0 = time.time()
    result = func(*args, **kwargs)
    print(time.time() - t0)
    
    return result


def seeder(seed, func, *args, **kwargs):
    
    np.random.seed(seed)
    result = func(*args, **kwargs)
    np.random.seed()
    
    return result


def _process_data(data, transpose=False):

    # data has shape (n_samples, n_dimensions,)
    data = np.atleast_2d(data)
    assert data.ndim == 2
    
    ordered = data.shape[0] > data.shape[1]
    if transpose:
        if not ordered:
            return data.T
    assert ordered
    
    return data

