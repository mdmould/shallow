import time
import numpy as np


def timer(func, *args, **kwargs):
    
    t0 = time.time()
    result = func(*args, **kwargs)
    print(time.time() - t0)
    
    return result


def _process_data(data):

    # data has shape (n_samples, n_dimensions,)
    data = np.atleast_2d(data)
    assert data.ndim == 2
    assert data.shape[0] > data.shape[1]

    return data

