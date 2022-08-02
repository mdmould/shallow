import numpy as np

from .utils import _process_data


class Affine:
    
    def __init__(self, scale=1., shift=0.):
        
        self.scale = scale
        self.shift = shift
        
    def forward(self, x):
        
        return x * self.scale + self.shift
    
    def inverse(self, y):
        
        return (y - self.shift) / self.scale
    
    def jac_forward(self, x):
        
        return self.scale
    
    def jac_inverse(self, y):
        
        return 1. / self.scale
    
    def log_jac_forward(self, x):
        
        return np.log(self.scale)
    
    def log_jac_inverse(self, y):
        
        return -np.log(self.scale)


class Whiten(Affine):
    
    def __init__(self, data):
        
        data = _process_data(data)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        scale = 1. / std
        shift = -mean * scale
        super().__init__(scale=scale, shift=shift)


class MinMax(Affine):
    
    def __init__(self, data):
        
        data = _process_data(data)
        minimum = data.min(axis=0)
        maximum = data.max(axis=0)
        scale = 1. / (maximum - minimum)
        shift = -minimum * scale
        super().__init__(shift=shift, scale=scale)

