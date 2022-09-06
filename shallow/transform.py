import numpy as np


_all__ = ['Affine', 'Whiten', 'MinMax']


class Affine:
    
    def __init__(self, scale=1., shift=0.):
        
        self.scale = scale
        self.shift = shfit
        
    def forward(self, x):
        
        return x * self.scale + self.shift
    
    def inverse(self, y):
        
        return (y - self.shift) / self.scale
    
    def jac_forward(self, x):
        
        return np.product(self.scale)
    
    def jac_inverse(self, y):
        
        return np.product(1. / self.scale)
    
    def log_jac_forward(self, x):
        
        return np.sum(np.log(self.scale))
    
    def log_jac_inverse(self, y):
        
        return np.sum(-np.log(self.scale))


class Whiten(Affine):

    def __init__(self, data):

        # data has shape (n_samples, n_dimensions,)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        scale = 1. / std
        shift = -mean / std
        super().__init__(scale=scale, shift=shift)


class MinMax(Affine):

    def __init__(self, data):

        # data has shape (n_samples, n_dimensions)
        minimum = np.min(data, axis=0)
        maximum = np.max(data, axis=0)
        scale = 1. / (maximum - minimum)
        shift = -minimum / (maximum -minimum)
        super().__init__(scale=scale, shift=shift)

