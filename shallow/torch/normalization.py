import torch


def shift_scale_Normalize(inputs):
    
    mean = torch.mean(inputs, dim=0)
    std = torch.std(inputs, dim=0)
    shift = - mean / std
    scale = 1.0 / std
    
    return shift, scale


def shift_scale_MinMax(inputs):
    
    minimum = torch.min(inputs, dim=0)
    maximum = torch.max(inputs, dim=0)
    shift = - minimum / (maximum - minimum)
    scale = 1.0 / (maximum - minimum)
    
    return shift, scale

