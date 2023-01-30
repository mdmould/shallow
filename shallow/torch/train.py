import torch


def relative_error(pred, true):
    return ((true - pred) / true).abs().mean()


def cross_entropy(pred, true):
    return -(true * pred.log()).sum(dim=-1).mean()


def binary_cross_entropy(pred, true):
    pred = torch.cat((pred, 1 - pred), dim=-1)
    true = torch.cat((true, 1 - true), dim=-1)
    return cross_entropy(pred, true)


def kl_divergence(pred, true):
    return cross_entropy(pred, true) - cross_entropy(true, true)


def binary_kl_divergence(pred, true):
    return (
        binary_cross_entropy(pred, true) -
        binary_cross_entropy(true, true)
        )


## TODO: base training class
class Trainer:
    
    def __init__(
        self,
        
        ):
        
        pass

