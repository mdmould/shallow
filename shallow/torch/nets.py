import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from torch import nn

from ..utils import training_split
from .utils import (
    cpu, device, get_activation, get_loss, get_optimizer, shift_and_scale,
    )
from .train import Trainer


class AffineModule(nn.Module):
    
    def __init__(self, shift, scale):
        
        super().__init__()
        
        self.register_buffer('shift', torch.as_tensor(shift))
        self.register_buffer('scale', torch.as_tensor(scale))
        
    def forward(self, inputs):
        
        return inputs * self.scale + self.shift
    
    
class MultilayerPerceptron(nn.Module):
    
    def __init__(
        self,
        inputs=1,
        outputs=1,
        contexts=None,
        layers=1,
        hidden=1,
        activation='relu',
        dropout=0.0,
        batchnorm=False,
        ):
        
        pass
    
    def forward(self, inputs):
        
        pass
        

## TODO: sub-class MultilayerPerceptron
class ForwardNet(nn.Module):
    
    def __init__(
        self,
        inputs=1, # Number of input dimensions
        outputs=1, # Number of output dimensions
        contexts=None, # Number of conditional dimensions
        layers=1, # Number of hidden layers
        hidden=1, # Number of units in each hidden layer
        activation='relu', # Activation function
        dropout=0.0, # Dropout probability for hidden units, 0 <= dropout < 1
        batchnorm=False,
        output_activation=None, # None or activation function for output layer
        norm_inputs=False, # Standardize inputs, bool or array/tensor
        norm_outputs=False, # Standardize outputs, bool or array/tensor
        ):
        
        super().__init__()
        
        activation = get_activation(activation, functional=False)
        
        # Zero mean + unit variance per input dimension
        self.norm_inputs = False
        if norm_inputs is not False:
            # Place holder for loading state dict
            if norm_inputs is True:
                shift, scale = torch.zeros(inputs), torch.ones(inputs)
            # Input tensor to compute mean and variance from
            else:
                shift, scale = shift_and_scale(norm_inputs)
            self.pre = AffineModule(shift, scale)
            self.norm_inputs = True
        
        # Input
        modules = [nn.Linear(inputs, hidden), activation()]
        
        # Hidden
        for i in range(layers):
            modules += [nn.Linear(hidden, hidden), activation()]
            if dropout != 0.0:
                modules += [nn.Dropout(dropout)]
        
        # Output
        modules += [nn.Linear(hidden, outputs)]
        if output_activation:
            modules += [get_activation(output_activation, functional=False)()]
            
        # Zero mean + unit variance per output dimension
        self.norm_outputs = False
        if norm_outputs is not False:
            # Place holder for loading state dict
            if norm_outputs is True:
                shift, scale = torch.zeros(outputs), torch.ones(outputs)
            # Input tensor to compute mean and variance from
            else:
                shift, scale = shift_and_scale(norm_outputs)
                self.norm_outputs = True
            self.post = AffineModule(shift, scale)
            
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, inputs):
        
        outputs = inputs
        if self.norm_inputs:
            outputs = self.pre(outputs)
        outputs = self.sequential(outputs)
        if self.norm_outputs:
            outputs = self.post(outputs)
            
        return outputs
    
    
## TODO: make residual net
class ResidualBlock(nn.Module):
    
    def __init__(self):
        
        pass
    
    def forward(self):
        
        pass
    
    
class ResidualNet(nn.Module):
    
    def __init__(self):
        
        pass
    
    def forward(self):
        
        pass
    

## TODO: sub-class train.Trainer
def trainer(
    model,
    training_data,
    validation_data=None,
    loss='mse',
    optimizer='adam',
    learning_rate=1e-3,
    weight_decay=0.0,
    epochs=1,
    batch_size=None,
    batch_size_valid='train',
    shuffle=True,
    reduce=None,
    stop=None,
    stop_ifnot_finite=True,
    verbose=True,
    save=None,
    seed=None,
    ):
    
    if seed is not None:
        torch.manual_seed(seed)
        
    model = model.to(device)
    
    x, y = training_data
    x = torch.as_tensor(x, dtype=torch.float32, device=cpu)
    y = torch.as_tensor(y, dtype=torch.float32, device=cpu)
    if x.ndim == 1:
        x = x[..., None]
    if y.ndim == 1:
        y = y[..., None]
    assert x.shape[0] == y.shape[0]
    
    validate = False
    if validation_data is not None:
        validate = True
        
        if type(validation_data) is float:
            train, valid = training_split(x.shape[0], validation_data)
            x_valid = x[valid]
            y_valid = y[valid]
            x = x[train]
            y = y[train]
        else:
            x_valid, y_valid = validation_data
            
        x_valid = torch.as_tensor(x_valid, dtype=torch.float32, device=cpu)
        y_valid = torch.as_tensor(y_valid, dtype=torch.float32, device=cpu)
        if x_valid.ndim == 1:
            x_valid = x_valid[..., None]
        if y_valid.ndim == 1:
            y_valid = y_valid[..., None]
        assert x_valid.shape[0] == y_valid.shape[0]
        assert x_valid.shape[-1] == x.shape[-1]
        assert y_valid.shape[-1] == y.shape[-1]
        
        if (batch_size is None or
            ((batch_size_valid == 'train') and (batch_size is None))
            ):
            x_valid = x_valid[None, ...]
            y_valid = y_valid[None, ...]
        else:
            if batch_size_valid == 'train':
                batch_size_valid = batch_size
            x_valid = x_valid.split(batch_size_valid)
            y_valid = y_valid.split(batch_size_valid)
        
    if not shuffle:
        if batch_size is None:
            x = x[None, ...]
            y = y[None, ...]
        else:
            x = x.split(batch_size)
            y = y.split(batch_size)

    if type(loss) is str:
        loss = get_loss(loss)
        if type(loss) is type:
            loss = loss()
    assert callable(loss)
            
    optimizer = get_optimizer(optimizer)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    
    best_model = deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = np.inf
    losses = {'train': []}
    if validate:
        losses['valid'] = []
    if reduce is not None:
        epoch_reduce = 0
    
    for epoch in range(1, epochs+1):
        print('Epoch', epoch)
        
        # Training
        model = model.train()
        
        if shuffle:
            perm = torch.randperm(x.shape[0])
            x_train = x[perm]
            y_train = y[perm]
            if batch_size is None:
                x_train = x_train[None, ...]
                y_train = y_train[None, ...]
            else:
                x_train = x_train.split(batch_size)
                y_train = y_train.split(batch_size)
        else:
            x_train, y_train = x, y
                
        n = len(x_train)
        loop = zip(x_train, y_train)
        if verbose:
            loop = tqdm(loop, total=n)
        
        loss_train = 0
        for xx, yy in loop:
            optimizer.zero_grad()
            loss_step = loss(model(xx.to(device)), yy.to(device))
            
            if loss_step.isfinite():
                loss_isfinite = True
                loss_step.backward()
                optimizer.step()
                loss_train += loss_step.item()
            else:
                loss_isfinite = False
                if stop_ifnot_finite:
                    break

        loss_train /= n
        losses['train'].append(loss_train)
        loss_track = loss_train
        
        # Validation
        if validate:
            model = model.eval()
            with torch.inference_mode():
                
                n = len(x_valid)
                loop = zip(x_valid, y_valid)
                if verbose:
                    loop = tqdm(loop, total=n)
                
                loss_valid = 0
                for xx, yy in loop:
                    loss_step = loss(model(xx.to(device)), yy.to(device))
                    
                    if loss_step.isfinite():
                        loss_isfinite = True
                        loss_valid += loss_step.item()
                    else:
                        loss_isfinite = False
                        if stop_ifnot_finite:
                            break

                loss_valid /= n
                losses['valid'].append(loss_valid)
                loss_track = loss_valid

        if stop_ifnot_finite and not loss_isfinite:
            print('nan/inf loss, stopping')
            break

        if verbose:
            print(loss_train, end='')
            if validate:
                print(f', {loss_valid}', end='')
            print()
            
        if save is not None:
            np.save(f'{save}.npy', losses, allow_pickle=True)
            
        if loss_track < best_loss:
            if verbose:
                print('Loss improved')
            best_epoch = epoch
            best_loss = loss_track
            best_model = deepcopy(model.state_dict())
            if save is not None:
                torch.save(best_model, f'{save}.pt')
                
        if reduce is not None:
            if epoch - best_epoch == 0:
                epoch_reduce = epoch
            if epoch - epoch_reduce > reduce:
                epoch_reduce = epoch
                if verbose:
                    print(f'No improvement for {reduce} epochs, reducing lr')
                for group in optimizer.param_groups:
                    group['lr'] /= 2
                    
        if stop is not None:
            if epoch - best_epoch > stop:
                if verbose:
                    print(f'No improvement for {stop} epochs, stopping')
                break
                
    if verbose and (save is not None):
        print(save)
        
    model.load_state_dict(best_model)
    model.eval()
                
    return model, losses

    