import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from torch import nn

from .utils import (
    cpu, device, get_activation, get_loss, get_optimizer, shift_and_scale,
    )


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
        inputs=1, # Number of input dimensions
        outputs=1, # Number of output dimensions
        hidden=1, # Number of units in each hidden layer
        layers=1, # Number of hidden layers
        activation='relu', # Activation function
        output_activation=None, # None or activation function for outpt layer
        dropout=0.0, # Dropout probability for hidden units, 0 <= dropout < 1
        norm_inputs=False, # Standardize inputs, bool or array/tensor
        ):
        
        super().__init__()
        
        activation = get_activation(activation, functional=False)
        
        # Input
        modules = [nn.Linear(inputs, hidden), activation()]
        # Zero mean + unit variance per dimension
        if norm_inputs is not False:
            # Place holder for loading state dict
            if norm_inputs is True:
                shift, scale = 0.0, 1.0
            # Input tensor to compute mean and variance from
            else:
                shift, scale = shift_and_scale(norm_inputs)
            modules = [AffineModule(shift, scale)] + modules
        
        # Hidden
        for i in range(layers):
            modules += [nn.Linear(hidden, hidden), activation()]
            if dropout != 0.0:
                modules += [nn.Dropout(dropout)]
        
        # Output
        modules += [nn.Linear(hidden, outputs)]
        if output_activation:
            modules += [get_activation(output_activation, functional=False)()]
            
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, inputs):
        
        return self.sequential(inputs)
    
    
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
    shuffle=True,
    reduce=False,
    stop=False,
    verbose=True,
    save=False,
    seed=None
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
        
        if batch_size is None:
            x_valid = x_valid[None, ...]
            y_valid = y_valid[None, ...]
        else:
            x_valid = x_valid.split(batch_size)
            y_valid = y_valid.split(batch_size)
        
    if not shuffle:
        if batch_size is None:
            x = x[None, ...]
            y = y[None, ...]
        else:
            x = x.split(batch_size)
            y = y.split(batch_size)

    if type(loss) is str:
        loss = get_loss(loss)()
    else:
        assert callable(loss)
            
    optimizer = get_optimizer(optimizer)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    
    best_epoch = 0
    best_loss = np.inf
    losses = {'train': []}
    if validate:
        losses['valid'] = []
    if reduce:
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
            loss_step.backward()
            optimizer.step()
            loss_train += loss_step.item()
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
                    loss_valid += loss(model(xx.to(device)), yy.to(device)).item()
                loss_valid /= n
                losses['valid'].append(loss_valid)
                loss_track = loss_valid
            
        if verbose:
            print(loss_train, end='')
            if validate:
                print(f', {loss_valid}', end='')
            print()
            
        if save:
            np.save(f'{save}.npy', losses, allow_pickle=True)
            
        if loss_track < best_loss:
            if verbose:
                print('Loss improved', end='')
            best_epoch = epoch
            best_loss = loss_track
            best_model = deepcopy(model)
            if save:
                torch.save(best_model, f'{save}.pt')
                
        if reduce:
            if epoch - best_epoch == 0:
                epoch_reduce = epoch
            if epoch - epoch_reduce > reduce:
                epoch_reduce = epoch
                if verbose:
                    print(f'No improvement for {reduce} epochs, reducing lr')
                for group in optimizer.param_groups:
                    group['lr'] /= 2
                    
        if stop:
            if epoch - best_epoch > stop:
                if verbose:
                    print(f'No improvement for {stop} epochs, stopping')
                break
                
    if verbose and save:
        print(save)
                
    return best_model, losses

    