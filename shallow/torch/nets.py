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
        

class Sequential(nn.Module):
    
    def __init__(
        self,
        inputs=1,
        outputs=1,
        hidden=1,
        layers=1,
        activation='relu',
        output_activation=None,
        dropout=0.0,
        norm_inputs=None,
        ):
        
        super().__init__()
        
        activation = get_activation(activation, functional=False)
        
        # Input
        modules = [nn.Linear(inputs, hidden), activation()]
        if norm_inputs is not None:
            modules = [AffineModule(*shift_and_scale(norm_inputs))] + modules
        
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
    reduce=None,
    stop=None,
    verbose=True,
    save=None,
    seed=None
    ):
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    model = model.to(device)
    
    x_train, y_train = training_data
    x_train = torch.as_tensor(x_train, dtype=torch.float32, device=cpu)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=cpu)
    if x_train.ndim == 1:
        x_train = x_train[..., None]
    if y_train.ndim == 1:
        y_train = y_train[..., None]
    assert x_train.shape[0] == y_train.shape[0]
    
    if not shuffle:
        if batch_size is None:
            x_train = x_train[None, ...]
            y_train = y_train[None, ...]
        else:
            x_train = x_train.split(batch_size)
            y_train = y_train.split(batch_size)
    
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
        assert x_valid.shape[-1] == x_train.shape[-1]
        assert y_valid.shape[-1] == y_train.shape[-1]
        assert x_valid.shape[0] == y_valid.shape[0]
            
        if batch_size is None:
            x_valid = x_valid[None, ...]
            y_valid = y_valid[None, ...]
        else:
            x_valid = x_valid.split(batch_size)
            y_valid = y_valid.split(batch_size)
            
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
            permute = torch.randperm(x_train.shape[0])
            x = x_train[permute]
            y = y_train[permute]
            if batch_size is None:
                x = x[None, :]
                y = y[None, :]
            else:
                x = x.split(batch_size)
                y = y.split(batch_size)
        
        n = len(x)
        loss_train = 0
        loop = zip(x, y)
        if verbose:
            loop = tqdm(loop, total=n)
        for xx, yy in loop:
            xx = xx.to(device)
            yy = yy.to(device)
            optimizer.zero_grad()
            loss_step = loss(model(xx), yy)
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
                loss_valid = 0
                loop = zip(x_valid, y_valid)
                if verbose:
                    loop = tqdm(loop, total=n)
                for x, y in loop:
                    x = x.to(device)
                    y = y.to(device)
                    loss_valid += loss(model(x), y).item()
                loss_valid /= n
            
            losses['valid'].append(loss_valid)
            loss_track = loss_valid
            
        if verbose:
            print(loss_train, end='')
            if validate:
                print(f', {loss_valid}')
            
        if save:
            np.save(f'{save}.npy', losses, allow_pickle=True)
            
        if loss_track < best_loss:
            if verbose:
                print('Loss improved, saving')
            best_epoch = epoch
            if reduce:
                epoch_reduce = epoch
            best_loss = loss_track
            best_model = deepcopy(model)
            if save:
                torch.save(best_model, f'{save}.pt')
                
        if reduce:
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
                
    return best_model, losses

    