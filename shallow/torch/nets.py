import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from torch import nn

from ..utils import training_split
from .utils import (
    cpu,
    device,
    dtype,
    get_activation,
    get_loss,
    get_optimizer,
    shift_and_scale,
    )
from .training import Trainer ## TODO


class AffineModule(nn.Module):

    def __init__(self, shift, scale):
        
        super().__init__()
        
        self.register_buffer('shift', torch.as_tensor(shift))
        self.register_buffer('scale', torch.as_tensor(scale))
        
    def forward(self, inputs):

        return inputs * self.scale + self.shift


class NormModule(AffineModule):

    def __init__(self, inputs):
        
        if type(inputs) is int:
            shift = torch.zeros(inputs)
            scale = torch.ones(inputs)
            
        else:
            shift, scale = shift_and_scale(inputs)
            
        super().__init__(shift, scale)
        
        
class ContextBlock(nn.Module):
    
    def __init__(self, contexts, hidden):
        
        super().__init__()
        
        self.linear = nn.Linear(contexts, hidden)
        self.glu = nn.GLU(dim=-1)
        
    def forward(self, inputs, context):
        
        return self.glu(torch.cat((inputs, self.linear(context)), dim=-1))
    
    
class BaseBlock(nn.Module):
    
    def __init__(
        self,
        inputs=1,
        outputs=None,
        contexts=None,
        blocks=1,
        activation=None,
        dropout=0,
        batchnorm=False,
        ):

        super().__init__()

        modules = []
        
        if outputs is None:
            outputs = inputs

        for i in range(blocks):
            modules.append(nn.Linear(inputs, outputs))
            if batchnorm:
                modules.append(nn.BatchNorm1d(outputs))
            if activation is not None:
                modules.append(get_activation(activation, functional=False))
            
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
            
        self.sequential = nn.Sequential(*modules)
        
        if contexts is not None:
            self.context_block = ContextBlock(contexts, outputs)
        
    def forward(self, inputs, context=None):
        
        outputs = self.sequential(inputs)
        
        if context is not None:
            outputs = self.context_block(outputs, context)
            
        return outputs
    
    
class HiddenBlock(BaseBlock):
    
    def __init__(
        self,
        hidden=1,
        contexts=None,
        activation='relu',
        dropout=0,
        batchnorm=False,
        ):
        
        super().__init__(
            inputs=hidden,
            outputs=hidden,
            contexts=contexts,
            blocks=1,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
            )


class ResidualBlock(BaseBlock):
    
    def __init__(
        self,
        hidden=1,
        contexts=None,
        activation='relu',
        dropout=0,
        batchnorm=False,
        ):
        
        super().__init__(
            inputs=hidden,
            outputs=hidden,
            contexts=contexts,
            blocks=2,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
            )
        
    def forward(self, inputs, context=None):
        
        return inputs + super().forward(inputs, context)
    
    
class BaseNetwork(nn.Module):
    
    def __init__(
        self,
        inputs=1,
        outputs=1,
        contexts=None,
        residual=False,
        blocks=1,
        hidden=1,
        activation='relu',
        dropout=0.0,
        batchnorm=False,
        output_activation=None,
        norm_inputs=False,
        ):
        
        self.hidden_features = hidden # for nflows compatibility
        
        super().__init__()
        
        modules = []
        
        if contexts is not None:
            inputs += contexts
        modules.append(
            BaseBlock(inputs=inputs, outputs=hidden, activation=activation),
            )
        
        block = ResidualBlock if residual else HiddenBlock
        for _ in range(blocks):
            modules.append(
                block(hidden, contexts, activation, dropout, batchnorm),
                )
            
        modules.append(
            BaseBlock(
                inputs=hidden,
                outputs=outputs,
                activation=output_activation,
                ),
            )
            
        self.sequential = nn.Sequential(*modules)
        
        self.norm_inputs = False
        if norm_inputs is not False:
            self.norm_inputs = True
            self.pre = NormModule(
                inputs if norm_inputs is True else norm_inputs,
                )
            
    def forward(self, inputs, context=None):
        
        if context is not None:
            inputs = torch.cat((inputs, context), dim=-1)
            
        if self.norm_inputs:
            inputs = self.pre(inputs)
            
        return self.sequential(inputs)
    
    
class ForwardNetwork(BaseNetwork):
    
    def __init__(
        self,
        inputs=1,
        outputs=1,
        contexts=None,
        blocks=1,
        hidden=1,
        activation='relu',
        dropout=0.0,
        batchnorm=False,
        output_activation=None,
        norm_inputs=False,
        ):
        
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            contexts=contexts,
            residual=False,
            blocks=blocks,
            hidden=hidden,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
            output_activation=output_activation,
            norm_inputs=norm_inputs,
            )

    
class ResidualNetwork(BaseNetwork):

    def __init__(
        self,
        inputs=1,
        outputs=1,
        contexts=None,
        blocks=1,
        hidden=1,
        activation='relu',
        dropout=0.0,
        batchnorm=False,
        output_activation=None,
        norm_inputs=False,
        ):

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            contexts=contexts,
            residual=True,
            blocks=blocks,
            hidden=hidden,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
            output_activation=output_activation,
            norm_inputs=norm_inputs,
            )


## TODO: sub-class train.Trainer
def trainer(
    model,
    training_data,
    validation_data=None,
    norm_outputs=False,
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
    seed=None, ## TODO: leave this arg or let user do it?
    ):
    
    if seed is not None:
        torch.manual_seed(seed)
        
    model = model.to(device)
    
    if type(loss) is str:
        loss = get_loss(loss)
    assert callable(loss)
    loss_ = loss
            
    optimizer = get_optimizer(optimizer)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    
    x, y = training_data
    x = torch.as_tensor(x, dtype=dtype, device=cpu)
    y = torch.as_tensor(y, dtype=dtype, device=cpu)
    if x.ndim == 1:
        x = x[..., None]
    if y.ndim == 1:
        y = y[..., None]
    assert x.shape[0] == y.shape[0]
    
    if norm_outputs:
        post = NormModule(y)
        y = post(y)
        loss = lambda pred, true: loss_(post(pred), true)
    
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
            x_valid = torch.as_tensor(x_valid, dtype=dtype, device=cpu)
            y_valid = torch.as_tensor(y_valid, dtype=dtype, device=cpu)
            if x_valid.ndim == 1:
                x_valid = x_valid[..., None]
            if y_valid.ndim == 1:
                y_valid = y_valid[..., None]
            assert x_valid.shape[0] == y_valid.shape[0]
            assert x_valid.shape[-1] == x.shape[-1]
            assert y_valid.shape[-1] == y.shape[-1]
        
        if norm_outputs:
            y_valid = post(y_valid)
        
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
    
    best_model = deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = float('inf')
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

