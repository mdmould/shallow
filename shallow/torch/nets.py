from torch import nn
from torch.nn import functional as F

from .utils import get_tensor, get_activation, shift_scale_Normalize


class AffineFixed(nn.Module):
    
    def __init__(self, shift, scale):
        
        super().__init__()
        
        self.register_buffer('shift', shift)
        self.register_buffer('scale', scale)
        
    def forward(self, inputs):
        
        return inputs * self.scale + self.shift
    
    
class Normalize(AffineFixed):
    
    def __init__(self, inputs):
        
        super().__init__(*shift_scale_Normalize(inputs))        


class Sequential(nn.Module):
    
    def __init__(
        self,
        inputs=1,
        outputs=1,
        hidden=1,
        layers=1,
        activation='relu',
        output_activation=None,
        norm_inputs=None,
        dropout=0.0,
        ):
        
        super().__init__(self)
        
        activation = get_activation(activation)
        
        # Input
        modules = [nn.Linear(inputs, hidden), activation()]
        if norm_inputs is not None:
            modules = [Normalize(norm_inputs)] + modules
        
        # Hidden
        for i in range(layers):
            modules += [nn.Linear(hidden, hidden), activation()]
            if dropout != 0.0:
                modules += [nn.Dropout(dropout)]
        
        # Output
        modules += [nn.Linear(hidden, outputs)]
        if output_activation is not None:
            modules += [get_activation(output_activation)()]
            
        self.sequential = nn.Sequential(*modules)
        
    def forward(self, inputs):
        
        return self.sequential(inputs)

    