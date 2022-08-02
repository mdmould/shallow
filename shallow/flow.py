import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import re

tfk = tf.keras
tfb = tfp.bijectors
tfd = tfp.distributions

from .model import Model


class AutoregressiveFlow(Model):
    
    def __init__(
        self, dims=1, n_flows=1, n_layers=1, n_units=1, activation='relu',
        conditions=None, bounds=None, base=None, model_file=None,
        transform=None, conditions_transform=None, **made_kwargs,
        ):
        
        if bounds is not None:
            assert len(bounds) == dims
        
        self.dims = int(dims)
        self.n_flows = int(n_flows)
        self.n_layers = int(n_layers)
        self.n_units = int(n_units)
        self.activation = activation
        
        self.transform = transform
        
        if conditions is None:
            self.conditional = False
            self.conditions = None
        else:
            self.conditional = True
            self.conditions = int(conditions)
            if conditions_transform is None:
                self._conditions_transform = None
            else:
                self._conditions_transform = lambda xc: [
                    xc[0], conditions_transform(xc[1]),
                    ]

        self.bounds = bounds
        
        if base is None:
            self.base = tfd.MultivariateNormalDiag(loc=[0.]*self.dims)
        else:
            self.base = base
            
        self._name = 'maf'
            
        self.flow = tfd.TransformedDistribution(
            distribution=self.base,
            bijector=self._make_bijector(made_kwargs),
            )
        
        super().__init__(model_file=model_file)

    def _sample(self, sample_shape, condition=None):
        
        return self.flow.sample(
            sample_shape, bijector_kwargs=self._conditional_kwargs(condition),
            ).numpy()
    
    def _log_prob(self, value, condition=None):
        
        return self.flow.log_prob(
            value, bijector_kwargs=self._conditional_kwargs(condition),
            ).numpy()
    
    def predict(self):
        
        pass
    
    def forward(self, x, condition=None):
        
        return self.flow.bijector.forward(
            x, **self._conditional_kwargs(condition),
            ).numpy()
    
    def inverse(self, y, condition=None):
        
        return self.flow.bijector.inverse(
            y, **self._conditional_kwargs(condition),
            ).numpy()
        
    def _conditional_kwargs(self, condition=None):
        
        if not self.conditional:
            return {}
        
        if self._conditions_transform is not None:
            condition = self._conditions_transform(condition)
        
        return _recurrsive_kwargs(
            self.flow.bijector,
            {f'{self._name}.': {'conditional_input': condition}},
            )
        
    def _make_model(self):
        
        x = tfk.Input(shape=[self.dims], dtype=tf.float32)
        
        if self.conditional:
            c = tfk.Input(shape=[self.conditions], dtype=tf.float32)
            log_prob = self.log_prob(x, condition=c)
            model = tfk.Model([x, c], log_prob)
            
        else:
            log_prob = self.log_prob(x)
            model = tfk.Model(x, log_prob)
            
        model.compile(
            optimizer=tf.optimizers.Adam(),
            loss=lambda _, log_prob: -log_prob,
            )
        
        return model
        
    def _make_bijector(self):
        
        bijector = self._masked_or_inverse(self._make_stack_bijector())
        
        if self.bounds is None:
            return bijector
        
        return tfb.Chain([self._make_output_bijector(), bijector])
    
    def _masked_or_inverse(self):
        
        pass
        
    def _make_stack_bijector(self, made_kwargs):
        
        bijectors = []
        for i in range(self.n_flows):
            
            made = tfb.AutoregressiveNetwork(
                params=2,
                event_shape=[self.dims],
                conditional=self.conditional,
                conditional_event_shape=[self.conditions],
                hidden_units=[self.n_units]*self.n_layers,
                activation=self.activation,
                **made_kwargs,
                )
            maf = tfb.MaskedAutoregressiveFlow(made, name=self._name+str(i))
            bijectors.append(maf)
            
            if i < n_flows-1:
                permute = tfb.Permute(list(reversed(range(self.dims))))
                bijectors.append(permute)
                
        return tfb.Chain(bijectors)
        
    def _make_output_bijector(self):
        
        bijectors = []
        for i in range(self.dims):
            
            if self.bounds[i] is None:
                bijectors.append(tfb.Identity())
                
            else:
                lo, hi = self.bounds[i]
                shift = lo + 1.
                scale = (hi - lo) / 2.
                sigmoid = tfb.Chain(
                    [tfb.Scale(scale), tfb.Shift(shift), tfb.Tanh()],
                    )
                bijectors.append(sigmoid)
                
        return tfb.Blockwise(bijectors, block_sizes=[1]*self.dims)


class MAF(AutoregressiveFlow):
    
    def _masked_or_inverse(self, bijector):
        
        return bijector


class IAF(AutoregressiveFlow):
    
    def _masked_or_inverse(self, bijector):
        
        return tfb.Invert(bijector)


## TODO:
## check log loss
## generate batches from student for training
class TwoWayFlow:
    
    def __init__(self, teacher_kwargs, student_kwargs=None):
        
        self.maf = MAF(**teacher_kwargs)
        if student_kwargs is None:
            self.iaf = IAF(**teacher_kwargs)
        else:
            self.iaf = IAF(**student_kwargs)
            
        self.loss = lambda lp_teacher, lp_student: lp_student - lp_teacher
        
    def train(self):
        
        pass

    
# https://github.com/tensorflow/probability/issues/1410
# https://github.com/tensorflow/probability/issues/1006#issuecomment-663141106
def _recurrsive_kwargs(bijector, name_to_kwargs):

    if hasattr(bijector, 'bijectors'):
        return {
            b.name: make_bijector_kwargs(b, name_to_kwargs)
            for b in bijector.bijectors
            }
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if re.match(name_regex, bijector.name):
                return kwargs
    return {}

