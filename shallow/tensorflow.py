import os
import re
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfb = tfp.bijectors
tfd = tfp.distributions

from .utils import _process_data


def train(
    model, x, y, validation=None, epochs=1, batch_size=1, learning_rate=1e-3,
    optimizer='adam', loss=None, stop=None, name=None, callbacks=[], name=None,
    verbose=1,
    ):
    
    x = _process_data(x)
    y = _process_data(y)
    if validation_data is not None:
        assert len(validation_data) == 2
        validation_data = list(
            _process_data(validation_data[0]),
            _process_data(validation_data[1]),
            )
        
    if model._x_transform is not None:
        x = model._x_transform(x)
        if validation_data is not None:
            validation_data[0] = model._x_transform(validation_data[0])
    if model._y_transform is not None:
        y = model._y_transform(y)
        if validation_data is not None:
            validation_data[1] = model._y_transform(validation_data[1])
            
    if type(optimizer) is str:
        optimizer = dict(
            class_name=optimizer, config=dict(learning_rate=learning_rate),
            )
    optimizer = tfk.optimizers.get(optimizer)
    loss = tfk.losses.get(loss) if model._loss is None else model._loss
    
    callbacks += [tfk.callbacks.TerminateOnNaN()]
    monitor = 'loss' if validation_data is None else 'val_loss'
    if stop is not None:
        callbacks += [
            tfk.callbacks.EarlyStopping(
                monitor=monitor, min_delta=0, patience=int(stop),
                verbose=verbose, mode='min', baseline=None,
                restore_best_weights=True,
                ),
            ]
    if name is not None:
        if os.path.exists(name+'.h5') or os.path.exists(name+'.csv'):
            name += str(datetime.now()).replace(' ', '_')
        callbacks += [
            tfk.callbacks.ModelCheckpoint(
                filepath=name+'.h5', monitor=monitor, verbose=verbose,
                save_best_only=True, save_weights_only=True, mode='min',
                save_freq='epoch',
                ),
            tfk.callbacks.CSVLogger(
                filename=name+'.csv', seperator=',', append=False,
                ),
            ]
    
    model.compile(optimizer=optimizer, loss=loss)
    
    return model.fit(
        x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
        callbacks=callbacks, validation_data=validation_data, shuffle=True,
        )


class Model(tfk.Model):
    
    def __init__(self, model_file=None, x_transform=None, y_transform=None):
        
        self._model = self._make_model()
        if model_file is not None:
            self._model.load_weights(model_file)
            
        if x_transform is None:
            x_transform = lambda x: x
        if y_transform is None:
            y_transform = lambda y: y
        self._x_transform = x_transform
        self._y_transform = y_transform
        
    def __call__(self, x):
        
        return self.predict(x)
    
    def predict(self, x):
        
        return self._y_transform(
            self._model.predict_on_batch(self._x_transform(x)),
            )
        
    def _make_model(self):
        
        pass


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
            b.name: _recurrsive_kwargs(b, name_to_kwargs)
            for b in bijector.bijectors
            }
    else:
        for name_regex, kwargs in name_to_kwargs.items():
            if re.match(name_regex, bijector.name):
                return kwargs
    return {}    

    