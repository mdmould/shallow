import os
import datetime
import numpy as np
import tensorflow as tf

tfk = tf.keras

from .utils import _process_data


__all__ = ['Model']


class Model:
    
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
    
    def _make_loss(self):
        
        return 'mse'
    
    def _make_optimizer(self, optimizer, **kwargs):
        
        return tf.keras.optimizers.get(
            {'class_name': optimizer, 'config': kwargs},
            )
    
    def train(
        self, x, y, validation_data=None, epochs=1, batch_size=1,
        learning_rate=.001, optimizer='adam', loss=None, stop=None, save=False,
        callbacks=[], name=None, verbose=1,
        ):
        
        x = _process_data(x)
        if validation_data is not None:
            validation_data = list(validation_data)
            assert len(validation_data) == 2
            validation_data[0] = _process_data(validation_data[0])
            
        if self._x_transform is not None:
            x = self._x_transform(x)
            if validation_data is not None:
                validation_data[0] = self._x_transform(validation_data[0])
        if self._y_transform is not None:
            y = self._y_transform(y)
            if validation_data is not None:
                validation_data[1] = self._y_transform(validation_data[1])
                
        optimizer = self._make_optimizer(
            optimizer, learning_rate=learning_rate,
            )
        if loss is None:
            loss = self._make_loss()
        self._model.compile(optimizer=optimizer, loss=loss)
            
        callbacks += [tfk.callbacks.TerminateOnNaN()]
        monitor = 'loss' if validation_data is None else 'val_loss'
        if stop is not None:
            callbacks += [tfk.callbacks.EarlyStopping(
                monitor=monitor, min_delta=0, patience=int(stop),
                verbose=verbose, mode='min', baseline=None,
                restore_best_weights=True,
                )]
        if save:
            name = './model' if name is None else name
            if os.path.exists(name+'.h5') or os.path.exists(name+'.csv'):
                name += '_'.join(str(datetime.datetime.now()).split('_'))
            callbacks += [
                tfk.callbacks.ModelCheckpoint(
                    filepath=name+'.h5', monitor=monitor, verbose=verbose,
                    save_best_only=True, save_weights_only=True, mode='min',
                    save_freq='epoch',
                    ),
                tfk.callbacks.CSVLogger(
                    filename=name+'.csv', separator=',', append=False,
                    ),
                ]

        return self._model.fit(
            x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=callbacks, validation_data=validation_data, shuffle=True,
            )

