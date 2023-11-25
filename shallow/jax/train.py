from functools import partial
import jax
import jax.numpy as jnp
import jax_tqdm
import equinox
import optax


## TODO: move general parts from .flows and .nets here and have them call this
def _trainer(
    key,
    loss_fn,
    model,
    train,
    valid=None,
    batch_size=None,
    all_batches=True,
    epochs=1,
    patience=None,
    stop_if_inf=True,
    lr=1e-3,
    wd=0,
    print_batch=False,
    print_epoch=True,
    filter_spec=equinox.is_inexact_array,
    ):

    if type(train) is tuple:
        if valid is not None:
            assert type(valid) is tuple
            assert len(valid) == len(train)
            
    

    return None
