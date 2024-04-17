import jax
import jax.numpy as jnp
import equinox
from flowjax.wrappers import NonTrainable


class Seed:
    def __init__(self, seed = 0):
        self._seed = int(seed)
        self.key = jax.random.PRNGKey(seed)
    def __call__(self, num = 2):
        self.key, *keys = jax.random.split(self.key, num)
        return jnp.array(keys).squeeze()


def get_partition(model, filter_spec = equinox.is_inexact_array):
    params, static = equinox.partition(
        pytree = model,
        filter_spec = filter_spec,
        is_leaf = lambda leaf: isinstance(leaf, NonTrainable),
    )
    return params, static


def get_params(model, filter_spec = equinox.is_inexact_array):
    params = equinox.filter(
        pytree = model,
        filter_spec = filter_spec,
        is_leaf = lambda leaf: isinstance(leaf, NonTrainable),
    )
    return params


def count_params(model, filter_spec = equinox.is_inexact_array):
    params = get_params(model, filter_spec)
    num_params = params_to_array(params).size
    return num_params


def params_to_array(params):
    array = jax.flatten_util.ravel_pytree(params)[0]
    return array


def get_array_to_params(params):
    array, unflatten = jax.flatten_util.ravel_pytree(params)
    def array_to_params(array):
        return unflatten(array)
    return array_to_params


def save(file, model, filter_spec = equinox.is_inexact_array):
    params = equinox.filter(model, filter_spec)
    array = params_to_array(params)
    return jnp.save(file, array)


def load(file, model, filter_spec = equinox.is_inexact_array):
    params, static = equinox.partition(model, filter_spec)
    array_to_params = get_array_to_params(params)
    array = jnp.load(file)
    params = array_to_params(array)
    model = equinox.combine(params, static)
    return model
