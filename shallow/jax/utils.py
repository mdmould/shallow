import jax
import jax.numpy as jnp


def params_to_array(params):
    arrays, unflatten = jax.tree_util.tree_flatten(params)
    array = jnp.concatenate([a.flatten() for a in arrays])
    return array


## TODO: convert maps to jax transformations
def get_array_to_params(params):
    arrays, unflatten = jax.tree_util.tree_flatten(params)
    shapes = list(map(jnp.shape, arrays))
    lens = list(map(lambda shape: jnp.prod(jnp.array(shape)), shapes))
    idxs = list(jnp.cumsum(jnp.array(lens[:-1])))
    def array_to_params(array):
        flat_arrays = jnp.split(array, idxs)
        arrays = list(map(lambda z: jnp.reshape(*z), zip(flat_arrays, shapes)))
        params = jax.tree_util.tree_unflatten(unflatten, arrays)
        return params
    return array_to_params


def count_params(params):
    return params_to_array(params).size
