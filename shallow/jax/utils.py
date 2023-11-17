import jax
import jax.numpy as jnp
import equinox


def params_to_array(params):
    arrays, unflatten = jax.tree_util.tree_flatten(params)
    array = jnp.concatenate([a.flatten() for a in arrays])
    return array


def get_array_to_params(params):
    arrays, unflatten = jax.tree_util.tree_flatten(params)
    # shapes = [a.shape for a in arrays]
    shapes = list(map(jnp.shape, arrays))
    # lens = [np.prod(shape) for shape in shapes]
    lens = list(map(np.prod, shapes))
    idxs = np.cumsum(lens)[:-1]
    def array_to_params(array):
        flat_arrays = jnp.split(array, idxs)
        arrays = [a.reshape(shape) for a, shape in zip(flat_arrays, shapes)]
        params = jax.tree_util.tree_unflatten(unflatten, arrays)
        return params
    return array_to_params


def count_params(params):
    return params_to_array(params).size
