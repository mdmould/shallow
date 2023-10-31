import numpy as np

import jax
import jax.numpy as jnp
import equinox
import optimistix
import optax
import jax_tqdm

from flowjax.distributions import (
    StandardNormal,
    Normal,
    Transformed,
    )
from flowjax.bijections import (
    Invert,
    Affine,
    SoftPlus,
    Tanh,
    Chain,
    Concatenate,
    Stack,
    )
from flowjax.flows import (
    BlockNeuralAutoregressiveFlow,
    CouplingFlow,
    MaskedAutoregressiveFlow,
    )

from typing import ClassVar
from jax import Array
from jax.typing import ArrayLike
from flowjax.bijections.bijection import Bijection
from flowjax.utils import arraylike_to_array


# flow wrapper
# - bounding transform
# - params pytree to array
# - numerical solver for BNAF inverse

# training loop
# - scan batches and epochs


# modify flowjax.bijections.Affine to accept any non-zero scale
class FixedAffine(Bijection):
    """Elementwise affine transformation ``y = a*x + b``."""

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array = equinox.field(static=True)
    _scale: Array = equinox.field(static=True)

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
    ):
        """Initilaizes an affine transformation.

        ``loc`` and ``scale`` should broadcast to the desired shape of the bijection.

        Args:
            loc (ArrayLike): Location parameter. Defaults to 0.
            scale (ArrayLike): Scale parameter. Defaults to 1.
        """
        loc, scale = (arraylike_to_array(a, dtype=float) for a in (loc, scale))
        self.shape = jnp.broadcast_shapes(loc.shape, scale.shape)
        self.loc = jnp.broadcast_to(loc, self.shape)
        self._scale = jnp.broadcast_to(scale, self.shape)
        assert jnp.all(self._scale != 0)

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return x * self._scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        scale = self._scale
        return x * scale + self.loc, jnp.log(jnp.abs(scale)).sum()

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return (y - self.loc) / self._scale

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        scale = self._scale
        return (y - self.loc) / scale, -jnp.log(jnp.abs(scale)).sum()


# FixedAffine = Affine


def get_bounder(bounds):
    # unbounded
    if (bounds is None) or all(bound is None for bound in bounds):
        bijection = FixedAffine(0, 1)
    # one sided bounds
    elif any(bound is None for bound in bounds):
        # right side bounded
        if bounds[0] is None:
            loc = bounds[1]
            scale = -1
        # left side bounded
        elif bounds[1] is None:
            loc = bounds[0]
            scale = 1
        bijection = Chain([SoftPlus(), FixedAffine(loc, scale)])
    # two sided bounds
    else:
        loc = bounds[0]
        scale = bounds[1] - bounds[0]
        bijection = Chain(
            [Tanh(), FixedAffine(0.5, 0.5), FixedAffine(loc, scale)],
            )
    return bijection


def get_normer(norms):
    mean = jnp.mean(norms, axis=0)
    std = jnp.std(norms, axis=0)
    loc = - mean / std
    scale = 1 / std
    return FixedAffine(loc, scale)


class BoundedFlow(Transformed):
    def __init__(self, flow, bounds=[None], norms=None):
        bounder = Stack([get_bounder(bound) for bound in bounds])
        if norms is not None:
            debounded_norms = jax.vmap(bounder.inverse)(norms)
            denormer = Invert(get_normer(debounded_norms))
            bounder = Chain([denormer, bounder])
        super().__init__(flow.base_dist, Chain([flow.bijection, bounder]))


def flow_to_array(flow):
    params, static = equinox.partition(flow, equinox.is_inexact_array)
    arrays, unflatten = jax.tree_util.tree_flatten(params)
    array = jnp.concatenate([a.flatten() for a in arrays])
    return array


def get_array_to_flow(flow):
    params, static = equinox.partition(flow, equinox.is_inexact_array)
    arrays, unflatten = jax.tree_util.tree_flatten(params)
    shapes = [a.shape for a in arrays]
    lens = [np.prod(shape) for shape in shapes]
    idxs = np.cumsum(lens)[:-1]
    def array_to_flow(array):
        flat_arrays = jnp.split(array, idxs)
        arrays = [a.reshape(shape) for a, shape in zip(flat_arrays, shapes)]
        params = jax.tree_util.tree_unflatten(unflatten, arrays)
        flow = equinox.combine(static, params)
        return flow
    return array_to_flow


## TODO: make this work
def numerical_inverse(flow, z, solver=None, bounds=None):
    fn = lambda x, z: flow.bijection.inverse(x) - z
    if solver is None:
        solver = optimistix.Newton(rtol=1e-5, atol=1e-5)
    if flow.__class__.__name__ == 'BoundedFlow':
        initial = lambda z: flow.bijection[-1].transform(z)
    else:
        initial = lambda z: z
    if bounds is not None:
        lower = jnp.array([bound[0] for bound in bounds])
        upper = jnp.array([bound[1] for bound in bounds])
        options = dict(lower=lower, upper=upper)
    else:
        options = {}
    def single(z):
        x0 = initial(z)
        result = optimistix.root_find(
            fn, solver, x0, z, options=options,
            )
        return result.value
    x = jax.vmap(single)(z)
    return x


def numerical_sampling(flow, key, shape, solver=None):
    z = flow.base_dist.sample(key, shape)
    x = numerical_inverse(flow, z, solver)
    return x


def trainer(
    key,
    flow,
    x,
    valid=None,
    loss_fn=None,
    max_epochs=100,
    patience=10,
    lr=1e-3,
    batch_size=100,
    print_batches=False,
    print_epochs=True,
    ):
    
    pass








    

