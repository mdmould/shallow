import jax
import jax.numpy as jnp

from flowjax.bijections import (
    Invert,
    SoftPlus,
    Tanh,
    Chain,
    Stack,
    )


# modify flowjax.bijections.Affine to accept any non-zero scale
from typing import ClassVar
from jax import Array
from jax.typing import ArrayLike
from flowjax.bijections import Bijection
from flowjax.utils import arraylike_to_array

class Affine(Bijection):
    loc: Array
    scale: Array

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
        ):
        loc, scale = [arraylike_to_array(a, dtype=float) for a in (loc, scale)]
        self.shape = jnp.broadcast_shapes(loc.shape, scale.shape)
        self.cond_shape = None

        self.loc = jnp.broadcast_to(loc, self.shape)
        self.scale = jnp.broadcast_to(scale, self.shape)

    def transform(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        x, _ = self._argcheck_and_cast(x)
        scale = self.scale
        return x * scale + self.loc, jnp.log(jnp.abs(self.scale)).sum()

    def inverse(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        y, _ = self._argcheck_and_cast(y)
        scale = self.scale
        return (y - self.loc) / scale, -jnp.log(jnp.abs(self.scale)).sum()


def get_bounder(bounds):
    # unbounded
    if (bounds is None) or all(bound is None for bound in bounds):
        bijection = Affine(0, 1)
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
        bijection = Chain([SoftPlus(), Affine(loc, scale)])
    # two sided bounds
    ## TODO: try normal CDF instead
    else:
        loc = bounds[0]
        scale = bounds[1] - bounds[0]
        bijection = Chain(
            [Tanh(), Affine(0.5, 0.5), Affine(loc, scale)],
            )
    return bijection


def get_normer(norms):
    mean = jnp.mean(norms, axis=0)
    std = jnp.std(norms, axis=0)
    loc = - mean / std
    scale = 1 / std
    return Affine(loc, scale)


def get_pre(bounds=[None], norms=None):
    bounder = Stack([get_bounder(bound) for bound in bounds])
    if norms is not None:
        debounded_norms = jax.vmap(bounder.inverse)(norms)
        denormer = Invert(get_normer(debounded_norms))
        bounder = Chain([denormer, bounder])
    return bounder
    