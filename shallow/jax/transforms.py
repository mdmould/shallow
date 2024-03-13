import jax
import jax.numpy as jnp

from flowjax.bijections import (
    Chain,
    Identity,
    Invert,
    Stack,
    # Affine,
    Exp,
    SoftPlus,
    Tanh,
    )


# modify flowjax.bijections.Affine to accept any non-zero scale
from typing import ClassVar
from jax import Array
from jax.typing import ArrayLike
from flowjax.bijections import AbstractBijection
from flowjax.utils import arraylike_to_array

class Affine(AbstractBijection):
    """Elementwise affine transformation ``y = a*x + b``.

    ``loc`` and ``scale`` should broadcast to the desired shape of the bijection.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    scale: Array

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
    ):
        self.loc, self.scale = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (loc, scale)),
        )
        self.shape = scale.shape

    def transform(self, x, condition=None):
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        return x * self.scale + self.loc, jnp.log(jnp.abs(self.scale)).sum()

    def inverse(self, y, condition=None):
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        return (y - self.loc) / self.scale, -jnp.log(jnp.abs(self.scale)).sum()


def get_bounder(bounds):
    # unbounded
    if (bounds is None) or all(bound is None for bound in bounds):
        bijection = Identity()
        
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
        constraint = Exp()
        reflect = Affine(loc, scale)
        bijection = Chain([constraint, reflect])
        
    # two sided bounds
    ## TODO: try normal CDF instead
    else:
        loc = bounds[0]
        scale = bounds[1] - bounds[0]
        constraint = Tanh()
        rescale = Affine(0.5 * scale + loc, 0.5 * scale)
        bijection = Chain([constraint, rescale])

    return bijection


def get_normer(norms):
    mean = jnp.mean(norms, axis=0)
    std = jnp.std(norms, axis=0)
    loc = - mean / std
    scale = 1 / std
    return Affine(loc, scale)


def get_pre(bounds = None, norms = None):
    if bounds is None and norms is None:
        return Identity()
    elif bounds is not None and norms is None:
        return Stack(list(map(get_bounder, bounds)))
    elif bounds is None and norms is not None:
        return Invert(get_normer(norms))
    else:
        bounder = Stack(list(map(get_bounder, bounds)))
        debounded_norms = jax.vmap(bounder.inverse)(norms)
        denormer = Invert(get_normer(debounded_norms))
        return Chain([denormer, bounder])


def get_pre_stack1d(bounds = None, norms = None):
    if bounds is None and norms is None:
        return Identity()
    elif bounds is not None and norms is None:
        return Stack(list(map(get_bounder, bounds)))
    elif bounds is None and norms is not None:
        return Stack(list(map(lambda x: Invert(get_normer(x)), norms.T)))
    else:
        pres = []
        for bound, norm in zip(bounds, norms.T):
            bounder = get_bounder(bound)
            debounded_norm = jax.vmap(bounder.inverse)(norm)
            denormer = Invert(get_normer(debounded_norm))
            pres.append(Chain([denormer, bounder]))
        return Stack(pres)        
    
