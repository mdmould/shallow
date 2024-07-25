import jax
import jax.numpy as jnp
import equinox
from flowjax.bijections import (
    AbstractBijection,
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Invert,
    Stack,
    Tanh,
)
from collections.abc import Callable


def Affine(loc = 0, scale = 1):
    affine = AffinePositiveScale(loc, scale)
    loc, scale = jnp.broadcast_arrays(
        affine.loc, jnp.asarray(scale, dtype = float),
    )
    affine = equinox.tree_at(lambda tree: tree.scale, affine, scale)
    return affine


def Logistic(shape = ()):
    loc = jnp.ones(shape) * 0.5
    scale = jnp.ones(shape) * 0.5
    return Chain([Tanh(shape), Affine(loc, scale)])


def UnivariateBounder(bounds = None):
    # no bounds
    if (bounds is None) or all(bound is None for bound in bounds):
        return Identity()

    # bounded on one side
    elif any(bound is None for bound in bounds):
        # bounded on right-hand side
        if bounds[0] is None:
            loc = bounds[1]
            scale = -1
        # bounded on left-hand side
        elif bounds[1] is None:
            loc = bounds[0]
            scale = 1
        return Chain([Exp(), Affine(loc, scale)])

    # bounded on both sides
    else:
        loc = bounds[0]
        scale = bounds[1] - bounds[0]
        return Chain([Logistic(), Affine(loc, scale)])


def Bounder(bounds):
    return Stack(list(map(UnivariateBounder, bounds)))


def Colour(norms):
    mean = jnp.mean(norms, axis = 0)
    std = jnp.std(norms, axis = 0)
    return Affine(loc = mean, scale = std)


def Whiten(norms):
    return Invert(Colour)


def ColourAndBound(bounds = None, norms = None):
    if bounds is None and norms is None:
        return Identity()
    elif bounds is not None and norms is None:
        return Bounder(bounds)
    elif bounds is None and norms is not None:
        return Colour(norms)
    else:
        bounder = Bounder(bounds)
        colour = Colour(jax.vmap(bounder.inverse)(norms))
        return Chain([colour, bounder])


def get_post_stack1d(bounds = None, norms = None):
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

