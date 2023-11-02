import numpy as np
from tqdm import tqdm

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
    valid=None, ## TODO
    batch_size=None,
    max_epochs=1,
    patience=None,
    lr=1e-3,
    opt=None,
    loss_fn=None,
    print_batch=False,
    print_epoch=True,
    ):

    nx = x.shape[0]
    if batch_size is None:
        batch_size = nx
    splits = jnp.arange(batch_size, nx, batch_size)
    nbatches = len(splits) + 1

    params, static = equinox.partition(flow, equinox.is_inexact_array)
    if loss_fn is None:
        get_flow = lambda params: equinox.combine(static, params)
        loss_fn = lambda params, x: get_flow(params).log_prob(x).mean()
    loss_and_grad = jax.value_and_grad(loss_fn)

    if opt is None:
        opt = optax.adam(lr)
    state = opt.init(params)

    def step(params, state, x):
        loss, grad = loss_and_grad(params, x)
        updates, state = opt.update(grad, state)
        params = equinox.apply_updates(params, updates)
        return params, state, loss

    def get_batches(key):
        key, key_ = jax.random.split(key)
        xs = jax.random.permutation(key_, x)
        xs = jnp.array_split(x, slpits)
        return key, xs

    epoch_loop = range(max_epochs)
    if print_epoch:
        miniters = 1 if print_epoch is True else print_epoch
        loop_epoch = tqdm(epoch_loop, desc='epoch', miniters=miniters)

    losses = []
    for epoch in loop_epoch:
        key, xs = get_batches(key)
        loop_batch = xs
        if print_batch:
            miniters = 1 if print_batch is True else print_batch
            loop_batch = tqdm(loop_batch, desc='batch', miniters=miniters)

        loss = 0
        for x in loop_batch:
            params, state, lossx = step(params, state, x)
            loss += lossx
        loss /= nbatches
        losses.append(loss)

        if loss == min(losses):
            best_epoch = epoch
            best_params = params
        if epoch - best_epoch > patience:
            break

    flow = equinox.combine(static, best_params)

    return key, flow, losses


def _trainer_scan(
    key,
    flow,
    x,
    valid=None,
    batch_size=100,
    max_epochs=100,
    patience=10,
    lr=1e-3,
    opt=None,
    loss_fn=None,
    print_batch=False,
    print_epoch=True,
    ):

    nx = x.shape[0]
    
    params, static = equinox.partition(flow, equinox.is_inexact_array)
    if loss_fn is None:
        loss_fn = lambda params, x: -equinox.combine(static, params).log_prob(x).mean()
    loss_and_grad = jax.value_and_grad(loss_fn)

    if opt is None:
        opt = optax.adam(lr)
    state = opt.init(params)

    def train_batch(carry, x):
        params, state = carry
        loss, grad = loss_and_grad(params, x)
        updates, state = opt.update(grad, state)
        params = optax.apply_updates(params, updates)
        return (params, state), loss

    def get_batches(key):
        key, key_ = jax.random.split(key)
        xs = jax.random.permutation(key_, x)
        splits = jnp.arange(batch_size, nx, batch_size)
        xs = jnp.array_split(xs, splits)
        return key, xs

    def cond_loss(current, best):
        epoch, loss, params = current
        best_epoch, best_loss, best_params = best
        pred = loss < best_loss
        true_fn = lambda: current
        false_fn = lambda: best
        return jax.lax.cond(pred, true_fn, false_fn)

    def train_epoch(key, params, state):
        init = params, state
        # key, xs = get_batches(key)
        key, key_ = jax.random.split(key)
        xs = jax.random.permutation(key_, x)
        # splits = jnp.arange(batch_size, nx, batch_size)
        splits = list(range(batch_size, nx, batch_size))
        xs = jnp.array_split(xs, splits)
        (params, state), losses = jax.lax.scan(train_batch, init, xs)
        loss = losses.mean()
        return key, params, state, loss

    def cond_patience(carry, epoch):
        key, params, state, best = carry
        best_epoch, best_loss, best_params = best
        pred = epoch - best_epoch > patience
        true_fn = lambda carry: (carry, jnp.nan)
        def false_fn(carry):
            key, params, state, best = carry
            key, params, state, loss = train_epoch(key, params, state)
            current = epoch, loss, params
            best = cond_loss(current, best)
            return (key, params, state, best), loss
        return jax.lax.cond(pred, true_fn, false_fn, carry)

    if print_batch:
        batches = sum(divmod(nx, batch_size))
        if print_batch is True:
            print_batch = sum(divmod(batches, 10))
        train_batch = jax_tqdm.scan_tqdm(
            batches, print_rate=print_batch, message='batch',
            )(train_batch)

    if print_epoch:
        if print_epoch is True:
            print_epoch = sum(divmod(max_epochs, 10))
        train_epoch = jax_tqdm.scan_tqdm(
            max_epochs, print_rate=print_batch, message='epoch',
            )(train_epoch)

    best_epoch = 0
    best_loss = jnp.inf
    best_params = params
    best = best_epoch, best_loss, best_params

    scan_fn = cond_patience #if patience is not None else train_epoch
    init = key, params, state, best
    epochs = jnp.arange(max_epochs)
    carry, losses = jax.lax.scan(scan_fn, init, epochs)

    key, params, state, best = carry
    best_epoch, best_loss, best_params = best
    losses = losses[:best_epoch + patience]
    flow = equinox.combine(static, best_params)

    return key, flow, losses
    
    








    

