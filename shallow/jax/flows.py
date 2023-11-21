from functools import partial
from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import jax_tqdm
import equinox
import optax
# import optimistix

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

from .utils import params_to_array, get_array_to_params, count_params


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


## TODO: bounder bijection in case of no bounds
def get_flow(flow, bounds=[None], norms=None):
    bounder = Stack([get_bounder(bound) for bound in bounds])
    if norms is not None:
        debounded_norms = jax.vmap(bounder.inverse)(norms)
        denormer = Invert(get_normer(debounded_norms))
        bounder = Chain([denormer, bounder])
    base_dist = flow.base_dist
    bijection = Chain([flow.bijection, bounder])
    return Transformed(base_dist, bijection)


def get_filter(flow, filter_spec=equinox.is_inexact_array):
    params = jax.tree_util.tree_map(filter_spec, flow)
    filter_spec = equinox.tree_at(
        lambda tree: tree.bijection[-1], params, replace=False,
        )
    return filter_spec


## TODO: make this work
# def numerical_inverse(flow, z, solver=None, bounds=None):
#     fn = lambda x, z: flow.bijection.inverse(x) - z
#     if solver is None:
#         solver = optimistix.Newton(rtol=1e-5, atol=1e-5)
#     if flow.__class__.__name__ == 'BoundedFlow':
#         initial = lambda z: flow.bijection[-1].transform(z)
#     else:
#         initial = lambda z: z
#     if bounds is not None:
#         lower = jnp.array([bound[0] for bound in bounds])
#         upper = jnp.array([bound[1] for bound in bounds])
#         options = dict(lower=lower, upper=upper)
#     else:
#         options = {}
#     def single(z):
#         x0 = initial(z)
#         result = optimistix.root_find(
#             fn, solver, x0, z, options=options,
#             )
#         return result.value
#     x = jax.vmap(single)(z)
#     return x


# def numerical_sampling(flow, key, shape, solver=None):
#     z = flow.base_dist.sample(key, shape)
#     x = numerical_inverse(flow, z, solver)
#     return x


def nll(flow, x):
    return -flow.log_prob(x)


def ce(flow, x):
    return nll(flow, x).mean()


def trainer(
    key,
    flow,
    x,
    valid=None,
    batch_size=None,
    all_batches=True,
    epochs=1,
    patience=None,
    stop_if_inf=True,
    lr=1e-3,
    wd=0,
    loss_fn=None,
    print_batch=False,
    print_epoch=True,
    filter_spec=equinox.is_inexact_array,
    ):

    params, static = equinox.partition(flow, filter_spec)
    opt = optax.adamw(learning_rate=lr, weight_decay=wd)
    state = opt.init(params)

    if loss_fn is None:
        loss_fn = nll
    def loss_vmap(params, x):
        return jax.vmap(partial(loss_fn, equinox.combine(params, static)))(x)
    loss_batch = lambda params, x: loss_vmap(params, x).mean()
    loss_and_grad = jax.value_and_grad(loss_batch)

    xt = x
    if valid is not None:
        xv = valid
        if type(valid) is float:
            assert 0 < valid < 1
            nv = max(int(valid * x.shape[0]), 1)
            key, key_ = jax.random.split(key)
            shuffle = jax.random.permutation(key, x.shape[0])
            xv = x[shuffle][:nv]
            xt = x[shuffle][nv:]
    
    nt = xt.shape[0]
    if batch_size is None:
        batch_size = nt
    nbt, remt = divmod(nt, batch_size)

    def train_step(carry, ix):
        params, state = carry
        i, x = ix
        loss, grad = loss_and_grad(params, x)
        updates, state = opt.update(grad, state, params)
        params = equinox.apply_updates(params, updates)
        return (params, state), loss

    def valid_step(params, ix):
        i, x = ix
        return params, loss_batch(params, x)

    if all_batches:

        if nbt == 1:
            print_batch = False
            def train_scan(params, state, x):
                carry, losses = train_step((params, state), (None, x))
                return *carry, losses

        elif remt == 0:
            def train_scan(params, state, x):
                xs = x.reshape(nbt, batch_size, *x.shape[1:])
                carry, losses = jax.lax.scan(
                    train_step, (params, state), (jnp.arange(nbt), xs),
                    )
                return *carry, losses

        else:
            def train_scan(params, state, x):
                xs = x[:-remt].reshape(nbt, batch_size, *x.shape[1:])
                carry, losses = jax.lax.scan(
                    train_step, (params, state), (jnp.arange(nbt), xs),
                    )
                carry, loss = train_step(carry, (None, x[-remt:]))
                losses = jnp.concatenate([losses, jnp.array([loss])])
                return *carry, losses

        if valid is None:
            def epoch_step(key, params, state):
                key, tkey = jax.random.split(key)
                shuffle = jax.random.permutation(tkey, nt)
                params, state, losses = train_scan(params, state, xt[shuffle])
                loss = losses.mean()
                return key, params, state, (loss,)
    
        else:        
            nv = xv.shape[0]
            vbatch_size = batch_size
            if vbatch_size > nv:
                vbatch_size = nv
            nbv, remv = divmod(nv, vbatch_size)
    
            if nbv == 1:
                def valid_scan(params, x):
                    params, losses = valid_step(params, (None, x))
                    return losses

            elif remv == 0:
                def valid_scan(params, x):
                    xs = x.reshape(nbv, vbatch_size, *x.shape[1:])
                    params, losses = jax.lax.scan(
                        valid_step, params, (jnp.arange(nbv), xs),
                        )
                    return losses

            else:
                def valid_scan(params, x):
                    xs = x[:-remv].reshape(nbv, vbatch_size, *x.shape[1:])
                    params, losses = jax.lax.scan(
                        valid_step, params, (jnp.arange(nbv), xs),
                        )
                    params, loss = valid_step(params, (None, x[-remv:]))
                    return jnp.concatenate([losses, jnp.array([loss])])

            def epoch_step(key, params, state):
                key, tkey, vkey = jax.random.split(key, 3)
                shuffle = jax.random.permutation(tkey, nt)
                params, state, losses = train_scan(params, state, xt[shuffle])
                tloss = losses.mean()
                shuffle = jax.random.permutation(vkey, nv)
                vloss = valid_scan(params, xv[shuffle]).mean()
                return key, params, state, (tloss, vloss)

    else:
        print_batch = False

        if valid is None:
            def epoch_step(key, params, state):
                key, tkey = jax.random.split(key)
                xs = jax.random.choice(tkey, xt, shape=(batch_size,))
                (params, state), loss = train_step((params, state), (None, xs))
                return key, params, state, (loss,)

        else:
            def epoch_step(key, params, state):
                key, tkey, vkey = jax.random.split(key, 3)
                xs = jax.random.choice(tkey, xt, shape=(batch_size,))
                (params, state), tloss = train_step((params, state), (None, xs))
                xs = jax.random.choice(vkey, xv, shape=(batch_size,))
                params, vloss = valid_step(params, (None, xs))
                return key, params, state, (tloss, vloss)

    def cond_loss(carry, epoch):
        key, params, state, best = carry
        key, params, state, loss = epoch_step(key, params, state)
        best_epoch, best_loss, best_params = best
        best = jax.lax.cond(
            loss[-1] < best_loss,
            lambda: (epoch, loss[-1], params),
            lambda: best,
            )
        return key, params, state, best, loss

    pred_patience = lambda epoch, best_epoch: epoch > best_epoch + patience - 1
    pred_inf = lambda loss: jnp.logical_not(jnp.isfinite(loss))
    if patience is not None and not stop_if_inf:
        def pred_fn(loss, epoch, best_epoch):
            return pred_patience(epoch, best_epoch)
    elif patience is None and stop_if_inf:
        def pred_fn(loss, epoch, best_epoch):
            return pred_inf(loss)
    elif patience is not None and stop_if_inf:
        def pred_fn(loss, epoch, best_epoch):
            return jnp.logical_or(
                pred_patience(epoch, best_epoch), pred_inf(loss),
                )
    else:
        def pred_fn(loss, epoch, best_epoch):
            return False
 
    nanloss = (jnp.nan,) if valid is None else (jnp.nan, jnp.nan)

    def cond_patience(carry, epoch):
        key, params, state, best, stop = carry
        key, params, state, best, loss = jax.lax.cond(
            stop,
            lambda carry, epoch: (*carry, nanloss),
            cond_loss,
            (key, params, state, best),
            epoch,
            )
        best_epoch, best_loss, best_params = best
        stop = pred_fn(loss[-1], epoch, best_epoch)
        return (key, params, state, best, stop), loss

    prints_ = [print_epoch, print_batch]
    sizes = [epochs, nbt]
    if valid is not None:
        prints_.append(print_batch)
        sizes_.append(nbv)
    prints = []
    for print_, size in zip(prints_, sizes):
        if print_:
            if print_ is True:
                print_ = 1
            elif type(print_) is float:
                assert 0 < print_ <= 1
                print_ = max(int(print_ * size), 1)
            else:
                assert type(print_) is int
                assert 0 < print_ < size
        prints.append(print_)
    if valid is None:
        print_epoch, print_train = prints
    else:
        print_epoch, print_train, print_valid = prints
    tqdm._instances.clear()

    if print_epoch:
        cond_loss = jax_tqdm.scan_tqdm(
            epochs,
            print_rate=print_epoch,
            desc='epoch',
            position=0,
            leave=True,
            )(cond_loss)
    p = int(bool(print_epoch))
    if print_train:
        train_step = jax_tqdm.scan_tqdm(
            nbt, print_rate=print_train, desc='train', position=p, leave=False,
            )(train_step)
    if valid is not None and print_valid:
        valid_step = jax_tqdm.scan_tqdm(
            nbv, print_rate=print_valid, desc='valid', position=p, leave=False,
            )(valid_step)

    (key, params, state, best, stop), losses = jax.lax.scan(
        cond_patience,
        (key, params, state, (0, jnp.inf, params), False),
        jnp.arange(epochs),
        )
    best_epoch, best_loss, best_params = best
    
    flow = equinox.combine(best_params, static)

    if patience is not None or stop_if_inf:
        losses = jnp.array(losses)
        if jnp.any(~jnp.isfinite(losses)):
            cut = jnp.argwhere(~jnp.isfinite(losses))[:, 1].min()
            if (
                cut == best_epoch + patience + 1 and 
                jnp.isnan(losses[:, cut]).all()
                ):
                print('Stopped: patience reached')
            else:
                print('Stopped: loss is not finite')
            losses = losses[:, :cut]
    losses = {k: v for k, v in zip(['train', 'valid'], losses)}

    return flow, losses


# def _trainer(
#     key,
#     flow,
#     x,
#     valid=None, ## TODO: add validation steps
#     batch_size=None,
#     max_epochs=1,
#     patience=None,
#     lr=1e-3,
#     opt=None,
#     loss_fn=None,
#     print_batch=False,
#     print_epoch=True,
#     filter_spec=equinox.is_inexact_array,
#     ):
    
#     params, static = equinox.partition(flow, filter_spec)

#     nx = x.shape[0]
#     if batch_size is None:
#         batch_size = nx
#         print_batch = False
#     nbatch, remainder = divmod(nx, batch_size)
#     nbatch += int(remainder > 0)

#     if patience is not None:
#         if type(patience) is float:
#             assert 0 < patience < 1
#             patience = max(int(patience * max_epochs), 1)
#         else:
#             assert type(patience is int)
#             assert 0 < patience < max_epochs

#     if opt is None:
#         opt = optax.adam(lr)
#     state = opt.init(params)
    
#     if loss_fn is None:
#         loss_fn = lambda params, x: ce(equinox.combine(params, static), x)

#     prints = []
#     for print_, size in zip(
#         (print_batch, print_epoch), (nbatch, max_epochs),
#         ):
#         if print_:
#             if print_ is True:
#                 print_ = 1
#             elif type(print_) is float:
#                 assert 0 < print_ < 1
#                 print_ = max(int(size * print_), 1)
#             else:
#                 assert type(print_) is int
#                 assert 0 < print_ < size
#         prints.append(print_)
#     print_batch, print_epoch = prints
#     tqdm._instances.clear()

#     args = (
#         key,
#         static,
#         params,
#         x,
#         valid,
#         batch_size,
#         max_epochs,
#         patience,
#         opt,
#         state,
#         loss_fn,
#         print_batch,
#         print_epoch,
#         )
#     if remainder == 0:
#         if patience is None:
#             print('scanning batches and all epochs')
#             key, best_params, losses = _trainer_scan_epoch(*args)
#         else:
#             print('scanning batches and epochs up to patience')
#             # key, best_params, losses = _trainer_scan_batch(*args)
#             key, best_params, losses = _trainer_scan_epoch_with_patience(*args)
#     else:
#         print('looping batches and epochs')
#         key, best_params, losses = _trainer_loop(*args)
#     flow = equinox.combine(best_params, static)

#     return key, flow, losses


# def _trainer_loop(
#     key,
#     static,
#     params,
#     x,
#     valid,
#     batch_size,
#     max_epochs,
#     patience,
#     opt,
#     state,
#     loss_fn,
#     print_batch,
#     print_epoch,
#     ):

#     splits = jnp.arange(batch_size, x.shape[0], batch_size)
#     nbatches = len(splits) + 1

#     @jax.jit
#     def step(params, state, batch):
#         loss, grad = jax.value_and_grad(loss_fn)(params, batch)
#         updates, state = opt.update(grad, state)
#         params = equinox.apply_updates(params, updates)
#         return params, state, loss

#     loop_epoch = range(max_epochs)
#     if print_epoch:
#         loop_epoch = tqdm(
#             loop_epoch, desc='epoch', miniters=print_epoch, position=0,
#             )

#     losses = []
#     for epoch in loop_epoch:
#         key, key_ = jax.random.split(key)
#         batches = jnp.array_split(jax.random.permutation(key_, x), splits)
        
#         loop_batch = batches
#         if print_batch:
#             loop_batch = tqdm(
#                 loop_batch,
#                 desc='batch',
#                 miniters=print_batch,
#                 position=1,
#                 # leave=False,
#                 )

#         loss = 0
#         for batch in loop_batch:
#             params, state, lossx = step(params, state, batch)
#             loss += lossx
#         loss /= nbatches
#         losses.append(loss)

#         if loss == min(losses):
#             best_epoch = epoch
#             best_params = params

#         if patience is not None:
#             if epoch - best_epoch > patience:
#                 break

#     return key, best_params, jnp.array(losses)


# def _trainer_scan_batch(
#     key,
#     static,
#     params,
#     x,
#     valid,
#     batch_size,
#     max_epochs,
#     patience,
#     opt,
#     state,
#     loss_fn,
#     print_batch,
#     print_epoch,
#     ):

#     nbatch = x.shape[0] // batch_size
#     ndim = x.shape[1]

#     def train_batch(carry, idx_batch):
#         params, state = carry
#         idx, batch = idx_batch
#         loss, grad = jax.value_and_grad(loss_fn)(params, batch)
#         updates, state = opt.update(grad, state)
#         params = equinox.apply_updates(params, updates)
#         return (params, state), loss

#     if print_batch:
#         train_batch = jax_tqdm.scan_tqdm(
#             nbatch, print_rate=print_batch, message='batch',
#             )(train_batch)

#     @jax.jit
#     def train_epoch(key, params, state):
#         key, key_ = jax.random.split(key)
#         batches = jnp.reshape(
#             jax.random.permutation(key_, x),
#             (nbatch, batch_size, *x.shape[1:]),
#             )
#         (params, state), losses = jax.lax.scan(
#             train_batch, (params, state), (jnp.arange(nbatch), batches),
#             )
#         loss = losses.mean()
#         return key, params, state, loss

#     loop_epoch = range(max_epochs)
#     if print_epoch:
#         loop_epoch = tqdm(
#             loop_epoch, desc='epoch', miniters=print_epoch, position=0,
#             )

#     losses = []
#     for epoch in loop_epoch:
#         key, params, state, loss = train_epoch(key, params, state)
#         losses.append(loss)
        
#         if loss == min(losses):
#             best_epoch = epoch
#             best_params = params
            
#         if patience is not None:
#             if epoch - best_epoch > patience:
#                 break

#     return key, best_params, jnp.array(losses)


# def _trainer_scan_epoch_with_patience(
#     key,
#     static,
#     params,
#     x,
#     valid,
#     batch_size,
#     max_epochs,
#     patience,
#     opt,
#     state,
#     loss_fn,
#     print_batch,
#     print_epoch,
#     ):

#     nbatch = x.shape[0] // batch_size
#     ndim = x.shape[1]

#     def train_batch(carry, idx_batch):
#         params, state = carry
#         idx, batch = idx_batch
#         loss, grad = jax.value_and_grad(loss_fn)(params, batch)
#         updates, state = opt.update(grad, state)
#         params = equinox.apply_updates(params, updates)
#         return (params, state), loss

#     if print_batch:
#         train_batch = jax_tqdm.scan_tqdm(
#             nbatch, print_rate=print_batch, message='batch',
#             )(train_batch)

#     def cond_loss(current, best):
#         epoch, loss, params = current
#         best_epoch, best_loss, best_params = best
#         pred = loss < best_loss
#         true_fn = lambda: current
#         false_fn = lambda: best
#         return jax.lax.cond(pred, true_fn, false_fn)
        
#     def train_epoch(carry, epoch):
#         key, params, state = carry
#         key, key_ = jax.random.split(key)
#         batches = jnp.reshape(
#             jax.random.permutation(key_, x),
#             (nbatch, batch_size, *x.shape[1:]),
#             )
#         (params, state), losses = jax.lax.scan(
#             train_batch, (params, state), (jnp.arange(nbatch), batches),
#             )
#         loss = losses.mean()
#         return (key, params, state), loss

#     if print_epoch:
#         train_epoch = jax_tqdm.scan_tqdm(
#             max_epochs, print_rate=print_epoch, message='epoch',
#             )(train_epoch)

#     def cond_patience(carry, epoch):
#         key, params, state, best = carry
#         best_epoch, best_loss, best_params = best
#         pred = epoch > best_epoch + patience - 1
#         true_fn = lambda carry, epoch: (carry, jnp.nan)
#         def false_fn(carry, epoch):
#             key, params, state, best = carry
#             (key, params, state), loss = train_epoch((key, params, state), epoch)
#             best = cond_loss((epoch, loss, params), best)
#             return (key, params, state, best), loss
#         return jax.lax.cond(pred, true_fn, false_fn, carry, epoch)

#     best = 0, jnp.inf, params
#     (key, params, state, best), losses = jax.lax.scan(
#         cond_patience, (key, params, state, best), jnp.arange(max_epochs),
#         )
#     best_epoch, best_loss, best_params = best
#     losses = losses[:best_epoch+patience]

#     return key, best_params, losses
    

# def _trainer_scan_epoch(
#     key,
#     static,
#     params,
#     x,
#     valid,
#     batch_size,
#     max_epochs,
#     patience,
#     opt,
#     state,
#     loss_fn,
#     print_batch,
#     print_epoch,
#     ):

#     nbatch = x.shape[0] // batch_size
#     ndim = x.shape[1]

#     def train_batch(carry, idx_batch):
#         params, state = carry
#         idx, batch = idx_batch
#         loss, grad = jax.value_and_grad(loss_fn)(params, batch)
#         updates, state = opt.update(grad, state)
#         params = equinox.apply_updates(params, updates)
#         return (params, state), loss

#     if print_batch:
#         train_batch = jax_tqdm.scan_tqdm(
#             nbatch, print_rate=print_batch, message='batch',
#             )(train_batch)

#     def cond_loss(current, best):
#         params, loss = current
#         best_params, best_loss = best
#         pred = loss < best_loss
#         true_fn = lambda: current
#         false_fn = lambda: best
#         return jax.lax.cond(pred, true_fn, false_fn)

#     def train_epoch(carry, epoch):
#         key, params, state, best = carry
#         key, key_ = jax.random.split(key)
#         batches = jnp.reshape(
#             jax.random.permutation(key_, x),
#             (nbatch, batch_size, *x.shape[1:]),
#             )
#         (params, state), losses = jax.lax.scan(
#             train_batch, (params, state), (jnp.arange(nbatch), batches),
#             )
#         loss = losses.mean()
#         best = cond_loss((params, loss), best)
#         return (key, params, state, best), loss

#     if print_epoch:
#         train_epoch = jax_tqdm.scan_tqdm(
#             max_epochs, print_rate=print_epoch, message='epoch',
#             )(train_epoch)

#     best = params, jnp.inf
#     (key, params, state, best), losses = jax.lax.scan(
#         train_epoch, (key, params, state, best), jnp.arange(max_epochs),
#         )
#     best_params, best_loss = best

#     return key, best_params, losses
