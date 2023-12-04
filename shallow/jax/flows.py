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


def get_pre(bounds=[None], norms=None):
    bounder = Stack([get_bounder(bound) for bound in bounds])
    if norms is not None:
        debounded_norms = jax.vmap(bounder.inverse)(norms)
        denormer = Invert(get_normer(debounded_norms))
        bounder = Chain([denormer, bounder])
    return bounder


## TODO: bounder bijection in case of no bounds
def get_flow(flow, bounds=[None], norms=None):
    bijection = Chain([flow.bijection, get_pre(bounds, norms)])
    return Transformed(flow.base_dist, bijection)


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
#     if bounds:
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


def nll(flow, x, c=None):
    return -flow.log_prob(x, condition=c)


def ce(flow, x, c=None):
    return nll(flow, x, c=c).mean()


def trainer(
    key,
    flow,
    train,
    valid=None,
    batch_size=None,
    all_batches=True,
    epochs=1,
    patience=None,
    stop_if_inf=True,
    lr=1e-3,
    wd=None,
    opt=None,
    loss_fn=None,
    print_batch=False,
    print_epoch=True,
    filter_spec=equinox.is_inexact_array,
    ):

    if type(train) is tuple and len(train) == 2:
        xt, ct = tuple(map(jnp.asarray, train))
        assert xt.shape[0] == ct.shape[0]
        train = xt, ct
        conditional = True
    else:
        if type(train) is tuple:
            assert len(train) == 1
            train = train[0]
        xt = jnp.asarray(train)
        train = xt,
        conditional = False

    if valid is not None:
        if type(valid) is float:
            assert 0 < valid < 1
            nv = max(int(valid * xt.shape[0]), 1)
            key, pkey = jax.random.split(key)
            shuffle = jax.random.permutation(pkey, xt.shape[0])
            xv = xt[shuffle][:nv]
            xt = xt[shuffle][nv:]
            if conditional:
                cv = ct[shuffle][:nv]
                ct = ct[shuffle][nv:]
                train = xt, ct
                valid = xv, cv
            else:
                train = xt,
                valid = xv,
        elif conditional:
            assert type(valid) is tuple and len(valid) == 2
            xv, cv = tuple(map(jnp.asarray, valid))
            assert xv.shape[0] == cv.shape[0]
            assert cv.shape[1:] == ct.shape[1:]
            valid = xv, cv
        else:
            if type(valid) is tuple:
                assert len(valid) == 1
                valid = valid[0]
            xv = jnp.asarray(valid)
            valid = xv,
        assert xv.shape[1:] == xt.shape[1:]
        
    nt = xt.shape[0]
    if batch_size is None:
        batch_size = nt

    flow = equinox.nn.inference_mode(flow, False)
    params, static = equinox.partition(flow, filter_spec)
    if opt is None:
        if wd is None:
            opt = optax.adam(learning_rate=lr)
        else:
            opt = optax.adamw(learning_rate=lr, weight_decay=wd)
    elif callable(opt):
        if wd is None:
            opt = opt(learning_rate=lr)
        else:
            opt = opt(learning_rate=lr, weight_decay=wd)
    state = opt.init(params)

    if loss_fn is None:
        loss_fn = ce
    loss_batch = lambda params, *x: loss_fn(equinox.combine(params, static), *x)
    loss_and_grad = jax.value_and_grad(loss_batch)

    def train_step(carry, x):
        params, state = carry
        loss, grad = loss_and_grad(params, *x)
        updates, state = opt.update(grad, state, params)
        params = equinox.apply_updates(params, updates)
        return (params, state), loss

    def train_batch(carry, ix):
        i, x = ix
        return train_step(carry, x)

    def valid_step(params, x):
        return loss_batch(params, *x)

    def valid_batch(params, ix):
        i, x = ix
        return valid_step(params, x)

    if all_batches:
        if batch_size > nt:
            batch_size = nt
        nbt, remt = divmod(nt, batch_size)
        if valid:
            nv = xv.shape[0]
            vbatch_size = min(batch_size, nv)
            nbv, remv = divmod(nv, vbatch_size)
    
        def get_batch(key, x, batch_size):
            shuffle = jax.random.permutation(key, x[0].shape[0])
            return tuple(map(
                lambda x: x[shuffle].reshape(-1, batch_size, *x.shape[1:]), x,
                ))
            
        if nbt == 1 and remt == 0:
            def train_scan(key, params, state):
                x = get_batch(key, train, batch_size)
                x = tuple(map(lambda x: x[0], x))
                carry, losses = train_step((params, state), x)
                return *carry, losses

        elif remt == 0:
            def train_scan(key, params, state):
                x = get_batch(key, train, batch_size)
                carry, losses = jax.lax.scan(
                    train_batch, (params, state), (jnp.arange(nbt), x),
                    )
                return *carry, losses

        else: ## TODO: last leftover batch doesn't work with progress bar
            def train_scan(key, params, state):
                x = tuple(map(lambda x: x[:-remt], train))
                x = get_batch(key, x, batch_size)
                carry, losses = jax.lax.scan(
                    train_batch, (params, state), (jnp.arange(nbt), x),
                    )
                x = tuple(map(lambda x: x[-remt:], train))
                carry, loss = train_step(carry, x)
                return *carry, jnp.concatenate([losses, jnp.array([loss])])

        if valid:
            
            if nbv == 1 and remv == 0:
                def valid_scan(key, params):
                    x = get_batch(key, valid, vbatch_size)
                    x = tuple(map(lambda x: x[0], x))
                    return valid_step(params, x)

            elif remv == 0:
                def valid_scan(key, params):
                    x = get_batch(key, valid, vbatch_size)
                    return jax.lax.scan(
                        lambda carry, ix: (None, valid_batch(params, ix)),
                        None,
                        (jnp.arange(nbv), x),
                        )[1]

            else:
                def valid_scan(key, params):
                    x = tuple(map(lambda x: x[:-remv], valid))
                    x = get_batch(key, x, vbatch_size)
                    losses = jax.lax.scan(
                        lambda carry, ix: (None, valid_batch(params, ix)),
                        None,
                        (jnp.arange(nbv), x),
                        )[1]
                    x = tuple(map(lambda x: x[-remv:], valid))
                    loss = valid_step(params, x)
                    return jnp.concatenate([losses, jnp.array([loss])])

            def epoch_step(key, params, state):
                key, tkey, vkey = jax.random.split(key, 3)
                params, state, losses = train_scan(tkey, params, state)
                tloss = losses.mean()
                vloss = valid_scan(vkey, params).mean()
                return key, params, state, (tloss, vloss)

        else:
            def epoch_step(key, params, state):
                key, tkey = jax.random.split(key)
                params, state, losses = train_scan(tkey, params, state)
                loss = losses.mean()
                return key, params, state, (loss,)

    else:
        print_batch = False

        def get_batch(key, x, batch_size):
            idxs = jax.random.choice(key, x[0].shape[0], shape=(batch_size,))
            return tuple(map(lambda x: x[idxs], x))

        if valid:
            def epoch_step(key, params, state):
                key, tkey, vkey = jax.random.split(key, 3)
                x = get_batch(tkey, train, batch_size)
                (params, state), tloss = train_step((params, state), x)
                x = get_batch(vkey, valid, batch_size)
                vloss = valid_step(params, x)
                return key, params, state, (tloss, vloss)
        else:
            def epoch_step(key, params, state):
                key, tkey = jax.random.split(key)
                x = get_batch(tkey, train, batch_size)
                (params, state), loss = train_step((params, state), x)
                return key, params, state, (loss,)

    def cond_loss(carry, epoch):
        key, params, state, best = carry
        key, params, state, loss = epoch_step(key, params, state)
        best_epoch, best_loss, best_params = best
        best = jax.lax.cond(
            loss[-1] < best_loss, # -1 here refers to tloss or vloss, not epoch
            lambda: (epoch, loss[-1], params),
            lambda: best,
            )
        return key, params, state, best, loss

    pred_patience = lambda epoch, best_epoch: epoch > best_epoch + patience - 1
    pred_inf = lambda loss: jnp.logical_not(jnp.isfinite(jnp.array(loss))).all()
    if patience and not stop_if_inf:
        def pred_fn(loss, epoch, best_epoch):
            return pred_patience(epoch, best_epoch)
    elif not patience and stop_if_inf:
        def pred_fn(loss, epoch, best_epoch):
            return pred_inf(loss)
    elif patience and stop_if_inf:
        def pred_fn(loss, epoch, best_epoch):
            return jnp.logical_or(
                pred_patience(epoch, best_epoch), pred_inf(loss),
                )
    else:
        def pred_fn(loss, epoch, best_epoch):
            return False

    nanloss = (jnp.nan, jnp.nan) if valid else (jnp.nan,)
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
        stop = pred_fn(loss, epoch, best_epoch)
        return (key, params, state, best, stop), loss

    prints = []
    sizes = []
    if print_epoch:
        prints.append(print_epoch)
        sizes.append(epochs)
    if print_batch:
        prints.append(print_batch)
        sizes.append(nbt)
        if valid:
            prints.append(print_batch)
            sizes.append(nbv)
    for i in range(len(prints)):
        if prints[i]:
            if prints[i] is True:
                prints[i] = 1
            elif type(prints[i]) is float:
                assert 0 < prints[i] <= 1
                prints[i] = max(int(prints[i] * sizes[i]), 1)
            else:
                assert type(prints[i]) is int
                assert 0 < prints[i] <= sizes[i]

    if print_epoch:
        cond_loss = jax_tqdm.scan_tqdm(
            epochs,
            print_rate=prints.pop(0),
            desc='epoch',
            position=0,
            leave=True,
            )(cond_loss)

    p = int(bool(print_epoch))
    if print_batch:
        train_batch = jax_tqdm.scan_tqdm(
            nbt, print_rate=prints[0], desc='train', position=p, leave=False,
            )(train_batch)
        if valid:
            valid_batch = jax_tqdm.scan_tqdm(
                nbv,
                print_rate=prints[1],
                desc='valid',
                position=p,
                leave=False,
                )(valid_batch)

    tqdm._instances.clear()

    (key, params, state, best, stop), losses = jax.lax.scan(
        cond_patience,
        (key, params, state, (0, jnp.inf, params), False),
        jnp.arange(epochs),
        )
    best_epoch, best_loss, best_params = best

    flow = equinox.combine(best_params, static)
    flow = equinox.nn.inference_mode(flow, True)

    if stop:
        losses = jnp.array(losses)
        cut = jnp.argwhere(~jnp.isfinite(losses))[:, 1].min()
        if (
            patience and
            cut == best_epoch + patience + 1 and
            jnp.isnan(losses[:, cut]).all()
            ):
            print('Stopped: patience reached')
        else:
            print('Stopped: loss is not finite')
        losses = losses[:, :cut]
    losses = {k: v for k, v in zip(('train', 'valid'), losses)}

    return flow, losses


# def trainer(
#     key,
#     flow,
#     x,
#     valid=None,
#     batch_size=None,
#     all_batches=True,
#     epochs=1,
#     patience=None,
#     stop_if_inf=True,
#     lr=1e-3,
#     wd=0,
#     loss_fn=None,
#     print_batch=False,
#     print_epoch=True,
#     filter_spec=equinox.is_inexact_array,
#     ):

#     params, static = equinox.partition(flow, filter_spec)
#     opt = optax.adamw(learning_rate=lr, weight_decay=wd)
#     state = opt.init(params)

#     if loss_fn is None:
#         loss_fn = ce
#     # def loss_vmap(params, x):
#     #     return jax.vmap(partial(loss_fn, equinox.combine(params, static)))(x)
#     # loss_batch = lambda params, x: loss_vmap(params, x).mean()
#     loss_batch = lambda params, x: loss_fn(equinox.combine(params, static), x)
#     loss_and_grad = jax.value_and_grad(loss_batch)

#     xt = x
#     if valid is not None:
#         xv = valid
#         if type(valid) is float:
#             assert 0 < valid < 1
#             nv = max(int(valid * x.shape[0]), 1)
#             key, key_ = jax.random.split(key)
#             shuffle = jax.random.permutation(key_, x.shape[0])
#             xv = x[shuffle][:nv]
#             xt = x[shuffle][nv:]
    
#     nt = xt.shape[0]
#     if batch_size is None:
#         batch_size = nt
#     nbt, remt = divmod(nt, batch_size)

#     def train_step(carry, x):
#         params, state = carry
#         loss, grad = loss_and_grad(params, x)
#         updates, state = opt.update(grad, state, params)
#         params = equinox.apply_updates(params, updates)
#         return (params, state), loss

#     def train_batch(carry, ix):
#         i, x = ix
#         return train_step(carry, x)

#     def valid_step(params, x):
#         return params, loss_batch(params, x)

#     def valid_batch(params, ix):
#         i, x = ix
#         return valid_step(params, x)

#     if all_batches:

#         if nbt == 1:
#             def train_scan(params, state, x):
#                 carry, losses = train_step((params, state), x)
#                 return *carry, losses

#         elif remt == 0:
#             def train_scan(params, state, x):
#                 xs = x.reshape(nbt, batch_size, *x.shape[1:])
#                 carry, losses = jax.lax.scan(
#                     train_batch, (params, state), (jnp.arange(nbt), xs),
#                     )
#                 return *carry, losses

#         else: ## TODO: last leftover batch doesn't work with progress bar
#             def train_scan(params, state, x):
#                 xs = x[:-remt].reshape(nbt, batch_size, *x.shape[1:])
#                 carry, losses = jax.lax.scan(
#                     train_batch, (params, state), (jnp.arange(nbt), xs),
#                     )
#                 carry, loss = train_step(carry, x[-remt:])
#                 losses = jnp.concatenate([losses, jnp.array([loss])])
#                 return *carry, losses

#         if valid is None:
#             def epoch_step(key, params, state):
#                 key, tkey = jax.random.split(key)
#                 shuffle = jax.random.permutation(tkey, nt)
#                 params, state, losses = train_scan(params, state, xt[shuffle])
#                 loss = losses.mean()
#                 return key, params, state, (loss,)
    
#         else:        
#             nv = xv.shape[0]
#             vbatch_size = batch_size
#             if vbatch_size > nv:
#                 vbatch_size = nv
#             nbv, remv = divmod(nv, vbatch_size)
    
#             if nbv == 1:
#                 def valid_scan(params, x):
#                     params, losses = valid_step(params, x)
#                     return losses

#             elif remv == 0:
#                 def valid_scan(params, x):
#                     xs = x.reshape(nbv, vbatch_size, *x.shape[1:])
#                     params, losses = jax.lax.scan(
#                         valid_batch, params, (jnp.arange(nbv), xs),
#                         )
#                     return losses

#             else: ## TODO: last leftover batch doesn't work with progress bar
#                 def valid_scan(params, x):
#                     xs = x[:-remv].reshape(nbv, vbatch_size, *x.shape[1:])
#                     params, losses = jax.lax.scan(
#                         valid_batch, params, (jnp.arange(nbv), xs),
#                         )
#                     params, loss = valid_step(params, x[-remv:])
#                     return jnp.concatenate([losses, jnp.array([loss])])

#             def epoch_step(key, params, state):
#                 key, tkey, vkey = jax.random.split(key, 3)
#                 shuffle = jax.random.permutation(tkey, nt)
#                 params, state, losses = train_scan(params, state, xt[shuffle])
#                 tloss = losses.mean()
#                 shuffle = jax.random.permutation(vkey, nv)
#                 vloss = valid_scan(params, xv[shuffle]).mean()
#                 return key, params, state, (tloss, vloss)

#     else:
#         print_batch = False

#         if valid is None:
#             def epoch_step(key, params, state):
#                 key, tkey = jax.random.split(key)
#                 xs = jax.random.choice(tkey, xt, shape=(batch_size,))
#                 (params, state), loss = train_step((params, state), xs)
#                 return key, params, state, (loss,)

#         else:
#             def epoch_step(key, params, state):
#                 key, tkey, vkey = jax.random.split(key, 3)
#                 xs = jax.random.choice(tkey, xt, shape=(batch_size,))
#                 (params, state), tloss = train_step((params, state), xs)
#                 xs = jax.random.choice(vkey, xv, shape=(batch_size,))
#                 params, vloss = valid_step(params, xs)
#                 return key, params, state, (tloss, vloss)

#     def cond_loss(carry, epoch):
#         key, params, state, best = carry
#         key, params, state, loss = epoch_step(key, params, state)
#         best_epoch, best_loss, best_params = best
#         best = jax.lax.cond(
#             loss[-1] < best_loss,
#             lambda: (epoch, loss[-1], params),
#             lambda: best,
#             )
#         return key, params, state, best, loss

#     pred_patience = lambda epoch, best_epoch: epoch > best_epoch + patience - 1
#     pred_inf = lambda loss: jnp.logical_not(jnp.isfinite(loss))
#     if patience and not stop_if_inf:
#         def pred_fn(loss, epoch, best_epoch):
#             return pred_patience(epoch, best_epoch)
#     elif patience is None and stop_if_inf:
#         def pred_fn(loss, epoch, best_epoch):
#             return pred_inf(loss)
#     elif patience is not None and stop_if_inf:
#         def pred_fn(loss, epoch, best_epoch):
#             return jnp.logical_or(
#                 pred_patience(epoch, best_epoch), pred_inf(loss),
#                 )
#     else:
#         def pred_fn(loss, epoch, best_epoch):
#             return False
 
#     nanloss = (jnp.nan,) if valid is None else (jnp.nan, jnp.nan)

#     def cond_patience(carry, epoch):
#         key, params, state, best, stop = carry
#         key, params, state, best, loss = jax.lax.cond(
#             stop,
#             lambda carry, epoch: (*carry, nanloss),
#             cond_loss,
#             (key, params, state, best),
#             epoch,
#             )
#         best_epoch, best_loss, best_params = best
#         stop = pred_fn(loss[-1], epoch, best_epoch)
#         return (key, params, state, best, stop), loss

#     prints = []
#     sizes = []
#     if print_epoch:
#         prints.append(print_epoch)
#         sizes.append(epochs)
#     if print_batch:
#         prints.append(print_batch)
#         sizes.append(nbt)
#         if valid:
#             prints.append(print_batch)
#             sizes.append(nbv)
#     for i in range(len(prints)):
#         if prints[i]:
#             if prints[i] is True:
#                 prints[i] = 1
#             elif type(prints[i]) is float:
#                 assert 0 < prints[i] <= 1
#                 prints[i] = max(int(prints[i] * sizes[i]), 1)
#             else:
#                 assert type(prints[i]) is int
#                 assert 0 < prints[i] <= sizes[i]

#     if print_epoch:
#         cond_loss = jax_tqdm.scan_tqdm(
#             epochs,
#             print_rate=prints.pop(0),
#             desc='epoch',
#             position=0,
#             leave=True,
#             )(cond_loss)
    
#     p = int(bool(print_epoch))
#     if print_batch:
#         train_batch = jax_tqdm.scan_tqdm(
#             nbt, print_rate=prints[0], desc='train', position=p, leave=False,
#             )(train_batch)
#         if valid:
#             valid_batch = jax_tqdm.scan_tqdm(
#                 nbv,
#                 print_rate=prints[1],
#                 desc='valid',
#                 position=p,
#                 leave=False,
#                 )(valid_batch)

#     tqdm._instances.clear()

#     (key, params, state, best, stop), losses = jax.lax.scan(
#         cond_patience,
#         (key, params, state, (0, jnp.inf, params), False),
#         jnp.arange(epochs),
#         )
#     best_epoch, best_loss, best_params = best
    
#     flow = equinox.combine(best_params, static)

#     if patience is not None or stop_if_inf:
#         losses = jnp.array(losses)
#         if jnp.any(~jnp.isfinite(losses)):
#             cut = jnp.argwhere(~jnp.isfinite(losses))[:, 1].min()
#             if patience is not None:
#                 if (
#                     cut == best_epoch + patience + 1 and 
#                     jnp.isnan(losses[:, cut]).all()
#                     ):
#                     print('Stopped: patience reached')
#                 else:
#                     print('Stopped: loss is not finite')
#             else:
#                 print('Stopped: loss is not finite')
#             losses = losses[:, :cut]
#     losses = {k: v for k, v in zip(['train', 'valid'], losses)}

#     return flow, losses
