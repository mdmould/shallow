from functools import partial
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import jax_tqdm
import equinox
import optax

from collections.abc import Sequence
from itertools import accumulate
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.bijections import Affine, Chain, Invert

from .transforms import get_pre
from .utils import params_to_array, get_array_to_params, count_params


class Independent(AbstractDistribution):
    shape: tuple[int]
    cond_shape: None
    split_idxs: tuple[int, ...]
    distributions: Sequence[AbstractDistribution]

    def __init__(self, distributions):
        shapes = [d.shape for d in distributions]
        for s in shapes:
            assert len(s) == 1
        sizes = [s[0] for s in shapes]
        self.shape = (sum(sizes),)
        self.cond_shape = None
        self.split_idxs = tuple(accumulate(sizes[:-1]))
        self.distributions = distributions

    def _log_prob(self, x, condition = None):
        xs = jnp.array_split(x, self.split_idxs)
        log_probs = [
            d._log_prob(x)
            for d, x in zip(self.distributions, xs, strict = True)
        ]
        return sum(log_probs)

    def _sample(self, key, condition = None):
        keys = jax.random.split(key, len(self.distributions))
        samples = [
            d._sample(k)
            for d, k in zip(self.distributions, keys, strict = True)
        ]
        return jnp.concatenate(samples)        


class BoundedFlow(Transformed):
    bounds: jax.Array
    
    def __init__(self, flow, bounds):
        super().__init__(flow.base_dist, flow.bijection)
        _bounds = []
        for bound in bounds:
            left = -jnp.inf if bound[0] is None else bound[0]
            right = jnp.inf if bound[1] is None else bound[1]
            _bounds.append([left, right])
        self.bounds = jnp.array(_bounds)

    def _log_prob(self, x, condition = None):
        below = x < self.bounds[:, 0]
        above = x > self.bounds[:, 1]
        pred = jnp.any(jnp.logical_or(below, above))
        true_fn = lambda: -INF
        false_fn = lambda: super(Transformed, self)._log_prob(x, condition)
        return jax.lax.cond(pred, true_fn, false_fn)

class BoundedFlowTrainable(Transformed):
    def __init__(self, flow):
        super().__init__(flow.base_dist, flow.bijection)

    def _log_prob(self, x, condition = None):
        lo = self.bijection[-1].loc
        hi = lo + self.bijection[-1].scale
        pred = jnp.any(jnp.logical_or(x < lo, x > hi))
        true_fn = lambda: -INF
        false_fn = lambda: super(Transformed, self)._log_prob(x, condition)
        return jax.lax.cond(pred, true_fn, false_fn)        


def filter_default(flow, filter_spec = equinox.is_inexact_array):
    return jax.tree_util.tree_map(filter_spec, flow)

def filter_independent(
    flow, filter_specs, filter_spec = equinox.is_inexact_array,
):
    filter_spec = filter_default(flow, filter_spec)
    filter_spec = equinox.tree_at(
        lambda tree: tree.distributions, filter_spec, replace = filter_specs,
    )
    return filter_spec
    
def filter_bounds(filter_spec):
    return equinox.tree_at(
        lambda tree: tree.bounds, filter_spec, replace = False,
    )

def filter_base(filter_spec):
    return equinox.tree_at(
        lambda tree: tree.base_dist, filter_spec, replace = False,
    )

def filter_bijection(filter_spec):
    return equinox.tree_at(
        lambda tree: tree.bijection, filter_spec, replace = False,
    )

def filter_tuple(filter_spec):
    return True, filter_spec


def bound_from_unbound(
    flow,
    bounds,
    norms = True,
    exp = True,
    filter_spec = equinox.is_inexact_array,
):
    if norms is True:
        posteriors = load_posteriors()[:, :, :-1]
        nsamples, nobs, ndim = posteriors.shape
        norms = posteriors.reshape(-1, ndim)
    else:
        norms = None

    post = get_pre(bounds = bounds, norms = norms, exp = exp)
    base_dist = flow.base_dist
    bijection = Chain([flow.bijection, post])
    flow = Transformed(base_dist, bijection)
    flow = BoundedFlow(flow, bounds)

    filter_spec = filter_default(flow, filter_spec)
    filter_spec = filter_bounds(filter_spec)
    filter_spec = filter_base(filter_spec)

    filter_spec = equinox.tree_at(
        lambda tree: tree.bijection[-1], filter_spec, replace = False,
    )

    return flow, filter_spec

def bound_from_bound(
    flow,
    bounds = None,
    trainable = False,
    filter_spec = equinox.is_inexact_array,
):
    ndim = flow.shape[0]
    if bounds is None:
        bounds = jnp.array([jnp.zeros(ndim), jnp.ones(ndim)]).T
    assert len(bounds) == ndim

    lo = flow.base_dist.bijection.transform(jnp.zeros(ndim))
    hi = flow.base_dist.bijection.transform(jnp.ones(ndim))
    to_unit = Invert(Affine(loc = lo, scale = hi - lo))

    bounds = jnp.asarray(bounds)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    from_unit = Affine(loc = lo, scale = hi - lo)

    base_dist = flow.base_dist
    bijection = Chain([flow.bijection, to_unit, from_unit])
    flow = Transformed(base_dist, bijection)

    if trainable:
        flow = BoundedFlowTrainable(flow)
        filter_spec = filter_default(flow, filter_spec)
    else:
        flow = BoundedFlow(flow, bounds)
        filter_spec = filter_default(flow, filter_spec)
        filter_spec = equinox.tree_at(
            lambda tree: tree.bijection[-1], filter_spec, replace = False,
        )
        filter_spec = filter_bounds(filter_spec)

    filter_spec = filter_base(filter_spec)
    filter_spec = equinox.tree_at(
        lambda tree: tree.bijection[-2], filter_spec, replace = False,
    )
    
    return flow, filter_spec


def nll(flow, x, c=None):
    return -flow.log_prob(x, condition=c)


def ce(flow, x, c=None):
    return nll(flow, x, c=c).mean()


def trainer(
    key,
    flow,
    train,
    valid = None,
    batch_size = None,
    all_batches = True,
    epochs = 1,
    patience = None,
    stop_if_inf = True,
    lr  =1e-3,
    wd = None,
    opt = None,
    loss_fn = None,
    print_batch = False,
    print_epoch = True,
    filter_spec = equinox.is_inexact_array,
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
    loss_and_grad = # jax.value_and_grad(loss_batch)
    loss_and_grad = equinox.filter_value_and_grad(loss_batch)

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
            loss[-1] < best_loss, # -1 here refers to tloss or vloss
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
        # cond_patience,
        equinox.filter_jit(cond_patience),
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
