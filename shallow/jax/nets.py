from functools import partial
import jax
import jax.numpy as jnp
import jax_tqdm
import equinox
import optax

from .train import _trainer


def mse(model, x, y):
    return jnp.mean(jnp.square(y - model(x)))


def mae(model, x, y):
    return jnp.mean(jnp.abs(y - model(x)))


def ce(model, x, y):
    return -jnp.sum(y * jnp.log(model(x)))


def bce(model, x, y):
    q = model(x)
    return -(y * jnp.log(q) + (1 - y) * jnp.log(1 - q))


def kl(model, x, y):
    return jnp.sum(y * (jnp.log(y) - jnp.log(model(x))))


def bkl(model, x, y):
    q = model(x)
    return (
        + y * (jnp.log(y) - jnp.log(q))
        + (1 - y) * (jnp.log(1 - y) - jnp.log(1 - q))
        )


def js(model, x, y):
    q = model(x)
    m = 0.5 * (y + q)
    return 0.5 * jnp.sum(
        + p * jnp.log(p)
        + q * jnp.log(q)
        - 2 * m * jnp.log(m)
        )


def bjs(model, x, y):
    q = model(x)
    m = 0.5 * (y + q)
    return 0.5 * (
        + y * (jnp.log(y) - jnp.log(m))
        + (1 - y) * (jnp.log(1 - y) - jnp.log(1 - m))
        + q * (jnp.log(q) - jnp.log(m))
        + (1 - q) * (jnp.log(1 - q) - jnp.log(1 - m))
        )


def trainer(
    key,
    model,
    train,
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

    params, static = equinox.partition(model, filter_spec)
    opt = optax.adamw(learning_rate=lr, weight_decay=wd)
    state = opt.init(params)

    if loss_fn is None:
        loss_fn = mse
    def loss_vmap(params, x, y):
        model = equinox.combine(params, static)
        loss_model = partial(loss_fn, model)
        return jax.vmap(loss_model)(x, y)
    loss_batch = lambda params, x, y: loss_vmap(params, x, y).mean()
    loss_and_grad = jax.value_and_grad(loss_batch)

    xt, yt = train
    assert xt.shape[0] == yt.shape[0]
    if valid is not None:
        if type(valid) is tuple:
            xv, yv = valid
            assert xv.shape[0] == yv.shape[0]
        elif type(valid) is float:
            assert 0 < valid < 1
            nv = max(int(valid * xt.shape[0]), 1)
            key, key_ = jax.random.split(key)
            shuffle = jax.random.permutation(key_, x.shape[0])
            xv = xt[shuffle][:nv]
            yv = yt[shuffle][:nv]
            xt = xt[shuffle][nv:]
            yt = yt[shuffle][nv:]
            
    nt = xt.shape[0]
    if batch_size is None or batch_size > nt:
        batch_size = nt
    nbt, remt = divmod(nt, batch_size)

    def train_step(carry, batch):
        params, state = carry
        loss, grad = loss_and_grad(params, *batch)
        updates, state = opt.update(grad, state, params)
        params = equinox.apply_updates(params, updates)
        return (params, state), loss

    def train_batch(carry, ibatch):
        i, batch = ibatch        
        return train_step(carry, batch)

    def valid_step(params, batch):
        return None, loss_batch(params, *batch)

    def valid_batch(params, ibatch):
        i, batch = ibatch
        return valid_step(params, batch)

    if all_batches:

        if nbt == 1:
            def train_scan(params, state, x, y):
                carry, losses = train_step((params, state), (x, y))
                return *carry, losses

        elif remt == 0:
            def train_scan(params, state, x, y):
                xs = x.reshape(nbt, batch_size, *x.shape[1:])
                ys = y.reshape(nbt, batch_size, *y.shape[1:])
                carry, losses = jax.lax.scan(
                    train_batch, (params, state), (jnp.arange(nbt), (xs, ys)),
                    )
                return *carry, losses

        else:
            def train_scan(params, state, x, y):
                xs = x[:-remt].reshape(nbt, batch_size, *x.shape[1:])
                ys = y[:-remt].reshape(nbt, batch_size, *y.shape[1:])
                carry, losses = jax.lax.scan(
                    train_batch, (params, state), (jnp.arange(nbt), (xy, ys)),
                    )
                carry, loss = train_step(carry, (x[-remt:], y[-remt:]))
                losses = jnp.concatenate([losses, jnp.array([loss])])
                return *carry, losses

        if valid is None:
            def epoch_step(key, params, state):
                key, tkey = jax.random.split(key)
                shuffle = jax.random.permutation(tkey, nt)
                params, state, losses = train_scan(
                    params, state, xt[shuffle], yt[shuffle],
                    )
                loss = losses.mean()
                return key, params, state, (loss,)

        else:
            nv = xv.shape[0]
            vbatch_size = batch_size
            if vbatch_size > nv:
                vbatch_size = nv
            nbv, remv = divmod(nv, vbatch_size)

            if nbv == 1:
                def valid_scan(params, x, y):
                    return valid_step(params, x, y)[1]

            elif remv == 0:
                def valid_scan(params, x, y):
                    xs = x.reshape(nbv, vbatch_size, *x.shape[1:])
                    ys = y.reshape(nbv, vbatch_size, *y.shape[1:])
                    return jax.lax.scan(
                        valid_batch, params, (jnp.arange(nbv), (xs, ys)),
                        )[1]

            else:
                def valid_scan(params, x, y):
                    xs = x[:-remv].reshape(nbv, vbatch_size, *x.shape[1:])
                    ys = y[:-remv].reshape(nbv, vbatch_size, *y.shape[1:])
                    losses = jax.lax.scan(
                        valid_batch, params, (jnp.arange(nbv), (xs, ys)),
                        )[1]
                    loss = valid_step(params, x[-remv:], y[-remv:])
                    return jnp.concatenate([losses, jnp.array([loss])])

            def epoch_step(key, params, state):
                key, tkey, vkey = jax.random.split(key, 3)
                shuffle = jax.random.permutation(tkey, nt)
                params, state, losses = train_scan(
                    params, state, xt[shuffle], yt[shuffle],
                    )
                tloss = losses.mean()
                shuffle = jax.random.permutation(vkey, nv)
                vloss = valid_scan(params, xv[shuffle], yv[shuffle]).mean()
                return key, params, state, (tloss, vloss)

    else:
        print_batch = False

        if valid is None:
            def epoch_step(key, params, state):
                key, tkey = jax.random.split(key)
                idxs = jax.random.choice(tkey, nt, shape=(batch_size,))
                (params, state), loss = train_step(
                    (params, state), (xt[idxs], yt[idxs]),
                    )
                return key, params, state, (loss,)

        else:
            def epoch_step(key, params, state):
                key, tkey, vkey = jax.random.split(key, 3)
                idxs = jax.random.choice(tkey, nt, shape=(batch_size,))
                (params, state), loss = train_step(
                    (params, state), (xt[idxs], yt[idxs]),
                    )
                idxs = jax.random.choice(vkey, nv, shape=(vbatch-size,))
                vloss = valid_step(params, (xv[idxs], yv[idxs]))[1]
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

    model = equinox.combine(best_params, static)

    if patience is not None or stop_if_inf:
        losses = jnp.array(losses)
        if jnp.any(~jnp.isfinite(losses)):
            cut = jnp.argwhere(~jnp.isfinite(losses))[:, 1].min()
            if patience is not None:
                if (
                    cut == best_epoch + patience + 1 and
                    jnp.isnan(losses[:, cut]).all()
                    ):
                    print('Stopped: patience reached')
                else:
                    print('Stopped: loss is not finite')
            else:
                print('Stopped: loss is not finite')
            losses = losses[:, :cut]
    losses = {k: v for k, v in zip(['train', 'valid'], losses)}
    
    return model, losses


# def trainer(
#     key,
#     model,
#     train,
#     valid=None,
#     batch_size=10,
#     max_epochs=100,
#     patience=10,
#     lr=1e-3,
#     wd=0,
#     loss_fn=None,
#     print_batch=False,
#     print_epoch=True,
#     filter_spec=equinox.is_inexact_array,
#     ):

#     params, static = equinox.partition(model, filter_spec)
#     opt = optax.adamw(learning_rate=lr, weight_decay=wd)
#     state = opt.init(params)

#     xt, yt = train
#     nt = xt.shape[0]
#     if batch_size is None:
#         batch_size = nt
#         print_batch = False
#     nbt, remt = divmod(nt, batch_size)

#     if loss_fn is None:
#         loss_fn = mse
#     def loss_vmap(params, x, y):
#         model = equinox.combine(params, static)
#         loss_model = partial(loss_fn, model)
#         return jax.vmap(loss_model)(x, y)
#     loss_batch = lambda params, x, y: loss_vmap(params, x, y).mean()
#     loss_and_grad = jax.value_and_grad(loss_batch)

#     def cond_loss(current, best):
#         epoch, loss, params = current
#         best_epoch, best_loss, best_params = best
#         pred = loss < best_loss
#         true_fn = lambda: current
#         false_fn = lambda: best
#         return jax.lax.cond(pred, true_fn, false_fn)

#     def train_step(carry, batch):
#         params, state = carry
#         x, y = batch
#         loss, grad = loss_and_grad(params, x, y)
#         updates, state = opt.update(grad, state, params)
#         params = equinox.apply_updates(params, updates)
#         return (params, state), loss

#     if nt == batch_size:
        
#         def train_scan(params, state, x, y):
#             carry, losses = train_step((params, state), (x, y))
#             return *carry, losses
            
#     elif remt == 0:
        
#         def train_scan(params, state, x, y):
#             xs = x.reshape(nbt, batch_size, *x.shape[1:])
#             ys = y.reshape(nbt, batch_size, *y.shape[1:])
#             carry, losses = jax.lax.scan(train_step, (params, state), (xs, ys))
#             return *carry, losses

#     else:
        
#         def train_scan(params, state, x, y):
#             xscan = x[:-remt].reshape(nbt, batch_size, *x.shape[1:])
#             yscan = y[:-remt].reshape(nbt, batch_size, *y.shape[1:])
#             carry, losses = jax.lax.scan(
#                 train_step, (params, state), (xscan, yscan),
#                 )
#             xleft = x[-remt:]
#             yleft = y[-remt:]
#             carry, loss = train_step(carry, (xleft, yleft))
#             losses = jnp.concatenate([losses, jnp.array([loss])])
#             return *carry, losses

#     if valid is None:

#         def epoch_step(carry, epoch):
#             key, params, state, best = carry
#             key, tkey = jax.random.split(key)
#             shuffle = jax.random.permutation(tkey, nt)
#             params, state, losses = train_scan(
#                 params, state, xt[shuffle], yt[shuffle],
#                 )
#             loss = losses.mean()
#             best = cond_loss((epoch, loss, params), best)
#             return (key, params, state, best), (loss,)

#         nanloss = jnp.nan,

#     else:

#         xv, yv = valid
#         nv = xv.shape[0]
#         vbatch_size = batch_size
#         if vbatch_size > nv:
#             vbatch_size = nv
#         nbv, remv = divmod(nv, vbatch_size)

#         if nv == vbatch_size:

#             def valid_scan(params, x, y):
#                 return loss_batch(params, x, y)

#         elif remv == 0:

#             def valid_scan(params, x, y):
#                 # xs = x.reshape(nbv, vbatch_size, *x.shape[1:])
#                 # ys = y.reshape(nbv, vbatch_size, *y.shape[1:])
#                 # return jax.vmap(partial(loss_batch, params))(xs, ys)
#                 # return jax.lax.scan(
#                 #     lambda carry, xy: (carry, loss_fn(params, *xy)),
#                 #     None,
#                 #     (xs, ys),
#                 #     )[1]
#                 losses = loss_vmap(params, x, y)
#                 losses = losses.reshape(nbv, vbatch_size)
#                 losses = losses.mean(axis=1)
#                 return losses

#         else:
#             def valid_scan(params, x, y):
#                 # xscan = x[:-remv].reshape(nbv, vbatch_size, *x.shape[1:])
#                 # yscan = y[:-remv].reshape(nbv, vbatch_size, *y.shape[1:])
#                 # losses = jax.vmap(partial(loss_batch, params))(xscan, yscan)
#                 # # losses = jax.lax.scan(
#                 # #     lambda carry, xy: (carry, loss_fn(params, *xy)),
#                 # #     None,
#                 # #     (xscan, yscan),
#                 # #     )[1]
#                 # xleft = x[-remv:]
#                 # yleft = y[-remv:]
#                 # loss = loss_batch(params, xleft, yleft)
#                 losses = loss_vmap(params, x, y)
#                 loss = losses[-remv:].mean()
#                 losses = losses[:-remv].reshape(nbv, vbatch_size)
#                 losses = losses.mean(axis=1)
#                 losses = jnp.concatenate([losses, jnp.array([loss])])
#                 return losses

#         def epoch_step(carry, epoch):
#             key, params, state, best = carry
#             key, tkey, vkey = jax.random.split(key, 3)
#             shuffle = jax.random.permutation(tkey, nt)
#             params, state, losses = train_scan(
#                 params, state, xt[shuffle], yt[shuffle],
#                 )
#             tloss = losses.mean()
#             shuffle = jax.random.permutation(vkey, nv)            
#             vloss = valid_scan(params, xv[shuffle], yv[shuffle]).mean()
#             best = cond_loss((epoch, vloss, params), best)
#             return (key, params, state, best), (tloss, vloss)

#         nanloss = jnp.nan, jnp.nan

#     def cond_patience(carry, epoch):
#         key, params, state, best = carry
#         best_epoch, best_loss, best_params = best
#         pred = epoch > best_epoch + patience
#         true_fn = lambda carry, epoch: (carry, nanloss)
#         false_fn = epoch_step
#         return jax.lax.cond(pred, true_fn, false_fn, carry, epoch)

#     if print_epoch:
#         pbar = jax_tqdm.scan_tqdm(max_epochs, print_rate=1, message='epoch')
#         epoch_step = pbar(epoch_step)
#     epoch_scan = epoch_step if patience is None else cond_patience

#     best = 0, jnp.inf, params
#     init = key, params, state, best
#     epochs = jnp.arange(max_epochs)
#     carry, losses = jax.lax.scan(epoch_scan, init, epochs)
#     key, params, state, best = carry
#     best_epoch, best_loss, best_params = best

#     model = equinox.combine(best_params, static)
#     losses = {label: loss for label, loss in zip(('train', 'valid'), losses)}
#     if patience is not None:
#         for label in losses:
#             losses[label] = losses[label][:best_epoch+patience+1]

#     return model, losses
