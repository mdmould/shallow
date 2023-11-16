from functools import partial
import jax
import jax.numpy as jnp
import jax_tqdm
import equinox
import optax


def mse(model, x, y):
    return jnp.square(y - model(x)).mean()


def mae(model, x, y):
    return jnp.abs(y - model(x)).mean()


## TODO: make y any dimension
def trainer(
    key,
    model,
    train,
    valid=None,
    batch_size=10,
    max_epochs=100,
    patience=10,
    lr=1e-3,
    wd=0,
    loss_fn=None,
    print_epoch=True,
    ):

    params, static = equinox.partition(model, equinox.is_inexact_array)

    xt, yt = train
    nt = xt.shape[0]
    if batch_size is None:
        batch_size = nt
    nbt, remt = divmod(nt, batch_size)

    opt = optax.adamw(learning_rate=lr, weight_decay=wd)
    state = opt.init(params)

    if loss_fn is None:
        loss_fn = mse
    def batched_loss(params, x, y):
        model = equinox.combine(params, static)
        vmap_loss = jax.vmap(partial(loss_fn, model))
        return vmap_loss(x, y).mean()
    loss_and_grad = jax.value_and_grad(batched_loss)

    def cond_loss(current, best):
        epoch, loss, params = current
        best_epoch, best_loss, best_params = best
        pred = loss < best_loss
        true_fn = lambda: current
        false_fn = lambda: best
        return jax.lax.cond(pred, true_fn, false_fn)

    def train_step(carry, batch):
        params, state = carry
        x, y = batch
        loss, grad = loss_and_grad(params, x, y)
        updates, state = opt.update(grad, state, params)
        params = equinox.apply_updates(params, updates)
        return (params, state), loss

    if nt == batch_size:
        
        def train_scan(params, state, x, y):
            carry, losses = train_step((params, state), (x, y))
            return *carry, losses
            
    elif remt == 0:
        
        def train_scan(params, state, x, y):
            xs = x.reshape(nbt, batch_size, *x.shape[1:])
            ys = y.reshape(nbt, batch_size, *y.shape[1:])
            carry, losses = jax.lax.scan(train_step, (params, state), (xs, ys))
            return *carry, losses

    else:
        
        def train_scan(params, state, x, y):
            xscan = x[:-remt].reshape(nbt, batch_size, *x.shape[1:])
            yscan = y[:-remt].reshape(nbt, batch_size, *y.shape[1:])
            carry, losses = jax.lax.scan(
                train_step, (params, state), (xscan, yscan),
                )
            xleft = x[-remt:]
            yleft = y[-remt:]
            carry, loss = train_step(carry, (xleft, yleft))
            losses = jnp.concatenate([losses, jnp.array([loss])])
            return *carry, losses

    if valid is None:

        def epoch_step(carry, epoch):
            key, params, state, best = carry
            key, tkey = jax.random.split(key)
            shuffle = jax.random.permutation(tkey, nt)
            params, state, losses = train_scan(
                params, state, xt[shuffle], yt[shuffle],
                )
            loss = losses.mean()
            best = cond_loss((epoch, loss, params), best)
            return (key, params, state, best), (loss,)

        nanloss = jnp.nan,

    else:

        xv, yv = valid
        nv = xv.shape[0]
        vbatch_size = batch_size
        nbv, remv = divmod(nv, vbatch_size)

        if nv == vbatch_size:

            def valid_scan(params, x, y):
                return batched_loss(params, x, y)

        elif remv == 0:

            def valid_scan(params, x, y):
                xs = x.reshape(nbv, vbatch_size, *x.shape[1:])
                ys = y.reshape(nbv, vbatch_size, *y.shape[1:])
                return jax.vmap(partial(batched_loss, params))(xs, ys)
                # return jax.lax.scan(
                #     lambda carry, xy: (carry, loss_fn(params, *xy)),
                #     None,
                #     (xs, ys),
                #     )[1]

        else:
            def valid_scan(params, x, y):
                xscan = x[:-remv].reshape(nbv, batch_size, *x.shape[1:])
                yscan = y[:-remv].reshape(nbv, batch_size, *y.shape[1:])
                losses = jax.vmap(partial(batched_loss, params))(xscan, yscan)
                # losses = jax.lax.scan(
                #     lambda carry, xy: (carry, loss_fn(params, *xy)),
                #     None,
                #     (xscan, yscan),
                #     )[1]
                xleft = x[remv:]
                yleft = y[remv:]
                loss = batched_loss(params, xleft, yleft)
                losses = jnp.concatenate([losses, jnp.array([loss])])
                return losses

        def epoch_step(carry, epoch):
            key, params, state, best = carry
            key, tkey, vkey = jax.random.split(key, 3)
            shuffle = jax.random.permutation(tkey, nt)
            params, state, losses = train_scan(
                params, state, xt[shuffle], yt[shuffle],
                )
            tloss = losses.mean()
            shuffle = jax.random.permutation(vkey, nv)
            vloss = valid_scan(params, xv[shuffle], yv[shuffle]).mean()
            best = cond_loss((epoch, vloss, params), best)
            return (key, params, state, best), (tloss, vloss)

        nanloss = jnp.nan, jnp.nan

    def cond_patience(carry, epoch):
        key, params, state, best = carry
        best_epoch, best_loss, best_params = best
        pred = epoch > best_epoch + patience
        true_fn = lambda carry, epoch: (carry, nanloss)
        false_fn = epoch_step
        return jax.lax.cond(pred, true_fn, false_fn, carry, epoch)

    if print_epoch:
        pbar = jax_tqdm.scan_tqdm(max_epochs, print_rate=1, message='epoch')
        epoch_step = pbar(epoch_step)
    epoch_scan = epoch_step if patience is None else cond_patience

    best = 0, jnp.inf, params
    init = key, params, state, best
    epochs = jnp.arange(max_epochs)
    
    carry, losses = jax.lax.scan(epoch_scan, init, epochs)
    key, params, state, best = carry
    best_epoch, best_loss, best_params = best

    model = equinox.combine(best_params, static)
    
    losses = {label: loss for label, loss in zip(('train', 'valid'), losses)}
    if patience is not None:
        for label in losses:
            losses[label] = losses[label][:best_epoch+patience+1]

    return model, losses
