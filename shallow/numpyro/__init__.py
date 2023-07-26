import jax
import jax.numpy as jnp
import optax
import numpyro
from tqdm import tqdm
from copy import deepcopy


## TODO: add params as direct input everywhere


class Transform:

    def __init__(self, params=None):

        self.params = params

    def _forward(self, x):

        y = self.forward(x)
        ladj = self.log_abs_det_jac(x, y)

        return y, ladj

    def _inverse(self, y):

        x = self.inverse(y)
        ladj = self.log_abs_det_jac(x, y)

        return x, -ladj

    def forward(self, x):

        raise NotImplementedError

    def inverse(self, y):

        raise NotImplementedError

    def log_abs_det_jac(self, x, y):

        raise NotImplementedError

    def get_params(self):

        return self.params_list_to_dict(self.params)

    def update_params(self, params):

        if type(params) is dict:
            params = self.params_dict_to_list(params)
        self.params = params

    def params_list_to_dict(self, params):

        return {
            jax.tree_util.keystr(loc): val
            for loc, val in jax.tree_util.tree_leaves_with_path(params)
            }

    def params_dict_to_list(self, params):

        unflatten = jax.tree_util.tree_flatten(self.params)[1]

        return jax.tree_util.tree_unflatten(
            unflatten, list(params.values()),
            )


class Inverse(Transform):

    def __init__(self, transform):

        self.transform = transform
        super().__init__(transform.params)

    def _forward(self, x):

        return self.transform._inverse(x)

    def _inverse(self, y):

        return self.transform._forward(y)

    def forward(self, x):

        return self.transform.inverse(x)

    def inverse(self, y):

        return self.transform.forward(y)

    def log_abs_det_jac(self, x, y):

        return -self.transform.log_abs_det_jac(y, x)


class Compose(Transform):

    def __init__(self, transforms):

        self.transforms = list(transforms)
        params = [
            transform.params for transform in self.transforms
            # if transform.params is not None
            ]
        super().__init__(params)

    def _forward(self, x, params):

        y = x
        ladj = jnp.zeros_like(y.shape[0])
        # for transform in self.transforms:
        #     y, ladj_ = transform._forward(y)
        for transform, params_ in zip(self.transforms, params):
            args = (y,) if params_ is None else (y, params_)
            y, ladj_ = transform._forward(*args)
            
            ladj += ladj_

        return y, ladj

    def _inverse(self, y):

        x = y
        ladj = jnp.zeros_like(x.shape[0])
        for transform in reversed(self.transforms):
            x, ladj_ = transform._inverse(x)
            ladj += ladj_

        return x, ladj

    def forward(self, x):

        y = x
        for transform in self.transforms:
            y = transform.forward(y)

        return y

    def inverse(self, y):

        x = y
        for transform in reversed(self.transforms):
            x = transform.inverse(x)

        return x

    def log_abs_det_jac(self, x, y):

        return self._forward(x)[1]


class Stack(Transform):

    def __init__(self, transforms, dims=None):

        if dims is None:
            dims = list(range(len(transforms)))
        for i in range(len(transforms)):
            dims[i] = jnp.atleast_1d(dims[i])
            assert dims[i].ndim == 1
        self.dims = dims
        params = [tranform.params for transform in transforms]
        super().__init__(params)

    def _forward(self, x):

        y = jnp.empty_like(x)
        ladj = jnp.zeros_like(x.shape[0])
        for dim, transform in zip(self.dims, self.transforms):
            y[..., dim], ladj_ = transform._forward(x[..., dim])
            ladj += ladj_

        return y, ladj

    def _inverse(self, y):

        x = jnp.empty_like(y)
        ladj = jnp.zeros_like(y.shape[0])
        for dim, transform in zip(self.dims, self.transforms):
            x[..., dim], ladj_ = transform._inverse(y[..., dim])
            ladj += ladj_

        return x, ladj_

    def forward(self, x):

        y = jnp.empty_like(x)
        for dim, transform in zip(self.dims, self.transforms):
            y[..., dim] = transform.forward(x[..., dim])

        return y

    def inverse(self, y):

        x = jnp.empty_like(y)
        for dim, transform in zip(self.dims, self.transforms):
            x[..., dim] = transform.inverse(y[..., dim])

        return x

    def log_abs_det_jac(self, x, y):

        ladj = jnp.zeros_like(x.shape[0])
        for dim, transform in zip(self.dims, self.transforms):
            ladj += transform.log_abs_det_jac(x[..., dim], y[..., dim])

        return ladj


class Permute(Transform):

    def __init__(self, permutation):

        self.permutation = permutation
        super().__init__()

    def forward(self, x):

        return x[..., self.permutation]

    def inverse(self, y):

        return self.forward(y)

    def log_abs_det_jac(self, x, y):

        return jnp.zeros_like(x.shape[0])


class Identity(Transform):

    def forward(self, x):

        return x

    def inverse(self, y):

        return self.forward(y)

    def log_abs_det_jac(self, x, y):

        return jnp.zeros_like(x.shape[0])


class Affine(Transform):

    def __init__(self, scale, shift):

        super().__init__()
        self.shift = shift
        self.scale = scale
        self.ladj = jnp.sum(jnp.log(jnp.abs(self.scale)))

    def forward(self, x):

        return x * self.scale + self.shift

    def inverse(self, y):

        return (y - self.shift) / self.scale

    def log_abs_det_jac(self, x, y):

        return jnp.full_like(x.shape[0], self.ladj)


class Exp(Transform):

    def forward(self, x):

        return jnp.exp(x)

    def inverse(self, y):

        return jnp.log(y)

    def log_abs_det_jac(self, x, y):

        return jnp.sum(x, axis=-1)


class Tanh(Transform):

    def forward(self, x):

        return jnp.tanh(x)

    def inverse(self, y):

        return jnp.arctanh(y)

    def log_abs_det_jac(self, x, y):

        return jnp.sum(
            2 * (math.log(2) - x - jax.nn.softplus(-2 * x)),
            axis=-1,
            )


class Sigmoid(Transform):

    def forward(self, x):

        return jax.nn.sigmoid(x)

    def inverse(self, y):

        return jnp.log(y) - jnp.log1p(-y)

    def log_abs_det_jac(self, x, y):

        return jnp.sum(
            -jax.nn.softplus(x) - jax.nn.softplus(-x),
            axis=-1,
            )


class BARN(Transform):

    def __init__(self, dims, factors, residual=None, rng=0):

        init_fn, self.apply_fn = numpyro.nn.BlockNeuralAutoregressiveNN(
            input_dim=dims, hidden_factors=factors, residual=residual,
            )
        if type(rng) is int:
            rng = jax.random.PRNGKey(rng)
        rng, rng_ = jax.random.split(rng)
        params = init_fn(rng_, (dims,))[1]
        super().__init__(params)

    def _forward(self, x, params):

        # y, ladj = self.apply_fn(self.params, x)
        y, ladj = self.apply_fn(params, x)
        ladj = jnp.sum(ladj, axis=-1)

        return y, ladj

    def forward(self, x):

        return self._forward(x)[0]

    def log_abs_det_jac(self, x):

        return self._forward(x)[1]


class Flow(Compose):

    def __init__(self, dims, transforms, bounds=None):

        self.base = numpyro.distributions.Normal(
            loc=jnp.zeros(dims), scale=jnp.ones(dims),
            ).to_event(1)

        if bounds is not None:
            assert len(bounds) == dims
            transforms = self._get_bounds_transforms(bounds) + transforms

        super().__init__(transforms)

    def _get_bounds_transform(self, bounds):

        transforms = []
        for bound in bounds:
            
            if (bound is None) or all (b is None for b in bound):
                transform = Identity()
                
            elif any(b is None for b in bound):
                if bound[0] is None:
                    shift = bound[1]
                    scale = -1
                elif bound[1] is None:
                    shift = bound[1]
                    scale = 1
                transform = Compose([
                    Inverse(Affine(scale, shift)),
                    Inverse(Exp()),
                    ])
                
            else:
                shift = min(bound)
                scale = max(bound) - min(bound)
                transform = Compose([
                    Inverse(Affine(scale, shift)),
                    Inverse(Sigmoid()),
                    ])
                
            transforms.append(transform)

        return Stack(transforms)

    def log_prob(self, x, params=None):

        if params is None:
            params = self.params
        y, ladj = self._forward(x, params)
        lp = self.base.log_prob(y) + ladj

        return lp

    def prob(self, x, params=None):

        return jnp.exp(self.log_prob(x, params))

    def sample(self, rng, n=1):

        if rng is int:
            rng = jax.random.PRNGKey(rng)
        y = self.base.sample(rng, (n,))
        x = self.inverse(y)

        return x


class BARF(Flow):

    def __init__(
        self,
        dims,
        flows,
        factors,
        residual=None,
        permute=None,
        bounds=None,
        rng=0,
        ):

        if type(rng) is int:
            rng = jax.random.PRNGKey(rng)

        transforms = []
        for i in range(flows):
            
            if (permute is not None) and (i > 0):
                if permute == 'reverse':
                    permutation = list(reversed(range(dims)))
                elif permute == 'random':
                    rng, rng_ = jax.random.split(rng)
                    permutation = jax.random.permutation(rng_, dims)
                transforms.append(Permute(permutation))

            residual = 'gated' if i < flows - 1 else None
            transforms.append(BARN(dims, factors, residual, rng))

        super().__init__(dims, transforms, bounds)


def trainer(data, flow, lr, steps):
    
    loss_fn = lambda params: -flow.log_prob(data, params).sum()
    loss_and_grad = jax.value_and_grad(loss_fn)
    
    params = flow.params
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def update(params, state):

        loss, grads = loss_and_grad(params)
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)

        return params, state, loss    

    losses = []
    for step in tqdm(range(steps)):
        params, state, loss = update(params, state)
        losses.append(loss)
        if loss == min(losses):
            best_params = deepcopy(params)

    flow.update_params(params)

    return flow, losses

