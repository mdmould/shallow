import jax
import jax.numpy as jnp

from flowjax.distributions import AbstractDistribution

from collections.abc import Sequence
from itertools import accumulate


INF = jnp.nan_to_num(jnp.inf).item()


class IndependentDistribution(AbstractDistribution):
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


class ParentDistribution(AbstractDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: tuple[int, ...] | None = None
    _distribution: AbstractDistribution

    def __init__(self, distribution):
        self.shape = distribution.shape
        self.cond_shape = distribution.cond_shape
        self._distribution = distribution
            
    def __getattr__(self, attr):
        return getattr(self._distribution, attr)
        
    def _log_prob(self, x, condition = None):
        return self._distribution._log_prob(x, condition)
        
    def _sample(self, key, condition = None):
        return self._distribution._sample(key, condition)


class BoundedDistribution(ParentDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: tuple[int, ...] | None = None
    _distribution: AbstractDistribution
    bounds: jax.Array
    _inf: float

    def __init__(self, distribution, bounds, inf = False):
        super().__init__(distribution)
        self._inf = jnp.inf if inf else INF

        _bounds = []
        for bound in bounds:
            if bound is None:
                left, right = -jnp.inf, jnp.inf
            else:
                left = -jnp.inf if bound[0] is None else bound[0]
                right = jnp.inf if bound[1] is None else bound[1]
            _bounds.append([left, right])
        self.bounds = jnp.array(_bounds)

    def _log_prob(self, x, condition = None):
        below = x < self.bounds[:, 0]
        above = x > self.bounds[:, 1]
        return jax.lax.cond(
            jnp.any(jnp.logical_or(below, above)),
            lambda: - self._inf,
            lambda: self._distribution._log_prob(x, condition),
        )


class NamedDistribution(ParentDistribution):
    shape: tuple[int, ...] = ()
    cond_shape: tuple[int, ...] | None = None
    distribution: AbstractDistribution
    names: tuple[str, ...]
    cond_names: tuple[str, ...] | None = None
    
    def __init__(self, distribution, names, cond_names = None):
        assert len(distribution.shape) == 1
        assert len(names) == distribution.shape[0]
        if distribution.cond_shape is None:
            assert cond_names is None
        if cond_names is not None:
            assert distribution.cond_shape is not None
            assert len(distribution.cond_shape) == 1
            assert len(cond_names) == distribution.cond_shape[0]
        
        super().__init__(distribution)

        self.names = names
        self.cond_names = cond_names

    def log_prob(self, x, condition = None):
        x = jnp.array(list(x.values()))
        x = jnp.moveaxis(x, 0, -1)
        if self.cond_shape is not None:
            condition = jnp.array(list(condition.values()))
            condition = jnp.moveaxis(condition, 0, -1)
        return self.distribution.log_prob(x, condition)

    def sample(self, key, sample_shape = (), condition = None):
        if self.cond_shape is not None:
            condition = jnp.array(list(condition.values()))
            condition = jnp.moveaxis(condition, 0, -1)
        x = self.distribution.sample(key, sample_shape, condition)
        x = jnp.moveaxis(x, -1, 0)
        return dict(zip(self.names, x))
