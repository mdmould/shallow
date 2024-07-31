import jax
import jax.numpy as jnp
import equinox
import interpax

from flowjax.bijections import (
    AbstractBijection,
    Affine as AffinePositiveScale,
    Chain,
    Exp,
    Identity,
    Invert,
    RationalQuadraticSpline,
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


# def get_post_stack1d(bounds = None, norms = None):
#     if bounds is None and norms is None:
#         return Identity()
#     elif bounds is not None and norms is None:
#         return Stack(list(map(get_bounder, bounds)))
#     elif bounds is None and norms is not None:
#         return Stack(list(map(lambda x: Invert(get_normer(x)), norms.T)))
#     else:
#         pres = []
#         for bound, norm in zip(bounds, norms.T):
#             bounder = get_bounder(bound)
#             debounded_norm = jax.vmap(bounder.inverse)(norm)
#             denormer = Invert(get_normer(debounded_norm))
#             pres.append(Chain([denormer, bounder]))
#         return Stack(pres)


class StandardNormalCDF(AbstractBijection):
    shape: tuple[int, ...] = ()
    cond_shape: None = None

    def transform(self, x, condition = None):
        return jax.scipy.stats.norm.cdf(x)

    def transform_and_log_det(self, x, condition = None):
        return self.transform(x), jax.scipy.stats.norm.logpdf(x)

    def inverse(self, y, condition = None):
        return jax.scipy.stats.norm.ppf(y)

    def inverse_and_log_det(self, y, condition = None):
        x = self.inverse(y)
        return x, -jax.scipy.stats.norm.logpdf(x)


def UnivariateRationalQuadraticSplineCDF(samples, bounds = None):
    assert len(samples.shape) == 1

    if bounds is None:
        bounds = -jnp.inf, jnp.inf
    else:
        assert len(bounds) == 2
        left = -jnp.inf if bounds[0] is None else bounds[0]
        right = jnp.inf if bounds[1] is None else bounds[1]
        bounds = left, right
    bounds = jnp.unique(jnp.nan_to_num(jnp.array(bounds)))
    assert bounds.shape == (2,)

    affine = Invert(Affine(loc = bounds[0], scale = bounds[1] - bounds[0]))

    points = jnp.unique(jnp.append(samples, bounds))
    points = jax.vmap(affine.transform)(points)
    cdf = jnp.linspace(0, 1, points.size)
    interp = interpax.Interpolator1D(points, cdf, method = 'monotonic')

    rqs = RationalQuadraticSpline(
        knots = points.size,
        interval = (0, 1),
        min_derivative = 0,
        softmax_adjust = 0,
    )
    rqs = equinox.tree_at(lambda tree: tree.x_pos, rqs, interp.x)
    rqs = equinox.tree_at(lambda tree: tree.y_pos, rqs, interp.f)
    rqs = equinox.tree_at(
        lambda tree: tree.derivatives, rqs, interp.derivs['fx'],
    )

    return Chain([affine, rqs])


def RationalQuadraticSplineCDF(samples, bounds = None):
    samples = jnp.asarray(samples)
    assert len(samples.shape) == 2
    bounds = [None] * samples.shape[1] if bounds is None else bounds
    assert len(bounds) == samples.shape[1]
    return Stack(list(map(
        UnivariateRationalQuadraticSplineCDF, samples.T, bounds,
    )))


class UnivariateEmpiricalCDF(AbstractBijection):
    shape: tuple
    cond_shape: None = None
    _interp: interpax.Interpolator1D

    def __init__(self, samples, bounds = None):
        from interpax._coefs import A_CUBIC

        assert len(samples.shape) == 1
        self.shape = ()

        if bounds is None:
            bounds = -jnp.inf, jnp.inf
        else:
            assert len(bounds) == 2
            left = -jnp.inf if bounds[0] is None else bounds[0]
            right = jnp.inf if bounds[1] is None else bounds[1]
            bounds = left, right
        bounds = jnp.unique(jnp.nan_to_num(jnp.array(bounds)))
        assert bounds.shape == (2,)
        
        points = jnp.unique(jnp.append(samples, bounds))
        cdf = jnp.linspace(0, 1, points.size)
        self._interp = interpax.Interpolator1D(points, cdf, method = 'monotonic')

    def transform(self, x, condition = None):
        return self._interp(x)

    def transform_and_log_det(self, x, condition = None):
        return self.transform(x), jnp.log(jnp.abs(self._interp(x, 1)))

    def inverse(self, y, condition = None):
        fq = y
        x = self._interp.x
        f = self._interp.f
        fx = self._interp.derivs['fx']

        i = jnp.searchsorted(f, fq, side = 'right')

        def find_root():
            dx = x[i] - x[i - 1]
            dxi = jnp.where(dx == 0, 0, 1 / dx)

            f0 = f[i - 1]
            f1 = f[i]
            fx0 = fx[i - 1] * dx
            fx1 = fx[i] * dx
        
            F = jnp.stack([f0, f1, fx0, fx1])
            coef = jnp.matmul(A_CUBIC, F)
    
            dt, ct, bt, at = coef
            xi = x[i - 1]
    
            a = at * dxi ** 3
            b = bt * dxi ** 2 - 3 * at * xi * dxi ** 3
            c = ct * dxi - 2 * bt * xi * dxi ** 2 + 3 * at * xi ** 2 * dxi ** 3
            d = dt - ct * xi * dxi + bt * xi ** 2 * dxi ** 2 - at * xi ** 3 * dxi ** 3

            d = d - fq
    
            roots = jnp.roots(jnp.array([a, b, c, d]), strip_zeros = False)
    
            # p = (3 * a * c - b ** 2) / (3 * a ** 2)
            # q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
            # u1 = - q / 2 + jnp.sqrt(q ** 2 / 4 + p ** 3 / 27)
            # u2 = - q / 2 - jnp.sqrt(q ** 2 / 4 + p ** 3 / 27)
            # tq = jnp.cbrt(u1) + jnp.cbrt(u2)
            # xq = tq - b / (3 * a)
    
            # d0 = b ** 2 - 3 * a * c
            # d1 = 2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d
            # cp = jnp.cbrt((d1 + jnp.sqrt(d1 ** 2 - 4 * d0 ** 3)) / 2)
            # cm = jnp.cbrt((d1 - jnp.sqrt(d1 ** 2 - 4 * d0 ** 3)) / 2)
            # C = jnp.where(cp == 0, cm, cp)
            # d0_over_C = jnp.where(C == 0, 0, d0 / C)
            # k = jnp.arange(3)
            # xi = (-1 + 3 ** 0.5 * 1j) / 2
            # roots = -(b + xi ** k * C + d0_over_C / xi ** k) / (3 * a)
    
            roots = jnp.where(roots.imag == 0, roots.real, jnp.inf)
            roots = jnp.where((x[i - 1] < roots) * (roots < x[i]), roots, jnp.inf)
            xq = jnp.min(roots)

            return xq

        return jax.lax.cond(fq == f[i - 1], lambda: x[i - 1], find_root)

    def inverse_and_log_det(self, y, condition = None):
        x = self.inverse(y)
        return x, -jnp.log(jnp.abs(self._interp(x, 1)))


def EmpiricalCDF(samples, bounds = None):
    samples = jnp.asarray(samples)
    assert len(samples.shape) == 2
    bounds = [None] * samples.shape[1] if bounds is None else bounds
    assert len(bounds) == samples.shape[1]
    return Stack(list(map(UnivariateEmpiricalCDF, samples.T, bounds)))


class _EmpiricalCDF(AbstractBijection):
    shape: tuple
    cond_shape: None = None
    _bijections: tuple[UnivariateEmpiricalCDF, ...]

    def __init__(self, samples, bounds):
        samples = jnp.asarray(samples)
        assert len(samples.shape) == 2
        self.shape = (samples.shape[1],)
        bounds = [None] * samples.shape[1] if bounds is None else bounds
        assert len(bounds) == self.shape[0]
        self._bijections = tuple(map(
            UnivariateEmpiricalCDF, samples.T, bounds,
        ))

    def transform(self, x, condition = None):
        single = lambda bijection, x: bijection.transform(x)
        ys = list(map(single, self._bijections, x.T))
        return jnp.stack(ys, axis = -1)

    def transform_and_log_det(self, x, condition = None):
        single = lambda bijection, x: bijection.transform_and_log_det(x)
        ys, log_dets = zip(*map(single, self._bijections, x.T))
        return jnp.stack(ys, axis = -1), sum(log_dets)

    def inverse(self, y, condition = None):
        single = lambda bijection, y: bijection.inverse(y)
        xs = list(map(single, self._bijections, y.T))
        return jnp.stack(xs, axis = -1)

    def inverse_and_log_det(self, y, condition = None):
        single = lambda bijection, y: bijection.inverse_and_log_det(y)
        xs, log_dets = zip(*map(single, self._bijections, y.T))
        return jnp.stack(xs, axis = -1), sum(log_dets)
