from math import isinf
from typing import Callable, Iterable, Iterator, Optional, Sequence, SupportsFloat, List as list

from . import _utils
from ._src import root_iter as _root_iter

__all__ = [
    "bisection",
    "chandrupatla",
    "default",
    "heun_ode",
    "midpoint_ode",
    "multivariate_bisection",
    "newton",
    "newton_ode",
    "non_simple",
    "rk45_ode",
    "secant",
    "stochastic",
]

def bisection(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    The bisection method ensures a very robust worst-case scenario.

    Does not converge fast for the best-case scenario. Other methods
    are recommended over the bisection method. For non-simple roots,
    use the non_simple method instead.

    Order of Convergence:
        1:
            Linear convergence.

    See also:
        non_simple:
            Fast convergence when |f(x)| ~ C * |x - root| ^ power.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.bisection(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def chandrupatla(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    Chandrupatla's method is a robust 3-point method using inverse
    quadratic interpolation, similar to Brent's method except for
    the fact that it uses an intelligent linearity check to determine
    if bisection should be used instead.

    Unlike Chandrupatla's original implementation, this one extends
    pyroot's secant implementation with more robust checks to ensure
    convergence by falling back to the secant method applicable. An
    advanced correction term is also used to obtain very high-order
    convergence for simple roots.

    Order of Convergence:
        1.820 or 1.839:
            Depending on f'(root), f''(root), and f'''(root), the
            order of convergence may be 1.820 or 1.839.

            The exact values are the roots of:
                1.820: x^9 - 7x^6 + 6x^3 - 1
                1.839: x^3 -  x^2 -  x   - 1

    See also:
        secant:
            A robust algorithm which ensures fast and tight bracketing
            for simple roots.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.chandrupatla(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

default = chandrupatla

def heun_ode(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float, float], float],
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    Heun's ODE method is the ODE equivalent of the trapezoidal method
    for numerical integration.

    Heun's ODE method uses 2 separate `fprime` evaluations per
    iteration to produce a more accurate estimate of the root compared
    to the newton_ode method. The additional `fprime` evaluation also
    makes Heun's ODE method more robust than the newton_ode method,
    giving it more accurate estimates during initial estimates. Similar
    to the newton_ode method, robust measures are taken to ensure
    tight brackets and worst-case convergence.

    Order of Convergence:
        3:
            Cubic convergence, similar to Halley's method.

    See also:
        midpoint_ode:
            Uses fewer `fprime` calls per iteration at the cost of a
            reduced order of convergence. Recommended if `fprime` calls
            are relatively expensive compared to `f` calls.
        newton_ode:
            Uses fewer `fprime` calls per iteration at the cost of a
            reduced order of convergence. Also has less arithmetic
            cost compared to other methods. Recommended if the
            cost of the algorithm itself is significantly more
            expensive than function calls.
        rk45_ode:
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        fprime:
            The derivative of `f`, giving `yprime = fprime(x, y)`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    elif not callable(fprime):
        return TypeError(f"expected a function for fprime, got {fprime!r}")
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.heun_ode(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def midpoint_ode(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float, float], float],
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    The midpoint ODE method is the ODE equivalent of the midpoint
    method for numerical integration.

    The midpoint ODE method uses only 1 `fprime` evaluation per
    iteration, similar to the newton_ode method, but does not use the
    derivative at the current point. Instead, a secant estimate of the
    root is made, and then the derivative is evaluated between that and
    the current estimate i.e. the derivative is evaluated at the
    midpoint. This gives more faster and more robust convergence than
    the newton_ode method.

    Order of Convergence:
        2.414:
            Between quadratic and cubic orders of convergence. Faster
            than the newton_ode method, but slower than the heun_ode
            method.

            The exact value is given by the root of:
                 x^2 - 2x - 1

    See also:
        heun_ode:
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        newton_ode:
            Uses fewer arithmetic operations per iteration. Recommended
            if the cost of the algorithm itself is significantly more
            expensive than function calls.
        rk45_ode:
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed extremely cheaply compared to `f`.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        fprime:
            The derivative of `f`, giving `yprime = fprime(x, y)`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    elif not callable(fprime):
        return TypeError(f"expected a function for fprime, got {fprime!r}")
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.midpoint_ode(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def multivariate_bisection(
    f: Callable[[list[float]], Iterable[float]],
    x1: Sequence[float],
    x2: Sequence[float],
    /,
    *,
    x: Optional[Sequence[float]] = None,
    y1: Optional[Sequence[float]] = None,
    y2: Optional[Sequence[float]] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[list[float]]:
    """
    A multivariate variant of the bisection method that drops
    guaranteed convergence for pseudo-guaranteed convergence.

    Multivariate bisection works by taking an initial search region and
    gradually dividing it into two halves. Rather than guaranteeing the
    root is in one half or the other, as is with traditional bisection,
    this method keeps both halves and remains unsure of which region
    contains the root. Each region is then bisected whenever the
    possibility of a root is detected within the region. A root is
    considered possible if both a positive and negative output is
    detected within the region.

    Each region, or bounding box, is defined as a collection of lower
    and upper bounds for each input dimension. Flags for the signs of
    each output are used to track if the bounding box has potential to
    contain a root. A counter for how many points have been sampled is
    saved to determine which bounding boxes should be searched.

    Within each bounding box, points are sampled. Initially, points are
    sampled at opposing corners of the bounding box with random points
    sifted between. Once every corner has been checked, only random
    points are sampled. The random points are sampled with bias towards
    the edges of each dimension.

    The overarching algorithm takes the initial bounding box and tracks
    all of the split up bounding boxes as iterations go on. The
    bounding box with the least samples so far is sampled from, and
    split boxes replace their original bounding box with the number of
    samples in each reset back to 0. This encourages searching in newly
    split boxes over older boxes.

    In theory, most functions should take roughly a constant number of
    iterations before splitting a bounding box, if the bounding box
    contains a bracketable root.

    Order of Convergence:
        1:
            Expectedly linear convergence. However, this is slowed down
            by the number of output dimensions.

    See also:
        bisection:
            Univariate equivalent.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for for every output.
        x1, x2:
            Two initial points which bound the region being searched
            for. `x1` should contain lower bounds for each input, while
            `x2` should contain upper bounds for each input.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.multivariate_type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    x1 = [float(x_) for x_ in x1]
    x2 = [float(x_) for x_ in x2]
    for i, (a1, a2) in enumerate(zip(x1, x2)):
        if a1 > a2:
            x1[i] = a2
            x2[i] = a1
        if isinf(x1[i]):
            x1[i] = _utils.sign(x1[i]) * _utils.FLOAT_MAX
        if isinf(x2[i]):
            x2[i] = _utils.sign(x2[i]) * _utils.FLOAT_MAX
    if x is not None:
        x = [float(x_) for x_ in x]
        if not all(L <= x_ <= U for L, x_, U in zip(x1, x, x2)):
            x = None
    if y1 is None:
        y1 = f(x1)
    if y2 is None:
        y2 = f(x2)
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, min(0.1 * U - 0.1 * L for L, U in zip(x1, x2)))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.multivariate_bisection(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def newton(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float], float],
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    The Newton-Raphson method is similar to the secant method but uses
    a given derivative instead of estimating it.

    Unlike the ODE methods, `fprime` only takes `x` as an argument.

    Order of Convergence:
        2:
            Quadratic convergence.

    See also:
        newton_ode:
            The ODE equivalent. Contains more information. Recommended
            if `fprime(x)` can be computed more efficiently if `f(x)` is
            given.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        fprime:
            The derivative of `f`, giving `yprime = fprime(x)`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    elif not callable(fprime):
        return TypeError(f"expected a function for fprime, got {fprime!r}")
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.newton(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def newton_ode(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float, float], float],
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    The Newton-Raphson ODE method is the ODE equivalent of Riemann sums
    for numerical integration.

    The Newton-Raphson ODE method uses 1 `fprime` evaluation per
    iteration with a fairly simple formula. Compared to similar
    methods, the arithmetic cost per iteration is slightly reduced.

    Order of Convergence:
        2:
            Quadratic convergence.

    See also:
        heun_ode:
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        midpoint_ode:
            Uses more arithmetic operations per iteration to gain
            increased order of convergence and robustness. Recommended
            if function calls are somewhat expensive.
        rk45_ode:
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed extremely cheaply compared to `f`.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        fprime:
            The derivative of `f`, giving `yprime = fprime(x, y)`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    elif not callable(fprime):
        return TypeError(f"expected a function for fprime, got {fprime!r}")
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.newton_ode(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def non_simple(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    The non-simple method excels at finding non-simple roots.

    Non-simple roots occur when `|f(x)| ~ C * |x - root| ^ power` and
    the `power` is not `1`. This is done by estimating the `power` each
    iteration and rescaling f(x) to make it linear. The secant method
    is then applied to the rescaled function values.

    Although superlinear convergence is eventually reached, many
    initial iterations may be spent performing slow approximations
    while the `power` is still inaccurate. Consider providing the
    `power` if it is known ahead of time.

    Order of Convergence:
        1.618:
            The same order of convergence as the secant method, but
            applies to non-simple roots such as `|x - 5| * (x - 5)`.

            The exact value is given by the root of:
                 x^2 - x - 1

    See also:
        secant:
            Equivalent method for simple roots.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.non_simple(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def rk45_ode(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float, float], float],
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    The RK45 ODE method is the ODE equivalent of Simpson's method for
    numerical integration.

    The RK45 ODE method uses 4 `fprime` evaluations per iteration to
    reach extremely high orders of convergence. The RK45 ODE method
    is recommended if `fprime` evaluations are exceptionally cheap to
    compute compared to `f` evaluations, such as for special functions
    like the error function.

    Order of Convergence:
        5:
            Quintic convergence.

    See also:
        heun_ode:
            Uses fewer `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        midpoint_ode:
            Uses significantly fewer `fprime` calls per iteration, but
            more arithmetic operations to compensate the order of
            convergence and its robustness. Recommended if function
            calls are somewhat expensive.
        newton_ode:
            Uses significantly fewer `fprime` calls per iteration as
            well as fewer arithmetic operations per iteration.
            Recommended if the cost of the algorithm itself is
            significantly more expensive than function calls.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        fprime:
            The derivative of `f`, giving `yprime = fprime(x, y)`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    elif not callable(fprime):
        return TypeError(f"expected a function for fprime, got {fprime!r}")
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.rk45_ode(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def secant(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    The secant method is a classical method which estimates the root
    using linear interpolations.

    Unlike the classical secant method, pyroot's implementation
    specializes in tight bracketing to guarantee convergence to the
    root. Unlike the Regula Falsi or Illinois methods, there is no
    decrease in order of convergence while maintaining a tight bracket.
    Unlike Dekker's method, worst-case linear convergence is guaranteed
    and the robustness of the algorithm is improved at the cost of more
    arithmetic operations per iteration due to a modification of
    Chandrupatla's method.

    Order of Convergence:
        1.618:
            The same order of convergence as the original secant
            method.

            The exact value is given by the root of:
                 x^2 - x - 1

    See also:
        chandrupatla:
            Uses higher order interpolation, obtaining a higher order
            of convergence at the cost of a bit more arithmetic
            operations per iteration.
        newton:
            Uses a given derivative instead of approximating the
            derivative using finite differences.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    y2 = float(y2)
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) need to have different signs")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.secant(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)

def stochastic(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    probability: float = 0.999,
    *,
    x: Optional[float] = None,
    y1: Optional[float] = None,
    y2: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[float]:
    """
    A stochastic root-finder for noisy functions.

    Uses repeated function evaluations until a confidence in the result
    is obtained. Computing `f(x)` with `n` evaluations only provides an
    accuracy of `O(1 / sqrt(n))`. The root is estimated an error of
    `O(log(n) / sqrt(n))` after `n` evaluations, which is nearly
    optimal.

    Order of Convergence:
        1:
            Sublinear convergence. Obtains an error of
            `O(log(n) / sqrt(n))` after `n` evaluations, which is
            nearly optimal.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) != sign(f(x2))`.
        probability:
            A desired probability for the accuracy used on each
            iteration. Function calls are repeated until a high
            amount of confidence in the accuracy of the result is
            reached.
        x, optional:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2, optional:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol, optional:
            An initial tolerance to help jump-start an initially tight
            bracket.

    Yields:
        x:
            An estimate for the root with guaranteed error.
    """
    exception = _utils.type_check(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    if exception is not None:
        raise exception
    if not isinstance(probability, SupportsFloat):
        raise TypeError(f"probability could not be interpreted as a real value, got {probability!r}")
    probability = float(probability)
    if not 0 < probability < 1:
        raise ValueError(f"probability could not be interpreted as a percentage (between 0 and 1), got {probability!r}")
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not _utils.is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = _utils.sign(x1) * _utils.FLOAT_MAX
    if isinf(x2):
        x2 = _utils.sign(x2) * _utils.FLOAT_MAX
    if y1 is not None:
        y1 = float(x1)
    if y2 is not None:
        y2 = float(x2)
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * x1 - 0.1 * x2))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return _root_iter.stochastic(f, x1, x2, _utils.inv_cdf(probability), x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
