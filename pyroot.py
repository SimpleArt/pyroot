"""
Example
-------
    >>> from pyroot import solver, solver_table, inf
    >>> def f(x):
    ...     return ((x - 1) * x + 2) * x - 5
    ...
    >>> x = solver(f, -inf, +inf)
    >>> x, f(x)
    (1.6398020042326555, 0.0)
    >>> print(solver_table(f, -inf, +inf))
      i              x               y
    ---  -------------  --------------
      0  -1.79769e+308  -inf
      1   1.79769e+308   inf
      2   0               -5
      3   1               -3
      4   4.09375         55.035
      5   2.54688         10.1277
      6   1.77344          0.979398
      7   1.65035          0.0720223
      8   1.63923         -0.00387468
      9   1.6398           1.76943e-05
     10   1.6398           1.31717e-11
     11   1.6398          -2.53042e-12
     12   1.6398           2.53042e-12
     13   1.6398           0
    x = 1.6398020042326555
"""
from typing import Any, Literal, Optional, Protocol, Union
from math import exp, inf, isinf, isnan, log, nan, sqrt
import sys

# Version dependent imports.
if sys.version_info[:2] >= (3, 9):
    from math import nextafter
    from collections.abc import Callable, Generator
    Dict = dict
else:
    from numpy import nextafter
    from typing import Callable, Dict, Generator

# Things for export.
__all__ = [
    "arithmetic_mean",
    "between",
    "bisection",
    "bracketing_method",
    "brent",
    "chandrupatla",
    "chandrupatla_mixed",
    "chandrupatla_quadratic",
    "dekker",
    "epsilon",
    "generalized_mean",
    "geometric_mean",
    "log_log_mean",
    "mean",
    "newton",
    "newton_quadratic",
    "nextafter",
    "sign",
    "signed_pow",
    "solver",
    "solver_generator",
    "solver_table",
]

# Store the machine precision.
epsilon: float = sys.float_info.epsilon
# Store the largest and smallest abs(float) other than 0 and inf.
# This is used for minimum precision and rounding inf into a usable value.
real_max: float = sys.float_info.max
real_min: float = sys.float_info.min

def sign(x: float) -> Literal[-1, 0, 1]:
    """
    Returns the sign of x.

    If x > 0, then return +1.
    If x < 0, then return -1.
    Otherwise return 0.
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def between(x1: float, x2: float, x3: float) -> bool:
    """Checks if x2 is between x1 and x3."""
    return x1 < x2 < x3 or x1 > x2 > x3

def signed_pow(x: float, order: float) -> float:
    """Equivalent to sign(x) * abs(x) ** order."""
    return sign(x) * abs(x) ** order

def arithmetic_mean(x: float, y: float, /) -> float:
    """Computes the arithmetic mean of two points."""
    # Avoid over-/under-flow.
    return x + 0.5 * (y - x) if sign(x) == sign(y) else 0.5 * (x + y)

def geometric_mean(x: float, y: float, /) -> float:
    """Computes the signed geometric mean of two points, and 0 if they have different signs."""
    # Avoid over-/under-flow.
    return sign(x) * sqrt(abs(x)) * sqrt(abs(y)) if sign(x) == sign(y) else 0.0

def log_log_mean(x: float, y: float, /) -> float:
    """Computes the signed geometric mean on a logarithmic scaling."""
    if sign(x) != sign(y):
        return 0.0
    else:
        return sign(x) * exp(geometric_mean(log(abs(x)), log(abs(y))))

def generalized_mean(x: float, y: float, /, order: float = 1) -> float:
    """
    Returns the signed generalized mean of x and y of the provided order.

    Equivalent to the formula
        ((x**order + y**order) / 2) ** (1/order)
    where `a**b` is taken as `signed_pow(a, b)`.

    Avoids overflowing, even if `signed_pow(...)` should overflow.

    Special Cases
    -------------

        Geometric Mean (order == 0):
            sign(x) * sqrt(abs(x * y)) if x and y are positive

        Arithmetic Mean (order == 1):
            (x + y) / 2

        x and y positive (order != 0):
            ((x**order + y**order) / 2) ** (1/order)
    """
    if order == 1:
        return arithmetic_mean(x, y)
    elif order == 0:
        return geometric_mean(x, y)
    # For other means, swap x and y appropriately, then rescale the
    # values before applying formulas.
    elif abs(x) > abs(y):
        x, y = y, x
    # Use numerically stable formula, assuming abs(y) >= abs(x).
    return y * (0.5 * abs(1 + sign(x) * sign(y) * abs(x/y) ** order)) ** (1 / order)

def mean(x: float, y: float, /, order: float = 1) -> float:
    """
    Returns the signed generalized mean of x and y of the provided order,
    except when order = 0 where the geometric mean on a logarithmic scale is used.

    Special Cases
    -------------

        Log Geometric Mean (order == 0):
            exp(sqrt(log(x) * log(y))) if x and y are positive

        Arithmetic Mean (order == 1):
            (x + y) / 2

        x and y positive (order != 0):
            ((x**order + y**order) / 2) ** (1/order)
    """
    return log_log_mean(x, y) if order == 0 else generalized_mean(x, y, order=order)

class BracketingMethod(Protocol):
    """
    Protocol for bracketing methods.

    Call Parameters
    ---------------

        t, between 0 and 1:
            Measures how much of a combination of x1 and x3 that x2 is.
            Satisfies x2 = x3 + t * (x1 - x3).

        (xi, yi) for i in (1, 2, 3, 4):
            A pair of (x, y) points.
            Satisfies the following invariants:
                x2 is between x1 and x3.
                x4 is not between x1 and x3.
                sign(x1) != sign(x2) == sign(x3).
            For example, the following ordering is possible:
                x4 < x1 < x2 < x3
            In other words,
                (x1 to x2) is the current bracket.
                (x1 to x3) is the previous bracket.
                (x1 to x4) or (x3 to x4) is the previous previous bracket,
                    depending on the sign of y4 compared to y1.

        *args, **kwargs:
            Any additional args or kwargs a method may need. For example, the
            Newt-safe method may take an `fprime` parameter for a derivative function.
            See a specific method's documentation for more accurate signature.

    Call Returns
    ------------

        t, between 0 and 1:
            The next estimate of the root should be:
                x = x2 + t * (x1 - x2)
            unless t == 0.5, in which case:
                x = mean(x1, x2, order)
            where the order depends on the solver() iteration.
    """

    def __call__(
        self,
        t: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        x3: float,
        y3: float,
        x4: float,
        y4: float,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        raise NotImplementedError

#=====================#
# Bracketing methods: #
#=====================#

def bisection(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """Always use the mean of x1 and x2."""
    return 0.5

def newton(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    fprime: Callable[[float], float] = None,
    classic: bool = False,
) -> float:
    """
    Estimates the root using a straight line.

    Parameters
    ----------

        fprime, optional:
            If fprime is not given, the secant method is used.
            If fprime is given, the Newton-Raphson method is used.
            **Note:
                If the NR method goes out of the bracketing interval,
                then bisection is used instead.

        classic, default False:
            If fprime is given and classic is True, then the classic
            Newton-Raphson method will be used. If classic is False,
            then a faster NR method is used.
            **Note:
                The classic NR method uses fprime(x2), but the faster
                NR method uses fprime(x) for some x between x1 and x2.
                In some cases this is undesirable because fprime(x2)
                might be computed when y2 = f(x2) is computed.

    Example
    -------

        >>> def f(x):
        ...     return x*x - 4
        ...
        >>> def fprime(x):
        ...     return 2 * x
        ...
        >>> solver(f, 0, 5, fprime, method="newton", classic=True)
        2.0

    Notes
    -----

        The Newton-Raphson method uses only (x2, y2).

        The faster NR method uses (x2, y2) and fprime(x),
        for some x between x1 and x2.
    """
    # Use the secant method by default.
    if fprime is None:
        return y2 / (y2 - y1)
    # Use the Newton-Raphson method.
    # Classic Newton's method takes x = x2.
    if classic:
        t = 0
    # Estimate the root using either inverse quadratic interpolation
    # or the secant method for the fast method.
    else:
        t = chandrupatla(t, x1, y1, x2, y2, x3, y3, x4, y4)
        if t == 0.5:
            t = y2 / (y2 - y1)
    # Estimate halfway between the root and x2.
    x = x2 + 0.5 * t * (x1 - x2)
    # Use the derivative.
    yprime = fprime(x)
    # If division by 0 occurs, use bisection.
    return 0.5 if yprime == 0 else y2 / (yprime * (x2 - x1))

def newton_quadratic(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """
    Estimates the given function using a polynomial interpolation
    through the first 3 given points.

    The root of the interpolating quadratic is only estimated using
    2 iterations of Newton's method. This is used to guarantee
    convergence to the correct root while avoiding square roots.
    """
    # Collect coefficients for Newton's polynomial interpolation.
    a = (y1 - y2) / (x1 - x2)
    b = (y2 - y3) / (x2 - x3)
    b = (a - b) / (x1 - x3)
    # If the points are collinear, use the secant estimate.
    if b == 0:
        return y2 / (y2 - y1)
    # Collect starting point for Newton's method.
    # Align with the convexity of the polynomial interpolation
    # to guarantee Newton's method lands in the current bracket.
    x = x2 if sign(b) == sign(y2) else x1
    # Apply Newton's method twice.
    for _ in range(2):
        # Numerically stable formula.
        x -= (y2 + a * (x-x2) + b * (x-x1) * (x-x2)) / (a + b * ((x-x1) + (x-x2)))
    # Rescale to get t.
    return (x - x2) / (x1 - x2)

def dekker(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """
    Estimates the root using Dekker's method.

    Uses a secant line from (x2, y2) to either (x1, y1) or (x3, y3), depending on
    which point is closest.

    Note
    ----
    If x3 is closer to x2 but using it does not result in a value
    between x1 and x2, then it is rejected and bisection is used.
    Division by 0 is checked here, and the solver checks if 0 < t < 1
    before defaulting to bisection.
    """
    # If x2 is closer to x1, then use (x1, y1).
    if abs(x2 - x1) <= abs(x2 - x3):
        return y2 / (y2 - y1)
    # If division by 0, then use bisection.
    elif y2 == y3:
        return 0.5
    # If x2 is closer to x3 and using (x3, y3) does
    # not result in division by 0, then use (x3, y3).
    else:
        return y2 * (x3 - x2) / ((y2 - y3) * (x1 - x3))

def brent(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """
    Estimates the root using Brent's method.

    If it can, inverse quadratic interpolation is used.
    Otherwise, secant interpolation is used.

    If the result is outside of the current bracket, or if
    convergence is too slow, then bisection is used instead.
    """
    # Check if bisection was previously used.
    flag = t == 0.5
    # Try inverse quadratic interpolation.
    if y2 != y3:
        # Numerically stable formulas.
        al = (x3 - x2) / (x1 - x2)
        a = y2 / (y1 - y2)
        b = y3 / (y1 - y3)
        c = y2 / (y3 - y2)
        d = y1 / (y3 - y1)
        t = a*b + c*d*al
    # Default to linear interpolation.
    else:
        t = y2 / (y2 - y1)
    # Check if bisection should be used.
    if (
        # Inverse quadratic interpolation out of bounds.
        not 0 < t < 1
        # Bisection was previously used
        # but x is going to be closer to x2 than x1.
        or flag and t >= 0.5
        # Interpolation was previously used
        # but the convergence was too slow.
        or not flag and t * abs(x1 - x2) >= 0.5 * abs(x3 - x4)
    ):
        t = 0.5
    return t

def chandrupatla(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """
    Estimates the root using Chandrupatla's method.

    Apply inverse quadratic interpolation whenever the interpolation
    is monotonic over the previous bracketing interval. This provides
    good guarantee of convergence.

    Intuitively, use (x1, y1, x2, y2, x3, y3) to check if the function
    looks very linear before attempting to use interpolation.

    If inverse quadratic interpolation should not be used,
    then default back to bisection.
    """
    # Measure the ratios of the differences.
    x = (x2 - x1) / (x3 - x1)
    y = (y2 - y1) / (y3 - y1)
    # Check if the inverse quadratic interpolation is monotonic.
    if y**2 < x < 1 - (1 - y)**2:
        # Numerically stable formulas.
        al = (x3 - x2) / (x1 - x2)
        a = y2 / (y1 - y2)
        b = y3 / (y1 - y3)
        c = y2 / (y3 - y2)
        d = y1 / (y3 - y1)
        return a*b + c*d*al
    # Default to bisection if interpolation shouldn't be used.
    else:
        return 0.5

def chandrupatla_quadratic(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """
    Estimates the root using Chandrupatla's method with quadratic
    interpolation instead of inverse quadratic interpolation.

    Apply quadratic interpolation whenever the interpolation is
    monotonic over the previous bracketing interval. This provides
    good guarantee of convergence.

    Intuitively, use (x1, y1, x2, y2, x3, y3) to check if the function
    looks very linear before attempting to use interpolation.

    If quadratic interpolation should not be used, then default back
    to bisection.

    Similar to newton_quadratic, the root of the quadratic
    interpolation is only estimated using 2 iterations of Newton's
    method to guarantee convergence to the correct root without
    using square roots.
    """
    # Measure the ratios of the differences.
    x = (x2 - x1) / (x3 - x1)
    y = (y2 - y1) / (y3 - y1)
    # Check if the quadratic interpolation is monotonic.
    if x**2 < y < 1 - (1 - x)**2:
        # Collect coefficients for Newton's polynomial interpolation.
        a = (y1 - y2) / (x1 - x2)
        b = (y2 - y3) / (x2 - x3)
        b = (a - b) / (x1 - x3)
        # If the points are collinear, use the secant estimate.
        if b == 0:
            return y2 / (y2 - y1)
        # Collect starting point for Newton's method.
        # Align with the convexity of the polynomial interpolation
        # to guarantee Newton's method lands in the current bracket.
        x = x2 if sign(b) == sign(y2) else x1
        # Apply Newton's method twice.
        for _ in range(2):
            # Numerically stable formula.
            x -= (y2 + a * (x-x2) + b * (x-x1) * (x-x2)) / (a + b * ((x-x1) + (x-x2)))
        # Rescale to get t.
        return (x - x2) / (x1 - x2)
    # Default to bisection if interpolation shouldn't be used.
    else:
        return 0.5

def chandrupatla_mixed(
    t: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """
    Estimates the root using Chandrupatla's method with both inverse
    and non-inverse quadratic interpolation. Chooses which should be
    used appropriately.

    Decides which method should be used based on which is more likely
    to cover (x2, y2). For example, quadratic interpolation is more
    appropriate if the function looks close to a quadratic near the
    root, while inverse quadratic interpolation is more appropriate
    if the function looks close to a radical.

    Apply interpolation whenever the interpolation is monotonic over
    the previous bracketing interval. This provides good guarantee of
    convergence.

    Intuitively, use (x1, y1, x2, y2, x3, y3) to check if the function
    looks very linear before attempting to use interpolation.

    If interpolation should not be used, then default back to bisection.

    Similar to newton_quadratic, the root of the quadratic
    interpolation is only estimated using 2 iterations of Newton's
    method to guarantee convergence to the correct root without
    using square roots.
    """
    # Measure the ratios of the differences.
    x = (x2 - x1) / (x3 - x1)
    y = (y2 - y1) / (y3 - y1)
    # Check if quadratic interpolation should be used.
    if x**2 < y < 1 - (1 - x)**2 and y * (1 - y) < x * (1 - x):
        # Collect coefficients for Newton's polynomial interpolation.
        a = (y1 - y2) / (x1 - x2)
        b = (y2 - y3) / (x2 - x3)
        b = (a - b) / (x1 - x3)
        # If the points are collinear, use the secant estimate.
        if b == 0:
            return y2 / (y2 - y1)
        # Collect starting point for Newton's method.
        # Align with the convexity of the polynomial interpolation
        # to guarantee Newton's method lands in the current bracket.
        x = x2 if sign(b) == sign(y2) else x1
        # Apply Newton's method twice.
        for _ in range(2):
            # Numerically stable formula.
            x -= (y2 + a * (x-x2) + b * (x-x1) * (x-x2)) / (a + b * ((x-x1) + (x-x2)))
        # Rescale to get t.
        return (x - x2) / (x1 - x2)
    # Check if inverse quadratic interpolation should be used.
    elif y**2 < x < 1 - (1 - y)**2:
        # Numerically stable formulas.
        al = (x3 - x2) / (x1 - x2)
        a = y2 / (y1 - y2)
        b = y3 / (y1 - y3)
        c = y2 / (y3 - y2)
        d = y1 / (y3 - y1)
        return a*b + c*d*al
    # Default to bisection if interpolation shouldn't be used.
    else:
        return 0.5

methods_dict: Dict[str, BracketingMethod] = {
    'bisection'              : bisection,
    'binary search'          : bisection,
    'newton'                 : newton,
    'newton raphson'         : newton,
    'newt safe'              : newton,
    'regula falsi'           : newton,
    'false position'         : newton,
    'secant'                 : newton,
    'muller'                 : newton_quadratic,
    'quadratic'              : newton_quadratic,
    'dekker'                 : dekker,
    'brent'                  : brent,
    'iq'                     : brent,
    'inverse quadratic'      : brent,
    'chandrupatla'           : chandrupatla,
    'inverse quadratic safe' : chandrupatla,
    'chandrupatla quadratic' : chandrupatla_quadratic,
    'quadratic safe'         : chandrupatla_quadratic,
    'chandrupatla mixed'     : chandrupatla_mixed,      # default
}

def solver(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    *args: Any,
    y1: float = nan,
    y2: float = nan,
    x: float = nan,
    method: Union[str, BracketingMethod] = chandrupatla_mixed,
    x_err: float = 0.0,
    r_err: float = 4096 * epsilon,
    x_tol: float = nan,
    r_tol: float = nan,
    refine: int = 15,
    mean: Callable[[float, float, float], float] = mean,
    **kwargs: Any,
) -> float:
    """
    Computes a root between x1 and x2 of the function f,
    provided that f(x1) and f(x2) have different signs.

    Parameters
    ----------

        f:
            The function whose root is being searched for.

        x1, x2:
            Two points where it is known that sign(f(x1)) != sign(f(x2)).
            It is not required that x1 < x2.

        **Note:
            Parameters above this point are positional-only parameters.

        *args, **kwargs:
            Any additional args or kwargs a method may need. For example, the
            Newt-safe method may take an `fprime` parameter for a derivative function.
            See a specific method's documentation for more accurate signature.

        **Note:
            Parameters below this point are keyword-only parameters.

        y1, y2, default f(x1), f(x2):
            Provide initially known values if necessary.

        x, default arithmetic_mean(x1, x2):
            The initial guess of the root. Useful if the initial bracketing points,
            x1 and x2, are fairly inaccurate, but a very accurate point is known
            ahead of time without a bracket. This is ignored if it is not between
            x1 and x2, otherwise it is used for the first iteration. May allow
            interpolation methods to rapidly bracket the root accurately.

        method: str
            Name may include uppercase letters, or either "_" or " "
                e.g., "Chandrupatla Mixed" or "chandrupatla_mixed".
            See below for details.
        method: BracketingMethod, default chandrupatla_mixed
            A bracketing method or the name of one (see above for options).
            The method used determines what the next point will be, unless it fails to
            converge rapidly enough, in which case bisection gains precedence.
        **Note:
            See methods_dict for bracketing methods.
        **Note:
            Using bisection i.e. 0.5 will actually use the arithmetic and geometric means,
            not just the standard arithmetic mean. This guarantees fast convergence when
            dealing with floats.

        x_err, default 0.0:
            The absolute error required for termination.
        r_err, default 4096 * epsilon ~ 1e-12 ~ 12 significant figures of precision:
            The relative error required for termination.
            By default, set to 4096 times the machine precision, but it can be expected
            that the end result is accurate all the way to machine precision.
            If higher precision is desired, 32 * epsilon ~ 1e-14 is recommended.
        x_tol: float, default r_tol * min(1, abs(x1 - x2))
            The absolute tolerance initially used to force iterations to move a certain
            distance instead of remaining stuck really close to x1 or x2.
        r_tol, default r_err / (1024 * epsilon) ~ 0.03125:
            The relative tolerance initially used to force iterations to move a certain
            distance instead of remaining stuck really close to x1 or x2.
        **Note:
            Although the error may only guarantee a certain amount of precision, the
            final iterations to refine the estimate will usually produce results accurate
            to machine precision anyways, especially if the root is simple.
        **Note:
            Once the initial tolerances are satisfied, they are reduced to the errors.
            All errors/tolerances also have a minimum limit greater than 0 according
            to real_min and epsilon for absolute and relative values respectively.

        refine:
            The floating-point values close to the final solution are checked to
            see if any additional accuracy may be obtained. The specified amount of
            additional iterations used on this step is taken. The procedure is secant
            estimates are used in combination with minimal tolerance provided by the
            nextafter function to round steps towards the middle of the bracket.
        **Note:
            One may wish to set this to 0 for some of the following reasons:
            - f cannot be computed to such precision.
            - Such precision is not needed.
            - f is expected to be well-behaved at the root, implying the final
              result does not require any refinement.

    Additional Notes
    ----------------

        If f(x) produces nan, iterations are halted, and a result is returned.

        If (x1, y1) and (x2, y2) do not give a bracket, then an estimate of the root
            is returned as either mean(x1, x2) or the secant estimate of the root.

        The returned result may not have abs(f(x)) as minimal. This is because the root
        needs to be surrounded on both sides instead of just close to 0. An anomalous
        example would be f(x) = 1/x, for which x = 0 is the only point surrounded by
        the bracket, despite being furthest from f(x) = 0.

    Example
    -------

        Finding the root of a cubic on the interval (-inf to +inf).
        Write the polynomial in Horner's method to accurately handle
        very large values of x properly.
        The result is reached exactly in only 13 function evaluations,
        despite the size of the initial bracketing interval.
        >>> def f(x):
        ...     return ((x - 1) * x + 2) * x - 5
        ...
        >>> x = solver(f, -inf, +inf)
        >>> x, f(x)
        (1.6398020042326555, 0.0)
    """
    # Choose a bracketing method.
    if isinstance(method, str):
        method = method.lower().replace("_", " ")
    method = methods_dict.get(method, method)
    # Cast values to floats to avoid weird integer stuff.
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    x = float(x)
    x_err = float(x_err)
    r_err = float(r_err)
    x_tol = float(x_tol)
    r_tol = float(r_tol)
    # Round infinity down to the largest non-inf floats.
    if isinf(x1):
        x1 = sign(x1) * real_max
    if isinf(x2):
        x2 = sign(x2) * real_max
    # Compute the bracketing points.
    if isnan(y1):
        y1 = f(x1)
    if isnan(y2):
        y2 = f(x2)
    # Stop if there's no bracket to search in.
    if x1 == x2 or isnan(y1) or isnan(y2) or y1 == y2:
        return mean(x1, x2)
    elif sign(y1) == sign(y2) or y1 == 0 or y2 == 0:
        return (y2*x1 - y1*x2) / (y2 - y1)
    # Set the default tolerances.
    if isnan(r_tol):
        r_tol = r_err / (1024 * epsilon)
    if isnan(x_tol):
        x_tol = r_tol * min(1, abs(x1 - x2))
    # Set minimum possible errors and tolerances possible.
    x_err = max(x_err, 32 * real_min)
    x_tol = max(x_err, x_tol)
    r_err = max(r_err, 32 * epsilon)
    r_tol = max(r_err, r_tol)
    # Set maximum possible relative errors and tolerances possible.
    r_err = min(r_err, 0.5)
    r_tol = min(r_tol, 0.5)
    # Initialize iteration variables.
    x3 = y3 = x4 = y4 = nan
    t = 0.5
    bisection_fails = 0
    # Choose to use arithmetic or geometric mean,
    # depending on how far apart x1 and x2 are.
    bisection_flag = x_tol > epsilon * abs(x1 - x2)
    # If the the points are very far apart, usually
    # the root is closer to the smaller point.
    if not bisection_flag and isnan(x) and sign(x1) == sign(x2):
        x = nextafter(*sorted([x1, x2], key=abs))
    # Estimate the order of the root as we go.
    order1 = 1
    order2 = 0.875
    order3 = nan
    def order_error(order: float) -> float:
        return signed_pow((signed_pow(y2, order) + (signed_pow(y1, order) - signed_pow(y2, order)) * (x - x2) / (x1 - x2) - signed_pow(y, order)), 1/order)
    # Loop until convergence.
    while abs(x1 - x2) > x_err + r_err * abs(x2) and y2 != 0 and not isnan(y2):
        # Skip if given x is in the interval.
        if between(x1, x, x2):
            t = (x - x2) / (x1 - x2)
        # Use t's formula.
        elif t != 0.5:
            x = x2 + t * (x1 - x2)
        # Use arithmetic mean for bisection when x1 and x2 are close.
        elif between(0.25 * x1, x2, 4 * x1):
            x = mean(x1, x2)
        # Use arithmetic or log-log mean for bisection at first.
        elif bisection_fails < 3:
            x = mean(x1, x2, bisection_flag)
        # Use generalized mean for bisection otherwise.
        else:
            x = mean(x1, x2, bisection_flag / (bisection_fails - 2))
        # Use lower tolerance once converging.
        if abs(x1 - x2) < 16 * (x_tol + r_tol * abs(x)):
            x_tol = x_err
            r_tol = r_err
        # Round towards the midpoint with the tolerance.
        tol = 0.25 * (x_tol + r_tol * abs(x))
        x += tol * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Update the estimate of the order of the root.
        try:
            new_order = order1 - order_error(order1) * (order1 - order2) / (order_error(order1) - order_error(order2))
        except (OverflowError, ZeroDivisionError):
            pass
        else:
            # Don't update it to nan, which may occur during inf / inf.
            if isnan(new_order):
                pass
            # Avoid chaotic changes to the order estimation by allowing it to at most either halve or double.
            elif isnan(order3):
                order1, order2, order3 = min(max(new_order, 0.5 * order1), 2 * order1), order1, order2
            # Use Aitken delta-squared acceleration.
            else:
                new_order -= (new_order - order1) ** 2 / ((new_order - order1) - (order1 - order2))
                order1, order2, order3 = min(max(new_order, 0.5 * order1), 2 * order1), order1, nan
        # Swap to ensure x is moved to x2 and sign(y1) != sign(y).
        if sign(y) == sign(y1):
            t = 1 - t
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        # Shift variables so that:
        # (x1, x2, x3) -> (x1, x, x2, x3)
        x2, x3, x4 = x, x2, x3
        y2, y3, y4 = y, y2, y3
        # Use bisection if x1 and x2 are too far apart (geometrically).
        if not between(0.125 * x1, x2, 8 * x1):
            # If the current type of mean used is not rapidly improving, switch means.
            if (x1 - x2) / (x1 - x3) > 0.125:
                bisection_flag = not bisection_flag
            bisection_fails += 1
        # Use bisection if the bracketing interval fails to halve.
        elif t < 0.5:
            bisection_fails += 1
        # Don't use bisection if the bracketing interval more than halved.
        elif t > 0.5:
            bisection_fails = 0
        # If the geometric mean is used and it fails to reduce the
        # bracketing interval, more arithmetic means are needed.
        elif (x1 - x2) / (x1 - x3) > 0.75:
            bisection_flag = not bisection_flag
            bisection_fails += 1
        # If the current mean used is only okay, switch means and
        # attempt to use the bracketing method instead.
        elif (x1 - x2) / (x1 - x3) > 0.125:
            bisection_flag = not bisection_flag
            bisection_fails = 0
        # If the geometric mean is used and significantly reduces the
        # bracketing interval, don't switch means and try the
        # bracketing method instead.
        else:
            bisection_fails = 0
        # Resort to bisection if convergence failed to happen more than 3 times in a row.
        if bisection_fails > 3:
            t = 0.5
        # If x2 and x3 are significantly closer to each other than x1,
        # attempt to use the secant method without (x1, y1). Over-step
        # to try to get closer to x1 and rapidly tighten the bracket.
        elif abs(x2 - x3) < epsilon * abs(x2 - x1):
            t = (x3 - x2) / (x3 - x1) * (abs(y2) / (abs(y3 - y2) + 2 * tol) + 0.125 * tol)
            # Approximation at where the root might be under Lipschitz continuity assumptions.
            t = (x3 - x2) / (x3 - x1) * (abs(y2) / (abs(y3 - y2) + 2 * tol) + 0.125 * tol)
            # Over-step to try to capture the opposite side of the bracket.
            if bisection_fails < 3:
                t *= 2
            # If bisection is going to be used next, over-step even further to try to avoid that.
            else:
                t *= 4
            # Use bisection if out of bounds.
            if not 0 < t < 1:
                t = 0.5
        # If x1 and x2 are significantly closer to each other than x3,
        # but x1 and x2 are not within a factor of 1024 of each other,
        # attempt to use the secant method without (x3, y3).
        elif abs(x1 - x2) < epsilon * abs(x2 - x3) and not between(x1 / 1024, x2, x1 * 1024):
            t = y2 / (y2 - y1)
            # Allow t = 1 to get closer to x1 using tolerance instead.
            if t == 1:
                x = x1
            # Use bisection if out of bounds.
            elif t == 0:
                t = 0.5
        # Try the current bracketing method if corrections are not necessary.
        elif bisection_fails == 2:
            try:
                t = method(t, x1, signed_pow(y1, order1), x2, signed_pow(y2, order1), x3, signed_pow(y3, order1), x4, signed_pow(y4, order1), *args, **kwargs)
            except OverflowError:
                t = method(t, x1, y1, x2, y2, x3, y3, x4, y4, *args, **kwargs)
            # Use bisection if out of bounds.
            if not 0 < t < 1:
                t = 0.5
        elif bisection_fails < 2:
            t = method(t, x1, y1, x2, y2, x3, y3, x4, y4, *args, **kwargs)
            # Use bisection if out of bounds.
            if not 0 < t < 1:
                t = 0.5
        # On the 3rd failure to bracket the root, attempt to balance
        # out the root with a shifted estimate of the root.
        else:
            # Try the Illinois method by faking the y values.
            # Also attempt to linearize the estimate of the root.
            try:
                y_temp = 0.5 * signed_pow(y3, order1) * ((x1 - x2) / (x3 - x2))
            except OverflowError:
                y_temp = 0.5 * y3 * ((x1 - x2) / (x3 - x2))
                t = method(t, x1, y_temp, x2, y2, x3, y3, x4, y4, *args, **kwargs)
            else:
                t = method(t, x1, y_temp, x2, signed_pow(y2, order1), x3, signed_pow(y3, order1), x4, signed_pow(y4, order1), *args, **kwargs)
            # Resort to double-stepping if Illinois has no effect.
            if t == method(t, x1, y1, x2, y2, x3, y3, x4, y4, *args, **kwargs):
                t *= 2
            # Don't go past bisection.
            t = min(0.5, t)
    # Refine the estimate.
    for i in range(refine):
        # Use a secant estimate of the root.
        # Note the formula is numerically stable and good for high precision.
        x = (y2*x1 - y1*x2) / (y2 - y1)
        if i > 3:
            try:
                x = (signed_pow(y2, order1) * x1 - signed_pow(y1, order1) * x2) / (signed_pow(y2, order1) - signed_pow(y1, order1))
            except (OverflowError, ZeroDivisionError):
                pass
        # Round towards the middle, except for the first iteration.
        if i > 0:
            x = nextafter(x, mean(x1, x2))
        # Stop if the root is found.
        if x == x1 or x == x2 or nextafter(nextafter(x1, x2), x2) == x2:
            break
        # Compute the next point.
        y = f(x)
        # Update the estimate of the order of the root.
        try:
            order1, order2 = order1 - order_error(order1) * (order1 - order2) / (order_error(order1) - order_error(order2)), order1
        except (OverflowError, ZeroDivisionError):
            pass
        # Stop if the root is found.
        if isnan(y) or y == 0:
            return x
        # Otherwise update the bounds.
        elif sign(y) == sign(y1):
            x1, y1 = x, y
        else:
            x2, y2 = x, y
    # If no best is found yet, return the next secant approximation of the root.
    return (y2*x1 - y1*x2) / (y2 - y1)

def solver_generator(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    *args: Any,
    y1: float = nan,
    y2: float = nan,
    x: float = nan,
    method: Union[str, BracketingMethod] = chandrupatla_mixed,
    x_err: float = 0.0,
    r_err: float = 4096 * epsilon,
    x_tol: float = nan,
    r_tol: float = nan,
    refine: int = 15,
    mean: Callable[[float, float, float], float] = mean,
    **kwargs: Any,
) -> Generator[float, Optional[float], None]:
    """
    Equivalent to solver(), except every point x is yielded after it is used.

    Yields
    ------
        x_i:
            The next computed point, but f(x_i) may not be computed on the last iteration.

    Returns
    -------
        x:
            The best estimate of the bracketed root, which may not be yielded last,
            and abs(f(x)) may not be the smallest. See below for more information.

    Notes
    -----
        See `help(solver)` for information related to the `solver()` in general.

        See `help(pyroot)` for sample usage.

        If generator.send() is used, then that point is used on the next
        iteration as long as it lies in the bracketing interval.

        Initially, x1 is yielded followed by x2, but nothing is collected from
        generator.send() for these first iterations. Use the x parameter for this instead.

        The returned result may not have abs(f(x)) as minimal. This is because the root
        needs to be surrounded on both sides instead of just close to 0. An anomalous
        example would be f(x) = 1/x, for which x = 0 is the only point surrounded by
        the bracket, despite being furthest from f(x) = 0.

        The last yielded value is not necessarily the final result. Instead, the
        returned value is the final result, and may be accessed using `yield from`.
        >>> def tabulate_results(f, x1, x2):
        ...     x: float
        ...     def get_x():
        ...         nonlocal x
        ...         x = yield from solver_generator(f, x1, x2)
        ...     print(tabulate([(i, x_i, f(x_i)) for i, x_i in enumerate(get_x())], ("i", "x", "y")))
        ...     print(f"Final result: {x}")

    Example
    -------
        >>> from tabulate import tabulate
        >>> def f(x):
        ...     return ((x - 1) * x + 2) * x - 5
        ...
        >>> solutions = enumerate(solver_generator(f, x1, x2))
        >>> print(tabulate([(i, x_i, f(x_i)) for i, x_i in solutions], ("i", "x", "y")))
          i              x               y
        ---  -------------  --------------
          0  -1.79769e+308  -inf
          1   1.79769e+308   inf
          2   0               -5
          3   8.98847e+307   inf
          4   0.0078125       -4.98444
          5   1.68361          0.30496
          6   0.491764        -4.13938
          7   1.62344         -0.11002
          8   1.64042          0.00417759
          9   1.6398           4.41056e-07
         10   1.6398          -2.34301e-12
         11   1.6398           2.13163e-14
         12   1.6398          -1.95399e-14
         13   1.6398           0
    """
    # Choose a bracketing method.
    if isinstance(method, str):
        method = method.lower().replace("_", " ")
    method = methods_dict.get(method, method)
    # Cast values to floats to avoid weird integer stuff.
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    x = float(x)
    x_err = float(x_err)
    r_err = float(r_err)
    x_tol = float(x_tol)
    r_tol = float(r_tol)
    # Round infinity down to the largest non-inf floats.
    if isinf(x1):
        x1 = sign(x1) * real_max
    if isinf(x2):
        x2 = sign(x2) * real_max
    yield x1
    yield x2
    # Compute the bracketing points.
    if isnan(y1):
        y1 = f(x1)
    if isnan(y2):
        y2 = f(x2)
    # Stop if there's no bracket to search in.
    if isnan(y1) or isnan(y2) or y1 == y2:
        x = mean(x1, x2)
        yield x
        return x
    elif sign(y1) == sign(y2) or y1 == 0 or y2 == 0:
        return (y2*x1 - y1*x2) / (y2 - y1)
    # Set the default tolerances.
    if isnan(r_tol):
        r_tol = r_err / (1024 * epsilon)
    if isnan(x_tol):
        x_tol = r_tol * min(1, abs(x1 - x2))
    # Set minimum possible errors and tolerances possible.
    x_err = max(x_err, 32 * real_min)
    x_tol = max(x_err, x_tol)
    r_err = max(r_err, 32 * epsilon)
    r_tol = max(r_err, r_tol)
    # Set maximum possible relative errors and tolerances possible.
    r_err = min(r_err, 0.5)
    r_tol = min(r_tol, 0.5)
    # Initialize iteration variables.
    x3 = y3 = x4 = y4 = nan
    t = 0.5
    bisection_fails = 0
    # Choose to use arithmetic or geometric mean,
    # depending on how far apart x1 and x2 are.
    bisection_flag = x_tol > epsilon * abs(x1 - x2)
    # If the the points are very far apart, usually
    # the root is closer to the smaller point.
    if not bisection_flag and isnan(x) and sign(x1) == sign(x2):
        x = nextafter(*sorted([x1, x2], key=abs))
    # Estimate the order of the root as we go.
    order1 = 1
    order2 = 0.875
    order3 = nan
    def order_error(order: float) -> float:
        return signed_pow((signed_pow(y2, order) + (signed_pow(y1, order) - signed_pow(y2, order)) * (x - x2) / (x1 - x2) - signed_pow(y, order)), 1/order)
    # Loop until convergence.
    while abs(x1 - x2) > x_err + r_err * abs(x2) and y2 != 0 and not isnan(y2):
        # Skip if given x is in the interval.
        if between(x1, x, x2):
            t = (x - x2) / (x1 - x2)
        # Use t's formula.
        elif t != 0.5:
            x = x2 + t * (x1 - x2)
        # Use arithmetic mean for bisection when x1 and x2 are close.
        elif between(0.25 * x1, x2, 4 * x1):
            x = mean(x1, x2)
        # Use arithmetic or log-log mean for bisection at first.
        elif bisection_fails < 3:
            x = mean(x1, x2, bisection_flag)
        # Use generalized mean for bisection otherwise.
        else:
            x = mean(x1, x2, bisection_flag / (bisection_fails - 2))
        # Use lower tolerance once converging.
        if abs(x1 - x2) < 16 * (x_tol + r_tol * abs(x)):
            x_tol = x_err
            r_tol = r_err
        # Round towards the midpoint with the tolerance.
        tol = 0.25 * (x_tol + r_tol * abs(x))
        x += tol * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Update the estimate of the order of the root.
        try:
            new_order = order1 - order_error(order1) * (order1 - order2) / (order_error(order1) - order_error(order2))
        except (OverflowError, ZeroDivisionError):
            pass
        else:
            # Don't update it to nan, which may occur during inf / inf.
            if isnan(new_order):
                pass
            # Avoid chaotic changes to the order estimation by allowing it to at most either halve or double.
            elif isnan(order3):
                order1, order2, order3 = min(max(new_order, 0.5 * order1), 2 * order1), order1, order2
            # Use Aitken delta-squared acceleration.
            else:
                try:
                    new_order -= (new_order - order1) ** 2 / ((new_order - order1) - (order1 - order2))
                except ZeroDivisionError:
                    pass
                order1, order2, order3 = min(max(new_order, 0.5 * order1), 2 * order1), order1, nan
        # Swap to ensure x is moved to x2 and sign(y1) != sign(y).
        if sign(y) == sign(y1):
            t = 1 - t
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        # Shift variables so that:
        # (x1, x2, x3) -> (x1, x, x2, x3)
        x2, x3, x4 = x, x2, x3
        y2, y3, y4 = y, y2, y3
        # Use bisection if x1 and x2 are too far apart (geometrically).
        if not between(0.125 * x1, x2, 8 * x1):
            # If the current type of mean used is not rapidly improving, switch means.
            if (x1 - x2) / (x1 - x3) > 0.125:
                bisection_flag = not bisection_flag
            bisection_fails += 1
        # Use bisection if the bracketing interval fails to halve.
        elif t < 0.5:
            bisection_fails += 1
        # Don't use bisection if the bracketing interval more than halved.
        elif t > 0.5:
            bisection_fails = 0
        # If the geometric mean is used and it fails to reduce the
        # bracketing interval, more arithmetic means are needed.
        elif (x1 - x2) / (x1 - x3) > 0.75:
            bisection_flag = not bisection_flag
            bisection_fails += 1
        # If the current mean used is only okay, switch means and
        # attempt to use the bracketing method instead.
        elif (x1 - x2) / (x1 - x3) > 0.125:
            bisection_flag = not bisection_flag
            bisection_fails = 0
        # If the geometric mean is used and significantly reduces the
        # bracketing interval, don't switch means and try the
        # bracketing method instead.
        else:
            bisection_fails = 0
        # Generate and collect the next point, if any.
        x = yield x2
        if x is not None:
            continue
        x = nan
        # Resort to bisection if convergence failed to happen more than 3 times in a row.
        if bisection_fails > 3:
            t = 0.5
        # If x2 and x3 are significantly closer to each other than x1,
        # attempt to use the secant method without (x1, y1). Over-step
        # to try to get closer to x1 and rapidly tighten the bracket.
        elif abs(x2 - x3) < epsilon * abs(x2 - x1):
            # Approximation at where the root might be under Lipschitz continuity assumptions.
            t = (x3 - x2) / (x3 - x1) * (abs(y2) / (abs(y3 - y2) + 2 * tol) + 0.125 * tol)
            # Over-step to try to capture the opposite side of the bracket.
            if bisection_fails < 3:
                t *= 2
            # If bisection is going to be used next, over-step even further to try to avoid that.
            else:
                t *= 4
            # Use bisection if out of bounds.
            if not 0 < t < 1:
                t = 0.5
        # If x1 and x2 are significantly closer to each other than x3,
        # but x1 and x2 are not within a factor of 1024 of each other,
        # attempt to use the secant method without (x3, y3).
        elif abs(x1 - x2) < epsilon * abs(x2 - x3) and not between(x1 / 1024, x2, x1 * 1024):
            t = y2 / (y2 - y1)
            # Allow t = 1 to get closer to x1 using tolerance instead.
            if t == 1:
                x = x1
            # Use bisection if out of bounds.
            elif t == 0:
                t = 0.5
        # Try the current bracketing method if corrections are not necessary.
        elif bisection_fails == 2:
            try:
                t = method(t, x1, signed_pow(y1, order1), x2, signed_pow(y2, order1), x3, signed_pow(y3, order1), x4, signed_pow(y4, order1), *args, **kwargs)
            except OverflowError:
                t = method(t, x1, y1, x2, y2, x3, y3, x4, y4, *args, **kwargs)
            # Use bisection if out of bounds.
            if not 0 < t < 1:
                t = 0.5
        elif bisection_fails < 2:
            t = method(t, x1, y1, x2, y2, x3, y3, x4, y4, *args, **kwargs)
            # Use bisection if out of bounds.
            if not 0 < t < 1:
                t = 0.5
        # On the 3rd failure to bracket the root, attempt to balance
        # out the root with a shifted estimate of the root.
        else:
            # Try the Illinois method by faking the y values.
            # Also attempt to linearize the estimate of the root.
            try:
                y_temp = 0.5 * signed_pow(y3, order1) * ((x1 - x2) / (x3 - x2))
            except OverflowError:
                y_temp = 0.5 * y3 * ((x1 - x2) / (x3 - x2))
                t = method(t, x1, y_temp, x2, y2, x3, y3, x4, y4, *args, **kwargs)
            else:
                t = method(t, x1, y_temp, x2, signed_pow(y2, order1), x3, signed_pow(y3, order1), x4, signed_pow(y4, order1), *args, **kwargs)
            # Resort to double-stepping if Illinois has no effect.
            if t == method(t, x1, y1, x2, y2, x3, y3, x4, y4, *args, **kwargs):
                t *= 2
            # Don't go past bisection.
            t = min(0.5, t)
    # If y2 is nan or 0, then the last result, x2, is the best estimate.
    if isnan(y2) or y2 == 0:
        return x2
    # Refine the estimate.
    for i in range(refine):
        # Use a secant estimate of the root.
        # Note the formula is numerically stable and good for high precision.
        x = (y2*x1 - y1*x2) / (y2 - y1)
        if i > 3:
            try:
                x = (signed_pow(y2 / y1, order1) * x1 - x2) / (signed_pow(y2 / y1, order1) - 1)
            except (OverflowError, ZeroDivisionError):
                pass
        # Round towards the middle, except for the first iteration.
        if i > 0:
            x = nextafter(x, mean(x1, x2))
        # Stop if the root is found.
        if x == x1 or x == x2 or nextafter(nextafter(x1, x2), x2) == x2:
            break
        # Compute the next point.
        yield x
        y = f(x)
        # Update the estimate of the order of the root.
        try:
            order1, order2 = order1 - order_error(order1) * (order1 - order2) / (order_error(order1) - order_error(order2)), order1
        except (OverflowError, ZeroDivisionError):
            pass
        # Stop if the root is found.
        if isnan(y) or y == 0:
            return x
        # Otherwise update the bounds.
        elif sign(y) == sign(y1):
            x1, y1 = x, y
        else:
            x2, y2 = x, y
    # If no best is found yet, return the next secant approximation of the root.
    x = (y2*x1 - y1*x2) / (y2 - y1)
    # Only yield if it's a new estimate.
    if x1 != x != x2:
        yield x
    return x

def solver_table(f: Callable[[float], float], *args: Any, **kwargs: Any) -> str:
    """
    A helper method for `solver_generator` which generates the `(i, x, y)` results and tabulates them.

    See `solver_generator` for additional documentation.

    Example
    --------
        >>> from pyroot import solver_table
        >>> print(solver_table(lambda x: x*x - 2, 0, float("inf")))
          i             x              y
        ---  ------------  -------------
          0  0              -2
          1  1.79769e+308  inf
          2  1              -1
          3  3.15625         7.96191
          4  2.07812         2.3186
          5  1.41791         0.0104761
          6  1.41421         2.2371e-11
          7  1.41421        -9.20375e-13
          8  1.41421         9.09051e-13
          9  1.41421        -4.44089e-16
         10  1.41421         8.88178e-16
         11  1.41421         4.44089e-16
        x = 1.4142135623730951
    """
    from tabulate import tabulate
    x: float
    def get_x():
        nonlocal x
        x = yield from solver_generator(f, *args, **kwargs)
    return tabulate([(i, x, f(x)) for i, x in enumerate(get_x())], ("i", "x", "y")) + f"\nx = {x}"
