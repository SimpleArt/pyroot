import math
import sys
from collections.abc import Callable, Iterator
from enum import Enum
from functools import partial
from math import exp, inf, isinf, isnan, log, nan, sqrt
from typing import Any, Literal, Optional, SupportsFloat, Union, overload

__all__ = ["root_in", "root_iter"]

FLOAT_EPSILON: float = 2 * sys.float_info.epsilon
FLOAT_MAX: float = sys.float_info.max
FLOAT_MIN: float = sys.float_info.min
FLOAT_SMALL_EPSILON: float = 2.0 ** -1074

DerivativeFreeMethod = Literal["bisect", "chandrupatla", "secant"]
ODEMethod = Literal["heun-ode", "midpoint-ode", "newt-ode", "rk45-ode"]
Method = Union[DerivativeFreeMethod, ODEMethod, Literal["newt-safe", "non-simple"]]

derivative_free_methods: tuple[DerivativeFreeMethod, ...] = ("bisect", "chandrupatla", "secant")  # type: ignore
ode_methods: tuple[ODEMethod, ...] = ("heun-ode", "midpoint-ode", "newt-ode", "rk45-ode")  # type: ignore
methods: tuple[Method, ...] = ("bisect", "chandrupatla", "heun-ode", "midpoint-ode", "newt-ode", "newt-safe", "non-simple", "rk45-ode", "secant")  # type: ignore

def float_mean(x1: float, x2: float, /) -> float:
    """
    Special mean of x1 and x2 for floats.

    Used for bisecting over floats faster than the arithmetic mean
    by converging first to the correct exponent before converging
    to the correct mantissa.
    """
    s = 0.5 * (sign(x1) + sign(x2))
    x1 *= s
    x2 *= s
    if abs(s) != 1:
        return 0.0
    elif x1 / 8 < x2 < x1 * 8:
        return s * (x1 + 0.5 * (x2 - x1))
    elif x1 <= 1 <= x2 or x1 >= 1 >= x2:
        return s
    elif sqrt(sqrt(x1)) < x2 < x1 * x1 * x1 * x1:
        return s * sqrt(x1) * sqrt(x2)
    elif x1 < 1:
        return s * exp(-sqrt(log(x1) * log(x2)))
    else:
        return s * exp(sqrt(log(x1) * log(x2)))

def is_between(lo: float, x: float, hi: float, /) -> bool:
    """Checks if `x` is between the `lo` and `hi` arguments."""
    return lo < x < hi or lo > x > hi

def mean(x1: float, x2: float, /) -> float:
    """Returns the arithmetic mean of x1 and x2 without overflowing."""
    return x1 + 0.5 * (x2 - x1) if sign(x1) == sign(x2) else 0.5 * (x1 + x2)

def _power_estimate_error(x1: float, x2: float, x3: float, y1: float, y2: float, y3: float, power: float, /) -> float:
    """Estimates the error of the power for the given 3 points."""
    if (power < 0) ^ (abs(y1) < abs(y3)):
        y1 /= y3
        y2 /= y3
        y3 /= y3
    else:
        y3 /= y1
        y2 /= y1
        y1 /= y1
    y1 = signed_pow(y1, power)
    y2 = signed_pow(y2, power)
    y3 = signed_pow(y3, power)
    if abs(x1 - x2) < abs(x2 - x3):
        return (y3 - y1) * (x2 - x1) / (x3 - x1) + y1 - y2
    else:
        return (y1 - y3) * (x2 - x3) / (x1 - x3) + y3 - y2

def power_estimate(x1: float, x2: float, x3: float, y1: float, y2: float, y3: float, power: float, /) -> float:
    """Estimates the the power where `(x, signed_pow(y, power))` forms a straight line."""
    if is_between(y1, y2, y3) ^ (power > 0):
        power *= -1
    if isinf(y1) or isinf(y2) or isinf(y3):
        return power
    p1 = power
    p2 = power * 1.1 + sign(power)
    yp1 = _power_estimate_error(x1, x2, x3, y1, y2, y3, p1)
    yp2 = _power_estimate_error(x1, x2, x3, y1, y2, y3, p2)
    p2 = power + 0.25 * secant(0.0, p2 - p1, yp1, yp2)
    yp2 = _power_estimate_error(x1, x2, x3, y1, y2, y3, p2)
    p1 = power + 0.5 * secant(0.0, p2 - p1, yp1, yp2)
    yp1 = _power_estimate_error(x1, x2, x3, y1, y2, y3, p1)
    p2 = secant(p1, p2, yp1, yp2)
    return p2 if not isnan(p2) else p1 if not isnan(p1) else power

def secant(x1: float, x2: float, y1: float, y2: float, power: Optional[float] = None, /) -> float:
    """Helper function to handle edge cases during secant interpolation e.g. overflow."""
    if isinf(y1) and isinf(y2) or y1 == y2:
        return x1 + 0.5 * (x2 - x1) if sign(x1) == sign(x2) else 0.5 * (x1 + x2)
    if abs(y1) < abs(y2):
        y1 /= y2
        y2 = 1.0
    else:
        y2 /= y1
        y1 = 1.0
    if power is None:
        pass
    elif power > 0:
        y1 = signed_pow(y1, power)
        y2 = signed_pow(y2, power)
    else:
        y1 = signed_pow(y1, -power)
        y2 = signed_pow(y2, -power)
        y1, y2 = y2, y1
    if sign(y1) != sign(y2):
        return (y1 * x2 - y2 * x1) / (y1 - y2)
    elif abs(y1) < abs(y2):
        return x1 - (x1 - x2) / (1 - y2 / y1)
    else:
        return x2 - (x2 - x1) / (1 - y1 / y2)

def sign(x: float, /) -> Literal[-1, 0, 1]:
    """Returns the sign of a real number: -1, 0, or 1."""
    return 1 if x > 0 else -1 if x < 0 else 0

def signed_pow(x: float, power: float, /) -> float:
    """Returns sign(x) * pow(abs(x), power)."""
    try:
        return sign(x) * math.pow(abs(x), power)
    except OverflowError:
        return sign(x) * inf

def type_check(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Optional[Union[Callable[[float], float], Callable[[float, float], float]]],
    power: float,
    x: Optional[float],
    y1: Optional[float],
    y2: Optional[float],
    abs_err: float,
    rel_err: float,
    abs_tol: Optional[float],
    rel_tol: Optional[float],
    method: Method,
    /,
) -> Optional[Union[TypeError, ValueError]]:
    """Performs input checks and potentially returns an error."""
    if not callable(f):
        return TypeError(f"expected a function for f, got {f!r}")
    elif not isinstance(x1, SupportsFloat):
        return TypeError(f"could not interpret x1 as a real number, got {x1!r}")
    elif isnan(float(x1)):
        return ValueError(f"x1 is not a number")
    elif not isinstance(x2, SupportsFloat):
        return TypeError(f"could not interpret x2 as a real number, got {x2!r}")
    elif isnan(float(x2)):
        return ValueError("x2 is not a number")
    elif float(x1) == float(x2):
        return ValueError("x1 == x2")
    elif method not in methods:
        return TypeError(f"invalid method given, got {method!r}")
    elif fprime is not None and method not in (*ode_methods, "newt-safe"):
        return TypeError(f"got unexpected argument 'fprime' for method {method!r}")
    elif fprime is not None and method in (*ode_methods, "newt-safe") and not callable(fprime):
        return TypeError(f"expected a function for fprime, got {fprime!r}")
    elif not isinstance(power, SupportsFloat):
        return TypeError(f"could not interpret power as a real number, got {power!r}")
    elif float(power) == 0:
        return ValueError(f"power must be non-zero")
    elif x is not None and not isinstance(x, SupportsFloat):
        return TypeError(f"x was given but could not be interpreted as a float, got {x!r}")
    elif x is not None and isnan(float(x)):
        return ValueError("x is not a number")
    elif y1 is not None and not isinstance(y1, SupportsFloat):
        return TypeError(f"y1 was given but could not be interpreted as a float, got {y1!r}")
    elif y1 is not None and isnan(float(y1)):
        return ValueError("y1 is not a number")
    elif y2 is not None and not isinstance(y2, SupportsFloat):
        return TypeError(f"y2 was given but could not be interpreted as a float, got {y2!r}")
    elif y2 is not None and isnan(float(y2)):
        return ValueError("y2 is not a number")
    elif not isinstance(abs_err, SupportsFloat):
        return TypeError(f"abs_err could not be interpreted as a float, got {abs_err!r}")
    elif isnan(float(abs_err)):
        return ValueError("abs_err is not a number")
    elif not isinstance(rel_err, SupportsFloat):
        return TypeError(f"rel_err could not be interpreted as a float, got {rel_err!r}")
    elif isnan(float(rel_err)):
        return ValueError("rel_err is not a number")
    elif abs_tol is not None and not isinstance(abs_tol, SupportsFloat):
        return TypeError(f"abs_tol was given but could not be interpreted as a float, got {abs_tol!r}")
    elif abs_tol is not None and isnan(float(abs_tol)):
        return ValueError("abs_tol is not a number")
    elif rel_tol is not None and not isinstance(rel_tol, SupportsFloat):
        return TypeError(f"rel_tol was given but could not be interpreted as a float, got {rel_tol!r}")
    elif rel_tol is not None and isnan(float(rel_tol)):
        return ValueError("rel_tol is not a number")
    else:
        return None

def bisect_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
    """
    The bisection method ensures a very robust worst-case scenario.

    Does not converge fast for the best-case scenario. Other methods
    are recommended over the bisection method. For non-simple roots,
    use the 'non-simple' method instead.

    Order of Convergence:
        1:
            Linear convergence.

    See also:
        'non-simple':
            Fast convergence when f(x) ~ C * (x - root) ^ power.
    """
    # Use an initial estimate.
    if x is not None and abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    # Initial convergence to {-1, 0, 1}.
    x = (sign(x1) + sign(x2)) / 2
    while (
        is_between(x1, x, x2)
        and not is_between(x1 / 8, x2, x1 * 8)
        and not (sign(x1) == sign(x2) and is_between(sqrt(abs(x1)), abs(x2), x1 * x1))
        and abs(x1 - x2) > abs_err + rel_err * abs(x2)
        and y2 != 0
    ):
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
        x = (sign(x1) + sign(x2)) / 2
    # Log-mean convergence.
    x_sign = sign(x1)
    x_abs = 1 if abs(x1) >= 1 else -1
    while (
        not is_between(x1 / 8, x2, x1 * 8)
        and not (sign(x1) == sign(x2) and is_between(sqrt(abs(x1)), abs(x2), x1 * x1))
        and abs(x1 - x2) > abs_err + rel_err * abs(x2)
        and y2 != 0
    ):
        x = x_sign * exp(x_abs * sqrt(abs(log(abs(x1)))) * sqrt(abs(log(abs(x2)))))
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    # Geometric-mean convergence.
    while (
        not is_between(x1 / 8, x2, x1 * 8)
        and abs(x1 - x2) > abs_err + rel_err * abs(x2)
        and y2 != 0
    ):
        x = x_sign * sqrt(abs(x1)) * sqrt(abs(x2))
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    # Arithmetic-mean convergence.
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        x = x1 + 0.5 * (x2 - x1)
        y = f(x)
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    return secant(x1, x2, y1, y2)

def bisect_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[float]:
    """
    The bisection method ensures a very robust worst-case scenario.

    Does not converge fast for the best-case scenario. Other methods
    are recommended over the bisection method. For non-simple roots,
    use the 'non-simple' method instead.

    Order of Convergence:
        1:
            Linear convergence.

    See also:
        'non-simple':
            Fast convergence when f(x) ~ C * (x - root) ^ power.
    """
    yield x1
    yield x2
    # Use an initial estimate.
    if x is not None and abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    # Initial convergence to {-1, 0, 1}.
    x = (sign(x1) + sign(x2)) / 2
    while (
        is_between(x1, x, x2)
        and not is_between(x1 / 8, x2, x1 * 8)
        and not (sign(x1) == sign(x2) and is_between(sqrt(abs(x1)), abs(x2), x1 * x1))
        and abs(x1 - x2) > abs_err + rel_err * abs(x2)
        and y2 != 0
    ):
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
        x = (sign(x1) + sign(x2)) / 2
    # Log-mean convergence.
    x_sign = sign(x1)
    x_abs = 1 if abs(x1) >= 1 else -1
    while (
        not is_between(x1 / 8, x2, x1 * 8)
        and not (sign(x1) == sign(x2) and is_between(sqrt(abs(x1)), abs(x2), x1 * x1))
        and abs(x1 - x2) > abs_err + rel_err * abs(x2)
        and y2 != 0
    ):
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x = x_sign * exp(x_abs * sqrt(abs(log(abs(x1)))) * sqrt(abs(log(abs(x2)))))
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    # Geometric-mean convergence.
    while (
        not is_between(x1 / 8, x2, x1 * 8)
        and abs(x1 - x2) > abs_err + rel_err * abs(x2)
        and y2 != 0
    ):
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x = x_sign * sqrt(abs(x1)) * sqrt(abs(x2))
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    # Arithmetic-mean convergence.
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        x = x1 + 0.5 * (x2 - x1)
        y = f(x)
        yield x
        if sign(y) == sign(y1):
            x1 = x2
            y1 = y2
        x2 = x
        y2 = y
    yield secant(x1, x2, y1, y2)

def chandrupatla_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
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
        'secant':
            A robust algorithm which ensures fast and tight bracketing
            for simple roots.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    x5 = x4 = x3 = x2
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = float_mean(x1, x2)
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use IQI if the points are highly linear.
        elif y_ratio * y_ratio < x_ratio < 1 - (1 - y_ratio) ** 2:
            a = y  / (y1 - y )
            b = y2 / (y1 - y2)
            c = y  / (y2 - y )
            d = y1 / (y2 - y1)
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x, x2 = x + a * b * (x1 - x) + c * d * (x2 - x), x
            y2 = y
            # Perform adjustment if convergence fails at least twice in a row.
            if bisect_fails >= 2:
                # High-order convergence for simple roots: ~ 1.82.
                x += (x - x2) * abs((x - x1) * ((x - x3) / (x - x5)))
                # Back-up Illinois method.
                x4 = secant(x1, x2, y1 / 2, y2)
                # Fall-back to the Illinois method if not properly converging.
                if not is_between(x4, x, x2):
                    y1 /= 2
                    x = x4
                    used_illinois = True
        # Use the secant method with x and x2.
        elif abs(x2 - x) < 1.25 * abs(x1 - x) and abs(y) < abs(y2) < 1.25 * abs(y1) and is_between(mean(x1, x), secant(x, x2, y, y2), x):
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2, x = x, secant(x, x2, y, y2)
            y2 = y
        # Use the secant method with x and x1.
        else:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    return secant(x1, x2, y1, y2)

def chandrupatla_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
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
        'secant':
            A robust algorithm which ensures fast and tight bracketing
            for simple roots.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    x5 = x4 = x3 = x2
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = float_mean(x1, x2)
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use IQI if the points are highly linear.
        elif y_ratio * y_ratio < x_ratio < 1 - (1 - y_ratio) ** 2:
            a = y  / (y1 - y )
            b = y2 / (y1 - y2)
            c = y  / (y2 - y )
            d = y1 / (y2 - y1)
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x, x2 = x + a * b * (x1 - x) + c * d * (x2 - x), x
            y2 = y
            # Perform adjustment if convergence fails at least twice in a row.
            if bisect_fails >= 2:
                # High-order convergence for simple roots: ~ 1.82.
                x += (x - x2) * abs((x - x1) * ((x - x3) / (x - x5)))
                # Back-up Illinois method.
                x4 = secant(x1, x2, y1 / 2, y2)
                # Fall-back to the Illinois method if not properly converging.
                if not is_between(x4, x, x2):
                    y1 /= 2
                    x = x4
                    used_illinois = True
        # Use the secant method with x and x2.
        elif abs(x2 - x) < 1.25 * abs(x1 - x) and abs(y) < abs(y2) < 1.25 * abs(y1) and is_between(mean(x1, x), secant(x, x2, y, y2), x):
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2, x = x, secant(x, x2, y, y2)
            y2 = y
        # Use the secant method with x and x1.
        else:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    yield secant(x1, x2, y1, y2)

def heun_ode_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
    """
    Heun's ODE method is the ODE equivalent of the trapezoidal method
    for numerical integration.

    Heun's ODE method uses 2 separate `fprime` evaluations per
    iteration to produce a more accurate estimate of the root compared
    to the 'newt-ode' method. The additional `fprime` evaluation also
    makes Heun's ODE method more robust than the 'newt-ode' method,
    giving it more accurate estimates during initial estimates. Similar
    to the 'newt-safe' method, robust measures are taken to ensure
    tight brackets and worst-case convergence.

    Order of Convergence:
        3:
            Cubic convergence, similar to Halley's method.

    See also:
        'midpoint-ode':
            Uses fewer `fprime` calls per iteration at the cost of a
            reduced order of convergence. Recommended if `fprime` calls
            are relatively expensive compared to `f` calls.
        'newt-ode':
            Uses fewer `fprime` calls per iteration at the cost of a
            reduced order of convergence. Also has less arithmetic
            cost compared to other methods. Recommended if the
            cost of the algorithm itself is significantly more
            expensive than function calls.
        'rk45-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use Huen's ODE method.
        else:
            yprime = fprime(x, y)
            if yprime == 0:
                k1 = inf
            else:
                k1 = y / yprime
            if not is_between(x1, x - k1, x2):
                x = secant(x1, x2, y1, y2)
                continue
            yprime = fprime(x - k1, 0)
            if yprime == 0:
                k2 = inf
            else:
                k2 = y / yprime
            x -= (k1 + k2) / 2
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    return secant(x1, x2, y1, y2)

def heun_ode_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[float]:
    """
    Heun's ODE method is the ODE equivalent of the trapezoidal method
    for numerical integration.

    Heun's ODE method uses 2 separate `fprime` evaluations per
    iteration to produce a more accurate estimate of the root compared
    to the 'newt-ode' method. The additional `fprime` evaluation also
    makes Heun's ODE method more robust than the 'newt-ode' method,
    giving it more accurate estimates during initial estimates. Similar
    to the 'newt-safe' method, robust measures are taken to ensure
    tight brackets and worst-case convergence.

    Order of Convergence:
        3:
            Cubic convergence, similar to Halley's method.

    See also:
        'midpoint-ode':
            Uses fewer `fprime` calls per iteration at the cost of a
            reduced order of convergence. Recommended if `fprime` calls
            are relatively expensive compared to `f` calls.
        'newt-ode':
            Uses fewer `fprime` calls per iteration at the cost of a
            reduced order of convergence. Also has less arithmetic
            cost compared to other methods. Recommended if the
            cost of the algorithm itself is significantly more
            expensive than function calls.
        'rk45-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use Huen's ODE method.
        else:
            yprime = fprime(x, y)
            if yprime == 0:
                k1 = inf
            else:
                k1 = y / yprime
            if not is_between(x1, x - k1, x2):
                x = secant(x1, x2, y1, y2)
                continue
            yprime = fprime(x - k1, 0)
            if yprime == 0:
                k2 = inf
            else:
                k2 = y / yprime
            x -= (k1 + k2) / 2
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    yield secant(x1, x2, y1, y2)

def midpoint_ode_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
    """
    The midpoint ODE method is the ODE equivalent of the midpoint
    method for numerical integration.

    The midpoint ODE method uses only 1 `fprime` evaluation per
    iteration, similar to the 'newt-ode' method, but does not use the
    derivative at the current point. Instead, a secant estimate of the
    root is made, and then the derivative is evaluated between that and
    the current estimate i.e. the derivative is evaluated at the
    midpoint. This gives more faster and more robust convergence than
    the 'newt-ode' method.

    Order of Convergence:
        2.414:
            Between quadratic and cubic orders of convergence. Faster
            than the 'newt-ode' method, but slower than the 'heun-ode'
            method.

            The exact value is given by the root of:
                 x^2 - 2x - 1

    See also:
        'heun-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        'newt-ode':
            Uses fewer arithmetic operations per iteration. Recommended
            if the cost of the algorithm itself is significantly more
            expensive than function calls.
        'rk45-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed extremely cheaply compared to `f`.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the midpoint ODE method.
        else:
            if abs(x - x2) < abs(x - x1) and abs(y) < abs(y2):
                k1 = secant(0.0, x - x2, y, y2)
            else:
                k1 = secant(0.0, x - x1, y, y1)
            if not is_between(x1, x - k1 / 2, x2):
                x = secant(x1, x2, y1, y2)
                continue
            yprime = fprime(x - k1 / 2, y / 2)
            if yprime == 0:
                k2 = inf
            else:
                k2 = y / yprime
            x -= k2
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    return secant(x1, x2, y1, y2)

def midpoint_ode_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[float]:
    """
    The midpoint ODE method is the ODE equivalent of the midpoint
    method for numerical integration.

    The midpoint ODE method uses only 1 `fprime` evaluation per
    iteration, similar to the 'newt-ode' method, but does not use the
    derivative at the current point. Instead, a secant estimate of the
    root is made, and then the derivative is evaluated between that and
    the current estimate i.e. the derivative is evaluated at the
    midpoint. This gives more faster and more robust convergence than
    the 'newt-ode' method.

    Order of Convergence:
        2.414:
            Between quadratic and cubic orders of convergence. Faster
            than the 'newt-ode' method, but slower than the 'heun-ode'
            method.

            The exact value is given by the root of:
                 x^2 - 2x - 1

    See also:
        'heun-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        'newt-ode':
            Uses fewer arithmetic operations per iteration. Recommended
            if the cost of the algorithm itself is significantly more
            expensive than function calls.
        'rk45-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed extremely cheaply compared to `f`.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x2 = x
            y2 = y
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            x2 = x
            y2 = y
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the midpoint ODE method.
        else:
            if abs(x - x2) < abs(x - x1) and abs(y) < abs(y2):
                k1 = secant(0.0, x - x2, y, y2)
            else:
                k1 = secant(0.0, x - x1, y, y1)
            if not is_between(x1, x - k1 / 2, x2):
                x = secant(x1, x2, y1, y2)
                continue
            yprime = fprime(x - k1 / 2, y / 2)
            if yprime == 0:
                k2 = inf
            else:
                k2 = y / yprime
            x -= k2
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    yield secant(x1, x2, y1, y2)

def newt_ode_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
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
        'heun-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        'midpoint-ode':
            Uses more arithmetic operations per iteration to gain
            increased order of convergence and robustness. Recommended
            if function calls are somewhat expensive.
        'rk45-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed extremely cheaply compared to `f`.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the Newton-Raphson method.
        else:
            yprime = fprime(x, y)
            if yprime == 0:
                x = inf
            else:
                x -= y / yprime
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    return secant(x1, x2, y1, y2)

def newt_ode_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
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
        'heun-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        'midpoint-ode':
            Uses more arithmetic operations per iteration to gain
            increased order of convergence and robustness. Recommended
            if function calls are somewhat expensive.
        'rk45-ode':
            Uses more `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed extremely cheaply compared to `f`.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the Newton-Raphson method.
        else:
            yprime = fprime(x, y)
            if yprime == 0:
                x = inf
            else:
                x -= y / yprime
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    yield secant(x1, x2, y1, y2)

def newt_safe_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
    """
    The Newton-Raphson method is similar to the secant method but uses
    a given derivative instead of estimating it.

    Unlike the ODE methods, `fprime` only takes `x` as an argument.

    Order of Convergence:
        2:
            Quadratic convergence.

    See also:
        'newt-ode':
            The ODE equivalent. Contains more information. Recommended
            if `fprime(x)` can be computed more efficiently if `f(x)` is
            given.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the Newton-Raphson method.
        else:
            yprime = fprime(x)
            if yprime == 0:
                x = inf
            else:
                x -= y / yprime
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    return secant(x1, x2, y1, y2)

def newt_safe_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[float]:
    """
    The Newton-Raphson method is similar to the secant method but uses
    a given derivative instead of estimating it.

    Unlike the ODE methods, `fprime` only takes `x` as an argument.

    Order of Convergence:
        2:
            Quadratic convergence.

    See also:
        'newt-ode':
            The ODE equivalent. Contains more information. Recommended
            if `fprime(x)` can be computed more efficiently if `f(x)` is
            given.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the Newton-Raphson method.
        else:
            yprime = fprime(x)
            if yprime == 0:
                x = inf
            else:
                x -= y / yprime
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    yield secant(x1, x2, y1, y2)

def nonsimple_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    power: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
    """
    The non-simple method excels at finding non-simple roots.

    Non-simple roots occur when `|f(x)| ~ C * |x - root| ^ power` when
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
        'secant':
            Equivalent method for simple roots.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Estimate the order of the root.
        power = power_estimate(x1, x, x2, y1, y, y2, power)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        y_min, _, y_max = sorted([abs(y), abs(y1), abs(y2)])
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            if 0.6 < power < 1.4:
                numerator = y / y_max - y1 / y_max
                denominator = y2 / y_max - y1 / y_max
            elif power > 0:
                numerator = signed_pow(y / y_max, power) - signed_pow(y1 / y_max, power)
                denominator = signed_pow(y2 / y_max, power) - signed_pow(y1 / y_max, power)
            else:
                numerator = signed_pow(y / y_min, power) - signed_pow(y1 / y_min, power)
                denominator = signed_pow(y2 / y_min, power) - signed_pow(y1 / y_min, power)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            if 0.6 < power < 1.4:
                numerator = y / y_max - y2 / y_max
                denominator = y1 / y_max - y2 / y_max
            elif power > 0:
                numerator = signed_pow(y / y_max, power) - signed_pow(y2 / y_max, power)
                denominator = signed_pow(y1 / y_max, power) - signed_pow(y2 / y_max, power)
            else:
                numerator = signed_pow(y / y_min, power) - signed_pow(y2 / y_min, power)
                denominator = signed_pow(y1 / y_min, power) - signed_pow(y2 / y_min, power)
        y_ratio = numerator / denominator
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x2 = x
            y2 = y
            x = float_mean(x1, x2)
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2, power)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least twice in a row.
        elif bisect_fails >= 2:
            x2 = x
            y2 = y
            y = y1 / 2 ** min(10 * sign(power), 1 / power, key=abs)
            x = secant(x1, x2, y, y2, power)
            used_illinois = True
        # Use the secant method with x and x2.
        elif abs(x2 - x) < 1.25 * abs(x1 - x) and abs(y) < abs(y2) < 1.25 * abs(y1) and is_between(mean(x1, x), secant(x, x2, y, y2), x):
            x2, x = x, secant(x, x2, y, y2, power)
            y2 = y
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2, power)
        # Use the secant method with x and x1.
        else:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2, power)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    if 0.6 < power < 1.4:
        return secant(x1, x2, y1, y2)
    else:
        return secant(x1, x2, y1, y2, power)

def nonsimple_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    power: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[float]:
    """
    The non-simple method excels at finding non-simple roots.

    Non-simple roots occur when `|f(x)| ~ C * |x - root| ^ power` when
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
        'secant':
            Equivalent method for simple roots.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Estimate the order of the root.
        power = power_estimate(x1, x, x2, y1, y, y2, power)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        y_min, _, y_max = sorted([abs(y), abs(y1), abs(y2)])
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            if 0.6 < power < 1.4:
                numerator = y / y_max - y1 / y_max
                denominator = y2 / y_max - y1 / y_max
            elif power > 0:
                numerator = signed_pow(y / y_max, power) - signed_pow(y1 / y_max, power)
                denominator = signed_pow(y2 / y_max, power) - signed_pow(y1 / y_max, power)
            else:
                numerator = signed_pow(y / y_min, power) - signed_pow(y1 / y_min, power)
                denominator = signed_pow(y2 / y_min, power) - signed_pow(y1 / y_min, power)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            if 0.6 < power < 1.4:
                numerator = y / y_max - y2 / y_max
                denominator = y1 / y_max - y2 / y_max
            elif power > 0:
                numerator = signed_pow(y / y_max, power) - signed_pow(y2 / y_max, power)
                denominator = signed_pow(y1 / y_max, power) - signed_pow(y2 / y_max, power)
            else:
                numerator = signed_pow(y / y_min, power) - signed_pow(y2 / y_min, power)
                denominator = signed_pow(y1 / y_min, power) - signed_pow(y2 / y_min, power)
        y_ratio = numerator / denominator
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x2 = x
            y2 = y
            x = float_mean(x1, x2)
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2, power)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least twice in a row.
        elif bisect_fails >= 2:
            x2 = x
            y2 = y
            y = y1 / 2 ** min(10 * sign(power), 1 / power, key=abs)
            x = secant(x1, x2, y, y2, power)
            used_illinois = True
        # Use the secant method with x and x2.
        elif abs(x2 - x) < 1.25 * abs(x1 - x) and abs(y) < abs(y2) < 1.25 * abs(y1) and is_between(mean(x1, x), secant(x, x2, y, y2), x):
            x2, x = x, secant(x, x2, y, y2, power)
            y2 = y
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2, power)
        # Use the secant method with x and x1.
        else:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2, power)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    if 0.6 < power < 1.4:
        yield secant(x1, x2, y1, y2)
    else:
        yield secant(x1, x2, y1, y2, power)

def rk45_ode_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
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
        'heun-ode':
            Uses fewer `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        'midpoint-ode':
            Uses significantly fewer `fprime` calls per iteration, but
            more arithmetic operations to compensate the order of
            convergence and its robustness. Recommended if function
            calls are somewhat expensive.
        'newt-ode':
            Uses significantly fewer `fprime` calls per iteration as
            well as fewer arithmetic operations per iteration.
            Recommended if the cost of the algorithm itself is
            significantly more expensive than function calls.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the RK45 ODE method.
        else:
            k1 = y / fprime(x, y)
            k2 = y / fprime(x - k1 / 2, y / 2)
            k3 = y / fprime(x - k2 / 2, y / 2)
            k4 = y / fprime(x - k3, 0)
            x -= (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    return secant(x1, x2, y1, y2)

def rk45_ode_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    fprime: Callable[[float, float], float],
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
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
        'heun-ode':
            Uses fewer `fprime` calls per iteration to gain increased
            order of convergence. Recommended if `fprime(x, y)` can
            be computed relatively cheaply compared to `f`.
        'midpoint-ode':
            Uses significantly fewer `fprime` calls per iteration, but
            more arithmetic operations to compensate the order of
            convergence and its robustness. Recommended if function
            calls are somewhat expensive.
        'newt-ode':
            Uses significantly fewer `fprime` calls per iteration as
            well as fewer arithmetic operations per iteration.
            Recommended if the cost of the algorithm itself is
            significantly more expensive than function calls.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        x2 = x
        y2 = y
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the RK45 ODE method.
        else:
            k1 = y / fprime(x, y)
            k2 = y / fprime(x - k1 / 2, y / 2)
            k3 = y / fprime(x - k2 / 2, y / 2)
            k4 = y / fprime(x - k3, 0)
            x -= (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Fall-back to the secant method if convergence fails.
            if not is_between(x1, x, x2):
                x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    yield secant(x1, x2, y1, y2)

def secant_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> float:
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
        'chandrupatla':
            Uses higher order interpolation, obtaining a higher order
            of convergence at the cost of a bit more arithmetic
            operations per iteration.
        'newt-safe':
            Uses a given derivative instead of approximating the
            derivative using finite differences.
    """
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x2 = x
            y2 = y
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least twice in a row.
        elif bisect_fails >= 2:
            x2 = x
            y2 = y
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the secant method with x and x2.
        elif abs(x2 - x) < 1.25 * abs(x1 - x) and abs(y) < abs(y2) < 1.25 * abs(y1) and is_between(mean(x1, x), secant(x, x2, y, y2), x):
            x2, x = x, secant(x, x2, y, y2)
            y2 = y
        # Use the secant method with x and x1.
        else:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    return secant(x1, x2, y1, y2)

def secant_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
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
        'chandrupatla':
            Uses higher order interpolation, obtaining a higher order
            of convergence at the cost of a bit more arithmetic
            operations per iteration.
        'newt-safe':
            Uses a given derivative instead of approximating the
            derivative using finite differences.
    """
    yield x1
    yield x2
    # Generate initial point using the Bisection method.
    if x is None:
        x = mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if sign(y) == sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        # Track how many times convergence fails in a row before using the Illinois method.
        if abs(x - x1) > 0.75 * abs(x1 - x2):
            bisect_fails += 1
        else:
            bisect_fails = 0
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        # Use the bisection method if not linear-ish.
        if not x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio:
            x2 = x
            y2 = y
            x = float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
            dx = mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least twice in a row.
        elif bisect_fails >= 2:
            x2 = x
            y2 = y
            y1 /= 2
            x = secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the secant method with x and x2.
        elif abs(x2 - x) < 1.25 * abs(x1 - x) and abs(y) < abs(y2) < 1.25 * abs(y1) and is_between(mean(x1, x), secant(x, x2, y, y2), x):
            x2, x = x, secant(x, x2, y, y2)
            y2 = y
        # Use the secant method with x and x1.
        else:
            x2 = x
            y2 = y
            x = secant(x1, x2, y1, y2)
        if x1 < x2:
            if x < x1:
                x = x1
        else:
            if x > x2:
                x = x2
    # Use the secant method on the final iteration for high precision.
    yield secant(x1, x2, y1, y2)

@overload
def root_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: DerivativeFreeMethod = ...,
) -> float:
    ...

@overload
def root_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float, float], float],
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: ODEMethod,
) -> float:
    ...

@overload
def root_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float], float],
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: Literal["newt-safe"],
) -> float:
    ...

@overload
def root_in(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    power: float = ...,
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: Literal["non-simple"],
) -> float:
    ...

def root_in(
    f,
    x1,
    x2,
    /,
    fprime=None,
    power=1.0,
    *,
    x=None,
    y1=None,
    y2=None,
    abs_err=0.0,
    rel_err=32 * FLOAT_EPSILON,
    abs_tol=None,
    rel_tol=None,
    method="chandrupatla",
):
    """
    Finds a bracketed root of a function.

    Bracketing methods allow roots of a function to be found quickly
    provided two initial points where one yields a negative value and
    the other yields a positive value.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) == -sign(f(x2))`.
        fprime:
            The derivative of `f` for 'newt-safe' or an ODE method.
        power, default 1:
            An estimate of the order of the root for 'non-simple'.
        x:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol:
            An initial tolerance to help jump-start an initially tight
            bracket.
        method, default 'chandrupatla':
            The method used to find the root. Different methods have
            different advantages and disadvantages, which should be
            considered. The default method is highly robust.

    Returns:
        x:
            The estimate of the root.
    """
    exception = type_check(f, x1, x2, fprime, power, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol, method)
    if exception is not None:
        raise exception
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = sign(x1) * FLOAT_MAX
    if isinf(x2):
        x2 = sign(x2) * FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    else:
        y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    else:
        y2 = float(y2)
    if sign(y1) != -sign(y2):
        raise ValueError("sign(f(x1)) is not the opposite of sign(f(x2))")
    if abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), FLOAT_MIN)
    rel_err = max(float(rel_err), 32 * FLOAT_EPSILON)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * FLOAT_EPSILON)
    else:
        rel_tol = max(float(rel_tol), rel_err)
    if abs_tol is None:
        abs_tol = rel_tol * min(1, abs(x1 - x2))
    else:
        abs_tol = max(float(abs_tol), abs_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    if method == "bisect":
        return bisect_in(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "chandrupatla":
        return chandrupatla_in(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "heun-ode":
        return heun_ode_in(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "midpoint-ode":
        return midpoint_ode_in(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "newt-safe":
        return newt_ode_in(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "newt-safe":
        return newt_safe_in(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "non-simple":
        return nonsimple_in(f, x1, x2, 1 / float(power), x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "rk45-ode":
        return rk45_ode_in(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "secant":
        return secant_in(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    else:
        assert False, f"Missing method {method!r}"

@overload
def root_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: DerivativeFreeMethod = ...,
) -> Iterator[float]:
    ...

@overload
def root_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float, float], float],
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: ODEMethod,
) -> Iterator[float]:
    ...

@overload
def root_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    fprime: Callable[[float], float],
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: Literal["newt-safe"],
) -> Iterator[float]:
    ...

@overload
def root_iter(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    /,
    power: float = ...,
    *,
    x: Optional[float] = ...,
    y1: Optional[float] = ...,
    y2: Optional[float] = ...,
    abs_err: float = ...,
    rel_err: float = ...,
    abs_tol: Optional[float] = ...,
    rel_tol: Optional[float] = ...,
    method: Literal["non-simple"],
) -> Iterator[float]:
    ...

def root_iter(
    f,
    x1,
    x2,
    /,
    fprime=None,
    power=1.0,
    *,
    x=None,
    y1=None,
    y2=None,
    abs_err=0.0,
    rel_err=32 * FLOAT_EPSILON,
    abs_tol=None,
    rel_tol=None,
    method="chandrupatla",
):
    """
    Finds a bracketed root of a function.

    Bracketing methods allow roots of a function to be found quickly
    provided two initial points where one yields a negative value and
    the other yields a positive value.

    Parameters:
        f:
            The objective function for which the `x` in `f(x) = 0` is
            solved for.
        x1, x2:
            Two initial points where `sign(f(x1)) == -sign(f(x2))`.
        fprime:
            The derivative of `f` for 'newt-safe' or an ODE method.
        power, default 1:
            An estimate of the order of the root for 'non-simple'.
        x:
            An initial estimate of the root which can be used to
            jump-start the initial convergence.
        y1, y2:
            Pre-computed initial values for `f(x1)` and `f(x2)`.
        abs_err, default 0:
            The desired absolute error.
        rel_err, default 32 * FLOAT_EPSILON:
            The desired relative error, default near machine precision.
        abs_tol, rel_tol:
            An initial tolerance to help jump-start an initially tight
            bracket.
        method, default 'chandrupatla':
            The method used to find the root. Different methods have
            different advantages and disadvantages, which should be
            considered. The default method is highly robust.

    Yields:
        x:
            The current estimate of the root. Use `functools.lru_cache`
            to cache the most recent `f(x)` evaluation, if necessary.

    Note:
        The final iteration of `f(x)` is not evaluated by pyroot.
    """
    exception = type_check(f, x1, x2, fprime, power, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol, method)
    if exception is not None:
        raise exception
    x1 = float(x1)
    x2 = float(x2)
    if x is not None:
        x = float(x)
        if not is_between(x1, x, x2):
            x = None
    if isinf(x1):
        x1 = sign(x1) * FLOAT_MAX
    if isinf(x2):
        x2 = sign(x2) * FLOAT_MAX
    if y1 is None:
        y1 = f(x1)
    else:
        y1 = float(y1)
    if y2 is None:
        y2 = f(x2)
    else:
        y2 = float(y2)
    if sign(y1) != -sign(y2):
        raise ValueError("sign(f(x1)) is not the opposite of sign(f(x2))")
    if abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    abs_err = max(float(abs_err), FLOAT_MIN)
    rel_err = max(float(rel_err), 32 * FLOAT_EPSILON)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * FLOAT_EPSILON)
    else:
        rel_tol = max(float(rel_tol), rel_err)
    if abs_tol is None:
        abs_tol = rel_tol * min(1, abs(x1 - x2))
    else:
        abs_tol = max(float(abs_tol), abs_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    if method == "bisect":
        return bisect_iter(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "chandrupatla":
        return chandrupatla_iter(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "heun-ode":
        return heun_ode_iter(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "midpoint-ode":
        return midpoint_ode_iter(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "newt-safe":
        return newt_ode_iter(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "newt-safe":
        return newt_safe_iter(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "non-simple":
        return nonsimple_iter(f, x1, x2, 1 / float(power), x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "rk45-ode":
        return rk45_ode_iter(f, x1, x2, fprime, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    elif method == "secant":
        return secant_iter(f, x1, x2, x, y1, y2, abs_err, rel_err, abs_tol, rel_tol)
    else:
        assert False, f"Missing method {method!r}"
