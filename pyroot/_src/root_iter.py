from heapq import heappop, heappush
from math import log, sqrt
from typing import Callable, Iterable, Iterator, Optional, Sequence, List as list

from pyroot import _utils

__all__ = [
    "bisection",
    "chandrupatla",
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
    x: Optional[float],
    y1: float,
    y2: float,
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[float]:
    if y2 == 0:
        yield x2
        return
    elif abs(x1 - x2) < abs_err + rel_err * abs(x2):
        yield _utils.secant(x1, x2, y1, y2)
        return
    elif y2 < 0 < y1:
        x1, x2, y1, y2 = x2, x1, y2, y1
    if x is not None:
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if y < 0:
            x1 = x
            y1 = y
        elif y > 0:
            x2 = x
            y2 = y
        else:
            return
    x = 0.5 * (_utils.sign(x1) + _utils.sign(x2))
    while (
        _utils.is_between(x1, x, x2)
        and not _utils.is_between(x1 / 8, x2, x1 * 8)
        and not (_utils.sign(x1) == _utils.sign(x2) and _utils.is_between(sqrt(abs(x1)), abs(x2), x1 * x1))
        and abs(x1 - x2) > abs_err + rel_err * abs(x)
    ):
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_err = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if y < 0:
            x1 = x
            y1 = y
        elif y > 0:
            x2 = x
            y2 = y
        else:
            return
        x = 0.5 * (_utils.sign(x1) + _utils.sign(x2))
    x_sign = 1 if x > 0 else -1
    x_abs = 1 if abs(x1) > 1 else -1
    while (
        not _utils.is_between(x1 / 8, x2, x1 * 8)
        and not (_utils.sign(x1) == _utils.sign(x2) and _utils.is_between(sqrt(abs(x1)), abs(x2), x1 * x1))
        and abs(x1 - x2) > abs_err + rel_err * abs(x)
    ):
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_err = rel_err
        x = x_sign * exp(x_abs * sqrt(log(abs(x1)) * log(abs(x2))))
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if y < 0:
            x1 = x
            y1 = y
        elif y > 0:
            x2 = x
            y2 = y
        else:
            return
    while (
        not _utils.is_between(x1 / 8, x2, x1 * 8)
        and abs(x1 - x2) > abs_err + rel_err * abs(x)
    ):
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_err = rel_err
        x = x_sign * sqrt(abs(x1)) * sqrt(abs(x2))
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        if y < 0:
            x1 = x
            y1 = y
        elif y > 0:
            x2 = x
            y2 = y
        else:
            return
    while abs(x1 - x2) > abs_err + rel_err * abs(x):
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_err = rel_err
        x = x1 + 0.5 * (x2 - x1)
        y = f(x)
        yield x
        if y < 0:
            x1 = x
            y1 = y
        elif y > 0:
            x2 = x
            y2 = y
        else:
            return
    yield _utils.secant(x1, x2, y1, y2)

def chandrupatla(
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
    yield x1
    yield x2
    if x is None:
        x = _utils.mean(x1, x2)
    x5 = x4 = x3 = x2
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = _utils.float_mean(x1, x2)
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = _utils.secant(x1, x2, y1, y2)
            dx = _utils.mean(x1, x2) - x
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
            x = _utils.secant(x1, x2, y1, y2)
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
                x4 = _utils.secant(x1, x2, y1 / 2, y2)
                # Fall-back to the Illinois method if not properly converging.
                if not _utils.is_between(x4, x, x2):
                    y1 /= 2
                    x = x4
                    used_illinois = True
        # Use the secant method with x and x2.
        elif (
            abs(x2 - x) < 1.25 * abs(x1 - x)
            and abs(y) < abs(y2) < 1.25 * abs(y1)
            and _utils.is_between(_utils.mean(x1, x), _utils.secant(x, x2, y, y2), x)
        ):
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2, x = x, _utils.secant(x, x2, y, y2)
            y2 = y
        # Use the secant method with x and x1.
        else:
            if abs(x - x2) < abs(x - x3):
                x5 = x4
                x4 = x3
                x3 = x2
            x2 = x
            y2 = y
            x = _utils.secant(x1, x2, y1, y2)
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def heun_ode(
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
    yield x1
    yield x2
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x = _utils.float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = _utils.secant(x1, x2, y1, y2)
            dx = _utils.mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = _utils.secant(x1, x2, y1, y2)
            used_illinois = True
        # Use Huen's ODE method.
        else:
            yprime = fprime(x, y)
            if yprime == 0:
                k1 = inf
            else:
                k1 = y / yprime
            if not _utils.is_between(x1, x - k1, x2):
                x = _utils.secant(x1, x2, y1, y2)
                continue
            yprime = fprime(x - k1, 0)
            if yprime == 0:
                k2 = inf
            else:
                k2 = y / yprime
            x -= (k1 + k2) / 2
            # Fall-back to the secant method if convergence fails.
            if not _utils.is_between(x1, x, x2):
                x = _utils.secant(x1, x2, y1, y2)
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def midpoint_ode(
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
    yield x1
    yield x2
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x = _utils.float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = _utils.secant(x1, x2, y1, y2)
            dx = _utils.mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = _utils.secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the midpoint ODE method.
        else:
            if abs(x - x2) < abs(x - x1) and abs(y) < abs(y2):
                k1 = _utils.secant(0.0, x - x2, y, y2)
            else:
                k1 = _utils.secant(0.0, x - x1, y, y1)
            if not _utils.is_between(x1, x - k1 / 2, x2):
                x = _utils.secant(x1, x2, y1, y2)
                continue
            yprime = fprime(x - k1 / 2, y / 2)
            if yprime == 0:
                k2 = inf
            else:
                k2 = y / yprime
            x -= k2
            # Fall-back to the secant method if convergence fails.
            if not _utils.is_between(x1, x, x2):
                x = _utils.secant(x1, x2, y1, y2)
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def newton(
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
    yield x1
    yield x2
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x = _utils.float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = _utils.secant(x1, x2, y1, y2)
            dx = _utils.mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = _utils.secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the Newton-Raphson method.
        else:
            yprime = fprime(x)
            if yprime == 0:
                x = inf
            else:
                x -= y / yprime
            # Fall-back to the secant method if convergence fails.
            if not _utils.is_between(x1, x, x2):
                x = _utils.secant(x1, x2, y1, y2)
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def newton_ode(
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
    yield x1
    yield x2
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x = _utils.float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = _utils.secant(x1, x2, y1, y2)
            dx = _utils.mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = _utils.secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the Newton-Raphson method.
        else:
            yprime = fprime(x, y)
            if yprime == 0:
                x = inf
            else:
                x -= y / yprime
            # Fall-back to the secant method if convergence fails.
            if not _utils.is_between(x1, x, x2):
                x = _utils.secant(x1, x2, y1, y2)
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def non_simple(
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
    yield x1
    yield x2
    power = 1.0
    power = 1.0
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Estimate the order of the root.
        power = _utils.power_estimate(x1, x, x2, y1, y, y2, power)
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
                numerator = _utils.signed_pow(y / y_max, power) - _utils.signed_pow(y1 / y_max, power)
                denominator = _utils.signed_pow(y2 / y_max, power) - _utils.signed_pow(y1 / y_max, power)
            else:
                numerator = _utils.signed_pow(y / y_min, power) - _utils.signed_pow(y1 / y_min, power)
                denominator = _utils.signed_pow(y2 / y_min, power) - _utils.signed_pow(y1 / y_min, power)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            if 0.6 < power < 1.4:
                numerator = y / y_max - y2 / y_max
                denominator = y1 / y_max - y2 / y_max
            elif power > 0:
                numerator = _utils.signed_pow(y / y_max, power) - _utils.signed_pow(y2 / y_max, power)
                denominator = _utils.signed_pow(y1 / y_max, power) - _utils.signed_pow(y2 / y_max, power)
            else:
                numerator = _utils.signed_pow(y / y_min, power) - _utils.signed_pow(y2 / y_min, power)
                denominator = _utils.signed_pow(y1 / y_min, power) - _utils.signed_pow(y2 / y_min, power)
        y_ratio = numerator / denominator
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x2 = x
            y2 = y
            x = _utils.float_mean(x1, x2)
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x2 = x
            y2 = y
            x = _utils.secant(x1, x2, y1, y2, power)
            dx = _utils.mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least twice in a row.
        elif bisect_fails >= 2:
            x2 = x
            y2 = y
            y = y1 / 2 ** min(10 * _utils.sign(power), 1 / power, key=abs)
            x = _utils.secant(x1, x2, y, y2, power)
            used_illinois = True
        # Use the secant method with x and x2.
        elif (
            abs(x2 - x) < 1.25 * abs(x1 - x)
            and abs(y) < abs(y2) < 1.25 * abs(y1)
            and _utils.is_between(_utils.mean(x1, x), _utils.secant(x, x2, y, y2), x)
        ):
            x2, x = x, _utils.secant(x, x2, y, y2, power)
            y2 = y
            if not _utils.is_between(x1, x, x2):
                x = _utils.secant(x1, x2, y1, y2, power)
        # Use the secant method with x and x1.
        else:
            x2 = x
            y2 = y
            x = _utils.secant(x1, x2, y1, y2, power)
    # Use the secant method on the final iteration for high precision.
    if 0.6 < power < 1.4:
        yield _utils.secant(x1, x2, y1, y2)
    else:
        yield _utils.secant(x1, x2, y1, y2, power)

def rk45_ode(
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
    yield x1
    yield x2
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x = _utils.float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x = _utils.secant(x1, x2, y1, y2)
            dx = _utils.mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least thrice in a row.
        elif bisect_fails >= 3:
            y1 /= 2
            x = _utils.secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the RK45 ODE method.
        else:
            k1 = y / fprime(x, y)
            k2 = y / fprime(x - k1 / 2, y / 2)
            k3 = y / fprime(x - k2 / 2, y / 2)
            k4 = y / fprime(x - k3, 0)
            x -= (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # Fall-back to the secant method if convergence fails.
            if not _utils.is_between(x1, x, x2):
                x = _utils.secant(x1, x2, y1, y2)
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def secant(
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
    yield x1
    yield x2
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    # Track how many times convergence fails in a row before using the Illinois method.
    bisect_fails = 0
    # Track if the Illinois method was used to use the Illinois-Bisection method next.
    used_illinois = False
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
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
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x2 = x
            y2 = y
            x = _utils.float_mean(x1, x2)
            used_illinois = False
        # Follow-up the Illinois method with the Illinois-Bisection method.
        elif used_illinois:
            x2 = x
            y2 = y
            x = _utils.secant(x1, x2, y1, y2)
            dx = _utils.mean(x1, x2) - x
            x += dx * (dx / (x1 - x2)) ** 2
            used_illinois = False
        # Force the Illinois method if convergence fails at least twice in a row.
        elif bisect_fails >= 2:
            x2 = x
            y2 = y
            y1 /= 2
            x = _utils.secant(x1, x2, y1, y2)
            used_illinois = True
        # Use the secant method with x and x2.
        elif (
            abs(x2 - x) < 1.25 * abs(x1 - x)
            and abs(y) < abs(y2) < 1.25 * abs(y1)
            and _utils.is_between(_utils.mean(x1, x), _utils.secant(x, x2, y, y2), x)
        ):
            x2, x = x, _utils.secant(x, x2, y, y2)
            y2 = y
        # Use the secant method with x and x1.
        else:
            x2 = x
            y2 = y
            x = _utils.secant(x1, x2, y1, y2)
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def stochastic(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    deviations: float,
    x: Optional[float],
    y1: Optional[float],
    y2: Optional[float],
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[float]:
    if y1 is None:
        y1 = f(x1)
    j = 1
    rssr = 0.0
    while j < 5 or abs(y1) < deviations * rssr / j:
        j += 1
        dy = f(x1) - y1
        if dy == 0 == rssr:
            pass
        elif rssr < dy * dy * (1 - 1 / j):
            rssr = abs(dy) * sqrt((1 - 1 / j) + (rssr / dy) ** 2)
        else:
            rssr *= sqrt(1 + (dy / rssr) ** 2 * (1 - 1 / j))
        y1 += dy / j
    if y2 is None:
        y2 = f(x2)
    j = 1
    i = 2
    while j < 5 or abs(y2) < deviations * rssr / j:
        j += 1
        dy = f(x2) - y2
        if dy == 0 == rssr:
            pass
        elif rssr < dy * dy * (1 - 1 / j):
            rssr = abs(dy) * sqrt((1 - 1 / j) + (rssr / dy) ** 2)
        else:
            rssr *= sqrt(1 + (dy / rssr) ** 2 * (1 - 1 / j))
        y2 += dy / j
    if _utils.sign(y1) == _utils.sign(y2):
        raise ValueError("f(x1) and f(x2) have the same sign")
    elif abs(y1) < abs(y2):
        x1, x2, y1, y2 = x2, x1, y2, y1
    yield x1
    yield x2
    # Generate initial point using the bisection method.
    if x is None:
        x = _utils.mean(x1, x2)
    while abs(x1 - x2) > abs_err + rel_err * abs(x2) and y2 != 0:
        if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
            abs_tol = abs_err
            rel_tol = rel_err
        x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
        y = f(x)
        yield x
        j = 1
        i += 1
        while abs(y) < deviations * rssr / j:
            j += 1
            dy = f(x) - y
            if dy == 0 == rssr:
                pass
            elif rssr < dy * dy * (1 - 1 / j):
                rssr = abs(dy) * sqrt((1 - 1 / j) + (rssr / dy) ** 2)
            else:
                rssr *= sqrt(1 + (dy / rssr) ** 2 * (1 - 1 / j))
            y += dy / j
        # Estimate linearity.
        if abs(x - x1) < abs(x - x2):
            x_ratio = (x - x1) / (x2 - x1)
            y_ratio = (y - y1) / (y2 - y1)
        else:
            x_ratio = (x - x2) / (x1 - x2)
            y_ratio = (y - y2) / (y1 - y2)
        # Swap points to ensure x replaces x2.
        if _utils.sign(y) == _utils.sign(y1):
            x1, x2, y1, y2 = x2, x1, y2, y1
        dx = abs(x1 - x2)
        x2 = x
        y2 = y
        # Use the bisection method if the interval is very large or the points are not linear-ish.
        if not (
            _utils.is_between(abs(x1) / _utils.FLOAT_EPSILON, abs(x2), abs(x1) * _utils.FLOAT_EPSILON)
            and x_ratio > (x_ratio * x_ratio + y_ratio * y_ratio - 1) / 2 < y_ratio
        ):
            x = _utils.float_mean(x1, x2)
        # Use the secant-bisection method.
        else:
            x = _utils.secant(x1, x2, y1, y2)
            x += (_utils.mean(x1, x2) - x) / i
            if x < _utils.mean(x1, x2) - dx / 4:
                x = _utils.mean(x1, x2) - dx / 4
            elif x > _utils.mean(x1, x2) + dx / 4:
                x = _utils.mean(x1, x2) + dx / 4
    # Use the secant method on the final iteration for high precision.
    yield _utils.secant(x1, x2, y1, y2)

def multivariate_bisection(
    f: Callable[[list[float]], Iterable[float]],
    x1: Sequence[float],
    x2: Sequence[float],
    x: Optional[Sequence[float]],
    y1: Sequence[float],
    y2: Sequence[float],
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[list[float]]:
    heap = [_utils.BoundingBox(x1, x2)]
    heap[0].update(y1)
    heap[0].update(y2)
    if x is None:
        x = [_utils.mean(L, U) for L, U in zip(heap[0]._lower, heap[0]._upper)]
    while heap[0]._n < 1024:
        box = heappop(heap)
        box.update(f(x))
        yield x
        if box.is_bounded():
            for sub_box in box.split():
                heappush(heap, sub_box)
            if all(U - L < abs_err + rel_err * abs(_utils.mean(L, U)) for L, U in zip(sub_box._lower, sub_box._upper)):
                yield [_utils.mean(L, U) for L, U in zip(box._lower, box._upper)]
                print(max(box._n for box in heap))
                return
        else:
            heappush(heap, box)
        x = next(heap[0].points)
    #raise ValueError("no roots found")
