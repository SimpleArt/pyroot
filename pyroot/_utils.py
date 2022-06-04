import math
import random
import sys
from itertools import islice
from math import cos, exp, inf, isnan, log, pi, sqrt
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Sequence, SupportsFloat, TypeVar, Union, List as list, Tuple as tuple

FLOAT_EPSILON: float = 2 * sys.float_info.epsilon
FLOAT_MAX: float = sys.float_info.max
FLOAT_MIN: float = sys.float_info.min
FLOAT_SMALL_EPSILON: float = 2.0 ** -1074

# Constants for estimating the normal cdf.
p: float = 0.3275911
a: list[float] = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]

def cdf(x: float) -> float:
    """Half-precision estimate of the normal cdf for `python <= 3.7`."""
    if x > 0:
        t = 1 / (1 + p * x / sqrt(2))
        return 1 - 0.5 * sum(ai * t ** i for i, ai in enumerate(a)) * t * exp(-0.5 * x * x)
    elif x < 0:
        t = 1 / (1 - p * x / sqrt(2))
        return 0.5 * sum(ai * t ** i for i, ai in enumerate(a)) * t * exp(-0.5 * x * x)
    else:
        return 0.5

def inv_cdf(p: float) -> float:
    """Approximation of the normal inverse cdf for `python <= 3.7`."""
    if p == 0:
        return -inf
    elif p == 0.5:
        return 0.0
    elif p == 0.9:
        return 1.2815519564098403
    elif p == 0.99:
        return 2.326347014821902
    elif p == 0.999:
        return 3.0902525951122355
    elif p == 1:
        return +inf
    elif not 0 < p < 1:
        raise ValueError("domain error")
    elif p > 0.5:
        return -inv_cdf(1 - p)
    x1 = -sqrt(-2 * log(2 * p))
    x2 = sqrt(2 * pi) * (p - 0.5)
    x = -sqrt(-log(2 * p))
    if x2 - x < 0.5 * (x2 - x1):
        x = x2 + 0.5 * (x1 - x2)
    y1 = cdf(x1)
    y2 = cdf(x2)
    while x2 - x1 > 1e-6 * (1 + abs(x)):
        x += 0.25e-6 * (1 + abs(x)) * sign((x1 - x) + (x2 - x))
        y = cdf(x)
        if y > p:
            x2 = x
            y2 = y
        else:
            x1 = x
            y1 = y
        x = ((y2 - p) * x1 - (y1 - p) * x2) / (y2 - y1)
        x += min(0.5, 0.25 * (x2 - x1)) * ((x1 - x) + (x2 - x))
    return ((y2 - p) * x1 - (y1 - p) * x2) / (y2 - y1)

def bits_of(n: int, /) -> Iterator[Literal[0, 1]]:
    while True:
        yield n % 2
        n //= 2

Self = TypeVar("Self", bound="BoundingBox")


class BoundingBox:
    """
    Bounding box class for tracking if a root potentially lies within
    a bounded multivariate region.
    """
    _lower: list[float]
    _n: int
    _points: Iterator[list[float]]
    _signs: Union[tuple[()], list[bool]]
    _upper: list[float]

    __slots__ = {
        "_lower":
            "Lower bounds of the search region.",
        "_n":
            "The number of estimations used so far.",
        "_points":
            "The points being generated in the box.",
        "_signs":
            "Flags for the signs of the output.",
        "_upper":
            "Upper bounds of the search region.",
    }

    def __init__(self: Self, lower: list[float], upper: list[float], /) -> None:
        self._n = 0
        self._lower = lower
        self._signs = ()
        self._upper = upper
        self._points = self.__points()

    def __eq__(self: Self, other: Any, /) -> bool:
        if isinstance(other, BoundingBox):
            return self._n == other._n and sum(self._signs) == sum(other._signs)
        else:
            return NotImplemented

    def __ge__(self: Self, other: Any, /) -> bool:
        if isinstance(other, BoundingBox):
            return self._n > other._n or self._n == other._n and sum(self._signs) <= sum(other._signs)
        else:
            return NotImplemented

    def __gt__(self: Self, other: Any, /) -> bool:
        if isinstance(other, BoundingBox):
            return self._n > other._n or self._n == other._n and sum(self._signs) < sum(other._signs)
        else:
            return NotImplemented

    def __le__(self: Self, other: Any, /) -> bool:
        if isinstance(other, BoundingBox):
            return self._n < other._n or self._n == other._n and sum(self._signs) >= sum(other._signs)
        else:
            return NotImplemented

    def __lt__(self: Self, other: Any, /) -> bool:
        if isinstance(other, BoundingBox):
            return self._n < other._n or self._n == other._n and sum(self._signs) > sum(other._signs)
        else:
            return NotImplemented

    def __ne__(self: Self, other: Any, /) -> bool:
        if isinstance(other, BoundingBox):
            return self._n != other._n or sum(self._signs) != sum(other._signs)
        else:
            return NotImplemented

    def __points(self: Self, /) -> Iterator[list[float]]:
        i = 0
        while (i >> (len(self._lower) - 1)) == 0:
            yield [bit * L + (1 - bit) * U for bit, L, U in zip(bits_of(i), self._lower, self._upper)]
            yield [bit * U + (1 - bit) * L for bit, L, U in zip(bits_of(i), self._lower, self._upper)]
            yield [mean(L, U) + 0.5 * cos(random.random() * pi) * (L - U) for L, U in zip(self._lower, self._upper)]
            i += 1
        while True:
            yield [mean(L, U) + 0.5 * cos(random.random() * pi) * (L - U) for L, U in zip(self._lower, self._upper)]

    def is_bounded(self: Self, /) -> bool:
        """Returns if every output has seen both a positive and negative sign."""
        return self._signs is not None and False not in self._signs

    def split(self: Self, /) -> tuple[Self, Self]:
        """Splits the bounding box into two halves along the longest dimension."""
        i = max(enumerate(self.lengths), key=lambda ix: ix[1])[0]
        middle = mean(self._lower[i], self._upper[i])
        upper = self._upper.copy()
        upper[i] = middle
        box = type(self)(self._lower, upper)
        lower = self._lower.copy()
        lower[i] = middle
        return (box, type(self)(lower, self._upper))

    def update(self: Self, y: Iterable[float], /) -> None:
        """Update the bounding box with an observed y value."""
        self._n += 1
        if isinstance(self._signs, tuple):
            if not isinstance(y, Sequence):
                y = [*y]
            self._signs = [False] * (2 * len(y))
        for i, yi in enumerate(y):
            self._signs[i if yi < 0 else ~i] = True

    @property
    def lengths(self: Self, /) -> Iterator[float]:
        """An iterator over the lengths in each input dimension."""
        return (U - L for L, U in zip(self._lower, self._upper))

    @property
    def points(self: Self, /) -> Iterator[list[float]]:
        """An iterator over the points which should be sampled next."""
        return self._points


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

def multivariate_type_check(
    f: Callable[[Sequence[float]], Sequence[float]],
    x1: Sequence[float],
    x2: Sequence[float],
    x: Optional[Sequence[float]],
    y1: Optional[Sequence[float]],
    y2: Optional[Sequence[float]],
    abs_err: float,
    rel_err: float,
    abs_tol: Optional[float],
    rel_tol: Optional[float],
    /,
) -> Optional[Union[TypeError, ValueError]]:
    """Performs multivariate input checks and potentially returns an error."""
    if not callable(f):
        return TypeError(f"expected a function for f, got {f!r}")
    elif not isinstance(x1, Sequence):
        return TypeError(f"could not interpret x1 as a vector, got {x1!r}")
    elif len(x1) <= 0:
        return ValueError(f"x1 must be a non-empty vector, got {x1!r}")
    elif not isinstance(x2, Sequence):
        return TypeError(f"could not interpret x2 as a vector, got {x2!r}")
    elif len(x2) <= 0:
        return ValueError(f"x2 must be a non-empty vector, got {x2!r}")
    elif len(x1) != len(x2):
        return ValueError(f"x1 and x2 must be vectors of equal sizes, got sizes of {len(x1)} and {len(x2)}")
    elif x is not None and not isinstance(x, Sequence):
        return TypeError(f"could not interpret x as a vector, got {x!r}")
    elif x is not None and len(x) != len(x1):
        return ValueError(f"x, x1, and x2 must be vectors of equal sizes, got sizes of {len(x)}, {len(x1)}, and {len(x2)}")
    elif y1 is not None and not isinstance(y1, Sequence):
        return TypeError(f"could not interpret y1 as a vector, got {y1!r}")
    elif y1 is not None and len(y1) <= 0:
        return ValueError(f"y1 must be a non-empty vector, got {y1!r}")
    elif y2 is not None and not isinstance(y2, Sequence):
        return TypeError(f"could not interpret y2 as a vector, got {y2!r}")
    elif y2 is not None and len(y2) <= 0:
        return ValueError(f"y2 must be a non-empty vector, got {y2!r}")
    elif y1 is not None is not y1 and len(y1) != len(y2):
        return ValueError(f"y1 and y2 must be vectors of equal sizes, got sizes of {len(y1)} and {len(y2)}")
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

def type_check(
    f: Callable[[float], float],
    x1: float,
    x2: float,
    x: Optional[float],
    y1: Optional[float],
    y2: Optional[float],
    abs_err: float,
    rel_err: float,
    abs_tol: Optional[float],
    rel_tol: Optional[float],
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
