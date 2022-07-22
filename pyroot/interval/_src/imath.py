import decimal
import itertools
import math
import operator
from collections.abc import Iterable
from decimal import Decimal, localcontext
from typing import Any, Optional, SupportsFloat, SupportsIndex, Type
from typing import TypeVar, Union, get_args, overload

from .fpu_rounding import *
from .interval import NOT_REAL, Interval, interval
from .lanczos import digamma_precise, gamma_precise, lgamma_precise
from .typing import RealLike, SupportsRichFloat

NOT_REAL = "could not interpret {} as an interval"

e = interval[math.e:math.nextafter(math.e, math.inf)]
inf = interval[math.inf:]

_PI = interval[math.pi:math.nextafter(math.pi, math.inf)]
_BIG_PI = Decimal(
    "3."
    "14159265358979323846264338327950288419716939937510582097494459230"
    "78164062862089986280348253421170679821480865132823066470938446095"
    "50582231725359408128481117450284102701938521105559644622948954930"
    "38196442881097566593344612847564823378678316527120190914564856692"
    "34603486104543266482133936072602491412737245870066063155881748815"
)

Self = TypeVar("Self", bound="PiMultiple")


class PiMultiple(Interval):

    __slots__ = ()

    def __abs__(self: Self) -> Self:
        iterator = iter(abs(self.coefficients)._endpoints)
        return type(self)(*zip(iterator, iterator))

    def __add__(self: Self, other: Union[Interval, RealLike]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__add__ is type(other).__add__:
            iterator = iter((self.coefficients + other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__add__ is type(other).__add__:
            return self.__as_interval__() + other
        elif isinstance(other, (Decimal, SupportsIndex)):
            intervals = []
            iterator = iter(self._endpoints)
            with localcontext() as ctx:
                if isinstance(other, SupportsIndex):
                    other = operator.index(other)
                    ctx.prec += other.bit_length() + 2
                else:
                    ctx.prec += (1 + abs(other)).log10() + 2
                for lower, upper in zip(iterator, iterator):
                    if math.isinf(lower):
                        L = lower
                    else:
                        with localcontext() as ctx:
                            ctx.prec += 2 * int(math.log10(1 + abs(lower)))
                            L = Decimal(lower) * _BIG_PI + other
                    if math.isinf(upper):
                        U = lower
                    else:
                        with localcontext() as ctx:
                            ctx.prec += 2 * int(math.log10(1 + abs(upper)))
                            U = Decimal(lower) * _BIG_PI + other
                    intervals.append((L, U))
            return Interval(*intervals)
        elif isinstance(other, SupportsFloat):
            return self.__as_interval__() + Interval(float_split(other))
        else:
            return NotImplemented

    def __and__(self: Self, other: Union[Interval, RealLike]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__and__ is type(other).__and__:
            iterator = iter((self.coefficients & other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__and__ is type(other).__and__:
            return self.__as_interval__() & other
        elif isinstance(other, get_args(RealLike)):
            if isinstance(other, SupportsIndex):
                other = operator.index(other)
            return self.__as_interval__() & Interval(float_split(other))
        else:
            return NotImplemented

    def __as_interval__(self: Self) -> Interval:
        return self.coefficients * _PI

    @classmethod
    def __dist__(cls: Type[Self], p: list[Interval], q: list[Interval]) -> Self:
        if not all(
            type(x).__dist__.__func__ is cls.__dist__.__func__
            for pq in (p, q)
            for x in pq
        ):
            return NotImplemented
        return pi * dist([x.coefficients for x in p], [x.coefficients for x in q])

    def __eq__(self: Self, other: Any) -> bool:
        if isinstance(other, PiMultiple) and type(self).__eq__ is type(other).__eq__:
            return self._endpoints == other._endpoints
        elif isinstance(other, Interval) and Interval.__eq__ is type(other).__eq__:
            return all(
                e1 * _PI.minimum == e1 * _PI.maximum == e2
                for e1, e2 in zip(self._endpoints, other._endpoints)
            )
        elif isinstance(other, get_args(RealLike)):
            if isinstance(other, SupportsIndex):
                other = operator.index(other)
            return self == Interval(float_split(other))
        else:
            return NotImplemented

    def __format__(self: Self, specifier: str) -> str:
        if self == pi:
            return "pi"
        else:
            return format(self.coefficients, specifier) + " * pi"

    @classmethod
    def __fsum__(cls: Type[Self], intervals: list[Interval]) -> Self:
        if not all(
            type(x).__fsum__.__func__ is cls.__fsum__.__func__
            for x in intervals
        ):
            return NotImplemented
        return pi * fsum(x.coefficients for x in intervals)

    def __getitem__(self: Self, args: Union[slice, tuple[slice, ...]]) -> Interval:
        return self.__as_interval__()[args]

    def __mul__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__mul__ is type(other).__mul__:
            with localcontext() as ctx:
                ctx.prec += 5
                pi_squared = _BIG_PI ** 2
                return Interval(
                    *[
                        (
                            float_down(pi_squared * Decimal(xi.maximum) * Decimal(yi.minimum)),
                            float_up(pi_squared * Decimal(xi.minimum) * Decimal(yi.maximum)),
                        )
                        for xi in x[0:].sub_intervals
                        for yi in y[:0].sub_intervals
                    ],
                    *[
                        (
                            float_down(pi_squared * Decimal(xi.minimum) * Decimal(yi.maximum)),
                            float_up(pi_squared * Decimal(xi.maximum) * Decimal(yi.minimum)),
                        )
                        for xi in x[:0].sub_intervals
                        for yi in y[0:].sub_intervals
                    ],
                    *[
                        (
                            float_down(pi_squared * Decimal(xi.minimum) * Decimal(yi.minimum)),
                            float_up(pi_squared * Decimal(xi.maximum) * Decimal(yi.maximum)),
                        )
                        for xi in x[0:].sub_intervals
                        for yi in y[0:].sub_intervals
                    ],
                    *[
                        (
                            float_down(pi_squared * Decimal(xi.maximum) * Decimal(yi.maximum)),
                            float_up(pi_squared * Decimal(xi.minimum) * Decimal(yi.minimum)),
                        )
                        for xi in x[:0].sub_intervals
                        for yi in y[:0].sub_intervals
                    ],
                )
            return self.__as_interval__() * other.__as_interval__()
        elif isinstance(other, Interval) and Interval.__mul__ is type(other).__mul__:
            iterator = iter((self.coefficients * other)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, get_args(RealLike)):
            iterator = iter((self.coefficients * other)._endpoints)
            return type(self)(*zip(iterator, iterator))
        else:
            return NotImplemented

    def __or__(self: Self, other: Union[Interval, RealLike]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__or__ is type(other).__or__:
            iterator = iter((self.coefficients | other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__or__ is type(other).__or__:
            iterator = iter((self.__as_interval__() | other)._endpoints)
            return Interval(*zip(iterator, iterator))
        elif isinstance(other, get_args(RealLike)):
            return self.__as_interval__() | other
        else:
            return NotImplemented

    def __pow__(self: Self, other: Union[Interval, RealLike], modulo: None = None) -> Interval:
        if modulo is not None:
            return NotImplemented
        elif isinstance(other, PiMultiple) and type(self).__pow__ is type(other).__pow__:
            iterator = iter((self.__as_interval__() ** (other.__as_interval__()))._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__pow__ is type(other).__pow__:
            return self.__as_interval__() ** other.__as_interval__()
        elif isinstance(other, (Decimal, SupportsFloat, SupportsIndex)):
            return self.__as_interval__() ** other
        else:
            return NotImplemented

    def __rmul__(self: Self, other: Union[Interval, RealLike]) -> Interval:
        if (
            isinstance(other, Interval)
            and type(other).__mul__ is Interval.__mul__
            or isinstance(other, get_args(RealLike))
        ):
            return self * other
        else:
            return NotImplemented

    def __sub__(self: Self, other: Union[Interval, RealLike]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__sub__ is type(other).__sub__:
            iterator = iter((self.coefficients - other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__sub__ is type(other).__sub__:
            return self.__as_interval__() - other.__as_interval__()
        elif isinstance(other, get_args(RealLike)):
            if isinstance(other, SupportsIndex):
                other = operator.index(other)
            return self + -other
        else:
            return NotImplemented

    def __truediv__(self: Self, other: Union[Interval, RealLike]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__truediv__ is type(other).__truediv__:
            return self.coefficients / other.coefficients
        elif isinstance(other, Interval) and Interval.__truediv__ is type(other).__truediv__:
            iterator = iter((self.coefficients / other)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, get_args(RealLike)):
            if isinstance(other, SupportsIndex):
                other = operator.index(other)
            iterator = iter((self.coefficients / other)._endpoints)
            return type(self)(*zip(iterator, iterator))
        else:
            return NotImplemented

    def __xor__(self: Self, other: Union[Interval, RealLike]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__xor__ is type(other).__xor__:
            iterator = iter((self.coefficients ^ other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__xor__ is type(other).__xor__:
            return self.__as_interval__() ^ other
        elif isinstance(other, get_args(RealLike)):
            if isinstance(other, SupportsIndex):
                other = operator.index(other)
            return self.__as_interval__() ^ other
        else:
            return NotImplemented

    @property
    def coefficients(self: Self) -> Interval:
        iterator = iter(self._endpoints)
        return Interval(*zip(iterator, iterator))

    @property
    def maximum(self: Self) -> float:
        m = super().maximum
        if m > 0:
            return m * _PI.maximum
        else:
            return m * _PI.minimum

    @property
    def minimum(self: Self) -> float:
        m = super().minimum
        if m < 0:
            return m * _PI.maximum
        else:
            return m * _PI.minimum


pi = PiMultiple((1.0, 1.0))
tau = 2 * pi

SPECIAL_ANGLES = {
    -1.0: (1, 1, -1, 2),
    -0.5: (2, 3, -1, 6),
    0.0: (1, 2, 0, 1),
    0.5: (1, 3, 1, 6),
    1.0: (0, 1, 1, 2),
}

@overload
def sym_mod(x: float, modulo: float) -> float: ...

@overload
def sym_mod(x: Decimal, modulo: Decimal) -> Decimal: ...

def sym_mod(x, modulo):
    remainder = Decimal(x).remainder_near(2 * modulo)
    return remainder if isinstance(x, Decimal) else float(remainder)

def cos_precise(x: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec += 2
        x = sym_mod(x, _BIG_PI) ** 2
        i = last_s = 0
        s = fact = num = sign = 1
        while s != last_s:
            last_s = s
            i += 2
            fact *= i * (i - 1)
            num *= x
            sign *= -1
            s += sign * num / fact
        return s

def sin_precise(x: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec += 2
        s = num = x = sym_mod(x, _BIG_PI)
        last_s = 0
        i = fact = sign = 1
        x **= 2
        while s != last_s:
            last_s = s
            i += 2
            fact *= i * (i - 1)
            num *= x
            sign *= -1
            s += sign * num / fact
        return s

def cos_sin_precise(x: Union[Decimal, SupportsIndex, float]) -> tuple[Decimal, Decimal]:
    with localcontext() as ctx:
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
            ctx.prec += x.bit_length() + 10
        elif isinstance(x, Decimal):
            ctx.prec += 2 * int((1 + abs(x)).log10()) + 10
        else:
            ctx.prec += 2 * int(math.log10(1 + abs(x))) + 10
        c = cos_precise(Decimal(abs(x)))
        s = sin_precise(Decimal(abs(x)))
        if x < 0:
            return (c, -s)
        else:
            return (c, s)

def acos(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()[-1.0:1.0]._endpoints)
    intervals = []
    result = PiMultiple()
    for lower, upper in zip(iterator, iterator):
        if lower in SPECIAL_ANGLES and upper in SPECIAL_ANGLES:
            L = SPECIAL_ANGLES[lower][:2]
            U = SPECIAL_ANGLES[upper][:2]
            result |= interval[div_down(U[0], U[1]) : div_up(L[0], L[1])] * pi
            if result._endpoints == (0.0, 1.0):
                return result
        else:
            intervals.append((acos_down(lower), acos_up(upper)))
    if len(intervals) > 0:
        return Interval(*intervals) | result
    else:
        return result

def acos_down(x: float) -> float:
    y = math.acos(x)
    if cos_sin_precise(y)[0] < x:
        return math.nextafter(y, 0.0)
    else:
        return y

def acos_up(x: float) -> float:
    y = math.acos(x)
    if cos_sin_precise(y)[0] > x:
        return math.nextafter(y, 4.0)
    else:
        return y

def acosh(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()[1.0:]._endpoints)
    return Interval(*[
        (acosh_down(lower), acosh_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def acosh_down(x: float) -> float:
    y = math.acosh(x)
    d = Decimal(y)
    if (d.exp() + (-d).exp()) / 2 > x:
        return math.nextafter(y, 0.0)
    else:
        return y

def acosh_up(x: float) -> float:
    y = math.acosh(x)
    d = Decimal(y)
    if (d.exp() + (-d).exp()) / 2 < x:
        return math.nextafter(y, math.inf)
    else:
        return y

def asin(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()[-1.0:1.0]._endpoints)
    intervals = []
    result = PiMultiple()
    for lower, upper in zip(iterator, iterator):
        if lower in SPECIAL_ANGLES and upper in SPECIAL_ANGLES:
            L = SPECIAL_ANGLES[lower][2:]
            U = SPECIAL_ANGLES[upper][2:]
            result |= interval[div_down(L[0], L[1]) : div_up(U[0], U[1])] * pi
            if result._endpoints == (-0.5, 0.5):
                return result
        else:
            intervals.append((asin_down(lower), asin_up(upper)))
    if len(intervals) > 0:
        return Interval(*intervals) | result
    else:
        return result

def asin_down(x: float) -> float:
    y = math.asin(x)
    if cos_sin_precise(y)[1] > x:
        return math.nextafter(y, -math.inf)
    else:
        return y

def asin_up(x: float) -> float:
    y = math.asin(x)
    d = Decimal(y)
    if cos_sin_precise(y)[1] < x:
        return math.nextafter(y, math.inf)
    else:
        return y

def asinh(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (asinh_down(lower), asinh_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def asinh_down(x: float) -> float:
    y = math.asinh(x)
    d = Decimal(y)
    if (d.exp() - (-d).exp()) / 2 > x:
        return math.nextafter(y, -math.inf)
    else:
        return y

def asinh_up(x: float) -> float:
    y = math.asinh(x)
    d = Decimal(y)
    if (d.exp() - (-d).exp()) / 2 < x:
        return math.nextafter(y, math.inf)
    else:
        return y

def atan(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    intervals = []
    pi_intervals = []
    for lower, upper in zip(iterator, iterator):
        if (
            lower in (-math.inf, -1.0, 0.0, 1.0, math.inf)
            and upper in (-math.inf, -1.0, 0.0, 1.0, math.inf)
        ):
            if math.isinf(lower):
                L = 0.5
            elif lower == 0.0:
                L = 0.0
            else:
                L = 0.25
            if lower < 0:
                L *= -1
            if math.isinf(upper):
                U = 0.5
            elif upper == 0.0:
                U = 0.0
            else:
                U = 0.25
            if upper < 0:
                U *= -1
            pi_intervals.append((L, U))
        else:
            intervals.append((atan_down(lower), atan_up(upper)))
    result = PiMultiple(*pi_intervals)
    if result != Interval((-0.5, 0.5)) * pi and len(intervals) > 0:
        return Interval(*intervals) | result
    else:
        return result

def atan_down(x: float) -> float:
    y = math.atan(x)
    c, s = cos_sin_precise(y)
    if s / c > x:
        return math.nextafter(y, -math.inf)
    else:
        return y

def atan_up(x: float) -> float:
    y = math.atan(x)
    c, s = cos_sin_precise(y)
    if s / c < x:
        return math.nextafter(y, math.inf)
    else:
        return y

def atan2(y: Union[Interval, RealLike], x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    x = x.__as_interval__()
    y = y.__as_interval__()
    intervals = []
    pi_intervals = []
    for xi in x[0:].sub_intervals:
        if xi.maximum == 0.0:
            continue
        for yi in y[0:].sub_intervals:
            if math.isinf(yi.maximum) or xi.minimum == 0.0:
                U = 0.5
            elif yi.maximum == 0.0 or math.isinf(xi.minimum):
                U = 0.0
            elif yi.maximum == xi.minimum * 4 or yi.maximum / 4 == xi.minimum:
                U = 0.25
            else:
                intervals.append((atan2_down(yi.minimum, xi.maximum), atan2_up(yi.maximum, xi.minimum)))
                continue
            if yi.minimum == 0.0 or math.isinf(xi.maximum):
                L = 0.0
            elif yi.minimum == xi.maximum * 4 or yi.minimum / 4 == xi.maximum:
                L = 0.25
            else:
                intervals.append((atan2_down(yi.minimum, xi.maximum), atan2_up(yi.maximum, xi.minimum)))
                continue
            pi_intervals.append((L, U))
        for yi in y[:0].sub_intervals:
            if math.isinf(yi.minimum) or xi.minimum == 0.0:
                L = -0.5
            elif yi.minimum == 0.0 or math.isinf(xi.minimum):
                L = -0.0
            elif yi.minimum == xi.minimum * 4 or yi.minimum / 4 == xi.minimum:
                L = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            if yi.maximum == 0.0 or math.isinf(xi.maximum):
                U = -0.0
            elif yi.maximum == xi.maximum * 4 or yi.maximum / 4 == xi.maximum:
                U = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            pi_intervals.append((L, U))
    for xi in x[:0].sub_intervals:
        if xi.minimum == 0.0:
            continue
        for yi in y[0:].sub_intervals:
            if math.isinf(yi.maximum) or xi.maximum == 0.0:
                L = -0.5
            elif yi.maximum == 0.0 or math.isinf(xi.maximum):
                L = -0.0
            elif yi.maximum == xi.maximum * 4 or yi.maximum / 4 == xi.maximum:
                L = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            if yi.minimum == 0.0 or math.isinf(xi.minimum):
                U = -0.0
            elif yi.minimum == xi.minimum * 4 or yi.minimum / 4 == xi.minimum:
                U = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            pi_intervals.append((L, U))
        for yi in y[:0].sub_intervals:
            if math.isinf(yi.minimum) or xi.maximum == 0.0:
                U = 0.5
            elif yi.minimum == 0.0 or math.isinf(xi.maximum):
                U = 0.0
            elif yi.minimum == xi.maximum * 4 or yi.minimum / 4 == xi.maximum:
                U = 0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.minimum), atan2_up(yi.minimum, xi.maximum)))
                continue
            if yi.maximum == 0.0 or math.isinf(xi.minimum):
                L = 0.0
            elif yi.maximum == xi.minimum * 4 or yi.maximum / 4 == xi.minimum:
                L = 0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.minimum), atan2_up(yi.minimum, xi.maximum)))
                continue
            pi_intervals.append((L, U))
    result = PiMultiple(*pi_intervals)
    if result != Interval((-0.5, 0.5)) * pi and len(intervals) > 0:
        return result | Interval(*intervals)
    else:
        return result

def atan2_down(y: float, x: float) -> float:
    z = math.atan2(y, x)
    c, s = cos_sin_precise(z)
    if s / c > Decimal(y) / Decimal(x):
        return math.nextafter(z, -math.inf)
    else:
        return z

def atan2_up(y: float, x: float) -> float:
    z = math.atan2(y, x)
    c, s = cos_sin_precise(z)
    if s / c < Decimal(y) / Decimal(x):
        return math.nextafter(z, math.inf)
    else:
        return z

def atanh(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (atanh_down(lower), atanh_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def atanh_down(x: float) -> float:
    y = math.atanh(x)
    d = 2 * Decimal(y)
    if d > 0:
        t = (1 - (-d).exp()) / (1 + (-d).exp())
    else:
        t = (d.exp() - 1) / (d.exp() + 1)
    if t > x:
        return math.nextafter(y, -math.inf)
    else:
        return y

def atanh_up(x: float) -> float:
    y = math.atanh(x)
    d = 2 * Decimal(y)
    if d > 0:
        t = (1 - (-d).exp()) / (1 + (-d).exp())
    else:
        t = (d.exp() - 1) / (d.exp() + 1)
    if t > x:
        return math.nextafter(y, math.inf)
    else:
        return y

def cos(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    elif isinstance(x, PiMultiple):
        intervals = []
        for sub_interval in abs(x).sub_intervals:
            lower = sym_mod(sub_interval._endpoints[0], 1)
            upper = sub_interval._endpoints[1] - sub_interval._endpoints[0] + lower
            if upper - lower > 2.0 or math.isinf(lower) or math.isinf(upper):
                return Interval((-1.0, 1.0))
            _upper = sym_mod(upper, 1)
            if lower < -0.75:
                L = -cos_precise((Decimal(lower) + 1) * _BIG_PI)
            elif lower < -0.25:
                L = sin_precise((Decimal(lower) + Decimal("0.5")) * _BIG_PI)
            elif lower <= 0.25:
                L = cos_precise(Decimal(lower) * _BIG_PI)
            elif lower <= 0.75:
                L = -sin_precise((Decimal(lower) - Decimal("0.5")) * _BIG_PI)
            else:
                L = -cos_precise((Decimal(lower) - 1) * _BIG_PI)
            if _upper < -0.75:
                U = -cos_precise((Decimal(_upper) + 1) * _BIG_PI)
            elif _upper < -0.25:
                U = sin_precise((Decimal(_upper) + Decimal("0.5")) * _BIG_PI)
            elif _upper <= 0.25:
                U = cos_precise(Decimal(_upper) * _BIG_PI)
            elif _upper <= 0.75:
                U = -sin_precise((Decimal(_upper) - Decimal("0.5")) * _BIG_PI)
            else:
                U = -cos_precise((Decimal(_upper) - 1) * _BIG_PI)
            intervals.append((
                -1.0
                if not -1.0 < lower <= upper < 1.0
                else float_down(min(L, U)),
                1.0
                if lower <= 0.0 <= upper or 2.0 <= upper
                else float_up(max(L, U)),
            ))
        return Interval(*intervals)
    x = abs(x.__as_interval__())
    if len(x._endpoints) == 0:
        return x
    elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
        return interval[-1.0:1.0]
    intervals = []
    iterator = iter(x._endpoints)
    for lower, upper in zip(iterator, iterator):
        with localcontext() as ctx:
            ctx.prec += 2 * int(math.log10(1 + abs(lower))) + 10
            _L = sym_mod(Decimal(lower), _BIG_PI)
            _U = Decimal(upper) - Decimal(lower) + _L
        CL = cos_sin_precise(lower)[0]
        CU = cos_sin_precise(upper)[0]
        if _L == -_BIG_PI or _BIG_PI <= _U:
            L = -1.0
        else:
            L = float_down(min(CL, CU))
        if _L <= 0 <= _U or 2 * _BIG_PI <= _U:
            U = 1.0
        else:
            U = float_up(max(CL, CU))
        intervals.append((L, U))
    return Interval(*intervals)

def cos_down(x: float) -> float:
    return float_down(cos_sin_precise(x)[0])

def cos_up(x: float) -> float:
    return float_up(cos_sin_precise(x)[0])

def cosh(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (
            1.0
            if lower <= 0.0 <= upper
            else min(cosh_down(lower), cosh_down(upper)),
            max(cosh_up(lower), cosh_up(upper)),
        )
        for lower, upper in zip(iterator, iterator)
    ])

def cosh_down(x: float) -> float:
    x = Decimal(x)
    try:
        return float_down((x.exp() + (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return float_down((x - Decimal(2).ln()).exp() + (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf

def cosh_up(x: float) -> float:
    x = Decimal(x)
    try:
        return float_up((x.exp() + (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return float_up((x - Decimal(2).ln()).exp() + (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf

def degrees(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    elif isinstance(x, PiMultiple):
        return x * 180 / pi
    with localcontext() as ctx:
        ctx.prec += 2
        coef = 180 / _BIG_PI
        return Interval(
            *[
                (float_down(coef * Decimal(xi.maximum)), float_up(coef * Decimal(xi.minimum)))
                for xi in x[:0].sub_intervals
            ],
            *[
                (float_down(coef * Decimal(xi.minimum)), float_up(coef * Decimal(xi.maximum)))
                for xi in x[0:].sub_intervals
            ],
        )

def digamma(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    intervals = []
    iterator = iter(x.__as_interval__()._endpoints)
    for lower, upper in zip(iterator, iterator):
        if lower > 0:
            intervals.append((digamma_down(lower), digamma_up(upper)))
            continue
        elif upper - lower >= 2.0 or math.ceil(lower) < math.floor(upper):
            return Interval((-math.inf, math.inf))
        elif lower == upper and lower.is_integer():
            intervals.append((-math.inf, -math.inf))
            intervals.append((math.inf, math.inf))
            continue
        elif lower.is_integer():
            L = -math.inf
        else:
            L = digamma_down(lower)
        if upper.is_integer():
            U = math.inf
        else:
            U = digamma_up(upper)
        if lower - upper - ((-upper) % 1.0) < -1.0:
            intervals.append((-math.inf, U))
            intervals.append((L, math.inf))
        else:
            intervals.append((L, U))
    return Interval(*intervals)

def digamma_down(x: float) -> float:
    if math.isinf(x):
        return x
    elif x < 0.5 and x.is_integer():
        return -math.inf
    elif x >= 0.5:
        return float_down(digamma_precise(x))
    with localcontext() as ctx:
        ctx.prec += 10
        if x < 0.0:
            d = -((-x) % 1.0)
            if d <= -0.5:
                d = x % 1.0
        else:
            d = x
        c, s = cos_sin_precise(_BIG_PI * Decimal(d))
        return float_down(digamma_precise(1 - x) - _BIG_PI * c / s)

def digamma_up(x: float) -> float:
    if math.isinf(x) or x < 0.5 and x.is_integer():
        return math.inf
    elif x >= 0.5:
        return float_up(digamma_precise(x))
    with localcontext() as ctx:
        ctx.prec += 10
        if x < 0.0:
            d = -((-x) % 1.0)
            if d <= -0.5:
                d = x % 1.0
        else:
            d = x
        c, s = cos_sin_precise(_BIG_PI * Decimal(d))
        return float_up(digamma_precise(1 - x) - _BIG_PI * c / s)

def dist(p: Iterable[Union[Interval, RealLike]], q: Iterable[Union[Interval, RealLike]], /) -> Interval:
    intervals = []
    for x in p:
        if isinstance(x, get_args(RealLike)):
            if isinstance(x, SupportsIndex):
                x = operator.index(x)
            x = Interval(float_split(x))
        elif not isinstance(x, Interval):
            raise TypeError(NOT_INTERVAL.format(repr(x)))
        intervals.append(x)
    p = intervals
    intervals = []
    for x in q:
        if isinstance(x, get_args(RealLike)):
            if isinstance(x, SupportsIndex):
                x = operator.index(x)
            x = Interval(float_split(x))
        elif not isinstance(x, Interval):
            raise TypeError(NOT_INTERVAL.format(repr(x)))
        intervals.append(x)
    q = intervals
    if len(p) != len(q):
        raise ValueError("both points must have the same number of dimensions")
    for pq in (p, q):
        for x in pq:
            result = type(x).__dist__(p, q)
            if result is not NotImplemented:
                return result
    intervals = []
    with localcontext() as ctx:
        ctx.prec = 50
        for sub_intervals in itertools.product(
            *[
                interval.__as_interval__().sub_intervals
                for interval in p
            ],
            *[
                interval.__as_interval__().sub_intervals
                for interval in reversed(q)
            ],
        ):
            partials = multi_add(*[
                float_down(min(
                    Decimal(sub_intervals[i].minimum) - Decimal(sub_intervals[~i].maximum),
                    Decimal(sub_intervals[i].maximum) - Decimal(sub_intervals[~i].minimum),
                    key=abs,
                ) ** 2)
                for i in range(len(p))
                if 0.0 not in sub_intervals[i] - sub_intervals[~i]
            ])
            if len(partials) == 1 or partials[-2] > 0:
                L = float_down(partials[-1])
            else:
                L = math.nextafter(float_up(partials[-1]), -math.inf)
            if L > 0.0 and math.isinf(L):
                L = math.nextafter(L, 0.0)
            partials = multi_add(*[
                float_up(max(
                    Decimal(sub_intervals[i].minimum) - Decimal(sub_intervals[~i].maximum),
                    Decimal(sub_intervals[i].maximum) - Decimal(sub_intervals[~i].minimum),
                    key=abs,
                ) ** 2)
                for i in range(len(p))
            ])
            if len(partials) == 1 or partials[-2] < 0:
                U = float_up(partials[-1])
            else:
                U = math.nextafter(float_down(partials[-1]), math.inf)
            if U < 0.0 and math.isinf(U):
                U = math.nextafter(U, 0.0)
            intervals.append((L, U))
    return sqrt(Interval(*intervals))

def erf_small_precise(x: float) -> Decimal:
    assert abs(x) <= 1.5
    with localcontext() as ctx:
        ctx.prec += 5
        d = Decimal(x)
        d2 = d ** 2
        acc = 0
        fk = Decimal("25.5")
        for i in range(25):
            acc = 2 + d2 * acc / fk
            fk -= 1
        return acc * d * (-d2).exp() / _BIG_PI.sqrt()

def erfc_precise(x: float) -> Decimal:
    assert 1.5 < x < 30.0
    with localcontext() as ctx:
        ctx.prec += 5
        d = Decimal(x)
        d2 = d ** 2
        a = 0
        da = Decimal("0.5")
        p = 1
        p_last = 0
        q = da + d2
        q_last = 1
        for i in range(50):
            a += da
            da += 2
            b = da + d2
            p_last, p = p, b * p - a * p_last
            q_last, q = q, b * q - a * q_last
        return p / q * d * (-d2).exp() / _BIG_PI.sqrt()

def erf(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (erf_down(lower), erf_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def erf_down(x: float) -> float:
    if abs(x) <= 1.5:
        return float_down(erf_small_precise(x))
    elif 1.5 < x < 30.0:
        return float_down(1 - erfc_precise(x))
    elif -30.0 < x < -1.5:
        return float_down(erfc_precise(-x) - 1)
    elif x < 0.0:
        return -1.0
    elif math.isinf(x):
        return 1.0
    else:
        return math.nextafter(1.0, 0.0)

def erf_up(x: float) -> float:
    if abs(x) <= 1.5:
        return float_up(erf_small_precise(x))
    elif 1.5 < x < 30.0:
        return float_up(1 - erfc_precise(x))
    elif -30.0 < x < -1.5:
        return float_up(erfc_precise(-x) - 1)
    elif x > 0.0:
        return 1.0
    elif math.isinf(x):
        return -1.0
    else:
        return math.nextafter(-1.0, 0.0)

def erfc(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (erfc_down(upper), erfc_up(lower))
        for lower, upper in zip(iterator, iterator)
    ])

def erfc_down(x: float) -> float:
    if abs(x) <= 1.5:
        return float_down(1 - erf_small_precise(x))
    elif 1.5 < x < 30.0:
        return float_down(erfc_precise(x))
    elif -30.0 < x < -1.5:
        return float_down(erfc_precise(-x) + 2)
    elif x > 0.0:
        return 0.0
    elif math.isinf(x):
        return 2.0
    else:
        return math.nextafter(2.0, 0.0)

def erfc_up(x: float) -> float:
    if abs(x) <= 1.5:
        return float_up(1 - erf_small_precise(x))
    elif 1.5 < x < 30.0:
        return float_up(erfc_precise(x))
    elif -30.0 < x < -1.5:
        return float_up(erfc_precise(-x) + 2)
    elif x < 0.0:
        return 2.0
    elif math.isinf(x):
        return 0.0
    else:
        return math.nextafter(0.0, 1.0)

def exp(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, Decimal):
        try:
            return Interval(float_split(x.exp()))
        except decimal.Overflow:
            return Interval((math.nextafter(math.inf, 0.0), math.inf))
    elif isinstance(x, SupportsIndex):
        try:
            return Interval(float_split(Decimal(operator.index(x)).exp()))
        except decimal.Overflow:
            return Interval((math.nextafter(math.inf, 0.0), math.inf))
    elif isinstance(x, get_args(RealLike)):
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (exp_down(lower), exp_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def exp_down(x: float) -> float:
    try:
        return float_down(Decimal(x).exp())
    except decimal.Overflow:
        return math.nextafter(math.inf, x)

def exp_up(x: float) -> float:
    try:
        result = float_up(Decimal(x).exp())
    except decimal.Overflow:
        return math.inf
    if result != 0.0 or math.isinf(x):
        return result
    else:
        return math.nextafter(0.0, 1.0)

def expm1_precise(x: float) -> Decimal:
    with localcontext() as ctx:
        ctx.prec += 5
        d = Decimal(x)
        if abs(d) < 1e-4:
            return d * (1 + d / 2 * (1 + d / 3 * (1 + d / 4)))
        else:
            return d.exp() - 1

def expm1(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, Decimal):
        if abs(x) < 1e-4:
            with localcontext() as ctx:
                ctx.prec += 5
                return Interval(float_split(
                    x * (1 + x / 2 * (1 + x / 3 * (1 + x / 4)))
                ))
        try:
            return Interval(float_split(x.exp() - 1))
        except decimal.Overflow:
            return Interval((math.nextafter(math.inf, 0.0), math.inf))
    elif isinstance(x, SupportsIndex):
        try:
            return Interval(float_split(Decimal(operator.index(x)).exp() - 1))
        except decimal.Overflow:
            return Interval((math.nextafter(math.inf, 0.0), math.inf))
    elif isinstance(x, get_args(RealLike)):
        x = Interval(float_split(x))
    else:
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (expm1_down(lower), expm1_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def expm1_down(x: float) -> float:
    try:
        return float_down(expm1_precise(x))
    except decimal.Overflow:
        return math.nextafter(math.inf, x)

def expm1_up(x: float) -> float:
    try:
        result = float_up(expm1_precise(x))
    except decimal.Overflow:
        return math.inf
    if result != -1.0 or math.isinf(x):
        return result
    else:
        return math.nextafter(-1.0, 0.0)

def fsum(intervals: Iterable[Union[Interval, RealLike]]) -> Interval:
    temp = []
    for x in intervals:
        if isinstance(x, get_args(RealLike)):
            if isinstance(x, SupportsIndex):
                x = operator.index(x)
            x = Interval(float_split(x))
        elif not isinstance(x, Interval):
            raise TypeError(NOT_INTERVAL.format(repr(x)))
        temp.append(x)
    intervals = temp
    for x in intervals:
        result = type(x).__fsum__(intervals)
        if result is not NotImplemented:
            return result
    results = []
    with localcontext() as ctx:
        ctx.prec = 50
        for sub_intervals in itertools.product(*[
            interval.__as_interval__().sub_intervals
            for interval in intervals
        ]):
            partials = multi_add(*[
                sub_interval.minimum
                for sub_interval in sub_intervals
            ])
            if len(partials) == 1 or partials[-2] > 0:
                L = float_down(partials[-1])
            else:
                L = math.nextafter(float_up(partials[-1]), -math.inf)
            if L > 0.0 and math.isinf(L):
                L = math.nextafter(L, 0.0)
            partials = multi_add(*[
                sub_interval.maximum
                for sub_interval in sub_intervals
            ])
            if len(partials) == 1 or partials[-2] < 0:
                U = float_up(partials[-1])
            else:
                U = math.nextafter(float_down(partials[-1]), math.inf)
            if U < 0.0 and math.isinf(U):
                U = math.nextafter(U, 0.0)
            results.append((L, U))
    return Interval(*results)

LGAMMA_POS_MIN_X = Decimal(
    "1.46163214496836234126595423257"
)
LGAMMA_POS_MIN_Y = Decimal(
    "-0.1214862905358496080955145571"
)

DIGAMMA_ROOTS_CACHE: dict[int, tuple[float, float]] = {}

def digamma_root(n: int) -> tuple[float, float]:
    if n not in DIGAMMA_ROOTS_CACHE:
        L = 0.0
        U = 0.5
        dx = 1
        while True:
            x = _BIG_PI / digamma_precise(1 - n - L)
            theta = atan_up(float_down(x))
            c, s = cos_sin_precise(theta)
            L = float_down((Decimal(theta) - c * (s + c * x)) / _BIG_PI)
            x = _BIG_PI / digamma_precise(1 - n - U)
            theta = atan_down(float_up(x))
            c, s = cos_sin_precise(theta)
            U = float_up((Decimal(theta) - c * (s + c * x)) / _BIG_PI)
            if dx >= U - L:
                break
            dx = U - L
        DIGAMMA_ROOTS_CACHE[n] = (L, U)
    return DIGAMMA_ROOTS_CACHE[n]

LGAMMA_MINS_CACHE: dict[int, Decimal] = {}

def lgamma_min(n: int) -> float:
    if n not in LGAMMA_MINS_CACHE:
        L, U = digamma_root(n)
        M = (Decimal(L) + Decimal(U)) / 2
        maximum = (Interval((1, 1)) - n - L).maximum
        minimum = (Interval((1, 1)) - n - U).minimum
        if abs(Decimal(maximum) + (n - 1) + M) < abs(Decimal(minimum) + (n - 1) + M):
            D = digamma_precise(maximum)
            D -= (Decimal(maximum) + (n - 1) + M) * (D - digamma_precise(minimum)) / (Decimal(maximum) - Decimal(minimum))
            L = lgamma_precise(maximum)
            L -= (Decimal(maximum) + (n - 1) + M) * (L - lgamma_precise(minimum)) / (Decimal(maximum) - Decimal(minimum))
        else:
            D = digamma_precise(minimum)
            D -= (Decimal(minimum) + (n - 1) + M) * (D - digamma_precise(maximum)) / (Decimal(minimum) - Decimal(maximum))
            L = lgamma_precise(minimum)
            L -= (Decimal(minimum) + (n - 1) + M) * (L - lgamma_precise(maximum)) / (Decimal(minimum) - Decimal(maximum))
        LGAMMA_MINS_CACHE[n] = (
            _BIG_PI.ln()
            - (1 + (D / _BIG_PI) ** 2).ln() / 2
            - L
        )
    return float_down(LGAMMA_MINS_CACHE[n])

def gamma_extrema(n: int) -> float:
    lgamma_min(n)
    return (-1) ** n * float_down(LGAMMA_MINS_CACHE[n].exp())

def lgamma(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, Decimal):
        if x.is_infinite():
            return inf if x > 0 else interval
        elif int(x) == x:
            x = int(x)
        else:
            L, U = float_split(x)
            with localcontext() as ctx:
                ctx.prec += 2
                if abs(x - Decimal(L)) < abs(x - Decimal(U)):
                    LG = lgamma_precise(L) + (x - Decimal(L)) * digamma_precise(L)
                else:
                    LG = lgamma_precise(U) + (x - Decimal(U)) * digamma_precise(U)
                return Interval(float_split(LG))
    if isinstance(x, SupportsIndex):
        x = operator.index(x)
        if x <= 0:
            return inf
        else:
            return Interval(float_split(lgamma_precise(float(x))))
    elif isinstance(x, SupportsRichFloat):
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    intervals = []
    iterator = iter(x.__as_interval__()._endpoints)
    for lower, upper in zip(iterator, iterator):
        if lower > LGAMMA_POS_MIN_X:
            L = lgamma_down(lower)
            U = lgamma_up(upper)
        elif math.isinf(lower):
            L = -math.inf
            U = math.inf
        elif 0.0 <= lower < LGAMMA_POS_MIN_X < upper:
            L = float_down(LGAMMA_POS_MIN_Y)
            if lower <= 0.0:
                U = math.inf
            else:
                U = max(lgamma_up(lower), lgamma_up(upper))
        elif lower == upper < 0.5 and lower.is_integer():
            L = U = math.inf
        elif digamma(Interval((lower, upper))).maximum < 0.0:
            L = lgamma_down(upper)
            U = lgamma_up(lower)
        elif digamma(Interval((lower, upper))).minimum > 0.0:
            L = lgamma_down(lower)
            U = lgamma_up(upper)
        elif digamma_down(lower) < 0:
            L = lgamma_min(math.floor(lower))
            if upper > LGAMMA_POS_MIN_X:
                L = min(L, float_down(LGAMMA_POS_MIN_Y))
            if upper - lower >= 1.0 or math.floor(lower) + 1 <= upper or lower.is_integer():
                U = math.inf
            else:
                U = max(lgamma_up(lower), lgamma_up(upper))
        else:
            L = lgamma_min(math.floor(lower))
            L = min(L, lgamma_down(lower))
            if upper > LGAMMA_POS_MIN_X:
                L = min(L, float_down(LGAMMA_POS_MIN_Y))
            U = math.inf
        intervals.append((L, U))
    return Interval(*intervals)

def lgamma_down(x: float) -> float:
    if math.isinf(x):
        return x
    elif x < 0.5 and x.is_integer():
        return math.inf
    y = float_down(lgamma_precise(x))
    if 0 < x < math.inf and math.isinf(y):
        return math.nextafter(y, 0.0)
    else:
        return y

def lgamma_up(x: float) -> float:
    if math.isinf(x) or x < 0.5 and x.is_integer():
        return math.inf
    else:
        return float_up(lgamma_precise(x))

def gamma(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, Decimal):
        if x.is_infinite():
            return inf if x > 0 else interval
        elif int(x) == x:
            x = int(x)
        else:
            L, U = float_split(x)
            with localcontext() as ctx:
                ctx.prec += 2
                if abs(x - Decimal(L)) < abs(x - Decimal(U)):
                    LG = lgamma_precise(L) + (x - Decimal(L)) * digamma_precise(L)
                else:
                    LG = lgamma_precise(U) + (x - Decimal(U)) * digamma_precise(U)
                if x > 0 or x.remainder_near(2) >= 0:
                    return Interval(float_split(LG.exp()))
                else:
                    return Interval(float_split(-LG.exp()))
    if isinstance(x, SupportsIndex):
        x = operator.index(x)
        if x <= 0:
            return Interval((-math.inf, -math.inf), (math.inf, math.inf))
        elif x >= 179:
            return Interval((math.nextafter(math.inf, 0.0), math.inf))
        else:
            return Interval(float_split(gamma_precise(float(x))))
    elif isinstance(x, SupportsRichFloat):
        x = Interval(float_split(x))
    else:
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    intervals = []
    iterator = iter(x.__as_interval__()._endpoints)
    with localcontext() as ctx:
        ctx.prec += 5
        for lower, upper in zip(iterator, iterator):
            if lower > LGAMMA_POS_MIN_X:
                L = gamma_down(lower)
                U = gamma_up(upper)
            elif math.isinf(lower):
                return Interval((-math.inf, math.inf))
            elif lower == upper < 0.5 and lower.is_integer():
                intervals.append((-math.inf, -math.inf))
                L = U = math.inf
            elif 0.0 <= lower <= upper < LGAMMA_POS_MIN_X:
                L = gamma_down(upper)
                U = gamma_up(lower)
            elif lower >= 0.0:
                L = float_down(LGAMMA_POS_MIN_Y.exp())
                U = max(gamma_up(lower), gamma_up(upper))
            elif digamma(Interval((lower, upper))).maximum <= 0.0:
                if math.floor(lower) % 2 == 0:
                    L = gamma_down(upper)
                    U = gamma_up(lower)
                else:
                    L = gamma_down(lower)
                    U = gamma_up(upper)
            elif digamma(Interval((lower, upper))).minimum >= 0.0:
                if math.floor(lower) % 2 == 0:
                    L = gamma_down(lower)
                    U = gamma_up(upper)
                else:
                    L = gamma_down(upper)
                    U = gamma_up(lower)
            elif digamma_down(lower) <= 0.0:
                n = math.floor(lower)
                if n % 2 == 0:
                    L = gamma_extrema(n)
                    if upper > LGAMMA_POS_MIN_X:
                        L = min(L, float_down(LGAMMA_POS_MIN_Y.exp()))
                    U = math.inf
                    intervals.append((L, U))
                    if n + 1 >= upper:
                        continue
                    L, U = digamma_root(n)
                    M = (Decimal(L) + Decimal(U)) / 2
                    L = -math.inf
                    if M + (n + 1) <= upper or n + 1 < upper and digamma_up(upper) >= 0.0:
                        U = gamma_extrema(n + 1)
                    else:
                        U = gamma_up(upper)
                    intervals.insert(-1, (L, U))
                    continue
                else:
                    L = -math.inf
                    U = gamma_extrema(n)
                    intervals.append((L, U))
                    if n + 1 >= upper:
                        continue
                    L, U = digamma_root(n)
                    M = (Decimal(L) + Decimal(U)) / 2
                    if M + (n + 1) <= upper or n + 1 < upper and digamma_up(upper) >= 0.0:
                        L = gamma_extrema(n + 1)
                    else:
                        L = gamma_down(upper)
                    if upper > LGAMMA_POS_MIN_X:
                        L = min(L, float_down(LGAMMA_POS_MIN_Y.exp()))
                    U = math.inf
            else:
                n = math.floor(lower)
                if n % 2 == 0:
                    L = -math.inf
                    U = gamma_extrema(n + 1)
                    intervals.append((L, U))
                    if n + 2 >= upper:
                        continue
                    L, U = digamma_root(n + 1)
                    M = (Decimal(L) + Decimal(U)) / 2
                    if M + (n + 2) <= upper or n + 2 < upper and digamma_up(upper) >= 0.0:
                        L = min(gamma_down(lower), gamma_extrema(n + 2))
                    else:
                        L = min(gamma_down(lower), gamma_down(upper))
                    U = math.inf
                else:
                    L = gamma_extrema(n + 1)
                    U = math.inf
                    intervals.append((L, U))
                    if n + 2 >= upper:
                        continue
                    L, U = digamma_root(n + 1)
                    M = (Decimal(L) + Decimal(U)) / 2
                    L = -math.inf
                    if M + (n + 2) <= upper or n + 2 < upper and digamma_up(upper) >= 0.0:
                        U = min(gamma_up(lower), gamma_extrema(n + 2))
                    else:
                        U = min(gamma_up(lower), gamma_up(upper))
                    intervals.insert(-1, (L, U))
                    continue
            intervals.append((L, U))
    return Interval(*intervals)

def gamma_down(x: float) -> float:
    y = float_down(gamma_precise(x))
    if 0 < x < math.inf and math.isinf(y):
        return math.nextafter(y, 0.0)
    else:
        return y

def gamma_up(x: float) -> float:
    return float_up(gamma_precise(x))

def hypot(*coordinates: Union[Interval, RealLike]) -> Interval:
    for x in coordinates:
        if isinstance(x, get_args(RealLike)):
            if isinstance(x, SupportsIndex):
                x = operator.index(x)
            x = Interval(float_split(x))
        elif not isinstance(x, Interval):
            raise TypeError(NOT_INTERVAL.format(repr(x)))
    if len(coordinates) == 0:
        return Interval((0, 0))
    elif isinstance(coordinates[0], Interval):
        zero = 0 * coordinates[0]
    else:
        zero = Interval((0, 0))
    return dist(coordinates, (zero,) * len(coordinates))

def log(x: Union[Interval, RealLike], base: Optional[Union[Interval, RealLike]] = None) -> Interval:
    if isinstance(x, (Decimal, Interval)):
        pass
    elif isinstance(x, SupportsIndex):
        x = Decimal(operator.index(x))
    elif isinstance(x, SupportsRichFloat):
        x = Interval(float_split(x))
    else:
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if base is None or isinstance(base, (Decimal, Interval)):
        pass
    elif isinstance(base, SupportsIndex):
        base = Decimal(operator.index(base))
    elif isinstance(base, SupportsRichFloat):
        base = Interval(float_split(base))
    else:
        raise TypeError(NOT_INTERVAL.format(repr(base)))
    if isinstance(x, Interval):
        if x[0:] == Interval():
            return Interval()
    elif x < 0:
        return Interval()
    if isinstance(base, Interval):
        if base[0:] == Interval():
            return Interval()
    elif None is not base < 0:
        return Interval()
    if x == 0:
        return -inf if base is None else -inf / log(base)
    elif x == 1:
        if isinstance(base, Interval) and 1 in base or base is not None and base == 1:
            return interval
        else:
            return Interval((0, 0))
    elif x == math.inf:
        return inf if base is None else inf / log(base)
    elif base == 0:
        return -log(x) / inf
    elif base == 1:
        return log(x) / Interval((0, 0))
    elif base == math.inf:
        return log(x) / inf
    elif isinstance(x, Decimal):
        if base is None:
            return Interval(float_split(x.ln()))
        elif base == x:
            return Interval((1, 1))
        elif isinstance(base, Interval):
            iterator = iter(base[0:].__as_interval__()._endpoints)
        else:
            iterator = iter((base, base))
        return Interval(*[
            (*sorted((x.logb(lower), x.logb(upper))),)
            for lower, upper in zip(iterator, iterator)
        ])
    elif isinstance(base, Decimal):
        iterator = iter(x._endpoints)
        if base > 1:
            return Interval(*[
                (
                    float_down(Decimal(lower).logb(base)),
                    float_up(Decimal(upper).logb(base)),
                )
                for lower, upper in zip(iterator, iterator)
            ])
        else:
            return Interval(*[
                (
                    float_down(Decimal(upper).logb(base)),
                    float_up(Decimal(lower).logb(base)),
                )
                for lower, upper in zip(iterator, iterator)
            ])
    x = x[0:].__as_interval__()
    if base is None:
        iterator = iter(x._endpoints)
        return Interval(*[
            (log_down(lower), log_up(upper))
            for lower, upper in zip(iterator, iterator)
        ])
    elif base == 10.0:
        iterator = iter(x._endpoints)
        return Interval(*[
            (log_down(lower, 10.0), log_up(upper, 10.0))
            for lower, upper in zip(iterator, iterator)
        ])
    elif not isinstance(base, Interval):
        base = Interval(float_split(base))
    base = base[0:].__as_interval__()
    return Interval(
        *[
            (log_down(XL, BU), log_up(XU, BL))
            for XL, XU in zip(*[iter(x._endpoints)] * 2)
            for BL, BU in zip(*[iter(base[:1]._endpoints)] * 2)
        ],
        *[
            (log_down(XL, BU), log_up(XU, BL))
            for XL, XU in zip(*[iter(x._endpoints)] * 2)
            for BL, BU in zip(*[iter(base[1:]._endpoints)] * 2)
        ],
    )

def log_down(x: float, base: Optional[float] = None) -> float:
    if base is None:
        if x == 0.0:
            return -math.inf
        return float_down(Decimal(x).ln())
    elif base == 0.0:
        if math.isinf(x):
            return -math.inf
        else:
            return 0.0
    elif base == 1.0:
        return -math.inf
    elif base == 10.0:
        if x == 0.0:
            return -math.inf
        else:
            return float_down(Decimal(x).log10())
    elif math.isinf(base):
        if x == 0.0:
            return -math.inf
        else:
            return 0.0
    else:
        if x == 0.0:
            return math.inf if base < 1.0 else -math.inf
        else:
            return float_down(Decimal(x).logb(base))

def log_up(x: float, base: Optional[float] = None) -> float:
    if base is None:
        if x == 0.0:
            return -math.inf
        return float_up(Decimal(x).ln())
    elif base == 0.0:
        if x == 0.0:
            return math.inf
        else:
            return 0.0
    elif base == 1.0:
        return math.inf
    elif base == 10.0:
        if x == 0.0:
            return -math.inf
        else:
            return float_up(Decimal(x).log10())
    elif math.isinf(base):
        if math.isinf(x):
            return math.inf
        else:
            return 0.0
    else:
        if x == 0.0:
            return -math.inf if base > 1.0 else math.inf
        else:
            return float_up(Decimal(x).logb(base))

def log10(x: Union[Interval, RealLike]) -> Interval:
    return log(x, 10)

def log1p(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, SupportsIndex):
        x = Decimal(operator.index(x))
    if isinstance(x, Decimal):
        if abs(x) < 1e-4:
            return Interval(float_split(log1p_precise(x)))
        else:
            return Interval(float_split((x + 1).ln()))
    elif isinstance(x, SupportsRichFloat):
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if not isinstance(x, Interval):
        x = Interval(float_split(x))
    iterator = iter(x.__as_interval__()[-1.0:]._endpoints)
    return Interval(*[
        (log1p_down(lower), log1p_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def log1p_precise(x: Union[Decimal, float]) -> Decimal:
    with localcontext() as ctx:
        ctx.prec += 2
        d = Decimal(x)
        d /= d + 2
        t = 2 * d
        d **= 2
        s = 0
        for n in range(1, 6, 2):
            s += t / n
            t *= d
        return s

def log1p_down(x: float) -> float:
    if abs(x) < 1e-4:
        return float_down(log1p_precise(x))
    else:
        return log_down(x + 1)

def log1p_up(x: float) -> float:
    if abs(x) < 1e-4:
        return float_up(log1p_precise(x))
    else:
        return log_up(x + 1)

def log2(x: Union[Interval, RealLike]) -> Interval:
    return log(x, 2)

def pow(x: Union[Interval, RealLike], y: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    return x ** y

def radians(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    return x / 180 * pi

def sin(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, (Decimal, SupportsIndex)):
        if isinstance(x, Decimal) and x.is_infinite():
            return Interval((-1.0, 1.0))
        return Interval(float_split(cos_sin_precise(x)[1]))
    elif not isinstance(x, Interval):
        x = Interval(float_split(x))
    elif isinstance(x, PiMultiple):
        intervals = []
        for sub_interval in x.sub_intervals:
            lower = sym_mod(sub_interval._endpoints[0], 1)
            upper = sub_interval._endpoints[1] - sub_interval._endpoints[0] + lower
            if upper - lower > 2.0 or math.isinf(lower) or math.isinf(upper):
                return Interval((-1.0, 1.0))
            _upper = sym_mod(upper, 1)
            if lower < -0.75:
                L = -sin_precise((Decimal(lower) + 1) * _BIG_PI)
            elif lower < -0.25:
                L = -cos_precise((Decimal(lower) + Decimal("0.5")) * _BIG_PI)
            elif lower <= 0.25:
                L = sin_precise(Decimal(lower) * _BIG_PI)
            elif lower <= 0.75:
                L = cos_precise((Decimal(lower) - Decimal("0.5")) * _BIG_PI)
            else:
                L = -sin_precise((Decimal(lower) - 1) * _BIG_PI)
            if _upper < -0.75:
                U = -sin_precise((Decimal(_upper) + 1) * _BIG_PI)
            elif _upper < -0.25:
                U = -cos_precise((Decimal(_upper) + Decimal("0.5")) * _BIG_PI)
            elif _upper <= 0.25:
                U = sin_precise(Decimal(_upper) * _BIG_PI)
            elif _upper <= 0.75:
                U = cos_precise((Decimal(_upper) - Decimal("0.5")) * _BIG_PI)
            else:
                U = -sin_precise((Decimal(_upper) - 1) * _BIG_PI)
            intervals.append((
                -1.0
                if lower <= -0.5 <= upper or 1.5 <= upper
                else float_down(min(L, U)),
                1.0
                if lower <= 0.5 <= upper or 2.5 <= upper
                else float_up(max(L, U)),
            ))
        return Interval(*intervals)
    x = x.__as_interval__()
    if len(x._endpoints) == 0:
        return x
    elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
        return interval[-1.0:1.0]
    intervals = []
    iterator = iter(x._endpoints)
    for lower, upper in zip(iterator, iterator):
        with localcontext() as ctx:
            ctx.prec += 2 * int(math.log10(1 + abs(lower))) + 10
            _L = sym_mod(Decimal(lower), _BIG_PI)
            _U = Decimal(upper) - Decimal(lower) + _L
        SL = cos_sin_precise(lower)[1]
        SU = cos_sin_precise(upper)[1]
        if _L <= -_BIG_PI / 2 <= _U or 3 * _BIG_PI / 2 <= _U:
            L = -1.0
        else:
            L = float_down(min(SL, SU))
        if _L <= _BIG_PI / 2 <= _U or 5 * _BIG_PI / 2 <= _U:
            U = 1.0
        else:
            U = float_up(max(SL, SU))
        intervals.append((L, U))
    return Interval(*intervals)

def sin_down(x: float) -> float:
    return float_down(cos_sin_precise(x)[1])

def sin_up(x: float) -> float:
    return float_up(cos_sin_precise(x)[1])

def sinh(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (
            min(sinh_down(lower), sinh_down(upper)),
            max(sinh_up(lower), sinh_up(upper)),
        )
        for lower, upper in zip(iterator, iterator)
    ])

def sinh_down(x: float) -> float:
    x = Decimal(x)
    try:
        return float_down((x.exp() - (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return float_down((x - Decimal(2).ln()).exp() - (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf if x > 0 else -math.inf

def sinh_up(x: float) -> float:
    x = Decimal(x)
    try:
        return float_up((x.exp() - (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return float_up((x - Decimal(2).ln()).exp() - (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf if x > 0 else -math.inf

def sqrt(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()[0:]._endpoints)
    return Interval(*[
        (sqrt_down(lower), sqrt_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def sqrt_down(x: float) -> float:
    y = math.sqrt(x)
    partials = mul_precise(y, y)
    if partials[-1] > x:
        return math.nextafter(y, 0.0)
    elif partials[-1] < x or len(partials) == 1 or partials[-2] < 0.0:
        return y
    else:
        return math.nextafter(y, 0.0)

def sqrt_up(x: float) -> float:
    y = math.sqrt(x)
    partials = mul_precise(y, y)
    if partials[-1] < x:
        return math.nextafter(y, math.inf)
    elif partials[-1] > x or len(partials) == 1 or partials[-2] > 0.0:
        return y
    else:
        return math.nextafter(y, math.inf)

def tan(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, (Decimal, SupportsIndex)):
        if isinstance(x, Decimal) and x.is_infinite():
            return interval
        c, s = cos_sin_precise(x)
        return Interval(float_split(s / c))
    elif not isinstance(x, Interval):
        x = Interval(float_split(x))
    intervals = []
    if isinstance(x, PiMultiple):
        for sub_interval in x.sub_intervals:
            lower = sym_mod(sub_interval._endpoints[0], 0.5)
            upper = sub_interval._endpoints[1] - sub_interval._endpoints[0] + lower
            if lower == upper in (-0.5, 0.5):
                intervals.append((-math.inf, -math.inf))
                intervals.append((math.inf, math.inf))
                continue
            if not upper - lower < 1.0:
                return interval
            _upper = sym_mod(upper, 0.5)
            if abs(lower) <= 0.25:
                c = cos_precise(Decimal(lower) * _BIG_PI)
                s = sin_precise(Decimal(lower) * _BIG_PI)
                L = s / c
            elif lower == -0.5:
                L = Decimal("-Infinity")
            elif lower == 0.5:
                L = Decimal("Infinity")
            else:
                s = -cos_precise(Decimal(lower - lower / abs(lower) * 0.5) * _BIG_PI)
                c = sin_precise(Decimal(lower - lower / abs(lower) * 0.5) * _BIG_PI)
                L = s / c
            if abs(_upper) <= 0.25:
                c = cos_precise(Decimal(_upper) * _BIG_PI)
                s = sin_precise(Decimal(_upper) * _BIG_PI)
                U = s / c
            elif _upper == -0.5:
                U = Decimal("-Infinity")
            elif _upper == 0.5:
                U = Decimal("Infinity")
            else:
                s = -cos_precise(Decimal(_upper - _upper / abs(_upper) * 0.5) * _BIG_PI)
                c = sin_precise(Decimal(_upper - _upper / abs(_upper) * 0.5) * _BIG_PI)
                U = s / c
            if U - L < upper - lower:
                intervals.append((-math.inf, float_up(U)))
                intervals.append((float_down(L), math.inf))
            else:
                intervals.append((float_down(L), float_up(U)))
    else:
        iterator = iter(x.__as_interval__()._endpoints)
        for lower, upper in zip(iterator, iterator):
            if math.isinf(lower) or math.isinf(upper) or Decimal(upper) - Decimal(lower) > _BIG_PI:
                return interval
            c, s = cos_sin_precise(lower)
            L = s / c
            c, s = cos_sin_precise(upper)
            U = s / c
            if U - L < Decimal(upper) - Decimal(lower):
                intervals.append((-math.inf, float_up(U)))
                intervals.append((float_down(L), math.inf))
            else:
                intervals.append((float_down(L), float_up(U)))
    return Interval(*intervals)

def tan_down(x: float) -> float:
    c, s = cos_sin_precise(x)
    return float_down(s / c)

def tan_up(x: float) -> float:
    c, s = cos_sin_precise(x)
    return float_up(s / c)

def tanh(x: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[
        (tanh_down(lower), tanh_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

def tanh_down(x: float) -> float:
    d = 2 * Decimal(x)
    if d > 0:
        t = (1 - (-d).exp()) / (1 + (-d).exp())
    else:
        t = (d.exp() - 1) / (d.exp() + 1)
    return float_down(t)

def tanh_up(x: float) -> float:
    d = 2 * Decimal(x)
    if d > 0:
        t = (1 - (-d).exp()) / (1 + (-d).exp())
    else:
        t = (d.exp() - 1) / (d.exp() + 1)
    return float_up(t)

def unadd(x: Union[Interval, RealLike], y: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    x = x.__as_interval__()
    y = y.__as_interval__()
    result = interval
    for yi in y.sub_intervals:
        intervals = []
        for xi in x.sub_intervals:
            start = sub_down(xi.minimum, yi.minimum)
            stop = sub_up(xi.maximum, yi.maximum)
            if start > stop:
                raise ValueError(f"impossible to unadd {x!r} and {y!r}")
            intervals.append((start, stop))
        result &= Interval(*intervals)
    return result

def undiv(x: Union[Interval, RealLike], y: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    x = x.__as_interval__()
    y = y.__as_interval__()
    result = interval
    intervals = []
    for temp in y.sub_intervals:
        if temp == 0.0:
            continue
        if temp != temp[:0]:
            for yi in temp[0:].sub_intervals:
                for xi in x[0:].sub_intervals:
                    if xi.minimum == 0.0:
                        start = 0.0
                    elif yi.maximum == 0.0:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    else:
                        start = mul_down(xi.minimum, yi.maximum)
                    if xi.maximum == 0.0:
                        stop = 0.0
                    elif yi.minimum == 0.0:
                        stop = math.inf
                    else:
                        stop = mul_up(xi.maximum, yi.minimum)
                    if start > stop:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    intervals.append((start, stop))
                for xi in x[:0].sub_intervals:
                    if xi.maximum == 0.0:
                        stop = 0.0
                    elif yi.maximum == 0.0:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    else:
                        stop = mul_up(xi.maximum, yi.maximum)
                    if xi.minimum == 0.0:
                        start = 0.0
                    elif yi.minimum == 0.0:
                        start = -math.inf
                    else:
                        start = mul_down(xi.minimum, yi.minimum)
                    if start > stop:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    intervals.append((start, stop))
            result &= Interval(*intervals)
            intervals.clear()
        if temp != temp[0:]:
            for yi in temp[:0].sub_intervals:
                for xi in x[0:].sub_intervals:
                    if xi.minimum == 0.0:
                        stop = 0.0
                    elif yi.minimum == 0.0:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    else:
                        stop = mul_up(xi.minimum, yi.minimum)
                    if xi.maximum == 0.0:
                        start = 0.0
                    elif yi.maximum == 0.0:
                        start = -math.inf
                    else:
                        start = mul_down(xi.maximum, yi.maximum)
                    if start > stop:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    intervals.append((start, stop))
                for xi in x[:0].sub_intervals:
                    if xi.maximum == 0.0:
                        start = 0.0
                    elif yi.minimum == 0.0:
                        start = -math.inf
                    else:
                        start = mul_down(xi.maximum, yi.minimum)
                    if xi.minimum == 0.0:
                        stop = 0.0
                    elif yi.maximum == 0.0:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    else:
                        stop = mul_up(xi.minimum, yi.maximum)
                    if start > stop:
                        raise ValueError(f"impossible to undiv {x!r} and {y!r}")
                    intervals.append((start, stop))
            result &= Interval(*intervals)
            intervals.clear()
    return result

def unmul(x: Union[Interval, RealLike], y: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    x = x.__as_interval__()
    y = y.__as_interval__()
    result = interval
    intervals = []
    for temp in y.sub_intervals:
        if temp != temp[:0]:
            for yi in temp[0:].sub_intervals:
                for xi in x[0:].sub_intervals:
                    if xi.minimum == 0.0:
                        start = 0.0
                    elif yi.minimum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        start = div_down(xi.minimum, yi.minimum)
                    if xi.maximum == 0.0:
                        stop = 0.0
                    elif yi.maximum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        stop = div_up(xi.maximum, yi.maximum)
                    if start > stop:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    intervals.append((start, stop))
                for xi in x[:0].sub_intervals:
                    if xi.maximum == 0.0:
                        stop = 0.0
                    elif yi.minimum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        stop = div_up(xi.maximum, yi.minimum)
                    if xi.minimum == 0.0:
                        start = 0.0
                    elif yi.maximum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        start = div_down(xi.minimum, yi.maximum)
                    if start > stop:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    intervals.append((start, stop))
            result &= Interval(*intervals)
            intervals.clear()
        if temp != temp[0:]:
            for yi in temp[:0].sub_intervals:
                for xi in x[0:].sub_intervals:
                    if xi.minimum == 0.0:
                        stop = 0.0
                    elif yi.maximum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        stop = div_up(xi.minimum, yi.maximum)
                    if xi.maximum == 0.0:
                        start = 0.0
                    elif yi.minimum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        start = div_down(xi.maximum, yi.minimum)
                    if start > stop:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    intervals.append((start, stop))
                for xi in x[:0].sub_intervals:
                    if xi.maximum == 0.0:
                        start = 0.0
                    elif yi.maximum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        start = div_down(xi.maximum, yi.maximum)
                    if xi.minimum == 0.0:
                        stop = 0.0
                    elif yi.minimum == 0.0:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    else:
                        stop = div_up(xi.minimum, yi.minimum)
                    if start > stop:
                        raise ValueError(f"impossible to unmul {x!r} and {y!r}")
                    intervals.append((start, stop))
            result &= Interval(*intervals)
            intervals.clear()
    return result

def unrdiv(x: Union[Interval, RealLike], y: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    x = x.__as_interval__()
    y = y.__as_interval__()
    result = interval
    intervals = []
    for temp in y.sub_intervals:
        if temp == 0.0:
            continue
        if temp != temp[:0]:
            for yi in temp[0:].sub_intervals:
                for xi in x[0:].sub_intervals:
                    if xi.maximum == 0.0:
                        continue
                    if yi.maximum == 0.0:
                        start = 0.0
                    elif xi.maximum == 0.0:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    else:
                        start = div_down(yi.maximum, xi.maximum)
                    if yi.minimum == 0.0:
                        stop = 0.0
                    elif xi.minimum == 0.0:
                        stop = math.inf
                    else:
                        stop = div_up(yi.minimum, xi.minimum)
                    if start > stop:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    intervals.append((start, stop))
                for xi in x[:0].sub_intervals:
                    if xi.minimum == 0.0:
                        continue
                    if yi.minimum == 0.0:
                        stop = 0.0
                    elif xi.maximum == 0.0:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    else:
                        stop = div_up(yi.minimum, xi.maximum)
                    if yi.maximum == 0.0:
                        start = 0.0
                    elif xi.minimum == 0.0:
                        start = -math.inf
                    else:
                        start = div_down(yi.maximum, xi.minimum)
                    if start > stop:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    intervals.append((start, stop))
            result &= Interval(*intervals)
            intervals.clear()
        if temp != temp[0:]:
            for yi in temp[:0].sub_intervals:
                for xi in x[0:].sub_intervals:
                    if xi.maximum == 0.0:
                        continue
                    if yi.minimum == 0.0:
                        stop = 0.0
                    elif xi.maximum == 0.0:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    else:
                        stop = div_up(yi.minimum, xi.maximum)
                    if yi.minimum == 0.0:
                        start = 0.0
                    elif xi.maximum == 0.0:
                        start = -math.inf
                    else:
                        start = div_down(yi.maximum, xi.minimum)
                    if start > stop:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    intervals.append((start, stop))
                for xi in x[:0].sub_intervals:
                    if xi.minimum == 0.0:
                        continue
                    if yi.minimum == 0.0:
                        start = 0.0
                    elif xi.minimum == 0.0:
                        start = -math.inf
                    else:
                        start = div_down(yi.minimum, xi.minimum)
                    if yi.maximum == 0.0:
                        stop = 0.0
                    elif xi.maximum == 0.0:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    else:
                        stop = div_up(yi.maximum, xi.maximum)
                    if start > stop:
                        raise ValueError(f"impossible to unrdiv {x!r} and {y!r}")
                    intervals.append((start, stop))
            result &= Interval(*intervals)
            intervals.clear()
    return result

def unrsub(x: Union[Interval, RealLike], y: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    x = x.__as_interval__()
    y = y.__as_interval__()
    result = interval
    for yi in y.sub_intervals:
        intervals = []
        for xi in x.sub_intervals:
            start = sub_down(yi.maximum, xi.maximum)
            stop = sub_up(yi.minimum, xi.minimum)
            if start > stop:
                raise ValueError(f"impossible to unrsub {x!r} and {y!r}")
            intervals.append((start, stop))
        result &= Interval(*intervals)
    return result

def unsub(x: Union[Interval, RealLike], y: Union[Interval, RealLike]) -> Interval:
    if isinstance(x, get_args(RealLike)):
        if isinstance(x, SupportsIndex):
            x = operator.index(x)
        x = Interval(float_split(x))
    elif not isinstance(x, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(x)))
    if isinstance(y, get_args(RealLike)):
        if isinstance(y, SupportsIndex):
            y = operator.index(y)
        y = Interval(float_split(y))
    elif not isinstance(y, Interval):
        raise TypeError(NOT_INTERVAL.format(repr(y)))
    x = x.__as_interval__()
    y = y.__as_interval__()
    result = interval
    for yi in y.sub_intervals:
        intervals = []
        for xi in x.sub_intervals:
            start = add_down(xi.minimum, yi.maximum)
            stop = add_up(xi.maximum, yi.minimum)
            if start > stop:
                raise ValueError(f"impossible to unsub {x!r} and {y!r}")
            intervals.append((start, stop))
        result &= Interval(*intervals)
    return result
