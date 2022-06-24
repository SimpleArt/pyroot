import decimal
import math
import operator
from decimal import Decimal, localcontext
from typing import Any, Optional, SupportsFloat, SupportsIndex, TypeVar, Union, overload

from .fpu_rounding import *
from .interval import Interval, interval

e = interval[math.e:math.nextafter(math.e, math.inf)]
_PI = interval[math.pi:math.nextafter(math.pi, math.inf)]
_BIG_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592"
    "30781640628620899862803482534211706798214808651328230664709384460"
    "95505822317253594081284811174502841027019385211055596446229489549"
    "30381964428810975665933446128475648233786783165271201909145648566"
    "923460348610454326648213393607260249141273724587006606"
)

Self = TypeVar("Self", bound="PiMultiple")


class PiMultiple(Interval):

    __slots__ = ()

    def __abs__(self: Self) -> Self:
        iterator = iter(abs(self.coefficients)._endpoints)
        return type(self)(*zip(iterator, iterator))

    def __add__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__add__ is type(other).__add__:
            iterator = iter((self.coefficients + other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__add__ is type(other).__add__:
            return self.__as_interval__() + other
        elif isinstance(other, SupportsFloat):
            return self.__as_interval__() + float(other)
        else:
            return NotImplemented

    def __and__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__and__ is type(other).__and__:
            iterator = iter((self.coefficients & other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__and__ is type(other).__and__:
            return self.__as_interval__() & other
        elif isinstance(other, SupportsFloat):
            return self.__as_interval__() & float(other)
        else:
            return NotImplemented

    def __as_interval__(self: Self) -> Interval:
        return self.coefficients * _PI

    def __eq__(self: Self, other: Any) -> bool:
        if isinstance(other, PiMultiple) and type(self).__eq__ is type(other).__eq__:
            return self._endpoints == other._endpoints
        elif isinstance(other, Interval) and Interval.__eq__ is type(other).__eq__:
            return all(
                e1 * _PI.minimum == e1 * _PI.maximum == e2
                for e1, e2 in zip(self._endpoints, other._endpoints)
            )
        elif isinstance(other, SupportsFloat):
            other = float(other)
            return self == Interval((other, other))
        else:
            return NotImplemented

    def __getitem__(self: Self, args: Union[slice, tuple[slice, ...]]) -> Interval:
        return self.__as_interval__()[args]

    def __mul__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__mul__ is type(other).__mul__:
            return self.__as_interval__() * (other.coefficients * _PI)
        elif isinstance(other, Interval) and Interval.__mul__ is type(other).__mul__:
            iterator = iter((self.coefficients * other)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, SupportsFloat):
            iterator = iter((self.coefficients * float(other))._endpoints)
            return type(self)(*zip(iterator, iterator))
        else:
            return NotImplemented

    def __or__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__or__ is type(other).__or__:
            iterator = iter((self.coefficients | other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__or__ is type(other).__or__:
            iterator = iter((self.__as_interval__() | other)._endpoints)
            return Interval(*zip(iterator, iterator))
        elif isinstance(other, SupportsFloat):
            iterator = iter((self.__as_interval__() | float(other))._endpoints)
            return Interval(*zip(iterator, iterator))
        else:
            return NotImplemented

    def __pow__(self: Self, other: Union[Interval, float], modulo: None = None) -> Interval:
        if modulo is not None:
            return NotImplemented
        elif isinstance(other, PiMultiple) and type(self).__pow__ is type(other).__pow__:
            iterator = iter((self.__as_interval__() ** (other.coefficients * _PI))._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__pow__ is type(other).__pow__:
            iterator = iter((self.__as_interval__() ** other)._endpoints)
            return Interval(*zip(iterator, iterator))
        elif isinstance(other, SupportsIndex):
            iterator = iter((self.__as_interval__() ** operator.index(other))._endpoints)
            return Interval(*zip(iterator, iterator))
        elif isinstance(other, SupportsFloat):
            iterator = iter((self.__as_interval__() ** float(other))._endpoints)
            return Interval(*zip(iterator, iterator))
        else:
            return NotImplemented

    def __repr__(self: Self) -> str:
        if self == pi:
            return "pi"
        else:
            return super().__repr__() + " * pi"

    def __rmul__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, Interval):
            iterator = iter((other * self.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, SupportsFloat):
            iterator = iter((float(other) * self.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        else:
            return NotImplemented

    def __sub__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__sub__ is type(other).__sub__:
            iterator = iter((self.coefficients - other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__sub__ is type(other).__sub__:
            return self.__as_interval__() - other
        elif isinstance(other, SupportsFloat):
            return self.__as_interval__() - float(other)
        else:
            return NotImplemented

    def __truediv__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__truediv__ is type(other).__truediv__:
            return self.coefficients / other.coefficients
        elif isinstance(other, Interval) and Interval.__truediv__ is type(other).__truediv__:
            iterator = iter((self.coefficients / other)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, SupportsFloat):
            iterator = iter((self.coefficients / float(other))._endpoints)
            return type(self)(*zip(iterator, iterator))
        else:
            return NotImplemented

    def __xor__(self: Self, other: Union[Interval, float]) -> Interval:
        if isinstance(other, PiMultiple) and type(self).__xor__ is type(other).__xor__:
            iterator = iter((self.coefficients ^ other.coefficients)._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, Interval) and Interval.__xor__ is type(other).__xor__:
            return self.__as_interval__() ^ other
        elif isinstance(other, SupportsFloat):
            return self.__as_interval__() ^ float(other)
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
    if x >= 0:
        x %= 2 * modulo
        if x > modulo:
            x -= 2 * modulo
    else:
        x = -((-x) % (2 * modulo))
        if x < -modulo:
            x += 2 * modulo
    return x

def decimal_down(x: Decimal) -> float:
    y = float(x)
    if x < y:
        return math.nextafter(y, -math.inf)
    else:
        return y

def decimal_up(x: Decimal) -> float:
    y = float(x)
    if x > y:
        return math.nextafter(y, math.inf)
    else:
        return y

def cos_precise(x: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec += 2
        if x >= 0:
            x %= (2 * _BIG_PI)
            if x > _BIG_PI:
                x -= _BIG_PI
        else:
            x = -((-x) % (2 * _BIG_PI))
            if x < -_BIG_PI:
                x += _BIG_PI
        i = last_s = 0
        s = fact = num = sign = 1
        x *= x
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
        if x >= 0:
            x %= (2 * _BIG_PI)
            if x > _BIG_PI:
                x -= _BIG_PI
        else:
            x = -((-x) % (2 * _BIG_PI))
            if x < -_BIG_PI:
                x += _BIG_PI
        s = num = x
        last_s = 0
        i = fact = sign = 1
        x *= x
        while s != last_s:
            last_s = s
            i += 2
            fact *= i * (i - 1)
            num *= x
            sign *= -1
            s += sign * num / fact
        return s

def cos_sin_precise(x: float) -> tuple[Decimal, Decimal]:
    with localcontext() as ctx:
        ctx.prec += int(2 * math.log10(1 + abs(x))) + 10
        c = cos_precise(Decimal(abs(x)))
        s = sin_precise(Decimal(abs(x)))
        if x < 0:
            return (c, -s)
        else:
            return (c, s)

def acos(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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

def acosh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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

def asin(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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

def asinh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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

def atan(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    iterator = iter(x.__as_interval__()._endpoints)
    intervals = []
    result = PiMultiple()
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
            result |= interval[L:U] * pi
            if result._endpoints == (-0.5, 0.5):
                return result
        else:
            intervals.append((atan_down(lower), atan_up(upper)))
    if len(intervals) > 0:
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

def atan2(y: Union[Interval, float], x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if not isinstance(y, Interval):
        y = float(y)
        y = Interval((y, y))
    intervals = []
    result = PiMultiple()
    for xi in x[0:].sub_intervals:
        if xi.maximum == 0.0:
            continue
        for yi in y[0:].sub_intervals:
            if math.isinf(yi.maximum) or xi.minimum == 0.0:
                U = 0.5
            elif yi.maximum == 0.0 or math.isinf(xi.minimum):
                U = 0.0
            elif (
                math.frexp(yi.maximum)[0] == math.frexp(xi.minimum)[0]
                and math.frexp(yi.maximum)[1] == math.frexp(xi.minimum)[1] - 2
            ):
                U = 0.25
            else:
                intervals.append((atan2_down(yi.minimum, xi.maximum), atan2_up(yi.maximum, xi.minimum)))
                continue
            if yi.minimum == 0.0 or math.isinf(xi.maximum):
                L = 0.0
            elif (
                math.frexp(yi.minimum)[0] == math.frexp(xi.maximum)[0]
                and math.frexp(yi.minimum)[1] == math.frexp(xi.maximum)[1] - 2
            ):
                L = 0.25
            else:
                intervals.append((atan2_down(yi.minimum, xi.maximum), atan2_up(yi.maximum, xi.minimum)))
                continue
            result |= Interval((L, U)) * pi
        for yi in y[:0].sub_intervals:
            if math.isinf(yi.minimum) or xi.minimum == 0.0:
                L = -0.5
            elif yi.minimum == 0.0 or math.isinf(xi.minimum):
                L = -0.0
            elif (
                math.frexp(yi.minimum)[0] == -math.frexp(xi.minimum)[0]
                and math.frexp(yi.minimum)[1] == math.frexp(xi.minimum)[1] - 2
            ):
                L = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            if yi.maximum == 0.0 or math.isinf(xi.maximum):
                U = -0.0
            elif (
                math.frexp(yi.maximum)[0] == -math.frexp(xi.maximum)[0]
                and math.frexp(yi.maximum)[1] == math.frexp(xi.maximum)[1] - 2
            ):
                U = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            result |= Interval((L, U)) * pi
    for xi in x[:0].sub_intervals:
        if xi.minimum == 0.0:
            continue
        for yi in y[0:].sub_intervals:
            if math.isinf(yi.maximum) or xi.maximum == 0.0:
                L = -0.5
            elif yi.maximum == 0.0 or math.isinf(xi.maximum):
                L = -0.0
            elif (
                math.frexp(yi.maximum)[0] == -math.frexp(xi.maximum)[0]
                and math.frexp(yi.maximum)[1] == math.frexp(xi.maximum)[1] - 2
            ):
                L = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            if yi.minimum == 0.0 or math.isinf(xi.minimum):
                U = -0.0
            elif (
                math.frexp(yi.minimum)[0] == -math.frexp(xi.minimum)[0]
                and math.frexp(yi.minimum)[1] == math.frexp(xi.minimum)[1] - 2
            ):
                U = -0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.maximum), atan2_up(yi.minimum, xi.minimum)))
                continue
            result |= Interval((L, U)) * pi
        for yi in y[:0].sub_intervals:
            if math.isinf(yi.minimum) or xi.maximum == 0.0:
                U = 0.5
            elif yi.minimum == 0.0 or math.isinf(xi.maximum):
                U = 0.0
            elif (
                math.frexp(yi.minimum)[0] == math.frexp(xi.maximum)[0]
                and math.frexp(yi.minimum)[1] == math.frexp(xi.maximum)[1] - 2
            ):
                U = 0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.minimum), atan2_up(yi.minimum, xi.maximum)))
                continue
            if yi.maximum == 0.0 or math.isinf(xi.minimum):
                L = 0.0
            elif (
                math.frexp(yi.maximum)[0] == math.frexp(xi.minimum)[0]
                and math.frexp(yi.maximum)[1] == math.frexp(xi.minimum)[1] - 2
            ):
                L = 0.25
            else:
                intervals.append((atan2_down(yi.maximum, xi.minimum), atan2_up(yi.minimum, xi.maximum)))
                continue
            result |= Interval((L, U)) * pi
    if len(intervals) > 0:
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

def atanh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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

def cos(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    elif isinstance(x, PiMultiple):
        intervals = []
        for sub_interval in abs(x).sub_intervals:
            lower = sym_mod(sub_interval._endpoints[0], 1)
            upper = sub_interval._endpoints[1] - sub_interval._endpoints[0] + lower
            if not upper - lower < 2.0:
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
                else decimal_down(min(L, U)),
                1.0
                if lower <= 0.0 <= upper or 2.0 <= upper
                else decimal_up(max(L, U)),
            ))
        return Interval(*intervals)
    x = abs(x.__as_interval__())
    if len(x._endpoints) == 0:
        return x
    elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
        return interval[-1.0:1.0]
    iterator = iter(x._endpoints)
    return Interval(*[
        (
            -1.0
            if L == -_BIG_PI or _BIG_PI <= U
            else min(cos_down(lower), cos_down(upper)),
            1.0
            if L <= 0 <= U or 2 * _BIG_PI <= U
            else max(cos_up(lower), cos_up(upper)),
        )
        for lower, upper in zip(iterator, iterator)
        for L in [sym_mod(Decimal(lower), _BIG_PI)]
        for U in [Decimal(upper) - Decimal(lower) + L]
    ])

def cos_down(x: float) -> float:
    return decimal_down(cos_sin_precise(x)[0])

def cos_up(x: float) -> float:
    return decimal_up(cos_sin_precise(x)[0])

def cosh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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
        return decimal_down((x.exp() + (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return decimal_down((x - Decimal(2).ln()).exp() + (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf

def cosh_up(x: float) -> float:
    x = Decimal(x)
    try:
        return decimal_up((x.exp() + (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return decimal_up((x - Decimal(2).ln()).exp() + (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf

def exp(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    iterator = iter(x.__as_interval__()._endpoints)
    return Interval(*[(exp_down(lower), exp_up(upper)) for lower, upper in zip(iterator, iterator)])

def exp_down(x: float) -> float:
    try:
        return decimal_down(Decimal(x).exp())
    except decimal.Overflow:
        return math.nextafter(math.inf, x)

def exp_up(x: float) -> float:
    try:
        result = decimal_up(Decimal(x).exp())
    except decimal.Overflow:
        return math.inf
    if result != 0.0 or math.isinf(x):
        return result
    else:
        return math.nextafter(0.0, 1.0)

def log(x: Union[Interval, float], base: Optional[Union[Interval, float]] = None) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    x = x.__as_interval__()
    if base is None:
        iterator = iter(x[0:]._endpoints)
        return Interval(*[
            (log_down(lower), log_up(upper))
            for lower, upper in zip(iterator, iterator)
        ])
    elif isinstance(base, Interval):
        base = base.__as_interval__()[0:]
        return Interval(
            *[
                (log_down(XL, BU), log_up(XU, BL))
                for XL, XU in zip(*[iter(x[0:]._endpoints)] * 2)
                for BL, BU in zip(*[iter(base[0:1]._endpoints)] * 2)
            ],
            *[
                (log_down(XL, BU), log_up(XU, BL))
                for XL, XU in zip(*[iter(x[0:]._endpoints)] * 2)
                for BL, BU in zip(*[iter(base[1:]._endpoints)] * 2)
            ],
        )
    else:
        iterator = iter(x[0:]._endpoints)
        return Interval(*[
            (log_down(lower, base), log_up(lower, base))
            for lower, upper in zip(iterator, iterator)
        ])

def log_down(x: float, base: Optional[float] = None) -> float:
    if base is None:
        if x <= 0.0:
            return -math.inf
        return decimal_down(Decimal(x).ln())
    elif base == 1.0:
        return -math.inf
    else:
        if x <= 0.0:
            return math.inf if base < 1.0 else -math.inf
        return decimal_down(Decimal(x).ln() / Decimal(base).ln())

def log_up(x: float, base: Optional[float] = None) -> float:
    if base is None:
        if x <= 0.0:
            return -math.inf
        return decimal_up(Decimal(x).ln())
    elif base == 1.0:
        return math.inf
    else:
        if x <= 0.0:
            return -math.inf if base > 1.0 else math.inf
        return decimal_up(Decimal(x).ln() / Decimal(base).ln())

def sin(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    elif isinstance(x, PiMultiple):
        intervals = []
        for sub_interval in x.sub_intervals:
            lower = sym_mod(sub_interval._endpoints[0], 1)
            upper = sub_interval._endpoints[1] - sub_interval._endpoints[0] + lower
            if not upper - lower < 2.0:
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
                else decimal_down(min(L, U)),
                1.0
                if lower <= 0.5 <= upper or 2.5 <= upper
                else decimal_up(max(L, U)),
            ))
        return Interval(*intervals)
    x = x.__as_interval__()
    if len(x._endpoints) == 0:
        return x
    elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
        return interval[-1.0:1.0]
    iterator = iter(x._endpoints)
    return Interval(*[
        (
            -1.0
            if L <= -_BIG_PI / 2 <= U or 3 * _BIG_PI / 2 <= U
            else min(sin_down(lower), sin_down(upper)),
            1.0
            if L <= _BIG_PI / 2 <= U or 5 * _BIG_PI / 2 <= U
            else max(sin_up(lower), sin_up(upper)),
        )
        for lower, upper in zip(iterator, iterator)
        for L in [sym_mod(Decimal(lower), _BIG_PI)]
        for U in [Decimal(upper) - Decimal(lower) + L]
    ])

def sin_down(x: float) -> float:
    return decimal_down(cos_sin_precise(x)[1])

def sin_up(x: float) -> float:
    return decimal_up(cos_sin_precise(x)[1])

def sinh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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
        return decimal_down((x.exp() - (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return decimal_down((x - Decimal(2).ln()).exp() - (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf if x > 0 else -math.inf

def sinh_up(x: float) -> float:
    x = Decimal(x)
    try:
        return decimal_up((x.exp() - (-x).exp()) / 2)
    except decimal.Overflow:
        pass
    try:
        return decimal_up((x - Decimal(2).ln()).exp() - (-x - Decimal(2).ln()).exp())
    except decimal.Overflow:
        return math.inf if x > 0 else -math.inf

def sqrt(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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

def tan(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    intervals = []
    if isinstance(x, PiMultiple):
        for sub_interval in x.sub_intervals:
            lower = sym_mod(sub_interval._endpoints[0], 0.5)
            upper = sub_interval._endpoints[1] - sub_interval._endpoints[0] + lower
            if lower == upper in (-0.5, 0.5):
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
                intervals.append((-math.inf, decimal_up(U)))
                intervals.append((decimal_down(L), math.inf))
            else:
                intervals.append((decimal_down(L), decimal_up(U)))
    else:
        iterator = iter(x.__as_interval__()._endpoints)
        for lower, upper in zip(iterator, iterator):
            if Decimal(upper) - Decimal(lower) > _BIG_PI:
                return interval
            c, s = cos_sin_precise(lower)
            L = s / c
            c, s = cos_sin_precise(upper)
            U = s / c
            if U - L < Decimal(upper) - Decimal(lower):
                intervals.append((-math.inf, decimal_up(U)))
                intervals.append((decimal_down(L), math.inf))
            else:
                intervals.append((decimal_down(L), decimal_up(U)))
    return Interval(*intervals)

def tan_down(x: float) -> float:
    c, s = cos_sin_precise(x)
    return decimal_down(s / c)

def tan_up(x: float) -> float:
    c, s = cos_sin_precise(x)
    return decimal_up(s / c)

def tanh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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
    return decimal_down(t)

def tanh_up(x: float) -> float:
    d = 2 * Decimal(x)
    if d > 0:
        t = (1 - (-d).exp()) / (1 + (-d).exp())
    else:
        t = (d.exp() - 1) / (d.exp() + 1)
    return decimal_up(t)

def unadd(x: Union[Interval, float], y: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if not isinstance(y, Interval):
        y = float(x)
        y = Interval((y, y))
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

def undiv(x: Union[Interval, float], y: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if not isinstance(y, Interval):
        y = float(x)
        y = Interval((y, y))
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

def unmul(x: Union[Interval, float], y: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if not isinstance(y, Interval):
        y = float(x)
        y = Interval((y, y))
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

def unrdiv(x: Union[Interval, float], y: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if not isinstance(y, Interval):
        y = float(x)
        y = Interval((y, y))
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

def unrsub(x: Union[Interval, float], y: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if not isinstance(y, Interval):
        y = float(x)
        y = Interval((y, y))
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

def unsub(x: Union[Interval, float], y: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if not isinstance(y, Interval):
        y = float(x)
        y = Interval((y, y))
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
