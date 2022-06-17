import math
from decimal import Decimal, localcontext
from typing import Optional, Union, overload

from .fpu_rounding import *
from .interval import Interval, interval

e = Decimal("2.71828182845904523536028747")
pi = Decimal("3.1415926535897932384626433")

@overload
def cos(x: float) -> float: ...

@overload
def cos(x: Interval) -> Interval: ...

def cos(x):
    if isinstance(x, Interval):
        if len(x._endpoints) == 0:
            return x
        elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
            return interval[-1.0:1.0]
        iterator = iter(x._endpoints)
        return Interval(*[
            (
                -1
                if lower <= (upper // (2 * math.pi)) * (2 * math.pi) + (math.pi / 2) <= upper
                or lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (math.pi / 2) <= upper
                else min(cos_down(lower), cos_down(upper)),
                1
                if lower <= (upper // (2 * math.pi)) * (2 * math.pi) <= upper
                or lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (2 * math.pi) <= upper
                else max(cos_up(lower), cos_up(upper)),
            )
            for lower, upper in zip(iterator, iterator)
        ])
    else:
        return math.cos(x)

def cos_down(x: float) -> float:
    if float(x) * x < math.ulp(x):
        return math.nextafter(1.0, 0.0)
    with localcontext() as ctx:
        ctx.prec += 2
        t = Decimal(x) % (2 * pi)
        t **= 2
        i = lasts = 0
        s = fact = num = sign = 1
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= t
            sign *= -1
            s += sign * num / fact
        result = float(s - Decimal(0.5 * math.ulp(s)))
    if result < -1.0:
        return -1.0
    else:
        return result

def cos_up(x: float) -> float:
    if float(x) * x < math.ulp(x):
        return 1.0
    with localcontext() as ctx:
        ctx.prec += 2
        t = Decimal(x) % (2 * pi)
        t **= 2
        i = lasts = 0
        s = fact = num = sign = 1
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= t
            sign *= -1
            s += sign * num / fact
        result = float(s + Decimal(0.5 * math.ulp(s)))
    if result > 1.0:
        return 1.0
    else:
        return result

@overload
def cosh(x: float) -> float: ...

@overload
def cosh(x: Interval) -> Interval: ...

def cosh(x):
    if isinstance(x, Interval):
        iterator = iter(x._endpoints)
        return Interval(*[
            (
                1.0
                if lower <= 0.0 <= upper
                else min(cosh_down(lower), cosh_down(upper)),
                max(cosh_up(lower), cosh_up(upper)),
            )
            for lower, upper in zip(iterator, iterator)
        ])
    else:
        return math.cosh(x)

def cosh_down(x: float) -> float:
    try:
        result = math.cosh(x)
    except OverflowError:
        return math.inf
    if (e ** Decimal(x) + e ** Decimal(-x)) / 2 < result:
        return math.nextafter(result, 1.0)
    return result

def cosh_up(x: float) -> float:
    try:
        result = math.cosh(x)
    except OverflowError:
        return math.inf
    if (e ** Decimal(x) + e ** Decimal(-x)) / 2 > result:
        return math.nextafter(result, result + 1)
    return result

@overload
def exp(x: float) -> float: ...

@overload
def exp(x: Interval) -> Interval: ...

def exp(x):
    if isinstance(x, Interval):
        iterator = iter(x._endpoints)
        return Interval(*[(exp_down(lower), exp_up(upper)) for lower, upper in zip(iterator, iterator)])
    else:
        return math.exp(x)

def exp_down(x: float) -> float:
    try:
        result = math.exp(x)
    except OverflowError:
        return math.inf
    if e ** Decimal(x) < result:
        return math.nextafter(result, 0.0)
    else:
        return result

def exp_up(x: float) -> float:
    try:
        result = math.exp(x)
    except OverflowError:
        return math.inf
    if result == 0.0:
        return math.nextafter(0.0, 1.0)
    elif e ** Decimal(x) > result:
        return math.nextafter(result, 2 * result)
    else:
        return result

@overload
def log(x: float, base: float = ...) -> float: ...

@overload
def log(x: Union[Interval, float], base: Union[Interval, float] = ...) -> Interval: ...

def log(x, base=None):
    if base is None:
        if isinstance(x, Interval):
            iterator = iter(x[0:]._endpoints)
            return Interval(*[
                (log_down(lower), log_up(upper))
                for lower, upper in zip(iterator, iterator)
            ])
        else:
            return math.log(x)
    elif isinstance(base, Interval):
        base = base[0:]
        if isinstance(x, Interval):
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
            return Interval(
                *[
                    (log_down(x, BU), log_up(x, BL))
                    for BL, BU in zip(*[iter(base[0:1]._endpoints)] * 2)
                ],
                *[
                    (log_down(x, BU), log_up(x, BL))
                    for BL, BU in zip(*[iter(base[1:]._endpoints)] * 2)
                ],
            )
    else:
        if isinstance(x, Interval):
            iterator = iter(x[0:]._endpoints)
            return Interval(*[
                (log_down(lower, base), log_up(lower, base))
                for lower, upper in zip(iterator, iterator)
            ])
        else:
            return math.log(x, base)

def log_down(x: float, base: Optional[float] = None) -> float:
    if base is None:
        if x <= 0.0:
            return -math.inf
        y = math.log(x)
        if e ** Decimal(y) > x:
            return math.nextafter(y, y - 1)
    elif base == 1.0:
        return -math.inf
    else:
        if x <= 0.0:
            return math.inf if base < 1.0 else -math.inf
        y = math.log(x, base)
        if Decimal(base) ** Decimal(y) > x:
            return math.nextafter(y, y - 1)
    return y

def log_up(x: float, base: Optional[float] = None) -> float:
    if base is None:
        if x <= 0.0:
            return -math.inf
        y = math.log(x)
        if e ** Decimal(y) < x:
            return math.nextafter(y, y + 1)
    elif base == 1.0:
        return math.inf
    else:
        if x <= 0.0:
            return -math.inf if base > 1.0 else math.inf
        y = math.log(x, base)
        if Decimal(base) ** Decimal(y) < x:
            return math.nextafter(y, y + 1)
    return y

@overload
def sin(x: float) -> float: ...

@overload
def sin(x: Interval) -> Interval: ...

def sin(x):
    if isinstance(x, Interval):
        if len(x._endpoints) == 0:
            return x
        elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
            return interval[-1.0:1.0]
        iterator = iter(x._endpoints)
        return Interval(*[
            (
                -1
                if lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (math.pi / 2) <= upper
                or lower <= (upper // (2 * math.pi)) * (2 * math.pi) + (3 * math.pi / 2) <= upper
                else min(sin_down(lower), sin_down(upper)),
                1
                if lower <= (upper // (2 * math.pi)) * (2 * math.pi) + (math.pi / 2) <= upper
                or lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (3 * math.pi / 2) <= upper
                else max(sin_up(lower), sin_up(upper)),
            )
            for lower, upper in zip(iterator, iterator)
        ])
    else:
        return math.sin(x)

def sin_down(x: float) -> float:
    if x < 0:
        return -sin_up(-x)
    elif float(x) * x * x / 3 < math.ulp(x):
        return math.nextafter(x, 0.0)
    with localcontext() as ctx:
        ctx.prec += 2
        t = Decimal(abs(x)) % (2 * pi)
        s = num = t
        lasts = 0
        i = fact = sign = 1
        t **= 2
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= t
            sign *= -1
            s += sign * num / fact
        result = float(s - Decimal(0.5 * math.ulp(s)))
    if result < -1.0:
        return -1.0
    else:
        return result

def sin_up(x: float) -> float:
    if x < 0:
        return -sin_down(-x)
    elif float(x) * x * x / 3 < math.ulp(x):
        return float(x)
    with localcontext() as ctx:
        ctx.prec += 2
        t = Decimal(abs(x)) % (2 * pi)
        s = num = t
        lasts = 0
        i = fact = sign = 1
        t **= 2
        while s != lasts:
            lasts = s
            i += 2
            fact *= i * (i - 1)
            num *= t
            sign *= -1
            s += sign * num / fact
        result = float(s + Decimal(0.5 * math.ulp(s)))
    if result > 1.0:
        return 1.0
    else:
        return result

@overload
def sinh(x: float) -> float: ...

@overload
def sinh(x: Interval) -> Interval: ...

def sinh(x):
    if isinstance(x, Interval):
        iterator = iter(x._endpoints)
        return Interval(*[
            (
                min(sinh_down(lower), sinh_down(upper)),
                max(sinh_up(lower), sinh_up(upper)),
            )
            for lower, upper in zip(iterator, iterator)
        ])
    else:
        return math.sinh(x)

def sinh_down(x: float) -> float:
    try:
        result = math.sinh(x)
    except OverflowError:
        return math.inf if x > 0.0 else -math.inf
    if (e ** Decimal(x) - e ** Decimal(-x)) / 2 < result:
        return math.nextafter(result, result - 1.0)
    return result

def sinh_up(x: float) -> float:
    try:
        result = math.sinh(x)
    except OverflowError:
        return math.inf if x > 0.0 else -math.inf
    if (e ** Decimal(x) - e ** Decimal(-x)) / 2 > result:
        return math.nextafter(result, result + 1.0)
    return result

@overload
def sqrt(x: float) -> float: ...

@overload
def sqrt(x: Interval) -> Interval: ...

def sqrt(x):
    if isinstance(x, Interval):
        iterator = iter(x[0:]._endpoints)
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
        return math.nextafter(y, 2 * y)
    elif partials[-1] > x or len(partials) == 1 or partials[-2] > 0.0:
        return y
    else:
        return math.nextafter(y, 2 * y)

def unadd(x: Union[Interval, float], y: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = Interval((float(x),) * 2)
    if not isinstance(y, Interval):
        y = Interval((float(y),) * 2)
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
        x = Interval((float(x),) * 2)
    if not isinstance(y, Interval):
        y = Interval((float(y),) * 2)
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
        x = Interval((float(x),) * 2)
    if not isinstance(y, Interval):
        y = Interval((float(y),) * 2)
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
        x = Interval((float(x),) * 2)
    if not isinstance(y, Interval):
        y = Interval((float(y),) * 2)
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
        x = Interval((float(x),) * 2)
    if not isinstance(y, Interval):
        y = Interval((float(y),) * 2)
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
        x = Interval((float(x),) * 2)
    if not isinstance(y, Interval):
        y = Interval((float(y),) * 2)
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
