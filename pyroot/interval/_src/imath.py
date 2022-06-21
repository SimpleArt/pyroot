import decimal
import math
from decimal import Decimal, localcontext
from typing import Optional, Union

from .fpu_rounding import *
from .interval import Interval, interval

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
    mantissa, exponent = math.frexp(x)
    if not exponent > 0:
        mantissa = math.ldexp(mantissa, exponent)
        exponent = 0
    with localcontext() as ctx:
        ctx.prec += int(exponent * math.log10(2)) + 10
        d = Decimal(mantissa)
        c = cos_precise(d)
        s = sin_precise(d)
        for _ in range(exponent):
            c, s = (c + s) * (c - s), 2 * c * s
        return c, s

def acos(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    iterator = reversed(x[-1.0:1.0]._endpoints)
    return Interval(*[
        (acos_down(upper), acos_up(lower))
        for upper, lower in zip(iterator, iterator)
    ])

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
    iterator = iter(x[1.0:]._endpoints)
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
    iterator = iter(x[-1.0:1.0]._endpoints)
    return Interval(*[
        (asin_down(lower), asin_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

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
    iterator = iter(x._endpoints)
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
    iterator = iter(x._endpoints)
    return Interval(*[
        (atan_down(lower), atan_up(upper))
        for lower, upper in zip(iterator, iterator)
    ])

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

def atanh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    iterator = iter(x._endpoints)
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
    if len(x._endpoints) == 0:
        return x
    elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
        return interval[-1.0:1.0]
    iterator = iter(x._endpoints)
    return Interval(*[
        (
            -1
            if (
                lower < upper
                and (
                    lower <= (upper // (2 * math.pi)) * (2 * math.pi) + (math.pi / 2) <= upper
                    or lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (math.pi / 2) <= upper
                )
            )
            else min(cos_down(lower), cos_down(upper)),
            1
            if (
                lower < upper
                and (
                    lower <= (upper // (2 * math.pi)) * (2 * math.pi) <= upper
                    or lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (2 * math.pi) <= upper
                )
            )
            else max(cos_up(lower), cos_up(upper)),
        )
        for lower, upper in zip(iterator, iterator)
    ])

def cos_down(x: float) -> float:
    return decimal_down(cos_sin_precise(x)[0])

def cos_up(x: float) -> float:
    return decimal_up(cos_sin_precise(x)[0])

def cosh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
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
    iterator = iter(x._endpoints)
    return Interval(*[(exp_down(lower), exp_up(upper)) for lower, upper in zip(iterator, iterator)])

def exp_down(x: float) -> float:
    try:
        return decimal_down(Decimal(x).exp())
    except decimal.Overflow:
        return math.inf

def exp_up(x: float) -> float:
    try:
        return decimal_up(Decimal(x).exp())
    except decimal.Overflow:
        return math.inf

def log(x: Union[Interval, float], base: Optional[Union[Interval, float]] = None) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    if base is None:
        iterator = iter(x[0:]._endpoints)
        return Interval(*[
            (log_down(lower), log_up(upper))
            for lower, upper in zip(iterator, iterator)
        ])
    elif isinstance(base, Interval):
        base = base[0:]
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
    if len(x._endpoints) == 0:
        return x
    elif math.isinf(x._endpoints[0]) or math.isinf(x._endpoints[-1]):
        return interval[-1.0:1.0]
    iterator = iter(x._endpoints)
    return Interval(*[
        (
            -1
            if (
                lower < upper
                and (
                    lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (math.pi / 2) <= upper
                    or lower <= (upper // (2 * math.pi)) * (2 * math.pi) + (3 * math.pi / 2) <= upper
                )
            )
            else min(sin_down(lower), sin_down(upper)),
            1
            if (
                lower < upper
                and (
                    lower <= (upper // (2 * math.pi)) * (2 * math.pi) + (math.pi / 2) <= upper
                    or lower <= (upper // (2 * math.pi)) * (2 * math.pi) - (3 * math.pi / 2) <= upper
                )
            )
            else max(sin_up(lower), sin_up(upper)),
        )
        for lower, upper in zip(iterator, iterator)
    ])

def sin_down(x: float) -> float:
    return decimal_down(cos_sin_precise(x)[1])

def sin_up(x: float) -> float:
    return decimal_up(cos_sin_precise(x)[1])

def sinh(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    iterator = iter(x._endpoints)
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
        return math.nextafter(y, math.inf)
    elif partials[-1] > x or len(partials) == 1 or partials[-2] > 0.0:
        return y
    else:
        return math.nextafter(y, math.inf)

def tan(x: Union[Interval, float]) -> Interval:
    if not isinstance(x, Interval):
        x = float(x)
        x = Interval((x, x))
    iterator = iter(x._endpoints)
    return Interval(*[
        interval
        for lower, upper in zip(iterator, iterator)
        for L in [tan_down(lower)]
        for U in [tan_up(upper)]
        for interval in (
            [(L, U)]
            if (
                lower == upper
                or not (
                    lower <= (upper // math.pi) * math.pi - (math.pi / 2) <= upper
                    or lower <= (upper // math.pi) * math.pi + (math.pi / 2) <= upper
                )
            )
            else [(-math.inf, U), (L, math.inf)]
            if upper - lower < math.pi or 1 == len({
                x
                for x in {
                    (lower // math.pi) * math.pi + (math.pi / 2),
                    (upper // math.pi) * math.pi - (math.pi / 2),
                    (upper // math.pi) * math.pi + (math.pi / 2),
                }
                if lower <= x <= upper
            })
            else [(-math.inf, math.inf)]
        )
    ])

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
    iterator = iter(x._endpoints)
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
