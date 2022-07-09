import math
from decimal import Decimal
from typing import Iterator

from .typing import SupportsRichFloat

def split_bits(n: int) -> Iterator[int]:
    if n > 0:
        while n != 0:
            yield 1 << (n.bit_length() - 1)
            n -= 1 << (n.bit_length() - 1)
    else:
        while n != 0:
            yield -(1 << (n.bit_length() - 1))
            n += 1 << (n.bit_length() - 1)

def float_split(x: SupportsRichFloat) -> tuple[float, float]:
    y = float(x)
    if x < y:
        L = math.nextafter(y, -math.inf)
        U = y
    elif x > y:
        L = y
        U = math.nextafter(y, math.inf)
    else:
        L = U = y
    return (L, U)

def float_down(x: SupportsRichFloat) -> float:
    y = float(x)
    if x < y:
        return math.nextafter(y, -math.inf)
    else:
        return y

def float_up(x: SupportsRichFloat) -> float:
    y = float(x)
    if x > y:
        return math.nextafter(y, math.inf)
    else:
        return y

def multi_add(*args: float) -> list[float]:
    partials = []
    remaining = sorted(args)
    for _ in range(len(args)):
        if len(partials) == 0 or partials[-1] < 0:
            partials_add(partials, remaining.pop())
        else:
            partials_add(partials, remaining.pop(0))
    return partials

def partials_add(partials: list[float], x: float) -> list[float]:
    i = 0
    for y in partials:
        if abs(x) < abs(y):
            x, y = y, x
        total = x + y
        if math.isinf(total):
            partials[:] = [total]
            return partials
        error = y - (total - x)
        if error != 0.0:
            partials[i] = error
            i += 1
        x = total
    partials[i:] = [x]
    if len(partials) > 1 and x == 0.0:
        del partials[-1]
    return partials

def partials_times(partials: list[float], x: float) -> list[float]:
    temp = []
    for y in partials:
        for z in mul_precise(x, y):
            partials_add(temp, z)
    partials[:] = temp
    return partials

def add_precise(x: float, y: float) -> list[float]:
    if abs(x) < abs(y):
        x, y = y, x
    total = x + y
    error = y - (total - x)
    return [error, total]

def add_down(x: float, y: float) -> float:
    if isinstance(y, int):
        partials = multi_add(x, *[float(n) for n in split_bits(y)])
    else:
        partials = add_precise(x, y)
    if len(partials) > 1 and partials[-2] < 0.0:
        return math.nextafter(partials[-1], -math.inf)
    else:
        return partials[-1]

def add_up(x: float, y: float) -> float:
    if isinstance(y, int):
        partials = multi_add(x, *[float(n) for n in split_bits(y)])
    else:
        partials = add_precise(x, y)
    if len(partials) > 1 and partials[-2] > 0.0:
        return math.nextafter(partials[-1], math.inf)
    else:
        return partials[-1]

def sub_down(x: float, y: float) -> float:
    return add_down(x, -y)

def sub_up(x: float, y: float) -> float:
    return add_up(x, -y)

def mul_precise(x: float, y: float) -> list[float]:
    if math.isinf(x) or math.isinf(y) or math.isnan(x) or math.isnan(y):
        return [x * y]
    x_mantissa, x_exponent = math.frexp(x)
    y_mantissa, y_exponent = math.frexp(y)
    x_small = math.remainder(x_mantissa, math.ulp(x_mantissa) / math.sqrt(math.ulp(1.0)))
    x_large = x_mantissa - x_small
    y_small = math.remainder(y_mantissa, math.ulp(y_mantissa) / math.sqrt(math.ulp(1.0)))
    y_large = y_mantissa - y_small
    partials = [math.ldexp(x_large * y_large, x_exponent + y_exponent)]
    for u in (
        math.ldexp(x_large * y_small, x_exponent + y_exponent),
        math.ldexp(x_small * y_large, x_exponent + y_exponent),
        math.ldexp(x_small * y_small, x_exponent + y_exponent),
    ):
        partials_add(partials, u)
    return partials

def mul_down(x: float, y: float) -> float:
    if x * y == math.inf and not math.isinf(x) and not math.isinf(y):
        return math.nextafter(math.inf, 0.0)
    partials = mul_precise(x, y)
    if len(partials) == 1 or partials[-2] >= 0.0:
        return partials[-1]
    else:
        return math.nextafter(partials[-1], -math.inf)

def mul_up(x: float, y: float) -> float:
    if x * y == -math.inf and not math.isinf(x) and not math.isinf(y):
        return math.nextafter(-math.inf, 0.0)
    partials = mul_precise(x, y)
    if len(partials) == 1 or partials[-2] <= 0.0:
        return partials[-1]
    else:
        return math.nextafter(partials[-1], math.inf)

def div_down(x: float, y: float) -> float:
    if y == 0.0:
        return -math.inf
    elif math.isinf(x) and math.isinf(y):
        if x < 0.0 < y or y < 0.0 < x:
            return -math.inf
        else:
            return 0.0
    quotient = x / y
    if quotient == math.inf and not math.isinf(x):
        return math.nextafter(math.inf, 0.0)
    elif math.isinf(quotient):
        return quotient
    elif quotient != 0.0:
        partials = mul_precise(quotient, y)
        if y > 0:
            if partials[-1] > x:
                return math.nextafter(quotient, -math.inf)
            elif partials[-1] < x or len(partials) == 1 or partials[-2] < 0.0:
                return quotient
            else:
                return math.nextafter(quotient, -math.inf)
        else:
            if partials[-1] < x:
                return math.nextafter(quotient, -math.inf)
            elif partials[-1] > x or len(partials) == 1 or partials[-2] > 0.0:
                return quotient
            else:
                return math.nextafter(quotient, -math.inf)
    elif x == 0.0 or math.isinf(y) or not (x < 0.0 < y or y < 0.0 < x):
        return quotient
    else:
        return math.nextafter(0.0, -math.inf)

def div_up(x: float, y: float) -> float:
    if y == 0.0:
        return math.inf
    elif math.isinf(x) and math.isinf(y):
        if x < 0.0 < y or y < 0.0 < x:
            return 0.0
        else:
            return math.inf
    quotient = x / y
    if quotient == -math.inf and not math.isinf(x):
        return math.nextafter(-math.inf, 0.0)
    elif math.isinf(quotient):
        return quotient
    elif quotient != 0.0:
        partials = mul_precise(quotient, y)
        if y > 0:
            if partials[-1] < x:
                return math.nextafter(quotient, math.inf)
            elif partials[-1] > x or len(partials) == 1 or partials[-2] > 0.0:
                return quotient
            else:
                return math.nextafter(quotient, math.inf)
        else:
            if partials[-1] > x:
                return math.nextafter(quotient, math.inf)
            elif partials[-1] < x or len(partials) == 1 or partials[-2] < 0.0:
                return quotient
            else:
                return math.nextafter(quotient, math.inf)
    elif x == 0.0 or math.isinf(y) or x < 0.0 < y or y < 0.0 < x:
        return quotient
    else:
        return math.nextafter(0.0, math.inf)

def pow_down(x: float, y: float) -> float:
    try:
        result = math.pow(x, y)
    except OverflowError:
        return math.nextafter(x ** (y % 2) * math.inf, -math.inf)
    except ValueError:
        if y == round(y) and round(y) % 2 == 1:
            return -math.inf
        else:
            return math.inf
    if (
        not 0.0 != abs(x) != 1.0
        or math.isinf(x)
        or math.isnan(x)
        or math.isinf(y)
        or math.isnan(y)
    ):
        return result
    elif Decimal(x) ** Decimal(y) < result:
        return math.nextafter(result, -math.inf)
    else:
        return result

def pow_up(x: float, y: float) -> float:
    try:
        result = math.pow(x, y)
    except OverflowError:
        return math.nextafter(x ** (y % 2) * math.inf, math.inf)
    except ValueError:
        return math.inf
    if (
        not 0.0 != abs(x) != 1.0
        or math.isinf(x)
        or math.isnan(x)
        or math.isinf(y)
        or math.isnan(y)
    ):
        return result
    elif Decimal(x) ** Decimal(y) > result:
        return math.nextafter(result, math.inf)
    else:
        return result
