import math
from decimal import Decimal

def multi_add(*args: float) -> list[float]:
    partials = []
    for arg in args:
        partials_add(partials, arg)
    return partials

def partials_add(partials: list[float], x: float) -> list[float]:
    i = 0
    for y in partials:
        if abs(x) < abs(y):
            x, y = y, x
        total = x + y
        error = y - (total - x)
        if error != 0.0:
            partials[i] = error
            i += 1
        x = total
    partials[i:] = [x]
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
    partials = add_precise(x, y)
    if partials[0] < 0.0:
        return math.nextafter(partials[1], -math.inf)
    else:
        return partials[1]

def add_up(x: float, y: float) -> float:
    partials = add_precise(x, y)
    if partials[0] > 0.0:
        return math.nextafter(partials[1], math.inf)
    else:
        return partials[1]

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
    x = float(x)
    try:
        result = math.pow(x, y)
    except OverflowError:
        return math.nextafter(x ** (y % 2) * math.inf, -math.inf)
    if (
        round(y) == y
        or x <= 0.0
        or x == 1.0
        or math.isinf(x)
        or math.isnan(x)
        or math.isinf(y)
        or math.isnan(y)
    ):
        return result
    elif Decimal(x) ** Decimal(y) < result:
        return math.nextafter(result, 0.0)
    else:
        return result

def pow_up(x: float, y: float) -> float:
    x = float(x)
    try:
        result = math.pow(x, y)
    except OverflowError:
        return math.nextafter(x ** (y % 2) * math.inf, math.inf)
    if (
        round(y) == y
        or x <= 0.0
        or x == 1.0
        or math.isinf(x)
        or math.isnan(x)
        or math.isinf(y)
        or math.isnan(y)
    ):
        return result
    elif Decimal(x) ** Decimal(y) > result:
        return math.nextafter(result, 2 * result)
    else:
        return result
