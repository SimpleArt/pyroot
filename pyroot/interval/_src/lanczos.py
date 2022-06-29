"""
https://github.com/siemens/vanilc/blob/master/3rdParty/boost/math/special_functions/lanczos.hpp
https://github.com/siemens/vanilc/blob/master/3rdParty/boost/math/special_functions/gamma.hpp
"""
import decimal
import math
from collections.abc import Reversible
from decimal import Decimal, localcontext
from typing import Union

E = Decimal("2.71828182845904523536028747")

G = Decimal("10.90051099999999983936049829935654997826")

NUMERATORS = (*[
    Decimal(x)
    for x in (
        "38474670393.31776828316099004518914832218",
        "36857665043.51950660081971227404959150474",
        "15889202453.72942008945006665994637853242",
        "4059208354.298834770194507810788393801607",
        "680547661.1834733286087695557084801366446",
        "78239755.00312005289816041245285376206263",
        "6246580.776401795264013335510453568106366",
        "341986.3488721347032223777872763188768288",
        "12287.19451182455120096222044424100527629",
        "261.6140441641668190791708576058805625502",
        "2.506628274631000502415573855452633787834"
    )
],)

DENOMINATORS = (
    0,
    362880,
    1026576,
    1172700,
    723680,
    269325,
    63273,
    9450,
    870,
    45,
    1,
)

EXP_NUMERATORS = (*[
    Decimal(x)
    for x in (
        "709811.662581657956893540610814842699825",
        "679979.847415722640161734319823103390728",
        "293136.785721159725251629480984140341656",
        "74887.5403291467179935942448101441897121",
        "12555.29058241386295096255111537516768137",
        "1443.42992444170669746078056942194198252",
        "115.2419459613734722083208906727972935065",
        "6.30923920573262762719523981992008976989",
        "0.2266840463022436475495508977579735223818",
        "0.004826466289237661857584712046231435101741",
        "0.00004624429436045378766270459638520555557321"
    )
],)

def evaluate_rational(
    numerator: Reversible[Union[Decimal, int]],
    denominator: Reversible[Union[Decimal, int]],
    x: Decimal,
) -> Decimal:
    s1 = 0
    s2 = 0
    if x > 1:
        x = 1 / x
    else:
        numerator = reversed(numerator)
        denominator = reversed(denominator)
    for coef in numerator:
        s1 *= x
        s1 += coef
    for coef in denominator:
        s2 *= x
        s2 += coef
    return s1 / s2

def digamma_P_lanczos(x: Decimal) -> Decimal:
    derivative = [i * x for i, x in enumerate(EXP_NUMERATORS[1:], 1)]
    derivative.append(0)
    return evaluate_rational(derivative, EXP_NUMERATORS, x)

def digamma_Q_lanczos(x: Decimal) -> Decimal:
    derivative = [i * x for i, x in enumerate(DENOMINATORS[1:], 1)]
    derivative.append(0)
    return evaluate_rational(derivative, DENOMINATORS, x)

def gamma_lanczos(x: Decimal) -> Decimal:
    return evaluate_rational(NUMERATORS, DENOMINATORS, x)

def lgamma_exp_lanczos(x: Decimal) -> Decimal:
    return evaluate_rational(EXP_NUMERATORS, DENOMINATORS, x)

def digamma_precise(x: float) -> Decimal:
    from .imath import _BIG_PI, cos_precise, sin_precise
    with localcontext() as ctx:
        ctx.prec += 10
        if x > 0.0 and math.isinf(x):
            return Decimal("Infinity")
        elif x > 0.0:
            d = Decimal(x)
        elif math.isinf(x):
            assert False, "digamma_precise should not accept negative infinity"
        elif x.is_integer():
            assert False, "digamma_precise should not accept non-positive integers"
        else:
            d = 1 - Decimal(x)
        result = None
        if d >= 1e4:
            try:
                s = (
                    d.ln()
                    - 1 / (2 * d)
                    - 1 / (12 * d ** 2)
                    + 1 / (120 * d ** 4)
                    - 1 / (252 * d ** 6)
                    + 1 / (240 * d ** 8)
                    - 1 / (132 * d ** 10)
                    + 691 / (32760 * d ** 12)
                )
                if s == s - 1 / (12 * d ** 14):
                    result = s
            except decimal.Overflow:
                result = math.inf
        if result is None:
            d1 = d % 1
            dgh = d1 + G + Decimal("0.5")
            result = (
                (d1 + Decimal("0.5")) / dgh
                + dgh.ln()
                + digamma_P_lanczos(d1 + 1)
                - digamma_Q_lanczos(d1 + 1)
                - 1
            )
            dx = 0
            while d - d1 > 1.5:
                d1 += 1
                dx += 1 / d1
            while d - d1 < 0.5:
                dx -= 1 / d1
                d1 -= 1
            result += dx
        if x <= 0.0:
            theta = x % 1.0
            if theta > 0.5:
                theta = -((-x) % 1.0)
            c = cos_precise(_BIG_PI * Decimal(theta))
            s = sin_precise(_BIG_PI * Decimal(theta))
            result -= _BIG_PI * c / s
        return result

def gamma_precise(x: float) -> Decimal:
    from .imath import _BIG_PI, sin_precise
    with localcontext() as ctx:
        ctx.prec += 10
        if x >= 172.0:
            return Decimal("Infinity")
        elif x > 0.0 and x.is_integer():
            result = Decimal(1)
            for i in range(2, round(x)):
                result *= i
            return result
        elif x > 0.0:
            d = Decimal(x)
        elif math.isinf(x):
            assert False, "gamma_precise should not accept negative infinity"
        elif x.is_integer():
            assert False, "gamma_precise should not accept non-positive integers"
        else:
            d = 1 - Decimal(x)
        result = None
        if d >= 1e4:
            d += 1
            s = (
                1
                + 1 / (12 * d)
                + 1 / (288 * d ** 2)
                - 139 / (51840 * d ** 3)
            )
            if s == s - 571 / (2488320 * d ** 4):
                try:
                    result = (2 * d * _BIG_PI).ln() + (d.ln() - 1) * d + s.ln()
                except decimal.Overflow:
                    result = math.inf
        if result is None:
            d1 = d % 1
            result = ((d + G - Decimal("0.5")) / E) ** (d - Decimal("0.5")) * gamma_lanczos(d)
            dx = 1
            while d - d1 > 1.5:
                d1 += 1
                dx *= d1
            while d - d1 < 0.5:
                dx /= d1
                d1 -= 1
            result *= dx
        if x <= 0.0:
            theta = x % 2.0
            if theta > 1.0:
                theta = -((-x) % 2.0)
            result = _BIG_PI / (
                sin_precise(Decimal(theta) * _BIG_PI)
                * result
            )
        return result

def lgamma_precise(x: float) -> Decimal:
    from .imath import _BIG_E, _BIG_PI, sin_precise
    with localcontext() as ctx:
        ctx.prec += 10
        if x > 0.0 and math.isinf(x):
            return math.inf
        elif x > 0.0 and x.is_integer():
            result = Decimal(1)
            for i in range(2, round(x)):
                result *= i
            return result.ln()
        elif x > 0.0:
            d = Decimal(x)
        elif math.isinf(x):
            assert False, "lgamma_precise should not accept negative infinity"
        elif x.is_integer():
            assert False, "lgamma_precise should not accept non-positive integers"
        else:
            d = 1 - Decimal(x)
        if d >= 1e4:
            d += 1
            s = (
                1
                + 1 / (12 * d)
                + 1 / (288 * d ** 2)
                - 139 / (51840 * d ** 3)
            )
            if s == s - 571 / (2488320 * d ** 4):
                try:
                    result = (2 * d * _BIG_PI).ln() + (d.ln() - 1) * d + s.ln()
                except decimal.Overflow:
                    result = math.inf
        else:
            d1 = d % 1
            result = ((d1 + Decimal("0.5") + G).ln() - 1) * (d1 + Decimal("0.5")) + lgamma_exp_lanczos(d1 + 1).ln()
            dx = 0
            while d - d1 > 1.5:
                d1 += 1
                dx += d1.ln()
            while d - d1 < 0.5:
                dx -= d1.ln()
                d1 -= 1
            result += dx
        if x <= 0.0:
            theta = abs(x) % 2.0
            if theta > 1.0:
                theta = (-abs(x)) % 2.0
            result = (
                _BIG_PI.ln()
                - sin_precise(Decimal(theta) * _BIG_PI).ln()
                - result
            )
        return result

