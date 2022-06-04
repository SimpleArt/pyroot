"""
Contains performance tests for pyroot.solver().

Includes a variety of test cases found in the literature,
some of which have been extended to test the performance
of different methods with extreme bracketing intervals,
especially when an initial endpoint is a singularity or at
infinity, such as the interval from (0 to inf) for log(x).
"""
import sys
from typing import Any, Callable
from collections import defaultdict
from math import cos, e, exp, fsum, inf, isinf, log, nan, pi, sin, sqrt
from itertools import chain
from tabulate import tabulate
from pyroot._pyroot import methods_dict, real_max, real_min, sign, solver, solver_generator

# Version dependent imports.
if sys.version_info[:2] >= (3, 9):
    DefaultDict = defaultdict
    Dict = dict
    List = list
    Tuple = tuple
    from collections.abc import Iterator
else:
    from typing import DefaultDict, Dict, Iterator, List, Tuple

def bigpow(x: float, y: float) -> float:
    """
    Implements pow(x, y) with handling of overflow as
    using inf and some sign manipulation.

    If x == 0 and y == 0, then return 1.0.
    If y < 0, then return 1 / bigpow(x, -y), or inf in the case of division by 0.
    Otherwise y > 0, and the result is pow(x, y) as expected, using inf appropriately.
        If x < 0 and y == int(y), then the sign is sign(x) ** y.
        Otherwise the sign of x is used.
    """
    # Attempt to use the built-in pow.
    try:
        result = pow(x, y)
    # If division by 0 occurs, return inf.
    except ZeroDivisionError:
        return inf
    # If overflow occurs, return inf with an appropriate sign.
    # If sign(x) ** y would be complex, just use sign(x) instead.
    except OverflowError:
        return sign(x) ** y * inf if y == int(y) else sign(x) * inf
    # Check for complex values.
    else:
        return sign(x) * pow(abs(x), y) if isinstance(result, complex) else result

def bigexp(x: float) -> float:
    """Implements big exponentiation by using inf if it overflows."""
    try:
        return exp(x)
    except OverflowError:
        return inf

def biglog(x: float) -> float:
    """Implements big logarithm with log(0) == -inf and log(x<0) == nan."""
    return -inf if x == 0 else nan if x < 0 else log(x)

def bigsqrt(x: float) -> float:
    """Implements big square root with sqrt(x<0) == nan."""
    return nan if x < 0 else sqrt(x)

def bigsin(x: float) -> float:
    """Uses sin(inf) = 0 since sin(inf) usually doesn't actually matter."""
    return 0 if isinf(x) else sin(x)

def bigcos(x: float) -> float:
    """Uses cos(inf) = 0 since cos(inf) usually doesn't actually matter."""
    return 0 if isinf(x) else cos(x)

# Collect unique names for each method.
methods = ("brent", "chandrupatla", "quadratic safe", "chandrupatla mixed")
methods_dict = {name: methods_dict[name] for name in methods}
# Store all of the results.
results_dict: DefaultDict[Tuple[str, float, float, float], Dict[str, Tuple[int, float, float]]] = defaultdict(dict)

def results(f: Callable[[float], float], x1: float, x2: float, *args: Any, **kwargs: Any) -> Tuple[int, float, float]:
    """Returns the number of iterations and the final solution together as (i, x, y)."""
    x: float
    # Use a helper to collect the returned value from the generator.
    def get_x() -> Iterator[float]:
        nonlocal x
        x = yield from solver_generator(f, x1, x2, *args, **kwargs)
    # Exhaust the generator to collect x and count the number of iterations.
    # Don't count the first 2 bracketing points.
    i = sum(1 for _ in get_x()) - 2
    return i, x, f(x)

def store_results(name: str, f: Callable[[float], float], x1: float, x2: float, y1: float = nan, y2: float = nan, x: float = nan, *args: Any, **kwargs: Any) -> int:
    """Stores the results for each method."""
    for method in methods_dict:
        results_dict[name, x1, x2, y1, y2, x][method] = results(f, x1, x2, *args, y1=y1, y2=y2, x=x, method=method, **kwargs)

def rowify_results(sort_by_type: bool = True) -> Iterator[Tuple[str, str, str, str, str, float, ...]]:
    """Groups the results into rows for tabulating."""
    # Group iterations, x's, and y's together.
    if sort_by_type:
        for index in range(3):
            for (f, x1, x2, y1, y2, x), results in results_dict.items():
                yield (f, f"{x1:.2f}", f"{x2:.2f}", f"{y1:.2f}", f"{y2:.2f}", f"{x:.2f}", ("i", "x", "y")[index], *[row[index] for row in results.values()])
    # Group functions together.
    else:
        for (f, x1, x2, y1, y2, x), results in results_dict.items():
            for index in range(3):
                yield (f, f"{x1:.2f}", f"{x2:.2f}", f"{y1:.2f}", f"{y2:.2f}", f"{x:.2f}", ("i", "x", "y")[index], *[row[index] for row in results.values()])
    # Find the maximum amount of iterations.
    yield ("max iterations", "nan", "nan", "nan", "nan", "nan", "i", *[
        max(row[method][0] for _, row in results_dict.items())
        for method in methods_dict
    ])
    # Sum up the total iterations.
    yield ("total iterations", "nan", "nan", "nan", "nan", "nan", "i", *[
        sum(row[method][0] for _, row in results_dict.items())
        for method in methods_dict
    ])
    # Find the maximum amount of absolute error.
    yield ("max absolute error", "nan", "nan", "nan", "nan", "nan", "y", *[
        max(abs(row[method][2]) for _, row in results_dict.items())
        for method in methods_dict
    ])
    # L^1 error: sum of the absolute error with full accuracy.
    yield ("mean absolute error", "nan", "nan", "nan", "nan", "nan", "y", *[
        fsum(abs(row[method][2]) for _, row in results_dict.items()) / len(results_dict)
        for method in methods_dict
    ])
    # L^2 error: sum of the squared error with full accuracy.
    yield ("root mean squared error", "nan", "nan", "nan", "nan", "nan", "y", *[
        sqrt(fsum(row[method][2] ** 2 for _, row in results_dict.items()) / len(results_dict))
        for method in methods_dict
    ])

def tabulate_results(sort_by_type: bool = True, showindex: bool = True) -> str:
    """Tabulates the results for neater printing."""
    return tabulate(
        rowify_results(sort_by_type),
        headers=("function", "x1", "x2", "y1", "y2", "x", "type", *methods_dict),
        showindex=showindex,
    )

def binomial_cdf(k: int, n: int, p: float) -> float:
    """Computes the CDF of the binomial distribution."""
    # Boundary cases.
    if p in (0, 1):
        return 1.0 - p
    # Reduce computation with symmetry to improve accuracy.
    elif 2 * k > n:
        return 1 - binomial_cdf(n-k-1, n, 1-p)
    # Collect all of the terms for the binomial CDF sum.
    terms = []
    binomial_coef = 1
    for i in range(k+1):
        terms.append(binomial_coef * p**i * (1-p)**(n-i))
        binomial_coef = binomial_coef * (n-i) // (i+1)
    return fsum(terms)

test_cases = list(chain(
    # Compute the Lambert W function.
    (
        (f"lambert_w({n})", lambda x, n=n: x * bigexp(x) - n, min(log(n / log(n+1)), n * e), log(n+1), y1, y2)
        for y1, y2 in ((nan, nan), (-1, 1))
        for n in range(1, 10)
    ),
    (
        (f"lambert_w({n})", lambda x, n=n: x * bigexp(x) - n, -1, inf, y1, y2)
        for y1, y2 in ((nan, nan), (-1, 1))
        for n in range(1, 10)
    ),
    (
        (f"lambert_w({n})", lambda x, n=n: x * bigexp(x) - n, -1, inf, y1, y2, log(n+1))
        for y1, y2 in ((nan, nan), (-1, 1))
        for n in range(1, 10)
    ),
    # Compute the Bring radical.
    (
        (f"br({n})", lambda x, n=n: (x*x*x*x + 1) * x + n, -abs(n)-1, abs(n)+1, y1, y2)
        for y1, y2 in ((nan, nan), (-1, 1))
        for n in range(1, 10)
    ),
    (
        (f"br({n})", lambda x, n=n: (x*x*x*x + 1) * x + n, -inf, inf, y1, y2)
        for y1, y2 in ((nan, nan), (-1, 1))
        for n in range(1, 10)
    ),
    # Compute quantiles for the binomial cdf.
    (
        (f"binomial_CDF({k}, {n}, x) - {q/4}", lambda x, n=n, k=k, q=q: binomial_cdf(k, n, x) - q/4, 0, 1)
        for n in range(1, 17, 4)
        for k in range(0, n, 4)
        for q in range(1, 4)
    ),
    (
        ("e^(e^x - 2) - 1", lambda x: bigexp(bigexp(x) - 2) - 1, -4, 4/3),
        ("x^3 - x^2 - x - 1", lambda x: ((x - 1) * x - 1) * x - 1, 1, 2),
        ("e^x - 2", lambda x: bigexp(x) - 2, -4, 4/3),
        ("x^3 - 2x - 5", lambda x: (x*x - 2) * x - 5, 2, 3),
    ),
    # Singularities and near-singularities.
    (
        (f"1 - 1 / x^{n}", lambda x: 1 - bigpow(x, -n), x1, inf)
        for x1 in (0, 0.1)
        for n in range(1, 11)
    ),
    # Very convex functions.
    (
        (f"2x / e^{n} + 1 - 2 / e^{n}x", lambda x: 2*x*bigexp(-n) + 1 - 2*bigexp(-n*x), 0, 1)
        for n in range(1, 5)
    ),
    (
        (f"{1+(1-n)**i}x - (1-{n}x)^{i}", lambda x: (1+(1-n)**i)*x - (1-n*x)**i, 0, 1)
        for i in (2, 4)
        for n in range(1, 5)
    ),
    (
        (f"x^{j} - (1-x)^{k/4}", lambda x, j=j, k=k: x**j - (1-x)**(k/4), 0, 1)
        for j in range(1, 5)
        for k in range(5, 12)
    ),
    (
        (f"(x-1)e^{-n}x + x^{n}", lambda x, n=n: (x-1) * bigexp(-n*x) + x**n, 0, 1)
        for n in range(1, 5)
    ),
    (
        (f"(1 - {n}x) / ((1 - {n})x)", lambda x, n=n: -inf if x <= 0 else 1.0 if x >= 1e14 else (1 - n*x) / ((1-n) * x), 0, inf)
        for n in range(2, 6)
    ),
    (
        ("sqrt(x) - cos(x)", lambda x: bigsqrt(x) - bigcos(x), x1, x2)
        for x1, x2 in ((0, 1), (0, inf))
    ),
    (
        ("x^3 - 7x^2 + 14x - 6", lambda x: ((x - 7) * x + 14) * x - 6, x1, x2)
        for x1, x2 in ((0, 1), (0, inf), (-inf, inf), (3.2, 4))
    ),
    (
        ("x^4 - 2x^3 - 4x^2 + 4x + 4", lambda x: (((x - 2) * x - 4) * x + 4) * x + 4, x1, x2)
        for x1, x2 in ((-2, -1), (0, 2))
    ),
    (
        ("e^x - x^2 + 3x - 2", lambda x: inf if x > 300 else -inf if x < -300 else bigexp(x) + (-x + 3) * x - 2, x1, x2)
        for x1, x2 in ((0, 1), (-inf, inf))
    ),
    (
        ("2x cos(2x) - (x+1)^2", lambda x: 2*x * bigcos(2*x) - (x+1) ** 2, x1, x2)
        for x1, x2 in ((-3, -2), (-1, 0))
    ),
    (
        ("3x - e^x", lambda x: 3*x - bigexp(x), 1, 2),
        ("x + 3cos(x) - e^x", lambda x: x + 3*bigcos(x) - bigexp(x), 0, 1),
    ),
    (
        ("x^2 - 4x + 4 - log(x)", lambda x: bigpow(x-2, 2) - biglog(x), x1, x2)
        for x1, x2 in ((0, 2), (1, 2), (2, 4), (2, inf))
    ),
    (
        ("x + 1 - 2sin(pi x)", lambda x: x + 1 - 2 * bigsin(pi*x), x1, x2)
        for x1, x2 in ((-inf, 0), (0, 0.5), (0.5, 1), (0.5, inf))
    ),
    (
        (
            f"sum(({j}i-5)^2/(x-i^{j})^{k} for i in 1..20)",
            lambda x, n=n, j=j, k=k: inf if x == n**j else -inf if x == (n+1)**j else fsum((j*i - 5) ** 2 / (x - i**j) ** k for i in range(1, 21)),
            n ** j,
            (n+1) ** j,
        )
        for j, k in ((2, 3), (3, 5))
        for n in range(1, 12)
    ),
    (("x / e^x", lambda x: x * bigexp(-x), -9, 31),),
    # High order inflection points.
    (
        (f"x^{n} - 0.2", lambda x, n=n: bigpow(x, n) - 0.2, x1, x2)
        for x1, x2 in ((0, 5), (0, inf))
        for n in range(4, 14, 2)
    ),
    (
        (f"x^{n} - 1", lambda x, n=n: bigpow(x, n) - 1, x1, x2)
        for x1, x2 in ((0, 5), (0, inf))
        for n in range(4, 14, 2)
    ),
    (
        (f"x^{n} - 1", lambda x, n=n: bigpow(x, n) - 1, x1, x2)
        for x1, x2 in ((-0.95, 5), (-0.95, inf))
        for n in range(8, 16, 2)
    ),
    (
        (f"e^{n}x - 2", lambda x, n=n: bigexp(n * x) - 2, x1, x2)
        for x1, x2 in ((0, 1), (-inf, inf))
        for n in range(5, 30, 5)
    ),
    (
        (f"log(x) - {n}", lambda x, n=n: biglog(x) - n, bigexp(n-1), bigexp(n+1))
        for n in range(5, 30, 5)
    ),
    (
        (f"log(x) - {n}", lambda x, n=n: biglog(x) - n, 0, inf)
        for n in range(5, 30, 5)
    ),
    (
        (f"2x/e^{n} + 1 - 2/e^{n}x", lambda x, n=n: 2*x * bigexp(-n) + 1 - 2 * bigexp(-n*x), x1, x2)
        for x1, x2 in ((0, 1), (-inf, inf))
        for n in range(20, 120, 20)
    ),
    (
        (f"{1+(1-n)**i}x - (1-{n}x)^{i}", lambda x, n=n, i=i: (1 + (1-n)**i) * x - (1 - n*x) ** i, 0, 1)
        for i in (2, 4)
        for n in range(5, 25, 5)
    ),
    (
        (f"(1-x)^2 - x^{n}", lambda x, n=n: ((-bigpow(x, n-2) + 1) * x - 2) * x + 1, x1, x2)
        for x1, x2 in ((0, 1), (0, inf))
        for n in range(5, 25, 5)
    ),
    (
        (f"(1-x)^2 - x^{n}", lambda x, n=n: ((-bigpow(x, n-2) + 1) * x - 2) * x + 1, x1, x2, 1 - 1/n)
        for x1, x2 in ((0, 1), (0, inf))
        for n in range(5, 25, 5)
    ),
    (
        (f"(x-1)e^{-n}x + x^{n}", lambda x, n=n: (x-1) * bigexp(-n*x) + x ** n, 0, 1)
        for n in range(5, 25, 5)
    ),
    (
        (f"(1 - {n}x) / ((1 - {n})x)", lambda x, n=n: -inf if x <= 0 else 1.0 if x >= 1e14 else (1 - n*x) / ((1-n) * x), 0, inf)
        for n in range(5, 25, 5)
    ),
    (
        (f"x^(1/{n}) - {n}^(1/{n})", lambda x, n=n: x ** (1/n) - n ** (1/n), 0, inf)
        for n in range(5, 35)
    ),
    # High order roots.
    (
        (f"sign(x-1) |x-1|^{n/10}", lambda x, n=n: bigpow(x-1, n/10), 0, 10)
        for n in range(1, 50, 2)
    ),
    # Piecewise continuous functions.
    (("max(-1, x/1.5 + sin(x) - 1)", lambda x: max(-1, x/1.5 + bigsin(x) - 1), -inf, pi/2),),
    (
        (f"e^min(1, max(0, {n}x/0.002)) - 1.859", lambda x, n=n: bigexp(min(1, max(0, n*x/0.002))) - 1.859, -inf, inf)
        for n in range(10, 110, 10)
    ),
    ((
        "(e^(x+20)-2) (1+1e-10+sin(log(1+x^2)))",
        lambda x: (bigexp(x+20) - 2) * (1 + 1e-10 + bigsin(biglog(1 + x*x))) if x > sqrt(real_min) else 2 * biglog(-x),
        -inf,
        0,
    ),),
))

def main() -> None:
    """Run the performance tests."""
    # Run all of the test cases.
    for args in test_cases:
        store_results(*args)
    # Collect tabulated results.
    print(tabulate_results())
    print(
        "\n"
        "Types:\n"
        "i: The number of iterations used.\n"
        "x: The final result.\n"
        "y: The error given by y = f(x)."
    )

if __name__ == "__main__":
    main()
