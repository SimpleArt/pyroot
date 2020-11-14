import solver
from solver import sign
from math import exp, log, sin, cos, sqrt, pi
inf = float('inf')
nan = float('nan')
line = '-'*40
solver.print_result = False  # Set to true to see individual iterations
solver.return_iterations = not solver.print_result

# Big functions for infinite values

def bigpow(x, a):
    if x == 0 and a == 0: return 1.0
    if a < 0:
        res = bigpow(x, -a)
        if res == 0: return inf
        else: return 1/res
    if a == int(a):
        res = 1
        x = float(x)
        a = int(a)
        while a != 0:
            if a%2 == 1: res *= x
            x *= x
            a //= 2
        return inf
    else:
        if a > 1:
            bool = abs(x) < solver.realmax ** (1/a)
        else:
            bool = abs(x)**a < solver.realmax
        if bool:
            return sign(x)*abs(x)**a
        else:
            return sign(x)*inf

def bigexp(x):
    if x < -709.78: return 0.0
    if x > +709.78: return inf
    else: return exp(x)

def biglog(x):
    if x < 0: return nan
    if x == 0: return -inf
    if x == inf: return inf
    else: return log(x)

def bigsqrt(x):
    if x < 0: return nan
    if x == inf: return inf
    else: return sqrt(x)


methods = [
    'bisection',
    'secant',
    'dekker',
    'quadratic',
    'brent',
    'chandrupatla',
    'quadratic safe',
    'chandrupatla mixed'
]

iterations = {method : 0 for method in methods}

def test(str, f, x1, x2):
    if solver.print_result:
        print(str)
        print(f'[{x1}, {x2}]')
        print()
    for method in methods:
        x = solver.root_in(f, x1, x2, method)
        if solver.print_result: print(f'{method}: {x}')
        if solver.return_iterations: iterations[method] += x
    if solver.print_result: print(line)


# Lambert W function
# with tight bounds.
for n in range(1, 10):
    test(f'xe^x - {n}', lambda x: x*bigexp(x)-n, log(n+1), min(log(n/log(n+1)), n*exp(1)))

# Lambert W function
# with loose bounds.
for n in range(1, 10):
    test(f'xe^x - {n}', lambda x: x*bigexp(x)-n, inf, -1)

# Bring radical
# with tight bounds.
for n in range(-10, 12, 2):
    test(f'x^5 + x + {n}', lambda x: (x*x*x*x+1)*x+n, -1-abs(n), 1+abs(n))

# Bring radical
# with loose bounds.
for n in range(-10, 12, 2):
    test(f'x^5 + x + {n}', lambda x: (x*x*x*x+1)*x+n, -inf, inf)

# Quantile of Binomial Distribution
# 1st, 2nd, and 3rd quantiles.
def binomial_cdf(k, n, p):
    if p*(1-p) == 0: return 1-p
    sum = 0
    term = 1
    for i in range(1+int(k)):
        sum += term*(1-p)**(n-i)
        term *= (n-i)*p/(i+1)
    return sum

for n in range(1, 17, 4):
    for k in range(0, n, 4):
        for q in range(1, 4):
            test(f'Binomial_CDF({k}, {n}, x) - {q/4}', lambda x: binomial_cdf(k, n, x)-q/4, 0, 1)

test('e^x - 2', lambda x: bigexp(bigexp(x)-2)-1, -4, 4/3)

test('x^3 - x^2 - x - 1', lambda x: ((x-1)*x-1)*x-1, 1, 2)

test('e^x - 2', lambda x: bigexp(x)-2, -4, 4/3)

test('x^3 - 2x - 5', lambda x: (x*x-2)*x-5, 2, 3)

for n in range(1, 11):
    test(f'1 - 1/x^{n}', lambda x: 1-bigpow(x, -n), 0, inf)
    test(f'1 - 1/x^{n}', lambda x: 1-bigpow(x, -n), 0.1, inf)

for n in range(1, 5):
    test(f'2x/e^{n} + 1 - 2/e^{n}x', lambda x: 2*x*bigexp(-n)+1-2*bigexp(-n*x), 0, 1)

for n in range(1, 5):
    test(f'{1+(1-n)**2}x - (1-{n}x)^2', lambda x: (1+(1-n)**2)*x-(1-n*x)**2, 0, 1)

for n in range(1, 5):
    test(f'{1+(1-n)**4}x - (1-{n}x)^4', lambda x: (1+(1-n)**4)*x-(1-n*x)**4, 0, 1)

for j in range(1, 5):
    for k in range(5, 12):
        test(f'x^{j} - (1-x)^{k/4}', lambda x: x**j-(1-x)**(k/4), 0, 1)

for n in range(1, 5):
    test(f'(x-1)e^-{n}x + x^{n}', lambda x: (x-1)*bigexp(-n*x)+x**n, 0, 1)

for n in range(2, 6):
    test(f'(1 - {n}x) / ((1 - {n})x)', lambda x: -inf if x <= 0 else 1.0 if x >= 1e10 else (1-n*x)/((1-n)*x), 0, inf)

test('sqrt(x) - cos(x)', lambda x: bigsqrt(x)-cos(x), 0, 1)

test('sqrt(x) - cos(x)', lambda x: bigsqrt(x)-cos(x), 0, inf)

test('x^3 - 7x^2 + 14x - 6', lambda x: ((x-7)*x+14)*x-6, 0, 1)

test('x^3 - 7x^2 + 14x - 6', lambda x: ((x-7)*x+14)*x-6, 0, inf)

test('x^3 - 7x^2 + 14x - 6', lambda x: ((x-7)*x+14)*x-6, -inf, inf)

test('x^3 - 7x^2 + 14x - 6', lambda x: ((x-7)*x+14)*x-6, 3.2, 4)

test('x^4 - 2x^3 - 4x^2 + 4x + 4', lambda x: (((x-2)*x-4)*x+4)*x+4, -2, -1)

test('x^4 - 2x^3 - 4x^2 + 4x + 4', lambda x: (((x-2)*x-4)*x+4)*x+4, 0, 2)

test('e^x - x^2 + 3x - 2', lambda x: bigexp(x)+(-x+3)*x-2, 0, 1)

test('e^x - x^2 + 3x - 2', lambda x: inf if x > 300 else -inf if x < -300 else bigexp(x)+(-x+3)*x-2, -inf, inf)

test('2x cos(2x) - (x+1)^2', lambda x: 2*x*cos(2*x)-(x+1)**2, -3, -2)

test('2x cos(2x) - (x+1)^2', lambda x: 2*x*cos(2*x)-(x+1)**2, -1, 0)

test('3x - e^x', lambda x: 3*x-bigexp(x), 1, 2)

test('x + 3cos(x) - e^x', lambda x: x+3*cos(x)-bigexp(x), 0, 1)

test('x^2 - 4x + 4 - log(x)', lambda x: inf if x <= 0 else (x-4)*x+4-biglog(x), 0, 2)

test('x^2 - 4x + 4 - log(x)', lambda x: (x-4)*x+4-biglog(x), 1, 2)

test('x^2 - 4x + 4 - log(x)', lambda x: (x-4)*x+4-biglog(x), 2, 4)

test('x^2 - 4x + 4 - log(x)', lambda x: (x-4)*x+4-biglog(x), 2, inf)

test('x + 1 - 2sin(pi x)', lambda x: x+1 if x < -1e16 else x+1-2*sin(pi*x), -inf, 0)

test('x + 1 - 2sin(pi x)', lambda x: x+1-2*sin(pi*x), 0, 0.5)

test('x + 1 - 2sin(pi x)', lambda x: x+1-2*sin(pi*x), 0.5, 1)

test('x + 1 - 2sin(pi x)', lambda x: x+1 if x > 1e16 else x+1-2*sin(pi*x), 0.5, inf)

for n in range(1, 12):
    test('sum((2i-5)^2/(x-i^2)^3 for i in 1..20)', lambda x: inf if x == n*n else -inf if x == (n+1)**2 else sum((2*i-5)**2/(x-i*i)**3 for i in range(1, 21)), n*n, (n+1)**2)

for n in range(1, 12):
    test('sum((3i-5)^2/(x-i^3)^5 for i in 1..20)', lambda x: inf if x == n**3 else -inf if x == (n+1)**3 else sum((3*i-5)**2/(x-i**3)**5 for i in range(1, 21)), n**3, (n+1)**3)

test('x/e^x', lambda x: x*bigexp(x), -9, 31)

for n in range(4, 14, 2):
    test(f'x^{n} - 0.2', lambda x: x**n-0.2, 0, 5)
    test(f'x^{n} - 0.2', lambda x: bigpow(x, n)-0.2, 0, inf)

for n in range(4, 14, 2):
    test(f'x^{n} - 1', lambda x: x**n-1, 0, 5)
    test(f'x^{n} - 1', lambda x: bigpow(x, n)-1, 0, inf)

for n in range(8, 16, 2):
    test(f'x^{n} - 1', lambda x: x**n-1, -0.95, 5)
    test(f'x^{n} - 1', lambda x: bigpow(x, n)-1, -0.95, inf)

for n in range(8, 16, 2):
    test(f'x^{n} - 1', lambda x: x**n-1, -0.95, 5)
    test(f'x^{n} - 1', lambda x: bigpow(x, n)-1, -0.95, inf)

for n in range(4, 14, 2):
    test(f'x^(1/{n}) - 0.2', lambda x: bigpow(x, 1/n)-0.2, 0, 5)
    test(f'x^(1/{n}) - 0.2', lambda x: bigpow(x, 1/n)-0.2, 0, inf)

for n in range(5, 30, 5):
    test(f'e^{n}x - 2', lambda x: bigexp(n*x)-2, 0, 1)
    test(f'e^{n}x - 2', lambda x: bigexp(n*x)-2, -inf, inf)

for n in range(5, 30, 5):
    test(f'log(x) - n', lambda x: biglog(x)-n, exp(n-1), exp(n+1))
    test(f'log(x) - n', lambda x: biglog(x)-n, 0, inf)

for n in range(20, 120, 20):
    test(f'2x/e^{n} + 1 - 2/e^{n}x', lambda x: 2*x*bigexp(-n)+1-2*bigexp(-n*x), 0, 1)
    test(f'2x/e^{n} + 1 - 2/e^{n}x', lambda x: 2*x*bigexp(-n)+1-2*bigexp(-n*x), -inf, inf)

for n in range(5, 25, 5):
    test(f'{1+(1-n)**2}x - (1-{n}x)^2', lambda x: (1+(1-n)**2)*x-(1-n*x)**2, 0, 1)

for n in range(5, 25, 5):
    test(f'{1+(1-n)**4}x - (1-{n}x)^4', lambda x: (1+(1-n)**4)*x-(1-n*x)**4, 0, 1)

for n in range(5, 25, 5):
    test(f'x^2 - (1-x)^{n}', lambda x: x*x-bigpow(1-x, n), 0, 1)
    test(f'x^2 - (1-x)^{n}', lambda x: x*x-bigpow(1-x, n), -inf, 1)

for n in range(5, 25, 5):
    test(f'(x-1)e^-{n}x + x^{n}', lambda x: (x-1)*bigexp(-n*x)+x**n, 0, 1)

for n in range(5, 25, 5):
    test(f'(1 - {n}x) / ((1 - {n})x)', lambda x: -inf if x <= 0 else 1.0 if x >= 1e14 else (1-n*x)/((1-n)*x), 0, inf)

for n in range(2, 35):
    test(f'x^(1/{n}) - {n}^(1/{n})', lambda x: x**(1/n)-n**(1/n), 0, inf)

test('max(-1, x/1.5 + sin(x) - 1)', lambda x: max(-1,x/1.5+sin(x)-1), -inf, pi/2)

for n in range(10, 110, 10):
    test(f'e^min(1, max(0, {n}x/0.002)) - 1.859', lambda x: bigexp(min(1,max(0,n*x/0.002)))-1.859, -inf, inf)

for n in range(1, 50, 2):
    test(f'sign(x-1) |x-1|^{n/10}', lambda x: bigpow(x-1, n/10), 0, 10)

test(f'(e^(x+20) - 2) (1 + 10^-10 + sin(log(x^2+1)))', lambda x: (bigexp(x+20)-2)*(1+1e-10+sin(biglog(x*x+1) if -x < sqrt(solver.realmin) else 2*biglog(-x))), -inf, 0)


# Final results
print('Total iterations:')
for method in methods:
    print(f'{method}: {iterations[method]}')
