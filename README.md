![pyroot](https://raw.githubusercontent.com/SimpleArt/pyroot/media/images/logo.png)

Abstract
--------

The purpose of this Python library is to provide implementations of advanced bracketed root-finding methods for single-variable functions. These methods are meant to both guarantee convergence and also minimize the number of function calls made, even if extremely poor estimates of the root are initially provided or the function is not very well-behaved.

The following root-finding methods are implemented:

- [Bisection](https://en.wikipedia.org/wiki/Bisection_method) / Binary Search.
- [Newt-Safe](https://www.youtube.com/watch?v=FD3BPTMGJds) / [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method).
- [Secant](https://en.wikipedia.org/wiki/Secant_method) / [Regula Falsi](https://en.wikipedia.org/wiki/Regula_falsi) / False Position.
- [Muller's method](https://en.wikipedia.org/wiki/Muller%27s_method).
- [Dekker's method](https://en.wikipedia.org/wiki/Brent%27s_method#Dekker's_method).
- [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method).
- [Chandrupatla's method](https://dl.acm.org/doi/10.1016/S0965-9978%2896%2900051-8).
- Chandrupatla-Quadratic / Quadratic-safe method (experimental).
- Chandrupatla-Mixed method (default/experimental).

**Note**: The implementations may not precisely match specified methods. This is because a variety of additional features are incorporated to guarantee convergence of every method is rapid and tight, even in the face of initial estimates such as `(-inf, inf)`. For example, the traditional bisection method fails to convergence to the correct order of magnitude rapidly e.g. it goes from `(1e0, 1e300)` to `5e299` instead of `1e150`. Additionally, method's such as Brent's method may also produce "stagnant points", where the upper or lower bound of the root doesn't improve during iterations. For these reasons, modifications are made which generally improve convergence.

Example
-------

```python
from pyroot import solver, solver_table
from tabulate import tabulate  # https://pypi.org/project/tabulate/

inf = float("inf")

# A function whose root is being searched for.
def f(x):
    """x^3 - x^2 + 2x - 5 written with Horner's method."""
    return ((x - 1) * x + 2) * x - 5

# Use the default root-finder.
x = solver(f, -inf, +inf)
table = solver_table(f, -inf, +inf)

# Print the results.
print(f"x = {x}")
print(f"f(x) = {f(x)}")
print()
print(table)
```
Output:
```
x = 1.6398020042326555
f(x) = 0.0

  i              x               y
---  -------------  --------------
  0  -1.79769e+308  -inf
  1   1.79769e+308   inf
  2   0               -5
  3   1               -3
  4   4.09375         55.035
  5   2.54688         10.1277
  6   1.77344          0.979398
  7   1.65035          0.0720223
  8   1.63923         -0.00387468
  9   1.6398           1.76943e-05
 10   1.6398           1.31717e-11
 11   1.6398          -2.53042e-12
 12   1.6398           2.53042e-12
 13   1.6398           0
x = 1.6398020042326555
```

Rationale
---------

Many other root-finding methods can be found elsewhere, so why choose `pyroot`?

Although it is true that other root-finding method implementations exist, many have the problems outlined in the note above. Although some other implementations may provide faster convergence under favorable circumstances, they most likely provide *significantly* slower convergence under unfavorable circumstances. This makes them even more unviable if function calls can be extremely expensive. Additionally, several methods are provided here which are not widely found elsewhere.

Another reason to use `pyroot` is that many of these root-finding implementations are a part of very large packages, such as `scipy`. This makes them significantly larger dependencies than `pyroot`, which is a single module you can add directly into your project without issues.

Usage
-----

All of the methods involve the same base API, although some methods may further extend it e.g. the Newt-safe method includes an `fprime` parameter.

### Solver Parameters

The `solver` takes the following arguments:
```python
x = solver(f, x1, x2, *args, **kwargs, **options)
```

The required arguments for the `solver` API are the following 3 positional-only parameters:
- `f` is the function whose root you are searching for. It must accept 1 `float` argument.
- `x1, x2` are the first two estimates for the root. Requires `f(x1)` and `f(x2)` have different signs.

The following 10 keyword-only parameters (`**options`) allow some customizability:
- `y1, y2` may be `f(x1), f(x2)` if known ahead of time, or approximately known.
- `x` may be a better estimate of the root than `x1` or `x2`, but `f(x)` has unknown sign.
- `method` may be the string name of a bracketing method, or an implementation of one.
- `x_err, r_err` controls how long the `solver` runs, representating the absolute and relative error.
- `x_tol, r_tol` controls initial step-sizes, preventing slow initial convergence when the bracketing interval is significantly larger.
- `refine` controls the number of additional iterations ran after `x_err` or `r_err` are reached. By default set to run 15 iterations, but usually only takes 3 iterations.
- `mean` controls the behavior of bisection-like iterations. By default, employs tricks to improve convergence on large intervals.

**Note**: Nothing is included for termination based on the number of iterations. This is because most cases will terminate in less than 25 iterations already. If one truly wishes to terminate based on a set number of iterations, the `solver_generator` may be used to terminate based on a fixed number of iterations, or other termination conditions, such as the magnitude of `f(x)`. If one does this, it should be noted that the last produced iteration need not be the best estimate of the root.

### Method Parameters

Additional `*args` or `**kwargs` may be provided to be passed into the particular `method`, as follows:
```python
# Passes *args, **kwargs into the method.
x = solver(f, x1, x2, *args, **kwargs, **options)
```
For example:
```python
x = solver(f, x1, x2, fprime, method="newton")
```

### Solver Generator / Table

The `solver_generator` provides the same API as the `solver`, but instead of just returning the final result, every iteration is `yield`ed. This allows one to track iterations as they occur, send estimates in-between iterations, terminate early, and so on. The documentation includes examples of this combined with the `tabulate` [package](https://pypi.org/project/tabulate/) to allow prettier printing.

The `solver_table` provides the same API as the `solver_generator`, but instead of generating results, a stringified table of results is returned, using the `tabulate` package.

### Full API Documentation

For a full description of the `solver`, run the following code:
```python
from pyroot import solver

help(solver)
```
For a full description of a method, such as the Newt-safe method, run the following code:
```python
from pyroot import methods_dict

help(methods_dict["newt safe"])
```
