pyroot
======

Abstract
--------

Python library implementing advanced bracketed root-finding methods for single-variable functions. These methods are meant to both guarantee convergence and also minimize the number of function calls made, even if extremely poor estimates of the root are initially provided or the function is not very well-behaved.

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
from pyroot import solver, solver_generator
from tabulate import tabulate  # https://pypi.org/project/tabulate/

inf = float("inf")

# A function whose root is being searched for.
def f(x):
    """x^3 - x^2 + 2x - 5 written with Horner's method."""
    return ((x - 1) * x + 2) * x - 5

# Use the default root-finder.
x = solver(f, -inf, +inf)
iterations = [(i, x, f(x)) for i, x in enumerate(solver_generator(f, -inf, +inf))]

# Print the results.
print(f"x = {x})
print(f"f(x) = {f(x)})
print()
print("Iterations:")
print(tabulate(iterations, ("i", "x", "y")))
```
Output:
```
x = 1.6398020042326555
f(x) = 0.0

Iterations:
  i              x               y
---  -------------  --------------
  0  -1.79769e+308  -inf
  1   1.79769e+308   inf
  2   0               -5
  3   8.98847e+307   inf
  4   0.0078125       -4.98444
  5   1.68361          0.30496
  6   0.491764        -4.13938
  7   1.62344         -0.11002
  8   1.64042          0.00417759
  9   1.6398           4.41056e-07
 10   1.6398          -2.34301e-12
 11   1.6398           2.13163e-14
 12   1.6398          -1.95399e-14
 13   1.6398           0
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

The required arguments for the `solver` API are the following 3 positional-only arguments:
- `f` is the function whose root you are searching for. It must accept 1 `float` argument.
- `x1, x2` are the first two estimates for the root. Requires `f(x1)` and `f(x2)` have different signs.

The following 10 keyword-only arguments allow some customizability:
- `y1, y2` may be `f(x1), f(x2)` if known ahead of time, or approximately known.
- `x` may be a better estimate of the root than `x1` or `x2`, but `f(x)` has unknown sign.
- `method` may be the string name of a bracketing method, or an implementation of one.
- `x_err, r_err` controls how long the `solver` runs, representating the absolute and relative error.
- `x_tol, r_tol` controls initial step-sizes, preventing slow initial convergence when the bracketing interval is significantly larger.
- `refine` controls the number of additional iterations ran after `x_err` or `r_err` are reached. By default set to run 15 iterations, but usually only takes 3 iterations.
- `mean` controls the behavior of bisection-like iterations. By default, employs tricks to improve convergence on large intervals.

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

### Solver Generator

The `solver_generator` provides the same API as the `solver`, but instead of just returning the final result, every iteration is `yield`ed. This allows one to, for example, track iterations as they occur, and even send estimates in-between iterations. The documentation includes examples of this combined with the `tabulate` [package](https://pypi.org/project/tabulate/) to allow prettier printing.

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


