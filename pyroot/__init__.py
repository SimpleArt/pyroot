"""
GitHub:
    https://github.com/SimpleArt/pyroot

The purpose of this Python library is to provide implementations of
advanced bracketed root-finding methods for single-variable functions.
These methods are meant to both guarantee convergence and also minimize
the number of function calls made, even if extremely poor estimates of
the root are initially provided or the function is not very well-behaved.

Example
-------
    >>> from pyroot import solver, solver_table
    >>> def f(x):
    ...     return ((x - 1) * x + 2) * x - 5
    ...
    >>> inf = float("inf")
    >>> x = solver(f, -inf, +inf)
    >>> x, f(x)
    (1.6398020042326555, 0.0)
    >>> print(solver_table(f, -inf, +inf))
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
"""
from pyroot.pyroot import *
