"""
GitHub:
    https://github.com/SimpleArt/pyroot

The purpose of this Python library is to provide implementations of
advanced bracketed root-finding methods. For deterministic univariate
functions, `pyroot` provides guaranteed fast convergence, minimizing
the number of function calls needed, even if extremely poor initial
estimates are provided or if the function is not very well-behaved.
Additionally, implementations of bracketing-like methods are provided
for stochastic univariate and deterministic multivariate functions are
provided.

Example
-------
    >>> from math import inf
    >>> from pyroot import root_in, root_iter
    >>> 
    >>> def f(x):
    ...     return ((x - 1) * x + 2) * x - 5
    ...
    >>> x = solver(f, -inf, +inf)
    >>> x, f(x)
    (1.6398020042326555, 0.0)
    >>> 
    >>> for i, x in enumerate(root_iter(f, -inf, +inf)):
    ...     print(f"{i:>3d}  {x:>25.16E}  {f(x):>25.16f}")
    ... 
      0   -1.7976931348623157E+308                       -inf
      1    1.7976931348623157E+308                        inf
      2     0.0000000000000000E+00        -5.0000000000000000
      3     7.8125000000000000E-03        -4.9844355583190918
      4     2.5371467728470156E+00         9.9690821682950936
      5     8.6537227493019742E-01        -3.3700740034394583
      6     1.3056110594089907E+00        -1.8678270842189817
      7     1.8529844070606760E+00         1.6347344594925595
      8     1.6166487295283405E+00        -0.1550583332176085
      9     1.6407097528123704E+00         0.0061643449464510
     10     1.6397998719520530E+00        -0.0000144722990747
     11     1.6398020042561847E+00         0.0000000001596980
     12     1.6398020042326498E+00        -0.0000000000000400
     13     1.6398020042326615E+00         0.0000000000000400
     14     1.6398020042326555E+00         0.0000000000000000
"""
from . import root_in, root_iter

__all__ = ["root_in", "root_iter"]
__version__ = "0.3.4"
