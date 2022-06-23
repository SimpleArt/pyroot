from collections.abc import Callable, Iterator
from typing import Optional

import pyroot._utils as _utils
from ._src.interval import Interval
from ._src import root_all as _root_all

__all__ = ["bisect", "newton"]

def bisect(
    f: Callable[[Interval], Interval],
    interval: Interval,
    /,
    *,
    x: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[Interval]:
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * interval.minimum - 0.1 * interval.maximum))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return Interval(*[
        (solution.minimum, solution.maximum)
        for solution in _root_all.bisect(f, interval, x, abs_err, rel_err, abs_tol, rel_tol)
    ])

def newton(
    f: Callable[[Interval], Interval],
    interval: Interval,
    fprime: Callable[[Interval], Interval],
    /,
    *,
    x: Optional[float] = None,
    abs_err: float = 0.0,
    rel_err: float = 32 * _utils.FLOAT_EPSILON,
    abs_tol: Optional[float] = None,
    rel_tol: Optional[float] = None,
) -> Iterator[Interval]:
    abs_err = max(float(abs_err), 32 * _utils.FLOAT_MIN)
    if abs_tol is None:
        abs_tol = min(4.0625, abs(0.1 * interval.minimum - 0.1 * interval.maximum))
    rel_err = max(float(rel_err), 32 * _utils.FLOAT_EPSILON)
    abs_tol = max(float(abs_tol), abs_err)
    if rel_tol is None:
        rel_tol = rel_err / (1024 * _utils.FLOAT_EPSILON)
    rel_tol = max(float(rel_tol), rel_err)
    rel_err = min(rel_err, 0.5)
    rel_tol = min(rel_tol, 0.5)
    return Interval(*[
        (solution.minimum, solution.maximum)
        for solution in _root_all.newton(f, interval, fprime, x, abs_err, rel_err, abs_tol, rel_tol)
    ])
