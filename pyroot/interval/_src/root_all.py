from collections.abc import Callable, Iterator
from math import exp, log, sqrt
from typing import Optional

import pyroot._utils as _utils
from .interval import Interval

__all__ = ["bisect", "newton"]

def bisect(
    f: Callable[[Interval], Interval],
    interval: Interval,
    x: Optional[float],
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[Interval]:
    if len(interval._endpoints) == 0:
        return
    if x is None:
        intervals = [*interval.sub_intervals]
    else:
        x = float(x)
        intervals = [*interval[:x].sub_intervals, *interval[x:].sub_intervals]
    intervals = [
        interval
        for interval in reversed(intervals)
        if 0 in f(interval)
    ]
    previous = None
    while len(intervals) > 0:
        interval = intervals.pop()
        x1 = interval.minimum
        x2 = interval.maximum
        x = 0.5 * (_utils.sign(x1) + _utils.sign(x2))
        while (
            _utils.is_between(x1, x, x2)
            and not _utils.is_between(x1 / 8, x2, x1 * 8)
            and not (_utils.sign(x1) == _utils.sign(x2) and _utils.is_between(sqrt(abs(x1)), abs(x2), x1 * x1))
            and x2 - x1 > abs_err + rel_err * abs(x)
        ):
            if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
                x += 0.25 * (abs_err + rel_err * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
            else:
                x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
            left = interval[:x]
            right = interval[x:]
            if 0 in f(left):
                if 0 in f(right):
                    intervals.append(right)
                interval = left
            elif 0 in f(right):
                interval = right
            else:
                interval = None
                break
            x1 = interval.minimum
            x2 = interval.maximum
            x = 0.5 * (_utils.sign(x1) + _utils.sign(x2))
        if interval is None:
            continue
        x_sign = 1 if x > 0 else -1
        x_abs = 1 if abs(x1) > 1 else -1
        while (
            not _utils.is_between(x1 / 8, x2, x1 * 8)
            and not _utils.is_between(sqrt(abs(x1)), abs(x2), x1 * x1)
            and x2 - x1 > abs_err + rel_err * abs(x)
        ):
            x = x_sign * exp(x_abs * sqrt(log(abs(x1)) * log(abs(x2))))
            if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
                x += 0.25 * (abs_err + rel_err * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
            else:
                x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
            left = interval[:x]
            right = interval[x:]
            if 0 in f(left):
                if 0 in f(right):
                    intervals.append(right)
                interval = left
            elif 0 in f(right):
                interval = right
            else:
                interval = None
                break
            x1 = interval.minimum
            x2 = interval.maximum
        if interval is None:
            continue
        while (
            not _utils.is_between(x1 / 8, x2, x1 * 8)
            and x2 - x1 > abs_err + rel_err * abs(x)
        ):
            x = x_sign * sqrt(abs(x1)) * sqrt(abs(x2))
            if abs(x1 - x2) < 16 * (abs_tol + rel_tol * abs(x)):
                x += 0.25 * (abs_err + rel_err * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
            else:
                x += 0.25 * (abs_tol + rel_tol * abs(x)) * _utils.sign((x1 - x) + (x2 - x))
            left = interval[:x]
            right = interval[x:]
            if 0 in f(left):
                if 0 in f(right):
                    intervals.append(right)
                interval = left
            elif 0 in f(right):
                interval = right
            else:
                interval = None
                break
            x1 = interval.minimum
            x2 = interval.maximum
        if interval is None:
            continue
        while x2 - x1 > abs_err + rel_err * abs(x):
            x = x1 + 0.5 * (x2 - x1)
            left = interval[:x]
            right = interval[x:]
            if 0 in f(left):
                if 0 in f(right):
                    intervals.append(right)
                interval = left
            elif 0 in f(right):
                interval = right
            else:
                interval = None
                break
            x1 = interval.minimum
            x2 = interval.maximum
        if interval is None:
            pass
        elif previous is None:
            previous = interval
        elif interval.minimum - previous.maximum < abs_err + rel_err * abs(_utils.mean(interval.minimum, previous.maximum)):
            previous |= Interval((previous.maximum, interval.maximum))
        else:
            yield previous
            previous = interval
    if previous is not None:
        yield previous

def newton(
    f: Callable[[Interval], Interval],
    interval: Interval,
    fprime: Callable[[Interval], Interval],
    x: Optional[float],
    abs_err: float,
    rel_err: float,
    abs_tol: float,
    rel_tol: float,
    /,
) -> Iterator[Interval]:
    if len(interval._endpoints) == 0:
        return
    if x is None:
        intervals = [*interval.sub_intervals]
    else:
        x = float(x)
        intervals = [*interval[:x].sub_intervals, *interval[x:].sub_intervals]
    intervals = [
        interval
        for interval in reversed(intervals)
        if 0 in f(interval)
    ]
    previous = None
    while len(intervals) > 0:
        interval = intervals.pop()
        size = interval.size
        x = _utils.mean(interval.minimum, interval.maximum)
        if size > abs_err + rel_err * abs(x):
            pass
        elif previous is None:
            previous = interval
            continue
        elif interval.minimum - previous.maximum < abs_err + rel_err * abs(_utils.mean(interval.minimum, previous.maximum)):
            previous |= Interval((previous.maximum, interval.maximum))
            continue
        else:
            yield previous
            previous = interval
            continue
        y = f(Interval((x, x)))
        previous_interval = interval
        interval &= x - _utils.mean(y.minimum, y.maximum) / fprime(interval)
        if len(interval._endpoints) == 0:
            pass
        elif interval != previous_interval:
            if interval.size < abs_tol + rel_tol * abs(_utils.mean(interval.minimum, interval.maximum)):
                abs_current = abs_err
                rel_current = rel_err
            else:
                abs_current = abs_tol
                rel_current = rel_tol
            x = previous_interval.maximum
            x -= 0.25 * (abs_current + rel_current * abs(x))
            if x < interval.maximum:
                if 0 in f(interval[x:]):
                    intervals.append(interval[x:])
                interval = interval[:x]
                if len(interval._endpoints) == 0:
                    continue
            x = previous_interval.minimum
            x += 0.25 * (abs_current + rel_current * abs(x))
            if x > interval.minimum:
                iterator = reversed(interval[x:]._endpoints)
                intervals.extend(
                    interval
                    for upper, lower in zip(iterator, iterator)
                    for interval in [Interval((lower, upper))]
                    if 0 in f(interval)
                )
                interval = interval[:x]
                if 0 in f(interval):
                    intervals.append(interval)
            else:
                iterator = reversed(interval._endpoints)
                intervals.extend(
                    interval
                    for upper, lower in zip(iterator, iterator)
                    for interval in [Interval((lower, upper))]
                    if 0 in f(interval)
                )
        else:
            right = interval[x:]
            if 0 in f(right):
                intervals.append(right)
            left = interval[:x]
            if 0 in f(left):
                intervals.append(left)
    if previous is not None:
        yield previous