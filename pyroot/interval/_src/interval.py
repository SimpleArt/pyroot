from __future__ import annotations
import operator
from heapq import merge
from math import ceil, floor, inf, isinf, isnan
from typing import Any, Iterable, Iterator, Optional, SupportsFloat, SupportsIndex, TypeVar, Union, Tuple as tuple

from . import fpu_rounding as fpur

__all__ = ["Interval", "interval"]

Self = TypeVar("Self", bound="interval")


class Interval:
    _endpoints: tuple[float, ...]

    __slots__ = ("_endpoints",)

    def __init__(self: Self, /, *args: tuple[float, float]) -> None:
        for arg in args:
            if not isinstance(arg, tuple):
                raise TypeError(f"interval(...) expects tuples for arguments, got {arg!r}")
            elif len(arg) != 2:
                raise ValueError(f"interval(...) expects (lower, upper) for arguments, got {len(arg)!r} arguments")
            for x in arg:
                if not isinstance(x, SupportsFloat):
                    raise TypeError(f"could not interpret {x!r} as a real value")
                elif isnan(float(x)):
                    raise ValueError(f"could not interpret {x!r} as a real value")
        intervals = [
            (L, U)
            for lower, upper in args
            for L in [fpur.float_down(lower)]
            for U in [fpur.float_up(upper)]
            if L <= U
            if not isinf(L) or L < 0
            if not isinf(U) or U > 0
        ]
        intervals.sort()
        if len(intervals) == 0:
            self._endpoints = ()
            return
        endpoints = [intervals[0][0]]
        upper = intervals[0][1]
        for L, U in intervals:
            if L > upper:
                endpoints.append(upper)
                endpoints.append(L)
                upper = U
            elif upper < U:
                upper = U
        endpoints.append(upper)
        self._endpoints = (*endpoints,)

    def __abs__(self: Self, /) -> Self:
        return -self[:0] | self[0:]

    def __add__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval) and type(self).__add__ is type(other).__add__:
            return type(self)(*[
                (fpur.add_down(x.minimum, y.minimum), fpur.add_up(x.maximum, y.maximum))
                for x in self.sub_intervals
                for y in other.sub_intervals
            ])
        elif isinstance(other, SupportsFloat):
            return self + Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __and__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval) and type(self).__and__ is type(other).__and__:
            return type(self)(*[
                (max(x.minimum, y.minimum), min(x.maximum, y.maximum))
                for x in self.sub_intervals
                for y in other.sub_intervals
            ])
        elif isinstance(other, SupportsFloat):
            return self & Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __as_interval__(self: Self) -> Interval:
        return self

    def __contains__(self: Self, other: Any, /) -> bool:
        if not isinstance(other, SupportsFloat):
            return False
        return any(
            x.minimum <= other <= x.maximum
            for o in {*fpur.float_split(other)}
            for x in self.sub_intervals
        )

    def __eq__(self: Self, other: Any, /) -> bool:
        if isinstance(other, Interval) and type(self).__eq__ is type(other).__eq__:
            return self._endpoints == other._endpoints
        elif isinstance(other, SupportsFloat):
            return self._endpoints == float_split(other)
        else:
            return NotImplemented

    def __getitem__(self: Self, args: Union[slice, tuple[slice, ...]], /) -> Interval:
        if isinstance(args, slice):
            args = (args,)
        elif not isinstance(args, tuple):
            raise TypeError(f"interval[...] expects 0 or more slices, got {args!r}")
        for arg in args:
            if not isinstance(arg, slice):
                raise TypeError(f"interval[...] expects slices, got {arg!r}")
            elif arg.step is not None:
                raise TypeError(f"interval[...] expects [lower:upper] for arguments, got a step argument")
            elif arg.start is not None and not isinstance(arg.start, SupportsFloat):
                raise TypeError(f"could not interpret {arg.start} as a real value")
            elif arg.start is not None and isnan(float(arg.start)):
                raise ValueError(f"could not interpret {float(arg.start)!r} as a real value")
            elif arg.stop is not None and not isinstance(arg.stop, SupportsFloat):
                raise TypeError(f"could not interpret {arg.stop} as a real value")
            elif arg.stop is not None and isnan(float(arg.stop)):
                raise ValueError(f"could not interpret {float(arg.stop)!r} as a real value")
        return self & type(self)(*[
            (
                -inf if arg.start is None else fpur.float_down(arg.start),
                inf if arg.stop is None else fpur.float_up(arg.stop),
            )
            for arg in args
        ])

    def __invert__(self: Self, /) -> Self:
        iterator = iter([-inf, *self._endpoints, inf])
        return type(self)(*zip(iterator, iterator))

    def __mul__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval) and type(self).__mul__ is type(other).__mul__:
            intervals = []
            if 0 in self and len(other._endpoints) > 0 or 0 in other and len(self._endpoints) > 0:
                intervals.append((0, 0))
            for x in self[0:].sub_intervals:
                if x.maximum == 0:
                    continue
                for y in other[0:].sub_intervals:
                    if y.maximum == 0:
                        continue
                    try:
                        start = fpur.mul_down(x.minimum, y.minimum)
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = fpur.mul_up(x.maximum, y.maximum)
                    except OverflowError:
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[:0].sub_intervals:
                    if y.minimum == 0:
                        continue
                    try:
                        stop = fpur.mul_up(x.minimum, y.maximum)
                    except OverflowError:
                        continue
                    if isinf(stop):
                        continue
                    try:
                        start = fpur.mul_down(x.maximum, y.minimum)
                    except OverflowError:
                        intervals.append((-inf, stop))
                    else:
                        intervals.append((start, stop))
            for x in self[:0].sub_intervals:
                if x.minimum == 0:
                    continue
                for y in other[:0].sub_intervals:
                    if y.minimum == 0:
                        continue
                    try:
                        start = fpur.mul_down(x.maximum, y.maximum)
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = fpur.mul_up(x.minimum, y.minimum)
                    except OverflowError:
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[0:].sub_intervals:
                    if y.maximum == 0:
                        continue
                    try:
                        stop = fpur.mul_up(x.maximum, y.minimum)
                    except OverflowError:
                        continue
                    if isinf(stop):
                        continue
                    try:
                        start = fpur.mul_down(x.minimum, y.maximum)
                    except OverflowError:
                        intervals.append((-inf, stop))
                    else:
                        intervals.append((start, stop))
            return type(self)(*intervals)
        elif isinstance(other, SupportsFloat):
            other = float(other)
            return self * Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __neg__(self: Self, /) -> Self:
        iterator = reversed(self._endpoints)
        return type(self)(*[(-upper, -lower) for upper, lower in zip(iterator, iterator)])

    def __or__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval) and type(self).__or__ is type(other).__or__:
            return type(self)(
                *[(x.minimum, x.maximum) for x in self.sub_intervals],
                *[(x.minimum, x.maximum) for x in other.sub_intervals],
            )
        elif isinstance(other, SupportsFloat):
            return type(self)(*[(x.minimum, x.maximum) for x in self.sub_intervals], Interval(fpur.float_split(other)))
        else:
            return NotImplemented

    def __pos__(self: Self, /) -> Self:
        return self

    def __pow__(self: Self, other: Union[Interval, float], modulo: None = None, /) -> Interval:
        if modulo is not None:
            return NotImplemented
        elif isinstance(other, Interval) and type(self).__pow__ is type(other).__pow__:
            intervals = []
            for x in self[0:].sub_intervals:
                for y in other[0:].sub_intervals:
                    try:
                        start = fpur.pow_down(x.minimum, y.minimum)
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = fpur.pow_up(x.maximum, y.maximum)
                    except OverflowError:
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[:0].sub_intervals:
                    try:
                        start = fpur.pow_down(x.maximum, y.maximum)
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = fpur.pow_up(x.minimum, y.minimum)
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
            result = type(self)(*intervals)
            for y in other.sub_intervals:
                for i in range(ceil(y.minimum), floor(y.maximum)):
                    result |= self ** i
            return result
        elif isinstance(other, SupportsIndex):
            other = operator.index(other)
            intervals = []
            iterator = iter(self._endpoints)
            if other == 0:
                for _ in iterator:
                    intervals.append((1.0, 1.0))
                    break
            elif other % 2 == 0:
                for lower, upper in zip(iterator, iterator):
                    if upper <= 0:
                        intervals.append((fpur.pow_down(upper, other), fpur.pow_up(lower, other)))
                    elif lower >= 0:
                        intervals.append((fpur.pow_down(lower, other), fpur.pow_up(upper, other)))
                    else:
                        intervals.append((0.0, fpur.pow_up(max(lower, upper, key=abs), other)))
            else:
                intervals = [
                   (fpur.pow_down(lower, other), fpur.pow_up(upper, other))
                   for lower, upper in zip(iterator, iterator)
                ]
            return type(self)(*intervals)
        elif isinstance(other, SupportsFloat):
            other = Interval(fpur.float_split(other))
            if other.minimum.is_integer() and other.minimum == other.maximum:
                return self ** round(other.minimum)
            elif other.minimum > 0:
                iterator = iter(self[0:]._endpoints)
                intervals = [
                    (fpur.pow_down(lower, other.minimum), fpur.pow_up(upper, other.maximum))
                    for lower, upper in zip(iterator, iterator)
                ]
                return type(self)(*intervals)
            else:
                iterator = iter(self[0:]._endpoints)
                intervals = [
                    (fpur.pow_down(upper, other.maximum), fpur.pow_up(lower, other.minimum))
                    for lower, upper in zip(iterator, iterator)
                ]
                return type(self)(*intervals)
        else:
            return NotImplemented

    def __radd__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval):
            return other.__as_interval__() + self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            return self + Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __rand__(self: Self, other: float, /) -> Interval:
        if isinstance(other, Interval):
            return other.__as_interval__() & self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            return self & Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __repr__(self: Self, /) -> str:
        if len(self._endpoints) == 0:
            return "interval[()]"
        else:
            iterator = iter(self._endpoints)
            bounds = ", ".join([
                ":".join([
                    "-0.0" if lower == upper == 0 else "0.0" if lower == 0 else "" if isinf(lower) else repr(lower),
                    "0.0" if lower == upper == 0 else "-0.0" if upper == 0 else "" if isinf(upper) else repr(upper),
                ])
                for lower, upper in zip(iterator, iterator)
            ])
            return f"interval[{bounds}]"

    def __rmul__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval):
            return other.__as_interval__() * self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            return self * Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __ror__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval):
            return other.__as_interval__() | self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            return self | Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __rpow__(self: Self, other: Union[Interval, float], modulo: None = None, /) -> Interval:
        if modulo is None:
            return NotImplemented
        elif isinstance(other, Interval):
            return other.__as_interval__() ** self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            return Interval(fpur.float_split(other)) ** self
        else:
            return NotImplemented

    def __rsub__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval):
            return other.__as_interval__() - self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            return Interval(fpur.float_split(other)) - self
        else:
            return NotImplemented

    def __rtruediv__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval):
            return other.__as_interval__() / self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            other = float(other)
            return Interval(fpur.float_split(other)) / self
        else:
            return NotImplemented

    def __rxor__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval):
            return other.__as_interval__() ^ self.__as_interval__()
        elif isinstance(other, SupportsFloat):
            return self | Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __sub__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval) and type(self).__sub__ is type(other).__sub__:
            intervals = []
            for x in self.sub_intervals:
                for y in other.sub_intervals:
                    intervals.append((fpur.sub_down(x.minimum, y.maximum), fpur.sub_up(x.maximum, y.minimum)))
            return type(self)(*intervals)
        elif isinstance(other, SupportsFloat):
            return self - Interval(fpur.float_split(other))
        else:
            return NotImplemented

    def __truediv__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval) and type(self).__truediv__ is type(other).__truediv__:
            intervals = []
            if len(self._endpoints) != 0 != len(other._endpoints) and (0 in self and other != 0 or isinf(other.minimum) or isinf(other.maximum)):
                intervals.append((0, 0))
            for x in self[0:].sub_intervals:
                if x.maximum == 0:
                    continue
                for y in other[0:].sub_intervals:
                    if y.maximum == 0:
                        continue
                    try:
                        start = fpur.div_down(x.minimum, y.maximum)
                    except (OverflowError, ZeroDivisionError):
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = fpur.div_up(x.maximum, y.minimum)
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[:0].sub_intervals:
                    if y.minimum == 0:
                        continue
                    try:
                        stop = fpur.div_up(x.minimum, y.minimum)
                    except (OverflowError, ZeroDivisionError):
                        continue
                    try:
                        start = fpur.div_down(x.maximum, y.maximum)
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((-inf, stop))
                    else:
                        intervals.append((start, stop))
            for x in self[:0].sub_intervals:
                if x.minimum == 0:
                    continue
                for y in other[:0].sub_intervals:
                    if y.minimum == 0:
                        continue
                    try:
                        start = fpur.div_down(x.maximum, y.minimum)
                    except (OverflowError, ZeroDivisionError):
                        continue
                    try:
                        stop = fpur.div_up(x.minimum, y.maximum)
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[0:].sub_intervals:
                    if y.maximum == 0:
                        continue
                    try:
                        stop = fpur.div_up(x.maximum, y.maximum)
                    except (OverflowError, ZeroDivisionError):
                        continue
                    try:
                        start = fpur.div_down(x.minimum, y.minimum)
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((-inf, stop))
                    else:
                        intervals.append((start, stop))
            return type(self)(*intervals)
        elif isinstance(other, SupportsFloat):
            other = Interval(fpur.float_split(other))
            if other.minimum > 0:
                iterator = iter(self._endpoints)
                return type(self)(*[
                    (fpur.div_down(lower, L), fpur.div_up(upper, U))
                    for lower, upper in zip(iterator, iterator)
                    for L in [other.minimum if lower < 0.0 else other.maximum]
                    for U in [other.maximum if upper < 0.0 else other.minimum]
                ])
            elif other.maximum < 0:
                iterator = reversed(self._endpoints)
                return type(self)(*[
                    (fpur.div_down(upper, L), fpur.div_up(lower, U))
                    for lower, upper in zip(iterator, iterator)
                    for L in [other.minimum if upper < 0.0 else other.maximum]
                    for U in [other.maximum if lower < 0.0 else other.minimum]
                ])
            else:
                return interval[()]
        else:
            return NotImplemented

    def __xor__(self: Self, other: Union[Interval, float], /) -> Interval:
        if isinstance(other, Interval) and type(self).__xor__ is type(other).__xor__:
            iterator = merge(self._endpoints, other._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, SupportsFloat):
            return self | Interval(fpur.float_split(other))
        else:
            return NotImplemented

    @property
    def maximum(self: Self, /) -> float:
        if len(self._endpoints) == 0:
            raise ValueError(f"an empty interval has no maximum")
        else:
            return self._endpoints[-1]

    @property
    def minimum(self: Self, /) -> float:
        if len(self._endpoints) == 0:
            raise ValueError(f"an empty interval has no minimum")
        else:
            return self._endpoints[0]

    @property
    def size(self: Self, /) -> float:
        return sum(interval.maximum - interval.minimum for interval in self.sub_intervals)

    @property
    def sub_intervals(self: Self, /) -> Iterator[Self]:
        iterator = iter(self._endpoints)
        return map(type(self), zip(iterator, iterator))


interval = Interval((-inf, inf))
