import operator
from heapq import merge
from math import ceil, floor, inf, isinf, isnan
from typing import Any, Iterable, Iterator, Optional, SupportsFloat, SupportsIndex, TypeVar, Union, Tuple as tuple

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
            (lower, upper)
            for lower, upper in args
            if lower <= upper
            if not isinf(lower) or lower < 0
            if not isinf(upper) or upper > 0
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

    def __add__(self: Self, other: Union[Self, float], /) -> Self:
        if isinstance(other, Interval):
            return type(self)(*[
                (x.minimum + y.minimum, x.maximum + y.maximum)
                for x in self.sub_intervals
                for y in other.sub_intervals
            ])
        elif isinstance(other, SupportsFloat):
            other = float(other)
            return self + interval[other:other]
        else:
            return NotImplemented

    def __and__(self: Self, other: Union[Self, float], /) -> Self:
        if isinstance(other, Interval):
            return type(self)(*[
                (max(x.minimum, y.minimum), min(x.maximum, y.maximum))
                for x in self.sub_intervals
                for y in other.sub_intervals
            ])
        elif isinstance(other, SupportsFloat):
            other = float(other)
            if other in self:
                return type(self)((other, other))
            else:
                return type(self)()
        else:
            return NotImplemented

    def __contains__(self: Self, other: Any, /) -> bool:
        if not isinstance(other, SupportsFloat):
            return False
        other = float(other)
        return any(x.minimum <= other <= x.maximum for x in self.sub_intervals)

    def __eq__(self: Self, other: Any, /) -> bool:
        if isinstance(other, Interval):
            return self._endpoints == other._endpoints
        elif isinstance(other, SupportsFloat):
            return self._endpoints == (float(other),) * 2
        else:
            return NotImplemented

    def __getitem__(self: Self, args: Union[slice, tuple[slice, ...]], /) -> Self:
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
                -inf if arg.start is None else float(arg.start),
                inf if arg.stop is None else float(arg.stop),
            )
            for arg in args
        ])

    def __invert__(self: Self, /) -> Self:
        iterator = iter([-inf, *self._endpoints, inf])
        return type(self)(*zip(iterator, iterator))

    def __mul__(self: Self, other: Union[Self, float], /) -> Self:
        if isinstance(other, Interval):
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
                        start = x.minimum * y.minimum
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = x.maximum * y.maximum
                    except OverflowError:
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[:0].sub_intervals:
                    if y.minimum == 0:
                        continue
                    try:
                        stop = x.minimum * y.maximum
                    except OverflowError:
                        continue
                    if isinf(stop):
                        continue
                    try:
                        start = x.maximum * y.minimum
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
                        start = x.maximum * y.maximum
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = x.minimum * y.minimum
                    except OverflowError:
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[0:].sub_intervals:
                    if y.maximum == 0:
                        continue
                    try:
                        stop = x.maximum * y.minimum
                    except OverflowError:
                        continue
                    if isinf(stop):
                        continue
                    try:
                        start = x.minimum * y.maximum
                    except OverflowError:
                        intervals.append((-inf, stop))
                    else:
                        intervals.append((start, stop))
            return type(self)(*intervals)
        elif isinstance(other, SupportsFloat):
            other = float(other)
            return self * interval[other:other]
        else:
            return NotImplemented

    def __neg__(self: Self, /) -> Self:
        iterator = reversed(self._endpoints)
        return type(self)(*[(-upper, -lower) for upper, lower in zip(iterator, iterator)])

    def __or__(self: Self, other: Union[Self, float], /) -> Self:
        if isinstance(other, Interval):
            return type(self)(
                *[(x.minimum, x.maximum) for x in self.sub_intervals],
                *[(x.minimum, x.maximum) for x in other.sub_intervals],
            )
        elif isinstance(other, SupportsFloat):
            other = float(other)
            if other in self:
                return self
            else:
                return type(self)(*[(x.minimum, x.maximum) for x in self.sub_intervals], (other, other))
        else:
            return NotImplemented

    def __pos__(self: Self, /) -> Self:
        return self

    def __pow__(self: Self, other: Union[Self, float], modulo: None = None, /) -> Self:
        if modulo is not None:
            return NotImplemented
        elif isinstance(other, Interval):
            intervals = []
            for x in self[0:].sub_intervals:
                for y in other[0:].sub_intervals:
                    try:
                        start = x.minimum ** y.minimum
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = x.maximum ** y.maximum
                    except OverflowError:
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[:0].sub_intervals:
                    try:
                        start = x.maximum ** y.maximum
                    except OverflowError:
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = x.minimum ** y.minimum
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
            temp1 = 1 / self if other < 0 else self
            temp2 = interval[1:1]
            other = abs(other)
            if other % 2 == 0 and other != 0:
                neg = temp1[:0]
                pos = temp1[0:]
                temp1 = (neg * neg) | (pos * pos)
                other //= 2
            while other > 0:
                if other % 2 == 1:
                    temp2 *= temp1
                temp1 *= temp1
                other //= 2
            return temp2
        elif isinstance(other, SupportsFloat):
            other = float(other)
            if other == round(other):
                return self ** round(other)
            else:
                return self ** interval[other:other]
        else:
            return NotImplemented

    def __radd__(self: Self, other: float, /) -> Self:
        if isinstance(other, SupportsFloat):
            return self + other
        else:
            return NotImplemented

    def __rand__(self: Self, other: float, /) -> Self:
        if isinstance(other, SupportsFloat):
            return self & other
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

    def __rmul__(self: Self, other: float, /) -> Self:
        if isinstance(other, SupportsFloat):
            return self * other
        else:
            return NotImplemented

    def __ror__(self: Self, other: float, /) -> Self:
        if isinstance(other, SupportsFloat):
            return self | other
        else:
            return NotImplemented

    def __rpow__(self: Self, other: float, modulo: None = None, /) -> Self:
        if modulo is not None and isinstance(other, SupportsFloat):
            other = float(other)
            return interval[other:other] ** self
        else:
            return NotImplemented

    def __rsub__(self: Self, other: float, /) -> Self:
        if isinstance(other, SupportsFloat):
            return self * -1 + other
        else:
            return NotImplemented

    def __rtruediv__(self: Self, other: float, /) -> Self:
        if isinstance(other, SupportsFloat):
            other = float(other)
            return interval[other:other] / self
        else:
            return NotImplemented

    def __rxor__(self: Self, other: float, /) -> Self:
        if isinstance(other, SupportsFloat):
            return self | other
        else:
            return NotImplemented

    def __sub__(self: Self, other: Union[Self, float], /) -> Self:
        if isinstance(other, Interval):
            intervals = []
            for x in self.sub_intervals:
                for y in other.sub_intervals:
                    intervals.append((x.minimum - y.maximum, x.maximum - y.minimum))
            return type(self)(*intervals)
        elif isinstance(other, SupportsFloat):
            other = float(other)
            return self - interval[other:other]
        else:
            return NotImplemented

    def __truediv__(self: Self, other: Union[Self, float], /) -> Self:
        if isinstance(other, Interval):
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
                        start = x.minimum / y.maximum
                    except (OverflowError, ZeroDivisionError):
                        continue
                    if isinf(start):
                        continue
                    try:
                        stop = x.maximum / y.minimum
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[:0].sub_intervals:
                    if y.minimum == 0:
                        continue
                    try:
                        stop = x.minimum / y.minimum
                    except (OverflowError, ZeroDivisionError):
                        continue
                    try:
                        start = x.maximum / y.maximum
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
                        start = x.maximum / y.minimum
                    except (OverflowError, ZeroDivisionError):
                        continue
                    try:
                        stop = x.minimum / y.maximum
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((start, inf))
                    else:
                        intervals.append((start, stop))
                for y in other[0:].sub_intervals:
                    if y.maximum == 0:
                        continue
                    try:
                        stop = x.maximum / y.maximum
                    except (OverflowError, ZeroDivisionError):
                        continue
                    try:
                        start = x.minimum / y.minimum
                    except (OverflowError, ZeroDivisionError):
                        intervals.append((-inf, stop))
                    else:
                        intervals.append((start, stop))
            return type(self)(*intervals)
        elif isinstance(other, SupportsFloat):
            return self * (1 / float(other))
        else:
            return NotImplemented

    def __xor__(self: Self, other: Union[Self, float], /) -> Self:
        if isinstance(other, Interval):
            iterator = merge(self._endpoints, other._endpoints)
            return type(self)(*zip(iterator, iterator))
        elif isinstance(other, SupportsFloat):
            return self | other
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
        iterator = iter(self._endpoints)
        return sum(upper - lower for lower, upper in zip(iterator, iterator))

    @property
    def sub_intervals(self: Self, /) -> Iterator[Self]:
        iterator = iter(self._endpoints)
        return map(type(self), zip(iterator, iterator))


interval = Interval((-inf, inf))