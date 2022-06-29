import math
from typing import Callable

from .interval import Interval

def bisection(
    f: Callable[[Interval], Interval],
    x: Interval,
    /,
    abs_error: float = 0.0,
    rel_error: float = math.ulp(32),
) -> float:
    abs_error = max(float(abs_error), math.nextafter(0.0, 1.0))
    rel_error = max(float(rel_error), math.ulp(32))
    xs = [*x[math.nextafter(-math.inf, 0.0) : math.nextafter(math.inf, 0.0)].sub_intervals]
    ys = [f(x) for x in xs]
    max_y_min = max(y.minimum for y in ys)
    xs = [x for x, y in zip(xs, ys) if y.maximum > max_y_min]
    ys = [y for y in ys if y.maximum > max_y_min]
    while not all(x.size < abs_error + 0.5 * rel_error * abs(x.minimum + x.maximum) for x in xs):
        print(xs)
        x = xs.pop(max(
            (
                (i, x, y)
                for i, (x, y) in enumerate(zip(xs, ys))
                if x.size >= abs_error + 0.5 * rel_error * abs(x.minimum + x.maximum)
            ),
            key=lambda ixy: ixy[2].maximum
        )[0])
        middle = x.minimum + 0.5 * (x.maximum - x.minimum)
        xs.append(x[:middle])
        ys.append(f(xs[-1]))
        xs.append(x[middle:])
        ys.append(f(xs[-1]))
        max_y_min = max(y.minimum for y in ys)
        xs = [x for x, y in zip(xs, ys) if y.maximum > max_y_min]
        ys = [y for y in ys if y.maximum > max_y_min]
    return xs[0].minimum + 0.5 * (xs[0].maximum - xs[0].minimum)
