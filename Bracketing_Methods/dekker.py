def dekker(x, x1, x2, x3, fx, f1, f2, f3, error_x, flag):
    """Dekker's method combining secant with bisection"""
    
    t = f2 / (f2 - f1)
    if t <= 0 or t >= 1: t = 0.5
    return t
