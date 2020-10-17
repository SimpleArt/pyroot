def brent(x1, f1, x2, f2, x3, f3, x4, f4, t):
    """Modified Brent's method
    to fit with general bracketing method
    as well as improve convergence
    """
    
    flag = t == 0.5
    
    # Interpolation:
    if f1 != f3 and f2 != f3:
        # inverse quadratic interpolation
        al = (x3 - x2) / (x1 - x2)
        a = f2 / (f1 - f2)
        b = f3 / (f1 - f3)
        c = f2 / (f3 - f2)
        d = f1 / (f3 - f1)
        t = a*b + c*d*al
    else:
        # secant interpolation
        t = f2 / (f2 - f1)
    
    # bisection
    if t <= 0 or t >= 1 or (flag and t*abs(x1-x2) >= 0.5*abs(x2-x3)) or (not flag and t*abs(x1-x2) >= 0.5*abs(x4-x3)):
        t = 0.5
    
    return t
