def brent(x, x1, x2, x3, fx, f1, f2, f3, error_x, flag):
    """Modified Brent's method
    to fit with general bracketing method
    as well as improve convergence
    """
    
    """interpolation"""
    
    if f1 != f3 and f2 != f3:
        # inverse quadratic interpolation
        al = (x3 - x2) / (x1 - x2)
        a = f2 / (f1 - f2)
        b = f3 / (f1 - f3)
        c = f2 / (f3 - f2)
        d = f1 / (f3 - f1)
        t = a*b + c*d*al
    else:
        # secant
        t = f2 / (f2 - f1)
    
    """bisection"""
    if (t <= 0 or t >= 1) or (flag and t*abs(x1-x2) >= 0.5*abs(x2-x3)) or (not flag and t*abs(x1-x2) >= 0.5*abs(x-x3)) or (flag and   abs(x2-x3) < error_x*(1+abs(x2))) or (not flag and abs(x-x3) < error_x*(1+abs(x2))): t = 0.5
    
    return t
