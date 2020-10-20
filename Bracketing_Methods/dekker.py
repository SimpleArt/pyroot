def dekker(x1, f1, x2, f2, x3, f3, x4, f4, t):
    """Dekker's method using secant extrapolation"""
    
    if f2 == f3 or t >= 0.5: return f2 / (f2 - f1)
    else: return f2*(x3-x2)/((f2-f3)*(x1-x2))
