def dekker(x1, f1, x2, f2, x3, f3, x4, f4, t):
    """Dekker's method using secant extrapolation"""
    
    if f2 == f3: return 0.5
    else: return f2*(x3-x2)/((f2-f3)*(x1-x2))
