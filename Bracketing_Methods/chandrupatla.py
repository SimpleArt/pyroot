def chandrupatla(x1, f1, x2, f2, x3, f3, x4, f4, t):
    """Chandrupatla's method"""
    
    x = (x2-x1) / (x3-x1)
    y = (f2-f1) / (f3-f1)
    
    if y**2 < x and (1-y)**2 < 1-x: # f is nearly linear
        al = (x3-x2) / (x1-x2)
        a = f2 / (f1-f2)
        b = f3 / (f1-f3)
        c = f2 / (f3-f2)
        d = f1 / (f3-f1)
        return a*b + c*d*al
    else:
        return 0.5
