def sign(x):
    return (x>0)-(x<0)

def chandrupatla_mixed(x1, f1, x2, f2, x3, f3, x4, f4, t):
    """Chandrupatla's method which chooses the best of
    - inverse quadratic interpolation, and
    - quadratic interpolation.
    """
    
    x = (x2-x1) / (x3-x1)
    y = (f2-f1) / (f3-f1)
    
    # use quadratic interpolation
    if x**2 < y and (1-x)**2 < 1-y and y*(1-y) < x*(1-x):
        a = (f1-f2)/(x1-x2)
        b = (f2-f3)/(x2-x3)
        b = (a-b)/(x1-x3)
        if b == 0: return f2/(f2-f1)
        if sign(b) == sign(f2): r = x2
        else: r = x1
        
        # newton's method 3 times
        for i in range(3):
            r -= (f2+a*(r-x2)+b*(r-x1)*(r-x2))/(a+b*((r-x1)+(r-x2)))
        
        return (r-x2)/(x1-x2)
    
    # use inverse quadratic interpolation
    elif y**2 < x and (1-y)**2 < 1-x:
        al = (x3-x2) / (x1-x2)
        a = f2 / (f1-f2)
        b = f3 / (f1-f3)
        c = f2 / (f3-f2)
        d = f1 / (f3-f1)
        return a*b + c*d*al
    
    # use bisection
    else:
        return 0.5
