def newton_quadratic(x1, f1, x2, f2, x3, f3, x4, f4, t):
    a = (f1-f2)/(x1-x2)
    b = (f2-f3)/(x2-x3)
    b = (a-b)/(x1-x3)
    if b == 0: return f2/(f2-f1)
    if b*f2 > 0: r = x2
    else: r = x1
    for i in range(2):
        r -= (f2+a*(r-x2)+b*(r-x1)*(r-x2))/(a+b*((r-x1)+(r-x2)))
    return (r-x2)/(x1-x2)
