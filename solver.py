from Bracketing_Methods.bisection        import bisection
from Bracketing_Methods.secant           import secant
from Bracketing_Methods.dekker           import dekker
from Bracketing_Methods.brent            import brent
from Bracketing_Methods.newton_quadratic import newton_quadratic
from Bracketing_Methods.chandrupatla     import chandrupatla

def is_nan(x):
    return x != x
def is_infinite(x):
    return abs(x) == float('inf')

print("Run solver.help() for an explanation on how to use the solver program.")

print_flag = True

def help():
    print("""These methods allow one to find a root or relative extrema of a provided function, provided that an initial bracket is given.

To enable/disable printed results, set solver.print_flag to True/False.

======================================================================================================

root_in:

Run a bracketing method to find a root of f between x1 and x2.
  Requires f(x1) and f(x2) to have different signs.
  Otherwise only one iteration of the secant method is run.

The provided methods are:
- Bisection
- Regula Falsi / False Position
- Dekker
- Brent
- Muller
- Chandrupatla

Usage: solver.root_in(f, x1, x2, {method, iterations, error, f1, f2})

Example:
---------------------------------------------
>>> def f(x):
...   return x*x - x - 1
... 
>>> solver.root_in(f, 1, 2)
f(1.5000000000000000) = -0.25
f(1.6333333333333377) = 0.03444444444445427
f(1.6178006329113883) = -0.0005217450628995923
f(1.6180341697805805) = 4.0479695195827503e-07
f(1.6180339887496349) = -5.810907310888069e-13
f(1.6180339887498987) = 8.659739592076221e-15
f(1.6180339887498910) = -8.881784197001252e-15
1.6180339887498951
---------------------------------------------

@params:
  f: searched function
  x1, x2: bracketing points
  method: bracketing method to be used
  iterations: maximum number of iterations before pure bisection is used
  error: desired error* of |x1-x2| to terminate
    error*: (error + 5e-15*|x1+x2|), a combination of absolute and relative errors.
  f1, f2: initial points

@defaults:
  method: Chandrupatla's method
  iterations: 1000 iterations
  error: 1e-16 (near machine precision)
  f1, f2: computed as f(x1), f(x2)

======================================================================================================

optimize:

Local extrema may be found by bracketing the derivative.
  Requires g to be increasing on one side and decreasing on the other.
  This is measured by f(x) = g(x+dx) - g(x-dx), where dx is based on the error.
  solver.root_in(f, x1, x2, method, iterations, error) is then used.

Usage: solver.optimize(g, x1, x2, {method, iterations, error})

Example:
---------------------------------------------------
>>> def g(x):
...   return x*math.exp(x)
... 
>>> solver.optimize(g, -10.5, 6, 'chandrupatla')
g(-10.5000000000000000) = 23.52388866664127

g(6.000000000000000000) = 20.69693845669907

g(-2.25000000000000000) = 1.125
f(-2.25000000000000000) = -8.1250000061317e-08

g(-1.37285157401949700) = 0.2357014407670286
f(-1.37285157401949700) = -3.5950183718824746e-08

g(-0.71121935280072270) = -0.1114208899961231
f(-0.71121935280072270) = -9.069713424736392e-09

g(2.644390323599638700) = 6.944587195864736
f(2.644390323599638700) = 2.5067845932369437e-07

g(-0.57826341578574510) = -0.1385309046668266
f(-0.57826341578574510) = -4.439796974509136e-09

g(1.033063453906946800) = 2.083066350272663
f(1.033063453906946800) = 1.0265327476943753e-07

g(-0.50402936486417770) = -0.1461935886485614
f(-0.50402936486417770) = -1.9529936556850203e-09

g(0.264517044521384600) = 0.400561389028831
f(0.264517044521384600) = 4.4801037490938e-08

g(-0.46928826439943040) = -0.14780413303797862
f(-0.46928826439943040) = -8.101439519236919e-10

g(-0.44514524929657790) = -0.14814787196152124
f(-0.44514524929657790) = -2.2778223751629412e-11

g(-0.44445279786778380) = -0.1481481481088972
f(-0.44445279786778380) = -2.7150504067208203e-13

g(-0.44442774182830963) = -0.1481481479912224
f(-0.44442774182830963) = 5.428435478904703e-13

g(-0.44444444159748553) = -0.14814814814814814
f(-0.44444444159748553) = 1.1102230246251565e-16

g(-0.44444444751326584) = -0.14814814814814814
f(-0.44444444751326584) = -1.1102230246251565e-16

g(-0.44444444455537570) = -0.14814814814814814
-0.4444444445553757
---------------------------------------------------

@params:
  g: searched function
  x1, x2: bracketing points
  method: bracketing method to be used
  iterations: maximum number of iterations before pure bisection is used
  error: desired error* of |x1-x2| to terminate
    error*: (error + 5e-15*|x1+x2|), a combination of absolute and relative errors.

@defaults:
  method: bracket default
  iterations: bracket default
  error: 1e-8 since usually impossible to search further due to roundoff errors and catastrophic cancellation.

@return:
  x: the result returned by solver.root_in().
""")

def sign(x):
    """Returns the sign of x"""
    if x > 0: return 1
    if x < 0: return -1
    return 0

def bracketing_method(x1, f1, x2, f2, x3, f3, x4, f4, t):
    """Interface for the bracketing method.
    
    @params:
        x1, f1: bracketing point for x2
        x2, f2: last estimate of the root
        x3, f3: last removed point from the interval
        x4, f4: second last removed point from the interval, initially None
        t: last computed t
    
    @returns:
        t: the combination of x1 and x2 for the next iteration
           i.e. the next x is x1 + t*(x2-x1)
    """
    
    pass

def root_in(f, x1, x2,
                method = None,
                iterations = None,
                error = None,
                f1 = None,
                f2 = None):
    """Run a bracketing method to find a root of f between x1 and x2.
    Requires f(x1) and f(x2) to have different signs.
    Otherwise only one iteration of the secant method is run.
    
    @params:
        f: searched function
        x1, x2: bracketing points
        method: bracketing method to be used
        iterations: maximum number of iterations before pure bisection is used
        error: desired error* of |x1-x2| to terminate
            error*: (error + 5e-15*|x1+x2|), a combination of absolute and relative errors.
        f1, f2: optional initial points, computed if not given
    
    @defaults:
        method: Chandrupatla's method with inverse quadratic extrapolation.
        iterations: 1000 iterations.
        error: 1e-16 for double precision.
    
    @return:
        (f1*x2-f2*x1)/(f1-f2), the secant estimate with the final bracket
    """
    
    """Set bracketing method to be used"""
    if method == 'bisection':
        bracketing_method = bisection
    elif method in ['regula falsi', 'false position', 'secant']:
        bracketing_method = secant
    elif method == 'dekker':
        bracketing_method = dekker
    elif method == 'brent':
        bracketing_method = brent
    elif method in ['muller', 'quadratic']:
        bracketing_method = newton_quadratic
    else:
        bracketing_method = chandrupatla
    
    """Set default iterations and error"""
    if iterations is None: iterations = 1000
    if error is None: error = 1e-16
    
    """Compute initial points"""
    if f1 is None: f1 = f(x1)
    if f2 is None: f2 = f(x2)
    if is_nan(f1) or is_nan(f2): return 0.5*(x1+x2)
    if f1 == 0: return x1
    if f2 == 0: return x2
    if sign(f1) == sign(f2): return (f2*x1-f1*x2)/(f2-f1)
    if is_infinite(x2):
        x1, f1, x2, f2 = x2, f2, x1, f1
    
    """Initialize variables"""
    n = 0
    bisection_fails = 0
    x3, f3 = None, None
    t = 0.5 # Safe start is bisection
    
    """Loop until convergence"""
    while (is_infinite(x1) or is_infinite(x2) or abs(x1-x2) > error+5e-15*abs(x1+x2)) and f2 != 0 and not is_nan(f2):
        
        # Maximum number of iterations before pure bisection is used.
        if n == iterations:
            bracketing_method = bisection
        
        """Compute next point"""
        if is_infinite(x2):
            x = 0
        elif is_infinite(x1):
            t = 0.5
            x = x2 + (1.0+abs(x2))*sign(x1)
        else:
            if is_infinite(f1) or is_infinite(f2) or is_nan(t): t = 0.5
            x = x2 + t*(x1-x2)
            x += 0.25*(error+5e-15*abs(x1+x2))*sign(t-0.5)*sign(x2-x1) # Apply Tolerance
        
        fx = f(x)
        
        """======For seeing iterations======"""
        if print_flag: print(f'f({x})  \t= {fx}')
        
        """Swap to ensure x replaces x2"""
        if sign(f1) == sign(fx):
            t = 1-t
            x1, x2 = x2, x1
            f1, f2 = f2, f1
        
        x, x2, x3 = x3, x, x2
        fx, f2, f3 = f3, fx, f2
        
        """Update counters"""
        n += 1
        if t < 0.5: bisection_fails += 1
        else: bisection_fails = 0
        
        """Compute t for next iteration"""
        t = bracketing_method(x1, f1, x2, f2, x3, f3, x, fx, t)

        """Adjust t to ensure convergence"""
        if bisection_fails == 3:
            t *= 3
            if t > 0.5: t = 0.5
        if bisection_fails == 4 or t >= 1 or t <= 0 or is_nan(t):
            t = 0.5
    
    """Return secant iteration"""
    return (x1*f2-x2*f1)/(f2-f1)

def optimize(g, x1, x2,
                f = None,
                method = None,
                iterations = None,
                error = None):
    """Runs an optimization method to find relative extrema of g between x1 and x2.
    Requires g to be increasing on one side and decreasing on the other.
    This is measured by f(x) = g(x+dx) - g(x-dx), where dx is based on the error.
    solver.bracket(f, x1, x2, method, iterations, error) is then used.
    
    @params:
        g: searched function
        x1, x2: bracketing points
        f: derivative of g, approximated if not given
        method: bracketing method to be used
        iterations: maximum number of iterations before pure bisection is used
        error: desired error* of |x1-x2| to terminate.
            error*: (error + 5e-15*|x1+x2|), a combination of absolute and relative errors.
    
    @defaults:
        method: None, which uses the default method for bracket.
        iterations: 1000 iterations.
        error: 1e-8 since usually impossible to search further due to roundoff errors.
    
    @return:
        x: the result returned by solver.root_in().
    """
    
    """Set default iterations and error"""
    if error is None: error = 1e-8
    
    """Symmetric difference"""
    if f is None:
        def f(x):
            """======For seeing iterations======"""
            if print_flag: print(f'\ng({x})  \t= {g(x)}') # use 0.5*(g(x+dx)+g(x-dx)) to avoid an additional evaluation of g
            
            dx = error + 1e-8*abs(x)
            return g(x+dx) - g(x-dx)
    
    x = root_in(f, x1, x2, method, iterations, error)
    if print_flag: print(f'\ng({x})  \t= {g(x)}')
    return x
