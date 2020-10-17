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

print_flag = True

bracketing_method_dict = {
    'bisection' : bisection,
    'binary search' : bisection,
    'regula falsi' : secant,
    'false position': secant,
    'secant' : secant,
    'dekker' : dekker,
    'brent' : brent,
    'chandrupatla' : chandrupatla  # default
}

def help(input = None):
    if input is None:
        print("""Usage:
solver.root_in(f, x1, x2, {method, iterations, error, f1, f2})
solver.optimize(g, x1, x2, {method, iterations, error, f})

@params
    g: the searched function for the local extrema.
    f: the searched function for the root. Represents the derivative of g for solver.optimize.
    x1, x2: initial bracket.
    method: string name of the method used for solver.root_in.
    iterations: limit on the number of iterations before binary search is used instead of the chosen method.
    error: desired error of |x1-x2|.
    f1, f2: optional initial values of f(x1) and f(x2).

Use solver.help('root_in'), solver.help('optimize'), or solver.help('methods') for more specific information, or check the github wiki at https://github.com/SimpleArt/solver/wiki.
""")
    if input == 'root_in':
        print("""Usage:
solver.root_in(f, x1, x2, {method, iterations, error, f1, f2})

@params
    f: the searched function for the root.
    x1, x2: initial bracket.
    method: string name of the method used for solver.root_in.
    iterations: limit on the number of iterations before binary search is used instead of the chosen method.
    error: desired error of |x1-x2|.
    f1, f2: optional initial values of f(x1) and f(x2).

@defaults
    method: 'chandrupatla'
    iterations: 1000
    error: 1e-14

Attempts to solve f(x) = 0 between x1 and x2, given f(x1) and f(x2) have different signs, where convergence to a root can be guaranteed using binary search, a.k.a. bisection.

To guarantee convergence, binary search, is used after 4 consecutive iterations without the bracketing interval halving.

To avoid unnecessary binary search when the root is actually being approached rapidly, the distance from the last computed point is tripled after 3 consecutive iterations without the bracketing interval halving.

Use solver.help('methods') for more specific information, or check the github wiki at https://github.com/SimpleArt/solver/wiki.
""")
    if input == 'optimize':
        print("""Usage:
solver.optimize(g, x1, x2, {method, iterations, error, f})

@params
    g: the searched function for the local extrema.
    x1, x2: initial bracket.
    f: the searched function for the root. Represents the derivative of g for solver.optimize.
    method: string name of the method used for solver.root_in.
    iterations: limit on the number of iterations before binary search is used instead of the chosen method.
    error: desired error of |x1-x2|.

@defaults
    f: g(x+dx) - g(x-dx), where dx = error + 1e-8*abs(x)
    method: 'chandrupatla'
    iterations: 1000
    error: 1e-8

Attempts to find extreme values of g(x) between x1 and x2, given g(x) is increasing at one point and decreasing at the other. Uses solver.root_in to find the root of f(x), the derivative of g(x).

Use solver.help('methods') for more specific information, or check the github wiki at https://github.com/SimpleArt/solver/wiki.
""")
    if input == 'methods':
        print("""bracketing_method_dict = {
    'bisection' : bisection,
    'binary search' : bisection,
    'regula falsi' : secant,
    'false position': secant,
    'secant' : secant,
    'dekker' : dekker,
    'brent' : brent,
    'chandrupatla' : chandrupatla  # default
}
}

bisection   : returns the midpoint of the interval.
secant      : returns the secant estimate of the root using x1 and x2.
dekker      : returns the secant estimate of the root, using x2 and x3.
brent       : returns the inverse quadratic interpolation estimate of the root when possible, using x1, x2, and x3.
chandrupatla: returns the inverse quadratic interpolation estimate of the root when the interpolation is monotone, using x1, x2, and x3.

Usage:
bracketing_method(x1, f1, x2, f2, x3, f3, x4, f4, t)

@params:
    x1, f1: bracketing point for x2.
    x2, f2: last estimate of the root.
    x3, f3: last removed point from the interval.
    x4, f4: second last removed point from the interval, initially None.
    t: last computed t, used to check if bisection was last used.

@returns:
    t: the combination of x1 and x2 for the next iteration,
       i.e. the next x is x1 + t*(x2-x1).
""")


def sign(x):
    """Returns the sign of x"""
    if x > 0: return 1
    if x < 0: return -1
    return 0

def bracketing_method(x1, f1, x2, f2, x3, f3, x4, f4, t):
    """Interface for the bracketing method.
    
    @params:
        x1, f1: bracketing point for x2.
        x2, f2: last estimate of the root.
        x3, f3: last removed point from the interval.
        x4, f4: second last removed point from the interval, initially None.
        t: last computed t.
    
    @returns:
        t: the combination of x1 and x2 for the next iteration,
           i.e. the next x is x1 + t*(x2-x1).
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
    """
    
    """Set bracketing method to be used"""
    if method in bracketing_method_dict:
        bracketing_method = bracketing_method_dict[method]
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
        if print_flag: print(f'f({x}) \t= {fx}')
        
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
                method = None,
                iterations = None,
                error = None,
                f = None):
    """Runs an optimization method to find relative extrema of g between x1 and x2.
    Requires g to be increasing on one side and decreasing on the other.
    This is measured by f(x) = g(x+dx) - g(x-dx), where dx is based on the error.
    solver.bracket(f, x1, x2, method, iterations, error) is then used.
    """
    
    """Set default iterations and error"""
    if error is None: error = 1e-8
    
    """Symmetric difference"""
    if f is None:
        def f(x):
            # use g(x) = 0.5*(g(x+dx)+g(x-dx)) to avoid an additional evaluation of g
            
            """======For seeing iterations======"""
            if print_flag: print(f'\ng({x}) \t= {g(x)}')
            
            if is_infinite(x) and is_infinite(g(x)): return g(x) * sign(x)
            
            dx = error + 1e-8*abs(x)
            return g(x+dx) - g(x-dx)
    
    x = root_in(f, x1, x2, method, iterations, error)
    if print_flag: print(f'\ng({x}) \t= {g(x)}')
    return x
