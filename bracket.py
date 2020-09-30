from Bracketing_Methods.bisection import bisection
from Bracketing_Methods.secant import secant
from Bracketing_Methods.dekker import dekker
from Bracketing_Methods.brent import brent
from Bracketing_Methods.chandrupatla import chandrupatla

print("Run bracket_help() for an explanation on how to use the root-finder program.")

def bracket_help():
    print("""These scripts allow one to find a root of a provided function, provided that an initial bracket is given. A bracket is made of two points x1 and x2 where f(x1) and f(x2) have opposing signs. The methods provided are guaranteed to converge to a root (or discontinuity/singularity). The provided methods are:

- Bisection
- Regula Falsi / False Position
- Illinois
- Dekker
- Brent
- Chandrupatla

Usage:
Required parameters: bracket(f, x1, x2) returns a root of f between x1 and x2.
Optional parameters: bracket(f, x1, x2, {method, iterations, error_x, error_f, safe, f1, f2, x3, f3})

Example:
---------------------------
  def f(x):
    return x*x - x - 1
  
  print(bracket(f, 1, 2, 'bisection'))
---------------------------

@params:
  f: searched function
  x1, x2: bracketing points
  method: bracketing method to be used
  iterations: maximum number of iterations to terminate
  error_x: desired error* of |x1-x2| to terminate
  error_f: desired error* of |(x1-x2)*f2/(f1-f2)| to terminate
      error* is measured by error * (1 + |x2|), a combination of absolute and relative errors.
  safe: guarantees convergence in no more than 5 times slower than bisection by forcing bisection to be used every 5th iteration if the bracketing interval does not reduce
  f1, f2: initial points
  x3, f3: additional point for 3-point interpolation method

@defaults:
  method: Chandrupatla's method
  iterations: 10000 iterations
  error_x, error_f: 5e-16 (near machine precision)
  safe: true (guaranteed convergence)
""")

def sign(x):
    """Returns the sign of x"""
    if x > 0: return 1
    if x == 0: return 0
    return -1

def bracketing_method(x, x1, x2, x3, fx, f1, f2, f3, error_x, flag):
    """Interface for the bracketing method.
    
    @params:
        x, fx: twice removed point from the interval, initially None
        x1, f1: bracketing point for x2
        x2, f2: last estimate of the root
        x3, f3: last removed point from the interval
        error_x: desired error for x
        flag: whether or not bisection was last used
    
    @returns:
        t: the combination of x1 and x2 for the next iteration
           i.e. the next x is x1 + t*(x2-x1)
    """
    
    pass

def bracket(f, x1, x2,
                method = None,
                iterations = None,
                error_x = None,
                error_f = None,
                safe = None,
                f1 = None,
                f2 = None,
                x3 = None,
                f3 = None):
        """Run a bracketing method to find a root of f between x1 and x2.
        
        @params:
            f: searched function
            x1, x2: bracketing points
            method: bracketing method to be used
            iterations: maximum number of iterations to terminate
            error_x: desired error* of |x1-x2| to terminate
            error_f: desired error* of |(x1-x2)*f2/(f1-f2)| to terminate
                error* is measured by error * (1 + |x2|), a combination of absolute and relative errors.
            safe: guarantees convergence in no more than 5 times slower than bisection by forcing bisection to be used every 5th iteration if the bracketing interval does not reduce
            f1, f2: optional initial points, computed if not given
            x3, f3: optional additional point for 3-point interpolation method
        
        @defaults:
            method: Chandrupatla's method
            iterations: 10000 iterations
            error_x, error_f: 1e-15
            safe: true (guaranteed convergence)
        
        @return:
            (f1*x2-f2*x1)/(f1-f2), the secant estimate with the final bracket
        """
        
        """Set bracketing method to be used"""
        
        if method == 'bisection':
            bracketing_method = bisection
        elif method in ['regula falsi', 'false position', 'secant', 'illinois']:
            bracketing_method = secant
        elif method == 'dekker':
            bracketing_method = dekker
        elif method == 'brent':
            bracketing_method = brent
        else:
            bracketing_method = chandrupatla
        
        """Set default iterations, errors, and safe"""
        
        if iterations is None: iterations = 10000
        if error_x is None: error_x = 1e-15
        if error_f is None: error_f = 1e-15
        if safe is None: safe = True
        
        """Compute initial points if needed.
        Return if root.
        Return one secant iteration if no bracket.
        """
        
        if f1 is None: f1 = f(x1)
        if f2 is None: f2 = f(x2)
        if abs((x1-x2)*f1/(f1-f2)) < error_f+5e-16*abs(x1): return x1
        if abs((x1-x2)*f2/(f1-f2)) < error_f+5e-16*abs(x2): return x2
        if sign(f1) == sign(f2) or abs(x1-x2) < error_x+3e-16*abs(x1+x2): return (x1*f2-x2*f1)/(f2-f1)
        
        """Initialize variables"""
        
        no_of_bracket_failed_halve = 0
        n = 0
        
        """Safe start is bisection, unsafe start is secant"""
        if safe:
            t = 0.5
        else:
            t = f2/(f1-f2)
        
        """Loop until convergence"""
        
        while n < iterations and abs(x1-x2) > error_x+3e-16*abs(x1+x2) and abs((x1-x2)*f2/(f1-f2)) > error_f+3e-16*abs(x1+x2):
            n += 1
            no_of_bracket_failed_halve += 1
            
            """Compute next point"""
            
            x = x2 + t*(x1-x2)
            x += 0.25*(error_x+3e-16*abs(x1+x2))*sign(0.5-t) # tolerance shift
            fx = f(x)
            
            """======For seeing iterations======"""
            print(f'f({x}) = {fx}')
            
            """Update all variables"""
            
            if sign(fx) == sign(f1):
                t = 1-t
                x1, x2 = x2, x1
                f1, f2 = f2, f1
            elif method == 'illinois':
                f1 *= 0.5
            
            if t >= 0.5: no_of_bracket_failed_halve = 0
            x, x2, x3 = x3, x, x2
            fx, f2, f3 = f3, fx, f2
            
            """Compute t for next iteration"""
            
            t = bracketing_method(x, x1, x2, x3, fx, f1, f2, f3, error_x, t == 0.5)
            
            """Safety to ensure convergence"""
            
            if safe:
                if no_of_bracket_failed_halve == 4: t *= 2
                if no_of_bracket_failed_halve == 5 or t > 0.5:
                    t = 0.5
        
        return (x1*f2-x2*f1)/(f2-f1)
