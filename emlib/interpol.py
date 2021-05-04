from math import cos, pow, exp, log
from functools import partial


MAX_FLOAT = 3.40282346638528860e+38
E = 2.718281828459045235360287471352662497757247093
PHI = 1.61803398874989484820458683436563811772030917
PI = 3.141592653589793238462643383279502884197169399375105


def interpol_linear(x, x0, y0, x1, y1):
    """
    interpolate between (x0, y0) and (x1, y1) at point x
    """
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


def ilin1(x, y0, y1):
    """
    x: a number between 0-1. 0: y0, 1: y1
    """
    return y0 + (y1-y0)*x
    
# ~~~~~~~~~~~~~~~~~~~~~~~
# Halfcos(exp)
# ~~~~~~~~~~~~~~~~~~~~~~~


def interpol_halfcos(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    interpolate between (x0, y0) and (x1, y1) at point x
    """
    dx = ((x - x0) / (x1 - x0)) * 3.14159265358979323846 + 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)


def interpol_halfcosexp(x: float, x0: float, y0: float, x1: float, y1: float, exp:float
                        ) -> float:
    """
    interpolate between (x0, y0) and (x1, y1) at point x

    exp defines the exponential of the curve
    """
    dx = pow((x - x0) / (x1 - x0), exp)
    dx = (dx + 1.0) * 3.14159265358979323846
    return y0 + ((y1 - y0) * (1 + cos(dx)) / 2.0)


def fib(x):
    """
    taken from home.comcast.net/~stuartmanderson/fibonacci.pdf
    fib at x = e^(x * ln(phi)) - cos(x * pi) * e^(x * ln(phi))
               -----------------------------------------------
                                     sqrt(5)
    """
    x_mul_log_phi = x * 0.48121182505960348   # 0.48121182505960348 = log(PHI)
    return (
        (exp(x_mul_log_phi) - cos(x * PI) * exp(-x_mul_log_phi)) /
        2.23606797749978969640917366873127623544
    )


def interpol_fib(x,  x0,  y0,  x1,  y1):
    """
    fibonacci interpolation. it is assured that if x is equidistant to
    x0 and x1, then for the result y it should be true that

    y1 / y == y / y0 == ~0.618
    """
    dx = (x - x0) / (x1 - x0)
    dx2 = fib(40 + dx * 2)
    dx3 = (dx2 - 102334155) / (165580141)
    return y0 + (y1 - y0) * dx3


def ifib1(x,  y0,  y1):
    """
    fibonacci interpolatation within the interval 0, 1

    the same as
    >> interpol_fib(x, 0, y0, 1, y1)

    if x is negative, the interval is reversed
    >> interpol_fib(abs(x), 1, y1, 0, y0)
    """
    if x < 0:
        return interpol_fib(x * -1, 1, y1, 0, y0)
    else:
        return interpol_fib(x, 0, y0, 1, y1)


def fib_gen(x, phi=PHI):
    if phi <= 1.001:
        raise ValueError("phi should be > 1.001")
    return (
        (exp(x * log(phi)) - cos(x * PI) * exp(x * log(phi) * -1)) /
        2.23606797749978969640917366873127623544
    )


def interpol_expon(x, x0, y0, x1, y1, exp):
    """
    interpolate between (x0, y0) and (x1, y1) at point x

    exp defines the exponential of the curve
    """
    dx = (x - x0) / (x1 - x0)
    return y0 + pow(dx, exp) * (y1 - y0)

    
def getfunc(descr):
    """
    returns a function performing the given interpolation

    Example
    ~~~~~~~

    >>> f = getfunc("expon(2.0)")
    >>> f(2)
    4.0
    """
    if "(" in descr:
        descr, exps = descr.split("(")
        exp = float(exps[:-1])
    else:
        exp = 1
    if descr == 'linear':
        return interpol_linear
    elif descr == 'halfcos':   
        if exp == 1:
            return interpol_halfcos
        else:
            return partial(interpol_halfcosexp, exp=exp)
    elif descr == 'expon':
        return partial(interpol_expon, exp=exp)
    elif descr == 'fib':
        return interpol_fib
    else:
        raise ValueError("descr not supported")


try:
    from interpoltools import (
        interpol_linear, 
        interpol_halfcos,
        interpol_expon,
        interpol_halfcosexp,
        interpol_fib
    )
    _ACCELERATED = True
except ImportError:
    _ACCELERATED = False

