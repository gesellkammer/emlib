"""
Calculus utilities, based on calculus.jl

"""
from __future__ import annotations
import math
import struct as _struct
import random as _random
from typing import Callable

NAN = float('nan')
epsilon = math.ldexp(1.0, -53)      # smallest double such that eps+0.5!=0.5
maxfloat = float(2**1024 - 2**971)  # From the IEEE 754 standard
minfloat = math.ldexp(1.0, -1022)   # min positive normalized double
smalleps = math.ldexp(1.0, -1074)   # smallest increment for doubles < minfloat
infinity = math.ldexp(1.0, 1023) * 2


def nextafter(x:float, direction=1) -> float:
    """
    returns the next float after x in the direction indicated

    if not possible, returns x

    Args:
        x: the value to evaluate
        direction: if 1, the next float is searched upwards, otherwise downwards

    Returns:
        a next representable float from `x` in the direction indicated
    """
    if math.isnan(x) or math.isinf(x):
        return x
    # return small numbers for x very close to 0.0
    if -minfloat < x < minfloat:
        if direction > 0:
            return x + smalleps
        else:
            return x - smalleps

    # it looks like we have a normalized number
    # break x down into a mantissa and exponent
    m, e = math.frexp(x)

    # all the special cases have been handled
    if direction > 0:
        m += epsilon
    else:
        m -= epsilon
    return math.ldexp(m, e)


def eps(x:float) -> float:
    """
    Difference with the next representable float
    """
    if math.isinf(x):
        return NAN
    return abs(nextafter(x) - x)


def next_float_up(x:float) -> float:
    """
    Return the next representable float
    """
    # NaNs and positive infinity map to themselves.
    if math.isnan(x) or (math.isinf(x) and x > 0):
        return x

    # 0.0 and -0.0 both map to the smallest +ve float.
    if x == 0.0:
        x = 0.0

    n = _struct.unpack('<q', _struct.pack('<d', x))[0]
    if n >= 0:
        n += 1
    else:
        n -= 1
    return _struct.unpack('<d', _struct.pack('<q', n))[0]


def next_float_down(x:float) -> float:
    """
    return the previous representable float
    """
    return -next_float_up(-x)


def next_toward(x:float, y:float) -> float:
    """
    return the next representable float between x and y
    """
    # If either argument is a NaN, return that argument.
    # This matches the implementation in decimal.Decimal
    if math.isnan(x):
        return x
    if math.isnan(y):
        return y
    if y == x:
        return y
    elif y > x:
        return next_float_up(x)
    else:
        return next_float_down(x)


def cbrt(x:float) -> float:
    """ cubic root """
    return math.pow(x, 1.0 / 3)


def finite_difference_forward(func, x:float, h:float=None) -> float:
    epsilon = math.sqrt(eps(max(1, abs(x)))) if h is None else h
    # use machine-representable numbers
    return (func(x + epsilon) - func(x)) / epsilon


def finite_difference_central(func, x:float, h:float=None) -> float:
    epsilon = cbrt(eps(max(1, abs(x)))) if h is None else h
    return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon)


def finite_difference(func, x:float, mode='central') -> float:
    """
    derivative of func at x
    """
    if mode == 'forward':
        return finite_difference_forward(func, x)
    elif mode == 'central':
        return finite_difference_central(func, x)
    else:
        raise ValueError("mode must be 'forward' or 'central'")


def derivative(func) -> Callable[[float], float]:
    """
    return a new function representing the derivative of func
    """
    return lambda x: finite_difference_central(func, x)


def _integrate_adaptive_simpsons_inner(f:Callable, a:float, b:float, eps:float, S:float,
                                       fa:float, fb:float, fc:float, bottom:float) -> float:
    c = (a + b) / 2
    h = b - a
    d = (a + c) / 2
    g = (c + b) / 2
    fd = f(d)
    fe = f(g)
    Sleft = (h / 12) * (fa + 4 * fd + fc)
    Sright = (h / 12) * (fc + 4 * fe + fb)
    S2 = Sleft + Sright
    if bottom <= 0 or abs(S2 - S) <= 15 * epsilon:
        return S2 + (S2 - S) / 15
    inner = _integrate_adaptive_simpsons_inner
    return (
        inner(f, a, c, eps/2, Sleft, fa, fc, fd, bottom - 1) +
        inner(f, c, b, eps/2, Sright, fc, fb, fe, bottom - 1)
    )


def integrate_adaptive_simpsons(f:Callable, a:float, b:float, accuracy=10e-10,
                                max_iterations=50) -> float:
    c = (a + b) / 2
    h = b - a
    fa = f(a)
    fb = f(b)
    fc = f(c)
    S = (h / 6) * (fa + 4 * fc + fb)
    return _integrate_adaptive_simpsons_inner(f, a, b, accuracy,
                                              S, fa, fb, fc, max_iterations)


def integrate_monte_carlo(f:Callable, a:float, b:float, iterations:int):
    estimate = 0.0
    width = (b - a)
    for i in range(iterations):
        x = width * _random.random() + a
        estimate += f(x) * width
    return estimate / iterations


def integrate(f:Callable, a:float, b:float, method='simpsons') -> float:
    if method == 'simpsons':
        return integrate_adaptive_simpsons(f, a, b)
    elif method == 'montecarlo':
        return integrate_monte_carlo(f, a, b, 10000)
    else:
        raise ValueError("Unknown method of integration")
