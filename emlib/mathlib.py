"""
Miscellaneous math utilities

* base conversion
* derivatives
* dimension reduction
* range alternatives
* etc

"""
from __future__ import annotations
import operator as _operator
import random as _random
from functools import reduce
from fractions import Fraction
from math import gcd, sqrt, cos, sin, radians, ceil
import sys as _sys
import numpy as np

from emlib.typehints import T, T2, number_t, List, Tup, Opt, Iter, Seq, Callable


__all__ = ("PHI",
           "intersection",
           "frange",
           "fraction_range",
           "linspace",
           "clip",
           "prod",
           "lcm",
           "min_common_denominator",
           "convert_any_base_to_base_10",
           "convert_base_10_to_any_base",
           "convert_base",
           "euclidian_distance",
           "geometric_mean",
           "harmonic_mean",
           "split_interval_at_values",
           "derivative",
           "logrange",
           "randspace",
           "fib",
           "roundrnd",
           "roundres",
           "next_in_grid",
           "modulo_shortest_distance",
           "rotate2d",
           "optimize_parameter"
           )


# phi, in float (double) form and as Rational number with a precission of 2000
# iterations in the fibonacci row (fib[2000] / fib[2001])
PHI = 0.6180339887498949


def intersection(u1:T, u2:T, v1:T, v2:T) -> Opt[Tup[T, T]]:
    """
    return the intersection of (u1, u2) and (v1, v2) or None if no intersection

    Args:
        u1: lower bound of range U
        u2: higher bound of range U
        v1: lower bound of range V
        v2: higher bound of range V

    Returns:
        the intersection between range U and range V, or None if
        there is no intersection

    Example::

        >>> if intersec := intersection(0, 3, 2, 5):
        ...     x0, x1 = intersec

    """
    x0 = u1 if u1 > v1 else v1
    x1 = u2 if u2 < v2 else v2
    return (x0, x1) if x0 < x1 else None


def frange(start:float, stop:float=None, step:float=None) -> Iter[float]:
    """
    Like xrange(), but returns list of floats instead

    All numbers are generated on-demand using generators

    Args:
        start: start value of the range
        stop: stop value of the range
        step: step between values

    Returns:
        an iterator over the values

    Example
    -------

    >>> list(frange(4, step=0.5))
    [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5]

    """
    if stop is None:
        stop = float(start)
        start = 0.0
    if step is None:
        step = 1.0
    numiter = int((stop - start) / step)
    for i in range(numiter):
        yield start + step*i


def asFraction(x) -> Fraction:
    """ Convert x to a Fraction if it is not already one """
    if isinstance(x, Fraction):
        return x
    return Fraction(x)


def fraction_range(start:number_t, stop:number_t=None, step:number_t=None
                   ) -> Iter[Fraction]:
    """ Like range, but yielding Fractions """
    if stop is None:
        stop = asFraction(start)
        start = Fraction(0)
    else:
        start = asFraction(start)

    if step is None:
        step = Fraction(1)
    else:
        step = asFraction(step)
    while start < stop:
        yield start
        start += step


def linspace(start:float, stop:float, numitems:int) -> List[float]:
    """ Similar to numpy.linspace, returns a python list """
    dx = (stop - start) / (numitems - 1)
    return [start + dx*i for i in range(numitems)]


def linlin(x: T, x0:T, x1:T, y0:T, y1:T) -> T:
    """
    Convert x from range x0-x1 to range y0-y1
    """
    return (x - x0) * (y1 - y0) / (x1-x0) + y0


def clip(x:T, minvalue:T, maxvalue:T) -> T:
    """
    clip the value of x between minvalue and maxvalue
    """
    if x < minvalue:
        return minvalue
    elif x < maxvalue:
        return x
    return maxvalue


def lcm(*numbers: int) -> int:
    """
    Least common multiplier between a seq. of numbers

    Example
    -------

    >>> lcm(3, 4, 6)
    12

    """
    def lcm2(a, b):
        return (a*b) // gcd(a, b)

    return reduce(lcm2, numbers, 1)


def min_common_denominator(floats: Iter[float], limit:int = int(1e10)) -> int:
    """
    find the min common denominator to express floats as fractions

    Examples
    --------

    >>> common_denominator((0.1, 0.3, 0.8))
    10
    >>> common_denominator((0.1, 0.3, 0.85))
    20
    """
    fracs = [Fraction(f).limit_denominator(limit) for f in floats]
    return lcm(*[f.denominator for f in fracs])


def convert_base_10_to_any_base(x:int, base:int) -> str:
    """
    Converts given number x, from base 10 to base b

    Args:
        x: the number in base 10
        base: base to convert

    Returns:
        x expressed in base *base*, as string
    """
    assert(x >= 0)
    assert(1< base < 37)
    r = ''
    import string
    while x > 0:
        r = string.printable[x % base] + r
        x //= base
    return r


def convert_any_base_to_base_10(s:str, base:int) -> int:
    """
    Converts given number s, from base b to base 10

    Args:
        s: string representation of number
        base: base of given number

    Returns:
        s in base 10, as int
    """
    assert(1 < base < 37)
    return int(s, base)


def convert_base(s:str, frombase:int, tobase:int) -> str:
    """
    Converts s from base a to base b

    Args:
        s: the number to convert, expressed as str
        frombase: the base of *s*
        tobase: the base to convert to

    Returns:
        *s* expressed in base *tobase*
    """
    if frombase == 10:
        x = int(s)
    else:
        x = convert_any_base_to_base_10(s, frombase)
    if tobase == 10:
        return str(x)
    return convert_base_10_to_any_base(x, tobase)


def euclidian_distance(values: Seq[float], weights: Seq[float]=None) -> float:
    """
    Reduces distances in multiple dimensions to 1 dimension.

    e_distance_unweighted = sqrt(sum(value**2 for value in values))

    Args:
        values: distances to the origin
        weights: each dimension can have a weight (a scaling factor)

    Returns:
        the euclidian distance
    """
    if weights:
        s = sum(value**2 * weight for value, weight in zip(values, weights))
        return sqrt(s)
    return sqrt(sum(value**2 for value in values))


def weighted_euclidian_distance(pairs: List[Tuple(float, float)]) -> float:
    """
    Reduces distances in multiple dimensions to 1 dimension.

    e_distance_unweighted = sqrt(sum(value**2 for value in values))

    Args:
        pairs: a list of pairs (value, weight)

    Returns:
        the euclidian distance
    """

    values, weights = zip(*pairs)
    return euclidian_distance(values=values, weights=weights)


def prod(numbers: Seq[number_t]) -> number_t:
    """
    Returns the product of the given numbers
    ::

        x0 * x1 * x2 ... * xn | x in numbers
    """
    return reduce(_operator.mul, numbers)


def geometric_mean(numbers: Seq[number_t]) -> float:
    """
    The geometric mean is often used to find the mean of data measured in different units.
    """
    return prod(numbers) ** (1/len(numbers))


def harmonic_mean(numbers: Seq[number_t]) -> number_t:
    """
    The harmonic mean is used to calculate F1 score.

    (https://en.wikipedia.org/wiki/F-score)
    """
    T = type(numbers[0])
    one = T(1)
    return one/(sum(one/n for n in numbers) / len(numbers))


def split_interval_at_values(start: number_t, end: number_t, offsets: Seq[number_t]
                             ) -> List[Tup[number_t, number_t]]:
    """
    Split interval (start, end) at the given offsets

    Args:
        start: start of the interval
        end: end of the interval
        offsets: offsets to split the interval at. Must be sorted

    Returns:
        a list of (start, end) segments where no segment extends over any
        of the given offsets

    Example::

        >>> split_interval_at_values(1, 3, [1.5, 2])
        [(1, 1.5), (1.5,  2), (2, 3)]

        >>> split_interval_at_values(0.2, 4.3, list(range(10)))
        [(0.2, 1), (1, 2), (2, 3), (3, 4), (4, 4.3)]
    """
    assert end > start
    assert offsets

    if offsets[0] > end or offsets[-1] < start:
        # no intersection, return the original time range
        return [(start, end)]

    out = []
    for offset in offsets:
        if offset >= end:
            break
        if start < offset:
            out.append((start, offset))
            start = offset
    if start != end:
        out.append((start, end))

    assert len(out) >= 1
    return out


def derivative(func: Callable[[number_t], number_t], h=0) -> Callable[[number_t], float]:
    """
    Return a function which is the derivative of the given func

    **NB**: Calculated via complex step finite difference

    To find the derivative at x, do::

        derivative(func)(x)

    **VIA**: https://codewords.recurse.com/issues/four/hack-the-derivative
    """
    if h <= 0:
        h = _sys.float_info.min
    return lambda x: (float(func(x+h*1.0j))).imag / h


def logrange(start:float, stop:float, num=50, base=10) -> np.ndarray:
    """
    create an array [start, ..., stop] with a logarithmic scale
    """
    log = np.log
    if start == 0:
        start = 0.000000000001
    return np.logspace(log(start, base), log(stop, base), num, base=base)


def randspace(begin: float, end: float, numsteps: int, include_end=True
              ) -> List[float]:
    """
    go from begin to end in numsteps at randomly spaced steps

    Args:
        begin: start number
        end: end number
        numsteps: number of elements to generate
        include_end: include the last value (like np.linspace)

    Returns:
        a list of floats of size *numsteps* going from *begin* to *end*

    """
    if include_end:
        numsteps -= 1
    N = sorted(_random.random() for _ in range(numsteps))
    D = (end - begin)
    Nmin, Nmax = N[0], N[-1]
    out = []
    for n in N:
        delta = (n - Nmin) / Nmax
        out.append(delta * D + begin)
    if include_end:
        out.append(end)
    return out


def _fib2(N: float) -> Tup[float, float]:
    if N == 0:
        return 0, 1
    half_N, is_N_odd = divmod(N, 2)
    f_n, f_n_plus_1 = _fib2(half_N)
    f_n_squared = f_n * f_n
    f_n_plus_1_squared = f_n_plus_1 * f_n_plus_1
    f_2n = 2 * f_n * f_n_plus_1 - f_n_squared
    f_2n_plus_1 = f_n_squared + f_n_plus_1_squared
    if is_N_odd:
        return (f_2n_plus_1, f_2n + f_2n_plus_1)
    return (f_2n, f_2n_plus_1)


def fib(n: float) -> float:
    """
    calculate the fibonacci number *n* (accepts fractions)
    """
    # matrix code from http://blog.richardkiss.com/?p=398
    if n < 60:
        SQRT5 = 2.23606797749979  # sqrt(5)
        PHI = 1.618033988749895
        # PHI = (SQRT5 + 1) / 2
        return int(PHI ** n / SQRT5 + 0.5)
    else:
        return _fib2(n)[0]




def roundrnd(x: float) -> float:
    """
    Round *x* to its nearest integer, taking the fractional part as the probability

    3.5 will have a 50% probability of rounding to 3 or to 4
    3.1 will have a 10% probability of rounding to 3 and 90% prob. of rounding to 4
    """
    return int(x) + int(_random.random() > (1 - (x % 1)))


def roundres(x, resolution=1.0):
    """
    Round x with given resolution

    Example
    ~~~~~~~

    >>> roundres(0.4, 0.25)
    0.5
    >>> roundres(1.3, 0.25)
    1.25
    """
    return round(x / resolution) * resolution


def next_in_grid(x: float, step: float, offset=0.) -> float:
    return offset + ceil((x - offset) / step) * step


def modulo_shortest_distance(x, origin, mod):
    """
    Return the shortest distance to x from origin around a circle of modulo `mod`.

    A positive value means move clockwise, negative value means anticlockwise.
    Use abs to calculate the absolute distance

    Example
    -------

    Calculate the interval between two pitches, independently of octaves

    >>> interval = modulo_shortest_distance(n2m("D5"), n2m("B4"), 12)
    3
    """
    xclock = (x - origin) % mod
    xanti = (origin - x) % mod
    if xclock < xanti:
        return xclock
    return -xanti


def rotate2d(point: Tup[float, float],
             degrees: float,
             origin=(0, 0)) -> Tup[float, float]:
    """
    A rotation function that rotates a point around an origin

    Args:
        point:   the point to rotate as a tuple (x, y)
        degrees: the angle to rotate (counterclockwise)
        origin:  the point acting as pivot to the rotation

    Returns:
        the rotated point, as a tuple (x, y)
    """
    x = point[0] - origin[0]
    yorz = point[1] - origin[1]
    newx = (x*cos(radians(degrees))) - (yorz*sin(radians(degrees)))
    newyorz = (x*sin(radians(degrees))) + (yorz*cos(radians(degrees)))
    newx += origin[0]
    newyorz += origin[1]
    return newx, newyorz


def periodic_float_to_fraction(s: str) -> Fraction:
    """
    Convert a float with a periodic part to its fraction

    Args:
        s: the numer as string. Notate the periodic part 
           (for example 1/3=0.333...)
           as 0.(3, without repetitions. For example, 2.83333... as 2.8(3

    Returns:
        the fraction which results in the same periodic float

    Notation::

        12.3(17     2.3171717...
        123.45(67   123.45676767...

    """
    s2 = s.replace("(", "")
    x = float(s.replace("(", ""))
    numDecimals = len(s2.split(".")[1])
    lenPeriod = len(s.split("(")[1])
    factorA = 10 ** numDecimals
    factorB = 10 ** (numDecimals-lenPeriod)
    den = factorA - factorB
    num = int(x*factorA) - int(x*factorB)
    return Fraction(num, den)


def optimize_parameter(func, val: float, paraminit: float, maxerror=0.001,
                       maxiterations=100) -> Tup[float, int]:
    """
    Optimize one parameter to arrive to a desired value.

    Example
    -------

    .. code::

        # find the exponent of a bpf were its value at 0.1 is 1.25
        # (within the given relative error)
        >>> func = lambda param: bpf.expon(0, 1, 1, 6, exp=param)(0.1)
        >>> expon, numiter = optimize_parameter(func=func, val=1.25, paraminit=2)
        >>> bpf.expon(0, 1, 1, 6, exp=expon)(0.1)
        1.25

    Args:
        func: a function returning a value which will be compared to `val`
        val: the desired value to arrive to
        paraminit: the initial value of param
        maxerror: the max. relative error (0.001 is 0.1%)
        maxiterations: max. number of iterations

    Returns:
        a tuple (value, number of iterations)
    """
    param = paraminit
    for i in range(maxiterations):
        valnow = func(param)
        relerror = abs(valnow - val) / valnow
        if relerror < maxerror:
            break
        if valnow > val:
            param = param * (1+relerror)
        else:
            param = param * (1-relerror)
    return valnow, i
