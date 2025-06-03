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
import sys as _sys
import functools as _functools
import math
import numpy as np

try:
    from quicktions import Fraction
except ImportError:
    from fractions import Fraction

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterator, Callable, Sequence, TypeVar, TypeAlias, Union
    from fractions import Fraction
    number_t: TypeAlias = Union[Fraction, float]
    T = TypeVar("T", float, Fraction)
    T2 = TypeVar("T2", bound=number_t)


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
           "interpfib",
           "roundrnd",
           "roundres",
           "next_in_grid",
           "modulo_shortest_distance",
           "rotate2d",
           "optimize_parameter",
           "ispowerof2",
           "nextpowerof2",
           )


class NotFoundError(ValueError):
    pass


# phi, in float (double) form and as Rational number with a precission of 2000
# iterations in the fibonacci row (fib[2000] / fib[2001])
PHI = 0.6180339887498949


def intersection(u1: T, u2: T, v1: T, v2: T) -> tuple[T | None, T]:
    """
    return the intersection of (u1, u2) and (v1, v2) or None if no intersection

    Args:
        u1: lower bound of range U
        u2: higher bound of range U
        v1: lower bound of range V
        v2: higher bound of range V

    Returns:
        the intersection between range U and range V as a tuple (start, end). If no intersection is found,
        start will be None.

    Example::

        >>> x0, x1 = intersection(0, 3, 2, 5)
        >>> if x0 is not None:
        ...     # there is an intersection

    """
    x0 = u1 if u1 > v1 else v1
    x1 = u2 if u2 < v2 else v2
    return (x0, x1) if x0 < x1 else (None, 0.)


def overlap(u1: T, u2: T, v1: T, v2: T) -> tuple[T, T] | None:
    """
    return the intersection of (u1, u2) and (v1, v2) or None if no intersection

    Args:
        u1: lower bound of range U
        u2: higher bound of range U
        v1: lower bound of range V
        v2: higher bound of range V

    Returns:
        the intersection between range U and range V as a tuple (start, end). If no intersection is found,
        start will be None.

    Example::

        >>> if intersect := overlap(0, 3, 2, 5):
        ...     x0, x1 = intersect
        ...     # there is an intersection

    """
    x0 = u1 if u1 > v1 else v1
    x1 = u2 if u2 < v2 else v2
    return (x0, x1) if x0 < x1 else None


def hasintersect(u1: T, u2: T, v1: T, v2: T) -> bool:
    """
    True if the intervals (u1, u2) and (v1, v2) intersect
    """
    return u2 > v1 if u1 < v1 else v2 > u1


def frange(start: float, stop: float = None, step: float = 0.) -> Iterator[float]:
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
    if not step:
        step = 1.0
    numiter = int((stop - start) / step)
    for i in range(numiter):
        yield start + step*i


def asfraction(x) -> Fraction:
    """ Convert x to a Fraction if it is not already one """
    return x if isinstance(x, Fraction) else Fraction(x)


def fraction_range(start: number_t, stop: number_t = None, step: number_t = None
                   ) -> Iterator[Fraction]:
    """ Like range, but yielding Fractions """
    if stop is None:
        stopF = asfraction(start)
        startF = Fraction(0)
    else:
        startF = asfraction(start)
        stopF = asfraction(stop)

    if step is None:
        step = Fraction(1)
    else:
        step = asfraction(step)
    while startF < stopF:
        yield startF
        startF += step


def linspace(start: float, stop: float, numitems: int) -> list[float]:
    """ Similar to numpy.linspace, returns a python list """
    dx = (stop - start) / (numitems - 1)
    return [start + dx*i for i in range(numitems)]


def linlin(x: T, x0: T, x1: T, y0: T, y1: T) -> T:
    """
    Convert x from range x0-x1 to range y0-y1
    """
    return (x - x0) * (y1 - y0) / (x1-x0) + y0


def clip(x: T, minvalue: T, maxvalue: T) -> T:
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
        return (a*b) // math.gcd(a, b)

    return _functools.reduce(lcm2, numbers, 1)


def min_common_denominator(floats: Iterator[float], limit: int = int(1e10)) -> int:
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


def convert_base_10_to_any_base(x: int, base: int) -> str:
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


def convert_any_base_to_base_10(s: str, base: int) -> int:
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


def convert_base(s: str, frombase: int, tobase: int) -> str:
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


def euclidian_distance(values: Sequence[float], weights: Sequence[float]=None) -> float:
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
        s = sum(value*value * weight for value, weight in zip(values, weights))
        return math.sqrt(s)
    return math.sqrt(sum(value**2 for value in values))


def weighted_euclidian_distance(pairs: list[tuple[float, float]]) -> float:
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


def prod(numbers: Sequence[number_t]) -> number_t:
    """
    Returns the product of the given numbers
    ::

        x0 * x1 * x2 ... * xn | x in numbers
    """
    return _functools.reduce(_operator.mul, numbers)


def geometric_mean(numbers: Sequence[number_t]) -> float:
    """
    The geometric mean is often used to find the mean of data measured in different units.
    """
    return float(prod(numbers)) ** (1/len(numbers))


def harmonic_mean(numbers: Sequence[T]) -> T:
    """
    The harmonic mean is used to calculate F1 score.

    (https://en.wikipedia.org/wiki/F-score)
    """
    one = type(numbers[0])(1)
    return one/(sum(one/n for n in numbers) / len(numbers))


def split_interval_at_values(start: T, end: T, offsets: Sequence[T]
                             ) -> list[tuple[T, T]]:
    """
    Split interval (start, end) at the given offsets

    Args:
        start: start of the interval
        end: end of the interval
        offsets: offsets to split the interval at. Must be sorted

    Returns:
        a list of (start, end) segments where no segment extends over any
        of the given offsets


    Example
    ~~~~~~~

    .. code::

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


def logrange(start: float, stop: float, num=50, base=10, minstart=1e-12) -> np.ndarray:
    """
    create an array [start, ..., stop] with a logarithmic scale
    """
    if start == 0:
        start = minstart
    return np.logspace(math.log(start, base), math.log(stop, base), num, base=base)


def randspace(begin: float, end: float, numsteps: int, include_end=True
              ) -> list[float]:
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


def _fib2(N: float) -> tuple[float, float]:
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


def interpfib(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Fibonacci interpolation

    Interpolate between ``(x0, y0)`` and ``(x1, y1)`` at *x* with fibonacci interpolation

    It is assured that if *x* is equidistant to
    *x0* and *x1*, then for the result *y* it should be true that::

        y1 / y == y / y0 == ~0.618
    """
    dx = (x-x0)/(x1-x0)
    dx2 = fib(40+dx*2)
    dx3 = (dx2 - 102334155) / 165580141
    return y0 + (y1 - y0)*dx3


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
    -------

    .. code:: python

        >>> roundres(0.4, 0.25)
        0.5
        >>> roundres(1.3, 0.25)
        1.25
    """
    return round(x / resolution) * resolution


def next_in_grid(x: T, step: T, offset: T) -> T:
    """
    The next value in the grid defined by step/offset which is >= x

    Args:
        x: a value to fit to the next division of a grid
        step: the step of the grid
        offset: the offset of the grid

    Returns:
        the next value within the grid higher or equal to x

    Example
    ~~~~~~~

        >>> next_in_grid(4.3, 0.5, 0)
        4.5
        >>> from fractions import Fraction
        >>> next_in_grid(1.31, Fraction(1, 3))
        Fraction(4, 3)
    """
    grididx = math.ceil((x - offset) / step)
    if isinstance(step, Fraction):
        return step * Fraction(grididx) + Fraction(offset)
    else:
        assert isinstance(step, (int, float)) and isinstance(offset, (int, float))
        return float(step * grididx + offset)  # type: ignore


def modulo_shortest_distance(x: number_t, origin: number_t, mod: int):
    """
    Return the shortest distance to x from origin around a circle of modulo `mod`.

    A positive value means move clockwise, negative value means anticlockwise.
    Use abs to calculate the absolute distance

    Example
    -------

    Calculate the interval between note D5 and B3, independently of octaves

    .. code-block:: python

        >>> interval = modulo_shortest_distance(74, 59, 12)
        3
    """
    xclock = (x - origin) % mod
    xanti = (origin - x) % mod
    if xclock < xanti:
        return xclock
    return -xanti


def rotate2d(point: tuple[float, float],
             degrees: float,
             origin=(0, 0)
             ) -> tuple[float, float]:
    """
    A rotation function that rotates a point around an origin

    Args:
        point: the point to rotate as a tuple (x, y)
        degrees: the angle to rotate (counterclockwise)
        origin: the point acting as pivot to the rotation

    Returns:
        the rotated point, as a tuple (x, y)
    """
    x = point[0] - origin[0]
    yorz = point[1] - origin[1]
    newx = (x*math.cos(math.radians(degrees))) - (yorz*math.sin(math.radians(degrees)))
    newyorz = (x*math.sin(math.radians(degrees))) + (yorz*math.cos(math.radians(degrees)))
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


def fraction_to_decimal(numerator: int, denominator: int) -> str:
    """
    Converts a fraction to a decimal number with repeating period

    Args:
        numerator: the numerator of the fraction
        denominator: the denominator of the fraction

    Returns:
        the string representation of the resulting decimal. Any repeating
        period will be prefixed with '('

    Example
    ~~~~~~~

        >>> from emlib.mathlib import *
        >>> fraction_to_decimal(1, 3)
        '0.(3'
        >>> fraction_to_decimal(1, 7)
        '0.(142857'
        >>> fraction_to_decimal(100, 7)
        '14.(285714'
        >>> fraction_to_decimal(355, 113)
        '3.(1415929203539823008849557522123893805309734513274336283185840707964601769911504424778761061946902654867256637168'
    """
    result = [str(numerator//denominator) + "."]
    subresults = [numerator % denominator]
    numerator %= denominator
    while numerator != 0:
        numerator *= 10
        result_digit, numerator = divmod(numerator, denominator)
        result.append(str(result_digit))
        if numerator not in subresults:
            subresults.append(numerator)
        else:
            result.insert(subresults.index(numerator) + 1, "(")
            break
    return "".join(result)


def optimize_parameter(func,
                       val: float,
                       paraminit: float,
                       maxerror=0.001,
                       maxiterations=100
                       ) -> tuple[float, int]:
    """
    Optimize one parameter to arrive to a desired value.

    Will raise NotFoundError if the maximum iterations are reached without
    converging

    Args:
        func: a function returning a value which will be compared to `val`
        val: the desired value to arrive to
        paraminit: the initial value of param
        maxerror: the max. relative error (0.001 is 0.1%)
        maxiterations: max. number of iterations

    Returns:
        a tuple (value, number of iterations)


    Example
    -------

    .. code::

        # find the exponent of a bpf were its value at 0.1 is 1.25
        # (within the given relative error)
        >>> func = lambda param: bpf.expon(0, 1, 1, 6, exp=param)(0.1)
        >>> expon, numiter = optimize_parameter(func=func, val=1.25, paraminit=2)
        >>> bpf.expon(0, 1, 1, 6, exp=expon)(0.1)
        1.25

    """
    if val == 0:
        raise ValueError("Cannot optimize for 0")

    param = paraminit
    for i in range(maxiterations):
        valnow = func(param)
        relerror = abs(valnow - val) / val
        if relerror < maxerror:
            break
        if valnow > val:
            param = param * (1+relerror)
        else:
            param = param * (1-relerror)
    else:
        valnow = func(param)
        relerror = abs(valnow - val) / valnow
        raise NotFoundError(f"The maximum iterations were reached without converging."
                            f" Last param: {param}, value: {valnow}, relerror: {relerror}")

    return param, i


def intersection_area_between_circles(x1: float, y1: float, r1: float,
                                      x2: float, y2: float, r2: float
                                      ) -> float:
    """
    Calculate the area of the intersection between two circles

    Args:
        x1: x coord of the center of the 1st circle
        y1: y coord of the center of the 1st circle
        r1: ratio of the 1st circle
        x2: x coord of the center of the 2nd circle
        y2: y coord of the center of the 2nd circle
        r2: ratio of the 2nd circle

    Returns:
        the area of the intersection

    from https://www.xarg.org/2016/07/calculate-the-intersection-area-of-two-circles/
    """
    d = math.hypot(x2 - x1, y2 - y1)
    if d >= r1 + r2:
        return 0
    a = r1 * r1
    b = r2 * r2
    if d != 0:
        x = (a - b + d*d) / (2*d)
    else:
        # They share the same center!
        return math.pi*min(a, b)
    z = x * x
    if a < z:
        # One circle is embedded in the other?
        return math.pi * min(a, b)
    y = math.sqrt(a - z)
    if d <= abs(r2 - r1):
        return math.pi * min(a, b)
    return a * math.asin(y / r1) + b*math.asin(y / r2) - y * (x + math.sqrt(z + b - a))


def roman(n: int) -> str:
    """
    Convert an integer to its roman representation

    Args:
        n: the integer to convert

    Returns:
        the roman representation

    credit: https://www.geeksforgeeks.org/python-program-to-convert-integer-to-roman/
    """
    romans = {
        1: "I",
        5: "V",
        10: "X",
        50: "L",
        100: "C",
        500: "D",
        1000: "M",
        5000: "G",
        10000: "H"
    }

    div = 1
    while n >= div:
        div *= 10

    div //= 10
    res = ""
    while n:
        # main significant digit extracted into lastNum
        lastNum = n // div
        if lastNum <= 3:
            res += (romans[div] * lastNum)
        elif lastNum == 4:
            res += (romans[div] + romans[div * 5])
        elif 5 <= lastNum <= 8:
            res += (romans[div * 5] + (romans[div] * (lastNum - 5)))
        elif lastNum == 9:
            res += (romans[div] + romans[div * 10])
        n = math.floor(n % div)
        div //= 10
    return res


def fractional_factorial(x: float) -> float:
    """
    Ramanujan's approximation of factorial of x, where x does not need to be an integer

    .. note::

        If x is in fact an integer the returned value will be an integral float

    Via: https://www.johndcook.com/blog/2012/09/25/ramanujans-factorial-approximation/

    """
    if isinstance(x, int):
        return math.factorial(x)

    fact = math.sqrt(math.pi)*(x/math.e)**x
    fact *= (((8*x + 4)*x + 1)*x + 1/30.)**(1./6.)
    return fact


def ispowerof2(x: int) -> bool:
    """
    Is x a power of two?

    Via: https://stackoverflow.com/questions/600293/how-to-check-if-a-number-is-a-power-of-2
    """
    return (x != 0) and ((x & (x - 1)) == 0)


def nextpowerof2(x) -> int:
    """
    Return the lowest power of two >= x
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def gini(xs: Sequence[T], eps=1e-12, normalize=True) -> float:
    """
    Calculate the Gini coefficient for a given sequence of values.

    The Gini coefficient is a measure of statistical dispersion intended to represent
    income or wealth distribution within a nation or a social group.

    Args:
        xs (Sequence[T]): A sequence of values for which the Gini coefficient is to be calculated.

    Returns:
        The Gini coefficient value. A value of 1 indicates perfect inequality, while a value of 0
        indicates perfect equality.
    """
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = np.asarray(xs)
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + eps
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    index2 = 2 * index - (n + 1)
    coeff = float((np.sum(index2 * array)) / (n * np.sum(array)))
    if normalize:
        maxcoeff = (array.shape[0] * 2 - (n+1)) / n
        coeff /= maxcoeff
        if coeff > 1:
            coeff = 1
    if coeff < eps:
        coeff = 0
    return coeff


def exponcurve(num: int, exp: float, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """
    Generate an exponential curve between two points

    Args:
        num: number of points to generate
        exp: exponent of the curve
        x0: start x-coordinate
        y0: start y-coordinate
        x1: end x-coordinate
        y1: end y-coordinate

    Returns:
        a numpy array of shape (num,) containing the y-coordinates of the curve.

    """
    xs = np.linspace(x0, x1, num)
    dxs = (xs - x0) / (x1 - x0)
    ys = (dxs ** exp) * (y1 - y0) + y0
    return ys
