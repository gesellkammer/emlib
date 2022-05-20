from __future__ import annotations
import numpy as np


def _parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    Args:
        f: a vector (an array of sampled values over a regular grid)
        x: index for that vector
   
    Returns:
        (vx, vy), the coordinates of the vertex of a parabola that goes
        through point x and its two neighbors.
   
    Example
    =======

    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    .. code::

        >>> f = [2, 3, 1, 6, 4, 2, 3, 1]
        >>> parabolic(f, 3)
        (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def parabolic(x: float, Y, sr: int, offset=0.):
    """
    Quadratic interpolation estimating the true position of an inter-sample maximum

    Args:
        Y: a sequence of y values, sampled regularly at samplerate=sr
        x: where the parabolic curve should be evaluated
        sr: the samplerate of Y
        offset: the offset of Y
    """
    index = int((x - offset) / sr)
    return _parabolic(Y, index)


def _parabolic_polyfit(f, x, n):
    """Use the built-in polyfit() function to find the peak of a parabola

    Args:
        f: a vector and x is an index for that vector.
        n: the number of samples of the curve used to fit the parabola.

    """    
    a, b, c = np.polyfit(np.arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
    xv = -0.5 * b/a
    yv = a * xv**2 + b * xv + c
    return (xv, yv)


def parabolic_polyfit(x, n, Y, sr, offset=0):
    """Use the built-in polyfit() function to find the peak of a parabola
    
    Y: an array of y-values resulting of sampling a curve at `sr` 
       with `offset`
    
    n is the number of samples of the curve used to fit the parabola.
    """        
    index = int((x - offset) / sr)
    return _parabolic_polyfit(Y, index, n)
    
