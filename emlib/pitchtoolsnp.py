"""
Similar to pitchtools, but on numpy arrays
"""

import numpy as np
from emlib.pitchtools import A4, set_reference_freq, n2m


import sys
_EPS = sys.float_info.epsilon


def f2m_np(freqs: np.ndarray, out:np.ndarray=None) -> np.ndarray:
    """
    vectorized version of f2m

    freqs: an array of frequencies
    out: if given, put the result in out

    formula:

    if freq < 9:
        return 0
    return 12.0 * log(freq/A4, 2) + 69.0
    """
    freqs = np.asarray(freqs, dtype=float)
    if out is None:
        return 12.0 * np.log2(freqs/A4) + 69.0
    x = freqs/A4
    np.log2(x, out=x)
    x *= 12.0
    x += 69.0
    return x


def m2f_np(midinotes: np.ndarray, out:np.ndarray=None) -> np.ndarray:
    """
    Vectorized version of m2f

    midinotes: an array of midinotes
    out: if given, put the result here
    """
    midinotes = np.asarray(midinotes, dtype=float)
    out = np.subtract(midinotes, 69, out=out)
    out /= 12.
    out = np.power(2.0, out, out)
    out *= A4
    return out


def db2amp_np(db:np.ndarray, out:np.ndarray=None) -> np.ndarray:
    """
    Vectorized version of db2amp
    
    db: a np array of db values
    out: if given, put the result here
    """
    # amp = 10.0**(0.05*db)
    out = np.multiply(db, 0.05, out=out)
    out = np.power(10, out, out=out)
    return out


def amp2db_np(amp:np.ndarray, out:np.ndarray=None) -> np.ndarray:
    """
    Vectorized version of amp2db
    
    amp: a np array of db values
    out: if given, put the result here
    """
    # db = log10(amp)*20
    X = np.maximum(amp, _EPS, out=out)
    X = np.log10(X, out=X)
    X *= 20
    return X


def logfreqs(notemin=0.0, notemax=139.0, notedelta=1.0) -> np.ndarray:
    """
    Return a list of frequencies corresponding to the pitch range given

    notemin, notemax, notedelta: as used in arange (notemax is included)

    Example 1: generate a list of frequencies of all audible semitones
    
    >>> logfreqs(0, 139, notedelta=1)

    Example 2: generate a list of frequencies of instrumental 1/4 tones

    >>> logfreqs(n2m("A0"), n2m("C8"), 0.5) 
    """
    return m2f_np(np.arange(notemin, notemax+notedelta, notedelta))


def pianofreqs(start='A0', stop='C8') -> np.ndarray:
    """
    Generate an array of the frequencies representing all the piano keys
    """
    n0 = int(n2m(start))
    n1 = int(n2m(stop)) + 1
    return m2f_np(np.arange(n0, n1, 1))


def ratio2interval_np(ratios: np.ndarray) -> np.ndarray:
    """
    Vectorized version of r2i
    """
    out = np.log(ratios, 2)
    np.multiply(12, out, out=out)
    return out


def interval2ratio_np(intervals: np.ndarray) -> np.ndarray:
    """
    Vectorized version of i2r
    """
    out = intervals / 12.
    np.float_power(2, out, out=out)
    return out