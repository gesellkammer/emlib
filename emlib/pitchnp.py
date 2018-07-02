"""
Set of routinges to work with pitch

Routines ending with suffix _np accept np arrays


if peach is present, it is used for purely numeric conversions
(see github.com/gesellkammer/peach) 
"""

import numpy as np
from emlib import pitch as _pitch
from emlib.pitch import set_reference_freq

import sys
_EPS = sys.float_info.epsilon

    
def f2m_np(freqs, out=None):
    """
    vectorized version of f2m
    """
    A4 = _pitch.A4
    if out is None:
        out = freqs/A4
    else:
        np.multiply(freqs, 1.0/A4, out=out)
    # don't allow negative midi, and avoid divide by zero
    out.clip(min=9, out=out)   
    np.log2(out, out)
    out *= 12.0
    out += 69.0
    return out


def m2f_np(midinotes, out=None):
    A4 = _pitch.A4
    if out is None:
        out = midinotes - 69
    else:
        out = np.subtract(midinotes, 69, out=out)
    out /= 12.
    out = np.power(2.0, out, out)
    out *= A4
    return out


def db2amp_np(db, out=None):
    """
    convert dB to amplitude (0, 1)

    db: a np array of db values
    out: if None, the result will be put here
    """
    # amp = 10.0**(0.05*db)
    if out is None:
        out = 0.05 * db
    else:
        out = np.multiply(db, 0.05, out=out)
    out = np.power(10, out, out=out)
    return out


def amp2db_np(amp, out=None):
    # db = log10(amp)*20
    if out is None:
        X = np.maximum(amp, _EPS)
    else:
        X = np.maximum(amp, _EPS, out=out)
    X = np.log10(X, out=X)
    X *= 20
    return X


def logfreqs(notemin=0, notemax=139, notedelta=1.0):
    # type: (float, float, float) -> np.ndarray
    """
    Return a list of frequencies corresponding to the pitch range given

    notemin, notemax, notedelta: as used in arange (notemax is included)

    Examples:

    1) generate a list of frequencies of all audible semitones
    
    >>> logfreqs(0, 139, notedelta=1)

    2) Generate a list of frequencies of instrumental 1/4 tones

    >>> logfreqs(n2m("A0"), n2m("C8"), 0.5) 
    """
    return m2f_np(np.arange(notemin, notemax+notedelta, notedelta))


def pianofreqs(start='A0', stop='C8'):
    # type: (str, str) -> np.ndarray
    """
    Generate an array of the frequencies representing all the piano keys
    """
    n0 = int(_pitch.n2m(start))
    n1 = int(_pitch.n2m(stop)) + 1
    return m2f_np(np.arange(n0, n1, 1))


def ratio2interval_np(ratios):
    out = np.log(ratios, 2)
    np.multiply(12, out, out=out)
    return out


def interval2ratio_np(intervals):
    out = intervals / 12.
    np.float_power(2, out, out=out)
    return out


def pitchbend2cents(pitchbend, maxcents=200):
    return int(((pitchbend/16383.0)*(maxcents*2.0))-maxcents+0.5)


def cents2pitchbend(cents, maxcents=200):
    return int((cents+maxcents)/(maxcents*2.0)* 16383.0 + 0.5)