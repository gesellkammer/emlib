from __future__ import division as _division
from math import pi, sqrt


C = 343      # type: float
_PI2 = pi*2  # type: float


def resonfreq(V, A=pi * 0.01**2, L=0.05, c=C):
    # type: (float, float, float, float) -> float
    """
    resonance frequency of a helmholtz-resonator (in Hz)

    V: static volume of the cavity          (m^3)
    A: area of the neck                     (m^2)
    L: is the length of the neck            (m)
    c: speed of propagation                 (m/s)

    NB: magnitudes given in SI units (m, s, Hz)
    """
    a = sqrt(A / (L*pi))    # radius of the neck if the neck was a tube
    L1 = L + 1.7*a
    return c / _PI2 * sqrt(A/(V*L1))


def resonvolume(f, A=pi* 0.01**2, L=0.05, c=C):
    # type: (float, float, float, float) -> float
    """
    volume of a helmholz resonator of the given frequency (in m^3)
    the volume is given in m^3

    f: resonance frequency                  (Hz)
    A: area of the neck                     (m) 
       NB: circular area= PI * r^2. r = sqrt(A/PI)
    L: length of the neck                   (m)
    c: speed of propagation                 (m/s)   
    """
    a = sqrt(A / (L*pi))    # radius of the neck if the neck was a tube
    L1 = L + 1.7*a
    return A/((f*_PI2/c)**2 * L1)


def duct_unflanged_freq(L, radius, n, c=C):
    # type: (float, float, int, float) -> float
    """
    resonance frequency of an unflanged (open) duct

    L: length of the duct
    radius: radius of the duct
    n: 1,2,3. vibration mode

    from 'Acoustic Filters--David Russell.pdf'
    """
    f_n = n * c / (2*(L + 0.61*radius))
    return f_n


def duct_unflanged_length(freq, radius, n=1, c=C):
    # type: (float, float, int, float) -> float
    """
    calculate the length of the duct based on the observed resonant frequency
    
    freq   : resonant frequency measured
    radius : radius of the duct
    n      : harmonic number corresponding to the frequency observed
             (1 is the fundamental)
    c      : propagation speed of the medium

    Example
    -------

    # the most prominent resonance measured is 513 Hz, but in the spectrogram 
    # it is observed that this frequency corresponds to the second harmonic
    >>> duct_unflanged_length(513, 0.07, n=2)
    """
    L = n * c / (2*freq) - (0.61*radius)
    return L


def sphere_volume(r):
    # type: (float) -> float
    return 4/3. * pi * r**3


def sphere_radius(V):
    # type: (float) -> float
    return (V/(4/3.*pi)) ** (1/3.)
