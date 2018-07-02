"""
Frequency estimation of a signal with different algorithms
"""
from __future__ import division, print_function
from __future__ import absolute_import
import sys
import numpy as np
from scipy.signal import blackmanharris, fftconvolve
from typing import Tuple as Tup, Callable


def parabolic(f:np.ndarray, x:int) -> Tup[float, float]:
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f: a vector (an array of sampled values over a regular grid)
    x: index for that vector
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example
    =======

    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    >>> f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    >>> parabolic(f, 3)
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return xv, yv


def find(condition) -> np.ndarray:
    """Return the indices where ravel(condition) is true"""
    res, = np.nonzero(np.ravel(condition))
    return res


def freq_from_crossings(sig:np.ndarray, sr:int):
    """Estimate frequency by counting zero crossings
    
    sig: a sampled signal
    sr : sample rate

    Returns -> the frequency of the signal
    """
    # Find all indices right before a rising-edge zero crossing
    indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
    
    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices
    
    # More accurate, using linear interpolation to find intersample 
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
    
    # Some other interpolation based on neighboring points might be better. 
    # Spline, cubic, whatever
    return sr / np.mean(np.diff(crossings))


def freq_from_fft(sig:np.ndarray, sr:int) -> float:
    """Estimate frequency from peak of FFT

    sig: a sampled signal
    sr : sample rate

    Returns -> the frequency of the signal
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = np.fft.rfft(windowed)
    
    # Find the peak and interpolate to get a more accurate peak
    i = np.argmax(abs(f))    # Just use this for less-accurate, naive version
    true_i = parabolic(np.log(abs(f)), i)[0]
    
    # Convert to equivalent frequency
    return sr * true_i / len(windowed)


def freq_from_autocorr(sig:np.ndarray, sr:int) -> float:
    """
    Estimate frequency using autocorrelation
    
    sig: a sampled signal
    sr : sample rate

    Returns -> the frequency of the signal
    """
    # Calculate autocorrelation (same thing as convolution, but with 
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)/2:]
    
    # Find the first low point
    d = np.diff(corr)
    start = find(d > 0)[0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is 
    # not reliable for long signals, due to the desired peak occurring between 
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak) 
    return sr / px


def freq_from_HPS(sig, sr, maxharms=5):
    """
    Estimate frequency using harmonic product spectrum (HPS)
    
    """
    windowed = sig * blackmanharris(len(sig))
    c = abs(np.fft.rfft(windowed))
    freq = 0
    for x in range(2,maxharms):
        a = c[::x]  # Should average or maximum instead of decimating
        # a = max(c[::x],c[1::x],c[2::x])
        c = c[:len(a)]
        i = np.argmax(abs(c))
        try:
            true_i = parabolic(abs(c), i)[0]
        except IndexError:
            return freq
        freq = sr * true_i / len(windowed)
        print('Pass %d: %f Hz' % (x, freq))
        c *= a
    return freq


if __name__ == '__main__':
    filename = sys.argv[1]
    
    print('Reading file "%s"\n' % filename)
    import sndfileio
    signal, sr = sndfileio.sndread(filename)

    print('Calculating freq from FFT: %f Hz' % freq_from_fft(signal, sr))
    print('Calculating freq from zerocross: %f Hz: ' % freq_from_crossings(signal, sr))
    print('Calculating freq from autocorrelation: %f Hz' % freq_from_autocorr(signal, sr))
    print('Calculating freq from HPS: %f Hz' % freq_from_HPS(signal, sr))
