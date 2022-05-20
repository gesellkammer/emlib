"""

Routines for smoothing data

"""
from __future__ import annotations
import numpy
from collections import deque
from emlib.misc import isiterable
from emlib.iterlib import window, izip


def wavg(values, weights):
    """
    weighted average

    Args:
        values, weights: iterators (does not need to be a full formed sequence)

    Returns:
        dotproduct(values, weights) / sum(weights)
    """
    ac_v = 0
    ac_w = 0
    for v, w in izip(values, weights):
        ac_v += v * w
        ac_w += w
    return ac_v / ac_w


def windowed_moving_average(sequence, windowSize, weights=1, complete=True):
    """
    return the weighted moving average of sequence with a
    window size = windowSize.
    weights can be a tuple defining the weighting in the window
    for instance if windowSize = 5, weights could be (0.1, 0.5, 1, 0.5, 0.1)
    if only one number is defined, a tupled is generated following a 
    interpol.halfcos curve
    """
    import interpoltools
    import numpy

    def parse_coefs(n, coef):
        # Asume que coef es un numero , definiendo una curva
        n = n - 1 + n % 2
        if n == 3:
            return numpy.array((1., coef, 1.), 'd')
        middle = int(n / 2)
        coefs = []
        for i in range(middle):
            coefs.append(interpoltools.interpol_halfcos(i, 0, 1, middle, coef))
        for i in range(middle, n):
            coefs.append(interpoltools.interpol_halfcos(i, middle, coef, n - 1, 1))
        return numpy.array(coefs)
    l = len(sequence)
    if not isiterable(weights):
        weights = parse_coefs(windowSize, weights)
    seq = numpy.array([sequence[i:l - (windowSize - i) + 1]
                      for i in range(windowSize)], 'd').transpose()
    return (seq * weights).sum(1) / sum(weights)


class MovingAverage:

    """
    efficient class for calculating a moving average
    it can be used as a function or as a generator
    usage:
    smoother = MovingAverage(5)
    smoothed_data = [smoother(i) for i in data]

    or

    smoothed_data = (x for x in smoother(data))

    but it cannot be mixed. Once used as one of it,
    it specializes itself.

    """

    def __init__(self, wsize=5, start_value=None):
        self.start_value = start_value
        self.wsize = int(wsize - 1 + wsize % 2)  # always odd

        # self.__call__ = self.__first_call__

    def __call__(self, n):
        try:
            return self._as_generator(iter(n))
        except:
            return self.__first_call__(n)

    def __first_call__(self, n):
        start_value = self.start_value if self.start_value is not None else n
        self.data = deque([start_value] * self.wsize)
        self._data_append = self.data.append
        self._data_popleft = self.data.popleft
        self.__call__ = self.__next_calls__
        return self(n)
        # return sum(self.data) / self.wsize

    def __next_calls__(self, n):
        self._data_append(n)
        self._data_popleft()
        return sum(self.data) / self.wsize

    def _as_generator(self, seq):
        start_value = self.start_value if self.start_value is not None else seq.next()
        wsize = self.wsize
        data = deque([start_value] * wsize)
        _data_append = data.append
        _data_popleft = data.popleft
        yield sum(data) / wsize
        for i in seq:
            _data_append(i)
            _data_popleft()
            yield sum(data) / wsize


def smooth(x, window_len=0.5, window='hanning', fixed=True):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal (a sequence)
        window_len: the dimension of the smoothing window
            if it is a float, it is indicates as a fraction of the length of the
            data sequence (0.5 = 0.5 * len(x)). The result must be always smaller
            than the size of the data array. 1 will be interpreted as a window size of 1!
            (if it were 1 * len(x), then the window would not be smaller than 
            the data array)
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 
                'blackman'
            flat window will produce a moving average smoothing.
        fixed: if True, the beginning and the end match the beginning and end 
               of the original

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    from: http://www.scipy.org/Cookbook/SignalSmooth
    """
    if fixed:
        return _smooth_fixed(x, window_len, window)
    x = numpy.asarray(x)
    if isinstance(window_len, float):
        window_len = int(x.size * window_len)

    if isinstance(window, str):
        if window == 'flat':  # moving average
            window = numpy.ones(window_len,'d')
        else:
            window = eval('numpy.' + window + '(window_len)')
    # if not, assume the passed window is already an array
    # assert isinstance(w, numpy.ndarray)
    else:
        window_len = len(window)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    s = numpy.r_[2 * x[0] - x[window_len:1:-1],x,2 * x[-1] - x[-1:-window_len:-1]]
    y = numpy.convolve(window / window.sum(),s,mode='same')
    out = y[window_len - 1:-window_len + 1]
    assert out.size == x.size
    return out


def _smooth_fixed(x, window_len, window):
    x = numpy.asarray(x)
    z0 = smooth(x[::-1], window_len, window, fixed=False)[::-1]
    z1 = smooth(x, window_len, window, fixed=False)
    return (z0 + z1) / 2


def unsmooth(x, ratio, smooth_ratio=0.5):
    return Unsmoother(x, smooth(x, smooth_ratio))(ratio)


class Unsmoother:
    # TODO: implementar otro tipo de interpolacion que no sea lineal

    def __init__(self, x, smoothed_x):
        self.x = x
        self.smoothed_x = smoothed_x

    def __call__(self, ratio):
        """
        1 will give you the original data
        0 will give you the smoothed data
        any other number will interpolate / extrapolate between these two
        """
        if ratio == 0:
            return self.smoothed_x
        elif ratio == 1:
            return self.x
        else:
            return self.x * ratio + self.smoothed_x * (1 - ratio)


def weighted_moving_average(values, weights, wsize):
    """
    return an iterator to the sequence of moving averages of
    values weighted by weights with a window size = wsize
    the length of the returned iterator is len(values) - wsize + 1
    """
    grouped_values = window(values, wsize)
    grouped_weights = window(weights, wsize)
    for v, w in izip(grouped_values, grouped_weights):
        yield wavg(v, w)


def rm3(s):
    temp = map(lambda x,y,z:[x,y,z], s[:-2], s[1:-1], s[2:])
    temp2 = []
    temp2_append = temp2.append
    for x in temp:
        x.sort()
        temp2_append(x[1])
    return temp2


def rm5(s):
    temp = map(lambda a,b,c,d,e: 
               [a,b,c,d,e], s[0:-4], s[1:-3], s[2:-2], s[3:-1], s[4:])
    temp2 = []
    for x in temp:
        x.sort()
        temp2.append(x[1])
    return temp2


if __name__ == '__main__':
    import doctest
    doctest.testmod()
