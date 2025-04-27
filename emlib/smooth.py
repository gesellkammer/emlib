"""

Routines for smoothing data

"""
from __future__ import annotations
import numpy
from emlib.iterlib import window
from typing import Iterator


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
    for v, w in zip(values, weights):
        ac_v += v * w
        ac_w += w
    return ac_v / ac_w


def movavg(values, wsize=5, start_value=None) -> Iterator[float]:
    """
    Moving average

    Args:
        values: iterator (does not need to be a full formed sequence)
        wsize: window size
        start_value: initial value, will use the first value of values if not provided

    Returns:
        iterator of moving averages
    """
    from collections import deque
    start_value = start_value if start_value is not None else values.next()
    data = deque([start_value] * wsize)
    datasum = sum(data)
    _append = data.append
    _pop = data.popleft
    yield datasum / wsize
    for i in values:
        _append(i)
        datasum += i - _pop()
        yield datasum / wsize


class MovingAverage:

    """
    Class for calculating a moving average efficiently

    Example
    ~~~~~~~

        smoother = MovingAverage(5)
        smoothdata = [smoother(i) for i in data]

    """

    def __init__(self, wsize=5, start_value=None):
        self.start_value = start_value
        self.wsize = int(wsize - 1 + wsize % 2)
        self._firstcall = True
        self._sum = 0

    def __call__(self, n):
        if self._firstcall:
            from collections import deque
            start_value = self.start_value if self.start_value is not None else n
            self.data = deque([start_value] * self.wsize)
            self._sum = start_value * self.wsize
            self._firstcall = False
        self.data.append(n)
        self._sum = sum = self._sum + n - self.data.popleft()
        return sum / self.wsize


def smooth(x, window_len=0.5, window='hanning', fixed=True):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Args:
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

    Returns:
        the smoothed signal

    Example
    ~~~~~~~

        t = linspace(-2,2,0.1)
        x = sin(t)+randn(len(t))*0.1
        y = smooth(x)

    .. seealso::

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter

    From: http://www.scipy.org/Cookbook/SignalSmooth
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
    """
    Class to unsmooth data
    """
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
    Weighted moving average

    Return an iterator to the sequence of moving averages of
    values weighted by weights with a window size = wsize
    the length of the returned iterator is len(values) - wsize + 1
    """
    grouped_values = window(values, wsize)
    grouped_weights = window(weights, wsize)
    for v, w in zip(grouped_values, grouped_weights):
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
