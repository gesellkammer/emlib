from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from scipy import signal
import numpy as np


def plot_power_spectrum(samples, samplerate, framesize=2048, window=('kaiser', 9)):
    """
    window: As passed to scipy.signal.get_window
        * `blackman`, `hamming`, `hann`, `bartlett`, `flattop`, `parzen`, `bohman`, 
          `blackmanharris`, `nuttall`, `barthann`, `kaiser` (needs beta), 
          `gaussian` (needs standard deviation)


    """
    w = signal.get_window(window, framesize)
    
    def func(s):
        return s * w
    
    import matplotlib.pyplot as plt
    return plt.psd(samples, framesize, samplerate, window=func)


def get_channel(samples, channel):
    if len(samples.shape) == 1:
        return samples
    return samples[:,channel]


def get_num_channels(samples):
    if len(samples.shape) == 1:
        return 1
    return samples.shape[1]


def _plot_samples_matplotlib(samples, samplerate, subsampling=1):
    import matplotlib.pyplot as plt
    if subsampling:
        samples = samples[::subsampling]
        samplerate = int(samplerate / subsampling)
    numch = get_num_channels(samples)
    t = np.linspace(0, len(samples)/samplerate, num=len(samples))
    for ch in range(numch):
        ax = plt.subplot(numch, 1, ch+1)
        data = get_channel(samples, ch)
        ax.plot(t, data)
    ax.set_xlabel("Time (s)")
    return True


def _plot_samples_pyqtgraph(samples, samplerate, subsampling=1):
    # TODO
    return False


def plot_samples(samples, samplerate, subsampling=1):
    backends = [
        ('pyqtgraph', _plot_samples_pyqtgraph),
        ('matplotlib', _plot_samples_matplotlib),
    ]
    for backend, func in backends:
        ok = func(samples, samplerate, subsampling)
        if ok:
            break
