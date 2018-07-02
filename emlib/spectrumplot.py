from matplotlib import pyplot as plt
import bpf4 as bpf
from .pitch import amp2db_np
# import numpy as np


def plot(partials):
    return plot_pyplot(partials)


def plot_pyplot(partials):
    fig, ax = plt.subplots(1, 1)
    colorcurve = bpf.linear(-90, 0, -30, 0.3, 0, 1)
        
    for partial in partials:
        X = partial.times
        Y = partial.freqs
        amps = partial.amps
        dbs = amp2db_np(amps)
        C = colorcurve.map(dbs)
        ax.plot(X, Y, color=(0.5, 0.5, 0.5, 0.5))
        ax.scatter(X, Y, s=16, c=C, linewidths=0, cmap=plt.cm.Greys, alpha=0.5)
    return ax


def plot_partial_pyplot(partial, ax):
    # ax.grey()
    raise NotImplementedError()
