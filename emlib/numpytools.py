import numpy as np
from numpy.lib.stride_tricks import as_strided


def npzip(*arrays):
    """
    zip 1-D arrays, similar to the built-in zip

    this is the same as np.column_stack but seems to be significantly faster
    all arrays should be the same shape

    To unzip, use:

    column0, column1 = a.transpose()
    """
    return np.concatenate(arrays).reshape(len(arrays), len(arrays[0])).transpose()


def npunzip(a):
    """
    column0, column1, ... = a.transpose()
    """
    return a.transpose()


def zipsort(a, b):
    """
    equivalent to

    a, b = unzip(sorted(zip(a, b)))

    If a and b are two columns of data, sort a keeping b in sync
    """
    indices = a.argsort()
    return a[indices], b[indices]

    
def smooth(a, kind="running", strength=0.05):
    """
    kind: "running" --> running mean 
          "gauss"   --> 1-D gaussian blur (not impl.)

    strength (0-1): how much smoothing to apply.
    """
    assert len(a) > 3
    if kind == "running":
        if strength is None:
            N = len(a)
        else:
            N = min(len(a) * 0.5 * strength, 3)
        K = np.ones(N, dtype=float) / N
        a_smooth = np.convolve(a, K, mode='same')
    else:
        raise NotImplementedError
    return a_smooth


def overlapping_frames(y, frame_length, hop_length):
    """
    Slice a time series into overlapping frames. Mainly
    used for dsp

    y: np.ndarray [shape=(n,)]
        Time series to frame, Must be one-dimensional
        and contiguous in memory

    frame_length: int > 0. 
        Length of the frame in samples
    
    hop_length: int > 0
        Number of samples to hop between frames

    Returns
    ~~~~~~~

    y_frames: np.ndarray [shape=(frame_length, N_FRAMES)]

    Examples

    Extract 2048-sample frames from `y` with a hop of 64 samples
    per frame

    samples, sr = sndread("monofile.wav")
    overlapping_frames(samples, frame_length=2048, hop_length=64)

    taken from librosa.util.frame
    """
    if not isinstance(y, np.ndarray):
        raise TypeError('Input must be of type np.ndarray, '
                        f'given type(y)={type(y)}')

    if y.ndim != 1:
        raise ValueError('Input must be one-dimensional, '
                         f'given y.ndim={y.ndim}')

    if len(y) < frame_length:
        raise ValueError(f'Buffer is too short (n={len(y)})'
                         f' for frame_length={frame_length}')

    if hop_length < 1:
        raise ValueError(f'Invalid hop_length: {hop_length}')

    if not y.flags['C_CONTIGUOUS']:
        raise ValueError('Input buffer must be contiguous.')

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames
