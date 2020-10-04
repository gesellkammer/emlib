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
        N = len(a) if strength is None else min(len(a) * 0.5 * strength, 3)
        K = np.ones(N, dtype=float) / N
        a_smooth = np.convolve(a, K, mode='same')
    else:
        raise ValueError(f"{kind} is not a valid kind. Valid options: 'running' ")
    return a_smooth


def overlapping_frames(y, frame_length, hop_length):
    """
    Slice a time series into overlapping frames.

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


def chunks(data: np.ndarray, chunksize: int, hop:int=None, padwith=0):
    """
    Iterate over data in chunks of chunksize. Returns a generator

    Args:
        data: the array to be iterated in chunks
        chunksize: the size of each chunk
        hop: the amount of elements to skip between chunks
        padwidth: value to pad when a chunk is not big enough
            Give None to avoid padding

    """
    numframes = len(data)
    if hop is None:
        hop = chunksize
    n = 0
    if padwith is None:
        while n < numframes:
            chunk = data[n:n+chunksize]
            yield chunk
            n += hop
    else:
        while n < numframes:
            chunk = data[n:n+chunksize]
            lenchunk = len(chunk)
            if lenchunk < chunksize:
                chunk = padarray(chunk, chunksize - lenchunk, padwith)
                yield chunk
                break
            yield chunk
            n += hop


def padarray(arr, numelements, padwith=0):
    """
    Pad a 1D array to the right with 0s, or a 2D array down with zeros

    Pad 1D with 2 elements

    1 2 3 4   -> 1 2 3 4 0 0

    Pad 2D with 2 elements

    0   1  2      0  1  2
    10 11 12  -> 10 11 12
    20 21 22     20 21 22
                  0  0  0
                  0  0  0
    """
    numdims = len(arr.shape)
    if numdims == 1:
        return np.pad(arr, (0, numelements), mode='constant', constant_values=padwith)
    elif numdims == 2:
        return np.pad(arr, [(0, numelements), (0, 0)], mode='constant', constant_values=padwith)
    else:
        raise ValueError("Only 1D or 2D arrays supported")


def linlin(xs:np.ndarray, x0:float, x1:float, y0:float, y1: float) -> np.ndarray:
    """
    Map xs from range x0-x1 to y0-y1

    Args:
        xs: the array of values between x0 and x1
        x0: the min. value of xs
        x1: the max. value of xs
        y0: the min. value of the remapped array
        y1: the max. value of the remapped array

    Returns:
        the remapped values
    """
    # (xs - x0) / (x1-x0) * (y1-y0) + y0
    xs = xs - x0
    xs /= x1 - x0
    xs *= y1 - y0
    xs += y0
    return xs