"""
Miscellaneous utilities for working with numpy arrays
"""
from __future__ import annotations
import numpy as np
from numpy.lib.stride_tricks import as_strided


def interlace(*arrays: np.ndarray) -> np.ndarray:
    """
    Interweave multiple arrays into a flat array in the form

    Example::

        A = [a0, a1, a2, ...]
        B = [b0, b1, b2, ...]
        C = [c0, c1, c2, ...]
        interlace(A, B, C)
        -> [a0, b0, c0, a1, b1, c1, ...]

    Args:
        *arrays: the arrays to interleave. They should be 1D arrays of the
            same length

    Returns:
        a 1D array with the elements of the given arrays interleaved

    """
    assert all(a.size == arrays[0].size and a.dtype == arrays[0].dtype for a in arrays)
    size = arrays[0].size * len(arrays)
    out = np.empty((size,), dtype=arrays[0].dtype)
    for i, a in enumerate(arrays):
        out[i::len(arrays)] = a
    return out


def npzip(*arrays: np.ndarray) -> np.ndarray:
    """
    zip 1-D arrays, similar to the built-in zip

    This is the same as np.column_stack but seems to be significantly faster
    all arrays should be the same shape

    To unzip, use::

        column0, column1 = a.transpose()
    """
    return np.concatenate(arrays).reshape(len(arrays), len(arrays[0])).transpose()


def npunzip(a: np.ndarray) -> np.ndarray:
    """
    column0, column1, ... = a.transpose()
    """
    return a.transpose()


def zipsort(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort one array, keep the other synched

    Equivalent to::

        a, b = unzip(sorted(zip(a, b)))

    If a and b are two columns of data, sort a keeping b in sync
    """
    indices = a.argsort()
    return a[indices], b[indices]

    
def smooth(a: np.ndarray, kind="running", strength=0.05) -> np.ndarray:
    """
    Smooth the values in a

    Args:
        a: the array to smoothen
        kind: the procedure used. One of "running", "gauss"
        strength: how strong should the smoothing be?

    Returns:
        the resulting array

    """
    assert len(a) > 3
    if kind == "running":
        N = len(a) if strength is None else min(len(a) * 0.5 * strength, 3)
        K = np.ones(N, dtype=float) / N
        a_smooth = np.convolve(a, K, mode='same')
    else:
        raise ValueError(f"{kind} is not a valid kind. Valid options: 'running' ")
    return a_smooth


def overlapping_frames(y: np.ndarray, frame_length: int, hop_length: int):
    """
    Slice a time series into overlapping frames.

    Args:
        y: np.ndarray - Time series to frame, Must be one-dimensional and
            contiguous in memory
        frame_length: int - Length of the frame in samples
        hop_length: int - Number of samples to hop between frames

    Returns:
        the frames, a np.ndarray of shape=(frame_length, N_FRAMES)

    Examples
    --------

        # Extract 2048-sample frames from `y` with a hop of 64 samples
        # per frame
        >>> samples, sr = sndread("monofile.wav")
        >>> overlapping_frames(samples, frame_length=2048, hop_length=64)

    **NB**: Taken from librosa.util.frame
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

    Returns:
        a generator with chunks of data of chunksize or less
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

    Pad 1D with 2 elements::

        1 2 3 4   -> 1 2 3 4 0 0

    Pad 2D with 2 elements::

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


def astype(a: np.ndarray, typedescr):
    """
    The same as: `if a.dtype != typedescr: a = as.astype(typedescr)`
    
    """
    return a if a.dtype == typedescr else a.astype(typedescr)


def _nearestlr(items: np.ndarray, seq: np.ndarray) -> np.ndarray:
    irs = np.searchsorted(seq, items, 'left')
    np.clip(irs, 0, len(seq) - 1, out=irs)
    ils = irs - 1
    rdiff = np.abs(seq[irs] - items)
    ldiff = np.abs(seq[ils] - items)
    out = np.choose(rdiff < ldiff, [ils, irs])
    return out


def _nearestl(items: np.ndarray, seq: np.ndarray) -> np.ndarray:
    idxs = np.searchsorted(seq, items, 'right')
    idxs -= 1
    if np.any(idxs < 0):
        raise ValueError("No values to the left!")
    return idxs


def _nearestr(items: np.ndarray, seq: np.ndarray) -> np.ndarray:
    idxs = np.searchsorted(seq, items, 'left')
    if np.any(idxs >= len(seq)):
        raise ValueError("No values to the right!")
    return idxs


def nearestindex(a: np.ndarray, grid: np.ndarray, left=True, right=True
                  ) -> np.ndarray:
    """
    For each value in `a` return the index into `grid` nearest to it

    To get the nearest element, do::

        indexes = nearest_index(a, grid)
        nearest_elements = grid[indexes]

    Args:
        a: events to match from. Does not need to be sorted
        grid: events to match against. Does not need to be sorted
        left: match events lower than the event from
        right: match events higher than the event from
    """
    if left and right:
        return _nearestlr(a, grid)
    elif left:
        return _nearestl(a, grid)
    elif right:
        return _nearestr(a, grid)
    else:
        raise ValueError("At least left or right must be true")

