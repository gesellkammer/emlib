from __future__ import absolute_import
from __future__ import print_function
import numpy, ctypes


def npzip(*arrays):
    """
    zip 1-D arrays, similar to the built-in zip

    this is the same as numpy.column_stack but seems to be significantly faster
    all arrays should be the same shape

    To unzip, use:

    column0, column1 = a.transpose()
    """
    return numpy.concatenate(arrays).reshape(len(arrays), len(arrays[0])).transpose()


def npunzip(a):
    """
    column0, column1, ... = a.transpose()
    """
    return a.transpose()

    
_buffrommem_rw = ctypes.pythonapi.PyBuffer_FromReadWriteMemory
_buffrommem_rw.restype = ctypes.py_object


def buffer_from_memory(int_pointer, dtype, size):
    """
    given the address of an array as an integer, create a buffer to this unicodedata

    NB: to convert it to a numpy array, use numpy.frombuffer

    int_pointer: the address of the pointer, as integer
    dtype: the type of the memory being referenced (float, int, etc)
    size: the size of the array
    
    The pointer could come from, for instance, a SWIG DoubleVector.
    This buffer can then be used with numpy.frombuffer to create a numpy array
    aliasing the memory of the DoubleVector, so that you can read and write
    to it as if it were a numpy array. It is only important to grab a reference
    to the original array which owns the memory
    
    >>> import loris, ctypes
    >>> vector = loris.DoubleVector(10)
    >>> buffer = buffer_from_memory(int(vector.this), float, 10)
    >>> a = numpy.frombuffer(buffer, float)
    >>> a[0] = 5
    >>> vector[0]
    5
    """
    try:
        datasize = numpy.dtype(dtype).itemsize
    except TypeError:
        raise TypeError("dtype not understood. It must be the type of the original array")
    pointer_ctype = {
        float: ctypes.c_double
    }.get(dtype)
    if pointer_ctype is None:
        raise TypeError("this type is not supported")
    pointer_factory = ctypes.POINTER(pointer_ctype)
    pointer = pointer_factory.from_address(int_pointer)
    buf = _buffrommem_rw(pointer, size * datasize)
    return buf
    

def numpy_array_from_swig_vector(vector, dtype):
    """
    NB: the original vector still owns the memory, so this array
        will be valid as long as the vector is alive
    """
    size = vector.size()
    buf = buffer_from_memory(int(vector.this), dtype, size)
    arr = numpy.frombuffer(buf, dtype)
    return arr


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
        K = numpy.ones(N, dtype=float) / N
        a_smooth = numpy.convolve(a, K, mode='same')
    else:
        raise NotImplementedError
    return a_smooth
