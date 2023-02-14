"""
More itertools
"""
from __future__ import annotations

import sys
from itertools import *
import operator as _operator
import collections as _collections
import random as _random
from math import inf
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Iterator, TypeVar, Callable, Sequence
    T = TypeVar("T")
    T2 = TypeVar("T2")

# ----------------------------------------------------------------------------
# 
#    protocol: seq, other parameters
#
#    Exception: functions that match the map protocol: func(predicate, seq)
#
# ----------------------------------------------------------------------------


def take(seq: Iterable[T], n:int) -> list[T]:
    """returns the first n elements of seq as a list"""
    return list(islice(seq, n))


def last(seq: Iterable[T]) -> T | None:
    """
    Returns the last element of seq or None if seq is empty

    if *seq* is an iterator, it will consume it
    """
    if isinstance(seq, _collections.Sequence):
        try:
            return seq[-1]
        except IndexError:
            return None
    else:
        x = None
        for x in seq:
            pass
        return x


def first(it: Iterable[T], default=None) -> T | None:
    return next(it, default)


def consume(iterator, n: int) -> None:
    """Advance the iterator n-steps ahead. If n is none, consume entirely.

    .. note::

        The only reason to consume an iterator is if it this has some
        sort of side-effect
    """
    if n is None:
        # feed the entire iterator into a zero-length deque
        _collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

        
def drop(seq: Iterable[T], n: int) -> Iterable[T]:
    """
    return an iterator over seq with n elements consumed

    Example
    -------

        >>> list(drop(range(10), 3))
        [3, 4, 5, 6, 7, 8, 9]
    """
    return islice(seq, n, None)


def nth(seq: Iterable[T], n: int) -> T:
    """Returns the nth item"""
    return next(islice(seq, n, n+1))


def pad(seq: Iterable[T], element:T2=None) -> Iterator[T | T2]:
    """
    Returns the elements in *seq* and then return *element* indefinitely.

    Useful for emulating the behavior of the built-in map() function.
    
    >>> take(pad(range(10), "X"), 15)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'X', 'X', 'X', 'X', 'X']
    """
    return chain(seq, repeat(element))


def ncycles(seq: Iterable[T], n: int) -> Iterable[T]:
    """Returns the sequence elements n times"""
    return chain(*repeat(seq, n))


class Accum:
    """
    Simple accumulator

    Example
    ~~~~~~~

    Iterate over seq until the sum of the items exceeds a given value

        >>> seq = range(999999)
        >>> takewhile((lambda item, accum=Accum(): accum(item) < 100), seq)
    """

    def __init__(self, init:T=0):
        self.value:T = init

    def __call__(self, value:T) -> T:
        self.value += value
        return self.value


def dotproduct(vec1: Iterable[T], vec2: Iterable[T]) -> T:
    """
    Returns the dot product (the sum of the product between each pair)
    between vec1 and vec2

    Args:
        vec1: a seq. of T
        vec2: a seq. of T

    Returns:
        the sum of the product of vec1_n * vec2_n for each n

    """
    return sum(map(_operator.mul, vec1, vec2))


def repeatfunc(func, times=None, *args):
    """
    Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    else:
        return starmap(func, repeat(args, times))


def pairwise(iterable: Iterable[T]) -> zip[tuple[T, T]]:
    """
    Similar to window(seq, size=2, step=1)
    
    Example
    ~~~~~~~

        >>> list(pairwise(range(4)))
        [(0, 1), (1, 2), (2, 3)]
    """    
    a, b = tee(iterable)
    try:
        next(b)
    except StopIteration:
        pass
    return zip(a, b)


def window(iterable: Iterable[T], size=3, step=1) -> Iterator[tuple[T, ...]]:
    """
    iterate over subseqs of iterable

    Args:
        iterable: an iterable
        size: the size of the window
        step: the step size

    Returns:
        an iterator over a tuple of items of iterable, windowed as indicated

    Example
    ~~~~~~~
    
    >>> seq = range(6)
    >>> list(window(seq, 3, 1))
    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    >>> list(window(seq, 3, 2))
    [(0, 1, 2), (2, 3, 4)]
    # the same as pairwise
    >>> list(window(range(5), 2, 1))
    [(0, 1), (1, 2), (2, 3), (3, 4)]
    """
    iterators = tee(iterable, size)
    for skip_steps, itr in enumerate(iterators):
        for _ in islice(itr, skip_steps):
            pass
    winiter = zip(*iterators)
    if step != 1:
        winiter = islice(winiter, 0, 99999999, step)
    return winiter


def window_fixed_size(seq: Iterable[T], size: int, maxstep: int
                           ) -> Iterable[tuple[T, ...]]:
    """
    A sliding window over subseqs of seq

    Each returned subseq has the given size and the step between
    each subseq is at most 'maxstep'. If the last window does not
    fit evenly, a smaller step is taken.
    If seq has less elements than `size`, a ValueError is raised

    .. note::

        The difference with `window` is that `window` drops the last
        elements if they don't fit evenly in the window size

    Example
    =======

    >>> list(window_fixed_size(range(10)), 5,2)
    [[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]
    # Notice that the step of the last window is 1 and not 2

    >>> list(window(range(10), 5, 2))
    [(0, 1, 2, 3, 4), (2, 3, 4, 5, 6), (4, 5, 6, 7, 8)]
    # element 9 is missing
    """
    cursor = 0
    try:
        seqlen = len(seq)
    except TypeError:
        seq = list(seq)
        seqlen = len(seq)
    if seqlen < size:
        raise ValueError("seq too small")
    while True:
        yield seq[cursor:cursor+size]
        step = min(maxstep, seqlen-size-cursor)
        if step == 0:
            break
        cursor += step



def iterchunks(seq, chunksize: int) -> Iterable[Tuple]:
    """
    Returns an iterator over chunks of seq of at most `chunksize` size.

    If seq is finite and not divisible by chunksize, the last chunk will
    have less than `chunksize` elements.

    Example
    ~~~~~~~

    >>> seq = range(20)
    >>> list(iterchunks(seq, 3))
    [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14), (15, 16, 17), (18, 19)]
    """
    class Sentinel(object):
        pass
    padded = chain(seq, repeat(Sentinel))
    for chunk in window(padded, size=chunksize, step=chunksize):
        if chunk[-1] is Sentinel:
            if chunk[0] is Sentinel:
                break
            yield tuple((x for x in chunk if x is not Sentinel))
            break
        else:
            yield chunk


def parse_range(start, stop:int=None, step:int=None) -> tuple[int, int, int]:
    """
    Given arguments as passed to `range`, resolved them in `(start, stop, step)`
    """
    if stop is None:
        stop = int(start)
        start = 0
    if step is None:
        step = 1
    return start, stop, step


def chunks(start:int, stop:int=None, step:int=None) -> Iterable[tuple[int, int]]:
    """
    Returns a generator of tuples (offset, chunksize)

    Example
    ~~~~~~~

    >>> list(chunks(0, 10, 3))
    [(0, 3), (3, 3), (6, 3), (9, 1)]

    """
    start, stop, step = parse_range(start, stop, step)
    while start < stop:
        size = min(stop - start, step)
        yield start, size
        start += step


def isiterable(obj, exclude=(str,)) -> bool:
    """Returns True if obj is iterable"""
    if exclude:
        return hasattr(obj, '__iter__') and (not isinstance(obj, exclude))
    return hasattr(obj, '__iter__')
    

def random_combination(iterable: Iterable, r):
    """Random selection from itertools.combinations(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(_random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_permutation(iterable, r=None):
    """Random selection from itertools.permutations(iterable, r)"""
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(_random.sample(pool, r))


def partialsum(seq: Iterable[T], start=0) -> Iterable[T]:
    """
    for each elem in seq return the partial sum

    .. code::

        n0 -> n0
        n1 -> n0 + n1
        n2 -> n0 + n1 + n2
        n3 -> n0 + n1 + n2 + n3
    """
    accum = start
    for i in seq:
        accum += i
        yield accum

def partialmul(seq: Iterable[T], start=1) -> Iterable[T]:
    """
    return the accumulated multiplication

    .. code::

        n0 -> n0
        n1 -> n1 * n0
        n2 -> n2 * n1 * n0
    """
    assert isiterable(seq)
    accum = start
    for i in seq:
        accum *= i
        yield accum

        
def partialavg(seq: Iterable[T]) -> T:
    """
    Return the partial average in seq
    """
    accum = 0
    i = 0
    for n in seq:
        accum += n
        i += 1
        yield accum / i


def ilen(seq: Iterable) -> int:
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = count()
    _collections.deque(zip(seq, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def avg(seq: Iterable[T], empty=0) -> T:
    """
    Return the average of seq, or `empty` if the seq. is empty

    .. note::

        If you know the size of seq, it is faster to do `sum(seq) / len_of_seq`

    Example
    ~~~~~~~

    >>> avg(range(1_000_000))
    499999.5
    """
    accum, i = 0, 0
    for x in seq:
        accum += x
        i += 1
    return accum / i if i else empty

        
def flatten(s: Iterable[T | Iterable[T]], exclude=(str,), levels=inf) -> Iterator[T]:
    """
    Return an iterator to the flattened items of sequence s

    Args:
        s: the sequence to flatten
        exclude: classes to exclude from flattening
        levels: how many levels to flatten

    Returns:
        the flattened version of *s*

    Example
    -------

    >>> seq = [1, [2, 3], [4, [5, 6]]]
    >>> list(flatten(seq))
    [1, 2, 3, 4, 5, 6]

    """
    for item in s:
        if isinstance(item, exclude or levels <= 0) or not hasattr(item, '__iter__'):
            yield item
        else:
            yield from flattened(item, exclude, levels-1)
    

def flatdict(d: dict) -> list:
    """
    Given a dictionary, return a flat list where keys and values are interleaved.

    This is similar to doing `flattened(d.items())`

    Example
    -------

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> flatdict(d)
    ['a', 1, 'b', 2, 'c', 3]
    """
    ks = d.keys()
    vs = d.values()
    out = [0] * (len(d)*2)
    out[::2] = ks
    out[1::2] = vs
    return out


def flattened(s: Iterable[T | Iterable[T]], exclude=(str,), levels: int = None, out: list=None
              ) -> list[T]:
    """
    Like flatten, but returns a list instead of an iterator

    Args:
        s: the seq to flatten.
        exclude: types to exclude
        levels: how many levels to flatten (None: flatten all levels)
        out: if given, the flattened result is appended to this list

    Returns:
        a list with the elements in *s* flattened
    """
    if levels is None:
        levels = sys.maxsize
    if out is None:
        out = []
    _flattened2(s, out, exclude, levels=levels)
    return out
    

def _flattened2(s, out: list, exclude, levels: int = None) -> None:
    for item in s:
        if isinstance(item, exclude or levels <= 0) or not hasattr(item, '__iter__'):
            out.append(item)
        else:
            _flattened2(item, out, exclude, levels-1)


def flattenonly(l, types):
    """
    Flatten only if subsequences are of type 'type' 

    Args:
        l: a seq
        type: a type or a tuple of types, as passed to isinstance

    Example
    -------

    flatten only list items, pass tuples untouched

    >>> l = [1, [2, 3], (4, 5)]
    >>> list(flattenonyl(l, list))
    [1, 2, 3, (4, 5)]

    """
    for elem in l:
        if isinstance(elem, types):
            yield from flattenonly(elem, types)
        else:
            yield elem

            
def zipflat(*seqs):
    """
    like izip but flat. It has the same effect as `flatten(zip(seqA, seqB), levels=1)`
    
    Example
    -------
    
    >>> list(zipflat(range(5), "ABCDE"))
    [0, 'A', 1, 'B', 2, 'C', 3, 'D', 4, 'E']
    """
    for elems in zip(*seqs):
        for elem in elems:
            yield elem

            
def butlast(seq: Iterable[T]) -> Iterator[T]:
    """
    iterate over seq[:-1]
    
    >>> list(butlast(range(5)))
    [0, 1, 2, 3]
    """
    seq_iter = iter(seq)
    lastitem = next(seq_iter)
    for i in seq_iter:
        yield lastitem
        lastitem = i

        
def butn(seq: Iterable[T], n: int) -> Iterable[T]:
    """
    iterate over seq[:-n]
    """
    seqi = iter(seq)
    # d = _collections.deque((next(seqi) for i in xrange(n)), n)
    d = _collections.deque((next(seqi) for _ in range(n)), n)
    for x in seqi:
        yield d.popleft()
        d.append(x)

        
def intercalate(seq: Iterable[T], item:T2) -> Iterator[T|T2]:
    """
    Intercalate *item* between elements of *seq*

    Example
    -------

    >>> list(intercalate(range(5), "X"))
    [0, 'X', 1, 'X', 2, 'X', 3, 'X', 4]
    """
    return butlast(zipflat(seq, repeat(item)))


def partialreduce(seq: Iterable[T], func: Callable, start=0) -> Iterator[T]:
    """
    Return an iterator of the partial values of the reduce operation.

    The last value is always equal to reduce(func, seq, start)
    """
    out = start
    it = iter(seq)
    while True:
        try:
            yield out
            out = func(out, next(it))
        except StopIteration:
            raise StopIteration

        
def mesh(xs: Iterable[T], ys: Iterable[T2]) -> Iterator[tuple[T, T2]]:
    """
    iterator over the lexicographical pairs

    ::

        (x1 y1) (x1 y2) (x1 y3) ... (x1 yn)
        (x2 y1) (x2 y2) (x2 y3) ... (x2 yn)
        ...
        (xn y1) ...                 (xn yn)

    Example
    -------

    .. code::

        In [4]: list(iterlib.mesh((1,2,3), "A B C".split()))
        Out[4]:
        [(1, 'A'),
         (1, 'B'),
         (1, 'C'),
         (2, 'A'),
         (2, 'B'),
         (2, 'C'),
         (3, 'A'),
         (3, 'B'),
         (3, 'C')]
    """
    for x in xs:
        for y in ys:
            yield (x, y)

            
def mesh3(A, B, C):
    """
    the same as mesh, but over 3 iterators
    """
    for a in A:
        for b in B:
            for c in C:
                yield (a, b, c)

                
def unique(seq: Iterable[T]) -> Iterator[T]:
    """
    Return only unique elements of a sequence (keeps order)

    If seq is not an iterator or order is not important it is more
    efficient to call set instead.

    .. note:: Elements must be hashable

    >>> tuple(unique((1, 2, 3)))
    (1, 2, 3)
    >>> tuple(unique((1, 2, 1, 3)))
    (1, 2, 3)

    """
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item

            
def interleave(seqs, pass_exceptions=()):
    """ 
    Interleave a sequence of sequences

    >>> list(interleave([[1, 2], [3, 4]]))
    [1, 3, 2, 4]

    >>> ''.join(interleave(('ABC', 'XY')))
    'AXBYC'

    Both the individual sequences and the sequence of sequences may be infinite

    Returns a lazy iterator
    """
    iters = map(iter, seqs)
    while iters:
        newiters = []
        for itr in iters:
            try:
                yield next(itr)
                newiters.append(itr)
            except (StopIteration,) + tuple(pass_exceptions):
                pass
        iters = newiters
        

def split_in_chunks(seq: Iterable[T], chunksize: int) -> Iterator[list[T]]:
    """
    splits a sequence into chunks

    Example
    -------

    >>> s = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    >>> list(split_in_chunks(s, 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    chunk = []
    for i, item in enumerate(seq):
        chunk.append(item)
        if i % chunksize == chunksize-1:
            yield chunk
            chunk = []
    # return [seq[i:i+chunksize] for i in range(0, len(seq), chunksize)]


def classify(s: Sequence[T], keyfunc: Callable[[T], T2]) -> dict[T2, list[T]]:
    """
    Split `s` according to `keyfunc`

    Args:
        s: the sequence to split
        keyfunc: a function taking an item of s and returning the key under which
            all similar items will be grouped

    Example
    ~~~~~~~

        >>> s = [
        ...     {'name': 'John', 'city': 'New York'},
        ...     {'name': 'Otto', 'city': 'Berlin'},
        ...     {'name': 'Jakob', 'city': 'Berlin'},
        ...     {'name': 'Bob', 'city': 'New York'}
        ... ]
        >>> groups = classify(s, lambda record: record['city'])
        {'Berlin': [{'name': 'Otto', 'city': 'Berlin'},
                    {'name': 'Jakob', 'city': 'Berlin'}],
         'New York': [{'name': 'John', 'city': 'New York'},
                      {'name': 'Bob', 'city': 'New York'}]}
    """
    groups = {}
    for item in s:
        key = keyfunc(item)
        group = groups.get(key)
        if group:
            group.append(item)
        else:
            groups[key] = [item]
    return groups


if __name__ == '__main__':
    import doctest
    doctest.testmod()
