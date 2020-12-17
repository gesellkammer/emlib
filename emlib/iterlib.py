from __future__ import annotations
from itertools import *
import operator as _operator
import collections as _collections
import random as _random
from math import inf
from typing import (
    Set, Any, List, TypeVar, Callable,
    Union as U,
    Optional as Opt,
    Iterator as Iter,
    Tuple as Tup
)
T = TypeVar("T")
T2 = TypeVar("T2")

# ----------------------------------------------------------------------------
# 
#    protocol: seq, other parameters
#
#    Exception: functions that match the map protocol: func(predicate, seq)
#
# ----------------------------------------------------------------------------


def take(seq: Iter[T], n:int) -> List[T]:
    """returns the first n elements of seq as a list"""
    return list(islice(seq, n))


def last(seq: Iter[T]) -> Opt[T]:
    """returns the last element of seq or None if seq is empty"""
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

    
def consume(iterator: Iter, n:int) -> None:
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    if n is None:
        # feed the entire iterator into a zero-length deque
        _collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

        
def quantify(iterable: Iter, pred: Callable=bool) -> int:
    """
    Count how many times the predicate is true
    
    Example
    =======
    
    >>> quantify("ABACA", lambda letter:letter=='A')
    3
    """
    return sum(map(pred, iterable))


def drop(seq: Iter[T], n: int) -> Iter[T]:
    """
    return an iterator over seq with n elements consumed
    
    >>> list(drop(range(10), 3))
    [3, 4, 5, 6, 7, 8, 9]
    """
    return islice(seq, n, None)


def nth(seq: Iter[T], n: int) -> T:
    """Returns the nth item"""
    return list(islice(seq, n, n+1))[0]


def pad(seq: Iter[T], element:T2=None) -> Iter[U[T, T2]]:
    """
    Returns the elements in seq and then return element indefinitely.

    Useful for emulating the behavior of the built-in map() function.
    
    >>> list(take(pad(range(10), "X"), 15))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'X', 'X', 'X', 'X', 'X']
    """
    return chain(seq, repeat(element))


def ncycles(seq: Iter[T], n: int) -> Iter[T]:
    """Returns the sequence elements n times"""
    return chain(*repeat(seq, n))


def dotproduct(vec1: Iter[T], vec2: Iter[T]) -> T:
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


def pairwise(iterable: Iter[T]) -> Iter[Tup[T, T]]:
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ..."
    
    similar to window(seq, size=2, step=1)
    
    Example
    =======
    
    >>> list(pairwise(range(4)))
    [(0, 1), (1, 2), (2, 3)]
    """    
    a, b = tee(iterable)
    try:
        next(b)
    except StopIteration:
        pass
    return zip(a, b)


def window(iterable: Iter[T], size=3, step=1) -> Iter[Tup[T, ...]]:
    """
    iterate over subseqs of iterable
    
    Example
    =======
    
    >>> seq = range(6)
    
    >>> list(window(seq, 3, 1))
    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    
    >>> list(window(seq, 3, 2))
    [(0, 1, 2), (2, 3, 4)]
    
    # the same as pairwise
    >>> assert list(window(range(5), 2, 1)) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    """
    iterators = tee(iterable, size)
    for skip_steps, itr in enumerate(iterators):
        for _ in islice(itr, skip_steps):
            pass
    window_itr = zip(*iterators)
    if step != 1:
        window_itr = islice(window_itr, 0, 99999999, step)
    return window_itr


def iterchunks(seq: Iter, chunksize: int) -> Iter[Tup]:
    """
    Returns an iterator over chunks of seq of at most `chunksize` size.
    If seq is finite and not divisible by chunksize, the last chunk will
    have less than `chunksize` elements.

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


def parse_range(start, stop:int=None, step:int=None) -> Tup[int, int, int]:
    if stop is None:
        stop = int(start)
        start = 0
    if step is None:
        step = 1
    return start, stop, step


def chunks(start:int, stop:int=None, step:int=None) -> Iter[Tup[int, int]]:
    """
    Returns a generator of tuples (offset, chunksize)

    Example
    =======

    chunks(0, 10, 3)
    (0, 3)
    (3, 3)
    (6, 3)
    (9, 1)
    """
    start, stop, step = _parse_range(start, stop, step)
    while start < stop:
        size = min(stop - start, step)
        yield start, size
        start += step


def isiterable(obj, exclude=(str,)) -> bool:
    if exclude:
        return hasattr(obj, '__iter__') and (not isinstance(obj, exclude))
    return hasattr(obj, '__iter__')
    

def grouper(seq: Iter, n: int, fillvalue=None) -> Iter:
    """
    Collect data into fixed-length chunks or blocks
    
    >>> for group in grouper('ABCDEFG', 3, fillvalue='x'): print(group)
    ('A', 'B', 'C')
    ('D', 'E', 'F')
    ('G', 'x', 'x')

    """
    args = [iter(seq)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def random_combination(iterable: Iter, r):
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


def partialsum(seq: Iter[T], start=0) -> Iter[T]:
    """
    for each elem in seq return the partial sum

    n0 -> n0
    n1 -> n0 + n1
    n2 -> n0 + n1 + n2
    n3 -> n0 + n1 + n2 + n3
    """
    accum = start
    for i in seq:
        accum += i
        yield accum

def partialmul(seq: Iter[T], start=1) -> Iter[T]:
    """
    return the accumulated multiplication

    n0 -> n0
    n1 -> n1 * n0
    n2 -> n2 * n1 * n0
    """
    assert isiterable(seq)
    accum = start
    for i in seq:
        accum *= i
        yield accum

        
def avgnow(seq: Iter[T]) -> T:
    """
    return the average of the elements of seq until now
    """
    accum = 0
    i = 0
    for n in seq:
        accum += n
        i += 1
        yield accum / i


def ilen(seq: Iter) -> int:
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = count()
    _collections.deque(zip(seq, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def avg(seq: Iter[T], empty=0) -> T:
    """ Return the average of seq, or `empty` if the seq. is empty

    NB: if you know the size of seq, it is faster to do
        `sum(seq) / len_of_seq`

    >>> avg(range(1_000_000))
    499999.5
    """
    accum, i = 0, 0
    for x in seq:
        accum += x
        i += 1
    return accum / i if i else empty

        
def flatten(s: Iter[U[T, Iter[T]]], exclude=(str,), levels=inf) -> Iter[T]:
    """
    return an iterator to the flattened items of sequence s
    strings are not flattened

    seq = [1, [2, 3], [4, [5, 6]]]
    list(flatten(seq))
    -> [1, 2, 3, 4, 5, 6]
    """
    if not hasattr(s, '__iter__') or isinstance(s, exclude):
        yield s
    else:
        for elem in s:
            if isinstance(elem, exclude) or levels <= 0:
                yield elem
            else:
                yield from flatten(elem, exclude, levels-1)


def flattened(s: Iter[U[T, Iter[T]]], exclude=(str,), levels=inf) -> List[T]:
    return list(flatten(s, exclude=exclude, levels=levels))


# as a reference, a non-recursive version. It is slower than flatten
def _flatten_nonrec(iterable):
    iterator, sentinel, stack = iter(iterable), object(), []
    pop = stack.pop
    append = stack.append
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = pop()
        elif isinstance(value, str):
            yield value
        else:
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                append(iterator)
                iterator = new_iterator

            
def flattenonly(l, types):
    """
    Flatten only if subsequences are of type 'type' 

    type: a type or a tuple of types, as passed to isinstance

    Example: flatten only list items, pass tuples untouched

    l = [1, [2, 3], (4, 5)]
    list(flattenonyl(l, list))
    --> [1, 2, 3, (4, 5)]
    """
    for elem in l:
        if isinstance(elem, types):
            yield from flattenonly(elem, types)
        else:
            yield elem

            
def zipflat(*seqs):
    """
    like izip but flat. It has the same effect as flatten(zip(seqA, seqB), levels=1)
    
    Example
    ~~~~~~~
    
    >>> list(zipflat(range(5), "ABCDE"))
    [0, 'A', 1, 'B', 2, 'C', 3, 'D', 4, 'E']
    """
    for elems in zip(*seqs):
        for elem in elems:
            yield elem

            
def butlast(seq):
    # type: (Iter[T]) -> Iter[T]
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

        
def butn(seq, n):
    # type: (Iter[T], int) -> Iter[T]
    """
    iterate over seq[:-n]
    """
    seqi = iter(seq)
    # d = _collections.deque((next(seqi) for i in xrange(n)), n)
    d = _collections.deque((next(seqi) for _ in range(n)), n)
    for x in seqi:
        yield d.popleft()
        d.append(x)

        
def intercalate(seq, item):
    # type: (Iter[T], Any) -> Iter[T]
    """
    Example
    =======
    
    >>> list(intercalate(range(5), "X"))
    [0, 'X', 1, 'X', 2, 'X', 3, 'X', 4]
    """
    return butlast(zipflat(seq, repeat(item)))


def apply_funcs(seq, *funcs):
    tees = tee(seq, len(funcs))
    return [func(t) for t, func in zip(tees, funcs)]


def reductions(seq, func, start=0):
    # type: (Iter[T], Callable, T) -> Iter[T]
    """
    return an iterator of the partial values of the reduction.
    the last value is always equal to reduce(func, seq, start)
    """
    out = start
    it = iter(seq)
    while True:
        try:
            yield out
            out = func(out, next(it))
        except StopIteration:
            raise StopIteration

        
def mesh(xs, ys):
    # type: (Iter[T], Iter[T2]) -> Iter[Tup[T, T2]]
    """
    iterator over the lexicographical pairs
    
    (x1 y1) (x1 y2) (x1 y3) ... (x1 yn)
    (x2 y1) (x2 y2) (x2 y3) ... (x2 yn)
    ...
    (xn y1) ...                 (xn yn)
    
    Example
    =======
    
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

                
def unique(seq):
    # type: (Iter[T]) -> Set[T]
    """ 
    Return only unique elements of a sequence. If seq is not an iterator,
    it is better to call set instead

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
        

if __name__ == '__main__':
    import doctest
    doctest.testmod()


del Set, Any, List, TypeVar, Callable, U, Opt, Iter, Tup
