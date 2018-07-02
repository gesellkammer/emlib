from __future__ import (division as _division,
                        absolute_import as _absolute_import,
                        print_function)
from itertools import *
import operator as _operator
import collections as _collections
import warnings
import random

# ----------------------------------------------------------------------------
# 
#    protocol: arg, seq
#
# ----------------------------------------------------------------------------


def take(n:int, seq):
    "returns the first n elements of seq as a list"
    return list(islice(seq, n))


def first(seq):
    "returns the first element of seq or None if the seq is empty"
    try:
        return next(seq)
    except StopIteration:
        return None

    
def last(seq):
    "returns the last element of seq or None if seq is empty"
    if isinstance(seq, _collections.Sequence):
        try:
            return seq[-1]
        except IndexError:
            return None
    else:
        for x in seq:
            pass
        return x

    
def consume(n, iterator):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        _collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def drop(n, seq):
    """
    return an iterator over seq with n elements consumed
    
    >>> list(drop(range(10), 3))
    [3, 4, 5, 6, 7, 8, 9]
    """
    return islice(seq, n, None)


def nth(n, seq):
    "Returns the nth item"
    return list(islice(seq, n, n+1))[0]


def pad(element, seq):
    """
    Returns the elements in seq and then return element indefinitely.

    Useful for emulating the behavior of the built-in map() function.
    
    >>> list(take(pad(range(10), "X"), 15))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'X', 'X', 'X', 'X', 'X']
    """
    return chain(seq, repeat(element))


def padlast(seq):
    """
    Returns elements in seq, then returns the last element indefinitely
    """
    for x in seq:
        yield x
    while True:
        yield x


def ncycles(n, seq):
    "Returns the sequence elements n times"
    return chain(*repeat(seq, n))


def dotproduct(A, B):
    return sum(a*b for a, b in zip(A, B))
    # return sum(imap(_operator.mul, vec1, vec2))


def repeatfunc(func, times=None, *args):
    """
    Repeat calls to func with specified arguments.

    Example:  repeatfunc(random.random)
    """
    if times is None:
        return starmap(func, repeat(args))
    else:
        return starmap(func, repeat(args, times))

    
def pairwise(iterable):
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


def window(seq, size, step=1):
    """
    iterate over subseqs of seq
    
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
    iterators = tee(seq, size)
    for skip_steps, itr in enumerate(iterators):
        for ignored in islice(itr, skip_steps):
            pass
    window_itr = zip(*iterators)
    if step != 1:
        window_itr = islice(window_itr, 0, 99999999, step)
    return window_itr


def iterchunks(chunksize, seq):
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
            yield tuple(filter(lambda x:x is not Sentinel, chunk))
            break
        else:
            yield chunk


def isiterable(obj, exclude=(str, bytes)):
    if exclude is None:
        return hasattr(obj, '__iter__')
    return hasattr(obj, '__iter__') and (not isinstance(obj, exclude))
    

def grouper(n, seq, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    
    >>> for group in grouper('ABCDEFG', 3, fillvalue='x'): print(group)
    ('A', 'B', 'C')
    ('D', 'E', 'F')
    ('G', 'x', 'x')

    """
    args = [iter(seq)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def groupby(f, coll):
    """ Group a collection by a key function

    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    >>> groupby(len, names)
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
    """
    d = dict()
    for item in coll:
        key = f(item)
        if key not in d:
            d[key] = []
        d[key].append(item)
    return d


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    assert isiterable(iterable, None)
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_permutation(iterable, r=None):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def partialsum(seq, start=0):
    """
    for each elem in seq return the partial sum

    n0 -> n0
    n1 -> n0 + n1
    n2 -> n0 + n1 + n2
    n3 -> n0 + n1 + n2 + n3
    """
    assert isiterable(seq)
    accum = start
    for i in seq:
        accum += i
        yield accum

        
def partialmul(seq, start=1):
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

        
def avgnow(seq):
    """
    return the average of the elements of seq until now
    """
    accum = 0
    i = 0
    for n in seq:
        accum += n
        i += 1
        yield accum / i

        
def flatten(s, exclude=(str, bytes)):
    """
    return an iterator to the flattened items of sequence s
    strings are not flattened
    """
    try:
        it = iter(s)
    except TypeError:
        yield s
        raise StopIteration
    for elem in it:
        if isinstance(elem, exclude):
            yield elem
        else:
            for subelem in flatten(elem, exclude):
                yield subelem

                
def flatten1(seq, exclude=(str, bytes)):
    """
    the same as flatten but only one level deep.

    >>> list(flatten1([(1,2), 3, (4, 5, 6), (7, 8), 9]))
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list( flatten1([(1, 2), 3, (4, 5, (6, 7)), 8, 9]) )
    [1, 2, 3, 4, 5, (6, 7), 8, 9]
    """
    for elem in seq:
        if hasattr(elem, '__iter__'):
            if isinstance(elem, exclude):
                yield elem
            else:
                for subelem in elem:
                    yield subelem
        else:
            yield elem

            
def zipflat(*seqs):
    """
    like izip but flat
    
    Example
    =======
    
    >>> list(zipflat(range(5), "ABCDE"))
    [0, 'A', 1, 'B', 2, 'C', 3, 'D', 4, 'E']
    """
    for elems in zip(*seqs):
        for elem in elems:
            yield elem

            
def butlast(seq):
    """
    iterate over seq[:-1]
    
    >>> list(butlast(range(5)))
    [0, 1, 2, 3]
    """
    seq_iter = iter(seq)
    last = next(seq_iter)
    for i in seq_iter:
        yield last
        last = i

        
def butn(n, seq):
    """
    iterate over seq[:-n]
    """
    seqi = iter(seq)
    d = _collections.deque((next(seqi) for i in range(n)), n)
    for x in seqi:
        yield d.popleft()
        d.append(x)

        
def intercalate(item, seq):
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

        
def mesh(*seqs):
    """
    iterator over the lexicographical pairings
    
    (x1 y1) (x1 y2) (x1 y3) ... (x1 yn)
    (x2 y1) (x2 y2) (x2 y3) ... (x2 yn)
    ...
    (xn y1) ...                 (xn yn)
    
    Example
    =======
    
    In [4]: list(iters.mesh((1,2,3), "A B C".split()))
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
    if len(seqs) == 2:
        for x in seqs[0]:
            for y in seqs[1]:
                yield (x,y)
    elif len(seqs) == 3:
        X, Y, Z = seqs
        for x in X:
            for y in Y:
                for z in Z:
                    yield (x, y, z)
    elif len(seqs) == 4:
        A, B, C, D = seqs
        for a in A:
            for b in B:
                for c in C:
                    for d in D:
                        yield (a, b, c, d)
    else:
        raise ValueError("how to do this efficiently??")

                
def unique(seq):
    """ 
    Return only unique elements of a sequence

    >>> tuple(unique((1, 2, 3)))
    (1, 2, 3)
    >>> tuple(unique((1, 2, 1, 3)))
    (1, 2, 3)

    """
    try:
        L = len(seq)
        return set(seq)
    except TypeError:
        return _unique_gen(seq)

    
def _unique_gen(seq):    
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

        
# def unpack(iterable, result=tuple):
#     """
#     Similar to python 3's *rest unpacking (don't use it in Python3)

#     result: how to pack the rest items. Pass None to get an iterator.
    
#     >>> x, y, rest = unpack('test')
#     >>> assert x == 't' and y == 'e' and rest == ('s', 't')
#     """
#     def how_many_unpacked():
#         import inspect, opcode
#         f = inspect.currentframe().f_back.f_back
#         if ord(f.f_code.co_code[f.f_lasti]) == opcode.opmap['UNPACK_SEQUENCE']:
#             return ord(f.f_code.co_code[f.f_lasti+1])
#         raise ValueError("Must be a generator on RHS of a multiple assignment!!")
#     iterator = iter(iterable)
#     has_items = True
#     amount_to_unpack = how_many_unpacked() - 1
#     item = None
#     for num in xrange(amount_to_unpack):
#         if has_items:        
#             try:
#                 item = next(iterator)
#             except StopIteration:
#                 item = None
#                 has_items = False
#         yield item
#     if has_items:
#         if result is not None:
#             yield result(iterator)
#         else:
#             yield iterator
#     else:
#         yield None
       

if __name__ == '__main__':
    import doctest
    doctest.testmod()
