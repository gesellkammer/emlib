# -*- coding: utf-8 -*-
from __future__ import annotations
import os as _os
import sys as _sys
import math
from bisect import bisect as _bisect
from collections import namedtuple as _namedtuple
import re as _re
import dataclasses
import warnings

import numpy as np
from fractions import Fraction

from emlib.typehints import T, T2, number_t, List, Tup, Opt, Iter, U, Seq, Func


# ------------------------------------------------------------
#     CHUNKS
# ------------------------------------------------------------


def reverse_recursive(seq: list):
    """
    >>> reverse_recursive([1, 2, [3, 4], [5, [6, 7]]])
    [[[7, 6], 5], [4, 3], 2, 1]

    Args:
        seq: a (possibly nested) list of elements

    Returns:
        a reversed version of `seq` where all sub-lists are also reversed.

    NB: only lists will be reversed, other iterable collection remain untouched

    """
    out = []
    for x in seq:
        if isinstance(x, list):
            x = reverse_recursive(x)
        out.append(x)
    out.reverse()
    return out


def wrap_by_sizes(flatseq: list, packsizes: Seq[int]) -> List[list]:
    """
    Example:

    >>> flatseq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> wrap_by_sizes(flatseq, [3, 5, 2])
    [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10]]

    Args:
        flatseq: a flat sequence of items
        packsizes: a list of sizes
    """
    def partialsum(seq):
        accum = 0
        for i in seq:
            accum += i
            out.append(accum)
        return out

    offsets = [0] + partialsum(packsizes)
    start = offsets[0]
    out = []
    for i in range(1, len(offsets)):
        end = offsets[i]
        out.append(flatseq[start:end])
        start = end
    return out


# ------------------------------------------------------------
#                                                            -
#                          SEARCH                            -
#                                                            -
# ------------------------------------------------------------


def nearest_element(item: float, seq: U[List[float], np.ndarray]) -> float:
    """
    Find the nearest element (the element, not the index) in seq

    **NB**: assumes that seq is sorted (this is not checked). seq can also be a
        numpy array, in which case searchsorted is used instead of bisect

    Args:
        item: the item to search
        seq: either a list of numbers, or a numpy array

    Returns:
        the value of the nearest element of seq

    Example::

        >>> seq = list(range(10))
        >>> nearest_element(4.1, seq)
        4
        >>> nearest_element(3.6, [1,2,3,4,5])
        4
        >>> nearest_element(200, np.array([3,5,20]))
        20
        >>> nearest_element(0.5, [0, 1])
        0
        >>> nearest_element(1, [1, 1, 1])
        1
    """
    # check boundaries
    seq0 = seq[0]
    if item <= seq0:
        return seq0
    seq1 = seq[-1]
    if item >= seq1:
        return seq1
    if isinstance(seq, np.ndarray):
        ir = seq.searchsorted(item, 'right')
    else:
        ir = _bisect(seq, item)
    element_r = seq[ir]
    element_l = seq[ir - 1]
    if abs(element_r - item) < abs(element_l - item):
        return element_r
    return element_l


def nearest_unsorted(x:T, seq:List[T]) -> T:
    """
    seq is a numerical sequence. it can be unsorted
    x is a number or a seq of numbers

    >>> assert nearest_unsorted(3.6, (1,2,3,4,5)) == 4
    >>> assert nearest_unsorted(4, (2,3,4)) == 4
    >>> assert nearest_unsorted(200, (3,5,20)) == 20
    """
    return min((abs(x - y), y) for y in seq)[1]


def nearest_index(item: number_t, seq: List[number_t]) -> int:
    """
    Return the index of the nearest element in seq to item
    Assumes that seq is sorted

    Example
    ~~~~~~~

    >>> seq = [0, 3, 4, 8]
    >>> nearest_index(3.1, seq)
    1
    >>> nearest_index(6.5, seq)
    3
    """
    ir = _bisect(seq, item)
    if ir >= len(seq) or ir <= 0:
        return ir
    il = ir - 1
    if abs(seq[ir] - item) < abs(seq[il] - item):
        return ir
    return il


def fuzzymatch(pattern:str, strings:List[str]) -> List[Tup[float, str]]:
    """
    return a subseq. of strings sorted by best score.
    Only strings representing possible matches are returned

    pattern: the string to search for within S
    strings: a list os possible strings
    """
    pattern = '.*?'.join(map(_re.escape, list(pattern)))

    def calculate_score(pattern, s):
        match = _re.search(pattern, s)
        if match is None:
            return 0
        return 100.0 / ((1 + match.start()) * (match.end() - match.start() + 1))

    S2 = []
    for s in strings:
        score = calculate_score(pattern, s)
        if score > 0:
            S2.append((score, s))
    S2.sort(reverse=True)
    return S2

# ------------------------------------------------------------
#
#    SORTING
#
# ------------------------------------------------------------


def sort_natural(seq: list, key=None) -> list:
    """
    sort a sequence 'l' of strings naturally, so that
    'item1' and 'item2' are before 'item10'

    key: a function to use as sorting key

    Examples
    ~~~~~~~~

    >>> seq = ["e10", "e2", "f", "e1"]
    >>> sorted(seq)
    ['e1', 'e10', 'e2', 'f']
    >>> sort_natural(seq)
    ['e1', 'e2', 'e10', 'f']

    >>> seq = [(2, "e10"), (10, "e2")]
    >>> sort_natural(seq, key=lambda tup:tup[1])
    [(10, 'e2'), (2, 'e10')]
    >>> sort_natural(seq, key=1) # this is the same as above
    [(10, 'e2'), (2, 'e10')]
    """
    import re

    if isinstance(key, int):
        import operator
        key = operator.itemgetter(key)

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    if key:
        return sorted(seq, key=lambda x: alphanum_key(key(x)))
    return sorted(seq, key=alphanum_key)


def sort_natural_dict(d: dict, recursive=True) -> dict:
    """
    sort dict d naturally and recursively
    """
    rows = []
    sorted_rows = {}
    if recursive:
        for key, value in d.items():
            if isinstance(value, dict):
                value = sort_natural_dict(value, recursive=recursive)
            rows.append((key, value))
            sorted_rows = sort_natural(rows)
    else:
        keys = list(d.keys())
        sorted_rows = [(key, d[key]) for key in sort_natural(keys)]
    return dict(sorted_rows)


def issorted(seq:list, key=None) -> bool:
    """
    returns True if seq is sorted

    if any(x0 > x1 for x0, x1 in pairwise(seq)
    """
    lastx = -float('inf')
    if key is not None:
        for x in seq:
            x = key(x)
            if x < lastx:
                return False
            lastx = x
        return True
    else:
        for x in seq:
            if x < lastx:
                return False
            lastx = x
        return True


def firstval(*values:T, default:T=None, sentinel=None) -> T:
    """
    Get the first value in values which is not sentinel

    For delayed execution (similar to a if b else c) a value can be a
    callable which should return the value in question

    Example::

        >>> default = firstval(a, lambda: lengthycomputation(), config['a'])
        # this is the same as
        # a if a is not None else a2 := lengthycomputation() if a2 is not None else config['a']
    """
    for value in values:
        if callable(value):
            value2 = value()
            if value2 is not sentinel:
                return value2
        elif value is not sentinel:
            return value
    return default


def zipsort(a:Seq[T], b:Seq[T2], key:Func=None, reverse=False
            ) -> Tup[List[T], List[T2]]:
    """
    Sort a and keep b in sync

    - equivalent to

    a, b = unzip(sorted(zip(a, b), key=key))

    Example
    ~~~~~~~

    >>> names = ['John', 'Mary', 'Nick']
    >>> ages  = [20,      10,     34]
    >>> ages, names = zipsort(ages, names)
    >>> names
    ('Mary', 'John', 'Nick')
    """
    zipped = sorted(zip(a, b), key=key, reverse=reverse)
    a, b = zip(*zipped)
    return (a, b)


def duplicates(seq:Seq[T], mincount=2) -> List[T]:
    """
    Find all elements in seq which are present at least `mincount` times
    """
    from collections import Counter
    counter = Counter(seq).items()
    return [item for item, count in counter if count >= mincount]


def remove_duplicates(seq: List[T]) -> List[T]:
    """
    Remove all duplicates in seq while keeping its order
    If order is not important, use list(set(seq))

    Args:
        seq: a list of elements (elements must be hashable)

    Returns:
        a new list with all the unique elements of seq in its
        original order
        NB: list(set(...)) does not keep order
    """
    # Original implementation:
    # seen = set()
    # seen_add = seen.add
    # return [x for x in seq if not (x in seen or seen_add(x))]

    # In python >= 3.7 we use the fact that dicts keep order:
    return list(dict.fromkeys(seq))


def fractional_slice(seq: Seq, step:float, start=0, end=-1) -> list:
    """
    Given a list of elements, take a slice similar to seq[start:end:step],
    but allows step to be a fraction

    Example:

    >>> fractional_slice(range(10), 1.5)
    >>> [0, 2, 3, 5, 6, 8]

    Args:
        seq: the sequence of elements
        step: the step size
        start: start index
        end: end index

    Returns:
        the resulting list

    """
    if step < 1:
        raise ValueError("step should be >= 1 (for now)")

    accum = 0
    out = [seq[start]]
    for elem in seq[start+1:end]:
        accum += 1
        if accum >= step:
            out.append(elem)
            accum -= step
    return out


def sec2str(seconds:float) -> str:
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    s = seconds % 60
    if h > 0:
        fmt = "{h}:{m:02}:{s:06.3f}"
    else:
        fmt = "{m:02}:{s:06.3f}"
    return fmt.format(**locals())


def parse_time(t:str) -> float:
    """
    Given a time in the format HH:MM:SS.mmm or any sub-form of it
    (SS.mmm, MM:SS, etc), return the time in seconds. This is the
    inverse of sec2str

    Args:
        t: the time as string

    Returns:
        seconds
    """
    parts = t.split(":")
    if len(parts) == 1:
        # only seconds
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[1]) + float(parts[0])*60
    elif len(parts) == 3:
        return float(parts[2]) + float(parts[1])*60 + float(parts[0])*3600
    else:
        raise ValueError("Format not understood")


def ljust(s: str, width: int, fillchar=" ") -> str:
    """
    Like str.ljust, but makes sure that the output is always the given width,
    even if s is longer than `width`
    """
    s = s if isinstance(s, str) else str(s)
    s = s.ljust(width, fillchar)
    if len(s) > width:
        s = s[:width]
    return s


# ------------------------------------------------------------
#
#     namedtuple utilities
#
# ------------------------------------------------------------


def namedtuple_addcolumn(namedtuples, seq, column_name, classname=""):  # type: ignore
    """

    namedtuples: a list of namedtuples
    seq: the new column
    column_name: name of the column
    classname: nane of the new namedtuple

    Returns a list of namedtuples with the added column
    """
    t0 = namedtuples[0]
    assert isinstance(t0, tuple) and hasattr(t0, "_fields"), "namedtuples should be a seq. of namedtuples"
    name = classname or namedtuples[0].__class__.__name__ + '_' + column_name  # type: str
    New_Tup = _namedtuple(name, t0._fields + (column_name,))  # type: ignore
    newtuples = [New_Tup(*(t + (value,)))
                 for t, value in zip(namedtuples, seq)]
    return newtuples


def namedtuples_renamecolumn(namedtuples, oldname, newname, classname=None):  # type: ignore
    """
    Rename the column of a seq of namedtuples

    >>> from collections import namedtuple
    >>> Person = namedtuple("Person", "firstname familyname")
    >>> people = [Person("John", "Smith"), Person("Amy", "Adams")]
    >>> people2 = namedtuples_renamecolumn(people, "firstname", "first_name")
    >>> people2[0]._fields
    ('first_name', 'familyname')
    """
    if classname is None:
        classname = "%s_R" % namedtuples[0].__class__.__name__
    newfields = [field if field != oldname else newname
                 for field in namedtuples[0]._fields]
    New_Tup = _namedtuple(classname, newfields)
    newtuples = [New_Tup(*t) for t in namedtuples]
    return newtuples


def namedtuple_extend(name:str, orig, columns: U[str, Tup[str,...]]):
    """
    create a new constructor with the added columns

    it returns the class constructor and an ad-hoc
    constructor which takes as arguments an instance
    of the original namedtuple and the additional args

    Args:
        name: new name for the type
        orig: an instance of the original namedtuple or the constructor itself
        columns : the columns to add
    Returns:
        a tuple (newtype, newtype_from_old)
    >>> from collections import namedtuple
    >>> Point = namedtuple("Point", "x y")
    >>> p = Point(10, 20)
    >>> Vec3, fromPoint = namedtuple_extend("Vec3", Point, "z")
    >>> Vec3(1, 2, 3)
    Vec3(x=1, y=2, z=3)
    >>> fromPoint(p, 30)
    Vec3(x=10, y=20, z=30)
    """
    if isinstance(columns, str):
        columns = columns.split()
    fields = orig._fields + tuple(columns)
    N = _namedtuple(name, fields)

    def new_from_orig(orig, *args, **kws):
        """
        orig: the original namedtuple from which to construct an extended version
        args, kws: the missing columns
        """
        return N(*(orig + args), **kws)

    return N, new_from_orig

# ------------------------------------------------------------
#
#     MISCELLANEOUS
#
# ------------------------------------------------------------


def isiterable(obj, exceptions:Tup[type, ...]=(str, bytes)) -> bool:
    """
    >>> assert isiterable([1, 2, 3])
    >>> assert not isiterable("test")
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, exceptions)


def isgenerator(obj):
    import types
    return isinstance(obj, types.GeneratorType)


def isgeneratorlike(obj):
    return hasattr(obj, '__iter__') and not hasattr(obj, '__len__')


def unzip(seq: Iter[T]) -> List[T]:
    """
    >>> a, b = (1, 2), ("A", "B")
    >>> list(zip(a, b))
    [(1, 'A'), (2, 'B')]
    >>> list( unzip(zip(a, b)) ) == [a, b]
    True
    """
    return list(zip(*seq))


def asnumber(obj, accept_fractions=True, accept_expon=False) -> U[int, float, Fraction, None]:
    """
    Return obj as number, or None of it cannot be converted
    to a number

    >>> asnumber(1)
    1
    >>> asnumber("3.4")
    3.4
    >>> asnumber("1/3", accept_fractions=True)
    Fraction(1, 3)
    >>> asnumber("hello") is None
    True
    """
    if hasattr(obj, '__float__'):
        return obj
    elif isinstance(obj, str):
        if accept_fractions and "/" in obj:
            return Fraction(obj)
        try:
            asint = int(obj)
            return asint
        except ValueError:
            pass
        if not accept_expon and _re.search(r"[eE][+-]", obj):
            return None
        try:
            asfloat = float(obj)
            return asfloat
        except ValueError:
            return None
    else:
        return None


def astype(type_, obj, construct=None):
    """
    Return obj as type. If obj is already of said type, obj itself is returned
    Otherwise, obj is converted to type. If a special contructor is needed,
    it can be given as `construct`

    obj: the object to be checkec/converted
    type_: the type the object should have
    construct: if given, a function (obj) -> obj of type `type_`
        Otherwise, type_ itself is used

    Example


    astype(list, (3, 4)) -> [3, 4]
    l = [3, 4]
    assert astype(list, l) is l --> True
    aslist = partial(astype, list)
    """
    return obj if isinstance(obj, type_) else (construct or type_)(obj)


def could_be_number(x) -> bool:
    """
    True if `x` can be interpreted as a number

    2          | True
    "0.4"      | True
    "3/4"      | True
    "inf"      | True
    "mystring" | False

    """
    raise Deprecated("Use asnumber(x) is not None")
    

def str_is_number(s:str, accept_exp=False, accept_fractions=False):
    """
    NB: fractions should have the form num/den, like 3/4, with no spaces in between
    """
    if accept_exp and accept_fractions:
        return could_be_number(s)
    import re
    if accept_exp:
        return re.fullmatch(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?", s) is not None
    elif accept_fractions:
        return bool(re.fullmatch(r"[-+]?[0-9]+/[0-9]+", s) or re.fullmatch(r"[-+]?[0-9]*\.?[0-9]+", s))
    else:
        return re.fullmatch(r"[-+]?[0-9]*\.?[0-9]+", s) is not None


def dictmerge(dict1: dict, dict2: dict) -> dict:
    """
    Merge the contents of the two dicts.
    If they have keys in common, the value in dict1 is overwritten
    by the value in dict2

    >>> a, b = {'A': 1, 'B': 2}, {'B': 20, 'C': 30}

    >>> dictmerge(a, b) == {'A': 1, 'B': 20, 'C': 30}
    True
    """
    out = dict1.copy()
    out.update(dict2)
    return out


def moses(pred, seq: Iter[T]) -> Tup[List[T], List[T]]:
    """
    return two lists: filter(pred, seq), filter(not pred, seq)

    Example
    ~~~~~~~

    >>> moses(lambda x:x > 5, range(10))
    ([6, 7, 8, 9], [0, 1, 2, 3, 4, 5])
    """
    trueitems = []
    falseitems = []
    for x in seq:
        if pred(x):
            trueitems.append(x)
        else:
            falseitems.append(x)
    return trueitems, falseitems


def allequal(xs: Seq) -> bool:
    """
    Return True if all elements in xs are equal
    """
    x0 = xs[0]
    return all(x==x0 for x in xs)


def make_replacer(conditions:dict) -> Func:
    """
    Create a function to replace many subtrings at once

    conditions: a dictionary of string:replacement

    Example
    ~~~~~~~

    replacer = makereplacer({"&":"&amp;", " ":"_", "(":"\\(", ")":"\\)"})
    replacer("foo & (bar)")
    -> "foo_&amp;_\(bar\)"

    See also: replacemany
    """
    rep = {_re.escape(k): v for k, v in conditions.items()}
    pattern = _re.compile("|".join(rep.keys()))
    return lambda txt: pattern.sub(lambda m: rep[_re.escape(m.group(0))], txt)


def dumpobj(obj) -> list:
    """
    return all 'public' attributes of this object
    """
    return [(item, getattr(obj, item)) for item in dir(obj) if not item.startswith('__')]


def can_be_pickled(obj) -> bool:
    """
    test if obj can be pickled
    """
    import pickle
    try:
        obj2 = pickle.loads(pickle.dumps(obj))
    except pickle.PicklingError:
        return False
    return obj == obj2


def snap_to_grid(x:t.Rat, tick:t.Rat, offset:t.Rat=0, nearest=True) -> t.Rat:
    """
    Given a grid defined by offset + tick * N, find the nearest element
    of that grid to a given x

    NB: the result will have the same type as x, so if x is float,
        then the result will be float, if it is a Fraction, then the
        result will be a fraction
    """
    assert isinstance(x, (float, Fraction))
    t = x.__class__
    if nearest:
        return t(round((x - offset) / tick)) * tick + offset
    else:
        return t(int((x - offset) / tick)) * tick + offset


def snap_array(X:np.ndarray, tick:t.Rat, offset:t.Rat=0,
               out:t.Opt[np.ndarray]=None, nearest=True) -> np.ndarray:
    """
    Assuming a grid t defined by

    t(n) = offset + tick*n

    snap the values of X to the nearest value of t

    NB: tick > 0
    """
    if tick <= 0:
        raise ValueError("tick should be > 0")

    if nearest:
        return _snap_array_nearest(X, tick, offset=float(offset), out=out)
    return _snap_array_floor(X, tick, offset=float(offset), out=out)


def _snap_array_nearest(X:np.ndarray, tick:t.Rat, offset=0, out=None) -> np.ndarray:
    if out is None:
        out = X.copy()
    if offset != 0:
        out -= offset
        out /= tick
        out = np.round(out, out=out)
        out *= tick
        out += offset
    else:
        out /= tick
        out = np.round(out, out=out)
        out *= tick
    return out


def _snap_array_floor(X: np.ndarray, tick:float, offset=0., out:np.ndarray=None) -> np.ndarray:
    if out is None:
        out = X.copy()
    if offset != 0:
        out -= offset
        out /= tick
        out = np.floor(out, out=out)
        out *= tick
        out += offset
    else:
        out /= tick
        out = np.floor(out, out=out)
        out *= tick
    return out


def snap_to_grids(x: number_t, ticks: Seq[number_t], offsets:Seq[number_t]=None,
                  mode='nearest') -> number_t:
    """
    A regular grid is defined as x = offset + tick*n, where n is an integer
    from -inf to inf.

    x: a number or a seq. of numbers
    ticks: a seq. of ticks, each tick defines a grid
    offsets: a seq. of offsets, or None to set offset to 0 for each grid
    mode: one of 'floor', 'ceil', 'nearest'

    Given a list of regular grids, snap the value of x to this grid

    Example
    ~~~~~~~

    snap a time to a grid of 16th notes
    >>> snap_to_grids(0.3, [1/4])
    0.25

    snap a time to a multiple grid of 16ths, 8th note triplets, etc.
    >>> snap_to_grids(0.3, [1/8, 1/6, 1/5])
    0.3333333333333333

    """
    assert isinstance(ticks, (list, tuple))
    assert offsets is None or isinstance(offsets, (list, tuple))

    if offsets is None:
        offsets = [Fraction(0)] * len(ticks)

    if any(tick <= 0 for tick in ticks):
        raise ValueError(f"all ticks must be > 0, got {ticks}")

    def snap_round(x, tick, offset):
        return round((x - offset) * (1 / tick)) * tick + offset

    def snap_floor(x, tick, offset):
        return math.floor((x - offset) * (1 / tick)) * tick + offset

    def snap_ceil(x, tick, offset):
        return math.ceil((x - offset) * (1 / tick)) * tick + offset

    func = {
        'floor': snap_floor,
        'ceil': snap_ceil,
        'nearest': snap_round,
        'round': snap_round
    }.get(mode)
    if func is None:
        raise ValueError(f"mode should be one of 'floor', 'ceil', 'round', but got {mode}")
    quants = [func(x, t, o) for t, o in zip(ticks, offsets)]
    quants.sort(key=lambda quant: abs(quant - x))
    return quants[0]


def distribute_in_zones(x:number_t, split_points:List[number_t], side="left") -> int:
    """
    Returns the index of a "zone" where to place x. A zone is a numeric range
    defined by an inclusive lower boundary and a non-inclusive higher boundary
    (NB: see distribute_in_zones_right for a non-inclusive lower and
    inclusive upper boundary). The edge zones extend to inf.

    Args:
        x: the number to assign a zone to
        split_points: the split points which define the zones
        side: if "left", a zone has an inclusive lower bound and a non-inclusive
            upper bound. "right" is the opposite

    Returns:
        the index of the zone

    Example
    ~~~~~~~

    # 1 and 5 define three zones: (-inf, 1], (1, 5], (5, inf)
    >>> distribute_in_zones(2, [1, 5])
    1
    >>> distribute_in_zones(5, [1, 5])
    2

    SEE ALSO: distribute_in_zones_right
    """
    if side == "right":
        return _distribute_in_zones_right(x, split_points)
    imin = 0
    imax = len(split_points)
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if split_points[imid] <= x:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _distribute_in_zones_right(x:number_t, split_points:Seq[number_t]) -> int:
    """
    the same as distribute_in_zones, but with right inclusive zones
    """
    imin = 0
    imax = len(split_points)
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if split_points[imid] < x:
            imin = imid + 1
        else:
            imax = imid
    return imin


def copyseq(seq: T) -> T:
    """
    return a copy of seq. if it is a list or a tuple, return a copy.
    if it is a numpy array, return a copy with no shared data
    """
    if isinstance(seq, np.ndarray):
        return seq.copy()
    return seq.__class__(seq)


def normalize_slice(slize: U[int, tuple]) -> Tup[int, Opt[int]]:
    """
    >>> normalize_slice((3,))
    (3, None)
    >>> normalize_slice((0, -2))
    (0, -2)
    >>> normalize_slice(4)
    (4, -4)
    """
    if isinstance(slize, int):
        return (slize, -slize)
    if len(slize) == 1:
        return (slize[0], None)
    elif len(slize) == 2:
        return slize
    else:
        raise ValueError("??")


def replace_subseq(seq: Seq[T], subseq: Seq[T], replacement: Seq[T], slize=None
                   ) -> List[T]:
    """
    Args:

        seq: the sequence to be transformed (NOT in-place)
        subseq: a sequence which is MAYBE a sub-sequence of seq
        replacement: the seq. to be put instead
        slize: a slice to be applied to seq before analysis.
            slice-notation   argument to slize
            [:4]             (0, 4)
            [1:]             (1,) or (1,None)
            [1:-2]           (1, -2)
            [2:-2]           (2, -2) or the shorthand 2,
                             which means "use a margin of 2 at each side"

    If subseq is a subsequence of seq, replace this with transform.
    If subseq is not found, return the original seq

    You can test if the seq has been found by testing for identity:

    Example
    ~~~~~~~

    >>> a = (1, 2, 1, 0, 0, 1, 1)
    >>> replace_subseq(a, (1, 0, 0), (1, 8, 8))  # it always returns a list!
    [1, 2, 1, 8, 8, 1, 1]

    # the subseq is not part of the seq, so a2 is a
    >>> a = "a b c d e f g".split()
    >>> a2 = replace_subseq(a, "h h h".split(), "j j j".split())
    >>> a2 is a
    True

    >>> a = "a b c d e f g".split()
    >>> a2 = replace_subseq(a, "b c d".split(), "x x x".split())
    >>> a2 is a
    False
    """
    from itertools import tee, islice

    def window(iterable, size=3, step=1):
        iterators = tee(iterable, size)
        for skip_steps, itr in enumerate(iterators):
            for _ in islice(itr, skip_steps):
                pass
        window_itr = zip(*iterators)
        if step != 1:
            window_itr = islice(window_itr, 0, 99999999, step)
        return window_itr

    if slize is not None:
        slize = normalize_slice(slize)
        sliced_seq = seq[slize[0]:slize[1]]
        seq2 = replace_subseq(sliced_seq, subseq, replacement)
        if sliced_seq is seq2:
            return seq
        out = list(seq)
        out[slize[0]:slize[1]] = seq2
        return out
    subseq = tuple(subseq)
    replacement = tuple(replacement)
    copy = list(seq)
    N = len(subseq)
    changed = False
    for i, win in enumerate(window(seq, N)):
        if win == subseq:
            copy[i:i+N] = replacement
            changed = True
    return copy if changed else seq


def seq_transform(seq: Seq[T], transforms, slize=None, maxiterations=20
                  ) -> Tup[List[T], bool]:
    """
    Transforms subsequences of seq

    transforms: a sequence of transforms. Each transform can be formulated as:
                - (subseq, transformation)
                  Example: ((0, 1, 0, 5), (0, 1, 5, 0))
                - a string of the format "0 1 0 5 -> 0 1 5 0"
                  Numbers will be converted to numbers
                transforms itself can be a multiline string holding individual
                string transforms

    Returns: (newsequence, stable) where stable is True if the transforms
    reached a steady state before maxiterations

    NB: an individual transformation can also have a slize.
    That means that the transform should be applied to a subsequence
    of the seq (see examples). Individual slizes of the transforms
    override the main slize argument (so the slize argument is really
    a default for the slize setting in each transform)

    Example
    ~~~~~~~

    >>> a = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> a2, stable = seq_transform(a, [                   \
        "0 0 0 1 1 1  -> 0 0 1 0 1 1",                    \
        "0 0 0 0 0 1 1 1 1 1 -> 0 0 1 0 0 1 1 0 1 1 [2:]" \
        ])
    >>> a2
    [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    """
    if slize is not None:
        slize = normalize_slice(slize)

    def uncomment(s):
        return s.split("#")[0]

    if isinstance(transforms, str):
        lines = [line for line in [uncomment(line).strip()
                                   for line in transforms.splitlines()]
                 if "->" in line]
        return seq_transform(seq, lines, maxiterations)
    valid_transforms = []
    for transform in transforms:
        if isinstance(transform, str) and "->" in transform:
            if "[" in transform:
                transform, transform_slice = transform.split("[")
                l, r = transform_slice.split("]")[0].split(":")
                l = int(l) if l else 0
                r = int(r) if r else None
                transform_slize = normalize_slice((l, r))
            else:
                transform_slize = slize
            subseq, trans = [s.strip().split() for s in transform.split("->")]
            subseq = list(map(asnumber, subseq))
            trans = list(map(asnumber, trans))
            valid_transforms.append((subseq, trans, transform_slize))
        else:
            if len(transform) == 2:
                transform = (transform[0], transform[1], slize)
            valid_transforms.append(transform)
    # --> We apply the transforms until there are no more changes or
    #     we reach the maxiterations.
    # --> If we reach the maxiterations, the transform system is
    #     not stable and there
    # are some oscillations
    changed_reg = []
    i = 0
    for i in range(maxiterations):
        changed = False
        for orig, transf, slize in valid_transforms:
            seq2 = replace_subseq(seq, orig, transf, slize)
            if not (seq2 is seq):
                changed = True
                seq = seq2
        changed_reg.append(changed)
        if changed_reg[-2:] == [False, False]:
            break
    return seq, i < maxiterations


def seq_contains(seq, subseq) -> Opt[Tup[int, int]]:
    """
    returns None if subseq is not contained in seq
    returns the (start, end) indices if seq contains subseq, so that
    
    Example::

        >>> seq, subseq = range(10), [3, 4, 5]
        >>> indices = seq_contains(seq, subseq)
        >>> assert seq[indices[0]:indices[1]] == subseq
    """
    for i in range(len(seq)-len(subseq)+1):
        for j in range(len(subseq)):
            if seq[i+j] != subseq[j]:
                break
        else:
            return i, i+len(subseq)
    return None


def pick_regularly(seq: Seq[T], numitems:int, start_idx:int=0, end_idx:int=0
                   ) -> List[T]:
    """
    Given a sequence, pick `numitems` from it at regular intervals
    The first and the last items are always included. The behaviour
    is similar to numpy's linspace

    Args:
        seq: a sequence of items
        numitems: the number of items to pick from seq
        start_idx: if given, the index to start picking from
        end_idx: if given, the index to stop

    Returns:
        a list of the picked items
    """
    if end_idx == 0:
        end_idx = len(seq)-1
    return [seq[round(i)] for i in np.linspace(start_idx, end_idx, numitems)]


def deepupdate(orig, updatewith):
    """
    recursively update orig with updatewith
    """
    for key, value in updatewith.items():
        if not isinstance(value, dict) or key not in orig:
            orig[key] = value
        else:
            orig[key] = deepupdate(orig[key], value)
    return orig


# ------------------------------------------------------------
#
#    Image and Pixels
#
# ------------------------------------------------------------


def fig2data(fig) -> np.ndarray:
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA
    channels and return it

    fig: a matplotlib figure
    returns a numpy 3D array of RGBA values
    """
    fig.canvas.draw()        # draw the renderer
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def pixels_to_cm(pixels: int, dpi=300) -> float:
    """
    convert a distance in pixels to cm

    pixels -> number of pixels
    dpi    -> dots (pixels) per inch
    """
    inches = pixels / dpi
    cm = inches * 2.54
    return cm


def cm_to_pixels(cm: float, dpi=300) -> float:
    """
    convert a distance in cm to pixels
    """
    inches = cm * 0.3937008
    pixels = inches * dpi
    return pixels


def inches_to_pixels(inches: float, dpi=300) -> float:
    return inches * dpi


def pixels_to_inches(pixels: int, dpi=300) -> float:
    return pixels / dpi


def page_dinsize_to_mm(pagesize: str, pagelayout: str) -> Tup[float, float]:
    """
    Convert a pagesize given as DIN size (A3, A4, ...) and page orientation
    into a tuple (height, width) in mm

    :param pagesize: size as DIN string (a3, a4, etc)
    :param pagelayout: portrait or landscape
    :return: a tuple (height, width) in mm
    """
    pagesize = pagesize.lower()
    if pagesize == 'a3':
        height, width = 420, 297
    elif pagesize == 'a4':
        height, width = 297, 210
    else:
        raise KeyError(f"pagesize {pagesize} not known")
    if pagelayout == 'landscape':
        height, width = width, height
    return height, width


# ------------------------------------------------------------
#
#    Decorators
#
# ------------------------------------------------------------


def returns_tuple(names, recname=None):
    """
    Decorator

    Modifies the function to return a namedtuple with the given names.

    Args:
        names: as passed to namedtuple, either a space-divided string,
            or a sequence of strings
        recname: a name to be given to the result as a whole. If nothing is
            given, the name of the decorated function is used.
    """
    from decorator import decorator
    from collections import defaultdict
    registry = {}
    if recname:
        nt = _namedtuple(recname, names)
        registry = defaultdict(lambda: nt)
    else:
        registry = {}

    @decorator
    def wrapper(func, *args, **kws):
        result = func(*args, **kws)
        nt = registry.get(func.__name__)
        if nt is None:
            nt = _namedtuple(func.__name__, names)
            registry[func.__name__] = nt
        return nt(*result)
    return wrapper


def returns_tuples(names, recname):
    """
    Decorator to modify a function returning a list of tuples to return
    a RecordList instead
    """
    from decorator import decorator
    from .containers import RecordList

    @decorator
    def wrapper(func, *args, **kws):
        result = func(*args, **kws)
        return RecordList(result, recname)
    return wrapper


def public(f):
    """
    Use a decorator to avoid retyping function/class names.

    Keeps __all__ updated

    * Based on an idea by Duncan Booth:
      http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
    * Improved via a suggestion by Dave Angel:
      http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1
    """
    publicapi = _sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ not in publicapi:  # Prevent duplicates if run from an IDE.
        publicapi.append(f.__name__)
    return f


def singleton(cls):
    """
    A decorator to create a singleton class

    @singleton
    class Logger(object):
        pass

    l = Logger()
    m = Logger()

    assert m is l
    """
    instances = {}

    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()


class runonce:
    """
    To be used as decorator. `func` will run only once, independent
    of the arguments passed the second time

    Example::

        # get_config() will only read the file the first time,
        # return the resulting dict for any further calls

        @runonce
        def get_config():
            config = json.load(open("/path/to/config.json"))
            return config

        config = get_config()

    """
    __slots__ = ('func', 'result', 'has_run')

    def __init__(self, func):
        self.func = func
        self.result = None
        self.has_run = False

    def __call__(self, *args, **kwargs):
        if self.has_run:
            return self.result

        self.result = self.func(*args, **kwargs)
        self.has_run = True
        return self.result


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def deprecated(func, msg=None):
    """
    To be used as

    oldname = deprecated(newname)

    """
    if msg is None:
        msg = f"Deprecated! use {func.__name__}"

    def wrapper(*args, **kws):

        warnings.warn(msg)
        return func(*args, **kws)

    return wrapper


def istype(obj, *types):
    """
    Examples
    --------

    >>> istype(1, int, float)
    True
    >>> istype((4, 0.5), (int, float))
    True
    >>> istype([4, 3], [int])
    True
    >>> istype({"foo": 4, "bar": 3}, {str:int})
    True

    """
    if len(types) > 1:
        return isinstance(obj, types)

    t = types[0]

    if isinstance(t, type):
        return isinstance(obj, t)

    if isinstance(t, tuple):
        return (isinstance(obj, tuple) and
                len(obj) == len(t) and
                all(istype(subobj, subt) for subobj, subt in zip(obj, t)))
    elif isinstance(t, list):
        if len(t) == 0:
            return isinstance(obj, list)
        elif len(t) == 1:
            subt = t[0]
            return isinstance(obj, list) and all(istype(subobj, subt)
                                                 for subobj in obj)
        else:
            raise ValueError(f"T should be [type], but found: {t}")
    elif isinstance(t, dict):
        assert len(t) == 1
        keyt, valt = list(t.items())[0]
        return all(istype(key, keyt) and istype(value, valt)
                   for key, value in obj.items())
    else:
        raise TypeError("T type not supported, see examples")


def assert_type(x, *types) -> bool:
    """
    Returns True if x is of the given type, raises TypeError otherwise
    See istype for examples

    Can be used as::

        assert_type(x, int, float)

    or::

        assert assert_type(x, [int])
    """
    if not istype(x, *types):
        raise TypeError(f"Expected type {types}, got {type(x).__name__}: {x}")
    return True


# --- crossplatform ---

def open_with_standard_app(path: str, force_wait=False) -> None:
    """
    Open path with the app defined to handle it by the user
    at the os level (xdg-open in linux, start in win, open in osx)

    This opens the default application in the background
    and returns immediately

    Returns a subprocess.Popen object. You can wait on this process
    by calling .wait on it. It can happen that the standard app is a server
    like application, where it is not possible to wait on it (like
    emacsclient or sublimetext). For those cases use force_wait=True,
    which opens a dialog that needs to be clicked in order to signal that
    editing is finished
    """
    import subprocess
    platform = _sys.platform
    if platform == 'linux':
        subprocess.call(["xdg-open", path])
    elif platform == "win32":
        # this function exists only in windows
        _os.startfile(path)
    elif platform == "darwin":
        subprocess.call(["open", path])
    else:
        raise RuntimeError(f"platform {platform} not supported")
    if force_wait:
        from emlib import dialogs
        dialogs.popupmsg("Close this dialog when finished")


def inside_jupyter() -> bool:
    """
    Are we running inside a jupyter notebook?
    """
    return session_type() == 'jupyter'


@runonce
def session_type() -> str:
    """
    Returns
        "jupyter" if running a jupyter notebook
        "ipython-terminal" if running ipython in a terminal
        "ipython" if running ipython outside a terminal
        "python" if a normal python

    NB: to check if we are inside an interactive session, check
    sys.flags.interactive == 1
    """
    try:
        # get_ipython should be available within an ipython/jupyter session
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return "jupyter"
        elif shell == 'TerminalInteractiveShell':
            return "ipython-terminal"
        else:
            return "ipython"
    except NameError:
        return "python"


def ipython_qt_eventloop_started() -> bool:
    """
    Are we running ipython / jupyter and the qt event loop has been started?
    ( %gui qt )
    """
    session = session_type()
    if session == 'ipython-terminal' or session == 'jupyter':
        # we are inside ipython so we can just call 'get_ipython'
        ip = get_ipython()
        return ip.active_eventloop == "qt"
    else:
        return False


def print_table(rows:list, headers=(), tablefmt:str=None, showindex=True) -> None:
    """
    Print rows as table
    
    Args:
        rows: a list of namedtuples or dataclass objects, all of the same kind 
        headers: override the headers defined in rows
        tablefmt: if None, a suitable default for the current situation will be used 
            (depending on if we are running inside jupyter or in a terminal, etc)
            Otherwise it is passed to tabulate.tabulate
        showindex: if True, add a column with the index of each row

    Returns:

    """
    if not rows:
        raise ValueError("rows is empty")

    row0 = rows[0]
    if dataclasses.is_dataclass(row0):
        if not headers:
            headers = [field.name for field in dataclasses.fields(row0)]
        rows = [dataclasses.astuple(row) for row in rows]
    elif isinstance(row0, tuple):
        if not headers:
            if not hasattr(row0, '_fields'):
                headers = [f"col{i}" for i in range(len(row0))]
                warnings.warn("The tuples passed don't seem to be namedtuples and no headers "
                              "provided. Using fallback headers")
            else:
                headers = row0._fields
    else:
        raise TypeError(f"rows should be a list of namedtuples or dataclass objects"
                        f", got {type(row0)}")

    import tabulate
    if inside_jupyter():
        from IPython.display import HTML, display
        if tablefmt is None:
            tablefmt = 'html'
        if tablefmt == 'html':
            html = tabulate.tabulate(rows, headers=headers, disable_numparse=True,
                                     tablefmt='html', showindex=showindex)
            display(HTML(html))
        else:
            print(tabulate.tabulate(rows, headers=headers, disable_numparse=True, 
                                    tablefmt=tablefmt, showindex=showindex))
    else:
        print(tabulate.tabulate(rows, headers=headers, showindex=showindex, tablefmt=tablefmt))


def replace_sigint_handler(handler):
    """
    Replace current SIGINT hanler with the given one, return
    the old one

    Args:
        handler: the new handler. A handler is a function taking no
        parameters and returning nothing

    Returns:
        the old handler

    """
    import signal
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    return original_handler


class temporary_sigint_handler:
    """
    Context manager to install a temporary sigint handler

    def handler():
        print("sigint detected!")

    with teporary_sigint_handler(handler):
        # Do something here, handler will be called if SIGINT (ctrl-c) is received
    """

    def __init__(self, handler):
        self.handler = handler
        self.original_handler = None

    def __enter__(self):
        self.original_handler = replace_sigint_handler(self.handler)

    def __exit__(self, type, value, traceback):
        replace_sigint_handler(self.original_handler)
        return True


def strip_lines(text: str) -> str:
    """
    Like .strip but for lines. Removes empty lines
    at the beginning or end of text, without touching
    lines in between
    """
    lines = text.splitlines()
    startidx, endidx = 0, 0

    for startidx, line in enumerate(lines):
        if line.strip():
            break

    for endidx, line in enumerate(reversed(lines)):
        if line.strip():
            break

    return "\n".join(lines[startidx:len(lines)-endidx])


def optimize_parameter(func, val:float, paraminit:float, maxerror=0.001) -> float:
    """
    Optimize one parameter to arrive to a desired value.

    Example:

        # find the exponent of a bpf were its value at 0.1 is 1.25
        (within the given relative error)

        expon = findparam(evalparam=lambda param: bpf.expon(0, 1, 1, 6, exp=param)(0.1),
                          val=1.25, paraminit=2)
        val = bpf.expon(0, 1, 1, 6, exp=expon)(0.1)
        print(val)

    Args:
        func: a function returning a value which will be compared to `val`
        val: the desired value to arrive to
        paraminit: the initial value of param
        maxerror: the max. relative error (0.001 is 0.1%)
    """
    param = paraminit
    while True:
        valnow = func(param)
        relerror = abs(valnow - val) / valnow
        if relerror < maxerror:
            return valnow
        if valnow > val:
            param = param * (1+relerror)
        else:
            param = param * (1-relerror)


#  ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#                             END
#  ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

if __name__ == '__main__':
    import doctest
    doctest.testmod()
