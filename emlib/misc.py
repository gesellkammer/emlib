"""

Miscellaneous functionality

* **Search**: `nearest_element`, `nearest_unsorted`, `nearest_index`
* **Sort**: `sort_natural`, `zipsort`, `issorted`
* **Namedtuples**: `namedtupled_addcolumn`, `namedtuple_extend`, etc.
* **Open files**: `open_with_standard_app`, `wait_for_file_modified`, `open_with`
* **Unit conversions**: `cm_to_pixels`, `page_dinsize_to_mm`, etc.
* **Other**: `singleton`

"""
# -*- coding: utf-8 -*-
from __future__ import annotations
import os as _os
import sys as _sys
from bisect import bisect as _bisect
import re as _re

import numpy as np


from typing import TYPE_CHECKING
if TYPE_CHECKING or 'sphinx' in _sys.modules:
    from typing import TypeVar, Sequence, Union, Callable, Any, Iterable
    T = TypeVar("T")
    T2 = TypeVar("T2")
    from fractions import Fraction
    import numbers
    num_t =  TypeVar("num_t", int, float, numbers.Rational)
    number_t = Union[float, numbers.Rational]


# ------------------------------------------------------------
#     CHUNKS
# ------------------------------------------------------------


def reverse_recursive(seq: list):
    """
    Reverse seq recursively

    Args:
        seq: a (possibly nested) list of elements

    Returns:
        a reversed version of `seq` where all sub-lists are also reversed.


    Example
    ~~~~~~~

        >>> reverse_recursive([1, 2, [3, 4], [5, [6, 7]]])
        [[[7, 6], 5], [4, 3], 2, 1]

    .. note:: only lists will be reversed, other iterable collection remain untouched

    """
    out = []
    for x in seq:
        if isinstance(x, list):
            x = reverse_recursive(x)
        out.append(x)
    out.reverse()
    return out


def _partialsum(seq: Sequence[num_t], init: num_t) -> list[num_t]:
    accum: num_t = init
    out = []
    for i in seq:
        accum += i
        out.append(accum)
    return out


def wrap_by_sizes(flatseq: list, packsizes: Sequence[int]) -> list[list]:
    """
    Wrap a flat seq using the given sizes

    Args:
        flatseq: a flat sequence of items
        packsizes: a list of sizes

    Returns:
        a list of groups, where each group is of size as given by packsizes

    Example
    ~~~~~~~

        >>> flatseq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> wrap_by_sizes(flatseq, [3, 5, 2])
        [[1, 2, 3], [4, 5, 6, 7, 8], [9, 10]]

    """
    offsets = [0] + _partialsum(packsizes, 0)
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


def nearest_element(item: float, seq: list[float] | np.ndarray) -> float:
    """
    Find the nearest element (the element, not the index) in seq

    **NB**: assumes that seq is sorted (this is not checked). seq can also be a
        numpy array, in which case searchsorted is used instead of bisect

    Args:
        item: the item to search
        seq: either a list of numbers, or a numpy array

    Returns:
        the value of the nearest element of seq

    Example
    ~~~~~~~

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
        ir = int(seq.searchsorted(item, 'right'))
    else:
        ir = _bisect(seq, item)
    element_r = seq[ir]
    element_l = seq[ir - 1]
    if abs(element_r - item) < abs(element_l - item):
        return element_r
    return element_l


def nearest_unsorted(x: num_t, seq: list[num_t]) -> num_t:
    """
    Find nearest item in an unsorted sequence

    Args:
        x: a number
        seq: a seq. of numbers (assumes it is not sorted)

    Returns:
        the item in seq. which is nearest to x

    .. note:: for sorted seq. use :func:`nearest_index`

    Example
    ~~~~~~~

        >>> assert nearest_unsorted(3.6, (1,2,3,4,5)) == 4
        >>> assert nearest_unsorted(4, (2,3,4)) == 4
        >>> assert nearest_unsorted(200, (3,5,20)) == 20
    """
    return min((abs(x - y), y) for y in seq)[1]


def nearest_index(item: num_t, seq: Sequence[num_t]) -> int:
    """
    Return the index of the nearest element in seq to item

    Args:
        item: a number
        seq: a sorted sequence of numbers

    Returns:
        the index of the nearest item

    .. note:: Assumes that seq is sorted

    Example
    ~~~~~~~

        >>> seq = [0, 3, 4, 8]
        >>> nearest_index(3.1, seq)
        1
        >>> nearest_index(6.5, seq)
        3

    .. seealso:: :func:`nearest_unsorted`
    """
    ir = _bisect(seq, item)
    seqlen = len(seq)
    if ir == seqlen:
        return seqlen - 1
    if ir == 0:
        return ir
    il = ir - 1
    return ir if seq[ir] - item < item - seq[il] else il


# ------------------------------------------------------------
#
#    SORTING
#
# ------------------------------------------------------------


def sort_natural(seq: list, key: Callable[[Any], str]=None) -> list:
    """
    sort a string sequence naturally

    Sorts the sequence so that 'item1' and 'item2' are before 'item10'

    Args:
        seq: the sequence to sort
        key: a function to convert an item in seq to a string

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
    """

    def convert(text: str):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key:str):
        return [convert(c) for c in _re.split('([0-9]+)', key)]

    if key is not None:
        return sorted(seq, key=lambda x: alphanum_key(key(x)))
    return sorted(seq, key=alphanum_key)


def sort_natural_dict(d: dict[str, Any], recursive=True) -> dict:
    """
    sort dict d naturally and recursively
    """
    rows: list[tuple[str, Any]] = []
    if recursive:
        for key, value in d.items():
            if isinstance(value, dict):
                value = sort_natural_dict(value, recursive=recursive)
            rows.append((key, value))
        sorted_rows = sort_natural(rows, key=lambda row: row[0])
    else:
        keys = list(d.keys())
        sorted_rows = [(key, d[key]) for key in sort_natural(keys)]
    return dict(sorted_rows)


def issorted(seq: Sequence, key=None) -> bool:
    """
    Returns True if seq is sorted

    Args:
        seq: the seq. to query
        key: an optional key to use

    Example
    ~~~~~~~

        >>> seq = [(10, "a"), (0, "b"), (45, None)]
        >>> issorted(seq, key=lambda item:item[0])
        False

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


def some(x, otherwise=False):
    """
    Returns ``x`` if it is not None, else ``otherwise``

    This allows code like::

        myvar = some(myvar) or default
        # If default does not need to be shortcircuited, then simply:
        myvar = some(myvar, default)

    instead of::

        myvar = myvar if myvar is not None else default
    """
    return x if x is not None else otherwise


def firstval(*values, sentinel=None):
    """
    Get the first value in values which is not sentinel.

    At least one of the values should differ from sentinel, otherwise
    an exception is raised. To allow short-cirtcuit lazy evaluation,
    a callable can be given as value, in which case the function
    will only be evaluated if the previous values where `sentinel`

    .. seealso:: :func:`some`


    Example
    ~~~~~~~

    .. code-block:: python

        config = {'a': 10, 'b': 20}
        def func(a=None, b=None):
            a = firstval(a, lambda: computation(), config['a'])

    """
    for value in values:
        if callable(value):
            value2 = value()
            if value2 is not sentinel:
                return value2
        elif value is not sentinel:
            return value
    raise ValueError(f"All values are {sentinel}")


def zipsort(a: Sequence[T], b: Sequence[T2], key: Callable | None = None, reverse=False
            ) -> tuple[list[T], list[T2]]:
    """
    Sort a and keep b in sync

    It is the equivalent of::

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
    return (list(a), list(b))


def duplicates(seq: Sequence[T], mincount=2) -> list[T]:
    """
    Find all elements in seq which are present at least `mincount` times
    """
    if mincount == 2:
        from . import iterlib
        return list(iterlib.duplicates(seq))

    from collections import Counter
    counter = Counter(seq).items()
    return [item for item, count in counter if count >= mincount]


def remove_duplicates(seq: Sequence[T]) -> list[T]:
    """
    Remove all duplicates in seq while keeping its order
    If order is not important, use list(set(seq))

    Args:
        seq: a list of elements (elements must be hashable)

    Returns:
        a new list with all the unique elements of seq in its
        original order

    .. note:: list(set(...)) does not keep order
    """
    # we use the fact that dicts keep order:
    return list(dict.fromkeys(seq))


def remove_last_matching(seq: list[T], func: Callable[[T], bool]) -> T | None:
    """
    Remove last element of *seq* matching the given condition, **in place**

    Args:
        seq: the list to modify
        func: a function taking an element of *seq*, should return True if
            this is the element to remove

    Returns:
        the removed element, or None if the condition was never met

    Example
    ~~~~~~~

        >>> a = [0, 1, 2, 3, 4, 5, 6]
        >>> remove_last_matching(a, lambda item: item % 2 == 1)
        >>> a
        [0, 1, 2, 3, 4, 6]
    """
    seqlen = len(seq)
    for i, x in enumerate(reversed(seq)):
        if func(x):
            return seq.pop(seqlen - i - 1)
    return None


def fractional_slice(seq: Sequence[T], step:float, start=0, end=-1) -> list[T]:
    """
    Take a slice similar to seq[start:end:step] with fractional step

    Args:
        seq: the sequence of elements
        step: the step size
        start: start index
        end: end index

    Returns:
        the resulting list


    Example
    ~~~~~~~

        >>> fractional_slice(range(10), 1.5)
        >>> [0, 2, 3, 5, 6, 8]

    """
    if step < 1:
        raise ValueError("step should be >= 1 (for now)")

    accum = 0.
    out = [seq[start]]
    for elem in seq[start+1:end]:
        accum += 1
        if accum >= step:
            out.append(elem)
            accum -= step
    return out


def sec2str(seconds:float, msdigits=3) -> str:
    """
    Convert seconds to a suitable string representation

    Args:
        seconds: time in seconds

    Returns:
        the equivalent time as string

    """
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    s = seconds % 60
    sint = int(s)
    sfrac = round((s - sint), msdigits)
    fmt = f"%.{msdigits}g"
    msstr = (fmt % sfrac)[1:2+msdigits]
    return f"{h}:{m:02}:{sint:02}{msstr}" if h > 0 else f"{m}:{sint:02}{msstr}"


def parse_time(t: str) -> float:
    """
    Parse a time string ``HH:MM:SS.mmm`` and convert it to seconds

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


# ------------------------------------------------------------
#
#     namedtuple utilities
#
# ------------------------------------------------------------


def namedtuple_addcolumn(namedtuples, seq, column_name: str, classname=""):  # type: ignore
    """
    Add a column to a sequence of named tuples

    Args:
        namedtuples: a list of namedtuples
        seq: the new column
        column_name: name of the column
        classname: nane of the new namedtuple

    Returns:
        a list of namedtuples with the added column
    """
    from collections import namedtuple
    t0 = namedtuples[0]
    assert isinstance(t0, tuple) and hasattr(t0, "_fields"), "namedtuples should be a seq. of namedtuples"
    name = classname or namedtuples[0].__class__.__name__ + '_' + column_name  # type: str
    NewTup = namedtuple(name, t0._fields + (column_name,))  # type: ignore
    newtuples = [NewTup(*(t + (value,)))
                 for t, value in zip(namedtuples, seq)]
    return newtuples


def namedtuples_renamecolumn(namedtuples: list, oldname: str, newname: str, classname=''
                             ) -> list:  # type: ignore
    """
    Rename the column of a seq of namedtuples

    Args:
        namedtuples: a list of namedtuples
        oldname: the name of the column which will be modified
        newname: the new name of the column
        classname: the name of the new namedtuple class

    Returns:
        the new namedtuples with the renamed column

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
    from collections import namedtuple
    NewTup = namedtuple(classname, newfields)
    newtuples = [NewTup(*t) for t in namedtuples]
    return newtuples


def namedtuple_extend(name: str, orig, columns: str | Sequence[str]):
    """
    Create a new namedtuple constructor with the added columns

    It returns the class constructor and an ad-hoc
    constructor which takes as arguments an instance
    of the original namedtuple and the additional args

    Args:
        name: new name for the type
        orig: an instance of the original namedtuple or the constructor itself
        columns : the columns to add

    Returns:
        a tuple (newtype, newtype_from_old)

    Example
    ~~~~~~~

        >>> from collections import namedtuple
        >>> Point = namedtuple("Point", "x y")
        >>> p = Point(10, 20)
        >>> Vec3, fromPoint = namedtuple_extend("Vec3", Point, "z")
        >>> Vec3(1, 2, 3)
        Vec3(x=1, y=2, z=3)
        >>> fromPoint(p, 30)
        Vec3(x=10, y=20, z=30)

    """
    from collections import namedtuple

    if isinstance(columns, str):
        columns = columns.split()
    fields = orig._fields + tuple(columns)
    N = namedtuple(name, fields)

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


def isiterable(obj, exceptions: tuple[type, ...]=(str, bytes)) -> bool:
    """
    Is `obj` iterable?

    Example
    ~~~~~~~

        >>> isiterable([1, 2, 3])
        True
        >>> isiterable("test")
        False
    """
    return hasattr(obj, '__iter__') and not isinstance(obj, exceptions)


def isgeneratorlike(obj):
    "Does ``obj`` behave like a generator? (it can be iterated but has no length)"
    return hasattr(obj, '__iter__') and not hasattr(obj, '__len__')


def asnumber(obj, accept_fractions=True, accept_expon=False
             ) -> int | float | Fraction | None:
    """
    Convert ``obj`` to a number or None if it cannot be converted


    Example
    ~~~~~~~

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
            from fractions import Fraction
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


def astype(type_, obj=None, factory=None):
    """
    Return obj as type.

    If obj is already of said type, obj itself is returned
    Otherwise, obj is converted to type. If a special contructor is needed,
    it can be given as `construct`. If no obj is passed, a partial function
    is returned which can check for that particular type

    Args:
        type_: the type the object should have
        obj: the object to be checkec/converted
        factory: if given, a function ``(obj) -> obj`` of type ``type_``

    Example
    ~~~~~~~

        >>> astype(list, (3, 4))
        [3, 4]
        >>> l = [3, 4]
        >>> astype(list, l) is l
        True
        >>> aslist = astype(list)
    """
    factory = factory or type_
    if obj is None:
        return lambda obj: obj if isinstance(obj, type_) else factory(obj)
    return obj if isinstance(obj, type_) else factory(obj)


def str_is_number(s: str, accept_exp=False, accept_fractions=False) -> bool:
    """
    Returns True if the given string represents a number

    Args:
        s: the string to inspect
        accept_exp: accept exponential notation
        accept_fractions: accept numbers of the form "3/4"

    Returns:
        True if s represents a number, False otherwise

    .. note::

        fractions should have the form num/den, like 3/4, with no spaces in between
    """
    if accept_exp and accept_fractions:
        return asnumber(s) is not None
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

    Example
    ~~~~~~~

        >>> a, b = {'A': 1, 'B': 2}, {'B': 20, 'C': 30}
        >>> dictmerge(a, b) == {'A': 1, 'B': 20, 'C': 30}
        True
    """
    import warnings
    warnings.warn("Deprecated, use dict1 | dict2")
    return dict1 | dict2


def moses(pred: Callable[[T], bool], seq: Iterable[T]
          ) -> tuple[list[T], list[T]]:
    """
    Divides *seq* into two lists: filter(pred, seq), filter(not pred, seq)

    Args:
        pred: a function predicate
        seq: the seq. to divide

    Returns:
        a tuple ``(true_elements, false_elements)``, where true_elements contains
        the items in *seq* for which *pred* evaluates to true, and *false_elements*
        contains the rest

    Example::

        >>> moses(lambda x:x > 5, range(10))
        ([6, 7, 8, 9], [0, 1, 2, 3, 4, 5])
    """
    trueseq, falseseq = [], []
    for x in seq:
        (trueseq if pred(x) else falseseq).append(x)
    return trueseq, falseseq


def allequal(xs: Sequence) -> bool:
    """
    Return True if all elements in xs are equal

    Args:
        xs: the seq. to query

    Returns:
        True if all elements in xs are equal
    """
    x0 = xs[0]
    return all(x==x0 for x in xs)


def dumpobj(obj) -> list[tuple[str, Any]]:
    """
    Return all 'public' attributes of this object
    """
    return [(item, getattr(obj, item))
            for item in dir(obj)
            if not item.startswith('__')]


def can_be_pickled(obj) -> bool:
    """
    Return True if obj can be pickled
    """
    import pickle
    try:
        obj2 = pickle.loads(pickle.dumps(obj))
    except pickle.PicklingError:
        return False
    return obj == obj2


def snap_to_grid(x: num_t, tick: num_t, offset: num_t = 0, nearest=True
                 ) -> num_t:
    """
    Find the nearest slot in a grid

    Given a grid defined by offset + tick * N, find the nearest element
    of that grid to a given x

    Args:
        x: the number to snap to the grid
        tick: distance between ticks of the grid
        offset: offset of the grid
        nearest: if True, snap to the nearest tick (the nearest
            of the next floor or ceil tick), otherwise to the
            floor tick

    Returns:
        the tick to which to snap x to


    .. note::

        the result will have the same type as *x*, so if *x* is float,
        the result will be float, if it is a Fraction, then the
        result will be a fraction

    Example
    ~~~~~~~

        >>> snap_to_grid(1.6, 0.5)
        1.5
        >>> from fractions import Fraction
        >>> snap_to_grid(Fraction(2, 3), Fraction(1, 5))
        Fraction(3, 5)

    """
    t = x.__class__
    if nearest:
        return t(round((x - offset) / tick)) * tick + offset
    else:
        return t(int((x - offset) / tick)) * tick + offset


def snap_array(X: np.ndarray,
               tick: float,
               offset: float = 0.,
               out: np.ndarray | None = None,
               nearest=True
               ) -> np.ndarray:
    """
    Snap the values of X to the nearest slot in a grid

    Assuming a grid t defined by ``t(n) = offset + tick*n``, snap the values of X
    to the nearest value of t

    Args:
        X: an array
        tick: the step value of the grid
        offset: the offset of the grid
        out: if given, snapped values are placed in this array
        nearest: if True, the nearest slot is selected. Otherwise the next lower
            (floor)

    Returns:
        an array containing the snapped values. This array will be *out* if it
        was given
    """
    if tick <= 0:
        raise ValueError("tick should be > 0")

    if nearest:
        return _snap_array_nearest(X, tick, offset=float(offset), out=out)
    return _snap_array_floor(X, tick, offset=float(offset), out=out)


def _snap_array_nearest(X: np.ndarray,
                        tick: number_t,
                        offset: number_t = 0.,
                        out: np.ndarray | None = None
                        ) -> np.ndarray:
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


def _snap_array_floor(X: np.ndarray, tick:float, offset=0., out: np.ndarray=None
                      ) -> np.ndarray:
    arr = out if out is not None else X.copy()
    if offset != 0:

        arr -= offset
        arr /= tick
        arr = np.floor(arr, out=arr)
        arr *= tick
        arr += offset
    else:
        arr /= tick
        arr = np.floor(arr, out=arr)
        arr *= tick
    return arr


def distribute_in_zones(x: num_t, split_points: Sequence[num_t], side="left") -> int:
    """
    Returns the index of a "zone" where to place x.

    A zone is a numeric range defined by an inclusive lower boundary and a
    non-inclusive higher boundary

    **NB**: see :func:`distribute_in_zones_right` for a non-inclusive lower and
    inclusive upper boundary. The edge zones extend to inf.

    Args:
        x: the number to assign a zone to
        split_points: the split points which define the zones
        side: if "left", a zone has an inclusive lower bound and a non-inclusive
            upper bound. "right" is the opposite

    Returns:
        the index of the zone

    Example::

        # 1 and 5 define three zones: (-inf, 1], (1, 5], (5, inf)
        >>> distribute_in_zones(2, [1, 5])
        1
        >>> distribute_in_zones(5, [1, 5])
        2

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


def _distribute_in_zones_right(x: num_t, split_points: Sequence[num_t]) -> int:
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


def seq_contains(seq: Sequence[T], subseq: Sequence[T]) -> tuple[int, int] | None:
    """
    Returns the (start, end) indexes if seq contains subseq, or None

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


def deepupdate(orig: dict, updatewith: dict) -> dict:
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



def pixels_to_cm(pixels: int, dpi=300) -> float:
    """
    Convert a distance in pixels to cm

    Args:
        pixels: number of pixels
        dpi: dots (pixels) per inch

    Returns:
        the corresponding value in cm
    """
    inches = pixels / dpi
    cm = inches * 2.54
    return cm


def cm_to_pixels(cm: float, dpi=300) -> float:
    """
    convert a distance in cm to pixels

    Args:
        cm: a value in cm
        dpi: dots-per-inch

    Returns:
        the corresponding value in pixels
    """
    inches = cm * 0.3937008
    pixels = inches * dpi
    return pixels


def inches_to_pixels(inches: float, dpi=300) -> float:
    """ Convert inches to pixels """
    return inches * dpi


def pixels_to_inches(pixels: int, dpi=300) -> float:
    """Convert pixels to inches"""
    return pixels / dpi


def page_dinsize_to_mm(pagesize: str, pagelayout: str) -> tuple[float, float]:
    """
    Return the (height, width) for a given DIN size and page orientation

    Args:
        pagesize: size as DIN string (a3, a4, etc)
        pagelayout: portrait or landscape

    Returns:
        a tuple (height, width) in mm


    ========== =================================
    Format     Width x Heigh (mm)
    ========== =================================
    A0         841 x 1189
    A1         594 x 841
    A2         420 x 594
    A3         297 x 420
    A4         210 x 297
    A5         148 x 210
    A6         105 x 148
    A7         74 x 105
    ========== =================================

    """
    pagesizes = {
        'a0': (1189, 841),
        'a1': (841, 594),
        'a2': (594, 420),
        'a3': (420, 297),
        'a4': (297, 210),
        'a5': (210, 148),
        'a6': (148, 105),
        'a7': (105, 74)
    }
    w, h = pagesizes.get(pagesize.lower(), (0, 0))
    if not w:
        raise ValueError(f"pagesize {pagesize} not known. Supported sizes: {pagesizes.keys()}")
    if pagelayout == 'portrait':
        w, h = h, w
    return h, w


# ------------------------------------------------------------
#
#    Decorators
#
# ------------------------------------------------------------
#

def public(f):
    """
    decorator - keeps __all__ updated

    **NB**: it has no performance penalty at runtime since the decorator
    just returns the passed function

    * Based on an idea by Duncan Booth:
      http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
    * Improved via a suggestion by Dave Angel:
      http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1
    """
    publicapi = _sys.modules[f.__module__].__dict__.setdefault('__all__', [])
    if f.__name__ not in publicapi:  # Prevent duplicates if run from an IDE.
        publicapi.append(f.__name__)
    return f


def singleton(cls: type):
    """
    A class decorator to create a singleton class

    Example
    -------

    ::

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def type_error_msg(x, *expected_types):
    """
    To be used when raising a TypeError

    Example::

        if isinstance(x, int):
            ...
        else:
            raise TypeError(type_error_msg(x, int))

        # This will raise a TypeError with the message
        # 'Expected type (int,), got str: "foo"'

    """
    return f"Expected type {expected_types}, got {type(x).__name__}: {x}"


# --- crossplatform ---

def _open_with_standard_app(path: str, wait: str | bool = False, min_wait=0.5,
                            timeout=0.
                            ) -> None:
    """
    Open path with the app defined to handle it at the os level

    Uses *xdg-open* in linux, *start* in win and *open* in osx.

    Args:
        path: the file to open
        wait: if True, we wait until the app has returned. This is in many cases
            not possible. If the app returns right away a dialog is created
            to make waiting explicit until the user confirms this dialog.
            Alternatively wait can be passed the string "modified", in which case
            we wait until the given file is modified
        min_wait: min. wait time. when waiting on app being closed. If the app
            closes before this time, a dialog appears asking for confirmation.
        timeout: a timeout for waiting on modified

    """
    import subprocess
    import time
    proc = None
    if _sys.platform == 'linux':
        proc = subprocess.Popen(["xdg-open", path])
    elif _sys.platform == "win32":
        # this function exists only in windows
        _os.startfile(path)  # type: ignore
    elif _sys.platform == "darwin":
        proc = subprocess.Popen(["open", path])
        min_wait = max(min_wait, 1)
    else:
        raise RuntimeError(f"platform {_sys.platform} not supported")

    if wait == "modified":
        wait_for_file_modified(path, timeout=timeout or 36000)
    elif wait:
        from emlib import dialogs
        if _sys.platform == "win32":
            dialogs.showInfo("Close this dialog when finished")
        elif proc is not None:
            t0 = time.time()
            proc.wait()
            if time.time() - t0 < min_wait:
                dialogs.showInfo("Close this dialog when finished")


def _split_command(s:str) -> list[str]:
    parts = s.split()
    parts = [p.replace('"', '') for p in parts]
    return parts


def open_with_app(path: str,
                  app: str | list[str] | None = None,
                  wait: bool | str = False,
                  shell=False,
                  min_wait=0.5,
                  timeout=0.) -> None:
    """
    Open a given file with a given app.

    It can either wait on the app to exit or wait until the file
    was modified. The app can be either a command as a string or a
    list of string arguments passed to *subprocess.Popen*

    Args:
        path: the path to the file to open
        app: a command-line string or a list of string arguments. If no app is given,
            we ask the os to open this file with its standard app
        wait: if True, wait until the app stops. If the app is a daemon
            app (it returns immediately), this situation
            is detected and a dialog is created which needs to be
            clicked in order for the function to return. Alternatively, wait can be
            "modified", in which case we wait until ``path`` has been modified; or
            "dialog", where a confirmation dialog is open for the user to signal
            when the editing is done
        shell: should app be started from a shell?
        min_wait: if the application returns before this time a wait
            dialog is created
        timeout: a timeout for wait_on_modified
    """
    if not app:
        assert not shell
        _open_with_standard_app(path, wait=wait, min_wait=min_wait, timeout=timeout)
        return

    import subprocess
    import time

    if shell:
        assert isinstance(app, str), "shell needs a command-line as string"
        proc = subprocess.Popen(f'{app} "{path}"', shell=True)
    else:
        args = app if isinstance(app, list) else app.split()
        args.append(path)
        proc = subprocess.Popen(args)
    t0 = time.time()
    if wait == "modified":
        wait_for_file_modified(path, timeout=timeout)
    elif wait == 'dialog':
        proc.wait()
        from emlib import dialogs
        dialogs.showInfo("Close this dialog when finished")

    elif wait:
        proc.wait()
        if time.time() - t0 < min_wait:
            from emlib import dialogs
            dialogs.showInfo("Close this dialog when finished")


def wait_for_file_modified(path: str, timeout: int | float = 0.) -> bool:
    """
    Wait until file is modified.

    This is useful when editing a file on an external application
    which runs in a daemon mode, meaning that opening a file in it
    might return immediately.

    Args:
        path: the path of the file to monitor
        timeout: how long should we wait for, in seconds

    Returns:
        True if the file was modified, False if it wasn't or if the operation
        timed-out
    """
    from watchdog.observers import Observer
    from watchdog.events import PatternMatchingEventHandler
    directory, base = _os.path.split(path)
    if not directory:
        directory = "."
    handler = PatternMatchingEventHandler(patterns=[base], ignore_directories=True, case_sensitive=True)
    observer = Observer()
    modified = False

    def on_modified(event):
        nonlocal modified
        modified = True
        observer.stop()

    handler.on_modified = on_modified
    observer.schedule(handler, path=directory, recursive=False)
    observer.start()
    if timeout is None:
        timeout = 360000  # 100 hours
    observer.join(timeout)
    return modified


def first_existing_path(*paths: str) -> str | None:
    """
    Returns the first path in paths which exists

    Args:
        *paths: the paths to test
        default: a default path returned when all other paths do not exist. It is
            not checked that this default path exists.

    Returns:
        the first existing path within the values given, None if no
        match was found

    """
    for p in paths:
        p = _os.path.expanduser(p)
        if _os.path.exists(p):
            return p
    return None


def html_table(rows: list,
               headers: list[str],
               maxwidths: list[int] | None = None,
               rowstyles: list[str] | None = None,
               tablestyle='',
               headerstyle=''
               ) -> str:
    """
    Create a html table

    Args:
        rows: the rows of the table, where each row is a sequence of cells
        headers: a list of column names
        maxwidths: if given, a list of max widths for each column
        rowstyles: if given, a list of styles, one for each column
        tablestyle: a style applied to the entire table
        headerstyle: a style applied to the table header

    Returns:
        a string with the generated HTML
    """
    parts = []
    _ = parts.append
    if tablestyle:
        _(f'<table style="{tablestyle}"')
    else:
        _("<table>")
    if headerstyle:
        _(f'<thead style="{headerstyle}"')
    else:
        _("<thead>")
    _("<tr>")
    if maxwidths is None:
        maxwidths = [0] * len(headers)
    if rowstyles is None:
        rowstyles = [''] * len(headers)
    for colname in headers:
        _(f'<th style="text-align:left">{colname}</th>')
    _("</tr></thead><tbody>")
    for row in rows:
        _("<tr>")
        for cell, maxwidth, rowstyle in zip(row, maxwidths, rowstyles):
            if rowstyle:
                cell = f'<span style="{rowstyle}">{cell}</span>'
            if maxwidth > 0:
                _(f'<td style="text-align:left;max-width:{maxwidth}px;">{cell}</td>')
            else:
                _(f'<td style="text-align:left">{cell}</td>')
        _("</tr>")
    _("</tbody></table>")
    return "".join(parts)


def print_table(rows: list, headers=(), tablefmt='', showindex=True, floatfmt: str|tuple[str, ...]='g') -> None:
    """
    Print rows as table

    Args:
        rows: a list of namedtuples or dataclass objects, all of the same kind
        headers: override the headers defined in rows
        tablefmt: if None, a suitable default for the current situation will be used
            (depending on if we are running inside jupyter or in a terminal, etc)
            Otherwise it is passed to tabulate.tabulate
        floatfmt: a format for all floats or a tuple of formats
        showindex: if True, add a column with the index of each row

    """
    if not rows:
        raise ValueError("rows is empty")

    import dataclasses
    row0 = rows[0]
    if dataclasses.is_dataclass(row0):
        if not headers:
            headers = [field.name for field in dataclasses.fields(row0)]
        rows = [dataclasses.astuple(row) for row in rows]
    elif isinstance(row0, (tuple, list)):
        if not headers:
            fields = getattr(row0, '_fields', None)
            headers = fields or [f"col{i}" for i in range(len(row0))]

    else:
        raise TypeError(f"rows should be a list of tuples, namedtuples or dataclass objects"
                        f", got {type(row0)}")

    import tabulate
    from .envir import inside_jupyter
    if inside_jupyter():
        from IPython.display import HTML, display
        if not tablefmt:
            tablefmt = 'html'
        disable_numparse = not floatfmt
        s = tabulate.tabulate(rows, headers=headers, disable_numparse=disable_numparse,
                              tablefmt=tablefmt, showindex=showindex, stralign='left',
                              floatfmt=floatfmt)
        if tablefmt == 'html':
            display(HTML(s))
        else:
            print(s)
    else:
        print(tabulate.tabulate(rows, headers=headers, showindex=showindex, tablefmt=tablefmt,
                                floatfmt=floatfmt))


def replace_sigint_handler(handler: Callable[[None], None]):
    """
    Replace current SIGINT hanler with the given one, return the old one

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

    Example::

        >>> def handler():
        ...    print("sigint detected!")

        >>> with teporary_sigint_handler(handler):
        ...    # Do something here, handler will be called if SIGINT (ctrl-c) is received
    """

    def __init__(self, handler):
        self.handler = handler
        self.original_handler = None

    def __enter__(self):
        self.original_handler = replace_sigint_handler(self.handler)

    def __exit__(self, type, value, traceback):
        replace_sigint_handler(self.original_handler)
        return True


def simplify_breakpoints(bps: list[T],
                         coordsfunc: Callable,
                         tolerance= 0.01
                         ) -> list[T]:
    """
    Simplify breakpoints in a breakpoint function

    Assuming a list of some objects building a multisegmented line
    in 2D, simplify this line by eliminating superfluous breakpoints
    which don't contribute (enough) to the resolution of this line

    Args:
        bps: a list of breakpoints
        coordsfunc: a function of the form (breakpoint) -> (x, y)
        tolerance: if the difference between two consecutive slopes is below this threshold
            we assume that the two lines are colinear and we don't need the middle point

    Returns:
        the list of simplified breakpoints. The first and last breakpoints of the original
        will always be part of the result

    Example::

        >>> @dataclasses.dataclass
        ... class Point:
        ...     name: str
        ...     x: float
        ...     y: float

        >>> points = [Point("A", 0, 0),
        ...           Point("B", 2, 0),
        ...           Point("C", 3, 0),
        ...           Point("D", 4, 1),
        ...           Point("E", 5, 2)]
        >>> simplify_breakpoints(points, coordsfunc=(lambda p: p.x, p.y))
        [Point(name="A", x=0, y=0), Point(name="C", x=3, y=0), Point(name="E", x=5, y=2)]
    """
    if len(bps) <= 3:
        return bps

    def colinear(A, B, C, tolerance=0.01):
        Ax, Ay = coordsfunc(A)
        Bx, By = coordsfunc(B)
        Cx, Cy = coordsfunc(C)
        slopeAB = (By - Ay) / (Bx - Ax)
        slopeBC = (Cy - By) / (Cx - Bx)
        return abs(slopeAB-slopeBC) < tolerance

    A = bps[0]
    B = bps[1]
    simplified = [A]

    for C in bps[2:]:
        if not colinear(A, B, C, tolerance=tolerance):
            simplified.append(B)
            A = B
        B = C

    simplified.append(bps[-1])
    return simplified


def rgb_to_hex(r: int, g: int, b: int) -> str:
    "Convert a color in rgb to its hex representation"
    return '#%02x%02x%02x'% (r, g, b)


_attrs_by_class: dict[type, list[str]] = {}


def find_attrs(obj, excludeprefix='_') -> list[str]:
    """
    Iterate over all attributes of objects.

    Args:
        obj: the object to query
        excludeprefix: attributes starting with this prefix will be excluded

    Returns:
        a list of all the attibutes (instance variables) of this object. Notice
        that results are cached by class so if an object has dynamic attributes
        these will not be detected


    .. note::
        This function will only return attributes, no methods,
        class variables, staticmethods, etc.

    Example
    -------

        >>> class Foo:
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        ...
        >>> class Bar(Foo):
        ...     def __init__(self, c):
        ...         super().__init__(1, 2)
        ...         self.c = c
        ...
        >>> bar = Bar(3)
        >>> find_attrs(bar)
        ['a', 'b', 'c']
    """
    cls = type(obj)
    if attrs := _attrs_by_class.get(cls):
        return attrs
    attrs = _find_attrs(obj, excludeprefix=excludeprefix)
    _attrs_by_class[cls] = attrs
    return attrs


def _find_attrs(obj, excludeprefix='_') -> list[str]:
    import inspect
    visited = set()
    out = []
    if hasattr(obj, "__dict__"):
        for attr in sorted(obj.__dict__):
            if attr not in visited:
                if not attr.startswith(excludeprefix):
                    out.append(attr)
                visited.add(attr)

    for cls in reversed(inspect.getmro(obj.__class__)):
        if hasattr(cls, "__slots__"):
            for attr in cls.__slots__:
                if hasattr(obj, attr) and attr not in visited:
                    if not attr.startswith(excludeprefix):
                        out.append(attr)
                    visited.add(attr)

    return out


class ReprMixin:
    """
    Mixin class to provide automatic __repr__
    """
    __slots__ = []

    def __repr__(self):
        attrs = find_attrs(self)
        reprstr = ", ".join(f"{attr}={repr(getattr(self, attr))}"
                            for attr in attrs)
        return f"{type(self).__name__}({reprstr})"

#  ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#                             END
#  ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

if __name__ == '__main__':
    import doctest
    doctest.testmod()
