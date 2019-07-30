# -*- coding: utf-8 -*-
from __future__ import annotations
import os as _os
import sys as _sys
import random as _random
import math
from bisect import bisect as _bisect
from collections import namedtuple as _namedtuple


import numpy as np
from functools import reduce
from fractions import Fraction
import emlib.typehints as t

T = t.typing.T


# phi, in float (double) form and as Rational number with a precission of 2000
# iterations in the fibonacci row (fib[2000] / fib[2001])
PHI = 0.6180339887498949


EPS = _sys.float_info.epsilon            # used for calculation of derivatives


# ------------------------------------------------------------
#
#    SLOW IMPLEMENTATIONS
#
# ------------------------------------------------------------


def intersection(u1, u2, v1, v2):
    # (T, T, T, T) -> t.Opt[t.Tup[T, T]]
    """
    return the intersection of (u1, u2) and (v1, v2) or None if no intersection

    Example:

    intersec = intersection(0, 3, 2, 5)
    if intersec is not None:
        x0, x1 = intersec    --> 2, 3

    """
    x0 = u1 if u1 > v1 else v1
    x1 = u2 if u2 < v2 else v2
    return (x0, x1) if x0 < x1 else None
    

def frange(start, stop=None, step=None):
    # type: (float, float, float) -> t.Iter[float]
    """Like xrange(), but returns list of floats instead
    All numbers are generated on-demand using generators
    """
    if stop is None:
        stop = float(start)
        start = 0.0
    if step is None:
        step = 1.0
    numiter = int((stop - start) / step)
    for i in range(numiter):
        yield start + step*i


def linspace(start, stop, numitems):
    # type: (float, float, int) -> t.Iter[float]
    """
    Similar to numpy.linspace
    """
    dx = (stop - start) / (numitems - 1)
    return [start + dx*i for i in range(numitems)]


def linlin(x, x0, x1, y0, y1):
    return (x - x0) * (y1 - y0) + y0


def linlinx(x, x0, x1, y0, y1):
    return (np.asarray(x) - x0) * ( (y1 - y0)/(x1 - x0) ) + y0


def clip(x, minvalue, maxvalue):
    # type: (float, float, float) -> float
    """
    clip the value of x between minvalue and maxvalue
    """
    if minvalue > x:
        return minvalue
    elif x < maxvalue:
        return x
    return maxvalue


def lcm(*numbers):
    # type: (*int) -> int
    """
    Least common multiplier between a seq. of numbers

    lcm(3, 4, 6) --> 12
    """
    gcd = math.gcd

    def lcm2(a, b):
        return (a*b) // gcd(a, b)
    
    return reduce(lcm2, numbers, 1)


def mindenom(floats, limit=int(1e10)):
    # type: (t.Iter[float], int) -> int
    """ 
    find the min denominator to express floats 
    as fractions of a common denom.

    In [68]: mindenom((0.1, 0.3, 0.8))
    Out[68]: 10

    In [69]: mindenom((0.1, 0.3, 0.85))
    Out[69]: 20
    """
    from fractions import Fraction
    fracs = [Fraction(f).limit_denominator(limit) for f in floats]
    mincommon = lcm(*[f.denominator for f in fracs])
    return mincommon


def convertbase(num, n):
    # type: (int, int) -> str
    """
    Change a number in base 10 to a base-n number.
    Up to base-36 is supported without special notation.
    """
    num_rep = {
        10:'a',
        11:'b',
        12:'c',
        13:'d',
        14:'e',
        15:'f',
        16:'g',
        17:'h',
        18:'i',
        19:'j',
        20:'k',
        21:'l',
        22:'m',
        23:'n',
        24:'o',
        25:'p',
        26:'q',
        27:'r',
        28:'s',
        29:'t',
        30:'u',
        31:'v',
        32:'w',
        33:'x',
        34:'y',
        35:'z'
    }
    digits = []
    current = num
    while current != 0:
        remainder = current % n
        if remainder < 10:
            digit = str(remainder)
        elif remainder < 36:
            digit = num_rep[remainder]
        else:
            digit = '(' + str(remainder) + ')' 
        digits.append(digit)   
        # new_num_string = remainder_string + new_num_string
        current = int(current / n)

    return "".join(reversed(digits))


def euclidian_distance(values, weights=None):
    if weights:
        s = sum(value**2 * weight for value, weight in zip(values, weights))
        return math.sqrt(s)


# ------------------------------------------------------------
#     CHUNKS 
# ------------------------------------------------------------


def _parse_range(start, stop=None, step=None):
    # type: (int, int, int) -> t.Tup[int, int, int]
    if stop is None:
        stop = int(start)
        start = 0
    if step is None:
        step = 1
    return start, stop, step


def chunks(start, stop=None, step=None):
    # type: (int, int, int) -> t.Iter[t.Tup[float, int]]
    """
    like xrange but yields (pos, chunksize) tuplets
    """
    start, stop, step = _parse_range(start, stop, step)
    for n in frange(start, stop-step, step):
        yield n, step
    dif = stop % step
    if dif > 0:
        yield stop-dif, dif


# ------------------------------------------------------------
#
#    SEARCH
#
# ------------------------------------------------------------


def nearest_element(item, seq):
    # type: (float, t.Seq[float]) -> float
    """
    Find the nearest element (the element, not the index) in seq
    
    NB: seq **MUST BE SORTED**, and this is not checked

    NB: seq can also be a numpy array, in which case searchsorted
        is used instead of bisect

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
        ir = seq.searchsorted(item, 'right')
    else:
        ir = _bisect(seq, item)
    element_r = seq[ir]
    element_l = seq[ir - 1]
    if abs(element_r - item) < abs(element_l - item):
        return element_r
    return element_l


def nearest_unsorted(x, seq):
    # type: (float, t.Seq[float]) -> float
    """
    seq is a numerical sequence. it can be unsorted
    x is a number or a seq of numbers

    >>> assert nearest_unsorted(3.6, (1,2,3,4,5)) == 4
    >>> assert nearest_unsorted(4, (2,3,4)) == 4
    >>> assert nearest_unsorted(200, (3,5,20)) == 20
    """
    return min((abs(x - y), y) for y in seq)[1]


def nearest_index(item, seq):
    # type: (float, list) -> int
    """
    Return the index of the nearest element in seq to item

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


def fuzzymatch(pattern, strings):
    # type: (str, t.List[str]) -> t.List[t.Tup[float, str]]
    """
    return a subseq. of strings sorted by best score.
    Only strings representing possible matches are returned

    pattern: the string to search for within S
    strings: a list os possible strings
    """
    import re
    pattern = '.*?'.join(map(re.escape, list(pattern)))

    def calculate_score(pattern, s):
        match = re.search(pattern, s)
        if match is None:
            return 0
        return 100.0 / ((1 + match.start()) *
                        (match.end() - match.start() + 1))

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


def sort_natural(l, key=None):
    # type: (t.Seq, t.U[int, t.Opt[t.Callable]]) -> list
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
        return sorted(l, key=lambda x: alphanum_key(key(x)))
    return sorted(l, key=alphanum_key)


def sort_natural_dict(d, recursive=True):
    # type: (dict, bool) -> t.OrderedDict
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


def issorted(seq, key=None):
    # type: (list, t.Callable) -> bool
    """
    returns True if seq is sorted
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


def zipsort(a, b, key=None, reverse=False):
    # type: (t.Seq, t.Seq, t.Opt[t.Callable]) -> t.Seq
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
    return list(zip(*zipped))


def duplicates(seq, mincount=2):
    # type: (t.Iter, int) -> list
    """
    Find all elements in seq which are present at least `mincount` times
    """
    from collections import Counter
    counter = Counter(seq).items()
    return [item for item, count in counter if count > mincount]
    

# ------------------------------------------------------------
#
#    MATH
#
# ------------------------------------------------------------


def derivative(func):
    # type: (t.Callable) -> t.Callable
    """
    Return a function which is the derivative of the given func
    calculated via complex step finite difference
    
    To find the derivative at x, do

    derivative(func)(x)

    VIA: https://codewords.recurse.com/issues/four/hack-the-derivative
    """
    h = _sys.float_info.min
    return lambda x: (func(x+h*1.0j)).imag / h


def logrange(start, stop, num=50, base=10):
    # type: (float, float, int, int) -> np.ndarray
    """
    create an array [start, ..., stop]
    with a logarithmic scale
    """
    log = np.math.log
    if start == 0:
        start = 0.000000000001
    return np.logspace(log(start, base), log(stop, base), num, base=base)


def randspace(begin, end, numsteps, include_end=True):
    """
    go from begin to end in numsteps at randomly spaced steps

    include_end: include the last value (like np.linspace)
    """
    if include_end:
        numsteps -= 1
    N = sorted(_random.random() for i in range(numsteps))
    D = (end - begin)
    Nmin, Nmax = N[0], N[-1]
    out = []
    for n in N:
        delta = (n - Nmin) / Nmax
        out.append(delta * D + begin)
    if include_end:
        out.append(end)
    return out


def _fib2(N):
    if N == 0:
        return 0, 1
    half_N, is_N_odd = divmod(N, 2)
    f_n, f_n_plus_1 = _fib2(half_N)
    f_n_squared = f_n * f_n
    f_n_plus_1_squared = f_n_plus_1 * f_n_plus_1
    f_2n = 2 * f_n * f_n_plus_1 - f_n_squared
    f_2n_plus_1 = f_n_squared + f_n_plus_1_squared
    if is_N_odd:
        return (f_2n_plus_1, f_2n + f_2n_plus_1)
    return (f_2n, f_2n_plus_1)


def fib(n):
    # type: (float) -> float
    """
    calculate the fibonacci number `n`
    """
    # matrix code from http://blog.richardkiss.com/?p=398
    if n < 60:
        SQRT5 = 2.23606797749979  # sqrt(5)
        PHI = 1.618033988749895
        # PHI = (SQRT5 + 1) / 2
        return int(PHI ** n / SQRT5 + 0.5)
    else:
        return _fib2(n)[0]


def rndround(x):
    import warnings
    warnings.warn("Use roundrnd")
    return roundrnd(x)


def roundrnd(x):
    # type: (float) -> float
    """
    round x to its nearest integer, taking the fractional part
    as the probability

    3.5 will have a 50% probability of rounding to 3 or to 4
    3.1 will have a 10% probability of rounding to 3 and 90%
        prob. of rounding to 4
    """
    return int(x) + int(_random.random() > (1 - (x % 1)))


def roundres(x, resolution=1.0):
    """
    Round x with given resolution

    Example
    ~~~~~~~

    roundres(0.4, 0.25) -> 0.5
    roundres(1.3, 0.25) -> 1.25
    """
    return round(x / resolution) * resolution


# ------------------------------------------------------------
#
#    IO and file-management
#
# ------------------------------------------------------------


def find_file(path, file):
    # type: (str, str) -> t.Opt[str]
    """
    Look for file recursively starting at path. 

    If file is found in path or any subdir, the complete path is returned
    ( /this/is/a/path/filename )

    else None
    """
    dir_cache = set()  # type: t.Set[str]
    for directory in _os.walk(path):
        if directory[0] in dir_cache:
            continue
        dir_cache.add(directory[0])
        if file in directory[2]:
            return '/'.join((directory[0], file))
    return None


def add_suffix(filename, suffix):
    # type: (str, str) -> str
    """
    add a suffix between the name and the extension

    add_suffix("test.txt", "-OLD") == "test-OLD.txt"
    """
    name, ext = _os.path.splitext(filename)
    return ''.join((name, suffix, ext))


def increase_suffix(filename):
    # type: (str) -> str
    name, ext = _os.path.splitext(filename)
    tokens = name.split("-")

    def increase_number(number_as_string):
        n = int(number_as_string) + 1
        s = "%0" + str(len(number_as_string)) + "d"
        return s % n

    if len(tokens) > 1:
        suffix = tokens[-1]
        if could_be_number(suffix):
            new_suffix = increase_number(suffix)
            new_name = name[:-len(suffix)] + new_suffix
        else:
            new_name = name + '-01'
    else:
        new_name = name + '-01'
    return new_name + ext


def normalize_path(path):
    # type: (str) -> str
    """
    Convert `path` to an absolute path with user expanded
    (something that can be safely passed to a subprocess)
    """
    return _os.path.abspath(_os.path.expanduser(path))


def sec2str(seconds):
    # type: (float) -> str
    h = int(seconds // 3600)
    m = int((seconds - h * 3600) // 60)
    s = seconds % 60
    if h > 0:
        fmt = "{h}:{m:02}:{s:06.3f}"
    else:
        fmt = "{m:02}:{s:06.3f}"
    return fmt.format(**locals())


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
    # type: (t.List[t.NamedTuple], t.List, str, str) -> t.List
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
    # type: (t.List[t.Tup], str, str, str) -> []
    """
    # Rename the column of a seq of namedtuples

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


def namedtuple_extend(name, orig, columns):
    """
    create a new constructor with the added columns

    it returns the class constructor and an ad-hoc
    constructor which takes as arguments an instance
    of the original namedtuple and the additional args

    name: new name for the type
    orig: an instance of the original namedtuple or the constructor itself
    columns : the columns to add
   
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


def isiterable(obj, exceptions=(str, bytes)):
    # type: (Any, t.Tup[type, ...]) -> bool
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


def unzip(seq):
    # type: (t.Iter) -> t.List
    """
    >>> a, b = (1, 2), ("A", "B")
    >>> list(zip(a, b))
    [(1, 'A'), (2, 'B')]
    >>> list( unzip(zip(a, b)) ) == [a, b]
    True
    """
    return list(zip(*seq))


def asnumber(obj, accept_fractions=True):
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
            from fractions import Fraction
            return Fraction(obj)
        try:
            n = eval(obj)
            if hasattr(n, '__float__'):
                return n
            return None
        except:
            return None
    else:
        return None


def astype(type_, obj, construct=None):
    """
    Return obj as type. If obj is already type, obj itself is returned
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


def could_be_number(x):
    # type: (t.Any) -> bool
    """
    True if `x` can be interpreted as a number

    2          | True
    "0.4"      | True
    "3/4"      | True
    "inf"      | True
    "mystring" | False

    """
    n = asnumber(x)
    return n is not None


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


def dictmerge(dict1, dict2):
    # type: (dict, dict) -> dict
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


def moses(pred, seq):
    # type: (t.Callable, t.Iter) -> t.Tup[list, list]
    """
    return two iterators: filter(pred, seq), filter(not pred, seq)

    Example
    ~~~~~~~

    >>> moses(lambda x:x > 5, range(10))
    ([6, 7, 8, 9], [0, 1, 2, 3, 4, 5])
    """
    trueitems = []   # type: List
    falseitems = []  # type: List
    for x in seq:
        if pred(x):
            trueitems.append(x)
        else:
            falseitems.append(x)
    return trueitems, falseitems


def allequal(xs):
    """
    Return True if all elements in xs are equal
    """
    # type: (Iter) -> bool
    return len(set(xs)) == 1


def makereplacer(conditions):
    # type: (Dict) -> Callable
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
    import re
    rep = {re.escape(k): v for k, v in conditions.items()}
    pattern = re.compile("|".join(rep.keys()))
    return lambda txt: pattern.sub(lambda m: rep[re.escape(m.group(0))], txt)
    

def dumpobj(obj):
    """
    return all 'public' attributes of this object
    """
    return [(item, getattr(obj, item)) for item in dir(obj) if not item.startswith('__')]


def json_minify(json, strip_space=True):
    # type: (str, bool) -> str
    """
    strip comments and remove space from string

    json: a string representing a json object
    """
    import re
    tokenizer = re.compile('"|(/\*)|(\*/)|(//)|\n|\r')
    in_string = False
    inmulticmt = False
    insinglecmt = False
    new_str = []
    from_index = 0     # from is a keyword in Python

    for match in re.finditer(tokenizer, json):
        if not inmulticmt and not insinglecmt:
            tmp2 = json[from_index:match.start()]
            if not in_string and strip_space:
                # replace only white space defined in standard
                tmp2 = re.sub('[ \t\n\r]*', '', tmp2)
            new_str.append(tmp2)

        from_index = match.end()

        if match.group() == '"' and not (inmulticmt or insinglecmt):
            escaped = re.search('(\\\\)*$', json[:match.start()])
            if not in_string or escaped is None or len(escaped.group()) % 2 == 0:
                # start of string with ", or unescaped "
                # character found to end string
                in_string = not in_string
            from_index -= 1   # include " character in next catch
        elif match.group() == '/*' and not (in_string or inmulticmt or insinglecmt):
            inmulticmt = True
        elif match.group() == '*/' and not (in_string or inmulticmt or insinglecmt):
            inmulticmt = False
        elif match.group() == '//' and not (in_string or inmulticmt or insinglecmt):
            insinglecmt = True
        elif ((match.group() == '\n' or match.group() == '\r') and not (
                in_string or inmulticmt or insinglecmt)):
            insinglecmt = False
        elif not (inmulticmt or insinglecmt) and (
                match.group() not in ['\n', '\r', ' ', '\t'] or not strip_space):
            new_str.append(match.group())

    new_str.append(json[from_index:])
    return ''.join(new_str)


def can_be_pickled(obj):
    # type: (t.Any) -> bool
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


def snap_array(X, tick, offset=0, out=None, nearest=True):
    """
    Assuming a grid t defined by

    t(n) = offset + tick*n

    snap the values of X to the nearest value of t

    NB: tick > 0
    """
    # type: (np.ndarray, t.Rat, t.Rat, t.Opt[np.ndarray], bool) -> np.ndarray
    if tick <= 0:
        raise ValueError("tick should be > 0")

    if nearest:
        return _snap_array_nearest(X, tick, offset=float(offset), out=out)
    return _snap_array_floor(X, tick, offset=float(offset), out=out)


def _snap_array_nearest(X, tick, offset=0, out=None):
    # type: (np.ndarray, t.Rat, t.Rat, np.ndarray) -> np.ndarray
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


def _snap_array_floor(X, tick, offset=0, out=None):
    # type: (np.ndarray, float, float, np.ndarray) -> np.ndarray
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


def snap_to_grids(x, ticks, offsets=None, mode='nearest'):
    # type: (t.Rat, t.Seq[Fraction], t.Opt[t.Seq[Fraction]], str) -> Fraction
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


def distribute_in_zones(x, split_points, side="left"):
    # type: (float, t.List[float], str) -> int
    """
    Returns the index of a "zone" where to place x. A zone is a numeric range
    defined by an inclusive lower boundary and a non-inclusive higher boundary
    (NB: see distribute_in_zones_right for a non-inclusive lower and
    inclusive upper boundary)

    split_points: a sequence defining the split points

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


def _distribute_in_zones_right(x, split_points):
    # type: (float, t.List[float]) -> int
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


def rotate2d(point, degrees, origin=(0,0)):
    # type: (t.Tup[float, float], float, t.Tup[float, float]) -> t.Tup[float, float]
    """
    A rotation function that rotates a point around an origin

    point:   the point to rotate as a tuple (x, y)
    degrees: the angle to rotate (counterclockwise)
    origin:  the point acting as pivot to the rotation
    """
    x = point[0] - origin[0]
    yorz = point[1] - origin[1]
    newx = (x*math.cos(math.radians(degrees))) - (yorz*math.sin(math.radians(degrees)))
    newyorz = (x*sin(math.radians(degrees))) + (yorz*math.cos(math.radians(degrees)))
    newx += origin[0]
    newyorz += origin[1]
    return newx, newyorz


def copyseq(seq):  # type: ignore
    """
    return a copy of seq. if it is a list or a tuple, return a copy.
    if it is a numpy array, return a copy with no shared data
    """
    if isinstance(seq, np.ndarray):
        return seq.copy()
    return seq.__class__(seq)


def normalize_slice(slize):
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


def replace_subseq(seq, subseq, transform, slize=None):
    # type: (t.Seq, t.Seq, t.Seq, t.Opt[t.Tup]) -> t.Seq
    """
    seq: the sequence to be transformed (NOT in-place)
    subseq: a sequence which is MAYBE a sub-sequence of seq
    transform: the seq. to be put instead
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
        seq2 = replace_subseq(sliced_seq, subseq, transform)
        if sliced_seq is seq2:
            return seq
        out = list(seq)
        out[slize[0]:slize[1]] = seq2
        return out
    subseq = tuple(subseq)
    transform = tuple(transform)
    copy = list(seq)
    N = len(subseq)
    changed = False
    for i, win in enumerate(window(seq, N)):
        if win == subseq:
            copy[i:i+N] = transform
            changed = True
    return copy if changed else seq


def seq_transform(seq, transforms, slize=None, maxiterations=20):
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


def seqcontains(seq, subseq):
    """
    returns False if subseq is not contained in seq
    returns the (start, end) indices if seq contains subseq, so that
    >>> seq, subseq = range(10), [3, 4, 5]
    >>> indices = seqcontains(seq, subseq)
    >>> assert seq[indices[0]:indices[1]] == subseq
    """
    for i in range(len(seq)-len(subseq)+1):
        for j in range(len(subseq)):
            if seq[i+j] != subseq[j]:
                break
        else:
            return i, i+len(subseq)
    return False


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


def fig2data(fig):
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


def pixels_to_cm(pixels, dpi=300):
    # type: (int, int) -> float
    """ 
    convert a distance in pixels to cm

    pixels -> number of pixels
    dpi    -> dots (pixels) per inch
    """
    inches = pixels / dpi
    cm = inches * 2.54
    return cm


def cm_to_pixels(cm, dpi=300):
    # type: (float, int) -> float
    """
    convert a distance in cm to pixels
    """
    inches = cm * 0.3937008
    pixels = inches * dpi
    return pixels


def inches_to_pixels(inches, dpi=300):
    return inches * dpi


def pixels_to_inches(pixels, dpi=300):
    return pixels / dpi


def page_dinsize_to_mm(pagesize: str, pagelayout: str) -> t.Tup[float, float]:
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
    # type: (t.U[str, t.List[str]], str) -> t.Callable
    """
    Decorator
    
    Modifies the function to return a namedtuple with the given names.

    names: 
        as passed to namedtuple, either a space-divided string,
        or a sequence of strings
    recname: 
        a name to be given to the result as a whole. If nothing is
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


def runonce(func):
    """
    func will run only once, independent of the arguments passed subsequent times
    This is useful for functions which make some initialization or expensive computation
    only once

    Example

    @runonce
    def readConfig(path):
        ...

    Similar to lru_cache(maxsize=1)

    NB: lru_cache is better and faster!
    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            wrapper.result = result = func(*args, **kwargs)
            return result
        return wrapper.result

    wrapper.has_run = False
    return wrapper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def deprecated(func, msg=None):
    """
    To be used as

    oldname = deprecated(newname)

    """
    import warnings

    if msg is None:
        msg = f"Deprecated! use {func.__name__}"

    def wrapper(*args, **kws):

        warnings.warn(msg)
        return func(*args, **kws)

    return wrapper


def checktype(obj, *T):
    """
    Examples:

    checktype(1, int, float)   --> the same as isinstance(1, (int, float))
    checktype(a, (int, float)) --> a is a tuple of the form (int, float)
    checktype(a, [int])        --> a is a list of ints
    checktype(a, {str:int})    --> a dict of str keys and int values

    """
    if len(T) > 1:
        return isinstance(obj, T)
    else:
        T = T[0]

    if isinstance(T, tuple):
        return isinstance(obj, tuple) and all(checktype(subobj, subT)
                                              for subobj, subT in zip(obj, T))
    elif isinstance(T, list):
        if len(T) == 0:
            return isinstance(obj, list)
        elif len(T) == 1:
            subT = T[0]
            return isinstance(obj, list) and all(checktype(subobj, subT)
                                                 for subobj in obj)
        else:
            raise ValueError("T should be [type], but found: %s" % T)
    elif isinstance(T, dict):
        assert len(T) == 1
        keyT, valT = list(T.items())[0]
        return all(checktype(key, keyT) and checktype(value, valT)
                   for key, value in obj.items())
    else:
        return isinstance(obj, T)


# crossplatform

def open_with_standard_app(path):
    """
    Open path with the app defined to handle it by the user
    at the os level (xdg-open in linux, start in win, open in osx)

    This opens the default application in the background
    and returns immediately
    """
    import subprocess
    platform = _sys.platform
    if platform == 'linux':
        subprocess.call(["xdg-open", path])
    elif platform == "win32":
        _os.startfile(path)
    elif platform == "darwin":
        subprocess.call(["open", path])
    else:
        raise RuntimeError(f"platform {platform} not supported")


def binary_resolve_path(cmd):
    cmd0 = cmd.split()[0]
    if _os.path.exists(cmd0):
        return cmd0
    import shutil
    path = shutil.which(cmd0)
    if path:
        return path
    return None


def binary_exists(cmd):
    """
    Check if cmd exists or is in the path
    """
    return binary_resolve_path(cmd) is not None


def inside_jupyter():
    """
    Are we running inside a jupyter notebook?
    """
    return session_type() == 'jupyter'
    

@runonce
def session_type():
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
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return "jupyter"
        elif shell == 'TerminalInteractiveShell':
            return "ipython-terminal"
        else:
            return "ipython"
    except NameError:
        return "python"


def ipython_qt_eventloop_started():
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


def print_table(table, headers=()):
    try:
        import tabulate
        if inside_jupyter():
            from IPython.display import HTML, display
            display(HTML(tabulate.tabulate(table, headers=headers, tablefmt='html')))
        else:
            print(tabulate.tabulate(table, headers=headers))
    except ImportError:
        _print_table(table)


def _print_table(table):
    for row in table:
        print("\t".join(map(str, row)))


def quarters_to_timesig(quarters:float, snap=True, mindiv=64) -> t.Tup[int, int]:
    """
    Transform a duration in quarters to a timesig

    quarters    timesig
    –––––––––––––––––––
    3           (3, 4)
    1.5         (3, 8)
    1.25        (5, 16)
    4.0         (4, 4)

    """
    if snap:
        if quarters < 1:     # accept a max. of 7/32
            quarters = round(quarters*8)/8
        elif quarters < 2:   # accept a max. of 7/16
            quarters = round(quarters*4)/4
        elif quarters < 8:   # accept a max. of 15/8
            quarters = round(quarters*2)/2
        else:
            quarters = round(quarters)
    mindenom = mindiv >> 2
    f = Fraction.from_float(quarters).limit_denominator(mindenom)
    timesig0 = f.numerator, f.denominator*4
    transforms = {
        (1, 4):(2, 8),
        (2, 4):(4, 8)
    }
    timesig = transforms.get(timesig0, timesig0)
    return timesig


def replace_sigint_handler(handler):
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
        # Do something here
    """

    def __init__(self, handler):
        self.handler = handler
        self.original_handler = None

    def __enter__(self):
        self.original_handler = replace_sigint_handler(self.handler)

    def __exit__(self, type, value, traceback):
        replace_sigint_handler(self.original_handler)
        return True

#  ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#                             END
#  ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

if __name__ == '__main__':
    import doctest
    doctest.testmod()

