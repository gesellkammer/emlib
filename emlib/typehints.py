from typing import (
    Optional as Opt,
    Tuple as Tup,
    Union as U,
    Sequence as Seq,   # a seq. of finite size
    Iterator as Iter,  # an iterator, possibly infinite, has no len
    List,
    Set,
    Dict,
    Deque,
    TypeVar,
    cast,          # force the typechecker
    NamedTuple,    # for mutable cases, use @dataclasses.dataclass
    overload,      # this makes only sense inside pyi files
    Callable,
    Any
)
import typing
from fractions import Fraction

Rat = U[int, float, Fraction]
Func = Callable

