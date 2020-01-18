"""
Here are things which depend only on external libraries

Here we define:

* types
* constants
* basic conversion functions

"""

import logging as _logging
from fractions import Fraction
from typing import Union, Tuple


num_t = Union[float, int, Fraction]
time_t = Union[float, int, Fraction]
pitch_t = Union['Note', float, str]
fade_t = Union[float, Tuple[float, float]]

MAXDUR = 99999
UNSET = object()


logger = _logging.getLogger(f"emlib.music_core")


def F(x: Union[Fraction, float, int], den=None, maxden=1000000) -> Fraction:
    if den is not None:
        return Fraction(x, den).limit_denominator(maxden)
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(maxden)


def asTime(x:num_t, maxden=128) -> Fraction:
    if isinstance(x, Fraction):
        return x.limit_denominator(maxden)
    return F(x, maxden)


