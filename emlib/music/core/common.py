"""
Common routines, only depend on base for types and constants
"""
from __future__ import annotations
from ._base import *
from .config import config
from .state import getState
from emlib.lib import isiterable
from emlib import iterlib
from emlib.pitchtools import *
import dataclasses
from bpf4 import bpf

dbToAmpCurve: bpf.BpfInterface = bpf.expon(
    -120, 0,
    -60, 0.0,
    -40, 0.1,
    -30, 0.4,
    -18, 0.9,
    -6, 1,
    0, 1,
    exp=0.333)


def astuple(obj):
    return obj if isinstance(obj, tuple) else tuple(obj)


def addColumn(mtx, col):
    """
    Add a column to a list of lists

    mtx = [[1,   2,  3],
           [11, 12, 13],
           [21, 22, 23]]

    addColumn(mtx, [4, 14, 24]) 

    [[1,   2,  3,  4],
      11, 12, 13, 14],
      21, 22, 23, 24]]


    Args:
        mtx: a list of tuples or lists
        col: the new column

    Returns:
        a list of tuples or lists with the given column added

    """
    if not isiterable(col):
        col = [col]*len(mtx)
        
    if isinstance(mtx[0], tuple):
        return [row+(elem,) for row, elem in zip(mtx, col)]
    elif isinstance(mtx[0], list):
        return [row+[elem] for row, elem in zip(mtx, col)]
    raise TypeError(f"mtx should be a seq. of tuples or lists, "
                    f"got {mtx} ({type(mtx[0])})")


def carryColumns(rows: list, sentinel=None) -> list:
    """
    Converts a series of rows with possibly unequal number of elements per row
    so that all rows have the same length, filling each new row with elements
    from the previous, if they do not have enough elements (elements are "carried"
    to the next row)
    """
    maxlen = max(len(row) for row in rows)
    initrow = [0] * maxlen
    outrows = [initrow]
    for row in rows:
        lenrow = len(row)
        if lenrow < maxlen:
            row = row + outrows[-1][lenrow:]
        if sentinel in row:
            row = row.__class__(x if x is not sentinel else lastx for x, lastx in zip(row, outrows[-1]))
        outrows.append(row)
    # we need to discard the initial row
    return outrows[1:]


def normalizeFade(fade: fade_t,
                  defaultfade: float
                  ) -> tuple[float, float]:
    """ Returns (fadein, fadeout) """
    if fade is None:
        fadein, fadeout = defaultfade, defaultfade
    elif isinstance(fade, tuple):
        assert len(fade) == 2, f"fade: expected a tuple or list of len=2, got {fade}"
        fadein, fadeout = fade
    elif isinstance(fade, (int, float)):
        fadein = fadeout = fade
    else:
        raise TypeError(f"fade: expected a fadetime or a tuple of (fadein, fadeout), got {fade}")
    return fadein, fadeout


_pitchinterpolToInt = {
    'linear':0,
    'cos':1,
    'freqlinear':2,
    'freqcos':3
}


_fadeshapeToInt = {
    'linear': 0,
    'cos': 1
}


@dataclasses.dataclass
class PlayArgs:
    delay: float = None
    dur: float = None
    chan: int = None
    gain: float = None
    fade: Union[None, float, tuple[float, float]] = None
    instr: str = None
    pitchinterpol: str = None
    fadeshape: str = None
    args: dict[str, float] = None
    priority: int = 1
    position: float = None

    def fill(self, **kws) -> PlayArgs:
        return dataclasses.replace(self, **kws)


class CsoundEvent:
    """
    Represents a standard event (a line of variable breakpoints)

    init: delay=0, chan=0, fadein=None, fadeout=None
    bp: delay, midi, amp, ...
    (all breakpoints should have the same length)

    protocol:

    i inum, delay, dur, p4, p5, ...
    i inum, delay, dur, chan, fadein, fadeout, bplen, *flat(bps)

    inum is given by the manager
    dur is calculated based on bps
    bplen is calculated based on bps (it is the length of one breakpoint)
    """
    __slots__ = ("bps", "delay", "chan", "fadein", "fadeout", "gain",
                 "instr", "pitchinterpol", "fadeshape", "stereo", "args",
                 "priority", "position")

    def __init__(self,
                 bps: List[tuple],
                 delay=0.0,
                 chan:int = None,
                 fade=None,
                 gain:float = 1.0,
                 instr=None,
                 pitchinterpol=None,
                 fadeshape=None,
                 args: dict[str, float] = None,
                 priority:int=1,
                 position:float=0):
        """
        bps (breakpoints): a seq of (delay, midi, amp, ...) of len >= 1.
        """
        bps = carryColumns(bps)

        if len(bps[0]) < 2:
            raise ValueError(f"A breakpoint should have at least (delay, pitch), "
                             f"but got {bps}")
        if len(bps[0]) < 3:
            bps = addColumn(bps, 1)
        assert len(bps[0])>= 3
        assert all(isinstance(bp, tuple) and len(bp) == len(bps[0]) for bp in bps)

        self.bps = bps
        fadein, fadeout = normalizeFade(fade, config['play.fade'])
        dur = self.getDur()
        self.delay = delay
        self.chan = chan or config['play.chan'] or 1
        self.gain = gain or config['play.gain']
        self.fadein = max(fadein, 0.0001)
        self.fadeout = fadeout if dur < 0 else min(fadeout, dur)
        self.instr = instr
        self.pitchinterpol = pitchinterpol or config['play.pitchInterpolation']
        self.fadeshape = fadeshape or config['play.fadeShape']
        self.args = args
        self.priority = priority
        self.position = position
        self._consolidateDelay()
        self._applyTimeFactor(getState().timefactor)

    @classmethod
    def fromPlayArgs(cls, bps:list[tuple[float, ...]], playargs: PlayArgs
                     ) -> CsoundEvent:
        """
        Construct a CsoundEvent from breakpoints and playargs

        Args:
            bps: the breakpoints
            playargs: playargs

        Returns:
            a new CsoundEvent
        """
        d = dataclasses.asdict(playargs)
        d.pop('dur', None)
        return cls(bps=bps, **d)

    def _consolidateDelay(self):
        delay0 = self.bps[0][0]
        if delay0 > 0:
            self.delay += delay0
            self.bps = [(bp[0]-delay0,)+bp[1:] for bp in self.bps]

    def _applyTimeFactor(self, timefactor):
        if timefactor == 1:
            return
        self.delay *= timefactor
        self.bps = [(bp[0]*timefactor,)+bp[1:] for bp in self.bps]

    def getDur(self) -> float:
        return self.bps[-1][0]

    def breakpointSize(self) -> int:
        return len(self.bps[0])

    def getArgs(self) -> List[float]:
        """
        returns pargs, beginning with p2

        idx parg    desc
        0   2       delay
        1   3       duration
        2   4       tabnum
        3   5       gain
        4   6       chan
        5   7       position
        6   8       fade0
        7   9       fade1
        8   0       pitchinterpol
        9   1       fadeshape
        0   2       numbps
        1   3       bplen
        22... data

        tabnum: if 0 it is discarded and filled with a valid number later
        """
        pitchinterpol = _pitchinterpolToInt[self.pitchinterpol]
        fadeshape = _fadeshapeToInt[self.fadeshape]

        args = [float(self.delay),
                self.getDur(),
                0,  # table index, to be filled later
                self.gain,
                self.chan,
                self.position,
                self.fadein,
                self.fadeout,
                pitchinterpol,
                fadeshape,
                len(self.bps),
                self.breakpointSize()]

        args.extend(iterlib.flatten(self.bps))
        assert all(isinstance(arg, (float, int, Fraction)) for arg in args), args
        return args

    def __repr__(self) -> str:
        lines = [f"CsoundLine(delay={float(self.delay):.3f}, gain={self.gain}, chan={self.chan}"
                 f", fadein={self.fadein}, fadeout={self.fadeout}"]
        for bp in self.bps:
            lines.append(f"    {float(bp[0]):.3f} {bp[1:]}")
        lines.append("")
        return "\n".join(lines)
