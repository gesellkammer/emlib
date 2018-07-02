from __future__ import division as _div
import warnings
from numbers import Number
from fractions import Fraction as R

from emlib import iterlib as _iterlib
from emlib.lib import returns_tuple, snap_to_grid
import typing as t



def measure_duration(timesig, tempo):
    # type: (t.Union[str, t.Tuple[int, int]]) -> R
    """
    calculate the duration of a given measure with the given tempo

    timesig: can be of the form "4/4" or (4, 4)
    tempo:   a tempo value corresponding to the denominator of the time
             signature
             
    Examples
    ~~~~~~~~

    >>> measure_duration("3/4", 120)        # 3 quarters, quarter=120
    1.5
    >>> measure_duration((3, 8), 60)        # 3/8 bar, 8th note=60
    3
    
    >>> assert all(measure_duration((n, 4), 60) == n for n in range(20))
    >>> assert all(measure_duration((n, 8), 120) == n / 2 for n in range(20))
    >>> assert all(measure_duration((n, 16), (8,60)) == n / 2 for n in range(40))
    """
    if isinstance(timesig, str):
        assert "/" in timesig
        num, den = map(int, timesig.split('/'))
    elif isinstance(timesig, (tuple, list)):
        num, den = timesig
    else:
        raise ValueError(
            "timesig must be a string like '4/4' or a tuple (4, 4)")
    if isinstance(tempo, (tuple, list)):
        tempoden, tempoval = tempo
    else:
        tempoval = tempo
        tempoden = den
    quarterdur = R(tempoden, 4) * R(60) / tempoval
    quarters_in_measure = R(4, den) * num
    return quarters_in_measure * quarterdur


@returns_tuple("linear2framed framed2linear")
def framed_time(offsets, durations):
    """
    Returns two bpfs to convert a value between linear and framed coords, and viceversa

    offsets: the start x of each frame
    durations: the duration of each frame

    Returns: linear2framed, framed2linear

    Example
    ~~~~~~~

    Imagine you want to apply a linear process to a "track" divided in
    non-contiguous frames. For example, a crescendo in density to all frames
    labeled "A".

    >>> from collections import namedtuple
    >>> Frame = namedtuple("Frame", "id start dur")
    >>> frames = map(Frame, [
        # id  start dur
        ('A', 0,    0.5),
        ('B', 0.5,  1),
        ('A', 1.5,  0.5),
        ('A', 2.0,  0.5),
        ('B', 2.5,  1)
    ])
    >>> a_frames = [frame for frame in frames if frame.id == 'A']
    >>> offsets = [frame.start for frame in a_frames]
    >>> durs = [frame.dur for frame in a_frames]
    >>> density = _bpf.linear(0, 0, 1, 1)  # linear crescendo in density
    >>> lin2framed, framed2lin = framed_time(offsets, durs)

    # Now to convert from linear time to framed time, call lin2framed
    >>> lin2framed(0.5)
    1.5
    >>> framed2lin(1.5)
    0.5
    """
    import bpf4 as bpf
    xs = [0] + list(_iterlib.partialsum(dur for dur in durations))
    pairs = []
    for (x0, x1), y in zip(_iterlib.pairwise(xs), offsets):
        pairs.append((x0, y))
        pairs.append((x1, y + (x1 - x0)))
    xs, ys = zip(*pairs)
    lin2framed = bpf.core.Linear(xs, ys)
    try:
        framed2lin = bpf.core.Linear(ys, xs)
    except ValueError:
        ys = _force_sorted(ys)
        framed2lin = bpf.core.Linear(ys, xs)
    return lin2framed, framed2lin


def _force_sorted(xs):
    EPS = 0
    out = []
    lastx = float('-inf')
    for x in xs:
        if x < lastx:
            x = lastx + EPS
        lastx = x
        out.append(x)
    return out


def fit_measures_to_frames(framelengths, timesigs, tempi,
                           initial_tempo=60, initial_timesig=(4, 4),
                           time_weight=30, tempo_weight=6,
                           preferredmeasure_weight=2,
                           preferredtempo_weight=2,
                           preservedenom_weight=2,
                           zigzag_weight=4):
    """
    fit a series of frames, each frame to one measure

    framelengths: a seq of durations
    timesigs: a dictionary of the sort:
        timesig: (temporange, weight), with
                  weight    : a number, the higher, the better a timesig is
                  temporange: a tuple (mintempo, maxtempo)

        Example:
            timesigs = {
                (3, 8) : (5,  (72, 144)),
                (4, 8) : (4,  (72, 144)),
                (3, 4) : (10, (40, 160)),
                ...
            }
    tempi:    a dict, specifying the weight of each tempo
        Example:
            tempi = {
                40: 3,
                48: 10,
                52: 5,
                ...
                144: 1
            }
    TODO: resto de los argumentos
    """
    # XXX: TODO! ver e.tempo.py, proj/orthogonal_features/forma.py
    pass


@returns_tuple("snapped division")
def quantize_duration(dur, possible_divisions=None):
    # type: (float, t.Sequence[t.Union[int, str]]) -> t.Tuple[float, int]
    """
    quantize duration dur to fit in a grid defined by the possible subdivisions

    dur: the duration to quantize (a float). 1 indicates a duration of a quarter note
    possible_divisions: a seq. of possible divisions of the whole note.
                        default=(2, 4, 8, 16, 32, 6, 12, 5, 10)
                        4  : "q"   : a quarter
                        8  : "q/2" : an 1/8th note
                        16 : "q/4" : a 16th note
                        12 : "q/3" : a triplet
                        6  : "h/3" : a quarter-triplet (3 quarters in the place of 2)
                        10 : "h/5" : an eigth note quintuplet
                        20 : "q/5" : a 16th quintuplet
                        
    Returns a tuple of: the quantized value and the division used
    """
    if possible_divisions is None:
        possible_divisions = (2, 4, 8, 16, 32, 6, 12, 5, 10, 20, 24)
    if any(isinstance(div, str) for div in possible_divisions):
        possible_divisions = [_str2dur(div)[1] if isinstance(div, str) else div
                              for div in possible_divisions]
    results = [(index, div, snap_to_grid(dur, 4 / div))
               for index, div in enumerate(possible_divisions)]
    diffs = sorted([(abs(snapped - dur), index, div, snapped)
                    for index, div, snapped in results])
    diff, _, div, snapped = diffs[0]
    return snapped, div


def find_nearest_duration(dur, possible_durations, direction="<>"):
    """
    dur: a Dur or a float (will be converted to Dur via .fromfloat)
    possible_durations: a seq of Durs
    direction: "<"  -> find a dur from possible_durations which is lower than dur
               ">"  -> find a dur from possible_durations which is higher than dur
               "<>" -> find the nearest dur in possible_durations

    Example
    ~~~~~~~

    >>> possible_durations = [0.5, 0.75, 1]
    >>> find_nearest_duration(0.61, possible_durations, "<>")
    0.5

    """
    possdurs = sorted(possible_durations, key=lambda d: float(d))
    inf = float("inf")
    if dur < possible_durations[0]:
        return possible_durations[0] if direction != "<" else None
    elif dur > possible_durations[-1]:
        return possible_durations[-1] if direction != ">" else None
    if direction == "<":
        nearest = sorted(possdurs, key=lambda d:abs(dur - d) if d < dur else inf)[0]
        return nearest if nearest < inf else None
    elif direction == ">":
        nearest = sorted(possdurs, key=lambda d:abs(dur - d) if d > dur else inf)[0]
        return nearest if nearest < inf else None
    elif direction == "<>":
        nearest = sorted(possdurs, key=lambda d:abs(dur - d))[0]
        return nearest
    else:
        raise ValueError("direction should be one of '>', '<', or '<>'")


def _str2dur(strdur: str) -> t.Tuple[int, int]:
    """
    parses a duration of the kind "q/5" (one of a quintuplet of a quarter note)
    into a subdivision of the whole note
    """
    mul = 1
    if "/" in strdur:
        kind, div = strdur.split("/")
        div = int(div)
        assert 2 <= div <= 12
        if len(kind) == 2:
            mul = int(kind[0])
            kind = kind[1]
        assert kind in ("q", "h", "e")
    else:
        kind = strdur
        div = 1
    num = mul
    den = {"e": 8, "q": 4, "h": 2}[kind] * div
    return num, den


DEFAULT_TEMPI = (
    60, 120, 90, 132, 48, 80, 96, 100, 72, 52,
    40, 112, 144, 45, 160, 108, 88, 76, 66, 69)


def tempo2beatdur(tempo):
    return 60 / tempo


@returns_tuple("best_tempi resulting_durs numbeats")
def best_tempo(duration, possible_tempi=DEFAULT_TEMPI,
               num_solutions=5, verbose=True):
    """
    Find best tempi that fit the given duration
    """
    remainings = [(duration % tempo2beatdur(tempo), i)
                  for i, tempo in enumerate(possible_tempi)]
    best_tempi = [possible_tempi[i] for remaining, i in
                  sorted(remainings)[:num_solutions]]
    numbeats = [int(duration / tempo2beatdur(tempo) + 0.4999)
                for tempo in best_tempi]
    resulting_durs = [tempo2beatdur(tempo) * n
                      for tempo, n in zip(best_tempi, numbeats)]
    if verbose:
        for tempo, dur, n in zip(best_tempi, resulting_durs, numbeats):
            print("Tempo: %f \t Resulting duration: %f \t Number of Beats: %d" %
                  (tempo, dur, n))
    else:
        return best_tempi, resulting_durs, numbeats


def translate_subdivision(subdivision, new_tempo, original_tempo=60):
    dur_subdiv = tempo2beatdur(original_tempo) / subdivision
    new_beat = tempo2beatdur(new_tempo)
    dur_in_new_tempo = dur_subdiv / new_beat
    return dur_in_new_tempo


def parse_dur(dur, tempo=60):

    def ratio_to_dur(num, den):
        return int(num) * (4 / int(den))
    if '//' in dur:
        d = ratio_to_dur(*dur.split('//'))
    elif '/' in dur:
        d = ratio_to_dur(*dur.split('/'))
    else:
        d = int(dur)
    return d * (60 / tempo)


def quarters_to_timesig(quarters:float, snap=True, mindiv=64) -> t.Tuple[int, int]:
    """
    Transform a (

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
    f = R.from_float(quarters).limit_denominator(mindenom)
    timesig0 = f.numerator, f.denominator*4
    transforms = {
        (1, 4):(2, 8),
        (2, 4):(4, 8)
    }
    timesig = transforms.get(timesig0, timesig0)
    return timesig


def possible_timesigs(tempo):
    """
    Return possible timesignatures for a given tempo

    Time signatures are given in fractions where 2.5 means 5/8, 3.5 means 7/8
    it is assumed that tempo refers to a quarter note
    """
    fractional_timesigs = [1.5, 2.5, 3.5, 4.5]
    int_timesigs = [2, 3, 4, 5, 6, 7, 8, 9]
    if tempo > 80:
        return int_timesigs
    return sorted(int_timesigs + fractional_timesigs)


@returns_tuple("best allsolutions")
def best_timesig(duration, tempo=60, possibletimesigs=None, maxmeasures=1,
                 tolerance=0.25):
    """
    possibletimesigs: a timesig is defined by a float where
                      1 = 1 * fig defining the tempo.
                      So if tempo=60, 1 is one beat of dur. 60
                      Assuming that tempo is defined for the quarter note,
                      1 = 1/4
                      1.5 = 3/8
                      3.5 = 7/8
                      4 = 4/4
                      etc.

                        If not given, a sensible default is assumed
    if maxmeasures > 1: solutions with multiple measures combined are searched
    """
    timesigs = possibletimesigs or possible_timesigs(tempo)
    assert (isinstance(timesigs, (list, tuple)) and
            all(isinstance(t, Number) for t in timesigs))
    if maxmeasures > 1:
        return _besttimesig_with_combinations(duration, tempo, timesigs,
                                              tolerance=tolerance,
                                              maxcombinations=maxmeasures)
    res = [(abs(timesig * (60 / tempo) - duration), timesig) for timesig in timesigs]
    res.sort()
    solutions = [r[1] for r in res]
    solutions = [sol for sol in solutions
                 if abs(sol * 60. / tempo - duration) <= tolerance]
    if not solutions:
        warnings.warn("No solution for the given tolerance. Try a different tempo")
        return None, None
    best = solutions[0]
    return best, solutions


def _besttimesig_with_combinations(duration, tempo, timesigs, maxcombinations=3,
                                   tolerance=0.25):
    assert isinstance(duration, (int, float, R))
    assert isinstance(tempo, (int, float, R)) and tempo > 0
    assert isinstance(timesigs, (tuple, list))
    assert isinstance(maxcombinations, int) and maxcombinations > 0
    import constraint
    p = constraint.Problem()
    possibledurs = [t * (60.0 / tempo) for t in timesigs]
    possibledurs += [0]
    V = range(maxcombinations)
    p.addVariables(V, possibledurs)

    def objective(solution):
        # this is the function to MINIMIZE
        values = solution.values()
        numcombinations = sum(value > 0 for value in values)
        # TODO: use timesig complexity here to make a more intelligent choice
        return numcombinations

    p.addConstraint(constraint.MinSumConstraint(duration - tolerance))
    p.addConstraint(constraint.MaxSumConstraint(duration + tolerance))
    solutions = p.getSolutions()
    if not solutions:
        warnings.warn("No solutions")
        return None
    solutions.sort(key=objective)

    def getvalues(solution):
        values = [value for name, value in sorted(solution.items()) if value > 0]
        values.sort()
        return tuple(values)

    solutions = map(getvalues, solutions)
    best = solutions[0]
    solutions = set(solutions)

    return best, solutions


