"""
Set of routinges to work with pitch

Routines ending with suffix _np accept np arrays


if peach is present, it is used for purely numeric conversions
(see github.com/gesellkammer/peach) 
"""

from __future__ import annotations
import math
import re as _re
import warnings
from typing import Tuple, List

import sys

_EPS = sys.float_info.epsilon


A4 = 442.0


def set_reference_freq(a4: float) -> None:
    """
    set the global value for A4
    """
    global A4
    A4 = a4


def f2m(freq: float) -> float:
    """
    Convert a frequency in Hz to a midi-note

    See also: set_reference_freq, temporaryA4
    """
    if freq < 9:
        return 0
    return 12.0 * math.log(freq / A4, 2) + 69.0


def freqround(freq: float) -> float:
    """
    round freq to next semitone
    """
    return m2f(round(f2m(freq)))


def m2f(midinote: float) -> float:
    """
    Convert a midi-note to a frequency

    See also: set_reference_freq, temporaryA4

    :type midinote: float|np.ndarray
    :rtype : float
    """
    return 2 ** ((midinote - 69) / 12.0) * A4


_flats = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B", "C"]
_sharps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"]


def _pitchname(pitchidx: int, micro: float) -> str:
    """
    Given a pitchindex (0-11) and a microtonal alteracion (between -0.5 and +0.5),
    return the pitchname which better represents pitchindex

    0, 0.4      -> C
    1, -0.2     -> Db
    3, 0.4      -> D#
    3, -0.2     -> Eb
    """
    blacknotes = {1, 3, 6, 8, 10}
    if micro < 0:
        if pitchidx in blacknotes:
            return _flats[pitchidx]
        else:
            return _sharps[pitchidx]
    elif micro == 0:
        return _sharps[pitchidx]
    else:
        if pitchidx in blacknotes:
            return _sharps[pitchidx]
        return _flats[pitchidx]


def parse_midinote(midinote: float) -> Tuple[int, float, int, str]:
    """
    Convert a midinote into its pitch components:
        pitchindex, alteration, octave, pitchname

    63.2   -> (3, 0.2, 4, "D#")
    62.8   -> (3, -0.2, 4, "Eb")
    """
    i = int(midinote)
    micro = midinote - i
    octave = int(midinote / 12.0) - 1
    ps = int(midinote % 12)
    cents = int(micro * 100 + 0.5)
    if cents == 50:
        if ps in (1, 3, 6, 8, 10):
            ps += 1
            micro = -0.5
        else:
            micro = 0.5
    elif cents > 50:
        micro = micro - 1.0
        ps += 1
        if ps == 12:
            octave += 1
            ps = 0
    pitchname = _pitchname(ps, micro)
    return ps, round(micro, 2), octave, pitchname


_notes3 = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"]
_enharmonics = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B", "C"]


def m2n(midinote):
    # type: (float) -> str
    i = int(midinote)
    micro = midinote - i
    octave = int(midinote / 12.0) - 1
    ps = int(midinote % 12)
    cents = int(micro * 100 + 0.5)
    if cents == 0:
        return str(octave) + _notes3[ps]
    elif cents == 50:
        if ps in (1, 3, 6, 8, 10):
            return str(octave) + _notes3[ps + 1] + "-"
        return str(octave) + _notes3[ps] + "+"
    elif cents == 25:
        return str(octave) + _notes3[ps] + "*"
    elif cents == 75:
        ps += 1
        if ps > 11:
            octave += 1
        if ps in (1, 3, 6, 8, 10):
            return str(octave)+ _enharmonics[ps] + "~"
        else:
            return str(octave) + _notes3[ps] + "~"
    elif cents > 50:
        cents = 100 - cents
        ps += 1
        if ps > 11:
            octave += 1
        if cents > 9:
            return "%d%s-%d" % (octave, _enharmonics[ps], cents)
        else:
            return "%d%s-0%d" % (octave, _enharmonics[ps], cents)
    else:
        if cents > 9:
            return "%d%s+%d" % (octave, _notes3[ps], cents)
        else:
            return "%d%s+0%d" % (octave, _notes3[ps], cents)


_notes2 = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

_r1 = _re.compile(r"(?P<pch>[A-Ha-h][b|#]?)(?P<oct>[-]?[\d]+)(?P<micro>[+|-][\d]*)?")
_r2 = _re.compile(r"(?P<oct>[-]?\d+)(?P<pch>[A-Ha-h][b|#]?)(?P<micro>[-|+|\*|~]\d*)?")


def n2m(note: str) -> float:
    """
    # first format: C#2, D4, Db4+20, C4*, Eb5~
    # snd format  : 2C#, 4D+, 7Eb-14

    + = 1/4 note sharp
    - = 1/4 note flat
    * = 1/8 note sharp
    ~ = 1/8 note flat
    """
    if note[0].isalpha():
        m = _r1.search(note)
    else:
        m = _r2.search(note)
    # m = _r1.search(note) or _r2.search(note)
    if not m:
        raise ValueError("Could not parse note " + note)
    groups = m.groupdict()
    pitchstr = groups["pch"]
    octavestr = groups["oct"]
    microstr = groups["micro"]

    pc = _notes2[pitchstr[0].lower()]

    if len(pitchstr) == 2:
        alt = pitchstr[1]
        if alt == "#":
            pc += 1
        elif alt == "b":
            pc -= 1
        else:
            raise ValueError("Could not parse alteration in " + note)
    octave = int(octavestr)
    if not microstr:
        micro = 0.0
    elif microstr == "+":
        micro = 0.5
    elif microstr == "-":
        micro = -0.5
    elif microstr == "*":
        micro = 0.25
    elif microstr == "~":
        micro = -0.25
    else:
        micro = int(microstr) / 100.0

    if pc > 11:
        pc = 0
        octave += 1
    elif pc < 0:
        pc = 12 + pc
        octave -= 1
    return (octave + 1) * 12 + pc + micro


def f2n(freq: float) -> str:
    """
    Convert freq. to notename
    """
    return m2n(f2m(freq))


def n2f(note: str) -> float:
    """
    notename -> freq
    """
    m = n2m(note)
    f = m2f(m)
    return f


def db2amp(db: float) -> float:
    """ 
    convert dB to amplitude (0, 1) 

    db: a value in dB
    """
    return 10.0 ** (0.05 * db)


def db2amp_np(*args, **kws):
    warnings.warn("deprecated, use emlib.pitchnp")
    from emlib import pitchnp

    return pitchnp.db2amp_np(*args, **kws)


def amp2db(amp: float) -> float:
    """
    convert amp (0, 1) to dB

    20.0 * log10(amplitude)

    :type amp: float|np.ndarray
    :rtype: float|np.ndarray

    """
    amp = max(amp, _EPS)
    return math.log10(amp) * 20


def logfreqs(notemin=0, notemax=139, notedelta=1.0):
    """
    Return a list of frequencies corresponding to the pitch range given

    notemin, notemax, notedelta: as used in arange (notemax is included)

    Examples:

    1) generate a list of frequencies of all audible semitones
    
    >>> logfreqs(0, 139, notedelta=1)

    2) Generate a list of frequencies of instrumental 1/4 tones

    >>> logfreqs(n2m("A0"), n2m("C8"), 0.5) 
    """
    from emlib import pitchnp

    return pitchnp.logfreqs(notemin=notemin, notemax=notemax, notedelta=notedelta)


def pianofreqs(start="A0", stop="C8") -> List[float]:
    """
    Generate an array of the frequencies representing all the piano keys

    Args:
        start: the starting note
        stop: the ending note

    Returns:
        a list of frequencies 
    """
    m0 = n2m(start)
    m1 = n2m(stop)
    midinotes = range(m0, m1+1)
    freqs = [m2f(m) for m in midinotes]
    return freqs
    

def ratio2interval(ratio: float) -> float:
    """
    Given two frequencies f1 and f2, calculate the interval between them
    
    f1 = n2f("C4")
    f2 = n2f("D4")
    interval = ratio2interval(f2/f1)   # --> 2 (semitones)
    """
    return 12 * math.log(ratio, 2)


def interval2ratio(interval: float) -> float:
    """
    Calculate the ratio r so that f1*r gives f2 so that
    the interval between f2 and f1 is the given one

    f1 = n2f("C4")
    r = interval2ratio(7)  # a 5th higher
    f2 = f2n(f1*r)  # --> G4
    """
    return 2 ** (interval / 12.0)


r2i = ratio2interval
i2r = interval2ratio


def pitchbend2cents(pitchbend: int, maxcents=200) -> int:
    return int(((pitchbend / 16383.0) * (maxcents * 2.0)) - maxcents + 0.5)


def cents2pitchbend(cents: int, maxcents=200) -> int:
    return int((cents + maxcents) / (maxcents * 2.0) * 16383.0 + 0.5)


_centsrepr = {
    '+': 50,
    '-': -50,
    '*': 25,
    '~': -25
}

def split_notename(notename: str) -> Tuple[int, str, int, int]:
    """
    Return (octave, letter, alteration (1=#, -1=b), cents)

    4C#+10  -> (4, "C", 1, 10)
    Eb4-15  -> (4, "E", -1, -15)
    """

    def parse_centstr(centstr: str) -> int:
        if not centstr:
            return 0
        cents = _centsrepr.get(centstr)
        if cents is None:
            cents = int(centstr)
        return cents

    if not notename[0].isdecimal():
        # C#4-10
        cursor = 1
        letter = notename[0]
        l1 = notename[1]
        if l1 == "#":
            alter = 1
            octave = int(notename[2])
            cursor = 3
        elif l1 == "b":
            alter = -1
            octave = int(notename[2])
            cursor = 3
        else:
            alter = 0
            octave = int(notename[1])
            cursor = 2
        centstr = notename[cursor:]
        cents = parse_centstr(centstr)
    else:
        # 4C#-10
        octave = int(notename[0])
        letter = notename[1]
        rest = notename[2:]
        cents = 0
        alter = 0
        if rest:
            r0 = rest[0]
            if r0 == "b":
                alter = -1
                centstr = rest[1:]
            elif r0 == "#":
                alter = 1
                centstr = rest[1:]
            else:
                centstr = rest
            cents = parse_centstr(centstr)
    return octave, letter.upper(), alter, cents


def split_cents(notename: str) -> "tuple[str, int]":
    """
    given a note of the form 4E- or 5C#+10, it should return (4E, -50) and (5C#, 10)
    """
    octave, letter, alter, cents = split_notename(notename)
    alterchar = "b" if alter == -1 else "#" if alter == 1 else ""
    return str(octave) + letter + alterchar, cents


def normalize_notename(notename: str) -> str:
    return m2n(n2m(notename))


def str2midi(s: str):
    """
    Accepts all that n2m accepts but with the addition of frequencies

    Possible values:

    "100hz", "200Hz", "4F+20hz", "8C-4hz"

    The hz part must be at the end
    """
    ending = s[-2:]
    if ending != "hz" and ending != "Hz":
        return n2m(s)
    freq = 0
    srev = s[::-1]
    minusidx = srev.find("-")
    plusidx = srev.find("+")
    if minusidx < 0 and plusidx < 0:
        return f2m(float(s[:-2]))
    if minusidx > 0 and plusidx > 0:
        if minusidx < plusidx:
            freq = -float(s[-minusidx:-2])
            notename = s[:-minusidx-1]
        else:
            freq = float(s[-plusidx:-2])
            notename = s[:-plusidx-1]
    elif minusidx > 0:
        freq = -float(s[-minusidx:-2])
        notename = s[:-minusidx-1]
    else:
        freq = float(s[-plusidx:-2])
        notename = s[:-plusidx-1]
    return f2m(n2f(notename) + freq)


def freq2mel(freq: float) -> float:
    return 1127.01048 * math.log(1. + freq/700)


def mel2freq(mel:float) -> float:
    return 700. * (math.exp(mel / 1127.01048) - 1.0)
