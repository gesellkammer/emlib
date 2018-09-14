"""
Set of routinges to work with pitch

Routines ending with suffix _np accept np arrays


if peach is present, it is used for purely numeric conversions
(see github.com/gesellkammer/peach) 
"""

from __future__ import division
import math
import re as _re
import warnings

import sys
_EPS = sys.float_info.epsilon


_A4 = 442.0


def set_reference_freq(a4: float) -> None:
    """
    set the global value for A4

    NB: you can get the current value calling m2f(69)
    """
    global _A4
    _A4 = a4
    
    
def f2m(freq: float) -> float:
    """
    Convert a frequency in Hz to a midi-note

    See also: set_reference_freq, temporary_A4
    """
    if freq < 9:
        return 0
    return 12.0 * math.log(freq/_A4, 2) + 69.0


def freqround(freq: float) -> float:
    """
    round freq to next semitone
    """
    return m2f(round(f2m(freq)))
    

def m2f(midinote: float) -> float:
    """
    Convert a midi-note to a frequency

    See also: set_reference_freq, temporary_A4

    :type midinote: float|np.ndarray
    :rtype : float
    """
    return 2**((midinote - 69) / 12.) * _A4


_notes3      = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C"]
_enharmonics = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B", "C"]


def m2n(midinote):
    # type: (float) -> str
    i = int(midinote)
    micro = midinote - i
    octave = int(midinote/12.0)-1
    ps = int(midinote % 12)
    cents = int(micro*100+0.5)
    if cents == 0:
        return str(octave) + _notes3[ps]
    elif cents == 50:
        if ps in (1, 3, 6, 8, 10):
            return str(octave) + _notes3[ps+1] + '-'
        return str(octave) + _notes3[ps] + '+'
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
            

_notes2 = {
    "c":0,
    "d":2,
    "e":4,
    "f":5,
    "g":7,
    "a":9,
    "b":11
}


def n2m(note: str) -> float:
    # first format: C#2, D+2, Db4+20
    # snd format  : 2C#, 4D+, 7Eb-14
    regexes = [
        "(?P<oct>[-]?[0-9]+)(?P<pch>[A-Ha-h][b|#]?)(?P<micro>[+|-][0-9]*)?",
        "(?P<pch>[A-Ha-h][b|#]?)(?P<oct>[-]?[0-9]+)(?P<micro>[+|-][0-9]*)?"
    ]
    for regex in regexes:
        m = _re.match(regex, note)
        if m:
            break
    else:
        raise ValueError("Could not parse note " + note)
    groups = m.groupdict()
    pitchstr = groups['pch']
    octavestr = groups['oct']
    microstr = groups['micro']
    pc = _notes2[pitchstr[0].lower()]
    if len(pitchstr) == 2:
        alt = pitchstr[1]
        if alt == "#":
            pc += 1
        elif alt == "b" or alt == "s":
            pc -= 1
        else:
            raise ValueError("Could not parse alteration in " + note)
    octave = int(octavestr)
    
    if not microstr:
        micro = 0.0
    elif microstr == '+':
        micro = 0.5
    elif microstr == '-':
        micro = -0.5
    else:
        micro = int(microstr)/100.
    
    if pc > 11:
        pc = 0
        octave += 1
    elif pc < 0:
        pc = 0
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
    return 10.0**(0.05*db)


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
    return math.log10(amp)*20


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
    

def pianofreqs(start='A0', stop='C8'):
    # type: (str, str) -> np.ndarray
    """
    Generate an array of the frequencies representing all the piano keys
    """
    from emlib import pitchnp
    return pitchnp.pianofreqs(start=start, stop=stop)


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
    return 2 ** (interval / 12.)


r2i = ratio2interval
i2r = interval2ratio 


def pitchbend2cents(pitchbend: int, maxcents=200) -> int:
    return int(((pitchbend/16383.0)*(maxcents*2.0))-maxcents+0.5)


def cents2pitchbend(cents:int, maxcents=200) -> int:
    return int((cents+maxcents)/(maxcents*2.0)* 16383.0 + 0.5)