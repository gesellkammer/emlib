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


A4 = 442   # type: float


def set_reference_freq(a4):
    # type: (float) -> None
    """
    set the global value for A4

    NB: you can get the current value calling m2f(69)
    NB2: to set it temporarilly, for instance, to 435 Hz, use 

    with temporary_A4(435):
        ...
    """
    global A4
    A4 = a4
    try:
        import peach
        peach.set_reference_freq(a4)
    except ImportError:
        pass

    
def f2m(freq):
    # type: (float) -> float
    """
    Convert a frequency in Hz to a midi-note

    See also: set_reference_freq, temporary_A4
    """
    if freq < 9:
        return 0
    return 12.0 * math.log(freq/A4, 2) + 69.0


def f2m_np(*args, **kws):
    warnings.warn("deprecated, use emlib.pitchnp.f2m_np")
    from emlib import pitchnp
    return pitchnp.f2m_np(*args, **kws)
    

def freqround(freq):
    # type: (float) -> float
    """
    round freq to next semitone
    """
    return m2f(round(f2m(freq)))
    

def m2f(midinote):
    # type: (float) -> float
    """
    Convert a midi-note to a frequency

    See also: set_reference_freq, temporary_A4

    :type midinote: float|np.ndarray
    :rtype : float
    """
    return 2**((midinote - 69) / 12.) * A4


def m2f_np(*args, **kws):
    warnings.warn("deprecated, use emlib.pitchnp")
    from emlib import pitchnp
    return pitchnp.m2f_np(*args, **kws)


_notes = (
    '-1C', '-1C#', '-1D', '-1D#', '-1E', '-1F', 
    '-1F#', '-1G', '-1G#', '-1A', '-1A#', '-1B', 
    '0C', '0C#', '0D', '0D#', '0E', '0F', '0F#', 
    '0G', '0G#', '0A', '0A#', '0B', '1C', '1C#', 
    '1D', '1D#', '1E', '1F', '1F#', '1G', '1G#', 
    '1A', '1A#', '1B', '2C', '2C#', '2D', '2D#', 
    '2E', '2F', '2F#', '2G', '2G#', '2A', '2A#', 
    '2B', '3C', '3C#', '3D', '3D#', '3E', '3F', 
    '3F#', '3G', '3G#', '3A', '3A#', '3B', '4C', 
    '4C#', '4D', '4D#', '4E', '4F', '4F#', '4G', 
    '4G#', '4A', '4A#', '4B', '5C', '5C#', '5D', 
    '5D#', '5E', '5F', '5F#', '5G', '5G#', '5A', 
    '5A#', '5B', '6C', '6C#', '6D', '6D#', '6E', 
    '6F', '6F#', '6G', '6G#', '6A', '6A#', '6B', 
    '7C', '7C#', '7D', '7D#', '7E', '7F', '7F#', 
    '7G', '7G#', '7A', '7A#', '7B', '8C', '8C#', 
    '8D', '8D#', '8E', '8F', '8F#', '8G', '8G#', 
    '8A', '8A#', '8B', '9C', '9C#', '9D', '9D#', 
    '9E', '9F', '9F#')


def m2n_(midinote):
    # type: (float) -> str
    base = int(midinote)
    rest = midinote - base
    if rest > 0.5:
        base += 1
        rest = rest - 1
    rest = int(rest * 100 + 0.5)
    if rest == 0:
        reststr = ""  
    elif rest == 50:
        reststr = '+'
    elif rest == -50:
        reststr = '-'
    elif rest > 0:
        reststr = '+%d' % rest
    elif rest < 0:
        reststr = '-%d' % abs(rest)
    return _notes[base] + reststr


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


def n2m(note):
    # type: (str) -> float
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
    pc = _notes2.get(pitchstr[0].lower())
    if pc is None:
        raise ValueError("Could not parse pitch: %s" % str(m.groups()))
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


def f2n(freq):
    # type: (float) -> str
    """
    Convert freq. to notename
    """
    return m2n(f2m(freq))


def n2f(note):
    # type: (str) -> float
    """
    notename -> freq
    """
    m = n2m(note)
    f = m2f(m)
    return f


def db2amp(db):
    # type: (float) -> float
    """ 
    convert dB to amplitude (0, 1) 

    db: a value in dB
    """
    return 10.0**(0.05*db)


def db2amp_np(*args, **kws):
    warnings.warn("deprecated, use emlib.pitchnp")
    from emlib import pitchnp
    return pitchnp.db2amp_np(*args, **kws)


def amp2db(amp):
    # type: (float) -> float
    """
    convert amp (0, 1) to dB

    20.0 * log10(amplitude)

    :type amp: float|np.ndarray
    :rtype: float|np.ndarray

    """
    amp = max(amp, _EPS)
    return math.log10(amp)*20


def amp2db_np(*args, **kws):
    warnings.warn("deprecated, use emlib.pitchnp")
    from emlib import pitchnp
    return pitchnp.amp2db_np(*args, **kws)


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


def ratio2interval(ratio):
    # type: (float) -> float
    """
    Given two frequencies f1 and f2, calculate the interval between them
    
    f1 = n2f("C4")
    f2 = n2f("D4")
    interval = ratio2interval(f2/f1)   # --> 2 (semitones)
    """
    return 12 * math.log(ratio, 2)


def ratio2interval_np(*args, **kws):
    warnings.warn("deprecated, use emlib.pitchnp")
    from emlib import pitchnp
    return pitchnp.ratio2interval_np(*args, **kws)


def interval2ratio(interval):
    # type: (float) -> float
    """
    Calculate the ratio r so that f1*r gives f2 so that
    the interval between f2 and f1 is the given one

    f1 = n2f("C4")
    r = interval2ratio(7)  # a 5th higher
    f2 = f2n(f1*r)  # --> G4
    """
    return 2 ** (interval / 12.)


def interval2ratio_np(*args, **kws):
    warnings.warn("deprecated, use emlib.pitchnp")
    from emlib import pitchnp
    return pitchnp.interval2ratio_np(*args, **kws)


r2i = ratio2interval
i2r = interval2ratio 


def pitchbend2cents(pitchbend, maxcents=200):
    return int(((pitchbend/16383.0)*(maxcents*2.0))-maxcents+0.5)


def cents2pitchbend(cents, maxcents=200):
    return int((cents+maxcents)/(maxcents*2.0)* 16383.0 + 0.5)