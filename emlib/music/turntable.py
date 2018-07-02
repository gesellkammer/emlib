from emlib.pitch import m2f, f2m, n2m, r2i
from emlib.mus import Note, Chord
import typing as _
import warnings as _warnings



Pitch = _.Union[float, str, Note]


def _asmidi(x:Pitch) -> float:
    """
    convert a Pitch (a notename or a midinote) to a midinote
    """
    if isinstance(x, str):
        return n2m(x)
    elif isinstance(x, Note):
        return x.midi
    if x > 127 or x <= 0:
        _warnings.warn(f"A midinote should be in the range 0-127, got: {x}")
    return x


def soundingPitch(pitch:Pitch, shift=0, rpm=45, origRpm=45) -> Note:
    """
    Return the sounding pitch

    A sound of pitch `pitch` was recorded at `origRpm` rpms on a turntable
    Return the sounding pitch if the turntable is running at `rpm` and
    has been pitchshifted by `shift` percent

    pitch   : recorded pitch (notename or midinote)
    shift   : positive or negative percent, as indicated in a turntable (+4, -8)
    rpm     : running rpm of the turntable (33 1/3, 45, 78)
    origRpm : rpm at which the pitch was recorded

    Example 1: find the pitch of a recorded sound running at rpm and
               shifted by `shift` percent

    sounding = shiftedPitch("A4", shift=0, rpm=33.333, reference=45) -> 4E-37
    """
    speed = (rpm / origRpm) * (100 + shift) / 100.0
    pitch = _asmidi(pitch)
    freq = m2f(pitch) * speed
    return Note(f2m(freq))


def shiftPercent(newPitch:Pitch, origPitch:Pitch, newRpm=45, origRpm=45) -> float:
    """
    Find the pitch shift (as percent, 0%=no shift) to turn origPitch into newPitch
    when running at newRpm
    """
    ratio = shiftRatio(newPitch=newPitch, origPitch=origPitch, newRpm=newRpm, origRpm=origRpm)
    return ratio * 100 - 100


def shiftRatio(newPitch:Pitch, origPitch:Pitch, newRpm=45, origRpm=45) -> float:
    """
    Calculate the speed ratio to turn origPitch into newPitch when running at the given rpm
    """
    newPitch = _asmidi(newPitch)
    origPitch = _asmidi(origPitch)
    frec = m2f(origPitch) * (newRpm / origRpm)
    fdes = m2f(newPitch)
    ratio = fdes / frec
    return ratio    


def findShifts(newPitch:Pitch, origPitch:Pitch, rpm=45, maxshift=10, possibleRpms=(33.33, 45)
               ) -> _.List[_.Tuple[int, float]]:
    """
    Given a recorded pitch at a given rpm, find configuration(s) (if possible)
    of rpm and shift which produces the desired pitch.

    Returns a (possibly empty) list of solutions of the form (rpm, shiftPercent)

    newPitch: (notename or midinote)
        pitch to produce
    origPitch: (notename or midinote)
        the pitch recorded on the turntable (as midinote or notename)
    rpm:
        the rpm at which the pitch is recorded at the turntable
    maxshift:
        the maximum shift (as percent, 0=no shift) either up or down
    """
    solutions = []
    newPitch = _asmidi(newPitch)
    origPitch = _asmidi(origPitch)
    for possibleRpm in possibleRpms:
        minPitch = soundingPitch(origPitch, shift=-maxshift, rpm=possibleRpm, origRpm=rpm)
        maxPitch = soundingPitch(origPitch, shift=maxshift, rpm=possibleRpm, origRpm=rpm)
        if minPitch <= newPitch <= maxPitch:
            solution = (possibleRpm, shiftPercent(newPitch, origPitch, possibleRpm, rpm))
            solutions.append(solution)
    return solutions


def findSourcePitch(sounding:Pitch, shift:float, rpm=45, origRpm=45) -> Note:
    """
    A sound was recorded at `origRpm` and is being shifted by `shift` percent. It sounds
    like `sounding`. Return which was the orginal sound recorded.
    """
    soundingFreq = m2f(_asmidi(sounding))
    origFreq = soundingFreq * (origRpm / rpm) * (100/(100+shift))
    return Note(f2m(origFreq))


def _normalizeRpm(rpm):
    if 33 <= rpm <= 33.34:
        rpm = 33.333
    assert rpm in (33.333, 45, 78)
    return rpm


class TurntableChord(Chord):
    def __init__(self, rpm, *notes):
        rpm = _normalizeRpm(rpm)
        super().__init__(notes)
        self.rpm = rpm

    def at(self, rpm, ratio=1):
        rpm = _normalizeRpm(rpm)
        finalratio = ratio * (rpm / self.rpm)
        ch = self.asChord().transpose(r2i(finalratio))
        ch.label = f"{rpm}x{int(ratio*100)}%"   
        return ch 

    def asChord(self):
        notes = [note for note in self]
        return Chord(notes)

