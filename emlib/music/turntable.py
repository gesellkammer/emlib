from emlib.pitch import m2f, f2m, n2m, r2i
from emlib.mus import Note, Chord
from emlib import typehints as t
import warnings as _warnings


Pitch = t.U[float, str, Note]


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


def ratio2shift(ratio:float):
    return (ratio - 1) * 100


def shift2ratio(shift:float):
    return (shift + 100) / 100


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
               ) -> t.List[t.Tup[int, float]]:
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

def findRatios(newPitch:Pitch, origPitch:Pitch, rpm=45, maxdev=0.1, possibleRpms=(33.33, 45)
               ) -> t.List[t.Tup[int, float]]:
    """
    Returns a list of (rpm, ratio)
    """
    shiftSolutions = findShifts(newPitch=newPitch, origPitch=origPitch, rpm=rpm, maxshift=maxdev*100, possibleRpms=possibleRpms)
    ratioSolutions = [(rpm, shift2ratio(shift)) for rpm, shift in shiftSolutions]
    return ratioSolutions
    

def findSourcePitch(sounding:Pitch, shift:float, rpm=45, origRpm=45) -> Note:
    """
    A sound was recorded at `origRpm` and is being shifted by `shift` percent. It sounds
    like `sounding`. Return which was the orginal sound recorded.
    """
    soundingFreq = m2f(_asmidi(sounding))
    origFreq = soundingFreq * (origRpm / rpm) * (100/(100+shift))
    return Note(f2m(origFreq))


def _normalizeRpm(rpm: float) -> float:
    if 33 <= rpm <= 33.34:
        rpm = 33.333
    assert rpm in (33.333, 45, 78)
    return rpm


class TurntableChord(Chord):
    def __init__(self, rpm:float, notes:t.List[Pitch], currentRpm=None, ratio=1) -> None:
        rpm = _normalizeRpm(rpm)
        origChord = Chord(notes)
        self.rpm = rpm
        self.ratio = ratio
        self.original = origChord
        self.currentRpm = currentRpm or rpm
        interval = self.interval
        notes = [note.transpose(interval) for note in origChord]
        super().__init__(notes)
        
    @property
    def interval(self):
        return r2i(self.speed)

    @property
    def speed(self):
        return (self.currentRpm / self.rpm) * self.ratio
    
    def at(self, rpm=None, ratio=1):
        rpm = rpm or self.currentRpm
        rpm = _normalizeRpm(rpm)
        return TurntableChord(self.rpm, self.original, currentRpm=rpm, ratio=ratio)

    def findRatios(self, pitch: Pitch, maxdev=0.1) -> t.List[t.Tup[int, float]]:
        if len(self) > 1:
            raise ValueError("This chord has multiple pitches")
        return findRatios(newPitch=pitch, origPitch=self.original[0], rpm=self.rpm, maxdev=maxdev)

    def findShifts(self, pitch:Pitch, maxshift=10):
        return findShifts(newPitch=pitch, origPitch = self.original[0], rpm=self.rpm, maxshift=maxshift)

    def findRatio(self, pitch: Pitch, maxdev=0.1) -> t.Opt[float]:
        """
        find ratio to make this turntable sound like pitch at the current rpm
        """
        ratios = self.findRatios(pitch=pitch, maxdev=maxdev)
        return ratios[0][1] if ratios else None