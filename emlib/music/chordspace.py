import bpf4 as bpf
from scipy import optimize
from emlib.typehints import List, Tup, Func
from functools import lru_cache
from emlib.pitchtools import n2m


_weightratio2exp = bpf.linear((1/10, 1/3), (1, 1), (10, 4))


def chordCurve(midinotes: List[float], 
               weights: List[float], 
               numiter: int, 
               weight2exp: Func[[float, float], float], 
               xs: List[float] = None,
               ) -> bpf.BpfInterface:
    
    """
    A chord curve is defined as a mapping between an abstract harmonic space (xs) and chord,
    with optional weighting of its components

    0                 C4
    1                 E4
    2                 A4
    8                 C#5+50

    midinotes: the list of the chord pitches, in ascending order
    weights: a list of corresponding weights
    numiter: determines the shape of the transition. The higher, the steeper the transition between 
             two components (see bpf.halfcos)
    weight2exp: a callback of the form (weight0, weight1) -> exponent
    xs: if given, determines the x position corresponding to each midinote.
    """
    def pairCurve(x0, midi0, weight0, x1, midi1, weight1):
        exp = weight2exp(weight0, weight1)
        return bpf.halfcos(x0, midi0, x1, midi1, exp=exp, numiter=numiter).outbound(0, 0)

    dx = 0.005
    curves = []
    if xs is None:
        xs = list(range(len(midinotes)))
    for i in range(len(midinotes)-1):
        c = pairCurve(xs[i], midinotes[i], weights[i], xs[i+1], midinotes[i+1], weights[i+1])
        curves.append(c)
    curve = bpf.core.Max(*curves)
    curve = curve[::dx]
    return curve


def parseChord(chordstr: str) -> Tup[List[float], List[float]]:
    """ 
    Given an string like "4C+ 4E 4G:3 4A 4A+:2 4B+80 7C:5 7C+:1 7E:2", interpret each
    space speparated substring as a note with an optional weight (the value after the :)

    Returns two lists of equal length: midinotes, weights

    This routine is used in conjunction with ChordAttractor
    """
    def _asmidi(n:str) -> float:
        if n[0].isalnum():
            return float(n)
        else:
            return n2m(n)
        
    def parseNote(n):
        if ":" in n:
            note, weightstr = n.split(":")
            return _asmidi(note), int(weightstr)
        return n, 1

    notes = chordstr.split()
    midinotes, weights = zip(*(parseNote(n) for n in notes))
    return midinotes, weights


class ChordAttractor:

    def __init__(self, 
                 midinotes: List[float], 
                 weights: List[float], 
                 numiter: int = 1, 
                 weightratio2exp: Func[[float], float] = None, 
                 strength=0.5, 
                 attract_width=0.46,
                 xs: List[float] = None
                 ) -> None:
        """
        midinotes: 
            the list of the chord pitches, in ascending order
        weights: 
            a list of corresponding weights
        numiter: 
            determines the shape of the transition. The higher, the steeper the transition between 
            two components (see bpf.halfcos)
        weightratio2exp: 
            a callback of the form (weightratio) -> exponent (weightratio = weight0/weight1)
        xs: 
            if given, determines the x position corresponding to each midinote.
        strength: 
            default strength 
        attract_width: 
            default width
        """
        self.midinotes = midinotes
        self.weights = weights
        self._numiter = numiter
        self._weightratio2exp = weightratio2exp or _weightratio2exp
        self._strength = strength
        self._attractWidth = attract_width
        self.curve = self._makeCurve()
        self._deriv = self.curve.derivative().abs()   # the rectified derivative
        self._inverted = self.curve.inverted()            # pitch -> step

    def _weight2exp(self, w0, w1):
        return self._weightratio2exp(w0/w1)

    def _makeCurve(self):
        return chordCurve(self.midinotes, self.weights, self._numiter, weight2exp=self._weight2exp)

    def getPitch(self, step: float, strength: float=-1) -> float:
        """
        Given a step (an abstract measure of pitch within a linear space), return the
        corresponding pitch according to the chord defined here.

        step:
            abstract pitch
        strength: a float between 0-1
            if 0, the corresponding pitch is returned, as defined in the chord curve
            if 1, the nearest component is returned.
            If given, overrides the value set at creation time
        """
        curve = self.curve
        strength = strength if strength >= 0 else self._strength
        if strength == 0:
            return curve(step)
        width = self._attractWidth
        sidewidth = width * 0.5
        stepmin = optimize.fminbound(self._deriv, step-sidewidth, step+sidewidth,
                                     xtol=0.001, disp=0)
        return curve(step * (1-strength) + stepmin*strength)

    def getStep(self, pitch: float) -> float:
        """
        Given a pitch, return the step which is mapped to that pitch
        """
        return self._inverted(pitch)

    def attract(self, pitch: float, strength: float=-1) -> float:
        """
        Return the pitch which is the result of attracting the given pitch to the defined 
        chord. 

        strength: 
            if given, overrides the attraction strength set at creation time.
            An strength of 0 just returns the given pitch, A higher strength (between
            0 and 1) will result in a pitch increasingly close to a component of this chord 
        """
        step = self.getStep(pitch)
        return self.getPitch(step, strength=strength)

    @lru_cache()
    def getPitchCurve(self, strength: float=-1, margin=2) -> bpf.BpfInterface:
        """
        Returns a bpf mapping pitch -> attracted pitch
        """
        bounds = (self.midinotes[0] - margin, self.midinotes[-1] + margin)
        return bpf.asbpf(lambda pitch: self.attract(pitch, strength=strength), bounds=bounds)
        

del List, Tup, Func


"""

Example

notes   = "4C+ 4E 4G:3 4A 4A+:2 4B+80 7C:5 7C+:1 7E:2".split()
midinotes, weights = parseChord(notes)
chordattr = ChordAttractor(midinotes, weights)
"""
