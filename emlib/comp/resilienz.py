import constraint
import typing as t

import bpf4 as bpf
from emlib.iterlib import window, flatten
from emlib import lib
from emlib.music.timescale import rateRelativeCurve, indexDistance
from math import sqrt
from emlib.music.core import Chord
from emlib.pitchtools import *
from emlib.music import combtones


default = {
    'slotValues': [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] + list(range(6, 40))
}


def _getSolutions(problem):
    out = []
    for sol in problem.getSolutionIter():
        values = list(zip(*sorted(sol.items())))[1]
        out.append(values)
    return out

class NoSolutionError(Exception):
    pass


def solveSlotsPerGroup(numGroups, numSlots, fixed=None, callback=None):
    """
    The idea is to partition a section into groups, where each group
    is a list of slots

    numGroups: a list of possible number of groups
    numSlots: a list of possible slots
    fixed: if given, a dict of idx:value, fixing the value of slot at idx
    callback: a callback to define extra constraints

    This is used as a preparatory step to solveSection

    Example 
    ~~~~~~~

    partition a section of dur. 50 into 4 or 5 groups, each
    group with a number of slots of 4, 5, or 6, but the
    first group should have 4 slots, second group should be bigger

    def myconstraints(problem, groups):
        problem.addConstraint(lambda g0, g1: g1 > g0, groups[:2])
    solveSlotsPerGroup(numGroups=[4, 5], numSlots=[4, 5, 6], fixed={0:4}, callback=myconstraints)
    """
    def solveSlots(numGroups, values):
        problem = constraint.Problem()
        variables = list(range(numGroups))
        problem.addVariables(variables, values)
        if fixed is not None:
            for idx, value in fixed.items():
                if idx < 0:
                    idx = len(variables) + idx
                print(f"Adding fix constraint: idx {idx} must be {value}")
                problem.addConstraint(lambda s, value=value: s == value, [idx])
        for s0, s1 in window(variables, 2):
            problem.addConstraint(lambda s0, s1: abs(s0-s1) <= 1, [s0, s1])
        for s0, s1, s2 in window(variables, 3):
            problem.addConstraint(lambda s0, s1, s2: abs(s2-s0) <= 1, [s0, s1, s2])
            problem.addConstraint(lambda s0, s1, s2: not(s0 == s1 == s2), [s0, s1, s2])
        problem.addConstraint(lambda butlast, last: butlast >= last, [variables[-2], variables[-1]])
        if numGroups >= 4:
            for group in window(variables, 4):
                problem.addConstraint(lambda *ss: not(ss[0] == ss[1] and ss[2]==ss[3]), group)
        if numGroups == 5:
            problem.addConstraint(lambda *ss: not(ss[0] == ss[4] and ss[1] == ss[3]))
        if numGroups >= 5:
            for group in window(variables, 5):
                problem.addConstraint(lambda *ss: not(ss[0] == ss[2] == ss[4] and ss[1] == ss[3]), group)
        if callback is not None:
            callback(problem, variables)

        return _getSolutions(problem)

    allsolutions = []
    for numGroups in numGroups:
        allsolutions.extend(solveSlots(numGroups, numSlots))
    return allsolutions


def _getIdxsPerGroup(slotsPerGroup):
    idx, out = 0, []
    for slots, nextslots in window(slotsPerGroup, 2):
        out.append([idx+j for j in range(slots)])
        idx += 1 if slots <= nextslots else slots - nextslots + 1
    out.append([idx+j for j in range(slotsPerGroup[-1])])
    return out


def _buildGroups(slots, idxsPerGroup):
    return [[slots[idx] for idx in idxs] for idxs in idxsPerGroup]


class Solution(t.NamedTuple):
    slots: t.List[float]
    groups: t.List[t.List[float]]
    score: float = 0
    ratings: t.Optional[dict] = None

    def duration(self):
        return sum(sum(group) for group in self.groups)


def _solveSection(sectionSecs:float, tempo:float, slotsPerGroup:t.List[t.List[int]], minGroupDur:float, 
                  maxSlope=None, maxIndexJump=None, maxConsecutive=2, relError=0.1, minSlot=None, 
                  slotDurs=None, callback=None
                  ) -> t.List[Solution]:
    """
    sectionSecs: 
        duration of section in seconds
    tempo: 
        tempo, in bpm (the value corresponds to the quarter)
    slotsPerGroup: 
        a list like [4, 4, 5, 4], indicating how many slots has each group. 
        (See solveSlotsPerGroup)
    minGroupDur: 
        min. duration of a group, in quarter notes
    maxSlope: 
        if given, the max. ratio between the last slot and the first slot
    relError: 
        rel. error of the final duration (0.1 equals a 10% error)
    minSlot: 
        the dur. (in quarter notes) of the shortest slot
    maxConsecutive:
        max. number of consecutive slots with same dur 
    maxIndexJump:
        max. difference between two slots, measured in index
        Given slotDurs = [0.5, 1, 2, 2.5], a jump from 0.5 to 2 would
        be 2 indices.
    """
    maxError = sectionSecs * relError
    numGroups = len(slotsPerGroup)
    sectionDur = sectionSecs * (tempo / 60)
    maxSlot = sectionDur / numGroups
    values = slotDurs or default['slotValues']
    minSlot = minSlot if minSlot is not None else min(values)

    values = [v for v in values if minSlot <= v <= maxSlot]
    idxsPerGroup = _getIdxsPerGroup(slotsPerGroup)
    idxsFlat = list(flatten(idxsPerGroup))
    numslots = max(idxsFlat) + 1
    slots = list(range(numslots))
    problem = constraint.Problem()
    problem.addVariables(slots, values)

    # ------------------ Constraints ------------------
    minvalue = min(values)
    problem.addConstraint(lambda s0: s0 == minvalue, variables=[slots[0]])
    for s0, s1 in window(slots, 2):
        problem.addConstraint(lambda s0, s1: 1 <= s1/s0 <= 1.618, variables=[s0, s1])
    
    def constrSum(*slots):
        slotsSum = sum(slots[i] for i in idxsFlat)
        return abs(slotsSum - sectionDur) < maxError
    problem.addConstraint(constrSum)
    if maxSlope is not None:
        problem.addConstraint(lambda s0, s1: s1/s0 < maxSlope, variables=[slots[0], slots[-1]])
    
    for group in idxsPerGroup:
        slotsInGroup = [slots[idx] for idx in group]
        problem.addConstraint(lambda *slots: sum(slots) >= minGroupDur, variables=slotsInGroup)
    for group0, group1 in window(idxsPerGroup, 2):
        problem.addConstraint(lambda s0, s1: s0 < s1, variables=[group0[-1], group1[-1]])
    if maxConsecutive is not None:
        for group in window(slots, maxConsecutive+1):
            problem.addConstraint(lambda *ss: len(set(ss))>1, variables=group)
    if maxIndexJump is not None:
        for s0, s1 in window(slots, 2):
            problem.addConstraint(lambda s0, s1: abs(indexDistance(values, s0, s1)) <= maxIndexJump, [s0, s1])
    if callback is not None:
        callback(problem, slots, idxsPerGroup)

    solutions = []
    for sol in problem.getSolutionIter():
        durs = list(zip(*sorted(sol.items())))[1]  # type: t.List[float]
        solutions.append(Solution(slots=durs, groups=_buildGroups(durs, idxsPerGroup)))
    return solutions


def _ascurve(curve) -> t.Optional[bpf.BpfInterface]:
    if isinstance(curve, bpf.BpfInterface):
        return curve
    elif isinstance(curve, (int, float)):
        return bpf.expon(0, 0, 1, 1, exp=curve)
    elif curve is None:
        return None
    else:
        raise TypeError(f"curve should be a bpf, the exponent of a bpf, or None, got {curve}")


class Rater:
    def __init__(self, slotsCurve=None, groupCurve=None,
                 groupCurveW=1.0, slotsCurveW=1.0, varianceW=1.0):
        """
        This class is just a helper to call rateSolution
        """
        self.slotsCurve = _ascurve(slotsCurve)
        self.groupCurve = _ascurve(groupCurve)
        self.groupCurveW = groupCurveW
        self.slotsCurveW = slotsCurveW
        self.varianceW = varianceW

    def __call__(self, solution: Solution) -> Solution:
        return rateSolution(solution, curve=self.slotsCurve, groupCurve=self.groupCurve, 
                            curveW=self.slotsCurveW, groupCurveW=self.groupCurveW, 
                            varianceW=self.varianceW)


def rateSolution(solution: Solution, curve=None, groupCurve=None,
                 groupCurveW=1.0, curveW=1.0, varianceW=1.0):
    """
    solution: the solution to rate
    curve: if given , the distance to curve is minimized per slot
    groupCurve: if given, the distance to curve is minimized, per group duration
    groupCurveW: group curve weight
    curveW: curve weight
    varianceW: variance weight
    """
    slots = solution.slots
    groups = solution.groups
    variance = len(set(slots))/len(slots)
    ratings = {'variance': (variance, varianceW)}
    if curve is not None:
        relcurve = _ascurve(curve)
        score = rateRelativeCurve(slots, relcurve)
        ratings['curve'] = (score, curveW)
    if groupCurve is not None:
        curve = _ascurve(groupCurve)
        groupdurs = [sum(group) for group in groups]
        score = rateRelativeCurve(groupdurs, curve)
        ratings['groupcrv'] = (score, groupCurveW)
    weights = [weight for rate, weight in ratings.values()]
    sumweights = sum(weights)
    score = sqrt(sum((rate*weight/sumweights)**2 
                     for k, (rate, weight) in ratings.items()
                     if not k.startswith("_")))
    return Solution(slots=slots, groups=groups, score=score, ratings=ratings)


def solveSection(possSlotsPerGroup, sectionSecs, tempo, minGroupDur, 
                 maxSlope=None, relError=0.1, minSlot=None, slotDurs=None,
                 maxIndexJump=None, maxConsecutive=2, callback=None,
                 rater=None,
                 showProgress=False,
                 report=True, reportMaxRows=10):
    """
    **This is the center of this module**
    
    possSlotsPerGroup: a list of possible slot distributions. 
        Each group has a form like [4, 4, 5, 4], which would imply 4 groups,
        the first and second with 4 slots (measures), the third with 5 slots, etc. 
        *NB*: this can be also calculated with constraints via solveSlotsPerGroup
        possSlotsPerGroup = solveSlotsPerGroup([4, 5], [4, 5, 6])
    sectionSecs: duration of the section, in seconds
    tempo: tempo of the quarter note
    minGroupDur: min. dur (in quarter notes) of a group
    maxSlope: if given, max. ratio between last slot and first slot
    relError: rel. error of the duration of the section
    minSlot: min. dur of a slot (in quarter notes)
    slotDurs: if given, the possible durations of a slot (in quarter notes)
    rater: Rater. If given, solutions will be rated and sorted with descending scores 
        (best solution comes first)
    """
    # possibleSlotsPerGroup = solveSlotsPerGroup(possGroups, possSlots)
    allsolutions = []
    for slotsPerGroup in possSlotsPerGroup:
        if showProgress:
            print(f"Solving: {slotsPerGroup}")
        solutions = _solveSection(sectionSecs, tempo=tempo, slotsPerGroup=slotsPerGroup,
                                  minGroupDur=minGroupDur,
                                  relError=relError, minSlot=minSlot,
                                  callback=callback, maxConsecutive=maxConsecutive,
                                  maxSlope=maxSlope, 
                                  maxIndexJump=maxIndexJump,
                                  slotDurs=slotDurs)
        allsolutions.extend(solutions)
    if not allsolutions:
        raise NoSolutionError("No solution!")

    if rater:
        allsolutions = [rater(sol) for sol in allsolutions]
        allsolutions.sort(reverse=True, key=lambda sol: sol.score)
    if report:
        table = [(sol.slots, sol.groups, sol.score, sum(sum(gr) for gr in sol.groups), _prettyRatings(sol.ratings))
                 for sol in allsolutions[:reportMaxRows]]
        lib.print_table(table)
    return allsolutions


def _prettyRatings(ratings: t.Dict[str, t.Tuple[float, float]]) -> str:
    return " ".join(f"{k}={int(v*1000)}x{w}" for k, (v, w) in ratings.items())


def turntableDiffReport(origRpm, *notes):
    ch0 = Chord(notes)
    ch45 = ch0.transpose(r2i(45/origRpm))
    ch33 = ch0.transpose(r2i(33.333/origRpm))
    
    def print_chord(ch, indent=8):
        s = " " * indent
        for n in ch:
            print(s, repr(n))

    def report(rpm):
        ch = ch45 if rpm == 45 else ch33
        print(f"{rpm} RPM")
        print_chord(ch)
        diffs = ch.difftones()
        if len(diffs) > 1:
            print("Difftones: ")
            print_chord(diffs)
        else:
            print("Difftones: ", diffs[0])
    report(45)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~")
    report(33)
