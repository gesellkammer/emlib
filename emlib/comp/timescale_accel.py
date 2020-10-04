from __future__ import annotations
import constraint
from emlib.music import timescale
from emlib.iterlib import pairwise, window, chain
from typing import Sequence as Seq, Union
from math import inf

Number = Union[int, float]


def _indexesPerGroup(framesPerGroup: Seq[int]) -> list[list[int]]:
    """
    assert _indexesPerGroup((4, 3, 5)) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11]]

    Args:
        framesPerGroup: how many frames per group in this section

    """
    totalFrames = sum(framesPerGroup)
    allIndexes = list(range(totalFrames))
    out = []
    i = 0
    for numFrames in framesPerGroup:
        out.append(allIndexes[i:i+numFrames])
        i += numFrames
    return out

def rule(problem, slots, func):
    problem.addConstraint(func, slots)


def _pack_slots(slots: list[Number], numFramesPerGroup: list[int]) -> list[list[Number]]:
    assert len(slots) == sum(numFramesPerGroup)
    groups = []
    i = 0
    for numFrames in numFramesPerGroup:
        groups.append(slots[i:i+numFrames])
        i += numFrames
    return groups


def solvePossibleGroupSizes(numGroups:int, possibleSizes: Seq[int], initialValue:int=None
                            ) -> list[list[int]]:
    """
    Given a number of groups and possible sizes for each group, return
    all possibilities

    Args:
        numGroups: the number of groups in this section
        possibleSizes: the possible number of frames per group
        initialValue: the initial number of frames for the first group

    Returns:

    """
    problem = constraint.Problem()
    variables = list(range(numGroups))
    problem.addVariables(variables, possibleSizes)
    smallestGroup = min(possibleSizes)

    if initialValue:
        rule(problem, [variables[0]], lambda first:first == initialValue)

    rule(problem, [variables[-1]], lambda last:last>smallestGroup)
    rule(problem, [variables[0]], lambda first:first>3)

    for slots in pairwise(variables):
        rule(problem, slots, lambda a, b:abs(a-b)<=2)
        rule(problem, slots, lambda a, b:not (a == b == smallestGroup))

    for slots in window(variables, 3):
        rule(problem, slots, lambda a, b, c:not (a-b>=1 and b-c>=1))
        rule(problem, slots, lambda a, b, c:not (a-b<=-1 and b-c<=-1))
        rule(problem, slots, lambda a, b, c:not (a == b == c))
        rule(problem, slots, lambda a, b, c:not (a == c and abs(a-b) == 2))
        rule(problem, slots, lambda a, b, c:not (a == b and a != c and abs(a-c)>=2))
        rule(problem, slots, lambda a, b, c:not (a != b and b == c and abs(a-c)>=2))

    if numGroups>=4:
        for slots in window(variables, 4):
            rule(problem, slots, lambda a, b, c, d:not (a == b and c == d and abs(a-c) == 2))
            rule(problem, slots, lambda a, b, c, d:not (a == b and c == d and a<c))

    if numGroups>=5:
        for slots in window(variables, 5):
            rule(problem, slots, lambda a, b, c, d, e:not (a == b == c and b == d))

    solutions = timescale.getSolutions(problem, numSlots=numGroups)
    return solutions


def possibleSectionSubdivisions(possibleNumGroups=(3, 4), possibleGroupSizes=(3, 4, 5),
                                initialValue:int=None) -> list[list[int]]:
    """
    Agreggates all possible solutions for the subdivision of a section

    Args:
        possibleNumGroups: the possible number of groups for this section
        possibleGroupSizes: the possible number of frames per group
        initialValue: the initial number of frames for the first group

    Returns:
        a list of subdivisions of the section:
            [[4, 4, 3, 3],
             [4, 5, 3, 4],
             ...
             ]
    """
    allSolutions = []
    for numGroups in possibleNumGroups:
        groupSizes = solvePossibleGroupSizes(numGroups, possibleGroupSizes, initialValue=initialValue)
        allSolutions.extend(groupSizes)
    return allSolutions


def solveSectionDistribution(sectionDur:float,
                             possibleFrameDurs:Seq[Number],
                             framesPerGroup: Seq[int],
                             initialValue=None, endValue=None,
                             minDur=-inf, maxRepeats=2,
                             maxContiguousRepeatsInGroup=2,
                             absError=None,
                             relError=0.1,
                             timeout=0, maxSolutions=None):
    """

    Args:
        sectionDur: the section duration
        possibleFrameDurs: the possible durations a frame can have
        framesPerGroup: a seq. of frames per group, like [4, 5, 4, 4], which indicates 4 groups with the
            given number of frames per group
        initialValue: the initial length of the first frame in the section
        endValue: the length of the last frame
        minDur: a min. duration for any frame
        maxRepeats: the max. number of frames which can have the same duration in a group
        maxContiguousRepeatsInGroup: the max. number of contiguous frames with the same duration
            in a group
        absError: the absolute error of the total duration
        relError: the relative error of the total duration.
            absError and relError are mutualy exclusive
        timeout: a max. time to find the first solution
        maxSolutions: stop searching after having found this many solutions

    Returns:
        a list of solutions
    """
    totalFrames = sum(framesPerGroup)
    allSlots = list(range(totalFrames))
    maxDur = initialValue if initialValue is not None else max(possibleFrameDurs)
    minDur = max(minDur, possibleFrameDurs[0])
    possibleValues = [f for f in possibleFrameDurs if minDur<=f<=maxDur]
    possibleValues.sort()

    if sectionDur>totalFrames*maxDur:
        raise ValueError("sectionDur is too big")

    if sectionDur<totalFrames*minDur:
        raise ValueError("sectionDur is too small")

    problem = constraint.Problem()
    problem.addVariables(allSlots, possibleFrameDurs)

    if initialValue:
        rule(problem, [allSlots[0]], lambda s0:s0 == initialValue)

    if endValue:
        rule(problem, [allSlots[-1]], lambda lastSlot:lastSlot == endValue)

    if not absError and not relError:
        absError = max(0.05*sectionDur, possibleValues[1])
    elif relError:
        absError = sectionDur*relError

    if absError:
        problem.addConstraint(constraint.MaxSumConstraint(sectionDur+absError), allSlots)
        problem.addConstraint(constraint.MinSumConstraint(sectionDur-absError), allSlots)

    allGroups = _indexesPerGroup(framesPerGroup)

    for group in allGroups:
        rule(problem, group, lambda *slots:all(s0>=s1 for s0, s1 in pairwise(slots)))
        rule(problem, group, lambda *slots:all((s0-s1)/s1<=0.5 for s0, s1 in pairwise(slots)))

    if maxRepeats:
        # the repeats are counted per group
        for group in allGroups:
            numberOfDifferentValues = len(group)-maxRepeats
            rule(problem, group, lambda *slots:len(set(slots))>=numberOfDifferentValues)

    if maxContiguousRepeatsInGroup:
        for group in allGroups:
            if len(group)<=maxContiguousRepeatsInGroup:
                continue
            for slots in window(group, maxContiguousRepeatsInGroup+1):
                rule(problem, slots, lambda *slots:len(set(slots))>1)
        maxContiguousRepeats = maxContiguousRepeatsInGroup+1
        for slots in window(allSlots, maxContiguousRepeats+1):
            rule(problem, slots, lambda *slots:len(set(slots))>1)

    # the last frame of the first group should be bigger than the last frame of the section
    rule(problem, [allGroups[0][-1], allGroups[-1][-1]],
         lambda a, b:a>b)

    for g in allGroups:
        if len(g)>=4:
            for slots in window(g, 4):
                rule(problem, slots, lambda a, b, c, d:not (a == b and c == d))

    # rules between contiguous groups
    for g0, g1 in pairwise(allGroups):
        rule(problem, (g0[0], g1[0]), lambda s0, s1:s0>=s1)
        rule(problem, (g0[-1], g1[-1]), lambda s0, s1:s0>=s1)
        rule(problem, (g0[-1], g1[0]), lambda lastOfPrev, firstOfNext:firstOfNext>lastOfPrev)
        rule(problem, g0+g1,
             lambda *slots, l0=len(g0), l1=len(g1):sum(slots[0:l0])>=sum(slots[l0:l0+l1]))
        rule(problem, [g0[0], g0[-1], g1[0], g1[-1]],
             lambda g0first, g0last, g1first, g1last:g0first != g1first or g0last != g1last)
        rule(problem, [g0[0], g0[1], g1[0], g1[1]], lambda g00, g01, g10, g11:g00+g01>=g10+g11)
        rule(problem, g0[:3]+g1[:3], lambda *slots:sum(slots[:3])>sum(slots[3:]))

        if len(g0)-len(g1)>=2:
            rule(problem, (g0[1], g1[0]), lambda s0, s1:s0>s1)

        if len(g0)>=len(g1):
            rule(problem, g0[-len(g1):]+g1,
                 lambda *slots, l=len(g1):not (slots[:l] == slots[l:]))

        if len(g0)<=len(g1):
            rule(problem, g0[:2]+g1[:2], lambda a, b, c, d:not (a == c and b == d))

    firstIndexes = [group[0] for group in allGroups]
    rule(problem, firstIndexes, lambda *slots:len(set(slots))>1)

    solutions = timescale.getSolutions(problem, totalFrames, maxSolutions=maxSolutions,
                                       timeout=timeout, timeoutSearch=timeout)
    if not solutions:
        return []
    groups = [_pack_slots(solution, framesPerGroup) for solution in solutions]
    return groups


def solveSection(sectionDur, possibleFrameDurs, initialValue=None, endValue=None, relError=0.1,
                 numGroups=(3, 4), framesPerGroup=(3, 4, 5), timeout=5, maxSolutions=50):
    if isinstance(framesPerGroup, int): framesPerGroup = list[framesPerGroup]
    if isinstance(numGroups, int): numGroups = list[numGroups]
    allSectionSubdivisions = possibleSectionSubdivisions(possibleNumGroups=numGroups,
                                                         possibleGroupSizes=framesPerGroup)
    allGroups = []
    for framesPerGroup in allSectionSubdivisions:
        print(f"solving configuration {framesPerGroup}")

        try:
            groups = solveSectionDistribution(sectionDur, possibleFrameDurs=possibleFrameDurs,
                                              framesPerGroup=framesPerGroup, initialValue=initialValue,
                                              endValue=endValue, relError=relError,
                                              maxSolutions=maxSolutions,
                                              timeout=timeout)
        except timescale.TimeoutError:
            print(f"---- timedout")
            continue

        print(f"---- {len(groups)} solutions")
        allGroups.extend(groups)
    return allGroups


