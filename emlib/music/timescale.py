import constraint
from emlib.iterlib import pairwise, window
from emlib import lib
import time
from math import sqrt, inf
import bpf4 as bpf
import typing as t
import copy
import logging

logger = logging.getLogger("emlib.timescale")


default = {
    'values': [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5] + list(range(6, 40))
}


def indexDistance(seq, elem0, elem1, exact=True):
    """
    Return the distance in indexes between elem1 and elem0
    if exact:
        assume that elem0 and elem1 are present in seq, otherwise the nearest
        element in seq is used
    """
    if not exact:
        elem0 = lib.nearest_element(elem0, seq)
        elem1 = lib.nearest_element(elem1, seq)
    return seq.index(elem1) - seq.index(elem0)


class Solver:
    def __init__(self, *, values=None, dur=None, absError=None, relError=None, timeout=None, fixedslots=None, 
                 maxIndexJump=None, maxRepeats=None, maxSlotDelta=None, minSlotDelta=None, 
                 monotonous='up', minvalue=None, maxvalue=None, callback=None):
        """
        Partition dur into timeslices

        dur            sum of all values
        values         possible values         
        maxIndexJump   max index distance between two consecutive slots
        absError       absolute error of dur
        relError       relative error (only one of absError or relError should be given)
        timeout        timeout for the solve function, in seconds
        fixedslots     a dictionary of the form {0: 0.5, 2: 3} would speficy that the 
                       slot 0 should have a value of 0.5 and the slot 2 a value of 3
        maxIndexJump   max. distance, in indices, between two slots
        maxRepeats     how many consecutive slots can have the same value
        maxSlotDelta   the max. difference between two slots
        minSlotDelta   the min. difference between two slots
        monotonous     possible values: 'up', 'down'. It indicates that all values 
                       should grow monotonously in the given direction
        minvalue       min. value for a slot
        maxvalue       max. value for a slot
                       These are convenience values, we could just filter values (from param. `values`)
                       which fall between these constraints (in fact this is what we do)
        callback       DEPRECATED
                       a function of the form callback(problem, slots) -> None. 
                       It should add constraints via problem.addConstraint  
        """
        self.dur = dur
        self.values = values or default['values'] 
        self.absError = absError
        self.fixedslots = fixedslots
        self.timeout = timeout
        self.maxIndexJump = maxIndexJump
        self.maxRepeats = maxRepeats
        self.maxSlotDelta = maxSlotDelta
        self.minSlotDelta = minSlotDelta
        self.monotonous = monotonous
        self.relError = relError
        self._constraintCallbacks = []

        minvalue = minvalue if minvalue is not None else -inf
        maxvalue = maxvalue if maxvalue is not None else inf
        self.values = [v for v in self.values if minvalue <= v <= maxvalue]
        if callback:
            logger.warning("callback param is deprecated. Use solver = Solver(....).addCallback(callback)")
            self.addCallback(callback)
    
    def copy(self):
        return copy.copy(self)

    def clone(self, **kws):
        out = self.copy()
        for key, val in kws.items():
            setattr(out, key, val)
        return out

    def solve(self, numslots):
        values = self.values
        dur = self.dur
        timeout = self.timeout
        problem = constraint.Problem()
        slots = list(range(numslots))
        problem.addVariables(slots, values)
        if dur is not None:
            if self.relError is None and self.absError is None:
                absError = min(values)
            elif self.absError is None:  
                absError = self.relError * dur
            elif self.relError is None:
                absError = self.absError
            else:
                absError = min(self.absError, self.relError*dur)    
            problem.addConstraint(constraint.MinSumConstraint(dur-absError))
            problem.addConstraint(constraint.MaxSumConstraint(dur+absError))
        
        if self.fixedslots:
            for idx, slotdur in self.fixedslots.items():
                try:
                    slot = slots[idx]
                    problem.addConstraint(lambda s, slotdur=slotdur: s==slotdur, variables=[slot])
                except IndexError:
                    pass
        
        self.applyConstraints(problem, slots)

        for callback in self._constraintCallbacks:
            callback(problem, slots)

        solutions = []
        t0 = time.time()
        for sol in problem.getSolutionIter():
            if not sol: continue
            vals = [sol[k] for k in range(numslots)]
            solutions.append(vals)
            if timeout and time.time() - t0 > timeout:
                print("timeout!")
                break
        return solutions

    def applyConstraints(self, problem, slots):
        constr = problem.addConstraint
        if self.monotonous is not None:
            if self.monotonous == 'up':
                for s0, s1 in pairwise(slots):
                    constr(lambda s0, s1:  s0 <= s1, variables=[s0, s1])
            elif self.monotonous == 'down':
                for s0, s1 in pairwise(slots):
                    constr(lambda s0, s1:  s0 >= s1, variables=[s0, s1])
            else:
                raise ValueError("monotonous should be 'up' or 'down'")
        if self.minSlotDelta is not None:
            for s0, s1 in pairwise(slots):
                constr(lambda s0, s1: abs(s1 - s0) >= self.minSlotDelta, variables=[s0, s1])
        if self.maxIndexJump is not None:
            for s0, s1 in pairwise(slots):
                constr(lambda s0, s1: abs(indexDistance(self.values, s0, s1)) <= self.maxIndexJump, variables=[s0, s1])
        if self.maxRepeats is not None:
            for group in window(slots, self.maxRepeats + 1):
                constr(lambda *values: len(set(values)) > 1, variables=group)
        if self.maxSlotDelta is not None:
            for s0, s1 in pairwise(slots):
                constr(lambda s0, s1: abs(s1 - s0) <= self.maxSlotDelta, variables=[s0, s1])

    def addCallback(self, func):
        """
        def growing(problem, slots):
            for s0, s1 in pairwise(slots):
                problem.addConstrain(lambda s0, s1: s0 < s1, variables=[s0, s1])
        solver.addCallback(growing)
        """
        self._constraintCallbacks.append(func)
        return self


def asCurve(curve) -> bpf.BpfInterface:
    if isinstance(curve, (int, float)):
        return bpf.expon(0, 0, 1, 1, exp=curve)
    return bpf.asbpf(curve)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             Extending a Solver
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


"""
There are 3 ways to extend a Solver:
    1) Subclass
    2) Add constraint callbacks

Add constraint callbacks
~~~~~~~~~~~~~~~~~~~~~~~~

foo = 0.5
def separateByFoo(problem, dur, slots):
    for s0, s1 in pairwise(slots):
        problem.addConstraint(lambda s0, s1: s1 - s0 >= foo, variables=[s0, s1])
solver.addConstraintCallback(separatedByFoo)
"""


class Rating(t.NamedTuple):
    name: str
    weight: float
    func: t.Callable
    exp: float = 1.0


class Solution(t.NamedTuple):
    slots: t.List[float]
    score: float = 0.0
    data: t.Optional[dict] = None


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

    def __init__(self, relcurve=None, abscurve=None, errorWeight=1, varianceWeight=1, curveWeight=3, curveExp=1):
        """
        relcurve: a bpf defined between x:0-1, y:0-1, or the exponent of an exponential curve
        """
        self.relcurve = relcurve
        self.abscurve = abscurve
        self.varianceWeight = varianceWeight
        self.curveWeight = curveWeight
        self.curveExp = curveExp
        self._ratings = []
        self._postinit()

    def _postinit(self):
        self.relcurve = asCurve(self.relcurve) if self.relcurve is not None else None
        self.abscurve = asCurve(self.abscurve) if self.abscurve is not None else None
        
    def __call__(self, solution: t.List[float]) -> Solution:
        numvalues = len(set(solution))
        ratedict = {
            'variance': (numvalues / len(solution), self.varianceWeight)
        }
        if self.relcurve is not None:
            relcurve = self.relcurve
        elif self.abscurve is not None:
            relcurve = (self.abscurve - solution[0]) / (solution[-1] - solution[0])
        else:
            relcurve = None
        
        if relcurve:
            score = rateRelativeCurve(solution, relcurve)
            ratedict['curve'] = (score**self.curveExp, self.curveWeight)

        for rating in self._ratings:
            score = rating.func(solution) ** rating.exp
            ratedict[rating.name] = (score, rating.weight)

        rates = list(ratedict.values())
        score = sqrt(sum((value**2)*weight for value, weight in rates) / sum(weight for _, weight in rates))
        data = {'ratings': ratedict}
        return Solution(slots=solution, score=score, data=ratedict)

    def clone(self, **kws):
        out = copy.copy(self)
        for k, v in kws.items():
            setattr(out, k, v)
        out._postinit()
        return out

    def addRating(self, name, weight, func):
        """
        Example: rate higher solutions which have a small error
    
        NB: put extra info in the lambda itself

        rater.addRating("minError", weight=2, 
                        func=lambda slots, dur=10, absError=2: abs(sum(slots)-dur)/absError)
        """
        self._ratings.append(Rating(name, weight, func))


def rateRelativeCurve(slots: t.List[float], relcurve: bpf.BpfInterface, plot=False) -> float:            
    solxs = lib.linspace(0, 1, len(slots))
    x0 = min(slots)
    x1 = max(slots)
    if x0 == x1:
        diff = 1
    else:
        solys = lib.linlinx(slots, min(slots), max(slots), 0, 1)
        solcurve = bpf.core.Linear(solxs, solys)
        if plot:
            solcurve.plot(), relcurve.plot(show=True)
        diff = (solcurve - relcurve).abs().integrate()
    assert diff <= 1, diff
    score = (1 - diff)
    return score


def solve(solver: Solver, numslots: t.Union[int, t.List[int]], rater: Rater=None,
          report=False, reportMaxRows=10) -> t.List[Solution]:
    """
    numslots: the number of slots to use, or a list of possible numslots

    Example
    ~~~~~~~

    values = [0.5, 1, 1.5, 2, 3, 5, 8]
    solver = Solver(values=values, dur=3, relError=0.1, monotonous='up')
    rater = Rater(relcurve=1.5)
    solutions = solve(solver=solver, numslots=4, rater=rater, report=True)
    best = solutions[0]

    """
    allsolutions = []
    possibleNumslots = numslots if lib.isiterable(numslots) else [numslots]
    for numslots in possibleNumslots:
        solutions = solver.solve(numslots)
        allsolutions.extend(solutions)
    ratedSolutions = []
    for sol in allsolutions:
        if rater is not None:
            sol = rater(sol)
        else:
            sol = Solution(sol, 0, None)
        ratedSolutions.append(sol)
    if rater is not None:
        ratedSolutions.sort(reverse=True, key=lambda solution: solution.score)
    if report:
        reportSolutions(ratedSolutions[:reportMaxRows])
    return ratedSolutions


def reportSolutions(solutions: t.List[Solution], plotbest=0, rater=None) -> None:
    """
    If given a rater, the solution will be plotted against the desired relcurve
    """
    table = []
    for solution in solutions:
        ratings = solution.data.get('ratings')
        if ratings:
            infostr = "\t".join([f"{key}: {value[0]:.3f}x{value[1]}={value[0]*value[1]:.3f}" for key, value in ratings.items()])
        else:
            infostr = ""
        row = (solution.slots, solution.score, infostr)
        table.append(row)
    lib.print_table(table)
    if plotbest and rater is not None and rater.relcurve is not None:
        for sol in solutions[:plotbest]:
            rateRelativeCurve(sol.slots, rater.relcurve)

