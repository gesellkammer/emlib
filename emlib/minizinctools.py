"""
Set of minizinc models to solve form problems

TODO: add documentation / examples
"""

from __future__ import annotations
import minizinc as mz
from typing import Union as U, List, Optional as Opt, Any


number_t = U[int, float]
solution_t = Any


class MiniZincModel:
    def __init__(self, params={}):
        self.model = mz.Model()
        self.params = params
        self.frozen = False

    def postprocessSolution(self, solution):
        """ this should be overloaded to extract the result var """
        return solution

    def addConstraint(self, c:str) -> None:
        self.model.add_string("constraint " + c)

    def solveSatisfy(self, timeout=None, numSolutions:int=0, solver="gecode"
                     ) -> list[solution_t]:
        if not self.frozen:
            self.model.add_string(f'solve satisfy;')
            self.frozen = True

        gecode = mz.Solver.lookup(solver)
        instance = mz.Instance(gecode, self.model)
        for param, value in self.params.items():
            instance[param] = value
        if numSolutions == 0 or numSolutions == "all":
            numSolutions = None
            allSolutions = True
        else:
            allSolutions = False
        result = instance.solve(nr_solutions=numSolutions, timeout=timeout, all_solutions=allSolutions)
        if result.solution is None:
            return []
        elif isinstance(result.solution, list):
            return [self.postprocessSolution(sol) for sol in result.solution]
        else:
            return [self.postprocessSolution(result.solution)]

    def solveMinimize(self, objective, solver="gecode") -> Opt[solution_t]:
        return self._solveOptimize(f'solve minimize {objective};', solver=solver)

    def solveMaximize(self, objective, solver="gecode") -> Opt[solution_t]:
        return self._solveOptimize(f'solve maximize {objective};', solver=solver)

    def _solveOptimize(self, objstr:str, solver="gecode"):
        if not self.frozen:
            self.model.add_string(objstr)
            self.frozen = True

        gecode = mz.Solver.lookup(solver)
        instance = mz.Instance(gecode, self.model)
        for param, value in self.params.items():
            instance[param] = value
        result = instance.solve()
        if result.solution is None:
            return None
        return self.postprocessSolution(result.solution)

    def solve(self, objective:str, solver="gecode", **kws):
        objectiveKind = objective.split()[0]
        if objectiveKind == "satisfy":
            return self.solveSatisfy(solver=solver, **kws)
        elif objectiveKind == "minimize":
            return self.solveMinimize(solver=solver, **kws)
        elif objectiveKind == "maximize":
            return self.solveMaximize(solver=solver, **kws)
        else:
            raise ValueError("Objective should start with satisfy, maximize or minimize")


    def __str__(self) -> str:
        lines = self.model._code_fragments
        return "\n".join(lines)