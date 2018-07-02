from emlib import typehints as t
import random

def _normalize(nums: t.List[t.Rat]) -> t.List[float]:
    total = sum(nums)
    return [float(n/total) for n in nums]


class WeightedChoooser:
    def __init__(self, choices: t.List, probabilities:t.List[t.Rat]):
        self.choices = choices
        self.probs: t.List[float] = _normalize(probabilities)

    def choose(self):
        # TODO
