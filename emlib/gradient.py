from __future__ import annotations
import bpf4 as bpf
import bisect
from emlib.interpol import interpol_linear


class Linear(object):
    def __init__(self, xs, bpfs):
        self.xs = xs
        self.bpfs = bpfs
        self.x0 = xs[0]
        self.x1 = xs[-1]

    @classmethod
    def frompairs(cls, *pairs):
        xs, bpfs = zip(*pairs)
        return cls(xs, bpfs)

    def __call__(self, x, y):
        if x <= self.x0:
            return self.bpfs[0](y)
        elif x >= self.x1:
            return self.bpfs[-1](y)
        else:
            index1 = bisect.bisect(self.xs, x)
            index0 = index1 - 1
            bpf0 = self.bpfs[index0]
            bpf1 = self.bpfs[index1]
            x0 = self.xs[index0]
            x1 = self.xs[index1]
            y = interpol_linear(x, x0, x1, bpf0(y), bpf1(y))
            return y


