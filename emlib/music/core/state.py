from ._base import *
from .config import config
from emlib.pitchtools import m2f, set_reference_freq
from dataclasses import dataclass
from typing import Any

@dataclass
class _State:
    a4: int = 442
    tempo: Fraction = Fraction(60)
    timefactor: Fraction = Fraction(1)
    renderer: Any = None

    def __post_init__(self):
        self.timefactor = Fraction(60) / Fraction(self.tempo)
        set_reference_freq(self.a4)

    def setTempo(self, tempo):
        self.tempo = F(tempo)
        self.timefactor = F(60) / self.tempo

    def setA4(self, a4):
        self.a4 = a4
        set_reference_freq(a4)


_statestack = [_State(a4=config.get('A4', m2f(69)), tempo=F(60))]


def pushState(a4:float=None,
              tempo:time_t=None,
              renderer=None
              ) -> _State:
    """
    Push a new state to the global state stack. A new state inherits values
    not set from the earlier state

    Args:
        a4: the reference frequency
        tempo: a tempo reference
        renderer: a play.OfflineRenderer
    """
    currState = _statestack[-1]
    a4 = a4 if a4 is not None else \
         currState.a4
    tempo = F(tempo) if tempo is not None else \
            currState.tempo
    renderer = renderer if renderer is not None else currState.renderer
    state = _State(a4=a4, tempo=tempo, renderer=renderer)

    _statestack.append(state)
    if a4 != m2f(69):
        set_reference_freq(a4)
    return state


def getState() -> _State:
    """
    Get current state
    """
    return _statestack[-1]


def popState() -> _State:
    """
    Pop a global state from the stack, return the now invalid state
    """
    if len(_statestack) == 1:
        return _statestack[-1]
    laststate = _statestack.pop()
    return laststate


def setTempo(rpm:float) -> None:
    getState().setTempo(rpm)


def setA4(a4:float) -> None:
    getState().setA4(a4)