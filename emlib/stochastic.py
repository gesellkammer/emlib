from __future__ import annotations
from emlib.misc import isiterable
from itertools import product
import numpy
from collections import defaultdict
import tabulate
import random
import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


try:
    from markovchain.markov import MarkovChain as _MarkovChain
except ImportError:
    raise ImportError("This module needs XXX (url)")

logger = logging.getLogger("emlib.stochastic")

STOP = "*END*"
EXCLUDED_STATES = {STOP}


def EXCLUDE_FUNC(prestate, poststate):
    if isiterable(prestate):
        intersection = EXCLUDED_STATES.intersection(set(prestate))
        if intersection:
            return True
        return False
    return prestate in EXCLUDED_STATES


class TransitionMatrix:
    def __init__(self, prestates, poststates, probabilities, round_ndigits=None,
                 exclude_empty_rows=False):
        self.prestates = prestates
        self.poststates = poststates
        self.probabilities = probabilities
        self._table = None
        self._round = round_ndigits
        self._exclude_empty = exclude_empty_rows

    def _get_table(self, ndigits=None):
        ndigits = ndigits if ndigits is not None else self._round
        possible_prestates = []
        for prestate in self.prestates:
            possible_prestates.extend(prestate)
        if not possible_prestates:
            raise ValueError("matrix is empty")
        maxlength = max(len(s) for s in possible_prestates)
        col1sep = " -> "

        def repr_key(key):
            return key.ljust(maxlength)

        def repr_prestate(prestate):
            return col1sep.join(repr_key(state) for state in prestate)

        def repr_prob_noround(prob):
            return str(prob) if prob > 0 else '--'   

        def repr_prob_round(prob):
            return str(round(prob, ndigits)) if prob > 0 else '--'
        rows = []
        repr_prob = repr_prob_round if ndigits is not None else repr_prob_noround
        for prestate in self.prestates:
            prestate_s = repr_prestate(prestate)
            probs = [self.probabilities[prestate][poststate]
                     for poststate in self.poststates]
            probs_strs = [repr_prob(self.probabilities[prestate][poststate])
                          for poststate in self.poststates]
            if self._exclude_empty and sum(probs) == 0:
                continue
            row = [prestate_s] + probs_strs
            rows.append(row)
        cols = ['prestate'] + self.poststates
        return tabulate.tabulate(cols)

    def clone(self, round_ndigits=None, exclude_empty_rows=False):
        return TransitionMatrix(self.prestates, self.poststates, self.probabilities,
                                round_ndigits, exclude_empty_rows)

    def _get_table_repr(self):
        return self.clone(round_ndigits=4, exclude_empty_rows=True)._get_table()

    def __repr__(self): 
        return self._get_table_repr().get_string()


class MarkovChain(_MarkovChain):

    def __init__(self, order, cyclic=False, add_stop=False):
        super().__init__(order)
        self._default_cyclic = cyclic
        self._default_add_stop = add_stop
        self._transition_matrix = None

    def _validate_defaults(self, cyclic, add_stop):
        cyclic = cyclic if cyclic is not None else self._default_cyclic
        add_stop = add_stop if add_stop is not None else self._default_add_stop
        return cyclic, add_stop

    def observe_string(self, s, cyclic=None, add_stop=None):
        cyclic, add_stop = self._validate_defaults(cyclic, add_stop)
        logger.debug("%s" % s)
        words = s.split()
        return self.observe_all(words, cyclic, add_stop)

    def observe_all(self, states, cyclic=None, add_stop=None):
        cyclic, add_stop = self._validate_defaults(cyclic, add_stop)
        if add_stop:
            states.append(STOP)
        return super().observe_all(states, cyclic)

    def observe(self, prestate, poststate, exclude_func=EXCLUDE_FUNC):
        if exclude_func(prestate, poststate):
            return None
        self._clear()
        return super().observe(prestate, poststate)

    def _clear(self):
        self._transition_matrix = None

    def random_step(self, prestate):
        """
        do one transition from the prestate
        
        prestate - a tuple, the prestate where to start.
        The class does not keep track of a current "position",
        so this must be passed as a parameter.
        """
        try:
            post = super().random_step(prestate)
        except Exception:
            post = None
        return post

    def _normalize_init(self, init):
        if init is None:
            init = self.get_random_prestate()
        elif self.order == 1 and not isiterable(init):
            init = [init]
        elif self.order > 1 and not isiterable(init):
            init = self.get_random_prestate_startingwith([init])
        elif isiterable(init) and self.order > len(init):
            init = self.get_random_prestate_startingwith(list(init))
        return init

    def get_random_prestate_startingwith(self, state):
        """
        state can be a single state, a sequence of states or a
        string as STATE1:STATE2:etc

        It returns None if no state found that starts with the given state
        """
        states = _parse_state(state)
        assert isiterable(states)
        if len(states) > self.order:
            raise ValueError("The state given has a higher order than this chain")
        order = self.order
        prestates = [prestate for prestate in self.get_prestates() if prestate[:order] == states]
        return random.choice(prestates) if prestates else None

    def generate(self, maxlength, init=None, honour_stop=True):
        init = self._normalize_init(init)
        out = []
        for state in self.random_walk(maxlength, init):
            if not state:
                break
            if honour_stop and state is STOP:
                break
            out.append(state)
        return out

    def get_prestates(self):
        states = [state for state in self.get_states() if state not in EXCLUDED_STATES]
        prestates = list(product(states, repeat=self.order))
        return prestates

    def get_poststates(self):
        return self.get_states()

    def get_transition_matrix(self):
        if self._transition_matrix is not None:
            return self._transition_matrix
        prestates = self.get_prestates()
        poststates = self.get_poststates()
        probabilities = numpy.zeros((len(prestates), len(poststates)), dtype=float)
        M = defaultdict(lambda:defaultdict(float))
        for i, prestate in enumerate(prestates):
            counts = defaultdict(float)
            for poststate in self.transitions[tuple(prestate)]:
                counts[poststate] = counts[poststate] + 1
            linesum = sum(counts.values())
            for j, poststate in enumerate(poststates):
                if linesum != 0 and counts[poststate] != 0:
                    v = counts[poststate] / linesum
                else:
                    v = 0
                probabilities[i, j] = v
                M[prestate][poststate] = v
        self._transition_matrix = TransitionMatrix(prestates, poststates, M)
        return self._transition_matrix

    def print_matrix(self):
        m = self.get_transition_matrix()
        print(m)


def _aslist(x:t.Iter) -> t.List:
    if isinstance(x, list):
        return x
    elif isiterable(x):
        return list(x)
    else:
        raise TypeError("x should be iterable")


def _parse_state(state: t.U[str, t.Seq]) -> t.List[str]:
    if isiterable(state):
        return _aslist(state)
    return state.split(":")

