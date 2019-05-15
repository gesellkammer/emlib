# ----------------------------------------------------------------------------
#
#   functions to convert between dB and musical dynamics
#   also makes a representation of the amplitude in terms of musical dynamics

from bisect import bisect as _bisect
import bpf4 as _bpf
from emlib.pitchtools import db2amp, amp2db
from emlib import typehints as t
from emlib import lib

_DYNAMICS = ('pppp', 'ppp', 'pp', 'p', 'mp',
             'mf', 'f', 'ff', 'fff', 'ffff')


class DynamicsCurve(object):
    
    def __init__(self, bpf, dynamics:t.Seq[str]=None):
        """
        shape: a bpf mapping 0-1 to amplitude(0-1)
        dynamics: a list of possible dynamics, or None to use the default

        NB: see .fromdescr
        """
        self.dynamics = lib.astype(tuple, dynamics if dynamics else _DYNAMICS)
        bpf = bpf.fit_between(0, len(self.dynamics)-1)
        self._amps2dyns, self._dyns2amps = _create_dynamics_mapping(bpf, self.dynamics)
        assert len(self._amps2dyns) == len(self.dynamics)

    @classmethod
    def fromdescr(cls, shape:float, mindb:-100.0, maxdb=0.0,
                  dynamics:t.Seq[str]=None) -> 'DynamicsCurve':
        """
        shape: the shape of the mapping ('linear', 'expon(2)', etc)
        mindb, maxdb: db value of minimum and maximum amplitude
        dynamics: the list of possible dynamics, ordered from soft to loud

        Example:

        DynamicsCurve.fromdescr('expon(3)', mindb=-80, dynamics='ppp pp p mf f ff'.split())

        """
        bpf = create_shape(shape, mindb, maxdb)
        return cls(bpf, dynamics)

    def amp2dyn(self, amp:float, nearest=True) -> str:
        """
        convert amplitude to a string representation of its corresponding
        musical dynamic as defined in DYNAMIC_TABLE

        nearest: if True, it searches for the nearest dynamic. Otherwise it gives 
                 the dynamic exactly inferior

       """
        curve = self._amps2dyns
        if amp < curve[0][0]:
            return curve[0][1]
        if amp > curve[-1][0]:
            return curve[-1][1]
        insert_point = _bisect(curve, (amp, None))
        if not nearest:
            floor = max(0, curve[insert_point-1])
            return curve[floor][1]
        amp0, dyn0 = curve[insert_point - 1]
        amp1, dyn1 = curve[insert_point]
        db = amp2db(amp)
        return dyn0 if abs(db-amp2db(amp0)) < abs(db-amp2db(amp1)) else dyn1

    def dyn2amp(self, dyn:str) -> float:
        """
        convert a dynamic expressed as a string to its 
        corresponding amplitude
        """
        amp = self._dyns2amps.get(dyn.lower())
        if amp is None:
            raise ValueError("dynamic %s not known" % dyn)
        return amp

    def dyn2db(self, dyn):
        # type: (str) -> float
        return amp2db(self.dyn2amp(dyn))

    def db2dyn(self, db):
        # type: (float) -> str
        """
        """
        return self.amp2dyn(db2amp(db))

    def dyn2index(self, dyn:str) -> int:
        """
        Convert the given dynamic to an integer index
        """
        try:
            return self.dynamics.index(dyn)
        except ValueError:
            raise ValueError("Dynamic not defined, should be one of %s" % self.dynamics)

    def index2dyn(self, idx:int) -> str:        
        return self.dynamics[idx]

    def amp2index(self, amp:float) -> str:
        return self.dyn2index(self.amp2dyn(amp))

    def index2amp(self, index:int) -> float:
        # type: (int) -> float
        return self.dyn2amp(self.index2dyn(index))

    def asdbs(self, step=1) -> 't.List[float]':
        """
        Convert the dynamics defined in this curve to dBs
        """
        indices = range(0, len(self.dynamics), step)
        dbs = [self.dyn2db(self.index2dyn(index)) for index in indices]
        assert dbs 
        return dbs


def _validate_dynamics(dynamics: t.Seq[str]) -> None:
    assert not set(dynamics).difference(_DYNAMICS), \
        "Dynamics not understood"


def _create_dynamics_mapping(bpf, dynamics:t.Seq[str]=None):
    """
    Calculate the global dynamics table according to the bpf given

    * bpf: a bpf from dynamic-index to amp
    * dynamics: a list of dynamics

    Returns:

    a tuple (amps2dyns, dyns2amps), where 
        - amps2dyns is a List of (amp, dyn)
        - dyns2amps is a dict mapping dyn -> amp
    """
    if dynamics is None:
        dynamics = _DYNAMICS
    assert isinstance(bpf, _bpf.core._BpfInterface)
    _validate_dynamics(dynamics)
    dynamics_table = [(bpf(i), dyn) for i, dyn in enumerate(dynamics)]
    dynamics_dict = {dyn: ampdb for ampdb, dyn, in dynamics_table}
    return dynamics_table, dynamics_dict


def create_shape(shape='expon(3)', mindb=-90, maxdb=0) -> _bpf.BpfInterface:
    """
    Return a bpf mapping 0-1 to amplitudes, as needed to be passed
    to DynamicsCurve

    * descr: a descriptor of the curve to use to map amplitude to dynamics
    * mindb, maxdb: the maximum and minimum representable amplitudes (in dB)
    * dynamics: - a list of dynamic-strings,
                - None to use the default (from pppp to ffff)
    
    If X is dynamic and Y is amplitude, an exponential curve with exp > 1
    will allocate more dynamics to the soft amplitude range, resulting in more
    resolution for small amplitudes.
    A curve with exp < 1 will result in more resolution for high dynamics
    """
    minamp, maxamp = db2amp(mindb), db2amp(maxdb)
    return _bpf.util.makebpf(shape, [0, 1], [minamp, maxamp])
    

_default = DynamicsCurve(create_shape("expon(4.0)", -80, 0))


def amp2dyn(amp, nearest=True):
    # type: (float, bool) -> str
    return _default.amp2dyn(amp, nearest)


def dyn2amp(dyn):
    # type: (str) -> float
    return _default.dyn2amp(dyn)


def dyn2db(dyn):
    # type: (str) -> float
    return _default.dyn2db(dyn)


def db2dyn(db, nearest=True):
    # type: (float, bool) -> str
    amp = db2amp(db)
    return _default.amp2dyn(amp, nearest)
   

def dyn2index(dyn):
    # type: (str) -> int
    return _default.dyn2index(dyn)


def index2dyn(idx):
    # type: (int) -> str
    return _default.index2dyn(idx)


def set_default_curve(shape, mindb=-90, maxdb=0, possible_dynamics=None):
    global _default
    _default = DynamicsCurve(shape, mindb, maxdb, possible_dynamics)


def get_default_curve():
    return _default
