import sys
import os
import ctcsound
import uuid as _uuid
from typing import Dict, List, Optional as Opt
import ctypes as _ctypes
import atexit as _atexit
import textwrap as _textwrap
import logging
from emlib import conftools
from emlib.snd import csound
from string import Template as _Template
from collections import namedtuple as _namedtuple
from contextlib import contextmanager as _contextmanager

import time
import json
import signal

__all__ = [
    'makeInstr',
    'getInstr',
    'unschedAll',
    'getManager',
    'availableInstrs',
    'stopEngine',
    'getEngine',
    'activeEngines',
    'config',
    'logger',
    'testout'
]


modulename = 'emlib.csoundengine'

logger = logging.getLogger(modulename)

_defaultconfig = {
    'sr': 0,           # 0 indicates the default sr of the backend. Will fallback to 44100 if the backend has no sr (portaudio, alsa)
    'numchannels': 0,  # 0 indicates the maximum channels available
    'ksmps': 64,
    'linux.backend': 'jack',
    'fallback_backend': 'portaudio',
    'A4': 442,
    'multisine.maxosc': 200,
    'fail_if_unmatched_pargs': False,
    'wait_poll_interval': 0.020
}

_validator = {
    'linux.backend::choices': ['jack', 'portaudio', 'pulse', 'alsa'],
    'fallback_backend::choices': ['portaudio'],
    'numchannels::range': (1, 128),
    'sr::choices': [0, 22050, 24000, 44100, 48000, 88200, 96000],
    'ksmps::choices': [16, 32, 64, 128, 256],
    'A4::range': (410, 460)
}

config = conftools.ConfigDict(modulename.replace(".", ":"), default=_defaultconfig, validator=_validator)
        

_SynthDef = _namedtuple("_SynthDef", "qname instrnum")


_MYFLTPTR = _ctypes.POINTER(ctcsound.MYFLT)

_csound_reserved_instrnum = 100
_csound_reserved_instr_turnoff = _csound_reserved_instrnum + 0


_csd:str = _Template("""
sr     = {sr}
ksmps  = {ksmps}
nchnls = {nchnls}
0dbfs  = 1
a4     = {a4}

instr _notifyDealloc
    iwhich = p4
    outvalue "__dealloc__", iwhich
    turnoff
endin

instr ${instr_turnoff}
    iwhich = p4
    turnoff2 iwhich, 4, 1
    turnoff
endin
""").safe_substitute(
    instr_turnoff=_csound_reserved_instr_turnoff)


_registry = {}


def fluidsf2Path():
    """
    Returns the path of the fluid sf2 file
    """
    key = 'fluidsf2_path'
    path = _registry.get(key)
    if path:
        return path
    if sys.platform == 'linux':
        path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
    else:
        raise RuntimeError("only works for linux right now")
    _registry[key] = path
    return path
   

def testout(dur=20, nchnls=2, group="default", sr:int=None, backend:str=None, outdev="dac"):
    """
    Test audio out
    """
    body = """
        iperiod = 1
        kchn init -1
        ktrig metro 1/iperiod
        kchn = (kchn + ktrig) % nchnls 
        anoise pinker
        outch kchn+1, anoise*0.1
        printk2 kchn
    """
    getEngine(name=group, nchnls=nchnls, sr=sr, backend=backend, outdev=outdev)
    instr = makeInstr(name="testout", body=body, group=group)
    return instr.play(dur=dur)


def _sigint_handler(sig, frame):
    raise KeyboardInterrupt("SIGINT (CTRL-C) while waiting")


def _set_sigint_handler():
    if _registry.get('sigint_handler_set'):
        return 
    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _sigint_handler)
    _registry['original_sigint_handler'] = original_handler
    _registry['sigint_handler_set'] = True


def _remove_sigint_handler():
    if not _registry.get('sigint_handler_set'):
        return 
    signal.signal(signal.SIGINT, _registry['original_sigint_handler'])
    _registry['sigint_handler_set'] = False


@_contextmanager
def safe_sigint():
    _set_sigint_handler()
    try:
        yield
    except:
        raise  # Exception is dropped if we don't reraise it.
    finally:
        _remove_sigint_handler()
        

class CsoundError(Exception): 
    pass


class CsoundEngine:

    tmpdir = "/tmp/emlib.csoundEngine"

    def __init__(self, sr:int=None, ksmps:int=None, backend:str=None, outdev="dac", a4:int=None, nchnls:int=None,
                 name:str=None, oscport:int=0):
        """
        NB: don't create instances directly, call getEngine
        """
        cfg = config
        backend = backend if backend is not None else cfg[f'{sys.platform}.backend']
        backends = csound.get_audiobackends()
        if backend not in backends:
            # should we fallback?
            fallback_backend = cfg['fallback_backend']
            if not fallback_backend:
                raise CsoundError(f"The backend {backend} is not available, no fallback backend defined")
            logger.error(f"The backend {backend} is not available. Fallback backend: {fallback_backend}")
            backend = fallback_backend
        sr = sr if sr is not None else cfg['sr']
        if sr == 0:
            sr = csound.get_sr(backend)
            if not sr:
                # failed to get sr for backend
                sr = 44100
                logger.error("Failed to get sr for backend {backend}, falling back to default sr: {sr}")
        if a4 is None:
            a4 = cfg['A4']
        if ksmps is None:
            ksmps = cfg['ksmps']
        self.sr = sr
        self.backend = backend
        self.a4 = a4
        self.ksmps = ksmps
        self.outdev = outdev
        self.uuid = _getUUID()
        self.name = name or "Unnamed"
        self.nchnls = nchnls if nchnls is not None else cfg['numchannels']
        self.oscport = oscport
        self.qualifiedName = f"{self.name}:{self.uuid}"
        self._fracnumdigits = 4
        self._cs = None
        self._pt = None
        self._exited = False
        self._csdstr = _csd
        self._instcounter = {}
        self._instrRegistry = {}
        self._outcallbacks = {}
        self._isOutCallbackSet = False
        self._globalcode = set()
        self._start()
        
    def __repr__(self):
        return f"CsoundEngine(name={self.name}, backend={self.backend}, out={self.outdev}, nchnls={self.nchnls}"

    def __del__(self):
        self.stop()

    @staticmethod
    def getTempDir():
        return CsoundEngine.tmpdir

    def _writeEngineInfo(self):
        info = self._getInfo()
        filename = self._getInfoPath()
        with open(filename, "w") as f:
            json.dump(info, f)

    def _getInfoPath(self):
        tmpdir = self.getTempDir()
        basename = self.qualifiedName + '.json'
        filename = os.path.join(tmpdir, basename)
        return filename

    def _getInfo(self):
        return {'name': self.name, 'uuid': self.uuid, 'oscport': self.oscport}

    def _getInstance(self, instnum):
        n = self._instcounter.get(instnum, 0)
        n += 1
        self._instcounter[instnum] = n
        return n

    def _getFractionalInstance(self, num, instance):
        frac = (instance / (10**self._fracnumdigits)) % 1
        return num + frac
        
    def _startCsound(self):        
        cs = ctcsound.Csound()
        orc = self._csdstr.format(sr=self.sr, ksmps=self.ksmps, nchnls=self.nchnls, backend=self.backend, a4=self.a4)
        options = ["-d", "-odac", "-+rtaudio=%s" % self.backend, "-m 0"]
        if self.backend == 'jack' and self.name is not None:
            clientname = self.name.strip().replace(" ", "_")
            options.append(f'-+jack_client=csoundengine.{clientname}')
        for opt in options:
            cs.setOption(opt)
        logger.debug("------------------------------------------------------------------")
        logger.debug("  Starting performance thread. ")
        logger.debug(f"     Options: {options}")
        logger.debug(orc)
        logger.debug("------------------------------------------------------------------")
        cs.compileOrc(orc)
        cs.start()
        pt = ctcsound.CsoundPerformanceThread(cs.csound())
        pt.play()
        self._cs = cs
        self._pt = pt
    
    def stop(self):
        if self._exited:
            return
        self._pt.stop()
        self._cs.stop()
        self._cs.cleanup()
        self._exited = True
        self._cs = None
        self._pt = None
        self._instcounter = {}
        self._instrRegistry = {}
        infopath = self._getInfoPath()
        if os.path.exists(infopath):
            os.remove(infopath)

    def _start(self):
        self._writeEngineInfo()
        self._startCsound()

    def restart(self) -> None:
        self.stop()
        self._start()
        
    def _outcallback(self, _, chan, valptr, chantypeptr):
        func = self._outcallbacks.get(chan)
        if not func:
            return
        val = ctcsound.cast(valptr, _MYFLTPTR).contents.value
        func(chan, val)

    def registerOutvalueCallback(self, chan:str, func) -> None:
        """
        Register a function `func` which will be called whenever a
        channel `chan` is changed in csound via the "outvalue" opcode

        chan: the name of a channel
        func: a function of the form `func(chan, newvalue)`
        """
        if not self._isOutCallbackSet:
            self._isOutCallbackSet = True
            self._cs.setOutputChannelCallback(self._outcallback)
        self._outcallbacks[bytes(chan, "ascii")] = func

    def getCsound(self):
        return self._cs

    def defInstr(self, instr:str, name:str=None) -> None:
        """
        Compile a csound instrument

        instr : the instrument definition, beginning with 'instr xxx'
        name  : name of the instrument, to keep track of definitions.
        """
        if not name:
            name = _getUUID()
        lines = [l for l in instr.splitlines() if l.strip()]
        instrnum = int(lines[0].split()[1])
        self._instrRegistry[name] = (instrnum, instr)
        self._cs.compileOrc(instr)
        logger.debug(f"defInstr: {name}")
        logger.debug(instr)

    def evalCode(self, code:str):
        """
        Evaluates code at instr0 (global code, only i-rate)
        Returns the value of the last evaluated line
        """
        codelines = [line.strip() for line in code.splitlines()]
        for line in codelines:
            if not line:
                continue
            if self._alreadyEvaluated(line):
                logger.debug(f"Resource already evaluated, skipping: {line}")
                return
            logger.debug(f'evalCode: evaluating line: "{line}"')
            out = self._cs.evalCode(line)
        return out

    def _alreadyEvaluated(self, code):
        unpureOpcodes = ['sfload']
        for opcode in unpureOpcodes:
            if opcode in code and code in self._globalcode:
                return True
        return False

    def sched(self, instrnum, delay=0, dur=-1, args=None) -> float:
        # type: (int, float, float, List) -> float
        """
        Schedule an instrument

        instrnum : the instrument number
        delay    : time to wait before instrument is started
        dur      : duration of the event
        args     : any other args expected by the instrument

        Returns: 
            the fractional number of the instr started. 
            This can be used to kill the event later on 
            (see unsched)
        """
        instance = self._getInstance(instrnum)
        instrfrac = self._getFractionalInstance(instrnum, instance)
        pargs = [instrfrac, delay, dur]
        if args:
            pargs.extend(args)
        self._pt.scoreEvent(0, "i", pargs)
        logger.debug(f"CsoundEngine.sched: scoreEvent(0, 'i', {pargs})  -> {instrfrac}")
        return instrfrac

    def unsched(self, instrfrac:float, delay:float=0) -> None:
        """
        mode: similar to turnoff2
        """
        logger.debug(f"CsoundEngin::unsched: {instrfrac}")
        self._pt.scoreEvent(0, "i", [_csound_reserved_instr_turnoff, delay, 0.1, instrfrac])


def _getUUID() -> str:
    return str(_uuid.uuid4())

def _isUUID(s: str) -> bool:
    if len(s) != 36:
        return False
    parts = s.split("-")
    if len(parts) != 5:
        return False
    if tuple(len(part) for part in parts) != (8, 4, 4, 4, 12):
        return False
    return True


_engines = {}   # type: Dict[str, CsoundEngine]
_managers = {}  # type: Dict[str, _InstrManager]


@_atexit.register
def _cleanup() -> None:
    _managers.clear()
    names = list(_engines.keys())
    for name in names:
        stopEngine(name)


def activeEngines():
    """
    Returns a list with the names of the active engines
    """
    return _engines.keys()


def getEngine(name="default",
              nchnls:int=None, sr:int=None, backend:str=None, 
              outdev="dac", a4:float=None,
              ) -> 'CsoundEngine':
    """
    This routine is only necessary if a csound engine needs to be started
    with specific parameters, which should not be saved for later.
    Otherwise, change the default values in config. 

    If an engine with this name has been defined, the engine will be returned,
    even if the settings differ. To change the settings, the engine must
    be stopped first.
    """
    engine = _engines.get(name)  # type: CsoundEngine
    if engine:
        return engine
    logger.info(f"***** Starting engine {name} *****")
    engine = CsoundEngine(sr=sr, backend=backend, outdev=outdev, a4=a4, nchnls=nchnls, name=name)
    _engines[name] = engine
    return engine


def stopEngine(name="default") -> None:
    engine = _engines.get(name)
    if not engine:
        raise KeyError("engine not found")
    engine.stop()
    del _engines[name]


class AbstrSynth:
    def stop(self):
        pass

    def isPlaying(self):
        pass

    def wait(self, pollinterval=None, sleepfunc=None):
        """
        Wait until this synth has stopped

        pollinterval: polling interval in seconds
        sleepfunc: the function to call when sleeping, defaults to time.sleep
        """
        if pollinterval is None:
            pollinterval = max(0.005, config['wait_poll_interval'])
        if sleepfunc is None: 
            sleepfunc = time.sleep
        with safe_sigint():
            while self.isPlaying():
                sleepfunc(pollinterval)
        

class Synth(AbstrSynth):
    """
    A user does NOT normally create a Synth. A Synth is created
    when a CsoundInstr is scheduled
    """

    def __init__(self, group:str, synthid:float, instrname:str=None) -> None:
        self.group = group
        self.synthid = synthid
        self._playing = True
        self.instrname = instrname

    def isPlaying(self) -> bool:
        return self._playing
    
    def getManager(self) -> '_InstrManager':
        return getManager(self.group)

    def stop(self, delay=0) -> None:
        self.getManager().unsched(self.synthid, delay=delay)
        self._playing = False


class SynthGroup(AbstrSynth):
    """
    A SynthGroup is used to control multiple (similar) synths created
    to work together (in additive synthesis, for example)
    """

    def __init__(self, synths) -> None:
        # type: (List[AbstrSynth]) -> None
        self.synths = synths

    def stop(self) -> None:
        for s in self.synths:
            s.stop()

    def isPlaying(self) -> bool:
        return any(s.isPlaying() for s in self.synths)


class CsoundInstr:
    __slots__ = ['body', 'name', 'initcode', 'group', 'meta', '_numpargs', '_recproc']

    def __init__(self, name:str, body: str, initcode: str = None, group="default", meta=None,
                 ) -> None:
        """
        *** A CsoundInstr is created via makeInstr, DON'T CREATE IT DIRECTLY ***

        To schedule a Synth using this instrument, call .play

        name    : the name of the instrument, if any. Use None to assign a UUID
        body    : the body of the instr (the text BETWEEN 'instr' end 'endin')
        initcode: code to be initialized at the instr0 level (tables, reading files, etc.)
        group   : the name of the group this instrument belongs to
        """
        errmsg = _checkInstr(body)
        if errmsg:
            raise CsoundError(errmsg)
        self.group = group
        self.name = name if name is not None else _getUUID()
        self.body = body
        self.initcode = initcode if initcode else None
        self.meta = meta if meta is not None else {}
        self._numpargs = None
        self._recproc = None

    def __repr__(self):
        header = f"CsoundInstr({self.name}, group={self.group})"
        pargs = self.getPargs()
        if pargs:
            args = ", ".join(self.getPargs())
        else:
            args = "--"
        sections = [
            header,
            f"> args: {args}",
        ]
        if self.initcode:
            sections.append("> init")
            sections.append(str(self.initcode))
        sections.append("> body")
        sections.append(self.body)
        return "\n".join(sections)

    def _getManager(self):
        return getManager(self.group)

    def getPargs(self):
        """
        Return the name of the pargs in this Instr.
        Args start at p4, since p1, p2 and p3 are always necessary
        """
        allpargs = csound.pargNames(self.body)
        pargs = []
        if not allpargs:
            return pargs
        minidx = min(allpargs.keys())
        minidx = min(4, minidx)
        maxidx = max(allpargs.keys())
        for idx in range(minidx, maxidx+1):
            pargs.append(allpargs.get(idx))
        return pargs

    def play(self, dur=-1, args:List=None, priority=1, delay=0.0, whenfinished=None):
        # type: (float, List, int, float) -> Synth
        """
        Schedules a Synth with this instrument.

        dur: the duration of the synth, or -1 to play until stopped
        args: args to be passed to the synth (p values, beginning with p4)
        priority: a number indicating order of execution. This is only important
            when depending on other synths
        delay: how long to wait to start the synth (this is always relative time)
        whenfinished: this function (if given) will be called when the synth is
            deallocated
        """
        self._checkArgs(args)
        manager = self._getManager()
        # args = [float(arg) for arg in args]
        return manager.sched(self.name, priority=priority, delay=delay, dur=dur,
                             args=args, whenfinished=whenfinished)

    def asOrc(self, instrid, sr:int, ksmps:int, nchnls:int=None) -> str:
        nchnls = nchnls if nchnls is not None else self._numchannels() 
        if self.initcode is None:
            initstr = ""
        else:
            initstr = self.initcode
        orc = """
        sr = {sr}
        ksmps = {ksmps}
        nchnls = {nchnls}
        0dbfs = 1

        {initstr}

        instr {instrid}
        
        {body}
        
        endin

        """.format(sr=sr, ksmps=ksmps, instrid=instrid, body=self.body, nchnls=nchnls, initstr=initstr)
        return orc

    def _numchannels(self):
        return 2

    def _numargs(self) -> int:
        if self._numpargs is None:
            self._numpargs = csound.numPargs(self.body)
        return self._numpargs

    def _checkArgs(self, args, fail=None) -> bool:
        lenargs = 0 if args is None else len(args)
        numargs = self._numargs()
        ok = numargs == lenargs
        fail = fail if fail is not None else config['fail_if_unmatched_pargs']
        if not ok:
            msg = f"expected {numargs} args, got {lenargs}"
            if fail:
                raise ValueError(msg)
            else:
                print(msg)
                logger.error(msg)
        return ok

    def rec(self, dur, outfile:str=None, args:List[float]=None, sr=44100, ksmps=64, 
            samplefmt='float', nchnls:int=None, block=True) -> str:
        """
        dur:       the duration of the recording
        outfile:   if given, the path to the generated soundfile. If not given, a temporary file will be
                   generated.
        args:      the seq. of pargs passed to the instrument (if any), beginning with p4
        sr:        the sample rate
        samplefmt: one of 16, 24, 32, or 'float'
        nchnls:    the number of channels of the generated soundfile. It defaults to 2
        block:     if True, the function blocks until done, otherwise rendering is asynchronous 
        """
        event = [0, dur]
        if args:
            event.extend(args)
        return self.recEvents(events=[event], outfile=outfile, sr=sr, ksmps=ksmps, 
                              samplefmt=samplefmt, nchnls=nchnls, block=block)

    def recEvents(self, events, outfile:str=None, sr=44100, ksmps=64, samplefmt='float', 
                  nchnls:int=None, block=True) -> str:
        """
        events:    a seq. of events, where each event is the list of pargs passed to the instrument,
                   as [delay, dur, p4, p5, ...]
        outfile:   if given, the path to the generated soundfile. If not given, a temporary file will be
                   generated.
        sr:        the sample rate
        samplefmt: one of 16, 24, 32, or 'float'
        nchnls:    the number of channels of the generated soundfile. It defaults to 2
        block:     if True, the function blocks until done, otherwise rendering is asynchronous 
        """
        nchnls = nchnls if nchnls is not None else self._numchannels()
        initstr = self.initcode if self.initcode is not None else ""
        outfile, popen = csound.recInstr(body=self.body, init=self.initcode, outfile=outfile,
                                         events=events, sr=sr, ksmps=ksmps, samplefmt=samplefmt, 
                                         nchnls=nchnls)
        if block:
            popen.wait()
        return outfile

    def stop(self):
        """
        Will stop all synths created with this instrument
        """
        self._getManager().unschedByName(self.name)


def _checkInstr(instr: str) -> str:
    """
    Returns an error message if the instrument is not well defined
    """
    lines = [l for l in (l.strip() for l in instr.splitlines()) if l]
    errmsg = ""
    if "instr" in lines[0] or "endin" in lines[-1]:
        errmsg = ("instr should be the body of the instrument,"
                  " without 'instr' and 'endin")
    return errmsg


class _InstrManager:
    """
    An InstrManager controls a series of instruments. It can have an 
    exclusive CsoundEngine associated, but this is an implementation detail.
    """
    
    def __init__(self, name="default") -> None:
        self.name: str = name
        self.instrDefs = {}  # type: Dict[str, 'CsoundInstr']

        self._bucketsize: int = 1000
        self._numbuckets: int = 10
        self._buckets = [{} for _ in range(self._numbuckets)]  # type: List[Dict[str, int]]
        self._synthdefs = {}                                   # type: Dict[str, Dict[int, _SynthDef]]
        self._synths = {}                                      # type: Dict[float, Synth]
        self._isDeallocCallbackSet = False
        self._whenfinished = {}
        
    def _deallocCallback(self, _, synthid):
        synth = self._synths.pop(synthid, None)
        if synth is None:
            logger.debug(f"synth {synthid} not found in Manager({self.name})._synths!")
            return
        synth._playing = False
        logger.debug(f"instr {synthid} deallocated!")
        callback = self._whenfinished.pop(synthid, None)
        if callback:
            callback(synthid)

    def getEngine(self) -> CsoundEngine:
        engine = getEngine(self.name)
        if not self._isDeallocCallbackSet:
            engine.registerOutvalueCallback("__dealloc__", self._deallocCallback)
        return engine

    def getInstrnum(self, instrname:str, priority=1) -> int:
        assert 1 <= priority < self._numbuckets - 1
        bucket = self._buckets[priority]
        instrnum = bucket.get(instrname)
        if instrnum is not None:
            return instrnum
        idx = len(bucket) + 1
        instrnum = self._bucketsize*priority + idx
        bucket[instrname] = instrnum 
        return instrnum

    def defInstr(self, name, body, initcode="", meta:dict=None):
        # type: (str, str, Opt[str]) -> CsoundInstr
        """
        name (str)     : a name to identify this instr, or None, in which case a UUID is created
        body (str)     : the body of the instrument
        initcode (str) : initialization code for the instr (ftgens, global vars, etc.)
        meta (dict)    : A dictionary to store metadata
        """
        if name is None:
            instr = self.findInstrByBody(body=body, initcode=initcode)
            name = instr.name if instr else _getUUID()
        else:
            instr = self.instrDefs.get(name)
        if instr:
            if body == instr.body and initcode == instr.initcode:
                logger.debug(f"The instruments are identical, reusing old instance")
                return instr
            logger.info("Instruments differ, old definition will be overwritten")
            logger.debug("new body: ")
            logger.debug(body)
            logger.debug("old body:")
            logger.debug(instr.body)
            logger.debug("new init: ")
            logger.debug(initcode)
            logger.debug("old init: ")
            logger.debug(instr.initcode)
            self._resetSynthdefs(name)
        instr = CsoundInstr(name=name, body=body, initcode=initcode, group=self.name, meta=meta)
        self.registerInstr(instr)
        return instr

    def findInstrByBody(self, body, initcode:str=None, onlyunnamed=False) -> 'Opt[CsoundInstr]':
        for name, instr in self.instrDefs.items():
            if onlyunnamed and not _isUUID(name):
                continue
            if body == instr.body and (not initcode or initcode == instr.initcode):
                return instr

    def registerInstr(self, instr:CsoundInstr, name:str=None) -> None:
        name = name or instr.name
        self.instrDefs[name] = instr
        if instr.initcode:
            self._evalInit(instr)
            
    def _evalInit(self, instr:CsoundInstr) -> None:
        logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> evaluating init code: ")
        logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " + instr.initcode)
        self.getEngine().evalCode(instr.initcode)

    def _resetSynthdefs(self, name):
        self._synthdefs[name] = {}

    def _registerSynthdef(self, name: str, priority: int, synthdef: _SynthDef) -> None:
        registry = self._synthdefs.get(name)
        if registry:
            registry[priority] = synthdef
        else:
            registry = {priority: synthdef}
            self._synthdefs[name] = registry

    def _makeSynthdef(self, name:str, priority:int) -> _SynthDef:
        """
        A SynthDef is a version of an Instrument with a given priority
        While making a SynthDef we send it already to the Engine
        """
        qname = _qualifiedName(name, priority)
        instrdef = self.instrDefs.get(name)
        instrnum = self.getInstrnum(name, priority)
        instrtxt = _instrWrapBody(instrdef.body, instrnum)
        self.getEngine().defInstr(instrtxt, name)
        synthdef = _SynthDef(qname, instrnum)
        self._registerSynthdef(name, priority, synthdef)
        return synthdef

    def getInstr(self, name:str) -> 'CsoundInstr':
        return self.instrDefs.get(name)

    def _getSynthdef(self, name:str, priority:int) -> 'Opt[_SynthDef]':
        registry = self._synthdefs.get(name)
        if not registry:
            return None
        return registry.get(priority)

    def prepareSched(self, instrname:str, priority:int=1) -> _SynthDef:
        synthdef = self._getSynthdef(instrname, priority)
        if synthdef is None:
            synthdef = self._makeSynthdef(instrname, priority)
        return synthdef

    def sched(self, instrname:str, priority:int=1, delay:float=0, dur:float=-1, args=[],
              whenfinished=None) -> Synth:
        synthdef = self.prepareSched(instrname, priority)
        synthid = self.getEngine().sched(synthdef.instrnum, delay=delay, dur=dur, args=args)
        if whenfinished is not None:
            self._whenfinished[synthid] = whenfinished
        synth = Synth(self.name, synthid=synthid, instrname=instrname)
        self._synths[synthid] = synth
        return synth

    def unsched(self, *synthids:float, delay=0) -> None:
        logger.debug(f"Manager: asking engine to unsched {synthids}")
        engine = self.getEngine()
        for synthid in synthids:
            engine.unsched(synthid, delay)
            self._synths.pop(synthid)
    
    def unschedByName(self, instrname:str):
        """
        Unschedule all playing synths created from given instr (as identified by the name)
        """
        synths = self.findSynthsByName(instrname)
        for synth in synths:
            self.unsched(synth.synthid)

    def unschedAll(self) -> None:
        synthids = [synth.synthid for synth in self._synths.values()]
        for synthid in synthids:
            self.unsched(synthid, delay=0)

    def findSynthsByName(self, instrname):
        """
        Return a list of active Synths created from the given instr
        """
        out = []
        for synthid, synth in self._synths.items():
            if synth.instrname == instrname:
                out.append(synth)
        return out


def _qualifiedName(name:str, priority:int) -> str:
    return f"{name}:{priority}"


def _instrWrapBody(body:str, instrnum:int, notify=True, dedent=False) -> str:
    if notify:
        s = """
        instr {instrnum}
        
        k__release release
        k__notified init 0
        
        if (k__release == 1) && (k__notified == 0) then
            k__notified = 1
            event "i", "_notifyDealloc", 0, -1, p1
        endif

        {body}

        endin
        """
    else:
        s = """
        instr {instrnum}

        {body}

        endin
        """
    s = s.format(instrnum=instrnum, body=body)
    if dedent:
        s = _textwrap.dedent(s)
    return s


def getManager(name="default") -> _InstrManager:
    """
    Get a specific Manager. A Manager controls a series of
    instruments and has its own csound engine
    """
    manager = _managers.get(name)
    if not manager:
        manager = _InstrManager(name)
        _managers[name] = manager
    return manager 


def unschedAll(group='default') -> None:
    man = getManager(group)
    man.unschedAll()


def makeInstr(body:str, initcode:str=None, name:str=None, group='default') -> CsoundInstr:
    """
    Creates a new CsoundInstr as part of group `group`

    To schedule a synth using this instrument use the .play method on the returned CsoundInstr

    See InstrSine for an example

    body    : the body of the instrument (the part between 'instr ...' and 'endin')
    initcode: the init code of the instrument (files, tables, etc.)
    name    : the name of the instrument, or None to assign a unique id
    group   : the group to handle the instrument
    """
    instr = getManager(group).defInstr(name=name, body=body, initcode=initcode)
    logger.debug(f"makeInstr: name: {instr.name}, body={instr.body}")
    return instr


def getInstr(name:str, group='default') -> 'Opt[CsoundInstr]':
    """
    Returns a CsoundInstr if an instrument was already defined, or None
    """
    man = getManager(name=group)
    instr = man.getInstr(name)
    return instr
    # return makeInstr(body=instrdef.body, initcode=instrdef.initcode, name=instrdef.name, group=group)


def availableInstrs(group='default'):
    return getManager(name=group).instrDefs.keys()

# -----------------------------------------------------------------------------


def InstrSineGliss(name='builtin.sinegliss', group='default'):
    body = """
        iDur, iAmp, iFreqStart, iFreqEnd passign 3
        imidi0 = ftom:i(iFreqStart)
        imidi1 = ftom:i(iFreqEnd)
        kmidi linseg imidi0, iDur, imidi1
        kfreq = mtof:k(kmidi)
        aenv linsegr 0, 0.01, 1, 0.05, 0
        a0 oscili iAmp, kfreq
        a0 *= aenv
        outs a0, a0
    """
    return makeInstr(body=body, name=name, group=group)


def InstrSine(name='builtin.sine', group='default'):
    body = """
        iDur, iChan, iAmp, iFreq passign 3
        kenv linsegr 0, 0.04, 1, 0.08, 0
        a0 oscil iAmp, iFreq
        a0 *= kenv
        outch iChan, a0
    """
    return makeInstr(body=body, name=name, group=group)


def InstrSines(numsines, group='default', sineinterp=True):
    """
    i = InstrSines(4)
    i.play(chan, gain, freq1, amp1, freq2, amp2, ...)

    def sinesplay(chan, gain, freqs):
        amps = [1/len(freqs)] * len(freqs)
        return i.play(chan, gain, *zip(freqs, amps))

    sinesplay(chan=1, gain=1, [440, 450, 460])
    """
    body = csound.genBodyStaticSines(numsines, sineinterp)
    name = f"csoundengine.sines{numsines}"
    if sineinterp:
        name += ".interp"
    return makeInstr(body=body, name=name, group=group)


_engineTempDir = CsoundEngine.getTempDir()
if not os.path.exists(_engineTempDir):
    os.makedirs(_engineTempDir)

del Dict, Opt, List