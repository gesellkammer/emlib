"""
This module handles playing of events

Each Note, Chord, Line, etc, can express its playback in terms of _CsoundLine
events.

A _CsoundLine event is a score line with the protocol

p2      3     4      5      6       7       8        9      ...
idelay, idur, igain, ichan, ifade0, ifade1, inumbps, ibplen

The rest of the pfields being a flat list of values representing
the breakpoints

A breakpoint is a tuple of values of the form (offset, pitch [, amp, ...])
The size if each breakpoint and the number of breakpoints are given
by inumbps, ibplen

An instrument to handle playback should be defined with `defPreset` which handles
breakpoints and only needs the audio generating part of the csound code

Whenever a note actually is played with a given preset, this preset is
actually sent to the csound engine and instantiated/evaluated.
"""


import os
import logging


from dataclasses import dataclass
from functools import lru_cache

from emlib.snd import csoundengine
from emlib import typehints as t

from .config import config

logger = logging.getLogger(f"emlib.mus")


@dataclass
class _InstrDef:
    body: str
    init: str


_csoundPrelude = \
"""
opcode passignarr, i[],ii
    istart, iend xin
    idx = 0
    icnt = iend - istart + 1
    iOut[] init icnt
    while idx < icnt do
        iOut[idx] = pindex(idx+istart)
        idx += 1
    od
    xout iOut
endop
"""


class _PresetManager:

    def __init__(self):
        self.instrdefs = {}
        self.makeBuiltin()

    def makeBuiltin(self) -> None:
        """
        Defines all builtin presets
        """
        gen = self.defPresetSimple
        sf = self.defPresetSoundfont
        gen('sine', "a0 oscili a(kamp), kfreq")
        gen('sine.cos', "a0 oscili a(kamp), kfreq", interpol='cos')
        gen('tri', "a0 vco2 kamp, kfreq, 12")
        gen('tri.cos', "a0 vco2 kamp, kfreq, 12", interpol='cos')
        gen('saw', "a0 vco2 kamp, kfreq, 0")
        gen('saw.cos', "a0 vco2 kamp, kfreq, 0", interpol='cos')
        gen('trilpf', "a0 K35_lpf vco2:a(kamp, kfreq, 12), kfreq * 5, 5")

        sf('piano', preset=148)
        sf('piano.cos', preset=148, interpol='cos')
        sf('clarinet', preset=61)
        sf('clarinet.cos', preset=61, interpol='cos')
        sf('oboe', preset=58)
        sf('flute', preset=42)
        sf('violin', preset=47)
        sf('reedorgan', preset=52)
        sf('reedorgan', preset=52, interpol='cos')

    def defPresetSimple(self, name, audiogen, init=None, interpol='linear') -> None:
        """
        Define a new simple instrument preset

        A simple instrument is a mono instrument with varying pitch (midinote)
        and amplitude. The new instrument is created by defining the audio
        generating part.

        Given the variables 'kpitch' and 'kamp', audiogen should generate 'a0'

        Example

        audiogen = 'a0 oscili a(kamp), mtof:k(kpitch)'

        name     : the name of the preset
        audiogen : audio generating csound code
        init     : global code needed by the audiogen part (usually a table definition)
        interpol : interpolation used by kpitch and kamp. One of 'linear', 'cos'
        """
        instrdef = _makePresetSimple(audiogen, init=init, interpol=interpol)
        self.instrdefs[name] = instrdef

    def defPresetSoundfont(self, name, sf2path:str = None, preset=0, interpol='linear') -> None:
        """
        Define a new soundfont instrument preset

        name    : the name of the preset
        sf2path : the path to the soundfont, or None to use the default fluidsynth
                  soundfont
        preset  : the preset to use
                  NB: to list all presets in a soundfont in linux, use
                  $ echo "inst 1" | fluidsynth $1 2>/dev/null | egrep '[0-9]{3}-[0-9]{3} '
        interpol: use linear or cos interpolation for pitch/amp
        """
        instrdef = _makePresetSoundfont(sf2path=sf2path, preset=preset, interpol=interpol)
        self.instrdefs[name] = instrdef

    def defPreset(self, name:str, body:str, init:str = None) -> None:
        instrdef = _InstrDef(body=body, init=init)
        self.instrdefs[name] = instrdef

    def getInstrDef(self, name:str) -> _InstrDef:
        return self.instrdefs.get(name)

    def definedPresets(self) -> t.Set[str]:
        return set(self.instrdefs.keys())


def _makePresetSimple(audiogen, init: str = None, interpol='linear') -> '_InstrDef':
    """
    Create an _InstrDef based on a template

    audiogen:
        the audio generating csound code. Takes `kpitch` and `kamp` and outputs `a0`
    init:
        any csound init code
    interpol:
        'linear' / 'cos'
        Sets the interpolation of kpitch / kamp

    Returns: an _InstrDef containing the resulting body and init, to be used as a preset

    ftgen 0, 0, 0, 

    """
    template = r"""
    ;  3  4       5      6      7       8        9
    idur, igain, ichan, ifade0, ifade1, inumbps, ibplen passign 3
    idatalen = inumbps * ibplen
    ipcount = idatalen + 9
    iArgs[] init ipcount
    iArgs passign 10, 9+ipcount
    ilastidx = idatalen - 1
    iTimes[]   init inumbps
    iPitches[] init inumbps
    iAmps[]    init inumbps
    iTimes     slicearray iArgs, 0, ilastidx, ibplen
    iPitches   slicearray iArgs, 1, ilastidx, ibplen
    iAmps      slicearray iArgs, 2, ilastidx, ibplen
    kt timeinsts

    {pitchampcalc}
    ; kpitch, kamp bpf kt, iTimes, iPitches, iAmps

    kfreq mtof kpitch
    
    {audiogen}
    ; a0 oscili a(kamp), mtof:k(kpitch)

    aenv linsegr 0, ifade0, igain, ifade1, 0
    ; aenv linseg 0, ifade0, igain, idur - (ifade0+ifade1), igain, ifade1, 0
    a0 *= aenv
    outch ichan, a0
    """
    if interpol == 'linear':
        pitchampcalc = 'kpitch, kamp bpf kt, iTimes, iPitches, iAmps'
    elif interpol == 'cos':
        pitchampcalc = """
            kpitch bpfcos kt, iTimes, iPitches
            kamp   bpfcos kt, iTimes, iAmps
            """
    else:
        raise ValueError(f"interpol should be one of 'linear', 'cos', got: {interpol}")
    body = template.format(audiogen=audiogen, pitchampcalc=pitchampcalc)
    return _InstrDef(body=body, init=init)


def defPreset(name: str, audiogen, init: str = None, interpol='linear'):
    """
    Define an instrument preset by specifying the audio generating csound code

    name:
        the name of the preset
    audiogen:
        a str containing csound code to generate audio (see below)
    init:
        if necessary, any global/init csound code needed by the instrument
    interpol:
        kind of pitch interpolation (linear / cos)

    audiogen has access to following variables:
    * kpitch: the pitch to synthesize, as midi note (possibly fractional)
    * kamp: the amplitude (0-1) to synthesize. Should be converted to audio to avoid clicks
    * audiogen should put the generated audio in a variable named 'a0'

    Example: a simple sine tone

    defPreset("mysine", audiogen="a0 oscili a(kamp), mtof:k(kpitch))
    """
    return _presetManager.defPresetSimple(name=name, audiogen=audiogen, init=init, interpol=interpol)


def _makePresetSoundfont(sf2path: str = None, preset=0, interpol='linear') -> '_InstrDef':
    """
    Define an instrument preset usable by .play, based on a soundfont (sf2)

    sf2path:
        path to a sf2 soundfont. If None, the default fluidsynth soundfont is used
    preset:
        preset to be loaded. To find the preset number, use
        echo "inst 1" | fluidsynth "path/to/sf2" | egrep '[0-9]{3}-[0-9]{3} '

    banks are not supported

    NB: at the moment all instruments generate a mono signal, even if they
        are played as stereo (the right chan is discarded)
    """
    if sf2path is not None:
        sf2path = os.path.abspath(sf2path)
        tabname = _soundfontToTabname(sf2path)
    else:
        sf2path = csoundengine.fluidsf2Path()
        tabname = 'gi_fluidsd'
    if not os.path.exists(sf2path):
        raise FileNotFoundError(f"Soundfont file {sf2path} not found")
    init = f'{tabname}  sfload "{sf2path}"'
    audiogen = f"""
    inote0 = p11
    ivel = p12*127
    iscale = 1/16384    
    a0, a1  sfinstr ivel, inote0, kamp*iscale, mtof:k(kpitch), {preset}, {tabname}, 1
    """
    return _makePresetSimple(audiogen=audiogen, init=init, interpol=interpol)


def _path2name(path):
    name = os.path.splitext(os.path.split(path)[1])[0].replace("-", "_")
    return name


@lru_cache(maxsize=100)
def getInstrPreset(instr: str = None, stereo=False):
    """
    instr: if None, use the default instrument as defined in config['play.instr']
    """
    if instr is None:
        instr = config['play.instr']
    instrdef = _presetManager.getInstrDef(instr)
    if instrdef is None:
        raise KeyError(f"Unknown instrument {instr}")
    group = config['play.group']
    body = instrdef.body
    init = instrdef.init
    name = f'{group}.preset.{instr}'
    if stereo:
        name += '.stereo'
        body += "\noutch ichan+1, a0\n"
    logger.debug(f"creating csound instr. name={name}, group={group}")
    startPlayEngine()
    csdinstr = csoundengine.makeInstr(name=name, body=body, init=init, group=group, check=False)
    logger.debug(f"Created {csdinstr}")
    return csdinstr


def makeFilteredSawPreset(name, cutoff, resonance=0.2, interpol='linear'):
    audiogen = f"""
    a0 vco2 a(kamp), mtof:k(kpitch), 10
    a0 moogladder a0, {cutoff}, {resonance}
    """
    if name is None:
        name = f"sawfilt{int(cutoff)}"
    return defPreset(name=name, audiogen=audiogen, interpol=interpol)


def _soundfontToTabname(sfpath: str) -> str:
    path = os.path.abspath(sfpath)
    return f"gi_sf2func_{hash(path)}"


def availableInstrPresets() -> t.Set[str]:
    """
    Returns a set of instr presets already defined
    """
    return _presetManager.definedPresets()


def startPlayEngine(nchnls=None) -> t.Opt[csoundengine.CsoundEngine]:
    """
    Start the play engine with a given configuration, if necessary.
    """
    engineName = config['play.group']
    if engineName in csoundengine.activeEngines():
        return
    nchnls = nchnls or config['play.numChannels']
    logger.info(f"Starting engine {engineName} (nchnls={nchnls})")
    return csoundengine.CsoundEngine(name=engineName, nchnls=nchnls, globalcode=_csoundPrelude)
    
def stopSynths(stop_engine=False, cancel_future=True, allow_fadeout=None):
    """
    Stops all synths (notes, chords, etc) being played

    If stopengine is True, the play engine itself is stopped
    """
    manager = getPlayManager()
    allow_fadeout = allow_fadeout if allow_fadeout is not None else config['play.unschedFadeout']
    manager.unschedAll(cancel_future=cancel_future, allow_fadeout=allow_fadeout)
    if stop_engine:
        stopPlayEngine()
        
def getPlayManager():
    group = config['play.group']
    return csoundengine.getManager(group)


def restart():
    group = config['play.group']
    manager = csoundengine.getManager(group)
    manager.unschedAll()
    manager.restart()


def isEngineActive():
    group = config['play.group']
    return csoundengine.getEngine(group) is not None


def getPlayEngine() -> t.Opt[csoundengine.CsoundEngine]:
    engine = csoundengine.getEngine(name=config['play.group'])
    if not engine:
        logger.debug("engine not started")
        return
    return engine


def stopPlayEngine():
    group = config['play.group']
    engine = csoundengine.getEngine(group)
    if not engine:
        logger.error("play engine not started")
        return
    engine.stop()


def getPresetManager():
    return _presetManager


_presetManager = _PresetManager()
