# -*- coding: utf-8 -*-
import functools as _functools
from collections import namedtuple
from fractions import Fraction
from math import sqrt
from functools import lru_cache
import subprocess
import logging
import os
import tempfile

import music21 as m21

from bpf4 import bpf
from emlib import lib
from emlib import conftools
from emlib.music import m21tools
from emlib.pitch import amp2db, db2amp, m2n, n2m, n2f, m2f, f2m, r2i
from emlib.iterlib import pairwise, window, flatten
from emlib.snd import csoundengine
import emlib.typehints as t


logger = logging.getLogger(f"emlib.mus")


_defaultconfig = {
    'showcents': True,
    'repr.showfreq': True,
    'chord.arpeggio': 'auto',
    'chord.adjustGain': True,
    'notation.semitone_divisions': 2,
    'm21.displayhook.install': True,
    'm21.displayhook.format': 'lily.png',
    'show.split': True,
    'show.centSep': ',',
    'show.scalefactor': 1.0, 
    'show.format': 'lily.png',
    'show.external': False,
    'use_musicxml2ly': True,
    'app.png': '/usr/bin/feh',
    'displayhook.install': True,
    'play.method': 'csound',
    'play.dur': 2.0,
    'play.gain': 0.5,
    'play.group': 'emlib.event',
    'play.instr': 'sine',
    'play.fade': 0.02,
    'play.numchannels': 2,
    'rec.block': False,
    'rec.gain': 1.0
}

_validator = {
    'notation.semitone_divisions::choices': [1, 2, 4],
    'm21.displayhook.format::choices': ['xml.png', 'lily.png'],
    'displayhook.format::choices': ['xml.png', 'lily.png'],
    'show.format::choices': 
        ['xml.png', 'xml.pdf', 'lily.png', 'lily.pdf', 'repr'],
    'play.method::choices': ['csound', 'midi'],
    'chord.arpeggio::choices':
        ['auto', 'always', 'never', True, False],
    'play.preset::choices':
        ['sine', 'piano', 'tri', 'clarinet'],
    'play.gain::range': (0, 1),
    'play.numchannels::type': int
}

def _checkConfig(cfg, key, value):
    if key == 'notation.semitone_divisions' and value == 4:
        showformat = cfg.get('show.format')
        if showformat and showformat.startswith('lily'):
            return ("\nlilypond backend (show.format) does not support 1/8 tones yet.\n"
                    "Either set config['notation.semitone_divisions'] to 2 or\n"
                    "set config['show.format'] to 'xml.png'")

config = conftools.ConfigDict(f'emlib:mus', _defaultconfig, _validator, validHook=_checkConfig)


def asNote(n, amp=None):
    # type: (t.U[Note, int, float, str, t.Tup[t.U[str, float], float]], float) -> Note
    """
    n: str    -> notename
       number -> midinote
    amp: 0-1

    you can also create a note by doing asNote((pitch, amp))
    """
    if isinstance(n, Note):
        out = n
    elif isinstance(n, (int, float)):
        out = Note(n, amp=amp)
    elif isinstance(n, str):
        out = Note(n2m(n), amp=amp)
    elif isinstance(n, tuple) and len(n) == 2 and amp is None:
        out = asNote(*n)
    else:
        raise ValueError(f"cannot express this as a Note: {n} ({type(n)})")
    return out


def asChord(obj, amp=None):
    # type: (t.U[Chord, t.List, t.Tup]) -> Chord
    if isinstance(obj, Chord):
        return obj
    elif isinstance(obj, (list, tuple, str)):
        return Chord(obj)
    elif hasattr(obj, "asChord"):
        out = obj.asChord()
        if not isinstance(out, Chord):
            raise TypeError(f"Called asChord on {obj} expecting to get a Chord, but got {type(out)}")
        return out
    else:
        raise ValueError(f"cannot express this as a Chord: {obj}")


_AmpNote = namedtuple("Note", "note midi freq db step")


def splitByAmp(midinotes, amps, numgroups=8, maxnotes_per_group=8):
    # type: (t.Iter[float], t.Iter[float], int, int) -> t.List[Chord]
    """
    split the notes by amp into groups (similar to a histogram based on amplitude)

    midinotes         : a seq of midinotes
    amps              : a seq of amplitudes in dB (same length as midinotes)
    numgroups         : the number of groups to divide the notes into
    maxnotes_per_group: the maximum of included notes per group, picked by loudness

    Returns: a list of chords with length=numgroups
    """
    curve = bpf.expon(
        -120, 0,
        -60, 0.0,
        -40, 0.1,
        -30, 0.4,
        -18, 0.9,
        -6, 1,
        0, 1,
        exp=0.333
    )  # type: bpf.BpfInterface
    step = (curve * numgroups).floor()
    # step2 = bpf.asbpf(db2amp) | bpf.linear(0, 0, 1, 1) * numgroups
    notes = []
    for note, amp in zip(midinotes, amps):
        db = amp2db(amp)
        notes.append(_AmpNote(m2n(note), note, m2f(note), db, int(step(db))))
    chords = [[] for _ in range(numgroups)]
    notes2 = sorted(notes, key=lambda n: n.db, reverse=True)
    for note in notes2:
        chord = chords[note.step]
        if len(chord) <= maxnotes_per_group:
            chord.append(note)
    for chord in chords:
        chord.sort(key=lambda n: n.db, reverse=True)       
    return [Chord(ch) for ch in chords]


def _notenameExtractCents(note):
    # type: (str) -> int
    assert isinstance(note, str)
    if "+" in note:
        n, c = note.split("+")
        return int(c) if c else 50
    elif "-" in note:
        n, c = note.split("-")
        return int(c) if c else -50
    else:
        return 0


class Pitch:
    def __init__(self, midinote, cents=0):
        self.midinote = midinote
        self.cents = cents 

    def __repr__(self):
        return f"Pitch(midinote={self.midinote}, cents={self.cents}"


def _jupyterMakeImage(path):
    from IPython.core.display import Image
    scalefactor = config.get('show.scalefactor', 1.0)
    if scalefactor != 1.0:
        imgwidth, imgheight = _imgSize(path)
        width = imgwidth * scalefactor
    else:
        width = None
    return Image(filename=path, embed=True, width=width)  # ._repr_png_()
    

def _jupyterShowImage(path):
    img = _jupyterMakeImage(path)
    return _jupyterDisplay(img)
    

def _pngShow(image, external=False):
    if external or not _inside_jupyter:
        _pngOpenExternal(image)
    else:
        _jupyterShowImage(image)
        

class _Base:
    _showableInitialized = False

    def __init__(self):
        self._pngimage = None

    def quantize(self, step=1.0):
        pass

    def transpose(self, step):
        pass

    def freqratio(self, ratio):
        """
        Transpose this by a given freq. ratio. A ratio of 2 equals to transposing an octave
        higher.
        """
        return self.transpose(r2i(ratio))

    def show(self, external=None, **options):
        """
        Show this as an image.

        external: 
            force opening the image in an external image viewer, even when
            inside a jupyter notebook. Otherwise, show will display the image
            inline
            
        options: any argument passed to .asmusic21


        NB: to use the music21 show capabilities, use note.asmusic21().show(...) or
            m21show(note.asmusic21())
        """
        external = external if external is not None else config['show.external']
        png = self.makeImage(**options)
        _pngShow(png, external=external)
        return self
        
    def _changed(self):
        # type: () -> None
        pass
        
    def makeImage(self, **options) -> str:
        """
        Creates an image representation, returns the path to the image

        options: any argument passed to .asmusic21

        """
        return makeImage(self, **options)

    def ipythonImage(self):
        from IPython.core.display import Image
        return Image(self.makeImage(), embed=True)
        
    @property
    def m21(self):
        # type: () -> m21.note.GeneralNote
        return self.asmusic21(pure=True)

    def asmusic21(self, **options):
        # type: (...) -> m21.stream.Stream
        pass

    @classmethod
    def _setJupyterHook(cls):
        if cls._showableInitialized:
            return
        from IPython.core.display import Image

        def reprpng(obj):
            imgpath = obj.makeImage()
            scalefactor = config.get('show.scalefactor', 1.0)
            if scalefactor != 1.0:
                imgwidth, imgheight = _imgSize(imgpath)
                width = imgwidth * scalefactor
            else:
                width = None
            return Image(filename=imgpath, embed=True, width=width)._repr_png_()
            
        _setJupyterHookForClass(cls, reprpng, fmt='image/png')

    def _playDur(self):
        return config['play.dur']

    def _play(self, delay:float, dur:float, gain:float, chan:int,
              csdinstr: 'csoundengine.CsoundInstr', fade:float):
        events = self._csoundEvents(delay=delay, dur=dur, chan=chan, gain=gain, fade=fade)
        if not events:
            raise ValueError("No events for obj {self}")
        if len(events) == 1:
            event = events[0]
            return csdinstr.play(dur=event[1], delay=event[0], args=event[2:])
        else:
            synths = [csdinstr.play(event[1], delay=event[0], args=event[2:]) for event in events]
            return _synthgroup(synths)

    def _csoundEvents(self, delay: float, dur: float, chan: int, gain: float, fade=0.0):
        raise NotImplementedError("This method should be overloaded")
    
    def _rec(self, delay:float, dur:float, gain:float, chan:int, outfile:str, csdinstr, sr:int, 
             block:bool, fade:float) -> str:
        """
        Called by .rec

        csdinstr: a CsoundInstrument
        block: if true, wait until offline recording is done
        delay: extra delay 
        dur: extra duration
        """
        events = self._csoundEvents(delay=delay, dur=dur, chan=chan, gain=gain, fade=fade)
        return csdinstr.recEvents(outfile=outfile, events=events, sr=sr, nchnls=chan,
                                  block=block)
        
    def play(self, dur=None, gain=None, delay=0, instr=None, chan=None, fade=None):
        # type: (float, float, float, str, int, float) -> csoundengine.AbstrSynth
        """
        Plays this object. Play is always asynchronous

        dur:   the duration of the event
        gain:  modifies the own amplitude for playback/recording (0-1)
        delay: delay in secons to start playback. 0=start immediately
        instr: which instrument to use (see defInstrPreset, availableInstrPresets)
        chan:  the channel to output to (an int starting with 1).
               If given, output is mono and the selected channel is used. Otherwise,
               output is stereo to channels 1, 2
        """
        gain = gain if gain is not None else config['play.gain']
        dur = dur if dur is not None else self._playDur()
        chan, stereo = (1, True) if chan is None else (chan, False)
        csdinstr = getInstrPreset(instr, stereo)
        fade = fade if fade is not None else csdinstr.meta.get('params.fade', config['play.fade'])
        return self._play(delay=delay, dur=dur, gain=gain, csdinstr=csdinstr, chan=chan, fade=fade)
        
    def rec(self, dur=None, outfile=None, gain=None, instr=None, chan=1, sr=44100, fade=None) -> str:
        gain = gain if gain is not None else config['rec.gain']
        dur = dur if dur is not None else self._playDur()
        csdinstr = getInstrPreset(instr)
        fade = fade if fade is not None else config['play.fade']
        outfile = self._rec(delay=0, dur=dur, gain=gain, csdinstr=csdinstr, outfile=outfile,
                            sr=sr, chan=chan, block=config['rec.block'], fade=fade)
        return outfile

_globalstate = {}


def instrumentDefinitions():
    instrdefs = _globalstate.get('instrdefs')
    if instrdefs:
        return instrdefs
    
    def makeFluidInstr(preset):
        template = """
        idur, ichan, iamp0, inote0, iamp1, inote1, ifade0, ifade1 passign 3
        iscale = 1/16384
        kt = lincos:k(linseg:k(0, idur, 1), 0, 1)
        kamp  linlin kt, iamp0, iamp1
        knote linlin kt, inote0, inote1
        kfreq   mtof knote
        ivel    bpf dbamp(iamp0), -120, 0, -90, 10, -70, 20, -24, 90, 0, 127
        a0, a1  sfinstr ivel, inote0, kamp*iscale, kfreq, {preset}, gi_fluidsf, 1
        ;; aenv    linsegr 1, 0.0001, 1, 0.2, 0
        aenv    linsegr 0, ifade0, 1, 0.2, 0
        a0 *= aenv
        outch ichan, a0
        """
        return template.format(preset=preset)

    fluidInit = f'gi_fluidsf  sfload "{csoundengine.fluidsf2Path()}"'

    instrdefs: t.Dict['str', t.Dict] = {
        'sine': dict(
            body="""
            idur, ichan, iamp0, inote0, iamp1, inote1, ifade0, ifade1 passign 3
            ifade0 = ifade0 > 0.02 ? ifade0 : 0.02
            ifade1 = ifade1 > 0 ? ifade1 : ifade1
            kt  linseg 0, idur, 1
            kt  lincos kt, 0, 1
            kamp  linlin kt, iamp0, iamp1
            knote linlin kt, inote0, inote1
            a0 oscili kamp, mtof:k(knote)
            aenv linseg 0, ifade0, 1, idur-(ifade0+ifade1), 1, ifade1, 0
            a0 *= aenv
            outch ichan, a0
            """
        ),
        'tri': dict(
            body="""
            idur, ichan, iamp0, inote0, iamp1, inote1, ifade0, ifade1 passign 3
            ifade0 = ifade0 > 0.02 ? ifade0 : 0.02
            ifade1 = ifade1 > 0 ? ifade1 : ifade1
            kt  linseg 0, idur, 1
            kt  lincos kt, 0, 1
            kamp  linlin kt, iamp0, iamp1
            knote linlin kt, inote0, inote1
            a0 vco2 kamp, mtof:k(knote), 12
            ; aenv linsegr 0, ifade0, 1, ifade1, 0
            aenv linseg 0, ifade0, 1, idur-(ifade0+ifade1), 1, ifade1, 0
            a0 *= aenv
            outch ichan, a0
            """
        ),
        'saw': dict(
            body="""
            idur, ichan, iamp0, inote0, iamp1, inote1, ifade0, ifade1 passign 3
            ifade0 = ifade0 > 0.02 ? ifade0 : 0.02
            ifade1 = ifade1 > 0 ? ifade1 : ifade1
            kt  linseg 0, idur, 1
            kt  lincos kt, 0, 1
            kamp  linlin kt, iamp0, iamp1
            knote linlin kt, inote0, inote1
            a0 vco2 kamp, mtof:k(knote), 10
            ; aenv linsegr 0, ifade0, 1, ifade1, 0
            aenv linseg 0, ifade0, 1, idur-(ifade0+ifade1), 1, ifade1, 0
            a0 *= aenv
            outch ichan, a0
            """
        ),
        'sawfilt': dict(
            body="""
            idur, ichan, iamp0, inote0, iamp1, inote1, ifade0, ifade1, icutoff, iresonance passign 3
            ifade0 = ifade0 > 0.02 ? ifade0 : 0.02
            icutoff = icutoff > 0 ? icutoff : 2000
            iresonance = iresonance > 0 ? iresonance : 0.2
            ifade1 = ifade1 > 0 ? ifade1 : ifade1
            kt  linseg 0, idur, 1
            kt  lincos kt, 0, 1
            kamp  linlin kt, iamp0, iamp1
            knote linlin kt, inote0, inote1
            a0 vco2 kamp, mtof:k(knote), 10
            a0 moogladder a0, icutoff, iresonance
            aenv linseg 0, ifade0, 1, idur-(ifade0+ifade1), 1, ifade1, 0
            a0 *= aenv
            outch ichan, a0
            """
        ),
        'piano': dict(
            body=makeFluidInstr(148),
            initcode=fluidInit
        ),
        'clarinet': dict(
            body=makeFluidInstr(61),
            initcode=fluidInit
        ),
        'oboe': dict(
            body=makeFluidInstr(58),
            initcode=fluidInit
        ),
        'flute': dict(
            body=makeFluidInstr(42),
            initcode=fluidInit
        ),
        'violin': dict(
            body=makeFluidInstr(47),
            initcode=fluidInit  
        ),
        'reedorgan': dict(
            body=makeFluidInstr(52),
            initcode=fluidInit  
        ),
    } 
    _globalstate['instrdefs'] = instrdefs
    return instrdefs


def _path2name(path):
    name = os.path.splitext(os.path.split(path)[1])[0].replace("-", "_")
    return name


@lru_cache(maxsize=100)
def getInstrPreset(instr:str=None, stereo=False):
    """
    instr: if None, use the default instrument as defined in config['play.instr']
    """
    if instr is None:
        instr = config['play.instr']
    instrdef = instrumentDefinitions().get(instr)
    if instrdef is None:
        raise KeyError(f"Unknown instrument {instr}")
    group = config['play.group']
    body = instrdef['body']
    name = f'emlib.event.preset.{instr}'
    if stereo:
        name += '.stereo'
        body += "\noutch ichan+1, a0\n"
    logger.debug(f"creating csound instr. name={name}, group={group}")
    from emlib.snd import csoundengine
    engine = startPlayEngine()
    csdinstr = csoundengine.makeInstr(name=name, body=body,
                                      initcode=instrdef.get('initcode'), group=group)
    logger.debug(f"Created {csdinstr}")
    return csdinstr


def defInstrPreset(name:str, body:str, init:str=None, params:dict=None) -> None:
    """
    Define an instrument preset usable by .play in Notes, Chords, etc.

    name: the name of the instr/preset
    body: the body of the instrument (csound code). The body is the code BETWEEN `instr` and `endin` 
    init: any code to set up the instr (instr 0)

    Structure of the instrument
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Calling convention:

        idur, ichan, iamp0, inote0, iamp1, inote1, ifade passign 3
    
    ichan: channel to output to (starts with 1)
    iamp0: start amplitude
    inote0: pitch start (as midinote)
    iamp1: end amplitude
    inote1: pitch end
    ifade: fade time 
    """
    defs = instrumentDefinitions()
    instrPreset = dict(body=body, initcode=init, params=params)
    if params:
        for key, value in params.items():
            key2 = 'params.' + key.strip()
            instrPreset[key2] = value
    defs[name] = instrPreset
    

def makeFilteredSawPreset(cutoff, resonance=0.2, name=None):
    body = """
    idur, ichan, iamp0, inote0, iamp1, inote1, ifade0, ifade1 passign 3
    ifade0 = ifade0 > 0.02 ? ifade0 : 0.02
    ifade1 = ifade1 > 0 ? ifade1 : ifade1
    kt  linseg 0, idur, 1
    kt  lincos kt, 0, 1
    kamp  linlin kt, iamp0, iamp1
    knote linlin kt, inote0, inote1
    a0 vco2 kamp, mtof:k(knote), 10
    a0 moogladder a0, {cutoff}, {resonance}
    aenv linseg 0, ifade0, 1, idur-(ifade0+ifade1), 1, ifade1, 0
    a0 *= aenv
    outch ichan, a0
    """.format(cutoff=cutoff, resonance=resonance)
    if name is None:
        name = f"sawfilt{int(cutoff)}"
    return defInstrPreset(name=name, body=body)


def _normalizeName(name):
    return name.replace("-", "_").replace(" ", "_")


def defSoundfontPreset(name:str, sf2path:str, preset=0) -> None:
    """
    Define an instrument preset usable by .play, based on a soundfont (sf2)

    name: name of the preset (to be used as note.play(instr=name))
    sf2path: path to a sf2 soundfont
    preset: preset to be loaded. To find the preset number, use

    echo "inst 1" | fluidsynth "path/to/sf2" | egrep '[0-9]{3}-[0-9]{3} '

    banks are not supported

    NB: at the moment all instruments generate a mono signal, even if they
        are played as stereo (the right chan is discarded)
    """
    sf2path = os.path.abspath(sf2path)
    if not os.path.exists(sf2path):
        raise FileNotFoundError(f"Soundfont file {sf2path} not found")
    tabname = f"gi_sf2func_{_normalizeName(name)}"
    init = f'{tabname}  sfload "{sf2path}"'
    body = f"""
    idur, ichan, iamp0, inote0, iamp1, inote1, ifade0, ifade1 passign 3
    iscale = 1/16384
    kt = lincos:k(linseg:k(0, idur, 1), 0, 1)
    kamp  linlin kt, iamp0, iamp1
    knote linlin kt, inote0, inote1
    kfreq   mtof knote
    ivel    bpf dbamp(iamp0), -120, 0, -90, 10, -70, 20, -24, 90, 0, 127
    a0, a1  sfinstr ivel, inote0, kamp*iscale, kfreq, {preset}, {tabname}, 1
    aenv    linsegr 0, ifade0, 1, 0.2, 0
    a0 *= aenv
    outch ichan, a0        
    """
    defInstrPreset(name, body=body, init=init)
    

def availableInstrPresets() -> t.List[str]:
    return list(instrumentDefinitions().keys())


def _asmidi(x) -> float:
    if isinstance(x, str):
        return n2m(x)
    elif isinstance(x, (int, float)):
        return x
    elif isinstance(x, Note):
        return x.midi
    raise TypeError(f"Expected a str, a Note or a midinote, got {x}")


def Hz(freq) -> 'Note':
    """
    Create a note from a given frequency
    """
    return Note(f2m(freq))


@_functools.total_ordering
class Note(_Base):

    def __init__(self, pitch, amp=None):
        # type: (t.U[float, str], float) -> None
        """
        pitch: a midinote or a note as a string
        amp  : amplitude 0-1.
        """
        _Base.__init__(self)
        self.midi: float = _asmidi(pitch)
        self.amp:float = amp if amp is not None else 1
        self._pitchResolution = 0

    def __hash__(self) -> int:
        return hash(self.midi)

    def __call__(self, cents: int) -> 'Note':
        """
        n = Note("C4")
        n(20)   -> C4+20
        """
        return self + (cents/100.)

    @property
    def h(self):
        return self + 0.5

    @property
    def f(self):
        return self - 0.5
        
    def shift(self, freq):
        # type: (float) -> Note
        """
        Return a copy of self, shifted in freq.

        >>> C3.shift(C3.freq)
        C4
        """
        return self.clone(pitch=f2m(self.freq + freq))

    def transpose(self, step):
        # type: (float) -> Note
        """
        Return a copy of self, transposed `step` steps
        """
        return self.clone(pitch=self.midi + step)

    def __call__(self, cents):
        return Note(self.midi + cents/100., self.amp)

    def __eq__(self, other):
        # type: (...) -> bool
        if isinstance(other, str):
            other = n2m(other)
        return self.__float__() == float(other)

    def __ne__(self, other):
        # type: (...) -> bool
        return not(self == other)

    def __lt__(self, other):
        # type: (SupportsFloat) -> bool
        if isinstance(other, str):
            other = n2m(other)
        return self.__float__() < float(other)

    @property
    def freq(self):
        # type: () -> float
        return m2f(self.midi)

    @freq.setter
    def freq(self, value):
        # type: (float) -> None
        self.midi = f2m(value)

    @property
    def name(self):
        # type: () -> str
        return m2n(self.midi)

    def roundedpitch(self, semitoneDivisions:int=0):
        semitoneDivision = semitoneDivisions or config['notation.semitone_divisions']
        res = 1 / semitoneDivisions
        midi = round(self.midi / res) * res
        return midi

    def overtone(self, n:int):
        return Note(f2m(self.freq * n))
    
    @property
    def cents(self):
        # type: () -> int
        return int((self.midi - int(self.midi)) * 100)
    
    def asmusic21(self, showcents=None, pure=False, stream=None):
        # type: (bool, bool) -> m21.base.Music21Object
        if self.midi == 0:
            return m21.note.Rest()
        divs = config['notation.semitone_divisions']
        basepitch = self.roundedpitch(divs)
        if divs == 4:
            note = m21tools.makeNote(basepitch)
        else:
            note = m21.note.Note(basepitch)
        if pure:
            return note
        cents = int((self.midi - basepitch) * 100)
        note.microtone = cents
        showcents = showcents if showcents is not None else config['showcents']
        if showcents:
            note.lyric = self.centsrepr
        clef = bestClef([self])
        part = stream or m21.stream.Part()
        part.append(clef)
        part.append(note)
        return part

    @property
    def centsrepr(self):
        # type: () -> str
        cents = self.cents
        if cents == 0:
            return ""
        elif 48 <= cents <= 52:
            return ""
        elif 24 <= cents <= 26:
            return "↑"
        elif 0 < cents < 75:
            return "%d" % cents
        else:
            return "–%d" % abs(100-cents)

    def _asTableRow(self):
        elements = [m2n(self.midi)]
        if config['repr.showfreq']:
            elements.append("%dHz" % int(self.freq))
        if self.amp < 1:
            elements.append("%ddB" % round(amp2db(self.amp)))
        return elements

    def __repr__(self):
        # type: () -> str
        elements = self._asTableRow()
        return f'{elements[0].ljust(3)} {" ".join(elements[1:])}'

    def __str__(self):
        return self.name
        # elements = self._asTableRow()
        # return f'{elements[0].ljust(6)} {" ".join(elem.rjust(6) for elem in elements[1:])}'
        
    def __float__(self):
        # type: () -> float
        return float(self.midi)

    def __int__(self):
        # type: () -> int
        return int(self.midi)

    def __mod__(self, other):
        # note%25 == note+.25
        if isinstance(other, int):
            return Note(self.midi + other/100., self.amp)
        raise TypeError(f"can't perform mid between a note and {other} ({other.__class__})")

    def __add__(self, other):
        # type: (U[Note, float, int]) -> U[Note, Chord]
        if isinstance(other, (int, float)):
            return Note(self.midi + other, self.amp)
        elif isinstance(other, Note):
            return Chord([self, other])
        elif isinstance(other, str):
            return self + asNote(other)
        raise TypeError(f"can't add {other} ({other.__class__}) to a Note")

    def __sub__(self, other):
        # type: (U[Note, float, int]) -> Note
        if isinstance(other, Note):
            raise TypeError("can't substract one note from another")
        elif isinstance(other, (int, float)):
            return Note(self.midi - other, self.amp)
        raise TypeError(f"can't substract {other} ({other.__class__}) from a Note")

    def quantize(self, step=1):
        # type: (float) -> Note
        """
        Returns a new Note, rounded to step
        """
        return self.clone(pitch=round(self.midi / step) * step)

    def clone(self, pitch=None, amp=None):
        # type: (Opt[float], Opt[float]) -> Note
        return Note(pitch=pitch if pitch is not None else self.midi,
                    amp=amp if amp is not None else self.amp)

    def copy(self):
        return Note(self.midi, self.amp)

    def _csoundEvents(self, delay, dur, chan, gain, fade=0):
        amp = self.amp*gain 
        midi = self.midi
        event = [delay, dur, chan, amp, midi, amp, midi, fade]
        event = _makeEvent(delay, dur, chan, amp, midi, endamp=amp, endmidi=midi, fade=fade)
        return [event]
        
    def gliss(self, dur, endpitch, endamp=None, start=0) -> 'Event':
        return Event(pitch=self.midi, amp=self.amp, dur=dur, endpitch=endpitch, endamp=endamp, start=start)

    def event(self, dur, start=0, endpitch=None, endamp=None) -> 'Event':
        endamp = endamp if endamp is not None else self.amp
        endpitch = endpitch if endpitch is not None else self.midi
        return Event(pitch=self.midi, amp=self.amp, dur=dur, endpitch=endpitch, endamp=endamp)
        

def F(x: t.U[Fraction, float, int]) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10000000)
    

def _normalizeFade(fade):
    """
    Returns (fadein, fadeout)
    """
    if isinstance(fade, (tuple, list)):
        if len(fade) != 2:
            raise IndexError(f"fade: expected a tuple or list of len=2, got {fade}")
        fadein, fadeout = fade
    elif isinstance(fade, (int, float)):
        fadein = fadeout = fade
    else:
        raise TypeError(f"fade: expected a fadetime or a tuple of (fadein, fadeout), got {fade}")
    return fadein, fadeout

def _makeEvent(delay, dur, chan, amp, midi, endamp=None, endmidi=None, fade=None, *pargs):
    """
    fade: a single value, or a tuple (fadein, fadeout)
    """
    endamp = endamp if endamp is not None else amp
    endmidi = endmidi if endmidi is not None else midi
    fade = fade if fade is not None else config['play.fade']
    fadein, fadeout = _normalizeFade(fade)
    ev = [delay, dur, chan, amp, midi, endamp, endmidi, fadein, fadeout]
    if pargs:
        ev.extend(pargs)
    return ev


class Event(Note):

    def __init__(self, pitch, dur, endpitch=None, amp=None, endamp=None, start=0, label=""):
        # type: (float, float, t.U[float, Fraction], t.U[float, Fraction], str) -> None
        """
        An event is a note with a duration. It can have a start time
        """
        if isinstance(pitch, Note):
            note = pitch
            pitch = note.midi
            amp = amp if amp is not None else note.amp
        super().__init__(pitch, amp)
        self.dur: Fraction = F(dur)
        self.start: Fraction = F(start)
        self.endmidi = _asmidi(endpitch) if endpitch is not None else self.midi
        self.endamp = endamp if endamp is not None else self.amp
        self.label: str = label

    @classmethod
    def fromNote(cls, note, dur=1, start=0):
        return Event(note.midi, amp=note.amp, dur=dur, start=start)

    @property
    def end(self) -> float:
        return self.start + self.dur

    def __repr__(self):
        # type: () -> str
        ampstr = " %ddB" % int(round(amp2db(self.amp))) if self.amp != 1 else ""
        label = " [%s]" % self.label if self.label else ""
        start = float(self.start)
        end = float(self.end)
        pitch = m2n(self.midi).ljust(7)
        if self.endmidi == self.midi:
            out = f"<{pitch}{start:.3f}:{end:.3f}{ampstr}{label}>"
        else:
            out = f"<{m2n(self.midi)}/{m2n(self.endmidi)} {start:.3f}:{end:.3f}{ampstr}{label}>"
        return out

    def copy(self):
        # type: () -> Event
        return Event(start=self.start, pitch=self.midi, dur=self.dur,
                     amp=self.amp, label=self.label)

    def clone(self, start=None, midi=None, dur=None, amp=None, label=None,
              endpitch=None, endamp=None):
        # type: (...) -> Event
        return Event(pitch=midi if midi is not None else self.midi, 
                     amp=amp if amp is not None else self.amp,
                     endpitch=endpitch if endpitch is not None else self.endmidi,
                     endamp=endamp if endamp is not None else self.endamp,
                     dur=dur if dur is not None else self.dur,
                     start=start if start is not None else self.start,
                     label=label if label is not None else self.label)
        
    def asmusic21(self, *args, pure=False, **kws):
        # type: (...) -> m21.note.Note
        if pure or self.endmidi == self.midi:
            note = Note.asmusic21(self, *args, **kws)
            note.duration = m21.duration.Duration(self.dur)
            return note
        else:
            voice = m21.stream.Voice()
            if self.start > 0:
                voice.append(m21.note.Rest(quarterLength=self.start))
            n0 = m21.note.Note(pitch=self.midi, quarterLenght=self.dur)
            n1 = m21.note.Note(pitch=self.endmidi, quarterLength=0.125)
            # n1.priority = -1
            voice.append(n0)
            voice.append(m21.spanner.Glissando([n0, n1]))
            voice.append(n1)
            return voice

    def _playDur(self):
        return self.dur

    def _csoundEvents(self, delay, dur, chan, gain, fade=0):
        """
        delay: extra delay (delay=delay+self.start)
        dur: extra dur
        gain: final gain = self.amp * gain
        """
        event = _makeEvent(delay=delay+self.start, dur=self.dur, chan=chan, 
                           amp=self.amp*gain, midi=self.midi, 
                           endamp=self.endamp*gain, endmidi=self.endmidi, 
                           fade=fade)
        return [event]

    def __hash__(self):
        return hash((self.midi, self.dur, self.start))


def _synthgroup(synths):
    return csoundengine.SynthGroup(synths)


class EventGroup(_Base, list):
    """
    A list of (possibly concurrent) Events

    The events have a start time which is relative to this group
    """

    def __init__(self, events=None, start=0):
        _Base.__init__(self)
        self.start = start
        if events is not None:
            self.extend(events)

    def append(self, event):
        if not isinstance(event, Event):
            raise TypeError(f"expected an Event, got {event}")
        super().append(event)

    def play(self, dur=None, instr=None, gain=None, chan=None, **kws):
        """
        Any value set here will be used by the .play method of each Event, overriding
        any individual value
        """
        if dur is not None:
            events = [ev.clone(dur=dur) for ev in self]
            gr = EventGroup(events)
            return gr.play(instr=instr, gain=gain, chan=chan, **kws)

        synths = [event.play(instr=instr, gain=gain, chan=chan, **kws) for event in self]
        return _synthgroup(synths)

    def _csoundEvents(self, delay, dur, chan, gain, fade=0):
        allCsoundEvents = []
        for event in self:
            csoundEvents = event._csoundEvents(delay=delay, dur=dur, chan=chan, gain=gain, fade=fade)
            allCsoundEvents.extend(csoundEvents)
        return allCsoundEvents

    def __hash__(self):
        return hash(tuple(hash(event) for event in self))

    def asmusic21(self, pure=False):
        streams = [event.asmusic21() for event in self]
        score = m21.stream.Score()
        for stream in streams:
            score.append(stream)
        return score

def concatEvents(events):
    # type: (List[Event]) -> EventSeq
    """
    Concatenate a series of events, forcing them to not overlap each other
    """
    t0 = 0
    newevents = []
    for event in events:
        t0 = max(event.start, t0)
        newevent = event.clone(start=t0)
        newevents.append(newevent)
        t0 = newevent.end
    return EventSeq(newevents)


class NoteSeq(_Base, list):
    """
    A seq. of Notes
    """

    def __init__(self, *notes, dur=1):
        # type: (Opt[Seq[Note]]) -> None
        self._hash = None
        self._eventSeq = None
        super().__init__()
        if notes:
            if len(notes) == 1 and lib.isiterable(notes[0]):
                notes = notes[0]
            self.extend(map(asNote, notes))
        self.dur = dur

    def __getitem__(self, *args):
        out = list.__getitem__(self, *args)
        if isinstance(out, list):
            out = self.__class__(out)
        return out    

    def _changed(self):
        super()._changed()
        self._hash = None
        self._eventSeq = None

    def asEventSeq(self, dur=None):
        dur = dur or self.dur
        if self._eventSeq is None:
            self._eventSeq = EventSeq([Event(pitch=note.midi, amp=note.amp, start=dur*i, dur=dur) for i, note in enumerate(self)])
        return self._eventSeq
        
    def asChord(self):
        return Chord(self)

    def asmusic21(self, split=None, pure=False, stream=None):
        voice = stream or m21.stream.Voice()
        clef = None
        for note in self:
            stream = note.asmusic21(pure=pure)
            for obj in stream.flat:
                if obj.isClassOrSubclass((m21.clef.Clef,)):
                    if obj != clef:
                        clef = obj
                    else:
                        continue
                voice.append(obj)
        return voice
        
    def __repr__(self):
        notestr = ", ".join(n.name for n in self)
        return f"NoteSeq({notestr})"
        
    def __hash__(self):
        if self._hash is not None:
            return self._hash
        self._hash = hash(tuple(hash(note) ^ 0xF123 for note in self))
        return self._hash

    def _csoundEvents(self, *args, **kws):
        dur = kws.get('dur', self.dur)
        return self.asEventSeq(dur=dur)._csoundEvents(*args, **kws)


class ChordSeq(_Base, list):
    """
    A seq. of Chords
    """

    def __init__(self, *chords):
        self._hash = None
        super().__init__()
        if chords:
            if len(chords) == 1 and isinstance(chords[0], (list, tuple)):
                chords = chords[0]
            self.extend((asChord(chord) for chord in chords))

    def __getitem__(self, *args):
        out = list.__getitem__(self, *args)
        if not out or isinstance(out, Chord):
            return out
        elif isinstance(out, list) and isinstance(out[0], Chord):
            return self.__class__(out)
        else:
            raise ValueError("__getitem__ returned {out}, expected a Chord or a list of Chords")

    def _changed(self):
        super()._changed()
        self._hash = None
            
    def asChord(self):
        """Join all the individual chords into one chord"""
        return Chord(list(set(flatten(self))))

    def asmusic21(self, split=None, showcents=None, pure=False):
        showcents = showcents if showcents is not None else config['showcents']
        if pure:
            voice = m21.stream.Voice()
            for chord in self:
                voice.append(chord.asmusic21(pure=True))
            return voice
        else:
            return splitChords(self, split=split, showcents=showcents)
        
    def __repr__(self):
        lines = ["ChordSeq "]
        lines.extend("   "+" ".join(n.name.ljust(6) for n in ch) for ch in self)
        return "\n".join(lines)
        
    def __hash__(self):
        if self._hash is not None:
            return self._hash
        self._hash = hash(tuple(hash(chord) ^ 0x1234 for chord in self))
        return self._hash

    def _csoundEvents(self, delay, dur, chan, gain, fade=0):
        # the dur is the duration of each chord
        allevents = []
        for i, chord in enumerate(self):
            allevents.extend(chord._csoundEvents(delay=delay+i*dur, dur=dur, chan=chan, gain=gain, fade=fade))
        return allevents

def _sequentializeEvents(events, start=None):
    now = start if start is not None else events[0].start
    out = []
    for ev in events:
        out.append(ev.clone(start=now))
        now += ev.dur
    return out


class EventSeq(_Base, list):
    """
    Non overlapping seq. of events (a Voice)
    """

    def __init__(self, events=None):
        # type: (Opt[Seq[Event]]) -> None
        self._hash = None
        super(EventSeq, self).__init__()
        if events:
            if all(ev.start == 0 for ev in events):
                events = _sequentializeEvents(events)
            self.extend(events)
            
    def _changed(self) -> None:
        super()._changed()
        self.sort(key=lambda ev: ev.start)
        self._hash = None

    @property
    def start(self):
        return self[0].start

    @property
    def end(self):
        return max(ev.start + ev.dur for ev in self)
        
    def removeOverlap(self):
        # type: () -> EventSeq
        events = EventSeq()
        for ev0, ev1 in pairwise(self):
            if ev0.start + ev0.dur <= ev1.start:
                events.append(ev0)
            else:
                dur = ev1.start - ev0.start
                if dur > 0:
                    newevent = ev0.copy()
                    newevent.dur = dur
                    events.append(newevent)
        events.append(self[-1])
        self[:] = events
        return self

    def shiftTime(self, start):
        """
        Shift this seq. to start at `start`
        """
        dt = start - self.start
        return EventSeq([ev.clone(start=ev.start+dt) for ev in self])

    def hasOverlap(self):
        return any(ev1.start - ev0.end > 0 for ev0, ev1 in pairwise(self))

    def asmusic21(self, split=None, pure=False):
        # type: () -> m21.stream.Voice
        if pure:
            split = False
        split = _normalizeSplit(split)
        from m21.duration import Duration
        if self.hasOverlap():
            self.removeOverlap()
        voice = m21.stream.Voice()
        now = 0
        for ev in self:
            if ev.start > now:
                voice.append(m21.note.Rest(duration=Duration(ev.start - now)))
            voice.append(ev.asmusic21())
            now = ev.start + ev.dur
        voice.sliceByBeat(inPlace=True)
        maxbeat = int(now)+1
        voice.sliceAtOffsets(range(maxbeat), inPlace=True)
        if pure:
            return voice
        m21stream = voice
        if split:
            midi0 = min(ev.midi for ev in self)
            midi1 = max(ev.midi for ev in self)
            if midi0 < 57 and midi1 > 63:
                m21stream = m21tools.splitVoice(voice)
        return m21stream

    def __repr__(self):
        lines = ["EventSeq"]
        for ev in self:
            lines.append("    " + repr(ev))
        return "\n".join(lines)

    def __hash__(self):
        if self._hash is not None:
            return self._hash
        self._hash = hash(tuple(hash(ev) for ev in self))
        return self._hash

    def _csoundEvents(self, *args, **kws):
        csdEvents = []
        for event in self:
            csdEvents.extend(event._csoundEvents(*args, **kws))
        return csdEvents


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Chord
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Chord(_Base, list):

    def __init__(self, *notes, label=None, amp=None):
        # type: (Any, Opt[str]) -> None
        """
        a Chord can be instantiated as:

            Chord(note1, note2, ...) or
            Chord([note1, note2, ...])

        where each note is either a Note, a notename ("C4", "E4+", etc) or a midinote

        label: str. If given, it will be used for printing purposes, if possible
        """
        _Base.__init__(self)
        self.label = label
        if notes:
            if lib.isgenerator(notes):
                notes = list(notes)
            if isinstance(notes[0], (list, tuple)):
                assert len(notes) == 1
                notes = notes[0]
            elif isinstance(notes[0], str) and len(notes) == 1:
                notes = notes[0].split()
            notes = [asNote(n, amp=amp) for n in notes]
            self.extend(notes)
            self.sortbypitch(inplace=True)

    def asmusic21(self, showcents=None, split=None, arpeggio=None, pure=False):
        showcents = showcents if showcents is not None else config['showcents']
        arpeggio = arpeggio if arpeggio is not None else config['chord.arpeggio']
        split = _normalizeSplit(split)
        notes = sorted(self.notes, key=lambda n: n.midi)
        arpeggio = _normalizeChordArpeggio(arpeggio, self)
        if pure:
            ch = m21.chord.Chord([n.asmusic21(pure=True) for n in notes])
            return ch
        if arpeggio:
            voice = m21.stream.Voice()
            for n in self:
                n2 = n.asmusic21(pure=True)
                n2.duration.quarterLength = 0.5
                if showcents:
                    n2.lyric = n.centsrepr
                voice.append(n2)
            m21stream = m21tools.splitVoice(voice) if split else voice
            return m21stream
        else:
            return splitChord(self, split=split)

    def __hash__(self):
        return hash(tuple(n.midi for n in self))

    @property
    def freqs(self):
        return [n.freq for n in self]

    def append(self, note):
        # type: (U[Note, float, str]) -> None
        self._changed()
        note = asNote(note)
        if note.freq < 17:
            logger.debug(f"appending a note with very low freq: {note.freq}")
        list.append(self, note)

    def extend(self, notes):
        # type: (t.Iter[t.U[Note, float, str]]) -> None
        for note in notes:
            super().append(asNote(note))
        self._changed()

    def insert(self, index, note):
        # type: (int, t.U[Note, float, str]) -> None
        self._changed()
        note = asNote(note)
        super(self.__class__, self).insert(index, note)

    def filter(self, func):
        """
        Example: filter out notes lower than the lowest note of the piano

        return ch.filter(lambda n: n > "A0")
        """
        return Chord([n for n in self if func(n)])
        
    def transpose(self, step):
        """
        Return a copy of self, transposed `step` steps
        """
        return Chord([note.transpose(step) for note in self])

    def shift(self, freq):
        return Chord([note.shift(freq) for note in self])
        
    def quantize(self, step=1.0):
        # type: (float) -> Chord
        """
        Quantize the pitch of the notes in this chord
        Two notes with the same pitch are considered equal if
        they quantize to the same pitch, independently of their
        amplitude. In the case of two equal notes, only the first
        one is kept.
        """
        seenmidi = set()
        notes = []
        for note in self:
            note2 = note.quantize(step)
            if note2.midi not in seenmidi:
                seenmidi.add(note2.midi)
                notes.append(note2)
        return Chord(notes)

    def __setitem__(self, i, obj):
        # type: (int, t.U[Note, float, str]) -> None
        self._changed()
        note = asNote(obj)
        super(self.__class__, self).__setitem__(i, note)

    def __add__(self, other):
        # type: (t.U[Note, float, str]) -> Chord
        if isinstance(other, Note):
            s = Chord(self)
            s.append(other)
            return s
        elif isinstance(other, (int, float)):
            s = [n + other for n in self]
            return Chord(s)
        elif isinstance(other, Chord):
            return Chord(list.__add__(self, other))
        elif isinstance(other, str):
            return self + asChord(other)
        raise TypeError("Can't add a Chord to a %s" % other.__class__.__name__)

    def splitbyamp(self, numchords=8, max_notes_per_chord=16):
        # type: (int, int) -> t.List[Chord]
        midinotes = [note.midi for note in self]
        amps = [note.amp for note in self]
        return splitByAmp(midinotes, amps, numchords, max_notes_per_chord)

    def sortbyamp(self, reverse=True, inplace=True):
        # type: (bool, bool) -> Chord
        if inplace:
            out = self
        else:
            out = self.__class__(self)
        out.sort(key=lambda n: n.amp, reverse=reverse)
        return out

    def loudest(self, n):
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        return self.sortbyamp(inplace=False, reverse=True)[:n]

    def sortbypitch(self, reverse=False, inplace=True):
        # type: (bool, bool) -> Chord
        if inplace:
            out = self
        else:
            out = self.__class__(self)
        out.sort(key=lambda n: n.midi, reverse=reverse)
        return out

    @property
    def notes(self):
        # type: () -> t.List[Note]
        return [note for note in self]

    def _csoundEvents(self, delay, dur, chan, gain, fade=0):
        adjustgain = config['chord.adjustGain']
        if adjustgain:
            gain *= 1/sqrt(len(self))
            logger.debug(f"playCsound: adjusting gain by {gain}")
        events = []
        for note in self:
            amp = note.amp*gain
            event = _makeEvent(delay=delay, dur=dur, chan=chan, amp=amp, midi=note.midi, fade=fade)
            events.append(event)
        return events
        
    def asSeq(self, dur=None) -> 'NoteSeq':
        return NoteSeq(*self, dur=dur)

    def __repr__(self):
        lines = []
        justs = [6, -6, -8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

        def justify(s, spaces):
            if spaces > 0:
                return s.ljust(spaces)
            return s.rjust(-spaces)

        cls = self.__class__.__name__
        indent = " " * len(cls)
            
        for i, n in enumerate(sorted(self.notes, key=lambda note:note.midi, reverse=True)):
            elements = n._asTableRow()
            line = " ".join(justify(element, justs[i]) for i, element in enumerate(elements))
            if i == 0:
                line = f"{cls} | " + line
            else:
                line = f"{indent} | " + line
            lines.append(line)
        return "\n".join(lines)

    def __getitem__(self, *args):
        out = list.__getitem__(self, *args)
        if isinstance(out, list):
            out = self.__class__(out)
        return out    
        
    def mapamp(self, curve, db=False):
        """
        Return a new Chord with the amps of the notes modified 
        according to curve
        
        Example #1: compress all amplitudes to 30 dB

        curve = bpf.linear(-90, -30, -30, -12, 0, 0)
        newchord = chord.mapamp(curve, db=True)

        curve:
            a func mapping amp -> amp
        db:
            if True, the value returned by func is interpreted as db
            if False, it is interpreted as amplitude (0-1)
        """
        notes = []
        if db:
            for note in self:
                db = curve(amp2db(note.amp))
                notes.append(note.clone(amp=db2amp(db)))
        else:
            for note in self:
                amp2 = curve(note.amp)
                notes.append(note.clone(amp=amp2))
        return self.__class__(notes)

    def setamp(self, amp):
        """
        Returns a new Chord where each note has the given amp. 
        This is a shortcut to

        ch2 = Chord([note.clone(amp=amp) for note in ch])

        See also: .scaleamp
        """
        return self.scaleamp(0, offset=amp)

    def scaleamp(self, factor:float, offset=0.0) -> 'Chord':
        """
        Returns a new Chord with the amps scales by the given factor
        """
        return Chord([note.clone(amp=note.amp*factor+offset) for note in self])

    def equalize(self, curve):
        # type: (t.Callable[[float], float]) -> Chord
        """
        Return a new Chord equalized by curve

        curve: a func(freq) -> gain
        """
        notes = []
        for note in self:
            gain = curve(note.freq)
            notes.append(note.clone(amp=note.amp*gain))
        return self.__class__(notes)

    def gliss(self, dur, endnotes, start=0):
        # type: (float, t.List[Note]) -> EventGroup
        """
        Example: semitone glissando in 2 seconds

        ch = Chord("C4", "E4", "G4")
        ch2 = ch.gliss(2, ch.transpose(-1))

        Example: gliss with diminuendo

        Chord("C4 E4", amp=0.5).gliss(5, Chord("E4 G4", amp=0).play()
        """
        if len(endnotes) != len(self):
            raise ValueError(f"The number of end notes {len(endnotes)} != the"
                             f"size of this chord {len(self)}")
        events = []
        for note, endnote in zip(self, endnotes):
            if isinstance(endnote, Note):
                endpitch = endnote.midi
                amp1 = endnote.amp
            else:
                endpitch = _asmidi(endnote)
                amp1 = note.amp
            events.append(note.gliss(dur=dur, endpitch=endpitch, endamp=amp1, start=start))
        return EventGroup(events)

    def difftones(self):
        """
        Return a Chord representing the difftones between the notes of this chord
        """
        from emlib.music.combtones import difftones
        return Chord(difftones(self))


def _normalizeChordArpeggio(arpeggio: t.U[str, bool], chord: Chord) -> bool:
    if isinstance(arpeggio, bool):
        return arpeggio
    if arpeggio == 'always':
        return True
    elif arpeggio == 'never':
        return False
    elif arpeggio == 'auto':
        return _isChordCrowded(chord)
    else:
        raise ValueError(f"arpeggio should be True, False, always, never or auto (got {arpeggio})")


def _isChordCrowded(chord: Chord) -> bool:
    return any(abs(n0.midi - n1.midi) <= 1 and abs(n1.midi - n2.midi) <= 1 
               for n0, n1, n2 in window(chord, 3))


def stopSynths(stopengine=False):
    """ 
    Stops all synths (notes, chords, etc) being played

    If stopengine is True, the play engine itself is stopped
    """
    group = config['play.group']
    csoundengine.getManager(group).unschedAll()
    if stopengine:
        csoundengine.stopEngine(group)


def startPlayEngine(nchnls=None):
    """
    Start the play engine with a given configuration. 
    """
    group = config['play.group'] 
    if group in csoundengine.activeEngines():
        return
    nchnls = nchnls or config['play.numchannels']
    logger.info(f"Starting engine {group} (nchnls={nchnls})")
    csoundengine.getEngine(name=group, nchnls=nchnls)
    

def stopPlayEngine():
    stopSynths(stopengine=True)
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Helper functions for Note, Chord, ...
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _pngOpenExternal(path, wait=False):
    app = config['app.png']
    cmd = f'{app} "{path}"'
    if wait:
        os.system(cmd)
    else:
        os.system(cmd + " &")


try:
    from IPython.core.display import display as _jupyterDisplay
except ImportError:
    _jupyterDisplay = _pngOpenExternal


def _setJupyterHookForClass(cls, func, fmt='image/png'):
    """ 
    Register func as a displayhook for class `cls`
    """
    if not _inside_jupyter:
        logger.debug("_setJupyterHookForClass: not inside IPython/jupyter, skipping")
        return 
    import IPython
    ip = IPython.get_ipython()
    formatter = ip.display_formatter.formatters[fmt]
    return formatter.for_type(cls, func)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# music 21
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _m21JupyterHook(enable=True) -> None:
    """
    Set an ipython-hook to display music21 objects inline on the
    ipython notebook
    """
    if not _inside_jupyter:
        logger.debug("_m21JupyterHook: not inside ipython/jupyter, skipping")
        return 
    from IPython.core.getipython import get_ipython
    from IPython.core import display
    ip = get_ipython()
    formatter = ip.display_formatter.formatters['image/png']
    if enable:
        def showm21(stream):
            fmt = config.get('m21.displayhook.format', 'xml.png')
            filename = str(stream.write(fmt))
            return display.Image(filename=filename)._repr_png_()

        dpi = formatter.for_type(m21.Music21Object, showm21)
        return dpi
    else:
        logger.debug("disabling display hook")
        formatter.for_type(m21.Music21Object, None)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# notenames
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


_enharmonic_sharp_to_flat = {
    'C#': 'Db',
    'D#': 'Eb',
    'E#': 'F',
    'F#': 'Gb',
    'G#': 'Ab',
    'A#': 'Bb',
    'H#': 'C'
}
_enharmonic_flat_to_sharp = {
    'Cb': 'H',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',
    'Hb': 'A#'
}


def enharmonic(n:str) -> str:
    n = n.capitalize()
    if "#" in n:
        return _enharmonic_sharp_to_flat[n]
    elif "x" in n:
        return enharmonic(n.replace("x", "#"))
    elif "is" in n:
        return enharmonic(n.replace("is", "#"))
    elif "b" in n:
        return _enharmonic_flat_to_sharp[n]
    elif "s" in n:
        return enharmonic(n.replace("s", "b"))
    elif "es" in n:
        return enharmonic(n.replace("es", "b"))


def generateNotes(start=12, end=127) -> t.List[Note]:
    """
    Generates all notes for interactive use.

    From an interactive session, 

    >>> locals().update(generate_notes())
    >>> C4(50).freq == (C4+0.5).freq
    True
    """
    notes = {}
    for i in range(start, end):
        notename = m2n(i)
        octave = notename[0]
        rest = notename[1:]
        rest = rest.replace('#', 'x')
        original_note = rest + str(octave)
        notes[original_note] = Note(i)
        if "x" in rest or "b" in rest:
            enharmonic_note = enharmonic(rest)
            enharmonic_note += str(octave)
            notes[enharmonic_note] = Note(i)
    return notes


def notes2ratio(n1, n2, maxdenominator=16) -> Fraction:
    """
    find the ratio between n1 and n2

    n1, n2: notes -> "C4", or midinote (do not use frequencies)

    Returns: a Fraction with the ratio between the two notes

    NB: to obtain the ratios of the harmonic series, the second note
        should match the intonation of the corresponding overtone of
        the first note

    C4 : D4       --> 8/9
    C4 : Eb4+20   --> 5/6
    C4 : E4       --> 4/5
    C4 : F#4-30   --> 5/7
    C4 : G4       --> 2/3
    C4 : A4       --> 3/5
    C4 : Bb4-30   --> 4/7
    C4 : B4       --> 8/15
    """
    f1, f2 = _asfreq(n1), _asfreq(n2)
    return Fraction.from_float(f1/f2, maxdenominator=maxdenominator)
    

def _asfreq(n) -> float:
    if isinstance(n, str):
        return n2f(n)
    elif isinstance(n, (int, float)):
        return m2f(n)
    elif isinstance(n, Note):
        return n.freq
    else:
        raise ValueError("cannot convert a %s to a frequency" % str(n))


def setJupyterHook() -> None:
    _Base._setJupyterHook()
    _m21JupyterHook()


def chordNeedsSplit(chord: Chord, splitpoint=60, tolerance=4) -> bool:
    midis = [note.midi for note in chord]
    ok = all(midi >= splitpoint-tolerance for midi in midis) or all(midi <= splitpoint+tolerance for midi in midis)
    return not ok

    
def _annotateChord(chord, notes, force=False):
    if all(note.cents == 0 for note in notes):
        return
    annotations = [note.centsrepr for note in sorted(notes, reverse=True)]
    if any(annotations):
        lyric = ",".join(annotation for annotation in annotations)
        chord.lyric = lyric
    
def _makeChord(notes, showcents=True):
    m21chord = m21.chord.Chord([m21.note.Note(n.midi) for n in notes])
    if showcents:
        _annotateChord(m21chord, notes, force=True)
    return m21chord

def bestClef(notes):
    # type: (t.List[Note]) -> m21.clef.Clef
    mean = sum(note.midi for note in notes) / len(notes)
    if mean > 90:
        return m21.clef.Treble8vaClef()
    elif mean > 59:
        return m21.clef.TrebleClef()
    else:
        return m21.clef.BassClef()
    # else:
    #     return m21.clef.Bass8vbClef()
    
def _splitNotes(chord, split):
    # type: (t.List[Note], float) -> t.Tup[t.List[Note], t.List[Note]]
    above, below = [], []
    for note in chord:
        (above if note.midi > split else below).append(note)
    return above, below

def _splitNotesIfNecessary(chord, split):
    # type: (t.List[Note], float) -> t.List[t.List[Note]]
    return [part for part in _splitNotes(chord, split) if part]

def splitChord(chord, split=60.0, showcents=True, showlabel=True):
    # type: (Chord, float, bool, bool) -> m21.stream.Score
    splitpoint = float(split)
    chords = _splitNotesIfNecessary(chord, split)
    parts = []
    for splitchord in chords:
        m21chord = _makeChord(splitchord, showcents=showcents)
        part = m21.stream.Part()
        clef = bestClef(splitchord)
        part.append(clef)
        if showlabel and chord.label:
            part.append(m21.expressions.TextExpression(chord.label))
            showlabel = False
        part.append(m21chord)
        parts.append(part)
    return m21.stream.Score(parts)


def _normalizeSplit(split):
    split = split if split is not None else config['show.split']
    if isinstance(split, bool):
        split = int(split) * 60
    return split
        

def splitChords(chords, split=60, showcents=True, showlabel=True):
    chordsabove, chordsbelow = [], []
    split = _normalizeSplit(split)
    for chord in chords:
        above, below = _splitNotes(chord, split)
        chordsabove.append(above)
        chordsbelow.append(below)
    rows = [chords for chords in (chordsabove, chordsbelow) if not all(not chord for chord in chords)]
    columns = zip(*rows)
    numrows = len(rows)
    labels = [chord.label for chord in chords]

    def makePart(row):
        allnotes = list(flatten(row))
        part = m21.stream.Part()
        part.append(bestClef(allnotes))
        return part

    parts = [makePart(row) for row in rows]
    for column, label in zip(columns, labels):
        if showlabel and label:
            parts[0].append(m21.expressions.TextExpression(label))
        for chord, part in zip(column, parts):
            if chord:
                part.append(_makeChord(chord, showcents=showcents))
            else:
                part.append(m21.note.Rest())
    return m21.stream.Score(parts)    
                

@lru_cache(maxsize=1000)
def makeImage(obj: _Base, outfile:str=None, fmt:str=None, **options) -> str:
    """
    obj     : the object to make the image from (a Note, Chord, etc.)
    outfile : the path to be generated
    fmt     : format used. One of 'xml.png', 'lily.png' (no pdf)
    options : any argument passed to .asmusic21

    NB: we put it here in order to make it easier to cache images
    """
    stream = obj.asmusic21(**options)
    fmt = fmt if fmt is not None else config['show.format'].split(".")[0] + ".png"
    logger.debug(f"makeImage: using format: {fmt}")
    method, fmt3 = fmt.split(".")
    if method == 'lily' and config['use_musicxml2ly']:
        if outfile is None:
            outfile = tempfile.mktemp(suffix="." + fmt3)
        path = m21tools.makeLily(stream, fmt3, outfile=outfile)
    else:
        tmpfile = stream.write(fmt)
        if outfile is not None:
            os.rename(tmpfile, outfile)
            path = outfile
        else:
            path = tmpfile
    return str(path)


def _imgSize(path:str) -> t.Tup[int, int]:
    """ returns (width, height) """
    import PIL
    im = PIL.Image.open(path)
    return im.size


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    makeImage.cache_clear()


def asMusic(obj):
    """
    Convert obj to a Note or Chord, depending on the input itself

    int, float      -> Note
    list (of notes) -> Chord
    "C4"            -> Note
    "C4 E4"         -> Chord
    """
    if isinstance(obj, _Base):
        return obj
    elif isinstance(obj, str):
        if " " in obj:
            return Chord(obj.split())
        return Note(obj)
    elif isinstance(obj, (list, tuple)):
        return Chord(obj)
    elif isinstance(obj, (int, float)):
        return Note(obj)


def gliss(obj1, obj2, dur=1, start=0):
    m1 = asMusic(obj1)
    m2 = asMusic(obj2)
    return m1.gliss(dur, m2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_inside_jupyter = lib.inside_jupyter()
if _inside_jupyter:
    setJupyterHook()
