# -*- coding: utf-8 -*-
import functools as _functools
from collections import namedtuple
from fractions import Fraction
from math import sqrt
from functools import lru_cache
import subprocess
import logging

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
    'show.format': 'lily.png',
    'use_musicxml2ly': True,
    'app.png': '/usr/bin/feh',
    'displayhook.install': True,
    'play.method': 'csound',
    'play.dur': 2.0,
    'play.gain': 0.5,
    'play.group': 'emlib.event',
    'play.instr': 'sine',
    'play.fade': 0.05,
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


def getConfig() -> conftools.ConfigDict:
    """
    Get configuration for this module
    """
    logger.warning("getConfig is deprecated, use config directly")
    return config

    
_initdone = False


def _init() -> None:
    global _initdone
    if _initdone:
        return
    if config["m21.displayhook.install"]:
        # displayhook for m21 objects
        m21_ipythonhook()
    if config["displayhook.install"]:
        # displayhook for our own objects (Note, Chord, etc)
        set_ipython_displayhook()
    _initdone = True
    

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


def asChord(chord):
    # type: (t.U[Chord, t.List, t.Tup]) -> Chord
    if isinstance(chord, Chord):
        return chord
    elif isinstance(chord, (list, tuple)):
        return Chord([asNote(n) for n in chord])
    else:
        raise ValueError(f"cannot express this as a Chord: {chord}")


_AmpNote = namedtuple("Note", "note midi freq db step")


def split_notes_by_amp(midinotes, amps, numgroups=8, maxnotes_per_group=8):
    # type: (t.Iter[float], t.Iter[float], int, int) -> t.List[Chord]
    """
    split the notes by amp into groups (similar to a histogram based on amplitude)

    midinotes         : a seq of midinotes
    amps              : a seq of amplitudes in dB (same length as midinotes)
    numgroups         : the number of groups to divide the notes into
    maxnotes_per_group: the maximum of included notes per group, picked by loudness

    Returns
    =======

    a list of chords with length=numgroups
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


def _notename_extract_cents(note):
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


class _Base:
    _showableInitialized = False

    def __init__(self):
        self._pngimage = None

    def quantize(self, step=1.0):
        pass

    def transpose(self, step):
        pass

    def freqratio(self, ratio):
        return self.transpose(r2i(ratio))

    def show(self, **options):
        """
        Show this as an image

        options: any argument passed to .asmusic21

        NB: to use the music21 show capabilities, use note.asmusic21().show(...) or
            m21show(note.asmusic21())
        """
        png = self.makeImage(**options)
        return _open_png(png)

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
    def _ipython_displayhook(cls):
        if cls._showableInitialized:
            return
        logger.debug("music: setting ipython displayhook")
        from IPython.core.display import Image

        def reprpng(obj):
            return Image(filename=obj.makeImage(), embed=True)._repr_png_()
            
        _ipython_displayhook(cls, reprpng, fmt='image/png')

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
        synths = [csdinstr.play(event[1], delay=event[0], args=event[2:]) for event in events]
        return _synthgroup(synths)

    def _csoundEvents(self, delay: float, dur: float, chan: int, gain: float, fade=0.0):
        raise NotImplementedError("This method should be overloaded")
    
    def _rec(self, delay:float, dur:float, gain:float, chan:int, outfile:str, csdinstr, sr:int, block:bool, fade:float) -> str:
        """
        Called by .rec

        csdinstr: a CsoundInstrument
        """
        events = self._csoundEvents(delay=delay, dur=dur, chan=chan, gain=gain, fade=fade)
        args = events[0][2:]
        outfile = csdinstr.rec(dur=dur, args=args, outfile=outfile, sr=sr, nchnls=chan,
                               block=block)
        return outfile

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
        fade = fade if fade is not None else config['play.fade']
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
        idur, ichan, iamp0, inote0, iamp1, inote1, ifade passign 3
        iscale = 1/16384
        kt = lincos:k(linseg:k(0, idur, 1), 0, 1)
        kamp  linlin kt, iamp0, iamp1
        knote linlin kt, inote0, inote1
        kfreq   mtof knote
        ivel    bpf dbamp(iamp0), -120, 0, -90, 10, -70, 20, -24, 90, 0, 127
        a0, a1  sfinstr ivel, inote0, kamp*iscale, kfreq, {preset}, gi_fluidsf, 1
        aenv    linsegr 1, 0.0001, 1, 0.2, 0
        a0 *= aenv
        outch ichan, a0
        """
        return template.format(preset=preset)

    fluidInit = f'gi_fluidsf  sfload "{csoundengine.fluidsf2Path()}"'

    instrdefs: t.Dict['str', t.Dict] = {
        'sine': dict(
            body="""
            idur, ichan, iamp0, inote0, iamp1, inote1, ifade passign 3
            ifade = ifade > 0.05 ? ifade : 0.05
            kt  linseg 0, idur, 1
            kt  lincos kt, 0, 1
            kamp  linlin kt, iamp0, iamp1
            knote linlin kt, inote0, inote1
            a0 oscili kamp, mtof:k(knote)
            aenv linsegr 0, ifade, 1, ifade, 0
            a0 *= aenv
            outch ichan, a0
            """
        ),
        'tri': dict(
            body="""
            idur, ichan, iamp0, inote0, iamp1, inote1, ifade passign 3
            ifade = ifade > 0.05 ? ifade : 0.05
            kt  linseg 0, idur, 1
            kt  lincos kt, 0, 1
            kamp  linlin kt, iamp0, iamp1
            knote linlin kt, inote0, inote1
            a0 vco2 kamp, mtof:k(knote), 12
            aenv linsegr 0, ifade, 1, ifade, 0
            a0 *= aenv
            outch ichan, a0
            """
        ),
        'saw': dict(
            body="""
            idur, ichan, iamp0, inote0, iamp1, inote1, ifade passign 3
            ifade = ifade > 0.05 ? ifade : 0.05
            kt  linseg 0, idur, 1
            kt  lincos kt, 0, 1
            kamp  linlin kt, iamp0, iamp1
            knote linlin kt, inote0, inote1
            a0 vco2 kamp, mtof:k(knote), 12
            aenv linsegr 0, ifade, 1, ifade, 0
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


def defInstrPreset(name, body, init=None):
    """
    Define an instrument preset usable by .play in Notes, Chords, etc.

    name: the name of the instr/preset
    body: the body of the instrument (csound code). The body is the code BETWEEN `instr` and `endin` 
    init: any code to set up the instr (instr 0) 
    """
    defs = instrumentDefinitions()
    defs[name] = dict(body=body, init=init)


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
        clef = _bestClef([self])
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
        amp, midi = self.amp*gain, self.midi
        event = [delay, dur, chan, amp, midi, amp, midi, fade]
        return [event]
        
    def gliss(self, dur, endpitch, endamp=None) -> 'Event':
        return Event(pitch=self.midi, amp=self.amp, dur=dur, endpitch=endpitch, endamp=endamp)
        

def F(x: t.U[Fraction, float, int]) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10000000)
    

def gliss(pitch, endpitch, dur, start=0, amp=None, endamp=None, label=""):
    return Event()


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
        
    def asmusic21(self, *args, **kws):
        # type: (...) -> m21.note.Note
        note = Note.asmusic21(self, *args, **kws)
        note.duration = m21.duration.Duration(self.dur)
        return note

    def _playDur(self):
        return self.dur

    def _csoundEvents(self, delay, dur, chan, gain, fade=0):
        amp, midi = self.amp*gain, self.midi
        event = [delay + self.start, self.dur, chan, self.amp*gain, self.midi, self.endamp*gain, self.endmidi, fade]
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

    def play(self, instr=None, gain=None, chan=None, **kws):
        """
        Any value set here will be used by the .play method of each Event, overriding
        any individual value
        """
        synths = [event.play(instr=instr, gain=gain, chan=chan, **kws) for event in self]
        return _synthgroup(synths)

    def __hash__(self):
        return hash(tuple(hash(event) for event in self))

    def asmusic21(self, pure=False):
        streams = [event.asmusic21(pure=True) for event in self]
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

    def __init__(self, *notes):
        # type: (Opt[Seq[Note]]) -> None
        self._hash = None
        self._eventSeq = None
        super().__init__()
        if notes:
            if len(notes) == 1 and lib.isiterable(notes[0]):
                notes = notes[0]
            self.extend(map(asNote, notes))

    def __getitem__(self, *args):
        out = list.__getitem__(self, *args)
        if isinstance(out, list):
            out = self.__class__(out)
        return out    

    def _changed(self):
        super()._changed()
        self._hash = None
        self._eventSeq = None

    def asEventSeq(self, dur=1):
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
        return self.asEventSeq()._csoundEvents(*args, **kws)


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
        chordstr = ", ".join(" ".join(n.name for n in ch) for ch in self)
        return f"ChordSeq[{chordstr}]"
        
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
            
    def _changed(self):
        super()._changed()
        self.sort(key=lambda ev: ev.start)
        self._hash = None

    @property
    def start(self):
        return self[0].start

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

    def __init__(self, *notes, label=None):
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
            if isinstance(notes[0], (list, tuple)):
                assert len(notes) == 1
                notes = notes[0]
            elif isinstance(notes[0], str) and len(notes) == 1:
                notes = notes[0].split()
            notes = set(map(asNote, notes))
            self.extend(notes)
            self.sortbypitch(inplace=True)

    def __hash__(self):
        return hash(tuple(n.midi for n in self))

    def append(self, note):
        # type: (U[Note, float, str]) -> None
        self._changed()
        note = asNote(note)
        if note.freq < 17:
            logger.debug(f"appending a note with very low freq: {note.freq}")
        super(self.__class__, self).append(note)

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
        return self.__class__(notes)

    def __setitem__(self, i, obj):
        # type: (int, t.U[Note, float, str]) -> None
        self._changed()
        note = asNote(obj)
        super(self.__class__, self).__setitem__(i, note)

    def __add__(self, other):
        # type: (t.U[Note, float, str]) -> Chord
        if isinstance(other, Note):
            s = self.copy()
            s.append(other)
            return s
        elif isinstance(other, (int, float)):
            s = [n + other for n in self]
            return Chord(s)
        elif isinstance(other, Chord):
            return Chord(list.__add__(self, other))
        raise TypeError("Can't add a Chord to a %s" % other.__class__.__name__)

    def splitbyamp(self, numchords=8, max_notes_per_chord=16):
        # type: (int, int) -> t.List[Chord]
        midinotes = [note.midi for note in self]
        amps = [note.amp for note in self]
        return split_notes_by_amp(midinotes, amps, numchords,
                                  max_notes_per_chord)

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

    def copy(self):
        # type: () -> Chord
        return self.__class__(self.notes)

    @property
    def notes(self):
        # type: () -> t.List[Note]
        return [note for note in self]

    def asmusic21(self, showcents=None, split=None, arpeggio=None, pure=False):
        showcents = showcents if showcents is not None else config['showcents']
        arpeggio = arpeggio if arpeggio is not None else config['chord.arpeggio']
        split = _normalizeSplit(split)
        notes = sorted(self.notes, key=lambda n: n.midi)
        arpeggio = _normalize_chord_arpeggio(arpeggio, self)
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
            return splitChord(self)
            
    def _csoundEvents(self, delay, dur, chan, gain, fade=0):
        adjustgain = config['chord.adjustGain']
        if adjustgain:
            gain *= 1/sqrt(len(self))
            logger.debug(f"playCsound: adjusting gain by {gain}")
        events = []
        for note in self:
            amp = note.amp*gain
            events.append([delay, dur, chan, amp, note.midi, amp, note.midi, fade])
        return events
        
    def _rec(self, delay: float, dur: float, gain: float, chan: int, outfile: str,
             sr: int, csdinstr: 'csoundengine.CsoundInstr', block: bool, **kwargs):
        events = self._csoundEvents(delay=delay, dur=dur, gain=gain, chan=chan)
        return csdinstr.recEvents(outfile=outfile, events=events, sr=sr, nchnls=chan,
                                  block=block)
        
    def asSeq(self, dur=0.5) -> 'NoteSeq':
        return NoteSeq(*self)
        # events = [Event(n.pitch, amp=n.amp, start=i*dur, dur=dur) for i, note in enumerate(self)]
        # return EventSeq(events)

    def __repr__(self):
        lines = []
        justs = [6, -6, -8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

        def justify(s, spaces):
            if spaces > 0:
                return s.ljust(spaces)
            return s.rjust(-spaces)

        for i, n in enumerate(sorted(self.notes, key=lambda note:note.midi, reverse=True)):
            elements = n._asTableRow()
            line = " ".join(justify(element, justs[i]) for i, element in enumerate(elements))
            if i == 0:
                line = "Chord | " + line
            else:
                line = "      | " + line
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
        """
        return self.scaleamp(0, offset=amp)

    def scaleamp(self, factor:float, offset=0.0) -> 'Chord':
        """
        Returns a new Chord with the amps scales by the given factor
        """
        return self.__class__([note.clone(amp=note.amp*factor+offset) for note in self])

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

    def gliss(self, dur, endnotes):
        # type: (float, t.List[Note]) -> EventGroup
        """
        Example: semitone glissando in 2 seconds

        ch = Chord("C4", "E4", "G4")
        ch2 = ch.gliss(2, ch.transpose(-1))
        """
        if len(endnotes) != len(self):
            raise ValueError(f"The number of end notes {len(endnotes)} != the"
                             f"size of this chord {len(self)}")
        events = []
        for note, endnote in zip(self, endnotes):
            if isinstance(endnote, Note):
                endpitch = endnote.midi
                endamp = endnote.amp
            else:
                endpitch = _asmidi(endnote)
                endamp = note.amp
            events.append(note.gliss(dur=dur, endpitch=endpitch, endamp=endamp))
        return EventGroup(events)

    def difftones(self):
        """
        Return a Chord representing the difftones between the notes of this chord
        """
        from emlib.music.combtones import difftones
        return Chord(difftones(self))


def _normalize_chord_arpeggio(arpeggio: t.U[str, bool], chord: Chord) -> bool:
    if isinstance(arpeggio, bool):
        return arpeggio
    if arpeggio == 'always':
        return True
    elif arpeggio == 'never':
        return False
    elif arpeggio == 'auto':
        return _is_chord_crowded(chord)
    else:
        raise ValueError(f"arpeggio should be True, False, always, never or auto (got {arpeggio})")


def _is_chord_crowded(chord: Chord) -> bool:
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

_POOL = None


def _getpool():
    global _POOL 
    if _POOL is not None:
        return _POOL
    from concurrent import futures
    _POOL = futures.ThreadPoolExecutor(2)
    return _POOL


def _m21show(obj, fmt=None, wait=False):
    fmt = fmt if fmt is not None else config['show.format']
    if fmt == 'repr':
        print(repr(obj))
        return
    if wait:
        obj.show(fmt)
    else:
        pool = _getpool()
        pool.submit(obj.show, fmt)


def _open_png(path, wait=False):
    app = config['app.png']
    proc = subprocess.Popen([app, path])
    if wait:
        proc.wait()


def _ipython_displayhook(cls, func, fmt='image/png'):
    """ 
    Register func as a displayhook for class `cls`
    """
    import IPython
    ip = IPython.get_ipython()
    if ip is None:
        logger.debug("_ipython_displayhook: not inside IPython/jupyter, skipping")
        return 
    formatter = ip.display_formatter.formatters[fmt]
    return formatter.for_type(cls, func)


# ------------------------------------------------------------
#
# music 21
#
# ------------------------------------------------------------

@lib.returns_tuple(["chords", "stream"])
def chords_to_music21(chords, labels=None):
    # type: (t.Seq[Chord], t.Seq[str]) -> t.Tup[t.List[Chord], m21.stream.Stream]
    """
    This function can be used after calling split_notes_by_amp 
    to generate a music21 stream

    chords: a seq of chords, where each chord is a seq of midinotes
    labels: labels to use for the chords, or None

    Returns: chords (a List of Chords), music21 stream
    
    Example
    ~~~~~~~

    >>> chords = [(60, 63, 65), (40, 45, 48)]
    """
    stream = m21.stream.Stream()
    chords2 = []
    for chord in chords:
        if chord:
            notes = [Note(note.midi) for note in chord]
            chord = Chord(notes)
            chords2.append(chord)
    chords3 = reversed(chords2)
    for i, chord in enumerate(chords3):
        ch = chord.as_m21()
        if labels is not None:
            ch.addLyric(labels[i])
        stream.append(ch)
    return chords2, stream


def showChords(chords, labels=None, method=None):
    if method is None:
        method = config['show.format']
    chords2, stream = chords_to_music21(chords, labels)
    stream.show(method)


show_chords = lib.deprecated(showChords)    


def showChord(notes, align='vert', method=None):
    """
    Display the notes as a chord

    notes      : a seq of midinotes or Note(s)
    align      : 'horiz' or 'vert'. Show the notes as chord or arpeggio
    showmethod : backend:method

    backend     methods available

    music21     musicxml
                midi
                lily

    """
    if method is None:
        method = config['show.format']
    if align == 'vert':
        stream = m21.stream.Stream()
        notes = [asNote(note) for note in notes]
        chord = Chord(notes)
        stream.append(chord.m21)
        stream.show(method)
    elif align == 'horiz':
        chords = [Chord(n) for n in notes]
        stream = splitchords(chords)
        stream.show(method)
        return stream

show_chord = lib.deprecated(showChord)


def m21_ipythonhook(enable=True) -> None:
    """
    Set an ipython-hook to display music21 objects inline on the
    ipython notebook
    """
    from IPython.core.getipython import get_ipython
    ip = get_ipython()
    if ip is None:
        logger.debug("m21_ipythonhook: not inside ipython/jupyter, skipping")
        return 
    from IPython.core import display
    formatter = ip.display_formatter.formatters['image/png']
    if enable:
        def showm21(stream):
            return display.Image(filename=str(stream.write('lily.png')))._repr_png_()

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


class _M2N(object):
    notes_sharp = "C C# D D# E F F# G G# A A# B C".split()
    notes_flat = "C Db D Eb E F Gb G Ab A Bb B C".split()
    enharmonic_sharp_to_flat = {
        'C#': 'Db',
        'D#': 'Eb',
        'E#': 'F',
        'F#': 'Gb',
        'G#': 'Ab',
        'A#': 'Bb',
        'H#': 'C'
    }
    enharmonic_flat_to_sharp = {
        'Cb': 'H',
        'Db': 'C#',
        'Eb': 'D#',
        'Fb': 'E',
        'Gb': 'F#',
        'Ab': 'G#',
        'Bb': 'A#',
        'Hb': 'A#'
    }


def _m2n(midinote):
    integer = int(midinote)
    cents = int((midinote - integer) * 100 + 0.5)
    if cents > 50:
        integer += 1
        cents -= 100
    octave = int(integer / 12) - 1
    pitch = integer % 12
    if cents >= 0:
        note = _M2N.notes_sharp[pitch]
    else:
        note = _M2N.notes_flat[pitch]
    if cents == 0:
        return note + str(octave)
    elif cents == 50:
        return "%s%d+" % (note, octave)
    elif cents == -50:
        return "%s%d-" % (note, octave)
    elif cents > 0:
        return "%s%d+%02d" % (note, octave, cents)
    elif cents < 0:
        return "%s%d-%02d" % (note, octave, abs(cents))


def enharmonic(n):
    n = n.capitalize()
    if "#" in n:
        return _M2N.enharmonic_sharp_to_flat[n]
    elif "x" in n:
        return enharmonic(n.replace("x", "#"))
    elif "is" in n:
        return enharmonic(n.replace("is", "#"))
    elif "b" in n:
        return _M2N.enharmonic_flat_to_sharp[n]
    elif "s" in n:
        return enharmonic(n.replace("s", "b"))
    elif "es" in n:
        return enharmonic(n.replace("es", "b"))


def generateNotes(start=12, end=127):
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


generate_notes = lib.deprecated(generateNotes)


def notes2ratio(n1, n2, maxdenominator=16):
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
    

def _asfreq(n):
    if isinstance(n, str):
        return n2f(n)
    elif isinstance(n, (int, float)):
        return m2f(n)
    elif isinstance(n, Note):
        return n.freq
    else:
        raise ValueError("cannot convert a %s to a frequency" % str(n))


def set_ipython_displayhook():
    _Base._ipython_displayhook()


def chordNeedsSplit(chord, splitpoint=60):
    midis = [note.midi for note in chord]
    ok = all(midi >= splitpoint for midi in midis) or all(midi <= splitpoint for midi in midis)
    return not ok
    
def _annotateChord(chord, notes, force=False):
    if all(note.cents == 0 for note in notes):
        return
    annotations = [note.centsrepr for note in sorted(notes, reverse=True)]
    lyric = ",".join(annotation for annotation in annotations)
    chord.lyric = lyric
    
def _makeChord(notes, showcents=True):
    m21chord = m21.chord.Chord([m21.note.Note(n.midi) for n in notes])
    if showcents:
        _annotateChord(m21chord, notes, force=True)
    return m21chord

def _bestClef(notes):
    mean = sum(note.midi for note in notes) / len(notes)
    if mean > 90:
        return m21.clef.Treble8vaClef()
    elif mean > 58:
        return m21.clef.TrebleClef()
    elif mean > 36:
        return m21.clef.BassClef()
    else:
        return m21.clef.Bass8vbClef()
    
def _splitNotes(chord, split):
    above, below = [], []
    for note in chord:
        (above if note.midi > split else below).append(note)
    return above, below


def splitChord(chord:Chord, split:t.U[int,float]=60, showcents=True):
    chords = _splitNotes(chord, split)
    parts = []
    for chord in chords:
        if chord:
            m21chord = _makeChord(chord, showcents=showcents)
            part = m21.stream.Part()
            clef = _bestClef(chord)
            part.append(clef)
            part.append(m21chord)
            parts.append(part)
    return m21.stream.Score(parts)


def _normalizeSplit(split):
    split = split if split is not None else config['show.split']
    if isinstance(split, bool):
        split = int(split) * 60
    return split
        

def splitChords(chords, split=60, showcents=True):
    chordsabove, chordsbelow = [], []
    split = _normalizeSplit(split)
    for chord in chords:
        above, below = _splitNotes(chord, split)
        chordsabove.append(above)
        chordsbelow.append(below)
    chords_seq = [chords for chords in (chordsabove, chordsbelow) if not all(not chord for chord in chords)]
    columns = zip(*chords_seq)
    numrows = len(chords_seq)

    def makePart(chords):
        part = m21.stream.Part()
        allnotes = list(flatten(chords))
        clef = _bestClef(allnotes)
        part.append(clef)
        return part

    parts = [makePart(chords) for chords in chords_seq]
    for column in columns:
        for chord, part in zip(column, parts):
            part.append( _makeChord(chord, showcents=showcents) if chord else m21.note.Rest() )
    return m21.stream.Score(parts)    
                

@lru_cache(maxsize=1000)
def makeImage(obj, outfile=None, **options) -> str:
    """
    options: any argument passed to .asmusic21

    NB: we put it here in order to make it easier to cache the images
    """
    stream = obj.asmusic21(**options)
    fmt = config['show.format'].split(".")[0] + ".png"
    logger.debug(f"makeImage: using format: {fmt}")
    method, fmt3 = fmt.split(".")
    if outfile is None:
        import tempfile
        outfile = tempfile.mktemp(suffix="." + fmt3)
    if method == 'lily' and config['use_musicxml2ly']:
        path = m21tools.makeLily(stream, fmt3, outfile=outfile)
    else:
        path = stream.write(fmt, outfile)
    return str(path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

_init()
