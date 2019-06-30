# from __future__ import annotations
import functools as _functools
from collections import namedtuple
from dataclasses import dataclass 
from fractions import Fraction
from math import sqrt
import os
import copy as _copy
import tempfile as _tempfile

import music21 as m21

from bpf4 import bpf
from emlib import lib as _lib
from emlib.music import m21tools
from emlib.music import m21fix
from emlib.pitchtools import amp2db, db2amp, m2n, n2m, m2f, f2m, r2i, n2f, str2midi, set_reference_freq
from emlib import iterlib as _iterlib
from emlib.music import scoring

from emlib.snd import csoundengine
import emlib.typehints as t

from .config import config, logger as _logger 
from . import m21funcs
from . import play
from . import tools


_Num = t.U[float, int, Fraction]
_Pitch = t.U['Note', float, str]
_AmpNote = namedtuple("Note", "note midi freq db step")
_T = t.TypeVar('_T')

_MAXDUR = 99999


def F(x: t.U[Fraction, float, int], den=None, maxden=1000000) -> Fraction:
    if den is not None:
        return Fraction(x, den).limit_denominator(maxden)
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(maxden)


@dataclass 
class _State:
    a4: int = 442
    tempo: Fraction = Fraction(60)
    timefactor: Fraction = Fraction(1)

    def __post_init__(self):
        self.timefactor = Fraction(60) / Fraction(self.tempo)

    def setTempo(self, tempo):
        self.tempo = F(tempo)
        self.timefactor = F(60) / self.tempo

    def setA4(self, a4):
        self.a4 = a4
        set_reference_freq(a4)


_statestack = [_State(a4=config.get('A4', 442), tempo=60.0)]


def pushState(a4=None, tempo=None):
    """
    Push a new state to the global state stack. A new state inherits values 
    not set from the earlier state
    """
    currState = _statestack[-1]
    a4 = a4 if a4 is not None else currState.a4
    tempo = tempo if tempo is not None else currState.tempo 
    tempo = F(tempo)
    state = _State(a4=a4, tempo=tempo)

    _statestack.append(state)
    set_reference_freq(a4)
    return state


pushState(a4=config.get('A4', 442), tempo=60.0)


def getState():
    """
    Get current state
    """
    return _statestack[-1]


def popState():
    """
    Pop a global state from the stack, return the now invalid state
    """
    if len(_statestack) == 1:
        return _statestack[-1]
    laststate = _statestack.pop()
    return laststate


def setTempo(rpm:float):
    getState().setTempo(rpm)


def setA4(a4:float):
    getState().setA4(a4)


def asTime(x: _Num):
    if not isinstance(x, Fraction):
        x = F(x)
    return x.limit_denominator(128)


def Hz(freq:float) -> 'Note':
    """
    Create a note from a given frequency

    a4 = Hz(442)
    """
    return Note(f2m(freq))


def _splitByAmp(midinotes, amps, numgroups=8, maxnotes_per_group=8):
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


class _CsoundLine:
    """
    Represents a standard event (a line of variable breakpoints)

    init: delay=0, chan=0, fadein=None, fadeout=None
    bp: delay, midi, amp, ...
    (all breakpoints should have the same length)

    protocol:

    i inum, delay, dur, p4, p5, ...
    i inum, delay, dur, chan, fadein, fadeout, bplen, *flat(bps)

    inum is given by the manager
    dur is calculated based on bps
    bplen is calculated based on bps (it is the length of one breakpoint)
    """

    def __init__(self, bps:t.List[tuple], delay=0.0, chan:int=0,
                 fadein:float=None, fadeout:float=None, gain:float=1.0):
        """
        bps (breakpoints): a seq of (delay, midi, amp, ...) of len >= 1.
        """
        bps = tools.fillColumns(bps)
        if len(bps[0]) < 2:
            raise ValueError(f"A breakpoint should have at least (delay, pitch), but got {bps}")
        if len(bps[0]) < 3:
            bps = tools.addColumn(bps, 1)
        assert all(len(bp) >= 3 for bp in bps)
        assert all(isinstance(bp, tuple) for bp in bps)
        self.bps = bps
        self.delay = delay
        self.chan = chan
        self.gain = gain
        self._consolidateDelay()
        self.fadein = max(fadein, 0.0001)
        dur = self.getDur()
        self.fadeout = fadeout if dur < 0 else min(fadeout, dur)
        
    def _consolidateDelay(self):
        delay0 = self.bps[0][0]
        if delay0 > 0:
            self.delay += delay0
            self.bps = [(bp[0] - delay0,) + bp[1:] for bp in self.bps]

    def getDur(self) -> float:
        return self.bps[-1][0]

    def breakpointSize(self) -> int:
        return len(self.bps[0])

    def getArgs(self) -> list:
        """
        get values from p4 on

        gain, chan, fade0, fade1, numbps, bplen, ... data
        """
        args = [self.gain, self.chan, self.fadein, self.fadeout, len(self.bps), self.breakpointSize()]
        args.extend(_iterlib.flatten(self.bps))
        return args

    def __repr__(self):
        lines = [f"CsoundLine(delay={float(self.delay):.3f}, gain={self.gain}, chan={self.chan}"
                 f", fadein={self.fadein}, fadeout={self.fadeout}"]
        for bp in self.bps:
            lines.append(f"    {float(bp[0]):.3f} {bp[1:]}")
        lines.append("")
        return "\n".join(lines)


def m21FromScoringEvents(events: t.List[scoring.Event], split=None, showcents=None, divsPerSemitone=None) -> m21.stream.Stream:
    import emlib.music.scoring.quantization as quant
    divsPerSemitone = divsPerSemitone if divsPerSemitone is not None else config['show.semitoneDivisions']
    showcents = showcents if showcents is not None else config['show.cents']
    showgliss = config['show.gliss']
    split = split if split is not None else config['show.split']
    if split and _midinotesNeedSplit([event.avgPitch() for event in events]):
        parts = quant.quantizeVoiceSplit(events, divsPerSemitone=divsPerSemitone,
                                         showcents=showcents, showgliss=showgliss)
        return m21tools.stackParts(parts)
    return quant.quantizeVoice(events, divsPerSemitone=divsPerSemitone,
                               showcents=showcents, showgliss=showgliss)

def _getValue(*values):
    """
    Get the first value in values which is not None

    For delayed execution (similar to a if b else c) a value can be a 
    callable which should return the value in question

    Example:

    default = _getValue(a, config['a'], lambda: lengthycomputation())
    """
    for value in values:
        if callable(value):
            value2 = value()
            if value2 is not None:
                return value2
        elif value is not None:
            return value

class _Base:
    _showableInitialized = False

    def __init__(self, label=None, dur:_Num=None, start:_Num=None, playargs:dict=None):
        self._pngimage = None
        self.label = label
        self.dur = F(dur) if dur is not None else None
        self.start = F(start) if start is not None else None
        self.playargs: dict = playargs or {}

    def setplay(self, **kws):
        self.playargs.update(kws)
        return self

    def clone(self:_T, **kws) -> _T:
        out = _copy.deepcopy(self)
        for key, value in kws.items():
            setattr(out, key, value)
        return out

    def copy(self:_T) -> _T:
        return _copy.deepcopy(self)

    def delay(self:_T, timeoffset:_Num) -> _T:
        """
        Return a copy of this object with an added time offset

        Example: create a seq. of syncopations

        n = Note("A4", start=0.5, dur=0.5)
        seq = Track([n, n.delay(1), n.delay(2), n.delay(3)])

        This is the same as 

        seq = Track([n, n>>1, n>>2, n>>3])
        """
        start = self.start or 0.0
        return self.clone(start=timeoffset + start)

    def __rshift__(self:_T, timeoffset:_Num) -> _T:
        return self.delay(timeoffset)

    def __lshift__(self:_T, timeoffset:_Num) -> _T:
        return self.delay(-timeoffset)

    @property
    def end(self) -> t.Opt[Fraction]:
        if self.dur is None:
            return None
        start = self.start if self.start is not None else 0
        return start + self.dur

    def quantize(self:_T, step=1.0) -> _T:
        pass

    def transpose(self:_T, step) -> _T:
        pass

    def freqratio(self:_T, ratio) -> _T:
        """
        Transpose this by a given freq. ratio. A ratio of 2 equals to transposing an octave
        higher.
        """
        return self.transpose(r2i(ratio))

    def show(self, external=None, fmt=None, **options) -> None:
        """
        Show this as an image.

        external: 
            force opening the image in an external image viewer, even when
            inside a jupyter notebook. Otherwise, show will display the image
            inline
        fmt:
            overrides the config setting 'show.format'
            One of 'xml.png', 'xml', 'xml.pdf', 'lily.png', 'lily.pdf'
            
        options: any argument passed to .asmusic21


        NB: to use the music21 show capabilities, use note.asmusic21().show(...) or
            m21show(note.asmusic21())
        """
        external = external if external is not None else config['show.external']
        png = self.makeImage(fmt=fmt, **options)
        tools.pngShow(png, external=external)
        
    def _changed(self) -> None: ...
        
    def makeImage(self, **options) -> str:
        """
        Creates an image representation, returns the path to the image

        options: any argument passed to .asmusic21

        """
        if config['show.cacheImages']:
            return _makeImageCached(self, **options)
        return _makeImage(self, **options)

    def ipythonImage(self):
        from IPython.core.display import Image
        return Image(self.makeImage(), embed=True)

    def scoringEvents(self) -> t.List[scoring.Event]: ...

    def music21objs(self) -> t.List[m21.Music21Object]:
        events = self.scoringEvents()
        return [event.asmusic21() for event in events]

    def asmusic21(self, **options) -> m21.stream.Stream:
        return m21FromScoringEvents(self.scoringEvents())

    def musicxml(self) -> str:
        """
        Return the representation of this object as musicxml
        """
        m = self.asmusic21()
        if config['m21.fixstream']:
            m21fix.fixStream(m)
        return m21tools.getXml(m)

    @classmethod
    def _setJupyterHook(cls):
        if cls._showableInitialized:
            return
        from IPython.core.display import Image

        def reprpng(obj):
            imgpath = obj.makeImage()
            scaleFactor = config.get('show.scaleFactor', 1.0)
            if scaleFactor != 1.0:
                imgwidth, imgheight = tools.imgSize(imgpath)
                width = imgwidth * scaleFactor
            else:
                width = None
            return Image(filename=imgpath, embed=True, width=width)._repr_png_()
            
        tools.setJupyterHookForClass(cls, reprpng, fmt='image/png')

    def csoundEvents(self, delay: float, dur: float, chan: int, gain: float, fade=0.0
                     ) -> t.List['_CsoundLine']:
        raise NotImplementedError("This method should be overloaded")

    def getEvents(self, delay:float=None, dur:float=None, chan:int=1,
                  gain=None, fade=None):
        gain = gain or config['play.gain']
        dur = self.dur or dur or self.dur or config['play.dur']
        if dur < 0:
            dur = _MAXDUR
        delay = (self.start or 0.0) + (delay or 0.0)
        events = self.csoundEvents(delay=delay,
                                   dur=dur,
                                   chan=chan, gain=gain, fade=fade)
        return events

    def play(self, dur=None, gain=None, delay=None, instr=None, chan=None, fade=None):
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
        # p4=ichan, p5=ifade0, p6=ifade1, p7=ibplen, p8... = bps
        playargs = self.playargs
        chan = _getValue(chan, playargs.get('chan'))
        chan, stereo = (1, True) if chan is None else (chan, False)
        instr = _getValue(instr, playargs.get('instr'))
        csdinstr = play.getInstrPreset(instr, stereo)
        fade = _getValue(fade, playargs.get('fade'), lambda:csdinstr.meta.get('params.fade', config['play.fade']))
        gain = _getValue(gain, playargs.get('gain'))
        events = self.getEvents(delay=delay, dur=dur, chan=chan, fade=fade, gain=gain)
        assert events, f"No events for obj {self}"
        if len(events) == 1:
            event = events[0]
            return csdinstr.play(dur=event.getDur(), delay=event.delay, args=event.getArgs())
        synths = [csdinstr.play(dur=ev.getDur(), delay=ev.delay, args=ev.getArgs()) for ev in events]
        return csoundengine.SynthGroup(synths)

    def chan(self, chan, **kws):
        return self.play(chan=chan, **kws)

    def rec(self, dur=None, outfile=None, gain=None, instr=None, chan=1, sr=44100,
            fade=None, block=None) -> str:
        gain = gain if gain is not None else config['rec.gain']
        dur = dur or self.dur or config['play.dur']
        fade = fade if fade is not None else config['play.fade']
        block = block if block is not None else config['rec.block']
        csdinstr = play.getInstrPreset(instr)
        events = self.csoundEvents(delay=0, dur=dur, chan=chan, gain=gain, fade=fade)
        outfile = csdinstr.recEvents(outfile=outfile, events=events, sr=sr, nchnls=chan,
                                     block=block)
        return outfile


class _Unset: pass

_UNSET = _Unset()


@_functools.total_ordering
class Note(_Base):

    def __init__(self, pitch: t.U[float, str], amp:float=None,
                 dur:Fraction=None, start:Fraction=None, endpitch:t.U[float, str]=None,
                 label:str=None):
        """
        pitch: a midinote or a note as a string
        amp  : amplitude 0-1.
        
        """
        _Base.__init__(self, label=label, dur=dur, start=start)
        self.midi: float = tools.asmidi(pitch)
        self.amp:float = amp if amp is not None else 1
        self.endmidi = tools.asmidi(pitch) if pitch is not None else None
    
    def __hash__(self) -> int:
        return hash((self.midi, self.dur, self.start, self.endmidi, self.label))

    def clone(self, pitch: t.U[float, str]=None, amp:float=None, dur:Fraction=_UNSET, 
              start:Fraction=_UNSET, label:str=None) -> 'Note':
        # we can't use the base .clone method because pitch can be anything
        return Note(pitch=pitch if pitch is not None else self.midi, 
                    amp=amp if amp is not None else self.amp, 
                    dur=dur if dur is not _UNSET else self.dur,
                    start=start if start is not _UNSET else self.start, 
                    label=label or self.label)

    def fill(self, amp:float=None, dur:Fraction=None, start:Fraction=None):
        """
        Similar to clone, but uses the information given only if
        this information is not already present in this note
        """
        amp = self.amp if self.amp is not None else amp
        dur = self.dur if self.dur is not None else dur
        start = self.start if self.start is not None else start
        if any(info is not None for info in (amp, dur, start)):
            return self.clone(amp=amp, dur=dur, start=start)
        return self

    # s: half semitone shart, f: half semitone flat, qs: quarter semitone shart, qf: qf quarter semitone flat
    @property
    def s(self):
        return self + 0.5

    @property
    def f(self):
        return self - 0.5

    @property
    def qs(self):
        return self + 0.25

    @property
    def qf(self):
        return self - 0.25

    def asChord(self):
        return Chord(self)
        
    def shift(self, freq:float) -> 'Note':
        """
        Return a copy of self, shifted in freq.

        C3.shift(C3.freq)
        -> C4
        """
        return self.clone(pitch=f2m(self.freq + freq))

    def transpose(self, step: float) -> 'Note':
        """
        Return a copy of self, transposed `step` steps
        """
        return self.clone(pitch=self.midi + step)

    def __call__(self, cents:int) -> 'Note':
        return Note(self.midi + cents/100., self.amp)

    def __eq__(self, other:_Pitch) -> bool:
        if isinstance(other, str):
            other = str2midi(other)
        return self.__float__() == float(other)

    def __ne__(self, other: _Pitch) -> bool:
        return not(self == other)

    def __lt__(self, other: _Pitch) -> bool:
        if isinstance(other, str):
            other = str2midi(other)
        return self.__float__() < float(other)

    @property
    def freq(self) -> float:
        return m2f(self.midi)

    @freq.setter
    def freq(self, value:float) -> None:
        self.midi = f2m(value)

    @property
    def name(self) -> str:
        return m2n(self.midi)

    def roundPitch(self, semitoneDivisions:int=0) -> float:
        divs = semitoneDivisions or config['show.semitoneDivisions']
        res = 1 / semitoneDivisions
        return self.quantize(res)
    
    def overtone(self, n:int) -> 'Note':
        return Note(f2m(self.freq * n))
    
    @property
    def cents(self) -> int:
        return tools.midicents(self.midi)

    @property
    def centsrepr(self) -> str:
        return tools.centsshown(self.cents)

    def scoringEvents(self) -> t.List[scoring.Note]:
        db = None if self.amp is None else amp2db(self.amp)
        dur = self.dur if self.dur is not None else config['defaultDuration']
        note = scoring.Note(self.midi, db=db, dur=dur, offset=self.start)
        if self.label:
            note.addAnnotation(self.label)
        return [note]

    def _asmusic21(self, showcents=None) -> m21.stream.Stream:
        if self.midi == 0:
            return m21.note.Rest()
        divs = config['show.semitoneDivisions']
        basepitch = float(self.roundPitch(divs))
        dur = self.dur if self.dur is not None else config['defaultDuration']
        note = m21funcs.m21Note(basepitch, quarterLength=dur)
        cents = int((self.midi - basepitch) * 100)
        note.microtone = cents
        showcents = showcents if showcents is not None else config['show.cents']
        if showcents:
            note.lyric = self.centsrepr
        part = m21.stream.Part()
        part.append(_bestClef([self]))
        part.append(note)
        return part

    def _asTableRow(self) -> t.List[str]:
        elements = [m2n(self.midi)]
        if config['repr.showfreq']:
            elements.append("%dHz" % int(self.freq))
        if self.amp < 1:
            elements.append("%ddB" % round(amp2db(self.amp)))
        return elements

    def __repr__(self) -> str:
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
        # type: (t.U[Note, float, int]) -> t.U[Note, Chord]
        if isinstance(other, (int, float)):
            return Note(self.midi + other, self.amp)
        elif isinstance(other, Note):
            return Chord([self, other])
        elif isinstance(other, str):
            return self + asNote(other)
        raise TypeError(f"can't add {other} ({other.__class__}) to a Note")

    def __xor__(self, freq):
        return self.shift(freq)

    def __sub__(self, other):
        # type: (t.U[Note, float, int]) -> Note
        if isinstance(other, Note):
            raise TypeError("can't substract one note from another")
        elif isinstance(other, (int, float)):
            return Note(self.midi - other, self.amp)
        raise TypeError(f"can't substract {other} ({other.__class__}) from a Note")

    def quantize(self, step=1.0) -> 'Note':
        """
        Returns a new Note, rounded to step
        """
        return self.clone(pitch=round(self.midi / step) * step)

    def copy(self) -> 'Note':
        return Note(self.midi, self.amp)

    def csoundEvents(self, delay, dur, chan, gain=1, fade=0) -> t.List[_CsoundLine]:
        amp = self.amp*gain
        midi = self.midi
        assert delay is not None
        assert dur is not None
        timefactor = getState().timefactor
        dur *= timefactor
        delay *= timefactor
        event = _makeEvent(delay, dur, chan, amp=amp, midi=midi, endamp=amp, endmidi=midi, fade=fade)
        return [event]
        
    def gliss(self, dur, endpitch, endamp=None, start=None) -> 'Event':
        return Event(pitch=self.midi, amp=self.amp, dur=dur, endpitch=endpitch, endamp=endamp, start=start)

    def asEvent(self, dur=None, start=None, endpitch=None, endamp=None) -> 'Event':
        endamp = endamp if endamp is not None else self.amp
        endpitch = endpitch if endpitch is not None else self.midi
        dur = dur or self.dur or config['defaultDuration']
        start = start or self.start or 0.0
        return Event(pitch=self.midi, amp=self.amp, dur=dur, endpitch=endpitch, endamp=endamp, start=start)


def Rest(dur:Fraction=1, start:Fraction=None) -> Note:
    return Note(pitch=0, dur=dur, start=start)


def asNote(n, amp:float=None, dur:Fraction=None) -> 'Note':
    """
    n: str    -> notename
       number -> midinote
    amp: 0-1

    you can also create a note by doing asNote((pitch, amp))
    """
    if isinstance(n, Note):
        out = n.fill(amp=amp, dur=dur)
    elif isinstance(n, (int, float)):
        out = Note(n, amp=amp)
    elif isinstance(n, str):
        midi = str2midi(n)
        out = Note(midi, amp=amp)
    elif isinstance(n, tuple) and len(n) == 2 and amp is None:
        out = asNote(*n)
    else:
        raise ValueError(f"cannot express this as a Note: {n} ({type(n)})")
    return out


def _makeEvent(delay, dur, chan, amp, midi, endamp=None, endmidi=None, fade=None) -> _CsoundLine:
    """
    fade: a single value, or a tuple (fadein, fadeout)
    """
    endamp = endamp if endamp is not None else amp
    endmidi = endmidi if endmidi is not None else midi
    fade = fade if fade is not None else config['play.fade']
    fadein, fadeout = tools.normalizeFade(fade)
    bps = [(0, midi, amp), (dur, endmidi, endamp)]
    return _CsoundLine(bps, delay=delay, chan=chan, fadein=fadein, fadeout=fadeout)


class Line(_Base):
    """ 
    A Line is a seq. of breakpoints, where each bp is of the form
    (delay, pitch, amp=1 [, ...])

    delay: the time offset to the first breakpoint. 
    pitch: the pitch as midinote or notename
    amp:   the amplitude (0-1), optional

    pitch, amp and any other following data can be 'carried'

    Line((0, "D4"), (1, "D5", 0.5), ..., fade=0.5)

    also possible:
    bps = [(0, "D4"), (1, "D5"), ...]
    Line(bps)   # without *

    a Line stores its breakpoints as
    [delayFromFirstBreakpoint, pitch, amp, ...]
    """

    def __init__(self, *bps, fade:float=None, label="", delay:_Num=0, interpol='linear', reltime=False):
        """
        bps: a tuple of the form (delay, pitch, [amp=1, ...])

            delay: the time offset to the beginning of the line
            pitch: the pitch as notename or midinote
            amp: a 0-1 amplitude

            In order to specify delay as delay from previous note, use reltime=True

        fade: a fade time, used for playback
        delay: time offset of the whole line
        interpol: pitch interpolation curve ('linear', 'halfcos') | Currently NOT used
        """
        if len(bps) == 1 and isinstance(bps[0], list):
            bps = bps[0]
        bps = tools.fillColumns(bps)
        if len(bps[0]) < 2:
            raise ValueError(f"A breakpoint should be at least (delay, pitch), got {bps}")
        if len(bps[0]) < 3:
            bps = tools.addColumn(bps, 1)
        # bps = tools.transformBreakpoints(bps)
        bps = [(bp[0], tools.asmidi(bp[1]))+_lib.astype(tuple, bp[2:])
               for bp in bps]
        if reltime:
            relbps = bps
            now = 0
            bps = []
            for _delay, *rest in relbps:
                now += _delay
                bp = (now, *rest)
                bps.append(bp)

        assert all(bp1[0] > bp0[0] for bp0, bp1 in _iterlib.pairwise(bps))
        super().__init__(dur=bps[-1][0], start=delay, label=label)
        self.bps = bps
        self.fade = fade if fade is not None else config['play.fade']
        self.interpol = interpol

    def getOffsets(self) -> t.List[_Num]:
        """
        Return absolute offsets of each breakpoint
        """
        start = self.start
        return [bp[0] + start for bp in self.bps]

    def csoundEvents(self, delay, dur, chan, gain, fade=None) -> t.List[_CsoundLine]:
        """
        delay: override own delay (use 0)
        dur: makes no sense, not used 
        gain: final gain = self.amp * gain
        """
        fade = fade if fade is not None else self.fade if self.fade is not None else config['play.fade']
        fadein, fadeout = tools.normalizeFade(fade)
        delay = delay + self.start
        timefactor = getState().timefactor
        delay *= timefactor

        # bp = (delay, pitch, amp, ...)
        if gain == 1 and timefactor == 1:
            bps = self.bps
        else:
            bps = [(bp[0] * timefactor, bp[1], bp[2]*gain) + bp[3:] for bp in self.bps]

        line = _CsoundLine(bps=bps, delay=delay, chan=chan, fadein=fadein, fadeout=fadeout)
        return [line]

    def __hash__(self):
        return hash((self.start, self.fade, *_iterlib.flatten(self.bps)))
        
    def __repr__(self):
        return f"Line(delay={self.start}, bps={self.bps})"

    def scoringEvents(self):
        notes = []
        offsets = self.getOffsets()
        group = scoring.makeId()
        for (bp0, bp1), offset in zip(_iterlib.pairwise(self.bps), offsets):
            dur = bp1[0]-bp0[0]
            gliss = bp0[1] != bp1[1]
            ev = scoring.Note(pitch=bp0[1], offset=offset, dur=dur, gliss=gliss, group=group)
            notes.append(ev)
        # notes[-1].endpitch = self.bps[-1][1]
        if(self.bps[-1][1] != self.bps[-2][1]):
            # add a last note if the last pair needed a glissando, to have a destination
            # point for it
            notes.append(scoring.Note(pitch=self.bps[-1][1], offset=offsets[-1], group=group,
                                      dur=asTime(config['show.lastBreakpointDur'])))
        return notes

    def _asmusic21(self, **kws) -> m21.Music21Object:
        import emlib.music.scoring.quantization as quant
        events = self.scoringEvents()
        divsPerSemitone = config['show.semitoneDivisions']
        showcents = config['show.cents']
        showgliss = config['show.gliss']
        return quant.quantizeVoice(events, divsPerSemitone=divsPerSemitone,
                                   showcents=showcents, showgliss=showgliss)

    def dump(self):
        elems = []
        if self.start:
            elems.append(f"delay={self.start}")
        if self.fade:
            elems.append(f"fade={self.fade}")
        if self.label:
            elems.append(f"label={self.label}")
        infostr = ", ".join(elems)
        header = f"Line: {infostr}"
        print(header)
        durs = [bp1[0] - bp0[0] for bp0, bp1 in _iterlib.pairwise(self.bps)]
        durs.append(0)
        rows = [(offset, offset+dur, dur) + bp
                for offset, dur, bp in zip(self.getOffsets(), durs, self.bps)]
        headers = ("start", "end", "dur", "offset", "pitch", "amp", "p4", "p5", "p6", "p7", "p8")
        _lib.print_table(rows, headers=headers)


class Event(Note):

    """
    An Event is a Note with duration. It can have endpitch, endamp and
    a start offset.
    """

    def __init__(self, pitch:_Pitch, dur:_Num, endpitch:_Pitch=None,
                 amp:float=None, endamp:float=None, start:_Num=0, label=""):
        """
        An event is a note with a duration. It can have a start time
        """
        if isinstance(pitch, Note):
            note = pitch
            pitch = note.midi
            amp = amp if amp is not None else note.amp
        super().__init__(pitch, amp=amp, dur=dur, start=start, label=label)
        self.endmidi = tools.asmidi(endpitch) if endpitch is not None else self.midi
        self.endamp = endamp if endamp is not None else self.amp

    def __repr__(self):
        # type: () -> str
        ampstr = " %ddB" % int(round(amp2db(self.amp))) if self.amp != 1 else ""
        label = " [%s]" % self.label if self.label else ""
        start = float(self.start) if self.start is not None else 0. 
        end = float(self.end)
        pitch = m2n(self.midi).ljust(7)
        if self.endmidi == self.midi:
            out = f"<{pitch}{start:.3f}:{end:.3f}{ampstr}{label}>"
        else:
            out = f"<{m2n(self.midi)}/{m2n(self.endmidi)} {start:.3f}:{end:.3f}{ampstr}{label}>"
        return out

    def asLine(self):
        bps = [
            (self.start, self.midi, self.amp),
            (self.start+self.dur, self.endmidi, self.endamp)
        ]
        return Line(bps)
    
    def scoringEvents(self):
        return [scoring.Note(self.midi, dur=self.dur, offset=self.start, endpitch=self.endmidi)]

    def asmusic21(self) -> m21.stream.Stream:
        return _Base.asmusic21(self)

    def csoundEvents(self, delay, dur, chan, gain, fade=0) -> t.List[_CsoundLine]:
        # delay has no effect
        start = self.start if self.start is not None else 0
        timefactor = getState().timefactor
        dur *= timefactor
        start *= timefactor
        event = _makeEvent(delay=start, dur=dur, chan=chan,
                           amp=self.amp*gain, midi=self.midi,
                           endamp=self.endamp*gain, endmidi=self.endmidi,
                           fade=fade)
        return [event]

    def __hash__(self):
        return hash((self.midi, self.endmidi, self.dur, self.start))




def concatEvents(events: t.List[Event]) -> 'EventSeq':
    """
    Concatenate a series of events, disregarding the start offset
    """
    t0 = F(0)
    newevents: t.List[Event] = []
    for event in events:
        t0 = max(event.start, t0)
        newevent = event.clone(start=t0)
        newevents.append(newevent)
        t0 = newevent.end
    return EventSeq(newevents)


# Line(0, "C4", 2, "E4", 3, "D4")


class NoteSeq(_Base, list):
    """
    A seq. of Notes
    """

    def __init__(self, *notes: t.Seq[Note], dur=None) -> None:
        self._hash = None
        super().__init__()
        if notes:
            if len(notes) == 1:
                n0 = notes[0]
                if _lib.isiterable(n0):
                    notes = n0
                elif isinstance(n0, str) and " " in n0:
                    notes = n0.split()
            self.extend(map(asNote, notes))
        self._noteDur = dur

    def __getitem__(self, *args):
        out = list.__getitem__(self, *args)
        if isinstance(out, list):
            out = self.__class__(out)
        return out    

    def _changed(self):
        super()._changed()
        self._hash = None

    def asEventSeq(self, dur=None):
        defaultdur = dur or self.dur or F(1)
        events = []
        now = 0
        for i, note in enumerate(self):
            dur = note.dur if note.dur > 0 else defaultdur
            ev = Event(pitch=note.midi, amp=note.amp, start=now, dur=dur)
            now += dur
            events.append(ev)
        return EventSeq(events)
        
    def asChord(self):
        return Chord(self)

    def _asmusic21(self):
        dur = self.dur or F(1, 2)
        showcents = config['show.cents']
        split = config['show.split']
        midinotes = [note.midi for note in self]
        if not split or not _midinotesNeedSplit(midinotes):
            part = m21.stream.Part()
            for note in self:
                m21note, centsdev = m21tools.makeNote(note.midi, showcents=showcents, quarterLength=dur)
                part.append(m21note)
            return part
        above, below = m21.stream.Part(), m21.stream.Part()
        splitpoint = 60.0
        for note in self:
            m21note, rest = m21funcs.m21Note(note.midi, quarterLength=dur), m21.note.Rest(quarterLength=dur)
            if note.midi<splitpoint:
                above.append(rest)
                below.append(m21note)
            else:
                above.append(m21note)
                below.append(rest)
        return m21tools.stackParts((above, below))

    def __repr__(self):
        notestr = ", ".join(n.name for n in self)
        return f"NoteSeq({notestr})"
        
    def __hash__(self):
        if self._hash is not None:
            return self._hash
        data = [note.midi for note in self]
        data.append(self._noteDur)
        if self.dur:
            self._hash = hash(tuple(hash(note) * self.dur for note in self))
        else:
            self._hash = hash(tuple(hash(note) ^ 0xF123 for note in self))
        return self._hash

    def scoringEvents(self) -> t.List[scoring.Event]:
        defaultdur = self._noteDur or F(1, 2)
        now = F(0)
        evs = []
        group = scoring.makeId()
        for note in self:
            ev = note.scoringEvents()[0]
            ev.dur = ev.dur or defaultdur
            ev.offset = now
            ev.group = group
            now += ev.dur
            evs.append(ev)
        return evs

    def csoundEvents(self, *args, **kws) -> t.List[_CsoundLine]:
        dur = kws.get('dur', self._noteDur)
        return self.asEventSeq(dur=dur).csoundEvents(*args, **kws)

    def __mul__(self, other):
        return self.__class__(list(self).__mul__(other))


def _sequentializeEvents(events: t.Seq[Event], start=None) -> t.List[Event]:
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

    def __init__(self, events:t.Seq[Event]=None) -> None:
        if events:
            if all(not ev.start for ev in events):
                events = _sequentializeEvents(events)
            events = sorted(events, key=lambda ev:ev.start)
            start = events[0].start
            end = events[-1].end
            dur = end - start
            super().__init__(dur=dur, start=start)
            self.extend(events)
        else:
            super().__init__()
        self._hash = None

    def _changed(self) -> None:
        super()._changed()
        self.sort(key=lambda ev: ev.start)
        self._hash = None
        if len(self):
            self.start = self[0].start
            self.dur = self[-1].end - self.start

    def extend(self, events):
        list.extend(self, events)
        self._changed()

    def isEmptyBetween(self, start, end):
        return not any(_lib.intersection(start, end, ev.start, ev.end) for ev in self)

    def append(self, event: Event) -> None:
        if event.start is None:
            event = event.clone(start=self.end)
        else:
            assert self.isEmptyBetween(event.start, event.end)
        list.append(self, event)
        self._changed()

    def removeOverlap(self) -> 'EventSeq':
        events = EventSeq()
        for ev0, ev1 in _iterlib.pairwise(self):
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

    def shiftTime(self, start:_Num) -> 'EventSeq':
        """
        Shift this seq. to start at `start`
        """
        dt = start - self.start
        return EventSeq([ev.clone(start=ev.start+dt) for ev in self])

    def hasOverlap(self) -> bool:
        return any(ev1.start - ev0.end > 0 for ev0, ev1 in _iterlib.pairwise(self))

    def scoringEvents(self):
        events = [event.scoringEvents()[0] for event in self]
        return events

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

    def csoundEvents(self, *args, **kws) -> t.List[_CsoundLine]:
        csdEvents = []
        for event in self:
            csdEvents.extend(event.csoundEvents(*args, **kws))
        return csdEvents


def N(pitch, dur:Fraction=None, start=None, endpitch=None):
    if isinstance(pitch, (tuple, list)):
        return Chord(pitch, dur=dur, start=start, endpitches=endpitch)
    elif isinstance(pitch, str):
        if " " in pitch:
            return Chord(pitch, dur=dur, start=start, endpitches=endpitch)
        else:
            return Note(pitch, dur=dur, start=start, endpitch=endpitch)
    else:
        return Note(pitch, dur=dur, start=start, endpitch=endpitch)


class Chord(_Base, list):

    def __init__(self, *notes:t.Any, amp:float=None, dur:Fraction=None,
                 start=None, endpitches=None, label:str=None) -> None:
        """
        a Chord can be instantiated as:

            Chord(note1, note2, ...) or
            Chord([note1, note2, ...])
            Chord("C4 E4 G4")

        where each note is either a Note, a notename ("C4", "E4+", etc), a midinote
        or a tuple (midinote, amp)

        label: str. If given, it will be used for printing purposes, if possible
        """
        _Base.__init__(self, label=label)
        self.amp = amp
        if dur is not None:
            dur = F(dur)
        if notes:
            # notes might be: Chord([n1, n2, ...]) or Chord(n1, n2, ...)
            if _lib.isgenerator(notes):
                notes = list(notes)
            if isinstance(notes[0], (list, tuple)):
                assert len(notes) == 1
                notes = notes[0]
            elif isinstance(notes[0], str) and len(notes) == 1:
                notes = notes[0].split()
            # determine dur
            if dur is None and any(isinstance(note, Note) and note.dur is not None 
                                   for note in notes):
                dur = max(note.dur if note.dur is not None else 0 for note in notes)
            notes2 = []
            for note in notes:
                if isinstance(note, Note):
                    if note.dur is not None and dur is not None:
                        note = note.clone(dur=dur, amp=amp)
                else:
                    note = asNote(note, amp=amp, dur=dur)
                notes2.append(note)
            self.extend(notes2)
            self.sortbypitch(inplace=True)
            
        self.endchord = asChord(endpitches) if endpitches else None
        self.dur = dur
        self.start = start
        self._hash = None

    @property
    def notes(self) -> t.List[Note]:
        return list(self)

    def scoringEvents(self) -> t.List[scoring.Event]:
        pitches = [note.midi for note in self]
        db = None if self.amp is None else amp2db(self.amp)
        annot = None if self.label is None else self.label
        endpitches = None if not self.endchord else [note.midi for note in self.endchord]
        return [scoring.Chord(pitches, db=db, annot=annot, endpitches=endpitches,
                              dur=self.dur, offset=self.start)]

    def asmusic21(self, arpeggio=None) -> m21.stream.Stream:
        arpeggio = _normalizeChordArpeggio(arpeggio, self)
        if arpeggio:
            dur = config['show.arpeggioDuration']
            return NoteSeq(self.notes, dur=dur).asmusic21()
        else:
            showcents = config['show.cents']
            split = config['show.split']
            return self._splitChord(split=split, showcents=showcents)

    def __hash__(self):
        if self._hash:
            return self._hash
        data = (self.dur, self.start) + tuple(n.midi for n in self)
        if self.endchord:
            data = data + tuple(n.midi for n in self.endchord)
        self._hash = h = hash(data)
        return h

    @property
    def freqs(self):
        return [n.freq for n in self]

    def _changed(self):
        self._hash = None

    def append(self, note):
        # type: (t.U[Note, float, str]) -> None
        self._changed()
        note = asNote(note)
        if note.freq < 17:
            _logger.debug(f"appending a note with very low freq: {note.freq}")
        list.append(self, note)

    def extend(self, notes) -> None:
        for note in notes:
            super().append(asNote(note))
        self._changed()

    def insert(self, index:int, note) -> None:
        self._changed()
        note = asNote(note)
        super(self.__class__, self).insert(index, note)

    def filter(self, func):
        """
        Example: filter out notes lower than the lowest note of the piano

        return ch.filter(lambda n: n > "A0")
        """
        return Chord([n for n in self if func(n)])
        
    def transpose(self, step:float) -> 'Chord':
        """
        Return a copy of self, transposed `step` steps
        """
        return Chord([note.transpose(step) for note in self])

    def shift(self, freq:float) -> 'Chord':
        return Chord([note.shift(freq) for note in self])

    def roundPitch(self, semitoneDivisions:int=0) -> float:
        divs = semitoneDivisions or config['show.semitoneDivisions']
        notes=[note.roundPitch(divs) for note in self]
        return self._withNewNotes(notes)
    
    def _withNewNotes(self, notes):
        return Chord(notes, start=self.start, dur=self.dur, amp=self.amp)

    def quantize(self, step=1.0) -> 'Chord':
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
        return self._withNewNotes(notes)

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
        elif isinstance(other, (Chord, str)):
            return Chord(list.__add__(self, asChord(other)))
        raise TypeError("Can't add a Chord to a %s" % other.__class__.__name__)

    def splitbyamp(self, numchords=8, max_notes_per_chord=16):
        # type: (int, int) -> t.List[Chord]
        midinotes = [note.midi for note in self]
        amps = [note.amp for note in self]
        return _splitByAmp(midinotes, amps, numchords, max_notes_per_chord)

    def sortbyamp(self, reverse=True, inplace=True) -> 'Chord':
        if inplace:
            out = self
        else:
            out = self.__class__(self)
        out.sort(key=lambda n: n.amp, reverse=reverse)
        return out

    def loudest(self, n:int) -> 'Chord':
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        return self.sortbyamp(inplace=False, reverse=True)[:n]

    def sortbypitch(self, reverse=False, inplace=True) -> 'Chord':
        if inplace:
            out = self
        else:
            out = self.__class__(self)
        out.sort(key=lambda n: n.midi, reverse=reverse)
        return out

    def csoundEvents(self, delay, dur, chan, gain, fade=0) -> t.List[_CsoundLine]:
        adjustgain = config['chord.adjustGain']
        if adjustgain:
            gain *= 1/sqrt(len(self))
            _logger.debug(f"playCsound: adjusting gain by {gain}")
        events = []
        # dur = dur or self.dur or config['play.dur']
        # delay = delay if delay is not None else self.start if self.start is not None else 0
        timefactor = getState().timefactor
        dur *= timefactor
        delay *= timefactor
        
        if self.endchord is None:
            for note in self:
                amp = note.amp*gain
                event = _makeEvent(delay=delay, dur=dur, chan=chan, amp=amp, midi=note.midi, fade=fade)
                events.append(event)
        else:
            for note0, note1 in zip(self.notes, self.endchord):
                ev = _makeEvent(delay=delay, dur=dur, chan=chan, 
                                amp=note0.amp*gain, endamp=note1.amp*gain,
                                midi=note0.midi, endmidi=note1.midi,
                                fade=fade)
                events.append(ev)
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
        Return a new Chord with the amps of the notes modified according to curve
        
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
        # type: (t.Func[[float], float]) -> Chord
        """
        Return a new Chord equalized by curve

        curve: a func(freq) -> gain
        """
        notes = []
        for note in self:
            gain = curve(note.freq)
            notes.append(note.clone(amp=note.amp*gain))
        return self.__class__(notes)

    def gliss(self, dur:float, endnotes, start=None) -> 'Chord':
        """
        Create a glissando between this chord and the endnotes given

        dur: the dur of the glissando
        endnotes: the end of the gliss, as Chord, list of Notes or string

        Example: semitone glissando in 2 seconds

        ch = Chord("C4", "E4", "G4")
        ch2 = ch.gliss(2, ch.transpose(-1))

        Example: gliss with diminuendo

        Chord("C4 E4", amp=0.5).gliss(5, Chord("E4 G4", amp=0).play()
        """
        endchord = asChord(endnotes)
        if len(endchord) != len(self):
            raise ValueError(f"The number of end notes {len(endnotes)} != the"
                             f"size of this chord {len(self)}")
        events = []
        startpitches = [note.midi for note in self]
        endpitches = [note.midi for note in endchord]
        assert len(startpitches) == len(endpitches)
        out = Chord(*startpitches, amp=self.amp, label=self.label, endpitches=endpitches)
        out.dur = asTime(dur)
        out.start = asTime(start) if start is not None else None
        return out

    def difftones(self):
        """
        Return a Chord representing the difftones between the notes of this chord
        """
        from emlib.music.combtones import difftones
        return Chord(difftones(*self))

    def isCrowded(self: 'Chord') -> bool:
        return any(abs(n0.midi-n1.midi)<=1 and abs(n1.midi-n2.midi)<=1
                   for n0, n1, n2 in _iterlib.window(self, 3))

    def _splitChord(self: 'Chord', split=60.0, showcents=None, showlabel=True) -> m21.stream.Score:
        split = 60 if isinstance(split, bool) else split
        showcents = showcents if showcents is not None else config['show.cents']
        parts = _splitNotesIfNecessary(self, float(split))
        score = m21.stream.Score()
        for notes in parts:
            midinotes = [n.midi for n in notes]
            m21chord = m21funcs.m21Chord(midinotes, showcents=showcents)
            part = m21.stream.Part()
            part.append(_bestClef(notes))
            if showlabel and self.label:
                part.append(m21funcs.m21Label(self.label))
                showlabel = False
            part.append(m21chord)
            if config['show.centsMethod'] == 'expression':
                m21tools.makeExpressionsFromLyrics(part)
            score.insert(0, part)
        return score


_CanBeChord = t.U[Chord, t.Seq[_Pitch], str]


def asChord(obj:_CanBeChord, amp=None, dur=None) -> 'Chord':
    if isinstance(obj, Chord):
        out = obj
    elif isinstance(obj, (list, tuple, str)):
        out = Chord(obj)
    elif hasattr(obj, "asChord"):
        out = obj.asChord()
        assert isinstance(out, Chord)
    elif isinstance(obj, (int, float)):
        out = Chord(asNote(obj))
    else:
        raise ValueError(f"cannot express this as a Chord: {obj}")
    if amp is not None or dur is not None:
        out = out.clone(amp=amp, dur=dur)
    return out
    

def _normalizeChordArpeggio(arpeggio: t.U[str, bool], chord: Chord) -> bool:
    if arpeggio is None:
        arpeggio = config['chord.arpeggio']

    if isinstance(arpeggio, bool):
        return arpeggio
    elif arpeggio is None:
        return config['show.arpeggio']
    elif arpeggio == 'auto':
        return chord.isCrowded()
    else:
        raise ValueError(f"arpeggio should be True, False, 'auto' (got {arpeggio})")


class ChordSeq(_Base, list):
    """
    A seq. of Chords
    """

    def __init__(self, *chords, dur=None):
        self._hash = None
        super().__init__()
        if len(chords) == 1 and isinstance(chords[0], (list, tuple)):
            chords = chords[0]
        if chords:
            chords = [asChord(chord) for chord in chords]
            default_dur = dur or config['defaultDuration']
            outchords = []
            for chord in chords:
                if chord.dur is None:
                    chord = chord.clone(dur=default_dur)
                self.append(chord)
        self.dur = dur

    def __getitem__(self, *args):
        out = list.__getitem__(self, *args)
        if not out or isinstance(out, Chord):
            return out
        elif isinstance(out, list) and isinstance(out[0], Chord):
            return self.__class__(out)
        else:
            raise ValueError("__getitem__ returned {out}, expected Chord or list of Chords")

    def _changed(self):
        super()._changed()
        self._hash = None

    def asChord(self):
        """Join all the individual chords into one chord"""
        return Chord(list(set(_iterlib.flatten(self))))

    def scoringEvents(self) -> t.List[scoring.Event]:
        events = []
        defaultDur = config['show.seqDuration']
        for chord in self:
            ev = chord.scoringEvents()[0]
            ev.dur = self.dur if self.dur is not None else defaultDur
            events.append(ev)
        return events

    def asmusic21(self, split=None, showcents=None, dur=None):
        showcents = showcents if showcents is not None else config['show.cents']
        dur = dur or self.dur or config['show.seqDuration']
        return splitChords(self, split=split, showcents=showcents, dur=dur)

    def __repr__(self):
        lines = ["ChordSeq "]
        lines.extend("   "+" ".join(n.name.ljust(6) for n in ch) for ch in self)
        return "\n".join(lines)

    def __hash__(self):
        if self._hash is not None:
            return self._hash
        self._hash = hash(tuple(hash(chord) ^ 0x1234 for chord in self))
        return self._hash

    def csoundEvents(self, delay, dur, chan, gain, fade=0) -> t.List[_CsoundLine]:
        allevents = []
        now = delay or 0.0
        timefactor = getState().timefactor
        delay *= timefactor
        default_dur = dur
        for i, chord in enumerate(self):
            chord_dur = chord.dur
            dur = chord_dur if chord_dur is not None and chord_dur > 0 else default_dur
            assert dur > 0
            allevents.extend(chord.csoundEvents(delay=now, dur=dur, 
                                                chan=chan, gain=gain, fade=fade))
            now += dur
        return allevents

    def cycle(self, dur):
        """
        Cycle the chords in this seq. until the given duration is reached 
        """
        out = ChordSeq()
        defaultDur = config['show.seqDuration']
        chordstream = _iterlib.cycle(self)
        totaldur = 0
        while totaldur < dur:
            chord = next(chordstream)
            if chord.dur is None:
                chord = chord.clone(dur=defaultDur)
            totaldur += chord.dur
            out.append(chord)
        return out


class Track(_Base):
    """
    A Track is a seq. of non-overlapping objects
    """
    def __init__(self, objs=None):
        self.timeline: t.List[_Base] = []
        self.instrs: t.Dict[_Base, str] = {}
        super().__init__(dur=0, start=0)
        if objs:
            for obj in objs:
                self.add(obj)

    def __iter__(self):
        return iter(self.timeline)

    def __getitem__(self, idx):
        return self.timeline.__getitem__(idx)

    def __len__(self):
        return len(self.timeline)

    def _changed(self):
        if self.timeline:
            self.dur = self.timeline[-1].end - self.timeline[0].start

    def endTime(self) -> Fraction:
        if not self.timeline:
            return Fraction(0)
        return self.timeline[-1].end

    def isEmptyBetween(self, t0, t1):
        if not self.timeline:
            return True
        if t0 >= self.timeline[-1].end:
            return True
        if t1 < self.timeline[0].start:
            return True
        for item in self.timeline:
            if _lib.intersection(item.start, item.end, t0, t1):
                return False
        return True

    def needsSplit(self):
        pass

    def add(self, obj:_Base, at=None, instr:str=None) -> None:
        """
        Add this object to this Track. If obj has already a given start,
        it will be inserted at that offset, otherwise it will be appended
        to the end of this Track. 

        1) To insert an untimed object (a Note, a Chord) to the Track at a 
           given offset, set its .start attribute, do track.add(chord.clone(start=...))
           or use the at param

        2) To append a timed object at the end of this track (overriding the start
           time of the object), do track.add(obj, at=track.endTime())

        **NB**: if the object already has timing information, the `at` parameter,
                if given, overrides this information.

        :param obj: the object to add (a Note, Chord, Event, etc.)
        :param at: if given, overrides the start value of obj
        :param instr: if given, it will be used for playing
        """
        defaultDur = config['defaultDuration']
        if at is not None:
            obj = obj.clone(start=at, dur=obj.dur or defaultDur)
        elif obj.start is None or obj.dur is None:
            obj = _asTimedObj(obj, start=self.endTime(), dur=defaultDur)

        if not self.isEmptyBetween(obj.start, obj.end):
            msg = f"obj {obj} ({obj.start}:{obj.start+obj.dur}) does not fit in track"
            raise ValueError(msg)
        assert obj.start is not None and obj.start >= 0 and obj.dur is not None and obj.dur > 0
        self.timeline.append(obj)
        self.timeline.sort(key=lambda obj:obj.start)
        if instr is not None:
            self.instrs[obj] = instr
        self._changed()

    def extend(self, objs:t.List[_Base]) -> None:
        objs.sort(key=lambda obj:obj.start)
        assert objs[0].start >= self.endTime()
        for obj in objs:
            self.timeline.append(obj)
        self._changed()

    def scoringEvents(self) -> t.List[scoring.Event]:
        allevents = []
        for obj in self.timeline:
            events = obj.scoringEvents()
            allevents.extend(events)
        return allevents

    def csoundEvents(self, delay: float, dur: float, chan: int, gain: float, fade=0.0
                     ) -> t.List['_CsoundLine']:
        allevents: t.List[_CsoundLine] = []
        for obj in self.timeline:
            events = obj.getEvents(gain=gain, fade=fade, chan=chan)
            allevents.extend(events)
        return allevents

    def play(self, **kws) -> csoundengine.SynthGroup:
        """
        kws: any kws is passed directly to each individual event
        """
        synths = []
        defaultInstr = kws.pop('instr') if 'instr' in kws else None
        for obj in self.timeline:
            instr = _getValue(obj.playargs.get('instr'), self.instrs.get(obj), defaultInstr)
            synth = obj.play(instr=instr, **kws)
            synths.append(synth)
        return csoundengine.SynthGroup(synths)

    def scoringTrack(self) -> scoring.Track:
        return scoring.Track(self.scoringEvents())


class TrackList(_Base, list):
    def __init__(self, tracks=None):
        _Base.__init__(self)
        if tracks:
            for track in tracks:
                self.append(track)

    # TODO


def _asTimedObj(obj: _Base, start, dur) -> _Base:
    """
    A TimedObj has a start time and a duration
    """
    assert start is not None and dur is not None
    dur = obj.dur if obj.dur is not None else dur

    start = obj.start if obj.start is not None else start
    start = asTime(start)
    dur = asTime(dur)
    obj2 = obj.clone(dur=dur, start=start)
    assert obj2.dur is not None and obj2.start is not None
    return obj2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# notenames
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def generateNotes(start=12, end=127) -> t.Dict[str, Note]:
    """
    Generates all notes for interactive use.

    From an interactive session, 

    locals().update(generate_notes())
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
            enharmonic_note = tools.enharmonic(rest)
            enharmonic_note += str(octave)
            notes[enharmonic_note] = Note(i)
    return notes


def setJupyterHook() -> None:
    _Base._setJupyterHook()
    tools.m21JupyterHook()


def _bestClef(notes:t.Seq[Note]) -> m21.clef.Clef:
    if not notes:
        return m21.clef.TrebleClef()
    mean = sum(note.midi for note in notes) / len(notes)
    if mean > 90:
        return m21.clef.Treble8vaClef()
    elif mean > 59:
        return m21.clef.TrebleClef()
    else:
        return m21.clef.BassClef()


def _splitNotes(notes: t.Seq[Note], splitpoint:float) -> t.Tup[t.List[Note], t.List[Note]]:
    """
    Given a seq. of notes, splitpoint them above and below the given splitpoint point

    :param notes: a seq. of Notes
    :param splitpoint: the pitch to split the notes
    :return: notes above and below
    """
    above, below = [], []
    for note in notes:
        (above if note.midi>splitpoint else below).append(note)
    return above, below


def _splitNotesIfNecessary(notes:t.Seq[Note], splitpoint:float) -> t.List[t.List[Note]]:
    return [part for part in _splitNotes(notes, splitpoint) if part]


def _midinotesNeedSplit(midinotes, splitpoint=60) -> bool:
    numabove = sum(int(midinote > splitpoint) for midinote in midinotes)
    return not(numabove == 0 or numabove == len(midinotes))


def _getSplitPoint(split) -> float:
    split = split if split is not None else config['show.split']
    if isinstance(split, bool):
        split = 60.0 if split else 0.0
    return split
        

def splitChords(chords:t.Seq[Chord], split=60, showcents=None, showlabel=True, dur=1,
                ) -> m21.stream.Score:
    showcents = showcents if showcents is not None else config['show.cents']
    dur = dur if dur is not None else config['show.seqDuration']
    chordsup, chordsdown = [], []
    split = _getSplitPoint(split)
    for chord in chords:
        above, below = _splitNotes(chord, split)
        chordsup.append(above)
        chordsdown.append(below)

    def isRowEmpty(chords):
        return all(not chord for chord in chords)

    rows = [row for row in (chordsup, chordsdown) if not isRowEmpty(row)]

    columns = zip(*rows)
    labels = [chord.label for chord in chords]

    def makePart(row):
        allnotes = list(_iterlib.flatten(row))
        part = m21.stream.Part()
        part.append(_bestClef(allnotes))
        return part

    parts = [makePart(row) for row in rows]
    for column, label in zip(columns, labels):
        if showlabel and label:
            parts[0].append(m21funcs.m21Label(label))
        for chord, part in zip(column, parts):
            if chord:
                midinotes = [n.midi for n in chord]
                part.append(m21funcs.m21Chord(midinotes, showcents=showcents, quarterLength=dur))
            else:
                part.append(m21.note.Rest(quarterLength=dur))
    return m21.stream.Score(parts)    
                

def _makeImage(obj: _Base, outfile:str=None, fmt:str=None, fixstream=True, **options) -> str:
    """
    obj     : the object to make the image from (a Note, Chord, etc.)
    outfile : the path to be generated
    fmt     : format used. One of 'xml.png', 'lily.png' (no pdf)
    options : any argument passed to .asmusic21

    NB: we put it here in order to make it easier to cache images
    """
    m21obj = obj.asmusic21(**options)
    if fixstream and isinstance(m21obj, m21.stream.Stream):
        m21obj = m21fix.fixStream(m21obj, inPlace=True)
    fmt = fmt if fmt is not None else config['show.format'].split(".")[0]+".png"
    _logger.debug(f"makeImage: using format: {fmt}")
    method, fmt3 = fmt.split(".")
    if method == 'lily' and config['use_musicxml2ly']:
        if fmt3 not in ('png', 'pdf'):
            raise ValueError(f"fmt should be one of 'lily.png', 'lily.pdf' (got {fmt})")
        if outfile is None:
            outfile = _tempfile.mktemp(suffix="."+fmt3)
        path = m21tools.renderViaLily(m21obj, fmt=fmt3, outfile=outfile)
    else:
        tmpfile = m21obj.write(fmt)
        if outfile is not None:
            os.rename(tmpfile, outfile)
            path = outfile
        else:
            path = tmpfile
    return str(path)


@_functools.lru_cache(maxsize=1000)
def _makeImageCached(*args, **kws) -> str:
    """
    obj     : the object to make the image from (a Note, Chord, etc.)
    outfile : the path to be generated
    fmt     : format used. One of 'xml.png', 'lily.png' (no pdf)
    options : any argument passed to .asmusic21

    NB: we put it here in order to make it easier to cache images
    """
    return _makeImage(*args, **kws)


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    _makeImageCached.cache_clear()


def asMusic(obj) -> _Base:
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


def gliss(obj1, obj2, dur=1, start=None):
    m1 = asMusic(obj1)
    m2 = asMusic(obj2)
    return m1.gliss(dur, m2, start=start)


def playmany(objs, **kws) -> csoundengine.SynthGroup:
    """
    Play multiple objects with the same parameters

    """
    synths = [obj.play(**kws) for obj in objs]
    return csoundengine.SynthGroup(synths)


def trill(note1, note2, totaldur, notedur=None) -> ChordSeq:
    """
    note1, note2: the notes of the trill. NB: these can be also chords
    totaldur: the total duration of the trill
    notedur: the duration of each note, if no duration is given

    returns a ChordSeq of at least the given duration
    """
    note1 = asChord(note1)
    note2 = asChord(note2)
    note1 = note1.clone(dur=note1.dur or notedur or F(1, 8))
    note2 = note2.clone(dur=note2.dur or notedur or F(1, 8))
    seq = ChordSeq(note1, note2)
    return seq.cycle(totaldur)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if tools.insideJupyter:
    setJupyterHook()

del t