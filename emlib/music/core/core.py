from __future__ import annotations
import functools as _functools
from math import sqrt
import os
import copy as _copy
import tempfile as _tempfile

import music21 as m21
from emlib import lib
from emlib.lib import firstval
from emlib.music import m21tools
from emlib.music import m21fix
from emlib.pitchtools import amp2db, db2amp, m2n, m2f, f2m, r2i, str2midi
from emlib import iterlib
from emlib.music import scoring
from emlib.snd import csoundengine

from ._base import *
from .common import CsoundEvent, PlayArgs, astuple
from .config import config, logger
from .state import getState
from . import m21funcs
from . import play
from . import tools

from emlib.typehints import U, T, Opt, List, Iter, Seq


_playkeys = {'delay', 'dur', 'chan', 'gain', 'fade', 'instr', 'pitchinterpol',
             'fadeshape', 'tabargs', 'position'}


def _clone(obj, **kws):
    out = _copy.copy(obj)
    for k, v in kws.items():
        setattr(out, k, v)
    return out


class MusicObj:
    _showableInitialized = False

    __slots__ = ('dur', 'start', 'label', '_playargs', '_hash')

    def __init__(self, label=None, dur:time_t = None, start: time_t = None):
        # A label can be used to identify an object within a group of
        # objects
        self.label: Opt[str] = label

        # A MusicObj can have a duration. A duration can't be 0
        if dur is not None:
            assert dur > 0
        self.dur: Opt[Fraction] = F(dur) if dur else None

        # start specifies a time offset for this object
        self.start: Opt[Fraction] = F(start) if start is not None else None

        # _playargs are set via .setplay and serve the purpose of
        # attaching playing parameters (like position, instrument)
        # # to an object
        self._playargs: Opt[PlayArgs] = None

        # All MusicObjs should be hashable. For the cases where
        # calculating the hash is expensive, we cache that here
        self._hash: int = 0

    @property
    def playargs(self):
        p = self._playargs
        if not p:
            self._playargs = p = PlayArgs()
        return p

    def setplay(self:T, **kws) -> T:
        """
        Pre-set any of the play arguments
         
        Args:
            **kws: any argument passed to .play 

        Returns:
            self
            
        Example:
            
            # a piano note
            note = Note("C4+25", dur=0.5).setplay(instr="piano")
        """
        for k, v in kws.items():
            if k not in _playkeys:
                raise KeyError(f"key {k} not known. "
                               f"Possible keys are {_playkeys}")
            setattr(self.playargs, k, v)
        return self

    def clone(self:T, **kws) -> T:
        """
        Clone this object, changing parameters if needed

        Args:
            **kws: any keywords passed to the constructor

        Returns:
            a clone of this objects, with the given arguments 
            changed
            
        Example:
            
            a = Note("C4+", dur=1)
            b = a.clone(dur=0.5)
        """
        out = _copy.deepcopy(self)
        for key, value in kws.items():
            setattr(out, key, value)
        return out

    def copy(self:T) -> T:
        return _copy.deepcopy(self)

    def delayed(self:T, timeoffset:time_t) -> T:
        """
        Return a copy of this object with an added time offset

        Example: create a seq. of syncopations

        n = Note("A4", start=0.5, dur=0.5)
        seq = Track([n, n.delay(1), n.delay(2), n.delay(3)])

        This is the same as 

        seq = Track([n, n>>1, n>>2, n>>3])
        """
        start = self.start or F(0)
        return self.clone(start=timeoffset + start)

    def __rshift__(self:T, timeoffset:time_t) -> T:
        return self.delayed(timeoffset)

    def __lshift__(self:T, timeoffset:time_t) -> T:
        return self.delayed(-timeoffset)

    @property
    def end(self) -> Opt[Fraction]:
        if not self.dur:
            return None
        start = self.start if self.start is not None else 0
        return start + self.dur

    def quantize(self:T, step=1.0) -> T:
        """ Returns a new object, rounded to step """
        raise NotImplementedError()

    def transpose(self:T, step) -> T:
        """ Transpose self by `step` """
        raise NotImplementedError()

    def freqratio(self:T, ratio) -> T:
        """ Transpose this by a given freq. ratio. A ratio of 2 equals
        to transposing an octave higher. """
        return self.transpose(r2i(ratio))

    def show(self, external=None, fmt=None, **options) -> None:
        """
        Show this as notation.

        Args:
            external: force opening the image in an external image viewer,
                even when inside a jupyter notebook. Otherwise, show will
                display the image inline
            fmt: overrides the config setting 'show.format'
                One of 'xml.png', 'xml', 'xml.pdf', 'lily.png', 'lily.pdf'
            options: any argument passed to .asmusic21

        NB: to use the music21 show capabilities, use note.asmusic21().show(...) or
            m21show(note.asmusic21())
        """
        if external is None: external = config['show.external']
        png = self.makeImage(fmt=fmt, **options)
        tools.pngShow(png, external=external)
        
    def _changed(self) -> None:
        """
        This method is called whenever the object changes its representation
        (a note changes its pitch inplace, the duration is modified, etc)
        This invalidates, among other things, the image cache for this 
        object
        """
        self._hash = None
        
    def makeImage(self, **options) -> str:
        """
        Creates an image representation, returns the path to the image

        Args:
            options: any argument passed to .asmusic21
        """
        # In order to be able to cache the images we put this
        # functionality outside of the class and use lru_cache
        if config['show.cacheImages']:
            return _makeImageCached(self, **options)
        return _makeImage(self, **options)

    def ipythonImage(self):
        """
        Generate a jupyter image from this object, to be used
        within a jupyter notebook

        Returns:
            an IPython.core.display.Image

        """
        from IPython.core.display import Image
        return Image(self.makeImage(), embed=True)

    def scoringEvents(self) -> List[scoring.Event]:
        """
        Each class should be able to return its notated form as
        an intermediate representation in the form of scoring.Events.
        These can then be converted into concrete notation via
        musicxml or lilypond

        Returns:
            A list of scoring.Event which best represent this
            object as notation
        """
        raise NotImplementedError("Subclass should implement this")


    def music21objs(self) -> List[m21.Music21Object]:
        """ Converts this obj to its closest music21 representation """
        events = self.scoringEvents()
        return [event.asmusic21() for event in events]

    def asmusic21(self, **options) -> m21.stream.Stream:
        """
        This method is used within .show, to convert this object
        into music notation. When using the musicxml backend
        we first convert our object/s into music21 and
        use the music21 framework to generate an image

        Args:
            **options: not used here, but classes inheriting from
                this may want to add customization

        Returns:
            a music21 stream which best represent this object as
            notation.

        NB: the music21 representation should be final, not thought to
            be embedded into another stream. For embedding we use
            an abstract representation of scoring objects which can
            be queried via .scoringEvents
        """
        return tools.m21FromScoringEvents(self.scoringEvents())

    def musicxml(self) -> str:
        " Return the representation of this object as musicxml "
        m = self.asmusic21()
        if config['m21.fixstream']:
            m21fix.fixStream(m)
        return m21tools.getXml(m)

    @classmethod
    def _setJupyterHook(cls) -> None:
        """
        Sets the jupyter display hook for this class

        """
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

    def _events(self, playargs:PlayArgs) -> List[CsoundEvent]:
        """
        This should be overriden by each class to generate CsoundEvents

        Args:
            playargs: a PlayArgs, structure, filled with given values,
                own .playargs values and config defaults (in that order)

        Returns:
            a list of CsoundEvents
        """
        raise NotImplementedError("Subclass should implement this")

    def _getDelay(self, delay:time_t=None) -> time_t:
        """
        This is here only to document how delay is calculated

        Args:
            delay: a delay to override playargs['delay']

        Returns:
            the play delay of this object
        """
        return firstval(delay, self.playargs.delay, 0)+(self.start or 0.)

    def _fillPlayArgs(self,
                      delay:float = None,
                      dur:float = None,
                      chan:int = None,
                      gain:float = None,
                      fade=None,
                      instr:str = None,
                      pitchinterpol:str = None,
                      fadeshape:str = None,
                      args: dict[str, float] = None,
                      position: float = None
                      ) -> PlayArgs:
        """
        Fill playargs with given values and defaults.

        The priority chain is:
            given value as param, prefilled value (playargs), config/default value
        """
        playargs = self.playargs
        dur = firstval(dur, playargs.dur, self.dur, config['play.dur'])
        if dur < 0:
            dur = MAXDUR
        return PlayArgs(
            dur = dur,
            delay =firstval(delay, playargs.delay, 0)+(self.start or 0.),
            gain = gain or playargs.gain or config['play.gain'],
            instr = instr or playargs.instr or config['play.instr'],
            chan = chan or playargs.chan or config['play.chan'],
            fade = firstval(fade, playargs.fade, config['play.fade']),
            pitchinterpol = pitchinterpol or playargs.pitchinterpol or config['play.pitchInterpolation'],
            fadeshape = fadeshape or playargs.fadeshape or config['play.fadeShape'],
            args = args or playargs.args,
            position = firstval(position, playargs.position, 0)
        )

    def events(self, delay:float=None, dur:float=None, chan:int=None,
               gain:float=None, fade=None, instr:str=None,
               pitchinterpol:str=None, fadeshape:str=None,
               args: dict[str, float] = None,
               position: float = None
               ) -> List[CsoundEvent]:
        """
        An object always has a start time. It can be unset (None), which defaults to 0
        but can also mean unset for contexts where this is meaningful (a sequence of Notes,
        for example, where they are concatenated one after the other, the start time
        is the end of the previous Note)

        All these attributes here can be set previously via .playargs (or
        using .setplay)

        Params:

            delay: A delay, if defined, is added to the start time.
            dur: play duration
            chan: the chan to play (or rec) this object
            gain: gain modifies .amp
            fade: fadetime or (fadein, fadeout)
            instr: the name of the instrument
            pitchinterpol: 'linear' or 'cos'
            fadeshape: 'linear' or 'cos'
            position: the panning position (0=left, 1=right). The left channel
                is determined by chan

        Returns:
            A list of _CsoundLine

        """
        playargs = self._fillPlayArgs(delay=delay, dur=dur, chan=chan, gain=gain,
                                      fade=fade, instr=instr,
                                      pitchinterpol=pitchinterpol, fadeshape=fadeshape,
                                      args=args,
                                      position=position)
        events = self._events(playargs)
        return events

    def play(self, 
             dur: float = None, 
             gain: float = None, 
             delay: float = None, 
             instr: str = None, 
             chan: int = None, 
             pitchinterpol: str = None,
             fade: float = None,
             fadeshape: str = None,
             args: dict[str, float] = None,
             position: float = None) -> csoundengine.AbstrSynth:
        """
        Plays this object. Play is always asynchronous. 
        By default, .play schedules this event to be renderer in realtime.
        
        NB: to record multiple events offline, see the example below

        Args:
            dur: the duration of the event
            gain: modifies the own amplitude for playback/recording (0-1)
            delay: delay in secons to start playback. 0=start immediately
            instr: which instrument to use (see defInstrPreset, availableInstrPresets)
            chan: the channel to output to (an int starting with 1).
            pitchinterpol: 'linear', 'cos', 'freqlinear', 'freqcos'
            fade: fade duration (can be a tuple (fadein, fadeout)
            fadeshape: 'linear' | 'cos'
            args: paramaters passed to the note through an associated table.
                A dict paramName: value
            position: the panning position (0=left, 1=right)

        Returns:
            A SynthGroup

        Example:
            # play a note
            note = Note(60).play(gain=0.1, chan=2)

            # record offline
            with play.rendering(sr=44100, outfile="out.wav"):
                Note(60).play(gain=0.1, chan=2)
                ... other objects.play(...)
        """
        events = self.events(delay=delay, dur=dur, chan=chan,
                             fade=fade, gain=gain, instr=instr,
                             pitchinterpol=pitchinterpol, fadeshape=fadeshape,
                             args=args, position=position)
        renderer: play._OfflineRenderer = getState().renderer
        if renderer is None:
            return playEvents(events)

        else:
            return renderer.schedMany(events)

    def rec(self, outfile:str=None, **kws) -> str:
        """
        Record the output of .play as a soundfile

        Args:
            outfile: the outfile where sound will be recorded. Can be
                None, in which case a filename will be generated
            **kws: any keyword passed to .play

        Returns:
            the path of the generated soundfile
        """
        events = self.events(**kws)
        return play.recEvents(events, outfile)



@_functools.total_ordering
class Note(MusicObj):

    __slots__ = ('midi', 'amp', 'endmidi')

    def __init__(self,
                 pitch: U[float, str],
                 amp:float=None,
                 dur:Fraction=None,
                 start:Fraction=None,
                 endpitch: U[float, str]=None,
                 label:str=None):
        """
        In its simple form, a Note is used to represent a Pitch.
        
        A Note must have a pitch. It is possible to specify
        an amplitude, a duration, a time offset (.start), 
        and an endpitch, resulting in a glissando
        
        Args:
            pitch: a midinote or a note as a string
            amp: amplitude 0-1 (optional)
            dur: the duration of this note (optional)
            start: start fot the note (optional)
            endpitch: if given, defines a glissando
            label: a label to identify this note
        """
        MusicObj.__init__(self, label=label, dur=dur, start=start)
        self.midi: float = tools.asmidi(pitch)
        self.amp:  Opt[float] = amp
        self.endmidi: float = tools.asmidi(endpitch) \
            if endpitch is not None else None
        
    def __hash__(self) -> int:
        return hash((self.midi, self.dur, self.start, self.endmidi, self.label))

    def clone(self, pitch:U[float, str]=UNSET, amp:float=UNSET,
              dur:Fraction=UNSET, start:Fraction=UNSET, label:str=UNSET
              ) -> Note:
        # we can't use the base .clone method because pitch can be anything
        return Note(pitch=pitch if pitch is not UNSET else self.midi,
                    amp=amp if amp is not UNSET else self.amp,
                    dur=dur if dur is not UNSET else self.dur,
                    start=start if start is not UNSET else self.start, 
                    label=label or self.label)

    # s: half semitone shart, f: half semitone flat,
    # qs: quarter semitone shart, qf: qf quarter semitone flat
    @property
    def s(self): return self + 0.5

    @property
    def f(self): return self - 0.5

    @property
    def qs(self): return self + 0.25

    @property
    def qf(self):  return self - 0.25

    def asChord(self): return Chord(self)

    def isRest(self): return self.amp == 0
        
    def shift(self, freq:float) -> Note:
        """
        Return a copy of self, shifted in freq.

        C3.shift(C3.freq)
        -> C4
        """
        return self.clone(pitch=f2m(self.freq + freq))

    def transpose(self, step: float) -> Note:
        """ Return a copy of self, transposed `step` steps """
        return self.clone(pitch=self.midi + step)

    def __call__(self, cents:int) -> Note:
        """
        Transpose this note by the given cents. This is mainly
        used intercatively, together with the predefined
        notes.
        
        C4(50) is the same as N(60.5)
        """
        return self + cents/100.
        
    def __eq__(self, other:pitch_t) -> bool:
        if isinstance(other, str):
            other = str2midi(other)
        return self.__float__() == float(other)

    def __ne__(self, other:pitch_t) -> bool:
        return not(self == other)

    def __lt__(self, other:pitch_t) -> bool:
        if isinstance(other, str):
            other = str2midi(other)
        return self.__float__() < float(other)

    @property
    def freq(self) -> float: return m2f(self.midi)

    @freq.setter
    def freq(self, value:float) -> None: self.midi = f2m(value)

    @property
    def name(self) -> str: return m2n(self.midi)

    def roundPitch(self, semitoneDivisions:int=0) -> Note:
        divs = semitoneDivisions or config['show.semitoneDivisions']
        res = 1 / divs
        return self.quantize(res)
    
    def overtone(self, n:int) -> 'Note':
        return Note(f2m(self.freq * n))
    
    @property
    def cents(self) -> int:
        return tools.midicents(self.midi)

    @property
    def centsrepr(self) -> str:
        return tools.centsshown(self.cents,
                                divsPerSemitone=config['show.semitoneDivisions'])

    def scoringEvents(self) -> List[scoring.Note]:
        db = None if self.amp is None else amp2db(self.amp)
        dur = self.dur or config['defaultDuration']
        note = scoring.Note(self.midi, db=db, dur=dur, offset=self.start)
        if self.label:
            note.addAnnotation(self.label)
        return [note]

    def _asTableRow(self) -> List[str]:
        elements = [m2n(self.midi)]
        if config['repr.showFreq']:
            elements.append("%dHz" % int(self.freq))
        if self.amp is not None and self.amp < 1:
            elements.append("%ddB" % round(amp2db(self.amp)))
        if self.dur:
            elements.append(f"dur={self.dur}")
        return elements

    def __repr__(self) -> str:
        elements = self._asTableRow()
        return f'{elements[0].ljust(3)} {" ".join(elements[1:])}'

    def __str__(self) -> str: return self.name

    def __float__(self) -> float: return float(self.midi)

    def __int__(self) -> int: return int(self.midi)

    def __add__(self, other) -> U[Note, Chord]:
        if isinstance(other, (int, float)):
            return Note(self.midi+other, 
                        amp=self.amp,
                        dur=self.dur,
                        start=self.start,
                        endpitch=self.endmidi+other if self.endmidi else None)
        elif isinstance(other, Note):
            return Chord([self, other])
        elif isinstance(other, str):
            return self + asNote(other)
        raise TypeError(f"can't add {other} ({other.__class__}) to a Note")

    def __xor__(self, freq) -> Note: return self.shift(freq)

    def __sub__(self, other: U[Note, float, int]) -> Note:
        if isinstance(other, Note):
            raise TypeError("can't substract one note from another")
        elif isinstance(other, (int, float)):
            return self + (-other)
        raise TypeError(f"can't substract {other} ({other.__class__}) from a Note")

    def quantize(self, step=1.0) -> Note:
        """ Returns a new Note, rounded to step """
        return self.clone(pitch=round(self.midi / step) * step)

    def _events(self, playargs: PlayArgs) -> List[CsoundEvent]:
        amp = 1.0 if self.amp is None else self.amp
        endmidi = self.endmidi or self.midi
        
        bps = [(0.,                  self.midi, amp), 
               (float(playargs.dur), endmidi,   amp)]
        
        return [CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs)]

    def gliss(self, dur:time_t, endpitch:pitch_t, endamp:float=None,
              start:time_t=None) -> Line:
        endnote = asNote(endpitch)
        start = firstval(self.start, start, 0.)
        endamp = firstval(endamp, self.amp, 1.)
        breakpoints = [(start, self.midi, self.amp),
                       (start+dur, endnote.midi, endamp)]
        return Line(breakpoints)


def Rest(dur:Fraction=1, start:Fraction=None) -> Note:
    """
    Create a Rest. A Rest is a Note with pitch 0 and amp 0

    Args:
        dur: duration of the Rest
        start: start of the Rest

    Returns:

    """
    assert dur is not None and dur > 0
    return Note(pitch=0, dur=dur, start=start, amp=0)


def asNote(n: U[float, str, Note],
           amp:float=None, dur:time_t=None, start:time_t=None) -> Note:
    """
    Convert n to a Note

    n: str    -> notename
       number -> midinote
       Note   -> Note
    amp: 0-1

    A Note can also be created via `asNote((pitch, amp))`
    """
    if isinstance(n, Note):
        if any(x is not None for x in (amp, dur, start)):
            return n.clone(amp=amp, dur=dur, start=start)
        return n
    elif isinstance(n, (int, float)):
        return Note(n, amp=amp, dur=dur, start=start)
    elif isinstance(n, str):
        midi = str2midi(n)
        return Note(midi, amp=amp, dur=dur, start=start)
    elif isinstance(n, tuple) and len(n) == 2 and amp is None:
        return asNote(*n)
    raise ValueError(f"cannot express this as a Note: {n} ({type(n)})")


class Line(MusicObj):
    """ 
    A Line is a seq. of breakpoints, where each bp is of the form
    (delay, pitch, [amp=1, ...])

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

    __slots__ = ('bps',)

    def __init__(self, *bps, label="", delay:num_t=0, reltime=False):
        """

        Args:
            bps: breakpoints, a tuple of the form (delay, pitch, [amp=1, ...]), where
                delay is the time offset to the beginning of the line
                pitch is the pitch as notename or midinote
                amp is an amplitude between 0-1
            delay: time offset of the line itself
            label: a label to add to the line
            reltime: if True, the first value of each breakpoint is a time offset
                from previous breakpoint
        """
        if len(bps) == 1 and isinstance(bps[0], list):
            bps = bps[0]
        bps = tools.carryColumns(bps)
        
        if len(bps[0]) < 2:
            raise ValueError("A breakpoint should be at least (delay, pitch)", bps)
        
        if len(bps[0]) < 3:
            bps = tools.addColumn(bps, 1)
        
        bps = [(bp[0], tools.asmidi(bp[1])) + astuple(bp[2:])
               for bp in bps]
        
        if reltime:
            now = 0
            absbps = []
            for _delay, *rest in bps:
                now += _delay
                absbps.append((now, *rest))
            bps = absbps
        assert all(all(isinstance(x, (float, int)) for x in bp) for bp in bps)
        assert all(bp1[0]>bp0[0] for bp0, bp1 in iterlib.pairwise(bps))
        super().__init__(dur=bps[-1][0], start=delay, label=label)
        self.bps = bps
        
    def getOffsets(self) -> List[num_t]:
        """ Return absolute offsets of each breakpoint """
        start = self.start
        return [bp[0] + start for bp in self.bps]

    def _events(self, playargs: PlayArgs) -> List[CsoundEvent]:
        return [CsoundEvent.fromPlayArgs(bps=self.bps, playargs=playargs)]

    def __hash__(self):
        return hash((self.start, *iterlib.flatten(self.bps)))
        
    def __repr__(self):
        return f"Line(start={self.start}, bps={self.bps})"

    def quantize(self, step=1.0) -> Line:
        """ Returns a new object, rounded to step """
        bps = [ (bp[0], tools.quantizeMidi(bp[1]), bp[2:])
                for bp in self.bps ]
        return Line(bps)

    def transpose(self, step: float) -> Line:
        """ Transpose self by `step` """
        bps = [ (bp[0], bp[1] + step, bp[2:])
                for bp in self.bps ]
        return Line(bps)

    def scoringEvents(self):
        offsets = self.getOffsets()
        group = scoring.makeId()
        notes = []
        for (bp0, bp1), offset in zip(iterlib.pairwise(self.bps), offsets):
            ev = scoring.Note(pitch=bp0[1], offset=offset, dur=bp1[0] - bp0[0],
                              gliss=bp0[1] != bp1[1], group=group)
            notes.append(ev)
        if(self.bps[-1][1] != self.bps[-2][1]):
            # add a last note if last pair needed a gliss (to have a destination note)
            notes.append(scoring.Note(pitch=self.bps[-1][1], 
                                      offset=offsets[-1], 
                                      group=group,
                                      dur=asTime(config['show.lastBreakpointDur'])))
        return notes

    def dump(self):
        elems = []
        if self.start:
            elems.append(f"delay={self.start}")
        if self.label:
            elems.append(f"label={self.label}")
        infostr = ", ".join(elems)
        print("Line:", infostr)
        durs = [bp1[0]-bp0[0] for bp0, bp1 in iterlib.pairwise(self.bps)]
        durs.append(0)
        rows = [(offset, offset+dur, dur) + bp
                for offset, dur, bp in zip(self.getOffsets(), durs, self.bps)]
        headers = ("start", "end", "dur", "offset", "pitch", "amp", "p4", "p5", "p6", "p7", "p8")
        lib.print_table(rows, headers=headers)


class NoteSeq(MusicObj):
    """
    A seq. of Notes. In a sequence, notes have no offset/start,
    each notes starts at the end of the previous one.

    Each note can have its own dur. or use a default
    (param notedur/config['default
    """

    __slots__ = ("notes", "_noteDur")

    def __init__(self, *notes: List[Note], notedur=None, start=None):
        MusicObj.__init__(self, start=start)
        if not notes:
            self.notes = []
        else:
            if len(notes) == 1:
                n0 = notes[0]
                if lib.isiterable(n0):
                    notes = n0
                elif isinstance(n0, str) and " " in n0:
                    notes = n0.split()
            self.notes = [asNote(n) for n in notes]
        self._noteDur = notedur

    def __len__(self) -> int:
        return len(self.notes)

    def __iter__(self):
        return iter(self.notes)

    def __getitem__(self, idx):
        out = self.notes.__getitem__(idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out    

    def asChord(self):
        return Chord(self)

    def __repr__(self):
        notestr = ", ".join(n.name for n in self)
        return f"NoteSeq({notestr})"
        
    def __hash__(self) -> int:
        if self._hash:
            return self._hash
        hashes = tuple(hash(note) for note in self)
        hashes += (self._noteDur, self.start)
        self._hash = hash(hashes)
        return self._hash

    def __mul__(self: T, other) -> T:
        return self.__class__(list(self).__mul__(other))

    def scoringEvents(self) -> List[scoring.Event]:
        defaultdur = self._noteDur or config['show.seqDuration']
        now = self.start or F(0)
        evs = []
        group = scoring.makeId()
        for note in self:
            ev = note.scoringEvents()[0]
            ev.dur = note.dur or defaultdur
            ev.offset = now
            ev.group = group
            now += ev.dur
            evs.append(ev)
        return evs

    def _events(self, playargs: PlayArgs) -> List[CsoundEvent]:
        now = float(self.start) if self.start else 0.
        events = []
        defaultDur = self._noteDur or config['defaultDuration']
        for note in self.notes:
            notedur = note.dur or defaultDur
            playargs2 = playargs.fill(dur=notedur, delay=now)
            now += notedur
            events.extend(note._events(playargs2))
        return events

    def transpose(self, step) -> NoteSeq:
        """ Return a transposed version of self """
        notes = [n.transpose(step) for n in self]
        return NoteSeq(notes, notedur=self._noteDur, start=self.start)

    def quantize(self, step=1.0) -> NoteSeq:
        """ Return a new NoteSeq with notes quantized to a grid
        defined by step """
        notes = [n.quantize(step) for n in self]
        return NoteSeq(notes, notedur=self._noteDur, start=self.start)


def N(pitch, dur:time_t=None, start:time_t=None, endpitch:pitch_t=None,
      amp:float=None
      ) -> U[Note, Chord]:
    """
    Create a Note. If pitch is a list of pitches, creates a Chord instead

    Args:
        pitch: a pitch (as float, int, str) or list of pitches (also a str
            with spaces, like "A4 C5"). If multiple pitches are passed,
            the result is a Chord
        dur: the duration of the note/chord (optional)
        start: the start time of the note/chord (optional)
        endpitch: the end pitch of the note/chord (optional, must match the
            number of pitches passes as start pitch)
        amp: the amplitude of the note/chord (optional)

    Returns:
        a Note or Chord, depending on the number of pitches passed
    """
    if isinstance(pitch, (tuple, list)):
        return Chord(pitch, dur=dur, start=start, endpitches=endpitch, amp=amp)
    elif isinstance(pitch, str):
        if " " in pitch:
            return Chord(pitch, dur=dur, start=start, endpitches=endpitch, amp=amp)
        else:
            return Note(pitch, dur=dur, start=start, endpitch=endpitch, amp=amp)
    else:
        return Note(pitch, dur=dur, start=start, endpitch=endpitch, amp=amp)


class Chord(MusicObj):

    __slots__ = ('amp', 'endchord', 'notes')

    def __init__(self, *notes, amp:float=None, dur:Fraction=None,
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
        self.amp = amp
        self._hash = None
        if dur is not None:
            assert dur > 0
            dur = F(dur)
        MusicObj.__init__(self, dur=dur, start=start, label=label)
        self.notes: List[Note] = []
        if notes:
            # notes might be: Chord([n1, n2, ...]) or Chord(n1, n2, ...)
            if lib.isgenerator(notes):
                notes = list(notes)
            if isinstance(notes[0], (list, tuple)):
                assert len(notes) == 1
                notes = notes[0]
            elif isinstance(notes[0], str) and len(notes) == 1:
                notes = notes[0].split()
            # determine dur
            if dur is None and any(isinstance(note, Note) and note.dur is not None 
                                   for note in notes):
                dur = max(note.dur if note.dur is not None else 0
                          for note in notes)
            for note in notes:
                if isinstance(note, Note):
                    note = note.clone(dur=None, amp=amp, start=None)
                else:
                    note = asNote(note, amp=amp, dur=dur, start=None)
                self.notes.append(note)
            self.sort()
            self.endchord = asChord(endpitches) if endpitches else None

    def __len__(self) -> int:
        return len(self.notes)

    def __getitem__(self, idx) -> U[Note, Chord]:
        out = self.notes.__getitem__(self, idx)
        if isinstance(out, list):
            out = self.__class__(out)
        return out

    def __iter__(self) -> Iter[Note]:
        return iter(self.notes)

    def scoringEvents(self) -> List[scoring.Event]:
        pitches = [note.midi for note in self.notes]
        db = None if self.amp is None else amp2db(self.amp)
        annot = None if self.label is None else self.label
        endpitches = None if not self.endchord else [note.midi for note in self.endchord]
        return [scoring.Chord(pitches, db=db, annot=annot,
                              endpitches=endpitches,
                              dur=self.dur, offset=self.start)]

    def asmusic21(self, arpeggio=None) -> m21.stream.Stream:
        arpeggio = _normalizeChordArpeggio(arpeggio, self)
        if arpeggio:
            dur = config['show.arpeggioDuration']
            return NoteSeq(self.notes, notedur=dur).asmusic21()
        else:
            return _splitChords([self], dur=self.dur or config['show.seqDuration'])

    def __hash__(self):
        if self._hash:
            return self._hash
        data = (self.dur, self.start, *(n.midi for n in self.notes))
        if self.endchord:
            data = (data, tuple(n.midi for n in self.endchord))
        self._hash = h = hash(data)
        return h

    def getFreqs(self):
        """ Return the frequencies of the notes in this chord """
        return [n.freq for n in self.notes]

    def append(self, note:pitch_t) -> None:
        """ append a note to this Chord """
        note = asNote(note)
        if note.freq < 17:
            logger.debug(f"appending a note with very low freq: {note.freq}")
        self.notes.append(note)
        self._changed()

    def extend(self, notes) -> None:
        """ extend this Chord with the given notes """
        for note in notes:
            self.notes.append(asNote(note))
        self._changed()

    def insert(self, index:int, note:pitch_t) -> None:
        self.notes.insert(index, asNote(note))
        self._changed()

    def filter(self, func) -> Chord:
        """
        Example: filter out notes lower than the lowest note of the piano

        return ch.filter(lambda n: n > "A0")
        """
        return Chord([n for n in self if func(n)])
        
    def transpose(self, step:float) -> Chord:
        """
        Return a copy of self, transposed `step` steps
        """
        return Chord([note.transpose(step) for note in self])

    def shift(self, freq:float) -> Chord:
        return Chord([note.shift(freq) for note in self])

    def roundPitch(self, semitoneDivisions:int=0) -> Chord:
        """
        Returns a copy of this chord, with pitches rounded according
        to semitoneDivisions

        Args:
            semitoneDivisions: if 2, pitches are rounded to the next
                1/4 tone

        Returns:
            the new Chord
        """
        divs = semitoneDivisions or config['show.semitoneDivisions']
        notes=[note.roundPitch(divs) for note in self]
        return self._withNewNotes(notes)
    
    def _withNewNotes(self, notes) -> Chord:
        return Chord(notes, start=self.start, dur=self.dur, amp=self.amp)

    def quantize(self, step=1.0) -> Chord:
        """
        Returns a copy of this chord, with the pitches
        quantized. Two notes with the same pitch are considered
        equal if they quantize to the same pitch, independently
        of their amplitude. In the case of two equal notes, only
        the first one is kept.
        """
        seenmidi = set()
        notes = []
        for note in self:
            note2 = note.quantize(step)
            if note2.midi not in seenmidi:
                seenmidi.add(note2.midi)
                notes.append(note2)
        return self._withNewNotes(notes)

    def __setitem__(self, i:int, obj:pitch_t) -> None:
        self.notes.__setitem__(i, asNote(obj))
        self._changed()

    def __add__(self, other:pitch_t) -> Chord:
        if isinstance(other, Note):
            s = Chord(self)
            s.append(other)
            return s
        elif isinstance(other, (int, float)):
            s = [n + other for n in self]
            return Chord(s)
        elif isinstance(other, (Chord, str)):
            return Chord(self.notes + asChord(other).notes)
        raise TypeError("Can't add a Chord to a %s" % other.__class__.__name__)

    def splitByAmp(self, numChords=8, maxNotesPerChord=16) -> List[Chord]:
        """
        Split the notes in this chord into several chords, according
        to their amplitude

        Args:
            numChords: the number of chords to split this chord into
            maxNotesPerChord: max. number of notes per chord

        Returns:
            a list of Chords
        """
        midis = [note.midi for note in self.notes]
        amps = [note.amp for note in self.notes]
        chords = tools.splitByAmp(midis, amps, numGroups=numChords,
                                  maxNotesPerGroup=maxNotesPerChord)
        return [Chord(chord) for chord in chords]

    def loudest(self, n:int) -> Chord:
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        return self.copy().sort(key='amp', reverse=True)[:n]

    def sort(self, key='pitch', reverse=False) -> Chord:
        """
        Sort INPLACE. If inplace sorting is undesired, use

        sortedchord = chord.copy().sort()

        Args:
            key: either 'pitch' or 'amp'
            reverse: similar as sort

        Returns:
            self
        """
        if key == 'pitch':
            self.notes.sort(key=lambda n: n.midi, reverse=reverse)
        elif key == 'amp':
            self.notes.sort(key=lambda n:n.amp, reverse=reverse)
        return self

    def _events(self, playargs) -> List[CsoundEvent]:
        gain = playargs.gain
        if config['chord.adjustGain']:
            gain *= 1/sqrt(len(self))
        if self.endchord is None:
            return sum((note._events(playargs) for note in self), [])
        events = []
        for note0, note1 in zip(self.notes, self.endchord):
            bps = [(0, note0.midi, note0.amp*gain),
                   (playargs.dur, note1.midi, note1.amp*gain)]
            events.append(CsoundEvent.fromPlayArgs(bps=bps, playargs=playargs))
        return events

    def asSeq(self, dur=None) -> 'NoteSeq':
        return NoteSeq(*self, notedur=dur)

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
            line = " ".join(justify(element, justs[i])
                            for i, element in enumerate(elements))
            if i == 0:
                line = f"{cls} | " + line
            else:
                line = f"{indent} | " + line
            lines.append(line)
        return "\n".join(lines)
        
    def mapamp(self, curve, db=False) -> Chord:
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

    def setamp(self, amp: float) -> Chord:
        """
        Returns a new Chord where each note has the given amp. 
        This is a shortcut to

        ch2 = Chord([note.clone(amp=amp) for note in ch])

        See also: .scaleamp
        """
        return self.scaleamp(0, offset=amp)

    def scaleamp(self, factor:float, offset=0.0) -> Chord:
        """
        Returns a new Chord with the amps scales by the given factor
        """
        return Chord([note.clone(amp=note.amp*factor+offset)
                      for note in self.notes])

    def equalize(self:T, curve) -> T:
        """
        Return a new Chord equalized by curve

        curve: a func(freq) -> gain
        """
        notes = []
        for note in self:
            gain = curve(note.freq)
            notes.append(note.clone(amp=note.amp*gain))
        return self.__class__(notes)

    def gliss(self, dur:float, endnotes, start=None) -> Chord:
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
        startpitches = [note.midi for note in self.notes]
        endpitches = [note.midi for note in endchord]
        assert len(startpitches) == len(endpitches)
        out = Chord(*startpitches, amp=self.amp, label=self.label, endpitches=endpitches)
        out.dur = asTime(dur)
        out.start = None if start is None else asTime(start)
        return out

    def difftones(self) -> Chord:
        """
        Return a Chord representing the difftones between the notes of this chord
        """
        from emlib.music.combtones import difftones
        return Chord(difftones(*self))

    def isCrowded(self) -> bool:
        return any(abs(n0.midi-n1.midi)<=1 and abs(n1.midi-n2.midi)<=1
                   for n0, n1, n2 in iterlib.window(self, 3))

    def _splitChord(self, splitpoint=60.0, showcents=None, showlabel=True) -> m21.stream.Score:
        if showcents is None: showcents = config['show.cents']
        parts = splitNotesIfNecessary(self.notes, float(splitpoint))
        score = m21.stream.Score()
        for notes in parts:
            midinotes = [n.midi for n in notes]
            m21chord = m21funcs.m21Chord(midinotes, showcents=showcents)
            part = m21.stream.Part()
            part.append(m21funcs.bestClef(midinotes))
            if showlabel and self.label:
                part.append(m21funcs.m21Label(self.label))
                showlabel = False
            part.append(m21chord)
            if config['show.centsMethod'] == 'expression':
                m21tools.makeExpressionsFromLyrics(part)
            score.insert(0, part)
        return score


def asChord(obj, amp:float=None, dur:float=None) -> Chord:
    """
    Create a Chord from `obj`

    Args:
        obj: a string with spaces in it, a list of notes, a Chord
        amp: the amp of the chord
        dur: the duration of the chord

    Returns:
        a Chord
    """
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
    

def _normalizeChordArpeggio(arpeggio: U[str, bool], chord: Chord) -> bool:
    if arpeggio is None: arpeggio = config['chord.arpeggio']

    if isinstance(arpeggio, bool):
        return arpeggio
    elif arpeggio == 'auto':
        return chord.isCrowded()
    else:
        raise ValueError(f"arpeggio should be True, False, 'auto' (got {arpeggio})")


class ChordSeq(MusicObj):
    """
    A seq. of Chords
    """
    __slots__ = ('chords', '_chorddur')

    def __init__(self, *chords, chorddur:float=None):
        self.chords: List[Chord] = []
        super().__init__()
        if len(chords) == 1 and isinstance(chords[0], (list, tuple)):
            chords = chords[0]
        if chords:
            chords = [asChord(chord) for chord in chords]
            default_dur = chorddur or config['defaultDuration']
            for chord in chords:
                if chord.dur is None:
                    chord = chord.clone(dur=default_dur)
                self.chords.append(chord)
        self._chorddur = chorddur

    def append(self, chord:Chord):
        self.chords.append(chord)

    def __len__(self) -> int:
        return len(self.chords)

    def __iter__(self) -> Iter[Chord]:
        return iter(self.chords)

    def __getitem__(self, idx):
        out = self.chords.__getitem__(self, idx)
        if not out or isinstance(out, Chord):
            return out
        elif isinstance(out, list):
            return self.__class__(out)
        else:
            raise ValueError("__getitem__ returned {out}, expected Chord or list of Chords")

    def asChord(self) -> Chord:
        """Join all the individual chords into one chord"""
        return Chord(list(set(iterlib.flatten(self))))

    def scoringEvents(self) -> List[scoring.Event]:
        events = []
        defaultDur = config['show.seqDuration']
        group = scoring.makeId()
        now = self.start or F(0)
        for chord in self:
            ev = chord.scoringEvents()[0]
            ev.dur = chord.dur or self.dur or defaultDur
            ev.offset = now
            ev.group = group
            now += ev.dur
            events.append(ev)
        return events

    def asmusic21(self, split=None, showcents=None, dur=None
                  ) -> m21.stream.Stream:
        if showcents is None: showcents = config['show.cents']
        dur = dur or self._chorddur or config['show.seqDuration']
        return _splitChords(self.chords, split=split,
                            showcents=showcents, dur=dur)

    def __repr__(self):
        lines = ["ChordSeq "]
        lines.extend("   "+" ".join(n.name.ljust(6) for n in ch) for ch in self)
        return "\n".join(lines)

    def __hash__(self):
        if self._hash:
            return self._hash
        self._hash = hash(tuple(hash(chord) ^ 0x1234 for chord in self))
        return self._hash

    def _events(self, playargs: PlayArgs) -> List[CsoundEvent]:
        now = playargs.delay or 0.
        allevents = []
        for i, chord in enumerate(self):
            d = chord.dur if (chord.dur and chord.dur > 0) else playargs.dur
            playargs2 = playargs.fill(delay=now, dur=d)
            allevents.extend(chord._events(playargs2))
            now += d
        return allevents

    def cycle(self, dur:float) -> ChordSeq:
        """
        Cycle the chords in this seq. until the given duration is reached 
        """
        out = ChordSeq()
        defaultDur = config['show.seqDuration']
        chordstream = iterlib.cycle(self)
        totaldur = 0
        while totaldur < dur:
            chord = next(chordstream)
            if chord.dur is None:
                chord = chord.clone(dur=defaultDur)
            totaldur += chord.dur
            out.append(chord)
        return out

    def transpose(self:ChordSeq, step) -> ChordSeq:
        chords = [obj.transpose(step) for obj in self.chords]
        return ChordSeq(chords, chorddur=self._chorddur)

    def quantize(self, step=1.0) -> ChordSeq:
        chords = [obj.quantize(step) for obj in self.chords]
        return ChordSeq(chords, chorddur=self._chorddur)


class Track(MusicObj):
    """
    A Track is a seq. of non-overlapping objects
    """

    def __init__(self, objs=None):
        self.timeline: List[MusicObj] = []
        self.instrs: dict[MusicObj, str] = {}
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
        super()._changed()

    def endTime(self) -> Fraction:
        if not self.timeline:
            return Fraction(0)
        return self.timeline[-1].end

    def isEmptyBetween(self, start:time_t, end:num_t):
        if not self.timeline:
            return True
        if start >= self.timeline[-1].end:
            return True
        if end < self.timeline[0].start:
            return True
        for item in self.timeline:
            if lib.intersection(item.start, item.end, start, end):
                return False
        return True

    def needsSplit(self) -> bool:
        pass

    def add(self, obj:MusicObj) -> None:
        """
        Add this object to this Track. If obj has already a given start,
        it will be inserted at that offset, otherwise it will be appended
        to the end of this Track. 

        1) To insert an untimed object (for example, a Note with start=None) to the Track
           at a given offset, set its .start attribute or do track.add(chord.clone(start=...))

        2) To append a timed object at the end of this track (overriding the start
           time of the object), do track.add(obj.clone(start=track.endTime()))

        obj: the object to add (a Note, Chord, Event, etc.)
        """
        if obj.start is None or obj.dur is None:
            obj = _asTimedObj(obj, start=self.endTime(), dur=config['defaultDuration'])
        if not self.isEmptyBetween(obj.start, obj.end):
            msg = f"obj {obj} ({obj.start}:{obj.start+obj.dur}) does not fit in track"
            raise ValueError(msg)
        assert obj.start is not None and obj.start >= 0 and obj.dur is not None and obj.dur > 0
        self.timeline.append(obj)
        self.timeline.sort(key=lambda obj:obj.start)
        self._changed()

    def extend(self, objs:List[MusicObj]) -> None:
        objs.sort(key=lambda obj:obj.start)
        assert objs[0].start >= self.endTime()
        for obj in objs:
            self.timeline.append(obj)
        self._changed()

    def scoringEvents(self) -> List[scoring.Event]:
        return sum((obj.scoringEvents() for obj in self.timeline), [])
                  
    def _events(self, playargs: PlayArgs) -> List[CsoundEvent]:
        return sum((obj._events(playargs) for obj in self.timeline), [])

    def play(self, **kws) -> csoundengine.SynthGroup:
        """
        kws: any kws is passed directly to each individual event
        """
        return csoundengine.SynthGroup([obj.play(**kws) for obj in self.timeline])

    def scoringTrack(self) -> scoring.Track:
        return scoring.Track(self.scoringEvents())

    def transpose(self:Track, step) -> Track:
        return Track([obj.transpose(step) for obj in self.timeline])

    def quantize(self:Track, step=1.0) -> Track:
        return Track([obj.quantize(step) for obj in self.timeline])


def _asTimedObj(obj: MusicObj, start, dur) -> MusicObj:
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


def generateNotes(start=12, end=127) -> dict[str, Note]:
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
    MusicObj._setJupyterHook()
    tools.m21JupyterHook()


def splitNotesOnce(notes: U[Chord, Seq[Note]], splitpoint:float, deviation=None,
                   ) -> tuple[List[Note], List[Note]]:
    """
    Split a list of notes into two lists, one above the splitpoint,
    one below

    Args:
        notes: a seq. of Notes
        splitpoint: the pitch to split the notes
        deviation: an acceptable deviation to fit all notes
            in one group (config: 'splitAcceptableDeviation')

    Returns:
        notes above and below

    """
    deviation = deviation or config['splitAcceptableDeviation']
    if all(note.midi>splitpoint-deviation for note in notes):
        above = [n for n in notes]
        below = []
    elif all(note.midi<splitpoint+deviation for note in notes):
        above = []
        below = [n for n in notes]
    else:
        above, below = [], []
        for note in notes:
            (above if note.midi>splitpoint else below).append(note)
    return above, below


def splitNotes(notes: Iter[Note], splitpoints:List[float], deviation=None
               ) -> List[List[Note]]:
    """
    Split notes at given splitpoints. This can be used to split a group of notes
    into multiple staves

    Args:
        notes: the notes to split
        splitpoints: a list of splitpoints
        deviation: an acceptable deviation to fit all notes
            in one group (config: 'splitAcceptableDeviation')

    Returns:
        A list of list of notes, where each list contains notes either above,
        below or between splitpoints
    """
    splitpoints = sorted(splitpoints)
    tracks = []
    above = notes
    for splitpoint in splitpoints:
        above, below = splitNotesOnce(above, splitpoint=splitpoint, deviation=deviation)
        if below:
            tracks.append(below)
        if not above:
            break
    return tracks


def splitNotesIfNecessary(notes:List[Note], splitpoint:float, deviation=None
                          ) -> List[List[Note]]:
    """
    Like splitNotesOnce, but returns only groups which have notes in them
    This can be used to split in more than one staves, like:

    Args:
        notes: the notes to split
        splitpoint: the split point
        deviation: an acceptable deviation, if all notes could fit in one part

    Returns:
        a list of parts (a part is a list of notes)

    """
    return [p for p in splitNotesOnce(notes, splitpoint, deviation) if p]


def _resolveSplitpoint(split) -> float:
    if split is None: split = config['show.split']
    if isinstance(split, bool):
        return 60.0 if split else 0.0
    return split
        

def _splitChords(chords:List[Chord], split=60, showcents=None,
                 showlabel=True, dur=1) -> m21.stream.Score:
    if showcents is None: showcents = config['show.cents']
    if dur is None: dur = config['show.seqDuration']
    chordsup, chordsdown = [], []
    splitpoint = _resolveSplitpoint(split)

    for chord in chords:
        above, below = splitNotesOnce(chord, splitpoint)
        chordsup.append(above)
        chordsdown.append(below)

    def isRowEmpty(chords: List[List[Note]]) -> bool:
        return all(not chord for chord in chords)

    rows = [row for row in (chordsup, chordsdown) if not isRowEmpty(row)]
    columns = zip(*rows)
    labels = [chord.label for chord in chords]

    def makePart(row: List[List[Note]]):
        part = m21.stream.Part()
        notes: List[Note] = list(iterlib.flatten(row))
        midinotes = [n.midi for n in notes]
        clef = m21funcs.bestClef(midinotes)
        part.append(clef)
        return part

    parts = [makePart(row) for row in rows]
    for column, label in zip(columns, labels):
        if showlabel and label:
            parts[0].append(m21funcs.m21Label(label))
        for chord, part in zip(column, parts):
            if chord:
                midis = [n.midi for n in chord]
                part.append(m21funcs.m21Chord(midis, showcents=showcents,
                                              quarterLength=dur))
            else:
                part.append(m21.note.Rest(quarterLength=dur))
    return m21.stream.Score(parts)
                

def _makeImage(obj: MusicObj, outfile:str=None, fmt:str=None, fixstream=True,
               **options) -> str:
    """
    Given a music object, make an image representation of it.
    NB: we put it here in order to make it easier to cache images

    Args:
        obj     : the object to make the image from (a Note, Chord, etc.)
        outfile : the path to be generated
        fmt     : format used. One of 'xml.png', 'lily.png' (no pdf)
        options : any argument passed to .asmusic21

    Returns:
        the path of the generated image
    """
    m21obj = obj.asmusic21(**options)
    if fixstream and isinstance(m21obj, m21.stream.Stream):
        m21obj = m21fix.fixStream(m21obj, inPlace=True)
    fmt = fmt or config['show.format'].split(".")[0]+".png"
    logger.debug(f"makeImage: using format: {fmt}")
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
    Given a music object, make an image representation of it.
    NB: we put it here in order to make it easier to cache images

    Args:
        obj     : the object to make the image from (a Note, Chord, etc.)
        outfile : the path to be generated
        fmt     : format used. One of 'xml.png', 'lily.png' (no pdf)
        options : any argument passed to .asmusic21

    Returns:
        the path of the generated image

    NB: we put it here in order to make it easier to cache images
    """
    return _makeImage(*args, **kws)


def resetImageCache() -> None:
    """
    Reset the image cache. Useful when changing display format
    """
    _makeImageCached.cache_clear()


def asMusic(obj) -> MusicObj:
    """
    Convert obj to a Note or Chord, depending on the input itself

    int, float      -> Note
    list (of notes) -> Chord
    "C4"            -> Note
    "C4 E4"         -> Chord
    """
    if isinstance(obj, MusicObj):
        return obj
    elif isinstance(obj, str):
        if " " in obj:
            return Chord(obj.split())
        return Note(obj)
    elif isinstance(obj, (list, tuple)):
        return Chord(obj)
    elif isinstance(obj, (int, float)):
        return Note(obj)


def gliss(a, b, dur:time_t=1, start:time_t=None) -> U[Note, Chord]:
    """
    Create a gliss. between a and b. a should implement
    the method .gliss (either a Note or a Chord)
    Args:
        a: the start object
        b: the end object (should have the same type as obj1)
        dur: the duration of the glissando
        start: the start time of the glissando

    Returns:

    """
    m1: U[Note, Chord] = asMusic(a)
    m2 = asMusic(b)
    assert isinstance(m2, type(m1))
    return m1.gliss(dur, m2, start=start)


class Group(MusicObj):
    """
    A Group represents a group of objects which belong together.

    a, b = Note(60, dur=2), Note(61, start=2, dur=1)
    h = Group((a, b))

    # TODO: apply start to .play, .rec and .show (.events)
    """

    def __init__(self, objects:List[MusicObj], start=0., label:str=None):
        assert isinstance(objects, (list, tuple))
        MusicObj.__init__(self, label=label, start=start)
        self.objs: List[MusicObj] = []
        self.objs.extend(objects)

    def append(self, obj:MusicObj) -> None:
        self.objs.append(obj)

    def __len__(self) -> int:
        return len(self.objs)

    def __iter__(self) -> Iter[MusicObj]:
        return iter(self.objs)


    def __getitem__(self, idx) -> U[MusicObj, List[MusicObj]]:
        return self.objs[idx]

    def __repr__(self):
        objstr = self.objs.__repr__()
        return f"Group({objstr})"

    def __hash__(self):
        hashes = [hash(obj) for obj in self.objs]
        return hash(tuple(hashes))

    #def play(self, **kws) -> csoundengine.SynthGroup:
    #    return playMany(self, **kws)

    def rec(self, outfile:str=None, sr:int=None, **kws) -> str:
        return recMany(self.objs, outfile=outfile, sr=sr, **kws)

    def events(self, **kws) -> List[CsoundEvent]:
        delay = kws.get('delay')
        if delay is None: delay = 0
        kws['delay'] = delay + self.start
        return getEvents(self.objs, **kws)

    def _events(self, playargs: PlayArgs) -> List[CsoundEvent]:
        return sum((obj._events(playargs) for obj in self.objs), [])

    def quantize(self, step=1.0) -> Group:
        return Group([obj.quantize(step=step) for obj in self])

    def transpose(self, step) -> Group:
        return Group([obj.transpose(step) for obj in self])

    def scoringEvents(self) -> List[scoring.Event]:
        events = sum((obj.scoringEvents() for obj in self.objs), [])
        if self.start != 0:
            events = [ev.clone(offset=ev.offset+self.start)
                      for ev in events]
        return events


def playEvents(events: List[CsoundEvent]) -> csoundengine.SynthGroup:
    """
    Play a list of events

    Args:
        events: a list of CsoundEvents

    Returns:
        A SynthGroup

    Example:u
        a = Chord("C4 E4 G4", dur=2)
        b = Note("1000hz", dur=4, start=1)
        events = events((a, b))
        playEvents(events)

    """
    synths = []
    for ev in events:
        csdinstr = play.makeInstrFromPreset(ev.instr)
        args = ev.getArgs()
        synth = csdinstr.play(delay=args[0],
                              dur=args[1],
                              args=args[3:],
                              tabargs=ev.args,
                              priority=ev.priority)
        synths.append(synth)
    return csoundengine.SynthGroup(synths)


def getEvents(objs, **kws) -> List[CsoundEvent]:
    """
    Collect events of multiple objects using the same parameters

    Args:
        objs: a seq. of objects
        **kws: keywords passed to play

    Returns:
        a list of the events
    """
    return sum((obj.events(**kws) for obj in objs), [])


def playMany(objs, **kws) -> csoundengine.SynthGroup:
    """
    Play multiple objects with the same parameters

    Args:
        objs: the objects to play
        kws: any keywords passed to play

    """
    return playEvents(getEvents(objs, **kws))


def recMany(objs: List[MusicObj], outfile:str=None, sr:int=None, **kws
            ) -> str:
    """
    Record many objects with the same parameters
    kws: any keywords passed to rec
    """
    allevents = getEvents(objs, **kws)
    return play.recEvents(outfile=outfile, events=allevents, sr=sr)


def trill(note1, note2, totaldur, notedur=None) -> ChordSeq:
    """
    Create a trill

    Args:
        note1: the first note of the trill (can also be a chord)
        note2: the second note of the trill (can also  be a chord)
        totaldur: total duration of the trill
        notedur: duration of each note

    Returns:
        A realisation of the trill as a ChordSeq of at least the
        given totaldur (can be longer if totaldur is not a multiple
        of notedur)
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


