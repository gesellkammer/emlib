# -*- coding: utf-8 -*-
import functools as _functools
from collections import namedtuple
from fractions import Fraction
from bpf4 import bpf
import music21
import logging
from . import m21tools
from emlib.pitch import *
from emlib import lib
from emlib import conftools
from emlib.iterlib import pairwise, window
from emlib.snd import csoundengine
from math import sqrt
from functools import lru_cache


from typing import Optional as Opt, Sequence as Seq, List, \
    Tuple, Union, Iterable as Iter, SupportsFloat as Float, Dict, \
    Callable


raise DeprecationWarning("This module is deprecated. Use emlib.mus")

logger = logging.getLogger(f"emlib.event")


_defaultconfig = {
    'showfreq': True,
    'show.split': True,
    'autoshow': False,
    'showcents': True,
    'chord.arpeggio': 'auto',
    'chord.adjustGain': True,
    'notation.pitchResolution': 0.5,
    'm21.displayhook.install': True,
    'm21.displayhook.format': 'lily.png',
    'show.format': 'lily.png',
    'displayhook.install': True,
    'play.method': 'csound',
    'play.dur': 2.0,
    'play.gain': 1.0,
    'play.group': 'emlib.event',
    'play.instr': 'sine',
}

_validator = {
    'm21.displayhook.format::choices':
        ['musicxml.png', 'lily.png'],
    'displayhook.format::choices':
        ['musicxml.png', 'lily.png'],
    'show.format::choices': 
        ['xml.png', 'xml.pdf', 'musicxml.png', 'musicxml.pdf', 'lily.png', 'lily.pdf', 'repr'],
    'play.method::choices':
        ['csound', 'midi'],
    'chord.arpeggio::choices':
        ['auto', 'always', 'never', True, False],
    'play.preset::choices':
        ['sine', 'piano', 'tri', 'clarinet'],
    'play.gain::range': (0, 1)
}

config = None


def getConfig() -> conftools.ConfigDict:
    """
    Get configuration for this module
    """
    global config
    if config is None:
        config = conftools.makeConfig(f'emlib:{__name__}', _defaultconfig, _validator)
    return config


def _init() -> None:
    if config is not None:
        return
    cfg = getConfig()
    # if we are in ipython, set displayhooks
    if _in_ipython():
        logger.debug("We are running inside ipython. Set displayhooks")
        if cfg["m21.displayhook.install"]:
            # displayhook for m21 objects
            m21_ipythonhook()
        if cfg["displayhook.install"]:
            # displayhook for our own objects (Note, Chord, etc)
            set_ipython_displayhook()
    

def _in_ipython():
    try:
        from IPython.core.getipython import get_ipython
        ipython = get_ipython()
        return ipython is not None
    except ImportError:
        return False    


def asnote(n, amp=-1):
    # type: (Union[Note, int, float, str, Tuple[Union[str, float], float]], float) -> Note
    """
    n: str    -> notename
       number -> midinote
    amp: 0-1

    you can also create a note by doing asnote((pitch, amp))
    """
    if isinstance(n, Note):
        out = n
    elif isinstance(n, (int, float)):
        out = Note(n, amp=amp)
    elif isinstance(n, str):
        out = Note(n2m(n), amp=amp)
    elif isinstance(n, tuple) and len(n) == 2 and amp == -1:
        out = asnote(*n)
    else:
        raise ValueError("cannot express this as a Note")
    return out


def aschord(chord):
    # type: (Union[Chord, List, Tuple]) -> Chord
    if isinstance(chord, Chord):
        return chord
    elif isinstance(chord, (list, tuple)):
        return Chord(map(asnote, chord))
    else:
        raise ValueError("cannot express this as a Chord")


_AmpNote = namedtuple("Note", "note midi freq db step")


def split_notes_by_amp(midinotes, amps, numgroups=8, maxnotes_per_group=8):
    # type: (Iter[float], Iter[float], int, int) -> List[Chord]
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

# ------------------------------------------------------------
#
# Note
#
# ------------------------------------------------------------

class _Base:
    _showableInitialized = False

    def __init__(self):
        self._pngfile = None

    def quantize(self, step=1.0):
        pass

    def transpose(self, step):
        pass


    def show(self, fmt=None, wait=False, **options):
        """
        options:

        showcents: shows cents as text
        """
        if fmt is None:
            fmt = getConfig()['show.format']
        return _m21show(self.asmusic21(**options), fmt, wait=wait)

    def _changed(self):
        # type: () -> None
        self._pngfile = None

    @property
    def m21(self):
        # type: () -> music21.stream.Stream
        return self.asmusic21()

    def asmusic21(self, **options):
        # type: (...) -> music21.stream.Stream
        pass

    @classmethod
    def _ipython_displayhook(cls):
        if cls._showableInitialized:
            return
        logger.debug("music: setting ipython displayhook")
        from IPython.core.display import Image

        def showpng(self):
            if self._pngfile is not None:
                pngfile = self._pngfile
            else:
                fmt = getConfig()['show.format']
                pngfile = self._pngfile = self.asmusic21().write(fmt)
            img = Image(filename=pngfile, embed=True)
            return img._repr_png_()

        _ipython_displayhook(cls, showpng, fmt='image/png')


class _Playable:

    def play(self, dur=0, delay=-1, gain=-1, instr:str=None) -> csoundengine.AbstrSynth:
        """
        dur: 
            if > 0, override the duration of the note/event. 
            For events without duration, set the default dur in the config (key='play.dur')
            use -1 to play forever
        amp: 
            override the amplitude of the event.
        delay:
            override the delay of the event
        """
        if dur == 0:
            dur = self._playDur()
        if gain < 0:
            gain = config['play.gain']
        if delay < 0:
            try:
                delay = self.start
            except AttributeError:
                delay = 0
        method = config['play.method']
        if method == 'csound':
            return self._playCsound(dur=dur, delay=delay, gain=gain, instr=instr)
        elif method == 'midi':
            return self._playMidi(dur=dur, delay=delay, gain=gain)
        else:
            raise ValueError(f"method {method} not supported")

    def _playDur(self):
        return config['play.dur']

    def _getInstr(self) -> csoundengine.CsoundInstr:
        return getInstrPreset(config['play.instr'])

    def _playCsound(self, dur: float, delay: float=0, gain: float=1.0, instr: str=None) -> csoundengine.AbstrSynth:
        csdinstr = self._getInstr() if instr is None else getInstrPreset(instr)
        logger.debug(f"_playCsound: dur={dur}\n instr: {instr}, args={(self.amp*gain, self.midi, self.midi)}")
        synth = csdinstr.play(dur, delay=delay,
                              args=(self.amp*gain, self.midi, self.midi))
        return synth

    def _playMidi(self, dur=0, delay=0, gain=-1):
        raise NotImplementedError()


_instrdefs: Dict['str', Dict] = {
    'sine': dict(
        body = """
        idur = p3
        iamp = p4
        inote0 = p5
        inote1 = p6
        knote = linseg:k(inote0, idur, inote1)
        kfreq = mtof(knote)
        a0 oscili iamp, kfreq
        aenv linsegr 0, 0.05, 1, 0.05, 0
        a0 *= aenv
        outs a0, a0
        """
    ),
    'tri': dict(
        body = """
        idur = p3
        iamp   = p4
        inote0 = p5
        inote1 = p6
        knote = linseg:k(inote0, idur, inote1)
        kfreq = mtof(knote)
        a0 vco2 iamp, kfreq, 12
        aenv linsegr 0, 0.05, 1, 0.05, 0
        a0 *= aenv
        outs a0, a0
        """
    ),
    'piano': dict(
        body = """
        idur = p3
        iamp = p4
        inote0 = p5
        inote1 = p6
        idb    dbamp iamp
        
        ivel   bpf idb, -120, 0, -90, 10, -70, 20, -24, 90, 0, 127
        ; idb2    bpf idb, -120, 0, -90, -40, -18, -6, 0, 0
        idb2 = idb
        iamp2 ampdb idb2
        knote  linseg inote0, idur, inote1
        kfreq mtof knote
        ; kamp = 1/32768 * iamp2
        kamp = 1 / 16384 * iamp2
        a0, a1 sfinstr ivel, inote0, kamp, kfreq, 148, gi_fluidsf, 1
        ; a0, a1 sfplay ivel, inote0, kamp, kfreq, gi_fluidsf_pianopreset, 1
        aenv linsegr 1, 0.0001, 1, 0.2, 0
        a0 *= aenv
        a1 *= aenv
        outs a0, a1
        """,
        initcode = f"""
        gi_fluidsf  sfload "{csoundengine.fluidsf2Path()}"
        """
    ),
    'clarinet': dict(
        body="""
        idur, iamp, inote0, inote1 passign 3
        idb    dbamp iamp
        ivel   bpf idb, -120, 0, -90, 10, -70, 20, -24, 90, 0, 127
        knote  linseg inote0, idur, inote1
        kfreq mtof knote
        kamp = 1 / 16384 * iamp
        a0, a1 sfinstr ivel, inote0, kamp, kfreq, 61, gi_fluidsf, 1
        aenv linsegr 0, 0.01, 1, 0.2, 0
        a0 *= aenv
        a1 *= aenv
        outs a0, a1
        """,
        initcode=f'gi_fluidsf sfload "{csoundengine.fluidsf2Path()}"'
    ),
    'oboe': dict(
        body="""
        idur, iamp, inote0, inote1 passign 3
        idb    dbamp iamp
        ivel   bpf idb, -120, 0, -90, 10, -70, 20, -24, 90, 0, 127
        knote  linseg inote0, idur, inote1
        kfreq mtof knote
        kamp = 1 / 16384 * iamp
        a0, a1 sfinstr ivel, inote0, kamp, kfreq, 58, gi_fluidsf, 1
        aenv linsegr 0, 0.01, 1, 0.2, 0
        a0 *= aenv
        a1 *= aenv
        outs a0, a1
        """,
        initcode=f'gi_fluidsf sfload "{csoundengine.fluidsf2Path()}"'
    )
} 


@lru_cache(maxsize=100)
def getInstrPreset(instr:str) -> csoundengine.CsoundInstr:
    instrdict = _instrdefs.get(instr)
    if instrdict is None:
        raise KeyError(f"Valid instrument presets are: {_instrdefs.keys()}, but got {instr}")
    group = getConfig()['play.group']
    name = f'emlib.event.preset.{instr}'
    logger.debug(f"creating csound instr. name={name}, group={group}")
    csdinstr = csoundengine.makeInstr(name=name, body=instrdict['body'], initcode=instrdict.get('initcode'), group=group)
    logger.debug(f"Created {csdinstr}")
    return csdinstr


def availableInstrPresets() -> List[str]:
    return list(_instrdefs.keys())


@_functools.total_ordering
class Note(_Base, _Playable):

    def __init__(self, pitch, amp=-1):
        # type: (Union[float, str], Opt[float]) -> None
        """
        pitch: a midinote or a note as a string
        amp  : amplitude 0-1. Use -1 to leave it unset
        """
        _Base.__init__(self)
        _Playable.__init__(self)
        midi = pitch if not isinstance(pitch, str) else n2m(pitch)  # type: float
        self.midi = midi
        self._amp = amp
        self._pitchResolution = 0

    def __hash__(self):
        return hash(self.midi)

    def __call__(self, cents):
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
        """
        Return a copy of self, transposed `step` steps
        """
        return self.clone(pitch=self.midi + step)

    @property
    def amp(self):
        # type: () -> float
        return self._amp if self._amp >= 0 else 1.0

    @amp.setter
    def amp(self, value):
        self._amp = value

    def __eq__(self, other):
        return self.__float__() == float(other)

    def __ne__(self, other):
        return not(self == other)

    def __lt__(self, other):
        # type: (Float) -> bool
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
    def ampdb(self):
        # type: () -> float
        return amp2db(self.amp)

    @ampdb.setter
    def ampdb(self, db):
        # type: (float) -> None
        self.amp = db2amp(db)

    @property
    def name(self):
        # type: () -> str
        return m2n(self.midi)

    @property
    def roundedpitch(self):
        # type: () -> float
        res = self._pitchResolution
        if res <= 0:
            self._pitchResolution = res = getConfig()['notation.pitchResolution']
        return round(self.midi / res) * res

    @property
    def cents(self):
        # type: () -> int
        return int((self.midi - self.roundedpitch) * 100)

    def asmusic21(self, showcents=None):
        # type: (Opt[bool]) -> music21.note.Note
        basepitch = self.roundedpitch
        if self.midi == 0:
            return music21.note.Rest()
        note = music21.note.Note(basepitch)
        cents = int((self.midi - basepitch) * 100)
        note.microtone = cents
        showcents = showcents if showcents is not None else getConfig()['showcents']
        if showcents:
            note.lyric = self.centsrepr
        return note

    @property
    def centsrepr(self):
        # type: () -> str
        cents = self.cents
        if cents == 0:
            return ""
        elif cents > 0:
            return "+%d" % cents
        else:
            return "–%d" % abs(cents)

    def __repr__(self):
        # type: () -> str
        pitchstr = m2n(self.midi).ljust(6)
        details = []
        if getConfig()['showfreq']:
            details.append("%s Hz" % str(int(self.freq)).ljust(3))
        if self._amp >= 0:
            details.append("%d dB" % round(amp2db(self.amp)))
        return "%s [%s]" % (pitchstr, ", ".join(details))

    def __float__(self):
        # type: () -> float
        return float(self.midi)

    def __int__(self):
        # type: () -> int
        return int(self.midi)

    def __add__(self, other):
        # type: (Union[Note, float, int]) -> Note
        if isinstance(other, Note):
            return Chord([self, other])
        elif isinstance(other, (int, float)):
            return Note(self.midi + other, self._amp)
        else:
            raise TypeError("can't add a Note to a %s" % str(other.__class__))

    def __sub__(self, other):
        # type: (Union[Note, float, int]) -> Note
        if isinstance(other, Note):
            raise TypeError("can't substract one note from another")
        elif isinstance(other, (int, float)):
            return Note(self.midi - other, self._amp)
        else:
            raise TypeError(
                "can't substract a %s from a Note" % str(
                    other.__class__))

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


def F(x: Union[Fraction, float, int]) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10000000)
    

class Event(Note):

    def __init__(self, midi, amp=-1, dur=1, start=-1, label=""):
        # type: (float, float, Union[float, Fraction], Union[float, Fraction], str) -> None
        """
        An event is a note with a duration. It can have a start time
        """
        super().__init__(midi, amp)
        self.dur: Fraction = F(dur)
        self.start: Fraction = F(start)
        self.label: str = label

    @classmethod
    def fromNote(cls, note, dur=1, start=-1):
        return Event(note.midi, amp=note.amp, dur=dur, start=start)

    @property
    def end(self) -> float:
        return self.start + self.dur

    def _playDur(self):
        return self.dur

    def __repr__(self):
        # type: () -> str
        ampstr = "(%ddB)" % int(round(amp2db(self.amp))) if self._amp is not None else ""
        label = "[%s]" % self.label if self.label else ""
        start = float(self.start)
        end = float(self.end)
        pitch = m2n(self.midi).ljust(6)
        out = f"{start:.3f}:{end:.3f} -> {pitch} {ampstr} {label}"
        return out

    def copy(self):
        # type: () -> Event
        return Event(start=self.start, midi=self.midi, dur=self.dur,
                     amp=self.amp, label=self.label)

    def clone(self, start=None, midi=None, dur=None, amp=None, label=None):
        # type: (...) -> Event
        return Event(midi  = midi if midi is not None else self.midi, 
                     amp   = amp if amp is not None else self.amp,
                     dur   = dur if dur is not None else self.dur,
                     start = start if start is not None else self.start,
                     label = label if label is not None else self.label)
        
    def asmusic21(self, *args, **kws):
        # type: (...) -> music21.note.Note
        note = Note.asmusic21(self, *args, **kws)
        note.duration = music21.duration.Duration(self.dur)
        return note


class EventSeq(_Base, list):
    """
    Non overlapping seq. of events (a Voice)
    """

    def __init__(self, events=None):
        # type: (Opt[Iter[Event]]) -> None
        super(EventSeq, self).__init__()
        if events:
            if any(ev.start >= 0 for ev in events):
                if any(ev.start < 0 for ev in events):
                    raise ValueError("events should all have a start time set, or none of them should have a start time set")
                self.extend(events)
            else:
                now = F(0)
                for ev in events:
                    ev2 = ev.clone(start=now)
                    now += ev.dur
                    self.append(ev2)

    def removeOverlap(self):
        # type: () -> EventSeq
        self.sort(key=lambda event:event.start)
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

    def hasOverlap(self):
        return any(ev1.start - ev0.end > 0 for ev0, ev1 in pairwise(self))

    @property
    def m21(self):
        return self.asmusic21(split=False)

    def asmusic21(self, split=None):
        # type: () -> music21.stream.Voice
        split = split if split is not None else getConfig()['show.split']
        from music21.duration import Duration
        if self.hasOverlap():
            self.removeOverlap()
        voice = music21.stream.Voice()
        now = 0
        for ev in self:
            if ev.start > now:
                voice.append(music21.note.Rest(duration=Duration(ev.start - now)))
            voice.append(ev.asmusic21())
            now = ev.start + ev.dur
        voice.sliceByBeat(inPlace=True)
        maxbeat = int(now)+1
        voice.sliceAtOffsets(range(maxbeat), inPlace=True)
        m21stream = voice
        if split:
            midi0 = min(ev.midi for ev in self)
            midi1 = max(ev.midi for ev in self)
            if midi0 < 57 and midi1 > 63:
                m21stream = m21tools.splitvoice(voice)
        return m21stream

# ------------------------------------------------------------
#
# Chord
#
# ------------------------------------------------------------


class Chord(_Base, list, _Playable):

    def __init__(self, notes=None):
        # type: (Opt[Iter[Note]]) -> None
        _Base.__init__(self)
        _Playable.__init__(self)
        if notes:
            notes = set(map(asnote, notes))
            self.extend(notes)
            self.sortbypitch(inplace=True)

    def append(self, note):
        # type: (Union[Note, float, str]) -> None
        self._changed()
        note = asnote(note)
        if note.freq < 17:
            logger.debug(f"appending a note with very low freq: {note.freq}")
        super(self.__class__, self).append(note)

    def extend(self, notes):
        # type: (Iter[Union[Note, float, str]]) -> None
        for note in notes:
            self.append(note)
        self._changed()

    def insert(self, index, note):
        # type: (int, Union[Note, float, str]) -> None
        self._changed()
        note = asnote(note)
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
        return Chord(notes)

    def __setitem__(self, i, obj):
        # type: (int, Union[Note, float, str]) -> None
        self._changed()
        note = asnote(obj)
        super(self.__class__, self).__setitem__(i, note)

    def __add__(self, other):
        # type: (Union[Note, float, str]) -> Chord
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
        # type: (int, int) -> List[Chord]
        midinotes = [note.midi for note in self]
        amps = [note.amp for note in self]
        return split_notes_by_amp(midinotes, amps, numchords,
                                  max_notes_per_chord)

    def sortbyamp(self, reverse=True, inplace=True):
        # type: (bool, bool) -> Chord
        if inplace:
            out = self
        else:
            out = Chord(self)
        out.sort(key=lambda n: n.amp, reverse=reverse)
        return out

    def loudest(self, n):
        """
        Return a new Chord with the loudest `n` notes from this chord
        """
        return self.sortbyamp(inplace=False)[:n]

    def sortbypitch(self, reverse=False, inplace=True):
        # type: (bool, bool) -> Chord
        if inplace:
            out = self
        else:
            out = Chord(self)
        out.sort(key=lambda n: n.midi, reverse=reverse)
        return out

    def copy(self):
        # type: () -> Chord
        return Chord(self.notes)

    @property
    def notes(self):
        # type: () -> List[Note]
        return [note for note in self]

    @property
    def m21(self):
        return self.asmusic21(split=False)

    def asmusic21(self, showcents=None, split=None, arpeggio=None):
        cfg = getConfig()
        showcents = showcents if showcents is not None else cfg['showcents']
        arpeggio = arpeggio if arpeggio is not None else cfg['chord.arpeggio']
        split = split if split is not None else cfg['show.split']
        notes = sorted(self.notes, key=lambda n: n.midi)
        arpeggio = _normalize_chord_arpeggio(arpeggio, self)
        if arpeggio:
            voice = music21.stream.Voice()
            for n in self:
                n2 = n.asmusic21()
                n2.duration.quarterLength = 0.5
                voice.append(n2)
            m21stream = m21tools.splitvoice(voice) if split else voice
            return m21stream
        else:
            ch = music21.chord.Chord([n.m21 for n in notes])
            if showcents:
                lyric = "\n".join(note.centsrepr for note in 
                                  sorted(self, reverse=True, key=lambda n:n.midi))
                ch.lyric = lyric
            if split:
                return m21tools.splitchords([ch], 60)
            return ch

    def maxnotes(self, maxnotes):
        """
        Returns a Chord with at most `maxnotes`, based on relevance
        (amp weighted by freq.)

        NB: at the moment we just use amplitude
        """
        return self.sortbyamp(inplace=False, reverse=True)[:maxnotes]
        

    def show(self, fmt=None, split=None, arpeggio=None, wait=False):
        """
        split: split this chord at this point, if necessary. To avoid splitting,
               set this to None
        splitamp: if > 0, the chord will be split by amplitude in so many
               subchords.
        """
        cfg = getConfig()
        fmt = fmt or cfg['show.format']
        m21stream = self.asmusic21(split=split, arpeggio=arpeggio)
        return _m21show(m21stream, fmt, wait=wait)

    def asSeq(self):
        return EventSeq(self)

    def __repr__(self):
        lines = ["Chord"]
        for n in sorted(self.notes, key=lambda note:note.midi, reverse=True):
            lines.append("    %s" % str(n))
        return "\n".join(lines)

    def __getitem__(self, *args):
        out = list.__getitem__(self, *args)
        if isinstance(out, list):
            out = Chord(out)
        return out    
        
    def filter(self, func):
        notes = [n for n in self if func(n)]
        return Chord(notes)

    def _playCsound(self, dur: float, delay: float=0, gain: float=1.0, instr: str=None, adjustGain=None) -> csoundengine.AbstrSynth:
        logger.debug(f"_playCsound: dur={dur} delay={delay}")
        adjustGain = adjustGain if adjustGain is not None else config['chord.adjustGain']
        if adjustGain:
            logger.debug(f"playCsound: adjusting gain by {gain/sqrt(len(self))}")
            notegain = gain/sqrt(len(self))
        else:
            notegain = gain
        csdinstr = self._getInstr() if instr is None else getInstrPreset(instr)
        synths = [csdinstr.play(dur, delay=delay, args=[note.amp*notegain, note.midi, note.midi]) for note in self]
        synth = csoundengine.SynthGroup(synths) 
        return synth

    def mapamp(self, curve, db=True):
        """
        Change the amps of the notes within this Chord, according to curve

        Example: compress all amplitudes to 30 dB

        :param curve:
        :param db:
        :return:
        """
        notes =  []
        if db:
            for note in self:
                db = curve(amp2db(note.amp))
                notes.append(note.clone(amp=db2amp(db)))
        else:
            for note in self:
                amp2 = curve(note.amp)
                notes.append(note.clone(amp=amp2))
        return Chord(notes)

    def equalize(self, curve):
        # type: (Callable[[float], float]) -> Chord
        """
        Return a new Chord equalized by curve

        curve: a func(freq) -> gain
        """
        notes = []
        for note in self:
            gain = curve(note.freq)
            notes.append(note.clone(amp=note.amp*gain))
        return Chord(notes)



def _normalize_chord_arpeggio(arpeggio: Union[str, bool], chord: Chord) -> bool:
    if isinstance(arpeggio, bool):
        return arpeggio
    if arpeggio == 'always':
        return True
    elif arpeggio == 'never':
        return False
    elif arpeggio == 'auto':
        return _is_chord_crowded(chord)
    else:
        raise ValueError(f"arpeggio should be True, False, always, never or auto. Got: {arpeggio}")


def _is_chord_crowded(chord: Chord) -> bool:
    return any(abs(n0.midi - n1.midi) <= 1 and abs(n1.midi - n2.midi) <= 1 
               for n0, n1, n2 in window(chord, 3))

def stopsynths():
    """ 
    Stops all synths (notes, chords, etc) being played
    """
    group = getConfig()['play.group']
    man = csoundengine.getManager(group)
    man.unschedAll()

 
# ------------------------------------------------------------
#
# Helper functions for Note, Chord, ...
#
# ------------------------------------------------------------


_POOL = None


def _getpool():
    global _POOL 
    if _POOL is not None:
        return _POOL
    from concurrent import futures
    _POOL = futures.ThreadPoolExecutor(2)
    return _POOL


def _m21show(obj, fmt=None, wait=False):
    fmt = fmt if fmt is not None else getConfig()['show.format']
    if fmt == 'repr':
        print(repr(obj))
        return
    if wait:
        obj.show(fmt)
    else:
        pool = _getpool()
        pool.submit(obj.show, fmt)


def _ipython_displayhook(cls, func, fmt='image/png'):
    """ 
    Register func as a displayhook for class `cls`
    """
    import IPython
    formatter = IPython.get_ipython().display_formatter.formatters[fmt]
    return formatter.for_type(cls, func)


# ------------------------------------------------------------
#
# music 21
#
# ------------------------------------------------------------

@lib.returns_tuple(["chords", "stream"])
def chords_to_music21(chords, labels=None):
    # type: (Seq[Chord], Opt[Seq[str]]) -> Tuple[List[Chord], music21.stream.Stream]
    """
    This function can be used after calling split_notes_by_amp 
    to generate a music21 stream

    chords: a seq of chords, where each chord is a seq of midinotes
    labels: labels to use for the chords, or None

    Returns: chords (a List of Chords), music21 stream
    
    Example
    =======
    >>> chords = [(60, 63, 65), (40, 45, 48)]
    """
    stream = music21.stream.Stream()
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


def chord_matrix_to_music21(chord_matrix, show=False):
    """
    [
        [chord1_amp1, chord1_amp2, chord1_amp3, ...],
        [chord2_amp1, chord2_amp2, chord2_amp3, ...],
        ...
    ]
    """
    from music21 import stream
    streams = [chords_to_music21(chords).stream for chords in chord_matrix]
    score = stream.Score()
    for s in streams:
        part = stream.Part()
        part.append(s)
        score.insert(0, part)
    if show:
        score.show()
    return score


def show_chords(chords, labels=None, method=None):
    if method is None:
        method = getConfig()['show.format']
    chords2, stream = chords_to_music21(chords, labels)
    stream.show(method)


def show_chord(notes, align='vert', method=None):
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
        method = getConfig()['show.format']
    if align == 'vert':
        stream = music21.stream.Stream()
        notes = [asnote(note) for note in notes]
        chord = Chord(notes)
        stream.append(chord.m21)
        stream.show(method)
    elif align == 'horiz':
        chords = [Chord(n) for n in notes]
        stream = splitchords(chords)
        stream.show(method)
        return stream


def n2m21(note="C4"):
    # type: (Union[str, Note]) -> music21.note.Note
    """
    convert a note (a string or a Note) to a music21 Note
    """
    assert isinstance(note, (str, Note))
    if isinstance(note, Note):
        return note.m21
    elif isinstance(note, str):
        midi = n2m(note)
        return music21.note.Note(midi)
    else:
        raise TypeError("excepted a Note or a notename as str")


def m21_ipythonhook(enable=True) -> None:
    """
    Set an ipython-hook to display music21 objects inline on the
    ipython notebook
    """
    from IPython.core.getipython import get_ipython
    from IPython.core import display
        
    formatter = get_ipython().display_formatter.formatters['image/png']
    if enable:
        def showm21(stream):
            return display.Image(filename=stream.write('lily.png'))._repr_png_()

        dpi = formatter.for_type(music21.Music21Object, showm21)
        return dpi
    else:
        logger.debug("disabling display hook")
        formatter.for_type(music21.Music21Object, None)
        
# ------------------------------------------------------------
#
# notenames
#
# ------------------------------------------------------------


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


def generate_notes(start=12, end=127):
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


def i2r(i, maxdenominator=16):
    """
    Interval to Ratio

    i: an interval in semitones

    Returns
    =======

    a Fraction with the ratio defining this interval
    """
    return Fraction.from_float(i).limit_denominator(maxdenominator)


def notes2ratio(n1, n2, maxdenominator=16):
    """
    find the ratio between n1 and n2

    n1, n2: notes -> "C4", or midinote (do not use frequencies)

    Returns
    =======

    a Fraction with the ratio between the two notes

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
    return i2r(f1/f2, maxdenominator=maxdenominator)


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


def splitchords(chords, splitpoint=60):
    """
    Split a seq. of chords in two staffs to visualize them as 
    a sequence
    """
    assert isinstance(chords, (list, tuple))
    assert all(isinstance(ch, Chord) for ch in chords)
    m21chords = [ch.m21 for ch in chords]
    return m21tools.splitchords(m21chords, split=splitpoint)


# ------------------------------------------------

_init()