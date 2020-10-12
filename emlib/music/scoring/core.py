from __future__ import annotations
import copy
import dataclasses
from fractions import Fraction as F
import uuid

from emlib.typehints import Opt, U, Seq, Iter
from emlib import lib, iterlib
from emlib.pitchtools import n2m, m2n, parse_midinote

import logging
logger = logging.getLogger("emlib.scoring")

pitch_t  = U[int, float, str]
number_t = U[int, float]
time_t = U[F, int, float]


@dataclasses.dataclass
class Annotation:
    text: str
    placement: str = 'above'
    fontSize: int = None


def makeId() -> uuid.UUID:
    return uuid.uuid4()


class Event:
    def __init__(self, dur:time_t=None, offset:number_t=None, annot:str="", db:float=0., dynamic:str=None,
                 articulation:str="", gliss:bool=False, instr:str=None, group:str="",
                 hidden=False):
        """
        
        Args:
            dur: duration of the event. Can be None.
            offset: start time. Can be None
            annot: an event can have many annotations. If given at construction time, this
                   will be the first annotations. See addAnnotation
            db: dynamic as decibel
            dynamic: a dynamic as str ('pp', 'ppp', etc)
            articulation: an articulation 'stacatto', 'tenuto', etc
            gliss: is this the start of a gliss? A gliss can be internal, from pitch to endpitch,
                      or external, from this event to another event. For an internal gliss.,
                      endpitch should be unset
            instr: which instrument should play this event (used for playback)
            group: if given, a string used to identify events that belong together. If unset,
                      it is interpreted as not belonging to any group
        """
        # to avoid confusion, optional attributes (dur, offset) will always have a default
        # value set. Call hasDur or hasOffset to check if the values were given
        self._hasDur = dur is not None
        self._dur: F = F(dur) if dur is not None else F(1)
        self._hasOffset = offset is not None
        self._offset: F = F(offset) if offset is not None else F(0)
        self.annots: Opt[list[Annotation]] = None
        self.dynamic: str = dynamic
        self.db = db if db is not None else 0
        self.articulation = articulation
        self.gliss = gliss
        self.instr = instr
        self.group = group
        self.hidden = hidden
        if annot:
            self.addAnnotation(annot)

    @property
    def offset(self) -> F:
        return self._offset

    @offset.setter
    def offset(self, value: Opt[F]) -> None:
        self._offset = F(value) if value is not None else F(0)
        self._hasOffset = value is not None

    @property
    def dur(self) -> F:
        return self._dur

    @dur.setter
    def dur(self, value: Opt[F]) -> None:
        self._dur = F(value) if value is not None else F(0)
        self._hasDur = value is not None

    @property
    def end(self):
        return self.offset + self.dur

    def hasDur(self):
        return self._hasDur

    def hasOffset(self):
        return self._hasOffset

    def getAnnotation(self) -> Opt[str]:
        """
        Returns the annotation of this event, or, if the event has multiple
        annotations, these are joined together as one text
        """
        if self.annots:
            return ":".join(annot.text for annot in self.annots)
        return None

    def avgPitch(self) -> float:
        pitches = self.getPitches()
        return sum(pitches)/len(pitches) if pitches else 0

    def getPitches(self) -> list[float]:
        raise NotImplementedError()

    def clone(self, **kws) -> Event:
        out = copy.deepcopy(self)
        for key, value in kws.items():
            setattr(out, key, value)
        return out

    def addAnnotation(self, annot: U[str, Annotation], placement='above', fontSize:int=None) -> None:
        annot = annot if isinstance(annot, Annotation) \
            else Annotation(annot, placement=placement, fontSize=fontSize)
        if self.annots is None:
            self.annots = [annot]
        else:
            assert isinstance(self.annots, list)
            self.annots.append(annot)

    def __hash__(self):
        return hash((self.dur, self.offset))

    def isSilence(self):
        return False

    def show(self, **kws):
        return Track([self]).show(**kws)


class Note(Event):
    def __init__(self, pitch:pitch_t, dur:time_t=None, endpitch:pitch_t=0,
                 offset:time_t=None, annot:str=None, **kws):
        """
        
        Args:
            pitch: the pitch of this Note. 0=silence 
            dur: the duration of this Note. 
            endpitch: the end pitch if this is a gliss.
            offset: the start time of the Note
            annot: a text annotation for this note
            **kws: any other keyword passed to Event (db, dynamic, articulation)
        """
        super().__init__(dur=dur, offset=offset, annot=annot, **kws)
        assert isinstance(pitch, (int, float, str))
        self.pitch = _asmidi(pitch)
        self.endpitch = _asmidi(endpitch)
        if self.endpitch != 0 and self.pitch != self.endpitch:
            self.gliss = True

    @classmethod
    def silence(cls, dur:time_t, offset:time_t=None) -> Note:
        return Note(pitch=0, dur=dur, offset=offset)

    def getPitches(self) -> list[float]:
        pitches = [self.pitch]
        if self.endpitch:
            pitches.append(self.endpitch)
        return pitches

    def isGlissInternal(self) -> bool:
        """ Is this gliss between pitch and endpitch? """
        return self.gliss and self.endpitch and self.pitch != self.endpitch

    def isGlissExternal(self) -> bool:
        """ Is this gliss between this event and the next? """
        return self.gliss and (self.endpitch == 0 or self.pitch == self.endpitch)

    def __repr__(self):
        frac = lambda x: f"{float(x):.3f}"
        ss = [f"pitch={self.pitch:.2f}, offset={frac(self.offset)}, dur={frac(self.dur)}"]
        if self.db < 0:
            ss.append(f", db={float(self.db):.1f}")
        if self.endpitch>0 and self.endpitch != self.pitch:
            ss.append(f", endpitch={self.endpitch:.2f}")
        if self.gliss:
            ss.append(f", gliss!")
        if self.annots:
            ss.append(f", {self.annots}")
        s = "".join(ss)
        return f"Note({s})"

    def isSilence(self) -> bool:
        """Is this note a silence?"""
        return self.pitch == 0

    def __hash__(self):
        return hash((self.pitch, self.dur, self.endpitch, self.offset))


def _asmidi(x: U[pitch_t, Note]) -> float:
    if isinstance(x, str):
        return n2m(x)
    elif isinstance(x, int):
        x = float(x)
    elif isinstance(x, Note):
        return x.pitch
    if not (0 <= x < 128):
        raise ValueError(f"x should be a midi note, got {x}")
    return x


class Chord(Event):
    def __init__(self, pitches:list[pitch_t], dur:time_t=None, offset:time_t=None,
                 endpitches:list[pitch_t]=None, **kws):
        super().__init__(dur=dur, offset=offset, **kws)
        assert all(isinstance(p, (float, int)) for p in pitches)
        self.pitches = pitches
        self.endpitches = endpitches
        if self.endpitches:
            self.gliss = True

    def __repr__(self):
        ss = [str(self.pitches)]
        if self.db < 0:
            ss.append(f", db={float(self.db):.1f}")
        if self.endpitches:
            ss.append(f", endpitches={self.endpitches}")
        if self.dur:
            ss.append(f", dur={self.dur}")
        if self.offset:
            ss.append(f", offset={self.offset}")
        if self.gliss:
            ss.append(f", gliss!")
        if self.annots:
            ss.append(f", {self.annots}")
        s = "".join(ss)
        return f"Chord({s})"

    def getPitches(self) -> list[float]:
        return self.pitches

    def isSilence(self) -> bool:
        return not self.pitches

    def __hash__(self):
        data = (float(self.dur), float(self.offset), *self.pitches)
        if self.endpitches:
            data += tuple(self.endpitches)
        return hash(data)

    def isGlissInternal(self):
        return self.gliss and self.endpitches != self.pitches

    def isGlissExternal(self):
        return self.gliss and not self.endpitches


class Track(list):
    def __init__(self, events: Iter[Event]=None, label:str=None, groupid:str=None):
        """
        A Track is a list of non-simultaneous events (a Part)

        Args:
            events: the events (notes, chords) in this track
            label: a label to identify this track in particular (a name)
            groupid: an identification (given by makeId), used to identify
                tracks which belong to a same group
        """
        if events:
            super().__init__(events)
        else:
            super().__init__()
        self.groupid:str = groupid
        self.label:str = label

    def __getitem__(self, item) -> Event:
        return super().__getitem__(item)

    def __iter__(self) -> Iter[Event]:
        return super().__iter__()

    def split(self) -> list['Track']:
        return splitEvents(self, groupid=self.groupid)

    def needsSplit(self) -> bool:
        return needsSplit(self)

    def show(self, **kws):
        score = makeScore([self], **kws)
        return score.show()


def stackEventsInplace(events: list[Event], start=0, overrideOffset=False) -> None:
    """
    This function stacks events together by placing an event at the end of the
    previous event whenever an event does not define its own offset

    Args:
        events: a list of events or a Track
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined

    Returns:
        a list of stacked events
    """
    now = events[0].offset if events[0].hasOffset() else start
    assert now is not None and now>=0
    for ev in events:
        if not ev.hasDur():
            ev.dur = ev.dur  # set duration explicitely

        if not ev.hasOffset() or overrideOffset:
            ev.offset = now
        now += ev.dur
    for ev1, ev2 in iterlib.pairwise(events):
        assert ev1.offset<ev2.offset


def stackEvents(events: list[Event], start=0., overrideOffset=False) -> list[Event]:
    """
    This function stacks events together by placing an event at the end of the
    previous event whenever an event does not define its own offset

    Args:
        events: a list of events
        start: the start time, will override the offset of the first event
        overrideOffset: if True, offsets are overriden even if they are defined

    Returns:
        a list of stacked events
    """
    now = events[0].offset if events[0].hasOffset() else start
    assert now is not None and now >= 0
    out = []
    for ev in events:
        assert ev.hasDur()
        if not ev.hasDur() or not ev.hasOffset() or overrideOffset:
            ev = ev.clone(offset=now, dur=ev.dur)
        now += ev.dur
        out.append(ev)
    for ev1, ev2 in iterlib.pairwise(out):
        assert ev1.offset < ev2.offset
    return out


def _nextInGrid(x: U[float, F], ticks: list[F]):
    return lib.snap_to_grids(x + F(1, 9999999), ticks, mode='ceil')


def snapTime(note: Event,
             divisors: list[int],
             mindur=F(1, 16),
             durdivisors: list[Note]=None) -> Event:
    """
    Quantize note's start and end to snap to a grid defined by divisors and durdivisors

    note:
        the note to be quantized
    divisors:
        a list of divisions of the pulse
    mindur:
        the min. duration of the note
    durdivisors:
        if left unspecified, then the same list of divisors is used for start
        and end of the note. Otherwise, it is possible to define a specific
        grid for the end also

    Returns a new Note with its offset and end quantized to the given grids
    """
    if durdivisors is None:
        durdivisors = divisors
    ticks = [F(1, div) for div in divisors]
    durticks = [F(1, div) for div in durdivisors]
    start = lib.snap_to_grids(note.offset, ticks)
    end = lib.snap_to_grids(note.offset + note.dur, durticks)
    if end - start <= mindur:
        end = _nextInGrid(start + mindur, ticks)
    return note.clone(offset=start, dur=end-start)


def fixOverlap(events: Seq[Event]) -> Seq[Event]:
    """
    Fix overlap between events. If two events overlap,
    the first event is cut, preserving the offset of the
    second event

    Args:
        events: the events to fix

    Returns:
        the fixed events
    """
    if len(events) < 2:
        return events
    out = []
    for n0, n1 in iterlib.pairwise(events):
        assert n0.offset < n1.offset, "Notes are not sorted!"
        if n1.offset < n0.offset+n0.dur:
            n0 = n0.clone(dur=n1.offset - n0.offset)
        out.append(n0)
    out.append(events[-1])
    return out


def fillSilences(events: Seq[Event], mingap=1/64) -> list[Event]:
    """
    Return a list of Notes with silences filled by Notes with pitch set
    to `silentPitch`

    Args:
        events: the notes to fill
        mingap: min. gap between two notes. If any notes differ by less
                   than this, the first note absorvs the gap
    Returns:
        a list of new Notes
    """
    assert events
    assert all(isinstance(ev, Event) for ev in events)
    out: list[Event] = []
    if events[0].offset > 0:
        out.append(Note.silence(offset=F(0), dur=events[0].offset))
    for ev0, ev1 in iterlib.pairwise(events):
        gap: F = ev1.offset - (ev0.offset + ev0.dur)
        assert gap >= 0, f"negative gap! = {gap}"
        if gap > mingap:
            out.append(ev0)
            rest = Note.silence(offset=ev0.offset+ev0.dur, dur=gap)
            out.append(rest)
        else:
            # adjust the dur of n0 to match start of n1
            out.append(ev0.clone(dur=ev1.offset - ev0.offset))
    out.append(events[-1])
    uncontiguous = [(n0, n1) for n0, n1 in iterlib.pairwise(out) if n0.offset+n0.dur != n1.offset ]
    if uncontiguous:
        for pair in uncontiguous:
            print(pair)
    return out


def _makeGroups(events:Seq[Event]) -> list[U[Event, list[Event]]]:
    """
    Given a seq. of events, elements which are grouped together are wrapped
    in a list, whereas elements which don't belong to any group are
    appended as is

    """
    out = []
    for groupid, elementsiter in iterlib.groupby(events, key=lambda event:event.group):
        if not groupid:
            out.extend(elementsiter)
        else:
            elements = list(elementsiter)
            elements.sort(key=lambda elem: elem.offset)
            out.append(elements)
    return out


def needsSplit(events: Seq[Event], threshold=1) -> bool:
    G, F, G15a = 0, 0, 0
    allpitches = sum((ev.getPitches() for ev in events), [])
    for pitch in allpitches:
        if 55 < pitch <= 93:
            G += 1
        elif 93 < pitch:
            G15a += 1
        else:
            F += 1
    numExceeded = sum(int(numnotes > threshold) for numnotes in (G, F, G15a))
    return numExceeded > 1


def splitEvents(events: Seq[Event], groupid=None) -> list[Track]:
    """
    Assuming that events are not simultenous, split the events into
    different tracks if the range makes it necessary

    Args:
        events: he events to split
        groupid: if given, this id will be used to identify the
            generated tracks (see makeId)

    Returns:
         list of Tracks (between 1 and 3, one for each clef)
    """
    G = []
    F = []
    G15a = []

    for event in events:
        if isinstance(event, Note):
            pitch = event.avgPitch()
            if 55 < pitch <= 93:
                G.append(event)
            elif 93 < pitch:
                G15a.append(event)
            else:
                F.append(event)
        elif isinstance(event, Chord):
            chordG = []
            chordF = []
            chord15a = []
            for pitch in event.pitches:
                if 55 < pitch <= 93:
                    chordG.append(pitch)
                elif 93 < pitch:
                    chord15a.append(pitch)
                else:
                    chordF.append(pitch)
            if chordG:
                G.append(event.clone(pitches=chordG))
            if chordF:
                F.append(event.clone(pitches=chordF))
            if chord15a:
                G15a.append(event.clone(pitches=chord15a))
        else:
            raise TypeError(f"Object not supported for splitting: {event} {type(event)}")
    groupid = groupid or makeId()
    tracks = [Track(track, groupid=groupid, label=name)
              for track, name in ((G15a, "G15a"), (G, "G"), (F, "F")) if track]
    return tracks


def packInTracks(events: Seq[Event], maxrange=36) -> list[Track]:
    """
    Pack a list of possibly simultaneous events into tracks, where the events
    within one track are NOT simulatenous. Events belonging to the same group
    are kept in the same track.

    Returns a list of Tracks (a Track is a list of Events)
    """
    from emlib.music import packing
    items = []
    groups = _makeGroups(events)
    for group in groups:
        if isinstance(group, Event):
            event = group
            if event.isSilence():
                continue
            item = packing.Item(obj=event, offset=event.offset, dur=event.dur, step=event.avgPitch())
        elif isinstance(group, list):
            dur = group[-1].end - group[0].offset
            step = sum(event.avgPitch() for event in group)/len(group)
            item = packing.Item(obj=group, offset=group[0].offset, dur=dur, step=step)
        else:
            raise TypeError(f"Expected an Event or a list thereof, got {type(group)}")
        items.append(item)

    tracks = packing.pack_in_tracks(items, maxrange=maxrange)

    def unwrapPackingTrack(track: packing.Track) -> list[Note]:
        out = []
        for item in track:
            obj = item.obj
            if isinstance(obj, list):
                out.extend(obj)
            else:
                out.append(obj)
        return out

    return [Track(unwrapPackingTrack(track)) for track in tracks]


def centsshown(centsdev: int, divsPerSemitone=4, snap=2) -> str:
    """
    Given a cents deviation from a chromatic pitch, return
    a string to be shown along the notation, to indicate the
    distance to its corresponding microtone. If we are very
    close to a notated pitch (depending on divsPerSemitone),
    then we don't show anything.

    Args:
        centsdev: the deviation from the chromatic pitch
        divsPerSemitone: divisions per semitone
        snap: if the difference to the microtone is within this error,
            we "snap" the pitch to the microtone

    Returns:
        the string to be shown alongside the notated pitch

    Example:
        centsshown(55, divsPerSemitone=4)
            "5"
    """
    # cents can be also negative (see self.cents)
    divsPerSemitone = divsPerSemitone
    pivot = int(round(100 / divsPerSemitone))
    dist = min(centsdev%pivot, -centsdev%pivot)
    if dist <= snap:
        return ""
    if centsdev < 0:
        # NB: this is not a normal - (minus) sign! We do this to avoid it being confused
        # with a syllable separator during rendering (this is currently the case
        # in musescore
        return f"–{-centsdev}"
    return str(int(centsdev))


class AbstractScore:

    def __init__(self, score, pageSize:str='a4', orientation='portrait', staffSize=12,
                 eventTracks: list[Track]=None, includeBranch=True) -> None:
        """
        Args:
            score: the internal representation of the score according to the backend
            pageSize: size of the page, such as 'a4' or 'a3'
            orientation: 'portrait' or 'landscape'
            staffSize: staff size in points
            includeBranch: ???
        """
        self.score = score
        self.pageSize: str = pageSize or 'a4'
        self.orientation: str = orientation or 'portrait'
        self.staffSize: int = staffSize or 12
        self.includeBranch: bool = includeBranch if includeBranch is not None else True
        self.eventTracks = eventTracks

    def show(self) -> None: ...
    def dump(self) -> None: ...
    def save(self, outfile) -> None: ...
    def play(self, instr=None):
        tracks = self.eventTracks
        events: list[Event] = []
        for track in tracks:
            events.extend(track)
        from .play import playNotes
        playNotes(events, defaultinstr=instr)


def makeScore(tracks: list[Track],
              pageSize='a4',
              orientation='portrait',
              staffSize=12,
              includeBranch=True,
              backend='music21',
              stackTracks=True,
              title="",
              composer=""
              ) -> AbstractScore:
    """
    Make a score from the list of tracks. Returns either
    an AbjScore or a Music21Score depending on backend

    Args:

        tracks:
            a list of Tracks, as returned for example by packInTracks(events)
        pageSize:
            a string like 'a4', 'a3', etc.
        orientation:
            'landscape' or 'portrait'
        staffSize:
            staff size
        backend:
            the backend to use (abjad only at the moment)
        includeBranch:
            add branck to the score
        stackTracks: if True, stackEvents is called on each track (see stackEvents)
        title: the title of the score
        composer: if given, add a composer annotation

    """
    from . import quantization
    assert all(isinstance(track, Track) for track in tracks), str(tracks)
    if stackTracks:
        tracks = [Track(stackEvents(track)) for track in tracks]

    if backend == 'abjad':
        from . import abjscore
        from emlib.music import abjadtools
        voices = [quantization.makeAbjadVoice(track) for track in tracks]
        score = abjadtools.voicesToScore(voices)
        return abjscore.AbjScore(score, eventTracks=tracks, pageSize=pageSize,
                                 orientation=orientation, staffSize=staffSize,
                                 includeBranch=includeBranch)
    elif backend == 'music21':
        from emlib.music import m21tools
        from . import music21score
        voices = [quantization.quantizeVoice(track) for track in tracks]
        m21score = m21tools.stackParts(voices)
        m21tools.scoreSetMetadata(m21score, title=title, composer=composer)
        return music21score.Music21Score(m21score, tracks=tracks, pageSize=pageSize)
    else:
        raise ValueError(f"Backend {backend} not supported. Possible backends: abjad, music21")


def midinotesNeedSplit(midinotes, splitpoint=60, margin=4) -> bool:
    if len(midinotes) == 0:
        return False
    numabove = sum(int(m > splitpoint - margin) for m in midinotes)
    numbelow = sum(int(m < splitpoint + margin) for m in midinotes)
    return bool(numabove and numbelow)


def clefNameFromMidinotes(midis: list[float]) -> str:
    """
    Given a list of midinotes, return the best clef to
    fit these notes

    """
    if not midis:
        return "treble"
    avg = sum(midis)/len(midis)
    if avg>80:
        return "treble8va"
    elif avg>58:
        return "treble"
    elif avg>36:
        return "bass"
    else:
        return "bass8vb"


def roundMidinote(a: float, divsPerSemitone=4) -> float:
    rounding_factor = 1/divsPerSemitone
    return round(a/rounding_factor)*rounding_factor


def hasSameAccidental(a: U[pitch_t, Note], b:U[pitch_t, Note], divsPerSemitone=4) -> bool:
    apitch = _asmidi(a)
    bpitch = _asmidi(b)
    return roundMidinote(apitch, divsPerSemitone) == roundMidinote(bpitch, divsPerSemitone)
