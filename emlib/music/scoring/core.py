from __future__ import annotations
import copy
import dataclasses
from fractions import Fraction as F
import uuid

import emlib.typehints as t
from emlib import lib, iterlib



@dataclasses.dataclass
class Annotation:
    text: str
    placement: str = 'above'


def makeId():
    return uuid.uuid4()


class Event:
    def __init__(self, dur:F=None, offset:F=None, annot:str="", db:float=0, dynamic:str=None,
                 articulation:str="", gliss:bool=False, instr:str=None, group:str="",
                 hidden=False):
        """

        :param dur: duration of the event. Can be None.
        :param offset: start time. Can be None
        :param annot: an event can have many annotations. If given at construction time, this
                      will be the first annotations. See addAnnotation
        :param db: dynamic as decibel
        :param dynamic: a dynamic as str ('pp', 'ppp', etc)
        :param articulation: an articulation 'stacatto', 'tenuto', etc
        :param gliss: is this the start of a gliss? A gliss can be internal, from pitch to endpitch,
                      or external, from this event to another event. For an internal gliss.,
                      endpitch should be unset
        :param instr: which instrument should play this event (used for playback)
        :param group: if given, a string used to identify events that belong together. If unser,
                      it is interpreted as not belonging to any group
        """
        # to avoid confusion, optional attributes (dur, offset) will always have a default
        # value set. If they were not given one, we preserve this in the _* variable,
        # which can be accessed via .hasDur, .hasOffset, etc.
        self._dur = dur
        self._offset = offset
        self.dur: F = F(dur) if dur is not None else F(1)
        self.offset: F = F(offset) if offset is not None else F(0)
        self.annots: t.List[Annotation] = None
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
    def end(self):
        return self.offset + self.dur

    def hasDur(self):
        return self._dur is not None

    def hasOffset(self):
        return self._offset is not None

    def getAnnotation(self) -> t.Opt[str]:
        """
        Returns the annotation of this event, or, if the event has multiple
        annotations, these are joined together as one text
        :return:
        """
        if self.annots:
            return ":".join(annot.text for annot in self.annots)
        return None

    def avgPitch(self) -> float:
        pitches = self.getPitches()
        return sum(pitches)/len(pitches) if pitches else 0

    def getPitches(self) -> list[float]:
        raise NotImplementedError()

    def clone(self, **kws):
        out = copy.deepcopy(self)
        for key, value in kws.items():
            setattr(out, key, value)
        return out

    def addAnnotation(self, annot, placement='above'):
        annot = annot if isinstance(annot, Annotation) else Annotation(annot, placement=placement)
        if self.annots is None:
            self.annots = [annot]
        else:
            assert isinstance(self.annots, list)
            self.annots.append(annot)

    def __hash__(self):
        return hash((self.dur, self.offset))

    def isSilence(self):
        return False


class Note(Event):
    def __init__(self, pitch:float, dur:F=None, endpitch:float=0, offset:F=None, **kws):
        super().__init__(dur=dur, offset=offset, **kws)
        assert isinstance(pitch, (int, float))
        self.pitch = pitch
        self.endpitch = endpitch
        if self.endpitch != 0 and self.pitch != self.endpitch:
            self.gliss = True

    @classmethod
    def silence(cls, dur:F, offset:F=0):
        return Note(pitch=0, dur=dur, offset=offset)

    def getPitches(self):
        pitches = [self.pitch]
        if self.endpitch:
            pitches.append(self.endpitch)
        return pitches

    def isGlissInternal(self):
        """ Is this gliss between pitch and endpitch? """
        return self.gliss and self.endpitch and self.pitch != self.endpitch

    def isGlissExternal(self):
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


class Chord(Event):
    def __init__(self, pitches:t.List[float], dur:F=None, offset:F=None,
                 endpitches:t.List[float]=None, **kws):
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
    def __init__(self, events: t.Iter[Event]=None, label:str=None, groupid:str=None):
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

    def __iter__(self) -> t.Iter[Event]:
        return super().__iter__()

    def split(self) -> t.List['Track']:
        return splitEvents(self, groupid=self.groupid)

    def needsSplit(self) -> bool:
        return needsSplit(self)


def _nextInGrid(x: t.U[float, F], ticks: t.List[F]):
    return lib.snap_to_grids(x + F(1, 9999999), ticks, mode='ceil')


def snapTime(note: Event,
             divisors: t.List[int],
             mindur=F(1, 16),
             durdivisors: t.List[Note]=None) -> Event:
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


def fixOverlap(events: t.Seq[Event]) -> t.Seq[Event]:
    """
    Fix overlap between events. If two events overlap,
    the first event is cut, preserving the offset of the
    second event

    :param events: the events to fix
    :return: the fixed events
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


def fillSilences(events: t.Seq[Event], mingap=1/64) -> t.List[Event]:
    """
    Return a list of Notes with silences filled by Notes with pitch set
    to `silentPitch`

    :param events: the notes to fill
    :param mingap: min. gap between two notes. If any notes differ by less
                   than this, the first note absorvs the gap
    :return: a list of new Notes
    """
    assert all(isinstance(ev, Event) for ev in events)
    out: t.List[Event] = []
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


def _makeGroups(events:t.Seq[Event]) -> t.List[t.U[Event, t.List[Event]]]:
    """
    Given a seq. of events, elements which are grouped together are wrapped
    in a list, whereas elements which don't belong to any group are
    appened as is

    :param events:
    :return:
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


def needsSplit(events: t.Seq[Event], threshold=1) -> bool:
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


def splitEvents(events: t.Seq[Event], groupid=None) -> t.List[Track]:
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


def packInTracks(events: t.Seq[Event], maxrange=36) -> t.List[Track]:
    """
    Pack a list of possibly simultaneous events into tracks, where the events
    within one track are NOT simulatenous. Events belonging to the same group
    are kept in the same track.

    Returns a list of Tracks (a Track is a list of Notes)
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

    def unwrapPackingTrack(track: packing.Track) -> t.List[Note]:
        out = []
        for item in track:
            obj = item.obj
            if isinstance(obj, list):
                out.extend(obj)
            else:
                out.append(obj)
        return out

    return [Track(unwrapPackingTrack(track)) for track in tracks]


def centsshown(centsdev, divsPerSemitone=4) -> str:
    """
    Given a cents deviation from a chromatic pitch, return
    a string to be shown along the notation, to indicate the
    true tuning of the note. If we are very close to a notated
    pitch (depending on divsPerSemitone), then we don't show
    anything. Otherwise, the deviation is always the deviation
    from the chromatic pitch

    :param centsdev: the deviation from the chromatic pitch
    :param divsPerSemitone: divisions per semitone
    :return: the string to be shown alongside the notated pitch
    """
    # cents can be also negative (see self.cents)
    divsPerSemitone = divsPerSemitone
    pivot = int(round(100 / divsPerSemitone))
    dist = min(centsdev%pivot, -centsdev%pivot)
    if dist <= 2:
        return ""
    if centsdev < 0:
        # NB: this is not a normal - sign! We do this to avoid it being confused
        # with a syllable separator during rendering (this is currently the case
        # in musescore
        return f"–{-centsdev}"
    return str(int(centsdev))


class AbstractScore:

    def __init__(self, score, pageSize:str='a4', orientation='portrait', staffSize=12,
                 eventTracks: t.List[Track]=None, includeBranch=True) -> None:
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
        events: t.List[Event] = []
        for track in tracks:
            events.extend(track)
        from .play import playNotes
        playNotes(events, defaultinstr=instr)


def makeScore(tracks: t.List[Track], pageSize='a4',
              orientation='portrait', staffSize=12,
              includeBranch=True, backend='music21'
              ) -> AbstractScore:
    """
    Make a score from the list of tracks. Returns either
    an AbsScore or a Music21Score depending on backend

    Args:

        tracks:
            a list of Tracks, as returned by packInTracks(events)
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

    Example:

        notes = [node2note(node) for node in nodes]
        tracks = packInTracks(notes)
        score = makescore(tracks)
        score.save("myscore.pdf")
    """
    from . import quantization
    assert all(isinstance(track, Track) for track in tracks), str(tracks)

    if backend == 'abjad':
        from . import abjscore
        from emlib.music import abjadtools
        voices = [quantization.makeAbjadVoice(track) for track in tracks]
        score = abjadtools.voicesToScore(voices)
        return abjscore.AbjScore(score, eventTracks=tracks, pageSize=pageSize, orientation=orientation,
                                 staffSize=staffSize, includeBranch=includeBranch)
    elif backend == 'music21':
        from emlib.music import m21tools
        from . import music21score
        voices = [quantization.quantizeVoice(track) for track in tracks]
        m21score = m21tools.stackParts(voices)
        return music21score.Music21Score(m21score, tracks=tracks, pageSize=pageSize)
    else:
        raise ValueError(f"Backend {backend} not supported. Possible backends are: abjad, music21")


def quantizeVoice(events: t.Seq[Event], grid="simple", divsPerSemitone=4, showcents=False):
    from . import quantization
    voice = quantization.quantizeVoice(events=events, grid=grid, showcents=showcents,
                                       divsPerSemitone=divsPerSemitone)
    return voice


def midinotesNeedSplit(midinotes, splitpoint=60, margin=4) -> bool:
    if len(midinotes) == 0:
        return False
    numabove = sum(int(m > splitpoint - margin) for m in midinotes)
    numbelow = sum(int(m < splitpoint + margin) for m in midinotes)
    return bool(numabove and numbelow)


def clefNameFromMidinotes(midis: t.List[float]) -> str:
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
