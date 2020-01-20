from __future__ import annotations
import music21 as m21
import abjad as abj
from abjadext import nauert
from emlib.music import m21tools
from emlib.typehints import *
import abjadtools
from emlib.music import m21fix

from .core import *


quantizationSearchTrees = {
    'simple': nauert.UnweightedSearchTree(
        definition={
            2: {
                2: {
                    2: None,
                    # 3: None,
                },
                3: None,
                # 5: None,
                # 7: None,
            },
            3: {
                2: None,
                # 3: None,
                # 5: None,
            },
            5: {
                2: None,
                # 3: None,
            },
            7: None,
            # 7: {
            #    2: None,
            #    },
            # 11: None,
            # 13: None,
        },
    )
}


def getSearchTree(name:str) -> nauert.UnweightedSearchTree:
    searchTree = quantizationSearchTrees.get(name)
    if not searchTree:
        possibleGrids = quantizationSearchTrees.keys()
        raise KeyError(f"grid {name} not known. It should be one of {list(possibleGrids)}")
    return searchTree


def makeAbjadVoice(notes: Seq[Note], grid="simple", glissUseMacros=True,
                   glissSkipSamePitch=True, annotationFontSize=5) -> abj.Voice:
    """
    Create an abjad Voice by quantizing the notes

    Args:
        notes: a seq. of Notes
        grid: XXX
        glissUseMacros: XXX
        glissSkipSamePitch: do not create a gliss. if the gliss would be 
            between notes of same pitch
        annotationFontSize: the font size for text annotations

    Returns:
        an abjad Voice
    """
    assert all(isinstance(note, Note) for note in notes)
    # notes = [snapTime(note, divisors) for note in notes]
    notes = fixOverlap(notes)
    continuousNotes = fillSilences(notes)
    durations = []

    # as of versoin 3, for a QEventSequence, a silence is represented
    # as a note with a duration represented as negative milliseconds
    # see https://github.com/Abjad/abjad-ext-nauert/blob/master/abjadext/nauert/QEventSequence.py
    # In version 2, a silence is represented as a positive dur with None as pitch
    indicateSilenceByNegativeDuration = True
    if indicateSilenceByNegativeDuration:
        for note in continuousNotes:
            millis = int(float(note.dur)*1000)
            if note.isSilence():
                millis = -millis
            durations.append(millis)
        pitches = [float(n.pitch - 60) for n in continuousNotes]
    else:
        # indicate silences by pitch=None
        durations = [int(float(note.dur)*1000) for note in continuousNotes]
        pitches = [None if n.isSilence() else float(n.pitch-60) for n in continuousNotes]
    pitchPairs = list(zip(durations, pitches))
    qseq = nauert.QEventSequence.from_millisecond_pitch_pairs(pitchPairs)

    def fixSilences(seq):
        seq2 = []
        for p in seq:
            if isinstance(p, nauert.PitchedQEvent) and p.pitches[0].number < -50:
                r = nauert.SilentQEvent(offset=p.offset)
                p = r
            seq2.append(p)
        return nauert.QEventSequence(seq2)

    qseq = fixSilences(qseq)
    search_tree = getSearchTree(grid)
    schema = nauert.MeasurewiseQSchema(search_tree=search_tree)
    quantizer = nauert.Quantizer()
    voice = quantizer(qseq, q_schema=schema)
    abjadtools.voiceSetClef(voice)
    attacks = abjadtools.getAttacks(voice)

    if len(attacks) != len(notes):
        print("Length mismatch! Skipping annotation")
        print("Attacks: ", attacks)
        print("Notes: ", notes)
    else:
        annotations = [note.getAnnotation() for note in notes]
        abjadtools.voiceAddAnnotation(voice, annotations, fontSize=annotationFontSize, attacks=attacks)
        abjadtools.voiceAddGliss(voice, [note.gliss for note in notes],
                                 usemacros=glissUseMacros,
                                 skipsame=glissSkipSamePitch,
                                 attacks=attacks)
    return voice


def quantizeAsAbjadRhythm(events: Seq[Event], grid="simple", annot=False, gliss=False
                          ) -> Tup[abj.Voice, Dict[int, Note]]:
    """
    Quantize the rhythm of events, converts it to an abjad voice where
    each note in events in represented by a pitch in ascending order

    The second argument is a mapping pitch->note, so that when iterating over voice,
    it is possible to retrieve the note by doing

    voice, mapping = quantizeAsAbjadRhythm(events)
    for lt in abjad.iterate(voice).logical_ties():
        if lt.is_pitched:
            pitch = lt[0].written_pitch.number
            originalnote = mapping[pitch]
            print(originalnote)
    """
    events = fixOverlap(events)
    continuousEvents = fillSilences(events)
    minpitch = 36
    indices = list(range(minpitch, minpitch+len(events)))
    durations = [int(float(ev.dur)*1000) for ev in continuousEvents]
    iter_indices = iter(indices)
    pitches = [None if n.isSilence() else next(iter_indices)-60 for n in continuousEvents]
    usedpitches = [p for p in pitches if p is not None]
    # assert len(usedpitches) == len(events)
    pitchPairs = list(zip(durations, pitches))
    qseq = nauert.QEventSequence.from_millisecond_pitch_pairs(pitchPairs)
    search_tree = getSearchTree(grid)
    schema = nauert.MeasurewiseQSchema(search_tree=search_tree)
    quantizer = nauert.Quantizer()
    voice = quantizer(qseq, q_schema=schema)
    abjadtools.voiceSetClef(voice)
    attacks = abjadtools.getAttacks(voice)
    if len(attacks) != len(events):
        print("Length mismatch!")
        print("Attacks: ", attacks)
        print("Events: ", events)
    mapping = {pitch+60:event for pitch, event in zip(usedpitches, events)}
    if annot:
        annotations = [note.getAnnotation() for note in events]
        abjadtools.voiceAddAnnotation(voice, annotations, fontSize=5, attacks=attacks)
    if gliss:
        abjadtools.voiceAddGliss(voice, [note.gliss for note in events],
                                 usemacros=False, skipsame=True, attacks=attacks)
    return voice, mapping


def _makeMusic21Voice(notes: Seq[Note], grid="simple") -> m21.stream.Part:
    """
    Create a music21 Part by quantizing these notes

    Args:
        notes: a seq. of Notes
        divisors: possible divisions of the quarter note
        grid: possible subdivisions of the pulse

    Returns:
        the generated music21 Part
    """
    abjvoice = makeAbjadVoice(notes=notes, grid=grid)
    m21voice = abjadtools.abjadToMusic21(abjvoice)
    return m21voice


def makeMusic21Voice(notes: Seq[Note], grid="simple", divsPerSemitone=4,
                     showcents=False) -> m21.stream.Part:
    """
    Create a music21.Voice with the given notes

    Args:
        notes: the notes in the part
        grid: XXX
        divsPerSemitone: max. subdivision of the semitone
        showcents: should we show the cents deviation as text?

    Returns:
        the generated music21 Part
    """
    if divsPerSemitone <= 2:
        return _makeMusic21Voice(notes, grid=grid)
    oversampling = 100
    notes2 = [n.clone(pitch=n.pitch*oversampling) for n in notes]
    m21part = _makeMusic21Voice(notes2, grid=grid)
    tied = False
    from emlib.music import m21tools
    for note in m21part.flat.getElementsByClass(m21.note.Note):
        midinote = round(note.pitch.ps/oversampling, 5)
        pitch, centsdev = m21tools.makePitch(midinote, divsPerSemitone=divsPerSemitone)
        note.pitch = pitch
        if showcents and not tied:
            annotation = centsshown(centsdev, divsPerSemitone=divsPerSemitone)
            # we put a space to fix a bug in conversion to lilypond, where if a "syllable"
            # is missing, all subsequent syllables shift to the left and alignment is broken
            # FIX: a better way would be not to use lyrics but text annotations
            note.lyric = annotation or " "
        tied = note.tie is not None and note.tie.type in ('start', 'continue')
    return m21part


def quantizeVoice(events: Seq[Event], grid="simple", divsPerSemitone=4,
                  showcents=False, showgliss=True, verify=True
                  ) -> m21.stream.Stream:
    """
    This is the center of this module: translate a series of non-overlapping
    scoring.Events (notes, chords) to a voice (a part) in musical notation
    (music21)

    Args:
        events: a seq. of scoring.Events
        grid: the grid used to quantize the divisions of the beat. One of {'simple'}
        divsPerSemitone: the number of divisions per semitone (4 = 1/8 tones)
        showcents: show the cents deviation to the chromatic pitch under the note
        showgliss: generate glissando lines when indicated by the given Events
        verify: verify the quantization

    Returns: 
        the generated music21 stream
    """
    abjvoice, mapping = quantizeAsAbjadRhythm(events, grid=grid)

    # Make index
    # _partMakeIndex(part, mapping)
    part = abjadtools.abjadToMusic21(abjvoice)
    for note in part.flat.getElementsByClass('Note'):
        # at this moment the mapped part consists only of notes, ordered chromatically
        event = _getEvent(note, mapping)
        if not event:
            raise Exception("??? {note}")
        note.editorial.origEvent = event
    if verify:
        assert all(note.editorial.origEvent is not None for note in part.getElementsByClass('Note'))

    m21fix.fixStream(part)
    _mappedPartApplyPitches(part, divsPerSemitone, showcents=showcents, inPlace=True)
    _mappedPartAnnotate(part, inPlace=True)
    if showgliss:
        _mappedPartApplyGliss(part, inPlace=True)
    _mappedPartRemoveIndex(part)
    return part


def quantizeVoiceSplit(events: Seq[Event], grid="simple", divsPerSemitone=4,
                       showcents=False, showgliss=True,
                       ) -> List[m21.stream.Stream]:
    """
    The same as quantizeVoice, but splits the events across staves if necessary

    Args:
        events: the events to quantize
        grid: one of XXX TODO: add documentation
        divsPerSemitone: divisions of the semitone (4 equals 1/8 tones)
        showcents: show cents deviations as annotations, when necessary
        showgliss: generate glissando lines when indicated by the given Events

    Returns:
        a list of generated parts (each part is a m21 stream)

    """
    above, below = [], []
    for ev in events:
        (above if ev.avgPitch() >= 60 else below).append(ev)
    parts = [quantizeVoice(part, grid=grid, divsPerSemitone=divsPerSemitone,
                           showcents=showcents, showgliss=showgliss)
             for part in (above, below) if part]
    return parts


def _getEvent(m21note:m21.note.Note, mapping:Dict[int, Event]
              ) -> Opt[Event]:
    if m21note.editorial.get('comment') == 'unmapped':
        return None
    index = int(m21note.pitch.ps)
    return mapping.get(index)


def _origEvent(m21obj:m21.note.NotRest) -> Opt[Event]:
    event = m21obj.editorial.origEvent
    if not event:
        raise Exception(f"Event not found: {m21obj}")
    return event


def _partMakeIndex(part: m21.stream.Stream, mapping: Dict[int, Event]
                   ) -> None:
    """
    Add a reference to the original scoring note responsible for each 
    m21 note within the note itself, as an editorial addition

    See quantizeVoice
    
    Args:

        part: the part as returned by quantizeAsAbjadRhythm / abjadToMusic21
        mapping:  
        
    Returns:
    """
    for note in part.flat.getElementsByClass('Note'):
        # at this moment, the mapped part consists only of notes, ordered
        # chromatically
        event = _getEvent(note, mapping)
        if not event:
            raise Exception("??? {note}")
        note.editorial.origEvent = event
    assert all(note.editorial.origEvent is not None for note in part.getElementsByClass('Note'))


def _mappedPartRemoveIndex(part:m21.stream.Stream) -> None:
    for event in part.flat.getElementsByClass(m21.note.NotRest):
        event.editorial.pop('origEvent', None)


def _mappedPartAnnotate(part: m21.stream.Stream, inPlace=False) -> m21.stream.Stream:
    part = part if inPlace else copy.deepcopy(part)
    attacks = m21tools.getAttacks(part.flat)
    for note in attacks:
        event = _origEvent(note)
        if not event:
            continue
        if event.annots:
            for annot in event.annots:
                m21tools.addTextExpression(note, text=annot.text, placement=annot.placement)
        if event.dynamic:
            m21tools.addDynamic(note, event.dynamic)
    return part


def _mappedPartApplyPitches(part: m21.stream.Stream, divsPerSemitone=4,
                            showcents=False, inPlace=False
                            ) -> m21.stream.Stream:
    """
    Args:
        part: the mapped part, where the pitches represent the indices
            to events in mapping
        divsPerSemitone: divisions per semitone
        inPlace: if True, operate on part itself

    Returns:
        the processed stream
    """
    if not inPlace:
        part = copy.deepcopy(part)
    assert isinstance(part, m21.stream.Stream)
    partflat : m21.stream.Stream = part.flat
    tiedFromBefore = False
    EMPTY_ANNOTATION = " "
    m21note: m21.note.Note
    for m21note in partflat.getElementsByClass(m21.note.Note):
        event = _origEvent(m21note)
        if not event:
            continue
        if isinstance(event, Note):
            midinote = event.pitch
            pitch, centsdev = m21tools.makePitch(midinote, divsPerSemitone=divsPerSemitone)
            m21note.pitch = pitch
            if tiedFromBefore:
                m21note.pitch.accidental.displayStatus = False
            if showcents and not tiedFromBefore:
                annotation = centsshown(centsdev, divsPerSemitone=divsPerSemitone)

                # we put a space to fix a bug in conversion to lilypond, where if a "syllable"
                # is missing, all subsequent syllables shift to the left and alignment is broken
                # TODO: better use text annotations instead of lyrics
                m21note.lyric = annotation if annotations is not None else EMPTY_ANNOTATION
            if event.hidden:
                m21tools.hideNotehead(m21note)
        elif isinstance(event, Chord):
            chord, centsdevs = m21tools.makeChord(event.pitches, duration=m21note.duration,
                                                  showcents=showcents and not tiedFromBefore)
            chord.editorial.origEvent = event
            chord.tie = m21note.tie
            if tiedFromBefore:
                for pitch in chord.pitches:
                    pitch.accidental.displayStatus = False
            if event.hidden:
                m21tools.hideNotehead(chord)
            m21note.getContextByClass('Measure').replace(m21note, chord)
            # m21note.getContextByClass('Part').replace(m21note, chord)
        tiedFromBefore = m21note.tie is not None and m21note.tie.type in ('start', 'continue')
    return part


def _glissInternal(tie, context, hideTied=True):
    if not tie:
        return
    event = _origEvent(tie[0])
    if not event or not event.gliss:
        return
    if isinstance(event, Note):
        gracenote = m21tools.addGraceNote(event.endpitch, anchorNote=tie[-1], nachschlag=True)
        # we add this to identify the grace note as unmapped, meaning that it has no 'origEvent'
        # stored in its editorial attribute
        gracenote.editorial.comment = "unmapped"
        m21tools.addGliss(tie[0], gracenote, stream=context)
    elif isinstance(event, Chord):
        gracenote = m21tools.addGraceNote(event.endpitches, anchorNote=tie[-1], nachschlag=True)
        gracenote.editorial.comment = "unmapped"
        m21tools.addGliss(tie[0], gracenote, stream=context)
    else:
        raise TypeError(f"type: {type(event)}")
    if hideTied and len(tie) > 1:
        for event in tie[1:]:
            m21tools.hideNotehead(event, hideAccidental=True)


def _mappedPartApplyGliss(part: m21.stream.Stream, hideTied=True,
                          inPlace=False) -> m21.stream.Stream:
    part = part if inPlace else copy.deepcopy(part)
    context = part.getContextByClass('Part')
    partflat = part.flat
    logicalTies = m21tools.logicalTies(partflat)

    if len(logicalTies) == 1:
        tie = logicalTies[0]
        _glissInternal(tie, context, hideTied=hideTied)
        return part

    for tie0, tie1 in iterlib.pairwise(logicalTies):
        attack0 = tie0[0]
        event = _origEvent(attack0)
        if not event or not event.gliss:
            continue
        if isinstance(event, (Note, Chord)):
            if event.isGlissInternal():
                _glissInternal(tie0, context, hideTied=hideTied)
            else:
                m21tools.addGliss(tie0[0], tie1[0], stream=context)
                if hideTied and len(tie0) > 1:
                    for note in tie0[1:]:
                        m21tools.hideNotehead(note, hideAccidental=True)

    _glissInternal(logicalTies[-1], context)
    return part


def m21FromEvents(events: list[Event], split=False, showCents=False,
                  showGliss=False, divsPerSemitone=4
                  ) -> m21.stream.Stream:
    """
    Creates a m21 stream from the events. If split is False, one Voice is created
    Otherwise events are split across as many staves as needed

    Args:
        events: the events to convert
        split: if True, split the events into as many staffs as needed
        showCents: show cents near the note
        showGliss: if True, show a glissando line when needed
        divsPerSemitone: divisions of the semitone

    Returns:
        a music21 Stream, containing a Voice or Score containing multiple
        voices

    """
    if split and midinotesNeedSplit([e.avgPitch() for e in events]):
        parts = quantizeVoiceSplit(events, divsPerSemitone=divsPerSemitone,
                                         showcents=showCents, showgliss=showGliss)
        return m21tools.stackParts(parts)
    return quantizeVoice(events,
                         divsPerSemitone=divsPerSemitone,
                         showcents=showCents,
                         showgliss=showGliss)
