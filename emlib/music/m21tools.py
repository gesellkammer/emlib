from __future__ import annotations
import os

import music21 as m21

from emlib import iterlib
from emlib import typehints as t
from emlib import lib

from emlib.pitchtools import n2m, m2n, split_notename, split_cents
from emlib.music import m21fix


def _splitchord(chord: m21.chord.Chord, 
                partabove: m21.stream.Part, 
                partbelow: m21.stream.Part, 
                split=60) -> t.Tup[t.List[m21.note.Note], t.List[m21.note.Note]]:
    above, below = [], []
    for i in range(len(chord)):
        note = chord[i]
        if note.pitch.midi >= split:
            above.append(note)
        else:
            below.append(note)

    def addnotes(part, notes, lyric=None):
        if not notes:
            rest = m21.note.Rest()
            if lyric:
                rest.lyric = lyric
            part.append(rest)
        else:
            ch = m21.chord.Chord(notes)
            lyric = "\n".join(note.lyric for note in reversed(notes) 
                              if note.lyric is not None)
            if lyric:
                ch.lyric = lyric
            part.append(ch)

    print("Chord lyric before split: ", chord.lyric)
    addnotes(partabove, above)
    addnotes(partbelow, below, lyric=chord.lyric)
    return above, below


def _asmidi(x) -> float:
    if isinstance(x, str):
        return n2m(x)
    elif isinstance(x, (int, float)):
        assert 0 <= x < 128, f"Expected a midinote (between 0-127), but got {x}"
        return x
    raise TypeError(f"Expected a midinote as number of notename, but got {x} ({type(x)})")


def logicalTies(stream:m21.stream.Stream) -> t.List[t.List[m21.note.NotRest]]:
    out = []
    current = []
    events = list(stream.getElementsByClass(m21.note.NotRest))
    if len(events) == 1:
        return [events]
    for ev0, ev1 in iterlib.pairwise(events):
        # if n0.pitch.ps != n1.pitch.ps or n0.tie is None:
        if ev0.tie is None:
            current.append(ev0)
            out.append(current)
            current = []
        elif ev0.tie is not None and ev0.tie.type in ('start', 'continue'):
            current.append(ev0)
    if events:
        current.append(events[-1])
    out.append(current)
    return out
    

def splitChords(chords: t.Seq[m21.chord.Chord], split: int=60, force=False) -> m21.stream.Score:
    """
    split a seq. of music21 Chords in two staffs
    """
    assert isinstance(split, (int, float))
    for chord in chords:
        assert isinstance(chord, m21.chord.Chord)
    partabove = m21.stream.Part()
    partabove.append(m21.clef.TrebleClef())
    partbelow = m21.stream.Part()
    partbelow.append(m21.clef.BassClef())
    allabove = []
    allbelow = []
    for chord in chords:
        above, below = _splitchord(chord, partabove=partabove, partbelow=partbelow, split=split)
        allabove.extend(above)
        allbelow.extend(below)
    parts = []
    if allabove or force:
        parts.append(partabove)
    if allbelow or force:
        parts.append(partbelow)
    return m21.stream.Score(parts)


def isTiedToPrevious(note:m21.note.Note) -> bool:
    """
    Is this note tied to the previous note?
    """
    prev = note.previous('Note')
    if not prev:
        return False
    tie: m21.tie.Tie = prev.tie
    return tie is not None and tie.type in ('start', 'continued')


def getAttacks(stream: m21.stream.Stream) -> t.List[m21.note.NotRest]:
    ties = logicalTies(stream)
    attacks = []
    for tie in logicalTies(stream):
        if tie:
            attacks.append(tie[0])
    return attacks
    

def splitVoice(voice: m21.stream.Stream, split: int=60) -> m21.stream.Score:
    """
    split a music21 Voice in two staffs
    """
    above = []
    below = []
    for obj in voice:
        if obj.isClassOrSubclass((m21.note.GeneralNote,)):
            above.append(obj)
            continue
        rest = m21.note.Rest(duration=obj.duration)
        if isinstance(obj, m21.note.Rest):
            above.append(obj)
            below.append(obj)
        else:
            if obj.pitch.midi >= split:
                above.append(obj)
                below.append(rest)
            else:
                below.append(obj)
                above.append(rest)
    partabove = m21.stream.Part()
    partabove.append(bestClef(above))
    for obj in above:
        partabove.append(obj)
    
    partbelow = m21.stream.Part()
    partbelow.append(bestClef(below))
    for obj in below:
        partbelow.append(obj)

    return m21.stream.Score([partabove, partbelow])


def bestClef(objs):
    avg = meanMidi(objs)
    if avg > 80:
        return m21.clef.Treble8vaClef()
    elif avg > 58:
        return m21.clef.TrebleClef()
    elif avg > 36:
        return m21.clef.BassClef()
    else:
        return m21.clef.Bass8vbClef()


def meanMidi(objs: t.Seq[m21.Music21Object]):
    n, s = 0, 0
    stream = m21.stream.Stream()
    for obj in objs:
        stream.append(obj)
    for obj in stream.flat:
        try: 
            for pitch in obj.pitches:
                s += pitch.ps 
                n += 1
        except AttributeError:
            pass
    if n:
        return s/n
    return 0


def makeNoteSeq(midinotes: t.Seq[float], dur=1, split=False) -> m21.stream.Stream:
    """
    Take a sequence of midi midinotes and create a Part (or a Score if
    split is True and midinotes need to be split between two staffs)

    midinotes: a seq. of midi values (fractional values are allowed)
    """
    s = m21.stream.Part()
    centroid = sum(midinotes)/len(midinotes)
    if centroid < 60:
        s.append(m21.clef.BassClef())
    for n in midinotes:
        s.append(m21.note.Note(n, quarterLength=dur))
    if split == 'auto' or split:
        if needsSplit(midinotes):
            return splitVoice(s)
        return s if not needsSplit(midinotes) else splitVoice(s)
    else:
        return s


def needsSplit(notes: t.U[t.Seq[float], m21.stream.Stream], splitpoint=60) -> bool:
    """

    notes:
        list of midinotes, or a m21 stream
    splitpoint:
        the note to use as splitpoint

    returns True if splitting is necessary
    """

    def _needsSplitMidinotes(midinotes):
        midi0 = min(midinotes)
        midi1 = max(midinotes)
        if midi0 < splitpoint - 7 and midi1 > splitpoint + 7:
            return True
        return False

    if isinstance(notes, list) and isinstance(notes[0], (int, float)):
        return _needsSplitMidinotes(notes)
    elif isinstance(notes, m21.stream.Stream):
        midinotes = [note.pitch.midi for note in notes.getElementsByClass(m21.note.Note)]
        return _needsSplitMidinotes(midinotes)
    else:
        raise TypeError(f"expected a list of midinotes or a m21.Stream, got {notes}")


def makeTimesig(num_or_dur: t.U[int, float], den:int=0) -> m21.meter.TimeSignature:
    """
    Create a m21 TimeSignature from either a numerator, denominator or from
    a duration in quarter notes.

    makeTimesig(2.5) -> 5/8
    makeTimesig(4)   -> 4/4
    """
    if den == 0:
        num, den = lib.quarters_to_timesig(num_or_dur)
    else:
        num = num_or_dur
    return m21.meter.TimeSignature(f"{num}/{den}")


def accidentalName(cents:int) -> str:
    """
    Given a number of cents, return the name of the accidental
    """
    accidentals = {
    # cents   name       pc mod
        0:   'natural',
        25:  'natural-up',
        50:  'quarter-sharp',
        75:  'sharp-down',
        100: 'sharp',
        125: 'sharp-up',
        150: 'three-quarters-sharp',

        -25: 'natural-down',
        -50: 'quarter-flat',
        -75: 'flat-up',
        -100:'flat',
        -125:'flat-down',
        -150:'three-quarters-flat'
    }
    if not -150 <= cents <= 150:
        raise ValueError("cents should be between -150 and +150")
    rndcents = round(cents/25)*25
    return accidentals.get(rndcents)


def makeAccidental(cents:int) -> m21.pitch.Accidental:
    """
    Make an accidental with possibly 1/8 tone alterration

    Example: create a C# 1/8 tone higher (C#+25)

    note = m21.note.Note(61)
    note.pitch.accidental = makeAccidental(125)
    """
    name = accidentalName(round(cents/25)*25)
    alter = round(cents / 50) * 50
    acc = m21.pitch.Accidental()
    acc.alter = alter
    # the non-standard-name should be done in the end, because otherwise
    # other settings (like .alter) will wipe it
    acc.set(name, allowNonStandardValue=True)
    return acc


def m21Notename(pitch: t.U[str, float]) -> str:
    """
    Convert a midinote or notename (like "4C+10") to a notename accepted
    by music21. The decimal part (the non-chromatic part, in this case "+10")
    will be discarded

    :param pitch: a midinote or a notename as returned by m2n
    :return: the notename accepted by music21
    """
    notename = pitch if isinstance(pitch, str) else m2n(pitch)
    if "-" in notename or "+" in notename:
        sep = "-" if "-" in notename else "+"
        notename = notename.split(sep)[0]
    if notename[0].isdecimal():
        # 4C# -> C#4
        return notename[1:] + notename[0]
    # C#4 -> C#4
    return notename


def makePitch(pitch: t.U[str, float], divsPerSemitone=4) -> t.Tup[m21.pitch.Pitch, int]:
    assert(isinstance(pitch, (str, int, float)))
    midinote = n2m(pitch) if isinstance(pitch, str) else pitch
    rounding_factor = 1/divsPerSemitone
    rounded_midinote = round(midinote/rounding_factor)*rounding_factor
    notename = m2n(rounded_midinote)
    octave, letter, alter, cents = split_notename(notename)
    basename, cents = split_cents(notename)
    m21notename = m21Notename(basename)
    accidental = makeAccidental(100*alter+cents)
    out = m21.pitch.Pitch(m21notename)
    out.accidental = accidental
    mididev = midinote-n2m(basename)
    centsdev = int(round(mididev*100))
    return out, centsdev


def _centsshown(centsdev, divsPerSemitone=4) -> str:
    """
    Given a cents deviation from a chromatic pitch, return
    a string to be shown along the notation, to indicate the
    true tuning of the note. If we are very close to a notated
    pitch (depending on divsPerSemitone), then we don't show
    anything. Otherwise, the deviation is always the deviation
    from the chromatic pitch

    :param centsdev: the deviation from the chromatic pitch
    :param divsPerSemitone: if given, overrides the value in the config
    :return: the string to be shown alongside the notated pitch
    """
    # cents can be also negative (see self.cents)
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


def makeNote(pitch: t.U[str, float], divsPerSemitone=4, showcents=False, **options
             ) -> t.Tup[m21.note.Note, int]:
    """
    Given a pitch as a (fractional) midinote or a notename, create a
    m21 Note with a max. 1/8 tone resolution.

    Any keyword option will be passed to m21.note.Note (for example,
    `duration` or `quarterLength`)

    :param pitch: the pitch of the resulting note (for example, 60.20, or "4C+20")
    :param divsPerSemitone: divisions per semitone (4=1/8 tones, possible values: 1, 2, 4)
    :param showcents: display the cents deviation as text
    :param options: any option will be passed to m21.note.Note
    :return: a tuple (m21.Note, cents deviation from the returned note)
    """
    assert isinstance(pitch, (str, int, float))
    pitch, centsdev = makePitch(pitch=pitch, divsPerSemitone=divsPerSemitone)
    note = m21.note.Note(60, **options)
    note.pitch = pitch
    if showcents:
        lyric = _centsshown(centsdev, divsPerSemitone=divsPerSemitone)
        if lyric:
            note.lyric = _centsshown(centsdev, divsPerSemitone=divsPerSemitone)
    return note, centsdev


def makeChord(pitches: t.Seq[float], divsPerSemitone=4, showcents=False, **options
              ) -> t.Tup[m21.chord.Chord, t.List[int]]:
    """
    Create a m21 Chord with the given pitches, adjusting the accidentals to divsPerSemitone
    (up to 1/8 tone). If showcents is True, the cents deviations to the written pitch
    are placed as a lyric attached to the chord.
    The cents deviations are returned as a second argument

    :param pitches: the midi notes
    :param divsPerSemitone: divisions per semitone (1, 2, 4)
    :param showcents: if True, cents deviation is added as lyric
    :param options: options passed to the Chord constructor (duration, quarterLength, etc)
    :return: a tuple (Chord, list of cents deviations)
    """
    notes, centsdevs = [], []
    pitches = sorted(pitches)
    for pitch in pitches:
        note, centsdev = makeNote(pitch, divsPerSemitone=divsPerSemitone, showcents=False)
        notes.append(note)
        centsdevs.append(centsdev)
    chord = m21.chord.Chord(notes, **options)
    if showcents:
        centsdevs.reverse()
        annotation = centsAnnotation(centsdevs)
        if annotation:
            chord.lyric = annotation
    return chord, centsdevs


def centsAnnotation(centsdevs:t.Seq[int]) -> str:
    """
    Given a list of cents deviations, construct an annotation as it would
    be placed as a lyric for a chord or a note

    :param centsdevs: the list of deviations from the written pitch
    :return: an annotation string to be attached to a chord or a note
    """
    annotations = [str(_centsshown(centsdev)) for centsdev in centsdevs]
    if not any(annotations):
        return ""
    return ",".join(annotations)


def addGraceNote(pitch:t.U[float, str, t.Seq[float]], anchorNote:m21.note.GeneralNote, dur=1/2,
                 nachschlag=False, context='Measure') -> m21.note.Note:
    """
    Add a grace note (or a nachschlag) to anchor note.A
    Anchor note should be part of a stream, to which the grace note will be added

    :param pitch: the pitch of the grace note (as midinote, or notename)
    :param anchorNote: the note the grace note will be added to
    :param dur: the written duration of the grace note
    :param nachschlag: if True, the grace note is added as nachschlag, after the anchor
    :param context: the context where anchor note is defined, as a str, or the
                    stream itself
    :return: the added grace note
    """
    stream = context if isinstance(context, m21.stream.Stream) else anchorNote.getContextByClass(context)
    if isinstance(pitch, (list, tuple)):
        grace = makeChord(pitch, quarterLength=dur)[0].getGrace()
    else:
        grace = makeNote(pitch, quarterLength=dur)[0].getGrace()
    grace.duration.slash = False
    if nachschlag:
        grace.priority = 2
    stream.insert(anchorNote.getOffsetBySite(stream), grace)
    return grace


def addGliss(start:m21.note.GeneralNote, end:m21.note.GeneralNote, linetype='solid',
             stream:t.U[str, m21.stream.Stream]='Measure') -> m21.spanner.Spanner:
    """
    Add a glissando between note0 and end. Both notes should already be part of a stream

    :param start: start note
    :param end: end note
    :param linetype: line type of the glissando
    :param stream: a concrete stream or a context class
    :return: the created Glissando
    """
    if stream is None:
        stream = 'Measure'
    if not isinstance(start, m21.note.NotRest):
        raise TypeError(f"Expected a Note or a Chord, got {type(start)}")
    gliss = m21.spanner.Glissando([start, end])
    gliss.lineType = linetype
    # this creates a <slide> tag when converted to xml
    if linetype == 'solid':
        gliss.slideType = 'continuous'
    context = stream if isinstance(stream, m21.stream.Stream) else start.getContextByClass(stream)
    context.insert(start.getOffsetBySite(context), gliss)
    return gliss


def _noteScalePitch(note: m21.note.Note, factor: t.Rat) -> None:
    """
    Scale the pitch of note INPLACE

    :param note: a m21 note
    :param factor: the factor to multiply pitch by
    """
    midinote = float(note.pitch.ps * factor)
    pitch, centsdev = makePitch(midinote)
    note.pitch = pitch   


def stackParts(parts:t.Seq[m21.stream.Stream], outstream:m21.stream.Stream=None
               ) -> m21.stream.Stream:
    """
    Create a score from the different parts given

    This solves the problem that a Score will stack Parts vertically,
    but append sny other stream horizontally

    """
    outstream = outstream or m21.stream.Score()
    for part in parts:
        part = lib.astype(m21.stream.Part, part)
        outstream.insert(0, part)
    return outstream


def attachToObject(obj:m21.Music21Object, thingToAttach:m21.Music21Object, contextclass):
    context = obj.getContextByClass(contextclass)
    offset = obj.getOffsetBySite(context)
    context.insert(offset, thingToAttach)


def addTextExpression(note:m21.note.GeneralNote, text:str, placement="above",
                      contextclass='Measure', fontSize:float=None,
                      letterSpacing:float=None, fontWeight:str=None) -> None:
    """
    Add a text expression to note. The note needs to be already inside a stream,
    since the text expression is added to the stream

    :param note: the note (or chord) to add text expr. to
    :param text: the text
    :param placement: above or below
    :param contextclass: the context in which note is defined (passed to getContextByClass)
    :return:
    """
    textexpr = makeTextExpression(text=text, placement=placement, fontSize=fontSize,
                                  letterSpacing=letterSpacing, fontWeight=fontWeight)
    attachToObject(note, textexpr, contextclass)


def makeTextExpression(text:str, placement="above",
                       contextclass='Measure', fontSize:float=None,
                       letterSpacing:float=None, fontWeight:str=None
                       ) -> m21.expressions.TextExpression:
    textexpr = m21.expressions.TextExpression(text)
    textexpr.positionPlacement = placement
    if fontSize:
        textexpr.style.fontSize = fontSize
    if letterSpacing:
        textexpr.style.letterSpacing = letterSpacing
    if fontWeight:
        textexpr.style.fontWeight = fontWeight
    return textexpr  


def makeExpressionsFromLyrics(part, **kws):
    """
    Iterate over notes and chords in part and if a lyric is present
    move it to a text expression

    kws: any attribute passed to makeTextExpression (placement, fontSize, etc)
    """
    part.makeMeasures(inPlace=True)
    for event in part.getElementsByClass(m21.note.NotRest):
        if event.lyric:
            text = event.lyric
            event.lyric = None 
            print(event)
            addTextExpression(event, text=text, **kws)
    

def hideNotehead(event:m21.note.NotRest, hideAccidental=True):
    if isinstance(event, m21.note.Note):
        event.style.hideObjectOnPrint = True
        if hideAccidental:
            event.pitch.accidental.displayStatus = False
    elif isinstance(event, m21.chord.Chord):
        for note in event:
            note.style.hideObjectOnPrint = True
            if hideAccidental:
                note.pitch.accidental.displayStatus = False
    else:
        raise TypeError(f"expected a Note or a Chord, got {type(event)}")


def addDynamic(note:m21.note.GeneralNote, dynamic:str, contextclass='Measure') -> None:
    attachToObject(note, m21.dynamics.Dynamic(dynamic), contextclass)


def scoreSchema(durs: t.Seq[float],
                default='rest',
                barlines: t.Dict[int, str] = None,
                labels: t.Dict[int, str]   = None,
                notes: t.Dict[int, float]  = None,
                separators: t.Dict[int, t.Dict] = None,
                tempo: int = None,
                ) -> m21.stream.Part:
    """
    Make an empty score where each measure is indicated by the duration
    in quarters.

    durs: a seq. of durations, where each duration indicates the length
          of each measure.
          e.g: 1.5 -> 3/8, 1.25 -> 5/16, 4 -> 4/4
    barlines:
        if given, a dictionary of measure_index: barline_style
        Possible styles are: 'double', 'dashed' or 'final'
    labels:
        if given, a dictionary of measure_index: label
        The label will be attached as a text expression to the measure
    notes:
        if given, a dictionary of measure_idx: midinote
        This note will be used instead of the default
    separators:
        if given, a dict of measure_idx: sep_dict where sep_dict
        can have the keys {'dur': duration, 'fill': 'rest' / 'fill': midinote}
        A separator adds a measure before the given idx. Separators don't affect
        measure indices used in other indicators (barlines, labels, notes)
    default: either 'rest' or a midinote, will be used to fill measures

    Returns: a **music21.Part**, which can be wrapped in a Score or used as is
    """
    part = m21.stream.Part()
    measnum = 0
    for i, dur in enumerate(durs):
        measnum += 1
        sep = separators.get(i) if separators else None
        if sep:
            sepdur = sep.get(dur, 1)
            sepfill = sep.get('fill', 'rest')
            sepmeas = m21.stream.Measure(number=measnum)
            sepmeas.timeSignature = makeTimesig(sepdur)
            if sepfill == 'rest' or sepfill == 0:
                sepmeas.append(m21.note.Rest(quarterLength=sepdur))
            else:
                sepmeas.append(m21.note.Note(sepfill, quarterLength=sepdur))
            part.append(sepmeas)
            measnum += 1
        meas = m21.stream.Measure(number=i+1)
        meas.timeSignature = makeTimesig(dur)
        barline = barlines.get(i) if barlines else None
        if barline:
            meas.append(m21.bar.Barline(style=barline))
        label = labels.get(i) if labels else None
        if label:
            meas.append(m21.expressions.TextExpression(label))
        midinote = notes.get(i) if notes is not None else None
        if midinote:
            meas.append(m21.note.Note(midinote, quarterLength=dur))
        elif default == 'rest':
            meas.append(m21.note.Rest(quarterLength=dur))
        else:
            meas.append(m21.note.Note(default, quarterLength=dur))
        part.append(meas)

    part[-1].append(m21.bar.Barline(style='final'))
    if tempo is not None:
        part[0].insert(0, m21.tempo.MetronomeMark(number=tempo))
    return part


def getXml(m21stream: m21.stream.Stream) -> str:
    """
    Generate musicxml from the given m21 stream, return it as a str

    :param m21stream:  a m21 stream
    :return: the xml generated, as string
    """
    exporter = m21.musicxml.m21ToXml.GeneralObjectExporter(m21stream)
    return exporter.parse().decode('utf-8')


def saveLily(m21stream, outfile: str) -> str:
    """
    Save to lilypond via musicxml2ly. Returns the saved path

    :param m21stream: the stream to save
    :param outfile: the name of the outfile
    """
    from emlib.music import lilytools
    xmlpath = str(m21stream.write('xml'))
    if not os.path.exists(xmlpath):
        raise RuntimeError("Could not write stream to xml")
    lypath = lilytools.musicxml2ly(str(xmlpath), outfile=outfile)
    return lypath


def renderViaLily(m21obj:m21.Music21Object, fmt:str=None, outfile:str=None, show=False) -> str:
    """
    Create a pdf or png via lilypond, bypassing the builtin converter
    (using musicxml2ly instead)

    To use the builtin method, use stream.write('lily.pdf') or stream.write('lily.png')

    m21obj:
        the stream to convert to lilypond
    fmt:
        one of 'png' or 'pdf'
    outfile:
        if given, the name of the lilypond file generated. Otherwise
        a temporary file is created
    show:
        if True, show the resulting file
    """
    if outfile is None and fmt is None:
        fmt = 'png'
    elif fmt is None:
        assert outfile is not None
        fmt = os.path.splitext(outfile)[1][1:]
    elif outfile is None:
        import tempfile
        assert fmt in ('png', 'pdf')
        outfile = tempfile.mktemp(suffix="."+fmt)
    else:
        ext = os.path.splitext(outfile)[1][1:]
        if fmt != ext:
            raise ValueError(f"outfile has an extension ({ext}) which does not match the format given ({fmt})")
    assert fmt in ('png', 'pdf')
    from emlib.music import lilytools
    xmlpath = str(m21obj.write('xml'))
    if not os.path.exists(xmlpath):
        raise RuntimeError("Could not write stream to xml")
    lypath = lilytools.musicxml2ly(str(xmlpath))
    if not os.path.exists(lypath):
        raise RuntimeError(f"Error converting {xmlpath} to lilypond {lypath}")    
    if fmt == 'png':
        outfile = lilytools.lily2png(lypath, outfile)
    elif fmt == 'pdf':
        outfile = lilytools.lily2pdf(lypath, outfile)
    else:
        raise ValueError(f"fmt should be png or pdf, got {fmt}")
    if not os.path.exists(outfile):
        raise RuntimeError(f"Error converting lilypond file {lypath} to {fmt} {outfile}")
    if show:
        lib.open_with_standard_app(outfile)
    return outfile


def makeImage(m21obj, outfile:str=None, fmt='xml.png', 
              fixstream=True, musicxml2ly=True) -> str:
    """
    Generate an image from m21obj

    outfile    : the file to write to, or None to create a temporary
    fmt        : the format, one of "xml.png" or "lily.png"
    fixstream  : see m21fix
    musicxml2ly: if fmt is lily.png and musicxml2ly is True, then conversion
                 is performed via the external tool `musicxml2ly`, otherwise 
                 the conversion routine provided by music21 is used
    
    returns: the path to the generated image file
    """
    if fixstream and isinstance(m21obj, m21.stream.Stream):
        m21obj = m21fix.fixStream(m21obj, inPlace=True)
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


def showImage(m21obj, fmt='xml.png'):
    imgpath = makeImage(m21obj, fmt=fmt)
    # TOOD: extract this and put it in its own module (emlib.interactive)
    from emlib.music.core import tools
    tools.pngShow(imgpath)

