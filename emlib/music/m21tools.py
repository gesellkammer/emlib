import os

import music21 as m21
from emlib import typehints as t
from emlib import lib
import warnings


def makeNote(pitch, resolution=4):
    """
    resolution: division of the semitone (4=1/8 tone) used to determine the accidental
        (does not affect the resolution of the note itself, only of its representation)
    """
    n = m21.note.Note(pitch)
    roundedpitch = round(pitch*resolution)/resolution
    alter = roundedpitch - int(roundedpitch)
    acc = n.pitch.accidental
    acc.setAttributeIndependently('alter', alter)
    if alter == 0.25:
        acc.setAttributeIndependently('name', acc.name+'-up')
    elif alter == 0.75:
        acc.setAttributeIndependently('name', acc.name+'-down')
    return n


def _splitchord(chord: m21.chord.Chord, 
                partabove: m21.stream.Part, 
                partbelow: m21.stream.Part, 
                split=60) -> None:
    # type: (m21.chord.Chord, m21.stream.Part, m21.stream.Part, int) -> t.Tup[m21.stream.Part, m21.stream.Part]
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


def meanMidi(objs):
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


def splitvoice(*args, **kws):
    warnings.warn("deprecation error: use splitVoice")
    return splitVoice(*args, **kws)


def noteSeq(midinotes: t.Seq[float], dur=1, split=False) -> m21.stream.Part:
    """
    Take a sequence of midi midinotes and create a Part

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


def needsSplit(notes: t.U[list, m21.stream.Stream], splitpoint=60) -> bool:
    def _needsSplitMidinotes(midinotes):
        midi0 = min(midinotes)
        midi1 = max(midinotes)
        if midi0 < splitpoint - 7 and midi1 > splitpoint + 7:
            return True
        return False
    if isinstance(notes, list) and isinstance(notes[0], (int, float)):
        return _needsSplitMidinotes(notes)
    elif isisntance(notes, m21.stream.Stream):
        midinotes = [note.pitch.midi for note in notes.getElementsByClass(m21.note.Note)]
        return _needsSplitMidinotes(midinotes)
    else:
        raise TypeError(f"expected a list of midinotes or a m21.Stream, got {notes}")


def makeTimesig(num_or_dur, den=0):
    """
    makeTimesig(2.5) == makeTimesig(5, 8)
    makeTimesig(4) == makeTimesig(4, 4)
    """
    if den == 0:
        dur = num_or_dur
        num, den = lib.quarters_to_timesig(num_or_dur)
    else:
        num = num_or_dur
    return m21.meter.TimeSignature(f"{num}/{den}")


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
    barlines: if given, a dictionary of measure_index: barline_style
          Possible styles are: 'double', 'dashed' or 'final'
    labels: if given, a dictionary of measure_index: label
          The label will be attached as a text expression to the measure
    notes: if given, a dictionary of measure_idx: midinote
          This note will be used instead of the default
    separators: if given, a dict of measure_idx: sep_dict where sep_dict
          can have the keys {'dur': duration, 'fill': 'rest' / 'fill': midinote}
          A separator adds a measure before the given idx. Separators don't affect
          measure indices used in other indicators (barlines, labels, notes)
    default: either 'rest' or a midinote, will be used to fill measures

    Returns: a **music21.Part**, which can be wrapped in a Score or used as is
    """
    part = m21.stream.Part()
    measnum = 0
    metronome_set = False
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


def makeLily(m21stream, fmt=None, outfile=None, show=False):
    """
    Create a pdf or png via lilypond, bypassing the builtin converter
    (use musicxml2ly instead)
    """
    if outfile is None and fmt is None:
        fmt = 'png'
    elif fmt is None:
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
    xmlpath = str(m21stream.write('xml'))
    if not os.path.exists(xmlpath):
        raise RuntimeError("Could not write stream to xml")
    lypath = lilytools.musicxml2ly(str(xmlpath))
    if not os.path.exists(lypath):
        raise RuntimeError(f"Could not convert {xmlpath} to lilypond {lypath}")
    if fmt == 'png':
        outfile = lilytools.lily2png(lypath, outfile)
    elif fmt == 'pdf':
        outfile = lilytools.lily2pdf(lypath, outfile)
    else:
        raise ValueError(f"fmt should be png or pdf, got {fmt}")
    if not os.path.exists(outfile):
        raise RuntimeError(f"Could not convert lilypond file {lypath} to {fmt} {outfile}")
    if show:
        lib.open_with_standard_app(outfile)
    return outfile


