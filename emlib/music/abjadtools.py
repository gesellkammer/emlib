from dataclasses import dataclass
from fractions import Fraction as F
import copy

import abjad as abj
from emlib.music import lilytools
from emlib import typehints as t
from emlib import iterlib
import textwrap
import music21 as m21


def voiceMeanPitch(voice: abj.Voice) -> float:
    # notes = abj.iterate(voice).by_class(abj.Note)
    notes = [obj for obj in abj.iterate(voice).leaves() if isinstance(obj, abj.Note)]
    pitches = [noteGetMidinote(n) for n in notes]
    if not pitches:
        return 60
    avg = sum(pitches) / len(pitches)
    return avg


def noteGetMidinote(note: abj.Note) -> float:
    return note.written_pitch.number + 60


def voiceSetClef(voice: abj.Voice) -> abj.Voice:
    pitch = voiceMeanPitch(voice)
    if pitch < 48:
        clefid = "bass_8"
    elif pitch < 62:
        clefid = "bass"
    elif pitch < 80:
        clefid = "treble"
    else:
        clefid = "treble^15"
    clef = abj.Clef(clefid)
    # abj.attach(clef, next(abj.iterate(voice).by_leaf()))
    abj.attach(clef, next(abj.iterate(voice).leaves()))
    return voice


def noteTied(n):
    """
    Returns a int describind the state:
        0: not tied
        1: tied forward
        2: tied backwards
        3: tied in both directions

    tied = noteTied(note)
    forward = tied & 0b01
    backwards = tied & 0b10
    both = tied & 0b11
    """
    insp = abj.inspect(n)
    spanners = insp.spanners()
    out = 0
    for spanner in spanners:
        if isinstance(spanner, abj.spanners.Tie) and len(spanner.leaves) > 1:
            leaves = spanner.leaves
            if leaves[0] is n:
                out = out | 0b01
            elif leaves[-1] is n:
                out = out | 0b10
            elif n in leaves:
                out = out | 0b11
            else:
                raise ValueError("This should not happen!") 
    return out


def isNoteTied(n):
    """
    Return if note is tied forward
    """
    return bool(noteTied(n) & 0b01)


def addLiteral(obj, text:str, position:str= "after") -> None:
    """
    add a lilypond literal to obj

    obj: an abjad object (a note, a chord)
    text: the text to add
    position: one of 'before', 'after'
    """
    assert position in ('before', 'after')
    abj.attach(abj.LilyPondLiteral(text, format_slot=position), obj)


def getAttacks(voice:abj.Voice) -> t.List[abj.Note]:
    """
    Returns a list of notes or gracenotes which represent an attack
    (they are not tied to a previous note)
    """
    leaves = abj.iterate(voice).leaves()
    attacks = []
    lasttie = None

    def getTie(note) -> t.Opt[abj.Tie]:
        insp = abj.inspect(note)
        if insp.has_spanner():
            spanners = abj.inspect(note).spanners()
            for span in spanners:
                if isinstance(span, abj.Tie):
                    return span
        return None

    for leaf in leaves:
        if isinstance(leaf, (abj.Rest, abj.Note)):
            #graces = abj.inspect(leaf).grace_container()
            #if graces:
            #    print(f"graces detected! leaf: {leaf} --- ")
            #    for grace in graces:
            #        if isinstance(grace, abj.Note):
            #            print(f"appending grace note: {grace}")
            #            attacks.append(grace)
            if isinstance(leaf, abj.Rest):
                lasttie = None
            else:
                note = leaf
                tie = getTie(note)
                if tie:
                    if tie != lasttie:
                        attacks.append(note)
                        lasttie = tie
                else:
                    lasttie = None
                    attacks.append(note)
    return attacks


@dataclass
class Event:
    pitch: float
    start: F
    dur: F


def getEvents(voice:abj.Voice) -> t.List[Event]:
    """
    extract a list of Events from an abjad voice

    voice: an abjad stream
    """
    pass


def scoreToLily(score: abj.Score, pageSize: str=None, orientation: str=None,
                staffSize: int=None) -> abj.LilyPondFile:
    """
    Create a LilyPondFile from a score by adding a header and setting score layout

    pageSize: a3 or a4
    orientation: portrait, landscape
    staffSize: the size of a staff, in points
    """
    if pageSize is None and orientation is None:
        paperDef = None
    else:
        paperSize = pageSize.lower() if pageSize else 'a4'
        paperOrientation = orientation if orientation else 'portrait'
        assert orientation in ('landscape', 'portrait')
        assert paperSize in ('a4', 'a3')
        paperDef = (paperSize, paperOrientation)
    lilyfile = abj.LilyPondFile.new(score,
                                    global_staff_size=staffSize,
                                    default_paper_size=paperDef)
    lilyfileAddHeader(lilyfile)
    return lilyfile


def saveLily(score: abj.Score, outfile: str=None,
             pageSize: str=None, orientation: str=None, staffSize: int=None) -> str:
    """
    Save the score as a .ly file, returns the path of the saved file

    :param score: the abj.Score to save
    :param outfile: the path of the lilypond file (or None to save to a temp file)
    :param pageSize: the size as str, one of "a4", "a3"
    :param orientation: one of 'landscape', 'portrait'
    :param staffSize: the size of the staff, in points. Default=12
    """
    import tempfile
    if outfile is None:
        outfile = tempfile.mktemp(suffix=".ly")
    lilyfile = scoreToLily(score, pageSize=pageSize, orientation=orientation, staffSize=staffSize)
    with open(outfile, "w") as f:
        f.write(format(lilyfile))
    return outfile


def savePdf(score: abj.Score, outfile: str,
            pageSize: str=None, orientation: str=None, staffSize: int=None) -> None:
    """
    Save this score as pdf

    :param score: the abj.Score to save
    :param outfile: the path to save to
    :param pageSize: the size as str, one of "a4", "a3"
    :param orientation: one of 'landscape', 'portrait'
    :param staffSize: the size of the staff, in points. Default=12

    """
    # we generate a lilyfile, then a pdf from there
    import tempfile
    lilyfile = tempfile.mktemp(suffix=".ly")
    saveLily(score, lilyfile, pageSize=pageSize, orientation=orientation, staffSize=staffSize)
    lilytools.lily2pdf(lilyfile, outfile)


def voicesToScore(voices: t.List[abj.Voice]) -> abj.Score:
    """
    voices: a list of voices as returned by [makevoice(notes) for notes in ...]

    :param voices: a list of voices
    :return: a Score
    """
    voices.sort(key=voiceMeanPitch, reverse=True)
    staffs = [abj.Staff([voice]) for voice in voices]
    score = abj.Score(staffs)
    return score


def lilyfileFindBlock(lilyfile: abj.LilyPondFile, blockname:str) -> t.Opt[int]:
    """
    Find the index of a Block. This is used to find an insertion
    point to put macro definitions
    Returns the index of the Block, or None if not found
    """
    for i, item in enumerate(lilyfile.items):
        if isinstance(item, abj.Block) and item.name == blockname:
            return i
    return None


def lilyfileAddHeader(lilyfile: abj.LilyPondFile, enablePointAndClick=False):
    """
    Adds a header to the given LyliPondFile
    """
    gliss_header = textwrap.dedent(r"""
        glissandoSkipOn = {
            \override NoteColumn.glissando-skip = ##t
            \hide NoteHead
            \override NoteHead.no-ledgers = ##t
        }

        glissandoSkipOff = {
            \revert NoteColumn.glissando-skip
            \undo \hide NoteHead
            \revert NoteHead.no-ledgers
        }

    """)
    blocks = [gliss_header]
    if not enablePointAndClick:
        blocks.append(r"\pointAndClickOff")
    blocktext = "\n".join(blocks)
    idx = lilyfileFindBlock(lilyfile, "score")
    lilyfile.items.insert(idx, blocktext)


def voiceAddAnnotation(voice: abj.Voice, annotations: t.List[t.Opt[str]], fontSize:int=10, attacks=None):
    """
    add the annotations to each note in this voice

    attacks: the result of calling getAttacks.
    """
    # prefix = "_" if orientation == "down" else "^"
    attacks = attacks or getAttacks(voice)
    # fontSize = fontSize if fontSize is not None else config['score.annotation.fontSize']

    if len(attacks) != len(annotations):
        for p in attacks: print(p)
        for p in annotations: print(p)
        # raise ValueError("Annotation mismatch")
    for attack, annotstr in zip(attacks, annotations):
        if annotstr:
            annots = annotstr.split(";")
            for annot in annots:
                if annot[0] not in  "_^":
                    prefix = "_"
                else:
                    prefix = annot[0]
                    annot = annot[1:]
                if fontSize <= 0:
                    literal = f'{prefix}"{annot}"'
                else:
                    literal = fr"{prefix}\markup {{\abs-fontSize #{fontSize} {{ {annot} }} }}"
                addLiteral(attack, literal, "after")
                # abjAddLiteral(attack, literal, "after")


def voiceAddGliss(voice: abj.Voice, glisses: t.List[bool], usemacros=True, skipsame=True, attacks=None):
    """
    Add glissando to the notes in the given voice

    voice: an abjad Voice
    glisses: a list of bools, where each value indicates if the corresponding
                    note should produce a sgliss.
    """
    attacks = attacks or getAttacks(voice)
    assert len(attacks) == len(glisses)
    # We use macros defined in the header. These are added when the file is saved
    # later on
    # usemacros = config['lilypond.gliss.usemacros']
    # skipsame = config['gliss.skipSamePitch']
    glissandoSkipOn = textwrap.dedent(r"""
        \override NoteColumn.glissando-skip = ##t
        \hide NoteHead
        \override NoteHead.no-ledgers =  ##t    
    """)
    glissandoSkipOff = textwrap.dedent(r"""
        \revert NoteColumn.glissando-skip
        \undo \hide NoteHead
        \revert NoteHead.no-ledgers
    """)

    def samenote(n0: abj.Note, n1: abj.Note) -> bool:
        return n0.written_pitch == n1.written_pitch

    for (note0, note1), gliss in zip(iterlib.pairwise(attacks), glisses):
        if gliss:
            if samenote(note0, note1) and skipsame:
                continue
            if usemacros:
                addLiteral(note0, "\glissando \glissandoSkipOn ", "after")
                addLiteral(note1, "\glissandoSkipOff", "before")
            else:
                addLiteral(note0, "\glissando " + glissandoSkipOn, "after")
                addLiteral(note1, glissandoSkipOff, "before")


def objDuration(obj) -> F:
    """
    Calculate the duration of obj.

    1/4 = 1 quarter note
    """
    if isinstance(obj, (abj.core.Tuplet, abj.core.Note, abj.core.Rest)):
        dur = abj.inspect(obj).duration()
        return dur
    else:
        raise TypeError(f"dur. not implemented for {type(obj)}")


def _abjTupleGetDurationType(tup: abj.core.Tuplet) -> str:
    tupdur = abj.inspect(tup).duration()
    mult = tup.multiplier
    if mult.denominator <= 3:
        durtype = {
            2: 'quarter',
            4: 'eighth',
            8: '16th',
            16: '32nd'
        }[tupdur.denominator]
    elif mult.denominator <= 7:
        durtype = {
            4: '16th',
            8: '32nd',
            16: '64th'
        }[tupdur.denominator]
    elif mult.denominator <= 15:
        durtype = {
            4: '16th',
            8: '32nd',
            16: '64th'
        }[tupdur.denominator]
    else:
        raise ValueError(f"??? {tup} dur: {tupdur}")
    return durtype


def _abjDurClassify(num, den) -> t.Tup[str, int]:
    durname = {
        1: 'whole',
        2: 'half',
        4: 'quarter',
        8: 'eighth',
        16: '16th',
        32: '32nd',
        64: '64th'
    }[den]
    dots = {
        1: 0,
        3: 1,
        7: 2
    }[num]
    return durname, dots


def noteGetMusic21Duration(abjnote: abj.Leaf, tuplet: abj.Tuplet=None) -> m21.duration.Duration:
    dur = abjnote.written_duration * 4
    dur = F(dur.numerator, dur.denominator)
    dur = m21.duration.Duration(dur)
    if tuplet:
        dur.appendTuplet(tuplet)
    return dur


def noteToMusic21(abjnote: abj.Note, tuplet: abj.Tuplet=None) -> m21.note.Note:
    """
    Convert an abjad to a music21 note

    abjnote: the abjad note to convert to
    tuplet: a lilipond tuplet, if applies
     """
    dur = noteGetMusic21Duration(abjnote, tuplet)
    pitch = noteGetMidinote(abjnote)
    m21note = m21.note.Note(pitch, duration=dur)
    return m21note


def extractMatching(abjobj, matchfunc):
    if not hasattr(abjobj, '__iter__'):
        if matchfunc(abjobj):
            yield abjobj
    else:
        for elem in abjobj:
            yield from extractMatching(elem, matchfunc)


def _abjtom21(abjobj, m21stream, level=0, durfactor=4, tup=None, state=None) -> m21.stream.Stream:
    """

    :param abjobj: the abjad object to convert
    :param m21stream: the stream being converted to
    :param level: the level of recursion
    :param durfactor: current duration factor
    :param tup: current m21 tuplet
    :param state: a dictionary used to pass global state
    :return: the music21 stream
    """
    indent = "\t"*level
    if state is None:
        state = {}
    debug = state.get('debug', False)

    def append(stream, obj):
        if debug:
            print(indent, f"{stream}  <- {obj}")
        stream.append(obj)

    if hasattr(abjobj, '__iter__'):
        if debug:
            print(indent, "iter", type(abjobj), abjobj)
        if isinstance(abjobj, abj.core.Measure):       # Measure
            meas0 = abjobj
            timesig = meas0.time_signature
            meas = m21.stream.Measure()
            oldtimesig = state.get('timesig')
            if timesig != oldtimesig:
                append(meas, m21.meter.TimeSignature(f"{timesig.numerator}/{timesig.denominator}"))
                state['timesig'] = timesig
            for elem in meas0:
                _abjtom21(elem, meas, level+1, durfactor, tup, state=state)
            append(m21stream, meas)
        elif isinstance(abjobj, abj.core.Voice):       # Voice
            voice0 = abjobj
            # voice = m21.stream.Voice()
            voice = m21.stream.Part()
            for meas in voice0:
                _abjtom21(meas, voice, level+1, durfactor, tup, state=state)
            append(m21stream, voice)
        elif isinstance(abjobj, abj.core.Tuplet):      # Tuplet
            mult = abjobj.multiplier
            newtup = m21.duration.Tuplet(mult.denominator, mult.numerator, bracket=True)
            m21durtype = _abjTupleGetDurationType(abjobj)
            newtup.setDurationType(m21durtype)
            if debug:
                print(indent, "Tuple!", mult, mult.numerator, mult.denominator, newtup)
            for elem in abjobj:
                _abjtom21(elem, m21stream, level+1, durfactor*abjobj.multiplier, tup=newtup, state=state)
            if debug:
                print(indent, "closing tuple")
        else:
            if debug:
                print("????", type(abjobj), abjobj)
    else:
        if debug:
            print("\t"*level, "tup: ", tup, "no iter", type(abjobj), abjobj)
        if isinstance(abjobj, (abj.core.Rest, abj.core.Note)):
            # check if it has gracenotes
            graces = abj.inspect(abjobj).grace_container()
            if graces:
                for grace in graces:
                    if isinstance(grace, abj.Note):
                        # add grace note to m21 stream
                        m21grace = noteToMusic21(grace).getGrace()
                        append(m21stream, m21grace)
            # abjdur = abjobj.written_duration
            # durtype, dots = _abjDurClassify(abjdur.numerator, abjdur.denominator)
            # dur = m21.duration.Duration(durtype, dots=dots)

            abjdur = abjobj.written_duration*4
            dur = m21.duration.Duration(F(abjdur.numerator, abjdur.denominator))
            if tup:
                dur.appendTuplet(tup)
            if isinstance(abjobj, abj.core.Rest):
                append(m21stream, m21.note.Rest(duration=dur))
            else:
                note = abjobj
                pitch = noteGetMidinote(note)
                m21note = m21.note.Note(pitch, duration=copy.deepcopy(dur))
                tie = noteTied(note)
                if debug:
                    print("\t"*level, "tie: ", tie)
                if isNoteTied(note):
                    m21note.tie = m21.tie.Tie()
                # m21note = copy.deepcopy(m21note)
                append(m21stream, m21note)
        else:
            if debug:
                print("\t"*level, "**** ???", type(abjobj), abjobj)
    return m21stream


def abjadToMusic21(abjadStream: abj.AbjadObject, debug=False) -> m21.stream.Stream:
    """
    Convert an abjad stream to a music21 stream

    :param abjadStream: an abjad stream
    :param debug: If True, print debugging information
    :return: the corresponding music21 stream
    """
    m21stream = m21.stream.Stream()
    out = _abjtom21(abjadStream, m21stream, state={'debug': debug})
    return out
