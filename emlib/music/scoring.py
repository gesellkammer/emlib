import os
import sys
import subprocess
import tempfile
import textwrap
from emlib import typehints as t
from fractions import Fraction as F
import abjad as abj
from emlib import lib
from emlib import iterlib
from emlib.music import packing
from emlib.music import lilytools
from emlib.conftools import makeConfig, getConfig as _getConfig

_moduleid = 'emlib:scoring'

config = makeConfig(
    _moduleid,
    default={
        'apps.pdf': '',
        'score.includebranch': True,
        'score.pagesize': 'a4',
        'score.orientation': 'portrait',
        'score.staffsize': 12,
        'lilypond.enablePointAndClick': False,
        'lilypond.gliss.use_macros': False,
        'gliss.skipSamePitch': True,
        'play.allowDiscontinuousGliss': False,
        'score.annotation.fontsize': 5
    }
)


def getConfig():
    return _getConfig(_moduleid)


class Note(t.NamedTuple):
    step: float
    offset: F
    dur: F
    annot:str = ""
    db: float = 0
    atriculation: str = ""
    gliss: bool = False
    stepend: float=0
    instr: str = None

    def __repr__(self):
        frac = lambda x: f"{float(x):.3f}"
        ss = [f"step={self.step:.2f}, offs={frac(self.offset)}, dur={frac(self.dur)}"]
        if self.db < 0:
            ss.append(f", db={float(self.db):.1f}")
        if self.gliss:
            ss.append(f", gliss!")
        if self.annot:
            ss.append(f", annot={self.annot}")
        if self.stepend > 0 and self.stepend != self.step:
            ss.append(f", stepend={self.stepend:.2f}")
        s = "".join(ss)
        return f"Note({s})"

    def clone(self, **kws) -> 'Note':
        """
        Replace any attribute of this Note
        """
        return self._replace(**kws)


def _next_in_grid(x: t.U[float, F], ticks: t.List[F]):
    return lib.snap_to_grids(x + F(1, 9999999), ticks, mode='ceil')


def snaptime(note: Note,
             divisors: t.List[int],
             mindur=F(1, 16),
             durdivisors: t.List[Note]=None) -> Note:
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
        end = _next_in_grid(start + mindur, ticks)
    return Note(note.step, start, end - start, note.annot)


def _fill_silences(notes: t.Seq[Note]) -> t.List[Note]:
    assert all(isinstance(note, Note) for note in notes)
    out = []  # type: t.List[Note]
    if notes[0].offset > 0:
        out.append(Note(0, F(0), notes[0].offset))
    for n0, n1 in iterlib.pairwise(notes):
        out.append(n0)
        gap = n1.offset - (n0.offset + n0.dur)
        assert gap >= 0, f"negative gap! = {gap}"
        if gap > 0:
            rest = Note(0, n0.offset+n0.dur, gap)
            out.append(rest)
    out.append(notes[-1])
    return out


def pack_in_tracks(notes: t.List[Note], maxrange=36) -> t.List[t.List[Note]]:
    """
    Pack a list of possibly simultaneous notes into tracks, where the notes
    within one track are NOT simulatenous.

    A track is just a list of Notes
    """
    items = [packing.Item(obj=note, offset=note.offset, dur=note.dur, step=note.step) for note in notes]
    tracks = packing.pack_in_tracks(items, maxrange=maxrange)

    def unwrap_track(track: packing.Track) -> t.List[Note]:
        return [item.obj for item in track]

    return [unwrap_track(track) for track in tracks]


_grid_simple = abj.quantizationtools.UnweightedSearchTree(
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


def abj_voice_meanpitch(voice):
    notes = abj.iterate(voice).by_class(abj.Note)
    pitches = [n.written_pitch.number + 60 for n in notes]
    avg = sum(pitches) / len(pitches)
    return avg


def abj_voice_setclef(voice):
    pitch = abj_voice_meanpitch(voice)
    if pitch < 48:
        clefid = "bass_8"
    elif pitch < 62:
        clefid = "bass"
    elif pitch < 80:
        clefid = "treble"
    else:
        clefid = "treble^15"
    clef = abj.Clef(clefid)
    abj.attach(clef, next(abj.iterate(voice).by_leaf()))
    return voice


def abj_add_literal(obj, text:str, position:str= "after") -> None:
    assert position in ('before', 'after')
    abj.attach(abj.LilyPondLiteral(text, format_slot=position), obj)


def _abj_lilyfile_findblock(lilyfile: abj.LilyPondFile, blockname:str) -> t.Opt[int]:
    """
    Find the index of the a Block. This is used to find an insertion
    point to put macro definitions

    Returns the index of the Block, or None if not found
    """
    for i, item in enumerate(lilyfile.items):
        if isinstance(item, abj.Block) and item.name == blockname:
            return i
    return None

def _abj_lilyfile_addheader(lilyfile: abj.LilyPondFile):
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
    if not config['lilypond.enablePointAndClick']:
        blocks.append(r"\pointAndClickOff")
    blocktext = "\n".join(blocks)
    idx = _abj_lilyfile_findblock(lilyfile, "score")
    lilyfile.items.insert(idx, blocktext)

def abj_get_attacks(voice:abj.Voice) -> t.List[abj.Note]:
    """
    Returns a list of notes or gracenotes which represent an attack
    (they are not tied to a previous note)
    """
    # notes = abj.iterate(voice).by_class(abj.Note)
    leafs = abj.iterate(voice).by_leaf()
    attacks = []
    lasttie = None

    def getTie(note) -> t.Opt[abj.Tie]:
        insp = abj.inspect(note)
        if insp.has_spanner():
            for span in abj.inspect(note).get_spanners():
                if isinstance(span, abj.Tie):
                    return span
        return None

    for leaf in leafs:
        graces = abj.inspect(leaf).get_grace_container()
        if graces:
            for grace in graces:
                if isinstance(grace, abj.Note):
                    attacks.append(grace)
        if isinstance(leaf, abj.Rest):
            lasttie = None
        elif isinstance(leaf, abj.Note):
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


def abj_voice_add_annotation(voice: abj.Voice, annotations: t.List[t.Opt[str]], fontsize:int=None):
    """
    add the annotations to each note in this voice
    """
    # prefix = "_" if orientation == "down" else "^"
    attacks = abj_get_attacks(voice)
    fontsize = fontsize if fontsize is not None else config['score.annotation.fontsize']

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
                if fontsize <= 0:
                    literal = f'{prefix}"{annot}"'
                else:
                    literal = fr"{prefix}\markup {{\abs-fontsize #{fontsize} {{ {annot} }} }}"
                abj_add_literal(attack, literal, "after")


def abj_voice_add_gliss(voice: abj.Voice, glisses: t.List[bool]):
    """
    Add glissando to the notes in the given voice

    voice: an abjad Voice
    glisses: a list of bools, where each value indicates if the corresponding
                    note should produce a sgliss.
    """
    attacks = abj_get_attacks(voice)
    assert len(attacks) == len(glisses)
    # We use macros defined in the header. These are added when the file is saved
    # later on
    use_macros = config['lilypond.gliss.use_macros']
    skipsame = config['gliss.skipSamePitch']
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
        print(note0, note1, gliss)
        if gliss:
            if samenote(note0, note1) and skipsame:
                continue
            if use_macros:
                abj_add_literal(note0, "\glissando \glissandoSkipOn ", "after")
                abj_add_literal(note1, "\glissandoSkipOff", "before")
            else:
                abj_add_literal(note0, "\glissando " + glissandoSkipOn, "after")
                abj_add_literal(note1, glissandoSkipOff, "before")



def abj_makevoice(notes: t.Seq[Note], divisors=None, grid="simple"):
    assert all(isinstance(note, Note) for note in notes)
    if divisors is None:
        divisors = [1, 2, 3, 4, 5, 6, 8, 10, 12]
    # notes = [snaptime(note, divisors) for note in notes]
    continuousnotes = _fill_silences(notes)
    durations = [int(float(n.dur) * 1000) for n in continuousnotes]
    pitches = [float(n.step - 60) for n in continuousnotes]
    qseq = abj.quantizationtools.QEventSequence.from_millisecond_pitch_pairs(list(zip(durations, pitches)))

    def fix_silences(seq):
        seq2 = []
        for p in seq:
            if isinstance(p, abj.quantizationtools.PitchedQEvent) and p.pitches[0].number < -50:
                r = abj.quantizationtools.SilentQEvent(offset=p.offset)
                p = r
            seq2.append(p)
        return abj.quantizationtools.QEventSequence(seq2)

    grids = {
        'simple':  _grid_simple
    }

    qseq = fix_silences(qseq)
    search_tree = grids.get(grid)
    if not search_tree:
        raise KeyError(f"grid {grid} not known. It should be one of {list(grids.keys())}")
    schema = abj.quantizationtools.MeasurewiseQSchema(search_tree=search_tree)
    quantizer = abj.quantizationtools.Quantizer()
    voice = quantizer(qseq, q_schema=schema)
    abj_voice_setclef(voice)
    attacks = abj_get_attacks(voice)

    if len(attacks) != len(notes):
        print("Length mismatch!")
        print(attacks, notes)
    else:
        annotations = [note.annot for note in notes]
        abj_voice_add_annotation(voice, annotations)
        abj_voice_add_gliss(voice, [note.gliss for note in notes])
    return voice


class Score:
    def __init__(self, score, pagesize: str=None, orientation: str=None, staffsize: int=None,
                 includebranch: bool=None) -> None:
        self.score = score
        self.pagesize = pagesize or config['score.pagesize']
        self.orientation = orientation or config['score.orientation']
        self.staffsize = staffsize or config['score.staffsize']
        self.includebranch = includebranch if includebranch is not None else config['score.includebranch']

    def show(self) -> None: ...
    def dump(self) -> None: ...
    def save(self, outfile) -> None: ...
    def extract_tracks(self) -> t.List[t.List[Note]]: ...
    def play(self, instr=None):
        tracks = self.extract_tracks()
        notes: t.List[Note] = []
        for track in tracks:
            notes.extend(track)
        playnotes(notes, instr=instr)


def abj_score2lily(score: abj.Score, pagesize: str=None, orientation: str=None,
                   staffsize: int=None) -> abj.LilyPondFile:
    """
    Create a LilyPondFile from a score by adding a header and setting score layout

    pagesize: a3 or a4
    orientation: portrait, landscape
    staffsize: the size of a staff, in points
    """
    if pagesize is None and orientation is None:
        paperdef = None
    else:
        papersize = pagesize.lower() if pagesize else 'a4'
        paperorient = orientation if orientation else 'portrait'
        assert orientation in ('landscape', 'portrait')
        assert papersize in ('a4', 'a3')
        paperdef = (papersize, paperorient)
    lilyfile = abj.LilyPondFile.new(score,
                                    global_staff_size=staffsize,
                                    default_paper_size=paperdef)
    _abj_lilyfile_addheader(lilyfile)
    return lilyfile


def _abj_save_lily(score: abj.Score, outfile: str=None,
                   pagesize: str=None, orientation: str=None, staffsize: int=None) -> str:
    """
    Save the score as a .ly file, returns the path of the saved file

    :param score: the abj.Score to save
    :param outfile: the path of the lilypond file (or None to save to a temp file)
    """
    if outfile is None:
        outfile = tempfile.mktemp(suffix=".ly")
    lilyfile = abj_score2lily(score, pagesize=pagesize, orientation=orientation, staffsize=staffsize)
    with open(outfile, "w") as f:
        f.write(format(lilyfile))
    return outfile


def _abj_save_pdf(score: abj.Score, outfile: str,
                  pagesize: str=None, orientation: str=None, staffsize: int=None) -> None:
    # we generate a lilyfile, then a pdf from there
    lilyfile = tempfile.mktemp(suffix=".ly")
    _abj_save_lily(score, lilyfile, pagesize=pagesize, orientation=orientation, staffsize=staffsize)
    lilytools.lily2pdf(lilyfile, outfile)


def _openpdf(pdf:str) -> None:
    app = getConfig().get('apps.pdf')
    pdf = os.path.abspath(os.path.expanduser(pdf))
    if not app:
        lib.open_with_standard_app(pdf)
    else:
        subprocess.call([app, pdf])


class AbjScore(Score):
    """
    Not to be created directly
    """
    def __init__(self, abjscore,
                 pagesize: str=None,
                 orientation: str=None,
                 staffsize: int=None,
                 includebranch: bool=None,
                 tracks=None) -> None:
        super().__init__(abjscore, pagesize=pagesize, orientation=orientation, staffsize=staffsize,
                         includebranch=includebranch)
        self._tracks = tracks

    def show(self):
        pdf = tempfile.mktemp(suffix='.pdf')
        self.save(pdf)
        _openpdf(pdf)  # this will block

    def dump(self):
        abjlilyfile = self._tolily()
        print(format(abjlilyfile))

    def _tolily(self) -> abj.LilyPondFile:
        lilyfile = _abj_score_to_lily(self.score, pagesize=self.pagesize, orientation=self.orientation,
                                      staffsize=self.staffsize)
        return lilyfile

    def _savelily(self, outfile:str=None) -> str:
        outfile = _abj_save_lily(self.score, outfile, pagesize=self.pagesize,
                                 orientation=self.orientation, staffsize=self.staffsize)
        return outfile

    def save(self, outfile:str) -> None:
        """
        Save this score.

        Supported formats are: lilypond (.ly), pdf (.pdf)
        """
        outfile = os.path.expanduser(outfile)
        base, ext = os.path.splitext(outfile)
        if ext == '.ly':
            self._savelily(outfile)
        elif ext == '.pdf':
            lilyfile = self._savelily(outfile)
            lilytools.lily2pdf(lilyfile, outfile)

    def extract_tracks(self) -> t.List[t.List[Note]]:
        """
        This extracts back a list of tracks from the current score
        """
        tracks = []
        voices = abj.iterate(self.score).by_class(abj.Voice)
        for voice in voices:
            track = _abj_voice_to_notes(voice)
            tracks.append(track)
        return tracks

def _abj_voice_to_notes(voice):
    raise NotImplementedError("this is not implemented!")

def _abj_voice_to_notes(voice):
    """
    Returns a list of Notes and Rests (a rest is given as a Note with
    step=0 and amp=0)

    tied notes are joint together
    rests are not joi
    :param voice:
    :return:
    """


def abj_voices2score(voices: t.List[abj.Voice]) -> AbjScore:
    """
    voices: a list of voices as returned by [makevoice(notes) for notes in ...]
    """
    voices.sort(key=abj_voice_meanpitch, reverse=True)
    staffs = [abj.Staff([voice]) for voice in voices]
    score = abj.Score(staffs)
    return score


def makescore(tracks: t.List[t.List[Note]], pagesize=None, orientation=None, staffsize=None,
              includebranch=None,
              backend='abjad') -> Score:
    """
    tracks:
        a list of Tracks, as returned by pack_in_tracks(notes)
    pagesize:
        a string like 'a4', 'a3', etc.
    orientation:
        'landscape' or 'portrait'
    staffsize:
        staff size

    backend:
        the backend to use (abjad only at the moment)

    Example:

    notes = [node2note(node) for node in nodes]
    tracks = pack_in_tracks(notes)
    score = makescore(tracks)
    score.save("myscore.pdf")
    """
    assert all(isinstance(track, list) for track in tracks), str(tracks)
    if backend == 'abjad':
        voices = [abj_makevoice(track) for track in tracks]
        score = abj_voices2score(voices)
        return AbjScore(score, tracks=tracks, pagesize=pagesize, orientation=orientation, staffsize=staffsize,
                        includebranch=includebranch)
    else:
        raise ValueError("Backend not supported")


from . scoring_play import playnotes
