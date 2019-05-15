import typing as t
from emlib.music import scoring
from emlib import iterlib
from .core import *
from .config import config
from emlib.pitchtools import amp2db, db2amp


def node2note(node: Node, showbranch:bool, showmeta:bool) -> t.Opt[scoring.Note]:
    assert node.step is not None
    assert node.weight > 0
    assert node.amp > 0
    db = amp2db(node.amp)
    if showbranch:
        annot = f"_{node.name}{branchid2str(node.branch)}"
    else:
        annot = f"_{node.name}"
    if showmeta:
        metastrs = [f"{key}:{value}" for key, value in node.data.items()]
        annotup = " ".join(metastrs)
        annot += f";^{annotup}"
    #gliss = node.data.get('gliss', False)
    note = scoring.Note(pitch=node.step, offset=node.offset, dur=node.dur, annot=annot, db=db,
                        stepend=node.stepend)
    assert note.step == node.step
    return note


def branch2notes(branch:Branch, showbranch:bool=None, showmeta:bool=None,
                 callback=None) -> t.List[scoring.Note]:
    """
    callback:
        a function of the form (node, note) -> note
        If given, the callback is called with the node and the generated note. The
        function can either return None, a new note, or any number of notes. The
        returned note(s) replace the original note.

    """
    showbranch = showbranch if showbranch is not None else config['score.showbranch']
    showmeta = showmeta if showmeta is not None else config['score.showmeta']
    allnodes = list(branch.flatview())
    nodes = [node for node in allnodes if node.weight > 0 and node.amp > 0 and node.step > 0]
    notes = [node2note(node, showbranch=showbranch, showmeta=showmeta)
             for node in nodes]
    assert not any(note is None for note in notes)
    if not config['score.gliss.allowsamenote'] and len(notes) > 1:
        notes2 = []  # type: t.List[scoring.Note]
        for n0, n1 in iterlib.pairwise(notes):
            if n0.gliss and n0.step == n1.step:
                n0 = n0._replace(gliss=False)
            notes2.append(n0)
        notes2.append(notes[-1])
        notes = notes2
    if callback:
        transformed_notes = []
        for node, note in zip(nodes, notes):
            out = callback(node, note)
            if out is None:
                transformed_notes.append(note)
            elif isinstance(out, scoring.Note):
                transformed_notes.append(out)
            elif isinstance(out, (list, tuple)):
                transformed_notes.extend(out)
            elif out == "SKIP":
                continue
            else:
                raise ValueError(f"expected a Note or None, got {out}")
        notes = transformed_notes
    return notes

def notes_scalegain(notes: t.List[scoring.Note], gain:float) -> t.List[scoring.Note]:
    return [note.clone(db=amp2db(db2amp(note.db)*gain)) for note in notes]



def makescore(branch: Branch, backend:str=None, showbranch:bool=None, pagesize:str=None,
              orientation:str=None, trackrange:int=None, showmeta:bool=None,
              callback=None) -> scoring.Score:
    """

    :param branch: a Branch of Nodes, as returned by LSystem.generate
    :param backend: "abjad" only at the moment
    :param showbranch: include the branch id as annotation in the score
    :param pagesize: a3, a4
    :param orientation: portait or landscape
    :param trackrange: the range within one track, that is, the max. allowed difference between the
                       lowest and highest note within a track. This is used to determine which
                       notes can be appended to which track
    :param callback: a function of the form (node, note) -> note
                     See branch2notes
    :return: a Score
    """
    showbranch = showbranch if showbranch is not None else config['score.showbranch']
    pagesize = (pagesize or config['score.pagesize']).lower()
    orientation = orientation or config['score.orientation']
    backend = backend or config['score.backend']
    trackrange = trackrange or config['score.track.maxrange']
    showmeta = showmeta if showmeta is not None else config['score.showmeta']
    assert orientation in ('landscape', 'portrait')
    assert pagesize in ('a4', 'a3')
    assert isinstance(showbranch, bool)
    assert backend in ('abjad',)
    assert isinstance(trackrange, int) and trackrange > 1

    if backend == 'abjad':
        return _makescore_abjad(branch, showbranch=showbranch, pagesize=pagesize,
                                orientation=orientation, trackrange=trackrange,
                                showmeta=showmeta)
    else:
        raise ValueError(f"Backend {backend} not supported")


def _makescore_abjad(branch: Branch, showbranch:bool, pagesize:str, orientation:str,
                     trackrange:int, showmeta:bool, callback=None) -> scoring.Score:
    notes = branch2notes(branch, showbranch=showbranch, showmeta=showmeta, callback=callback)
    tracks = scoring.pack_in_tracks(notes, maxrange=trackrange)
    return scoring.makescore(tracks, pagesize=pagesize, orientation=orientation,
                             includebranch=showbranch, backend='abjad')


def playbranch(branch:Branch, defaultinstr='piano', gliss=True, callback=None,
               gain=1, sr=44100, instrs=None):
    """
    Plays the Branch. Returns a process (subprocess.Popen)

    callback: a function of the form (node, scoring.Note) -> scoring.Note
    defaultinstr: the instr used for notes without an assigned instr
    instrs:
        a list of csound instrs. In order to have a note played by a given instr, the instr name
        must match the note.instr attr.
    """
    notes = branch2notes(branch, callback=callback)
    if gain != 1:
        notes = notes_scalegain(notes, gain)
    return scoring.playnotes(notes, defaultinstr=defaultinstr, gliss=gliss, sr=sr, instrs=instrs)
