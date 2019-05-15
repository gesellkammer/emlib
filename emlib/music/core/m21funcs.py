import music21 as m21
import emlib.typehints as t
from .config import config
from emlib.music import m21tools


def m21Note(pitch: t.U[str, float], showcents=None, **options) -> m21.note.Note:
    """
    Create a m21.note.Note, taking semitoneDivisions into account

    :param pitch: a notename or a midinote, possibly with fractional part
    :param showcents: if True, attach the cents representation as a lyric
    :param options: options passed to m21.note.Note
    :return: the generated note
    """
    divsPerSemitone = config['show.semitoneDivisions']
    showcents = showcents if showcents is not None else config['show.cents']
    note, centsdev = m21tools.makeNote(pitch, divsPerSemitone=divsPerSemitone, showcents=showcents,
                                       **options)
    return note


def m21Chord(midinotes:t.Seq[float], showcents=None, **options
             ) -> m21.chord.Chord:
    """
    Create a m21 Chord out of a seq. of midinotes
    """
    # m21chord = m21.chord.Chord([m21.note.Note(n.midi) for n in notes])
    divsPerSemi = config['show.semitoneDivisions']
    chord, cents = m21tools.makeChord(midinotes, showcents=showcents, divsPerSemitone=divsPerSemi,
                                      **options)
    return chord


def m21TextExpression(text:str, style:str=None) -> m21.expressions.TextExpression:
    """
    style: one of None (default), 'small', 'bold', 'label'
    """
    txtexp = m21.expressions.TextExpression(text)

    if style == 'small':
        txtexp.style.fontSize = 12.0
        txtexp.style.letterSpacing = 0.5
    elif style == 'bold':
        txtexp.style.fontWeight = 'bold'
    elif style == 'label':
        txtexp.style.fontSize = config.get('show.label.fontSize', 12.0)
        # txtexp.style.absoluteY = 40
    return txtexp


def m21Label(text:str):
    return m21TextExpression(text, style='label')