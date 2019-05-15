import os
from fractions import Fraction

import music21 as m21

from emlib import lib
from emlib.pitchtools import *
from emlib import typehints as t


from .config import config, logger

insideJupyter = lib.inside_jupyter()


_enharmonic_sharp_to_flat = {
    'C#': 'Db',
    'D#': 'Eb',
    'E#': 'F',
    'F#': 'Gb',
    'G#': 'Ab',
    'A#': 'Bb',
    'H#': 'C'
}
_enharmonic_flat_to_sharp = {
    'Cb': 'H',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#',
    'Hb': 'A#'
}


def enharmonic(n:str) -> str:
    n = n.capitalize()
    if "#" in n:
        return _enharmonic_sharp_to_flat[n]
    elif "x" in n:
        return enharmonic(n.replace("x", "#"))
    elif "is" in n:
        return enharmonic(n.replace("is", "#"))
    elif "b" in n:
        return _enharmonic_flat_to_sharp[n]
    elif "s" in n:
        return enharmonic(n.replace("s", "b"))
    elif "es" in n:
        return enharmonic(n.replace("es", "b"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Helper functions for Note, Chord, ...
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def addColumn(mtx, col):
    if not lib.isiterable(col):
        col = [col]*len(mtx)
    if isinstance(mtx[0], tuple):
        return [row+(elem,) for row, elem in zip(mtx, col)]
    elif isinstance(mtx[0], list):
        return [row+[elem] for row, elem in zip(mtx, col)]
    else:
        raise TypeError(f"mtx should be a seq. of tuples or lists, but got {mtx} ({type(mtx[0])})")


def fillColumns(rows: t.List[list], sentinel=None) -> t.List[list]:
    """
    Converts a series of rows with possibly unequeal number of elements per row
    so that all rows have the same length, filling each new row with elements
    from the previous, if they do not have enough elements (elements are "carried"
    to the next row)
    """
    maxlen = max(len(row) for row in rows)
    initrow = [0] * maxlen
    outrows = [initrow]
    for row in rows:
        lenrow = len(row)
        if lenrow < maxlen:
            row = row + outrows[-1][lenrow:]
        if sentinel in row:
            row = row.__class__(x if x is not sentinel else lastx for x, lastx in zip(row, outrows[-1]))
        outrows.append(row)
    # we need to discard the initial row
    return outrows[1:]


def asmidi(x) -> float:
    if isinstance(x, str):
        return n2m(x)
    elif isinstance(x, (int, float)):
        assert 0 <= x <= 200, f"Expected a midinote (0-127) but got {x}"
        return x
    elif hasattr(x, 'midi'):
        return x.midi
    raise TypeError(f"Expected a str, a Note or a midinote, got {x}")


def asfreq(n) -> float:
    """
    Convert a midinote, notename of Note to a freq.
    NB: a float value is interpreted as a midinote

    :param n: a note as midinote, notename or Note
    :return: a frequency taking into account the A4 defined in emlib.pitch
    """
    if isinstance(n, str):
        return n2f(n)
    elif isinstance(n, (int, float)):
        return m2f(n)
    elif hasattr(n, "freq"):
        return n.freq
    else:
        raise ValueError(f"cannot convert {n} to a frequency")


def notes2ratio(n1, n2, maxdenominator=16) -> Fraction:
    """
    find the ratio between n1 and n2

    n1, n2: notes -> "C4", or midinote (do not use frequencies)

    Returns: a Fraction with the ratio between the two notes

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
    f1, f2 = asfreq(n1), asfreq(n2)
    return Fraction.from_float(f1/f2).limit_denominator(maxdenominator)


def normalizeFade(fade:t.U[float, t.Tup[float, float]]) -> t.Tup[float, float]:
    """
    Returns (fadein, fadeout)
    """
    if isinstance(fade, tuple):
        if len(fade) != 2:
            raise IndexError(f"fade: expected a tuple or list of len=2, got {fade}")
        fadein, fadeout = fade
    elif isinstance(fade, (int, float)):
        fadein = fadeout = fade
    else:
        raise TypeError(f"fade: expected a fadetime or a tuple of (fadein, fadeout), got {fade}")
    return fadein, fadeout


def midicents(midinote: float) -> int:
    """
    Returns the cents to next chromatic pitch

    :param midinote: a (fractional) midinote
    :return: cents to next chromatic pitch
    """
    return int(round((midinote - round(midinote)) * 100))


def centsshown(centsdev, divsPerSemitone=None) -> str:
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
    divsPerSemitone = divsPerSemitone or config['show.semitoneDivisions']
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


def pngOpenExternal(path:str, wait=False) -> None:
    app = config.get(f'app.png')
    if not app:
        if wait:
            logger.debug("pngOpenExternal: called with wait=True," 
                         "but opening with standard app so can't wait")
        lib.open_with_standard_app(path)
        return
    cmd = f'{app} "{path}"'
    if wait:
        os.system(cmd)
    else:
        os.system(cmd + " &")


try:
    from IPython.core.display import display as jupyterDisplay
except ImportError:
    jupyterDisplay = pngOpenExternal


def setJupyterHookForClass(cls, func, fmt='image/png'):
    """
    Register func as a displayhook for class `cls`
    """
    if not insideJupyter:
        logger.debug("_setJupyterHookForClass: not inside IPython/jupyter, skipping")
        return
    import IPython
    ip = IPython.get_ipython()
    formatter = ip.display_formatter.formatters[fmt]
    return formatter.for_type(cls, func)


def imgSize(path:str) -> t.Tup[int, int]:
    """ returns (width, height) """
    import PIL
    im = PIL.Image.open(path)
    return im.size


def jupyterMakeImage(path: str):
    from IPython.core.display import Image
    scalefactor = config.get('show.scalefactor', 1.0)
    if scalefactor != 1.0:
        imgwidth, imgheight = imgSize(path)
        width = imgwidth*scalefactor
    else:
        width = None
    return Image(filename=path, embed=True, width=width)  # ._repr_png_()


def jupyterShowImage(path: str):
    img = jupyterMakeImage(path)
    return jupyterDisplay(img)


def pngShow(image, external=False):
    if external or not insideJupyter:
        pngOpenExternal(image)
    else:
        jupyterShowImage(image)


def m21JupyterHook(enable=True) -> None:
    """
    Set an ipython-hook to display music21 objects inline on the
    ipython notebook
    """
    if not insideJupyter:
        logger.debug("m21JupyterHook: not inside ipython/jupyter, skipping")
        return
    from IPython.core.getipython import get_ipython
    from IPython.core import display
    ip = get_ipython()
    formatter = ip.display_formatter.formatters['image/png']
    if enable:
        def showm21(stream):
            fmt = config['m21.displayhook.format']
            filename = str(stream.write(fmt))
            return display.Image(filename=filename)._repr_png_()

        dpi = formatter.for_type(m21.Music21Object, showm21)
        return dpi
    else:
        logger.debug("disabling display hook")
        formatter.for_type(m21.Music21Object, None)

