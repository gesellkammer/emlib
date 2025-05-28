"""
Routines to help draw shapes / labels within a matplotlib (pyplot) plot.
Implements the concept of a plotting profile, which makes it easier to
define sizes, colors, etc. for a series of elements.
"""
from __future__ import annotations
from functools import cache
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

import numpy as np

import typing as _t
if _t.TYPE_CHECKING:
    from matplotlib.colors import Colormap
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    _colort: _t.TypeAlias = float | tuple[float, float, float] | tuple[float, float, float, float]


defaultprofile = {
    'label_font': 'sans-serif',
    'label_size': 10,
    'label_alpha': 0.75,
    'line_alpha': 0.8,
    'line_style': 'solid',
    'edgecolor': 1,
    'facecolor': 0.8,
    'alpha': 0.75,
    'linewidth': 1,
    'annotation_color': (0, 0, 0),
    'annotation_alpha': 0.3,
    'autoscale': True,
    'background': (0, 0, 0),
    'colormap': 'jet'
}


def makeProfile(**kws):
    """
    Create a profile based on a default profile

    A profile is used to determine multiple defaults

    Example
    -------

    >>> makeProfile(
    ...     label_font="Roboto",
    ...     background=(10, 10, 10),
    ...     linewidth=2)
    """
    out = defaultprofile.copy()
    if kws:
        assert all(key in defaultprofile for key in kws), f"Unknown keys: {[k for k in kws if k not in defaultprofile]}"
        out |= kws
    return out


def drawLabel(ax: Axes, x: float, y: float, text: str, size=None, alpha=None,
              profile=defaultprofile) -> None:
    """
    Draw a text label at the given coordinates

    Args:
        ax: the plot axes
        x: x coordinate
        y: y coordinate
        text: the text
        size: size of the label. If given, overrides the profile's "label_size"
        alpha: if given, overrides the profile's "label_alpha"
        profile: the profile used (None = default)

    """
    family = profile['label_font']
    size = size if size is not None else profile['label_size']
    alpha = alpha if alpha is not None else profile['label_alpha']
    ax.text(x, y, text, ha="center", family=family, size=size, alpha=alpha)


def drawLine(ax: Axes, x0: float, y0: float, x1: float, y1: float,
             color: float = None, linestyle='solid', alpha: float = None,
             linewidth: float = None, label='', profile=defaultprofile, cmap='',
             autoscale: bool | None = None) -> None:
    """
    Draw a line from ``(x0, y0)`` to ``(x1, y1)``

    Args:
        ax: a Axes to draw on
        x0: x coord of the start point
        y0: y coord of the start point
        x1: x coord of the end point
        y1: y coord of the end point
        color: the color of the line as a value 0-1 within the colormap space
        linestyle: 'solid', 'dashed'
        alpha: a float 0-1
        label: if given, a label is plotted next to the line
        autoscale: autoscale axis if True
        profile: the profile (created via makeProfile) to use.

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> from emlib import matplotting
    >>> fig, ax = plt.subplots()
    >>> matplotting.drawLine(ax, 0, 0, 1, 1)
    >>> plt.show()
    """
    linewidth = linewidth if linewidth is not None else profile['line_width']
    alpha = alpha if alpha is not None else profile['line_alpha']
    color = color if color is not None else profile['edgecolor']
    X, Y = np.array([[x0, x1], [y0, y1]])
    assert linestyle in ('solid', 'dashed')
    colortup = _get_colormap(cmap or profile['colormap'])(color)
    line = mlines.Line2D(X, Y, lw=linewidth, alpha=alpha, color=colortup, linestyle=linestyle)
    ax.add_line(line)
    if label is not None:
        drawLabel(ax, x=(x0+x1)*0.5, y=y0, text=label, profile=profile)
    if autoscale or (autoscale is None and profile['autoscale']):
        autoscaleAxis(ax)


def _aslist(obj) -> list:
    if isinstance(obj, list):
        return obj
    return list(obj)


@cache
def _get_colormap(name: str) -> Colormap:
    return plt.get_cmap(name)


@cache
def _getcolor(color: _colort, colormap: str
              ) -> tuple[float, float, float, float]:
    if isinstance(color, tuple):
        return color if len(color) == 4 else color + (1,)

    return _get_colormap(colormap)(color)


def drawConnectedLines(ax: Axes,
                       pairs: list[tuple[float, float]],
                       connectEdges=False,
                       color: _colort = None,
                       alpha: float = None,
                       linewidth: float = None,
                       label='',
                       linestyle='',
                       profile=defaultprofile,
                       cmap=''
                       ) -> None:
    """
    Draw an open / closed poligon

    Args:
        ax: the plot axes
        pairs: a list of (x, y) pairs
        connectEdges: close the form, connecting start end end points
        color: the color to use. A float selects a color from the current color map
        alpha: alpha value of the lines
        linewidth: the line width
        label: an optional label to attach to the start of the lines
        linestyle: the line style, one of "solid", "dashed"
        profile: the profile used, or None for default
    """
    cmap = cmap or profile['colormap']
    if linewidth is None:
        linewidth = profile['line_width']
    if alpha is None:
        alpha = profile['line_alpha']
    if color is None:
        color = profile['edgecolor']
    if not linestyle:
        linestyle = profile['line_style']
    if connectEdges:
        pairs = pairs.copy()
        pairs.append(pairs[0])
    xs, ys = zip(*pairs)
    line = mlines.Line2D(xs, ys, lw=linewidth, alpha=alpha, color=_getcolor(color, cmap), linestyle=linestyle)
    ax.add_line(line)
    if label:
        avgx = sum(xs)/len(xs)
        avgy = sum(ys)/len(ys)
        drawLabel(ax, x=avgx, y=avgy, text=label, profile=profile)
    if profile['autoscale']:
        autoscaleAxis(ax)


def drawRect(ax: Axes, x0: float, y0: float, x1: float, y1: float,
             color: _colort, alpha: float = None, edgecolor: _colort = None, label='',
             profile=defaultprofile) -> None:
    """
    Draw a rectangle from point (x0, y0) to (x1, y1)

    Args:
        ax: the plot axe
        x0: x coord of the start point
        y0: y coord of the start point
        x1: x coord of the end point
        y1: y coord of the end point
        color: the face color
        edgecolor: the color of the edges
        alpha: alpha value for the rectangle (both facecolor and edgecolor)
        label: if given, a label is plotted at the center of the rectangle
        profile: the profile used, or None for default

    """
    cmap = profile['colormap']
    facecolor = color if color is not None else profile['facecolor']
    edgecolor = edgecolor if edgecolor is not None else profile['edgecolor']
    facecolor = _getcolor(facecolor, cmap)
    edgecolor = _getcolor(edgecolor, cmap)
    alpha = alpha if alpha is not None else profile['alpha']
    rect = Rectangle((x0, y0), x1-x0, y1-y0, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(rect)
    if label is not None:
        drawLabel(ax, x=(x0+x1)*0.5, y=(y0+y1)*0.5, text=label, profile=profile)
    if profile['autoscale']:
        autoscaleAxis(ax)


def drawRects(ax: Axes, data,
              facecolor: _colort = None,
              alpha: float = None,
              edgecolor: _colort = None,
              linewidth: float | None = None,
              profile=defaultprofile,
              autolim=True
              ) -> None:
    """
    Draw multiple rectangles

    Args:
        ax: the plot axes
        data: either a 2D array of shape (num. rectangles, 4), or a list of tuples
            (x0, y0, x1, y1), where each row is a rectangle
        facecolor: the face color
        edgecolor: the color of the edges
        alpha: alpha value for the rectangle (both facecolor and edgecolor)
        label: if given, a label is plotted at the center of the rectangle
        profile: the profile used, or None for default
        autolim: autoscale view
        linewidth: line width
    """
    facecolor = facecolor if facecolor is not None else profile['facecolor']
    edgecolor = edgecolor if edgecolor is not None else profile['edgecolor']
    linewidth = linewidth if linewidth is not None else profile['linewidth']
    cmap = profile['colormap']
    facecolor = _getcolor(facecolor, cmap)
    edgecolor = _getcolor(edgecolor, cmap)
    rects = []
    for coords in data:
        x0, y0, x1, y1 = coords
        rect = Rectangle((x0, y0), x1-x0, y1-y0)
        rects.append(rect)
    from matplotlib.collections import PatchCollection
    coll = PatchCollection(rects, linewidth=linewidth, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_collection(coll, autolim=True)
    if autolim:
        ax.autoscale_view()


def autoscaleAxis(ax: Axes) -> None:
    ax.relim()
    ax.autoscale_view(True,True,True)


def makeAxis(pixels: tuple[int, int] | None = None, dpi=96) -> Axes:
    """
    Create a plotting axes

    Args:
        pixels: the size of the plot, in pixels
        dpi: dots per inch

    Returns:
        the Axes

    """
    # plt.subplots(figsize=(20, 10))
    if pixels is None:
        fig, ax = plt.subplots()
        return ax

    if not isinstance(pixels, tuple):
        raise TypeError(f"pixels should be of the form (x, y), got {pixels}")

    import emlib.misc
    xinches = emlib.misc.pixels_to_inches(pixels[0], dpi=dpi)
    yinches = emlib.misc.pixels_to_inches(pixels[1], dpi=dpi)
    fig,ax = plt.subplots(figsize=(xinches, yinches), dpi=dpi)
    return ax


def drawBracket(ax: Axes, x0: float, y0: float, x1: float, y1: float,
                label='', color=None, linewidth: float = None, alpha: float = None,
                profile=defaultprofile) -> None:
    """
    Draw a bracket from (x0, y0) to (x1, y1)

    Args:
        ax: the plot axe
        x0: x coord of the start point
        y0: y coord of the start point
        x1: x coord of the end point
        y1: y coord of the end point
        color: the face color
        alpha: alpha value for the rectangle (both facecolor and edgecolor)
        label: if given, a label is plotted at the center of the rectangle
        linewidth: line width
        profile: the profile used, or None for default

    """
    if linewidth is None:
        linewidth = profile['linewidth']
    if color is None:
        color = profile['edgecolor']
    if alpha is None:
        alpha = profile['annotation_alpha']
    data = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    drawConnectedLines(ax, data, color=color, linewidth=linewidth, label=label, alpha=alpha)


def plotDurs(durs: list[float], y0=0.0, x0=0.0, height=1.0,
             labels: list[str] = None,
             color: _colort | None = None,
             ax: Axes = None,
             groupLabel='',
             profile=defaultprofile,
             stacked=False
             ) -> Axes:
    """
    Plot durations as contiguous rectangles

    Args:
        durs: the durations expressed in seconds
        y0: y of origin
        x0: x of origin
        height: the height of the drawn rectangles
        labels: if given, a label for each rectangle
        color: the color used for the rectangles
        ax: the axes to draw on. If not given, a new axes is created (and returned)
        groupLabel: a label for the group
        profile: the profile used, or None to use a default
        stacked: if True, the rectangles are drawn stacked vertically (the duration
            is still drawn horizontally). The result is then similar to a bars plot

    Returns:
        the plot axes. If *ax* was given, then it is returned; otherwise the new
        axes is returned.

    """
    if ax is None:
        ax = makeAxis()
    if color is None:
        color = profile['facecolor']
    if not stacked:
        x = x0
        data = []
        for i, dur in enumerate(durs):
            data.append((x, y0, x+dur, y0+height))
            x += dur
        drawRects(ax, data, facecolor=color)
        if groupLabel:
            sep = height * 0.05
            y1 = y0 + height
            x1 = x0 + sum(durs)
            drawBracket(ax, x0, y1+sep, x1, y1+sep*2, color=profile['annotation_color'])
            alpha = (profile['annotation_alpha'] +1) * 0.5
            drawLabel(ax, (x0+x1) * 0.5, y1 + sep, text=groupLabel, alpha=alpha)
    else:
        data = []
        y = y0
        for dur in durs:
            data.append((x0, y, x0+dur, y+height))
            y += height
        drawRects(ax, data, facecolor=color)
    return ax


def fig2data(fig: Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels

    Args:
        fig: a matplotlib figure

    Returns:
        a numpy 3D array of RGBA values
    """
    fig.canvas.draw()        # draw the renderer
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
