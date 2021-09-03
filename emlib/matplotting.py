"""
Routines to help draw shapes / labels within a matplotlib (pyplot) plot.
Implements the concept of a plotting profile, which makes it easier to
define sizes, colors, etc. for a series of elements.
"""
from __future__ import annotations
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from matplotlib import cm
import numpy as np
from emlib.misc import isiterable, pixels_to_inches
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


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
    'background': (0, 0, 0)
}



def makeProfile(default=defaultprofile, **kws):
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
    out = default.copy()
    for key, value in kws.items():
        if key not in default:
            raise KeyError(f"Key {key} not in default profile")
        out[key] = value
    return out


_colormap = plt.get_cmap('jet')  


def _get(profile: dict, key:str, fallback:dict=defaultprofile, value=None):
    if profile is not None and key in profile:
        return profile[key]
    return fallback.get(key, value)


def drawLabel(ax: plt.Axes, x: float, y: float, text: str, size=None, alpha=None,
              profile=None) -> None:
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
    family = _get(profile, 'label_font')
    size = _fallback(size, profile, 'label_size')
    alpha = _fallback(alpha, profile, 'label_alpha')
    plt.text(x, y, text, ha="center", family=family, size=size, alpha=alpha)


def drawLine(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float,
             color: float=None, linestyle:str = 'solid', alpha: float=None,
             linewidth:float=None, label: str=None, profile=None) -> None:
    """
    Draw a line from ``(x0, y0)`` to ``(x1, y1)``

    Args:
        ax: a plt.Axes to draw on
        x0: x coord of the start point
        y0: y coord of the start point
        x1: x coord of the end point
        y1: y coord of the end point
        color: the color of the line as a value 0-1 within the colormap space
        linestyle: 'solid', 'dashed'
        alpha: a float 0-1
        label: if given, a label is plotted next to the line
        profile: the profile (created via makeProfile) to use. Leave None to
            use the default profile

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> from emlib import matplotting
    >>> fig, ax = plt.subplots()
    >>> matplotting.drawLine(ax, 0, 0, 1, 1)
    >>> plt.show()
    """
    linewidth = _fallback(linewidth, profile, 'line_width')
    alpha = _fallback(alpha, profile, 'line_alpha')
    color = _fallback(color, profile, 'edgecolor')
    X, Y = np.array([[x0, x1], [y0, y1]])
    assert linestyle in ('solid', 'dashed')
    line = mlines.Line2D(X, Y, lw=linewidth, alpha=alpha, color=_colormap(color), linestyle=linestyle)
    ax.add_line(line)
    if label is not None:
        drawLabel(ax, x=(x0+x1)*0.5, y=y0, text=label, profile=profile)
    if _get(profile, 'autoscale'):
        autoscaleAxis(ax)


def _aslist(obj) -> list:
    if isinstance(obj, list):
        return obj
    return list(obj)


def _getcolor(color: Union[float, tuple]) -> Tuple[float, float, float, float]:
    if isinstance(color, tuple):
        return color
    return _colormap(color)


def _unzip(pairs):
    return zip(*pairs)


def drawConnectedLines(ax: plt.Axes, 
                       pairs: List[Tuple[float, float]],
                       connectEdges=False, 
                       color: Union[float, tuple] = None, 
                       alpha: float = None,
                       linewidth: float = None, 
                       label: str = None, 
                       linestyle: str = None,
                       profile: dict = None
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
    linewidth = _fallback(linewidth, profile, 'line_width')
    alpha = _fallback(alpha, profile, 'line_alpha')
    color = _fallback(color, profile, 'edgecolor')
    linestyle = _fallback(linestyle, profile, 'line_style')
    if connectEdges:
        pairs = pairs + (pairs[0])
    color = _getcolor(color)
    xs, ys = _unzip(pairs)
    line = mlines.Line2D(xs, ys, lw=linewidth, alpha=alpha, color=_getcolor(color), linestyle=linestyle)
    ax.add_line(line)
    if label is not None:
        avgx = sum(xs)/len(xs)
        avgy = sum(ys)/len(ys)
        drawLabel(ax, x=avgx, y=avgy, text=label, profile=profile)
    if _get(profile, 'autoscale'):
        autoscaleAxis(ax)


def drawRect(ax: plt.Axes, x0:float, y0:float, x1:float, y1:float,
             color=None, alpha:float=None, edgecolor=None, label:str=None,
             profile:dict=None) -> None:
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
    facecolor = _fallback(color, profile, 'facecolor')
    edgecolor = _fallback(edgecolor, profile, 'edgecolor')
    facecolor = _getcolor(facecolor)
    edgecolor = _getcolor(edgecolor)
    alpha = alpha if alpha is not None else _get(profile, 'alpha')
    rect = Rectangle((x0, y0), x1-x0, y1-y0, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
    ax.add_patch(rect)
    if label is not None:
        drawLabel(ax, x=(x0+x1)*0.5, y=(y0+y1)*0.5, text=label, profile=profile)
    if _get(profile, 'autoscale'):
        autoscaleAxis(ax)


def _many(value, numitems:int, key:str=None, profile:dict=None) -> list:
    if isiterable(value):
        return value
    elif value is None:
        return [_get(profile, key)] * numitems
    return [value] * numitems


def drawRects(ax: plt.Axes, data, facecolor=None, alpha:float=None, edgecolor=None,
              linewidth:float=None, profile:dict=None, autolim=True) -> None:
    """
    Draw multiple rectangles

    Args:
        ax: the plot axes
        data: either a 2D array of shape (num. rectangles, 4), or a list of tuples
            (x0, y0, x1, y1), where each row is a rectangle
        color: the face color
        edgecolor: the color of the edges
        alpha: alpha value for the rectangle (both facecolor and edgecolor)
        label: if given, a label is plotted at the center of the rectangle
        profile: the profile used, or None for default
        autolim: autoscale view
    """
    facecolor = _fallbackColor(facecolor, profile, key='facecolor')
    edgecolor = _fallbackColor(edgecolor, profile, key='edgecolor')
    linewidth = _fallback(linewidth, profile, 'linewidth')
    rects = []
    for coords in data:
        x0, y0, x1, y1 = coords
        rect = Rectangle((x0, y0), x1-x0, y1-y0)
        rects.append(rect)
    coll = PatchCollection(rects, linewidth=linewidth, alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_collection(coll, autolim=True)
    if autolim:
        ax.autoscale_view()


def autoscaleAxis(ax: plt.Axes) -> None:
    ax.relim()
    ax.autoscale_view(True,True,True)


def makeAxis(pixels: Tuple[int, int]=None, dpi=96) -> plt.Axes:
    """
    Create a plotting axes

    Args:
        pixels: the size of the plot, in pixels
        dpi: dots per inch

    Returns:
        the plt.Axes

    """
    # plt.subplots(figsize=(20, 10))
    if pixels is None:
        fig, ax = plt.subplots()
        return ax

    if not isinstance(pixels, tuple):
        raise TypeError(f"pixels should be of the form (x, y), got {pixels}")
    xinches = pixels_to_inches(pixels[0], dpi=dpi)
    yinches = pixels_to_inches(pixels[1], dpi=dpi)
    fig,ax = plt.subplots(figsize=(xinches, yinches), dpi=dpi)
    return ax


def _fallback(value, profile: dict, key: str):
    return value if value is not None else _get(profile, key)


def _fallbackColor(value, profile: dict, key: str) -> Tuple[float, float, float]:
    if value is not None:
        return _getcolor(value)
    return _getcolor(_get(profile, key))


def drawBracket(ax:plt.Axes, x0:float, y0:float, x1:float, y1:float,
                label:str=None, color=None, linewidth:float=None, alpha:float=None,
                profile:dict=None) -> None:
    """
    Draw a bracket from (x0, y0) to (x1, y1)

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
    linewidth = _fallback(linewidth, profile, 'linewidth')
    color = _fallback(color, profile, 'edgecolor')
    alpha = _fallback(alpha, profile, 'annotation_alpha')
    data = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    drawConnectedLines(ax, data, color=color, linewidth=linewidth, label=label, alpha=alpha)


def plotDurs(durs: List[float], y0=0.0, x0=0.0, height=1.0, labels:List[str]=None,
             color=None, ax=None, groupLabel:str=None, profile:dict=None, stacked=False
             ) -> plt.Axes:
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
    numitems = len(durs)
    labels = labels if isiterable(labels) else [labels]*numitems
    color = _fallbackColor(color, profile, 'facecolor')
    if not stacked:
        x = x0
        data = []
        for i, dur in enumerate(durs):
            data.append((x, y0, x+dur, y0+height))
            x += dur
        drawRects(ax, data, facecolor=color)
        if groupLabel is not None:
            sep = height * 0.05
            y1 = y0 + height
            x1 = x0 + sum(durs)
            drawBracket(ax, x0, y1+sep, x1, y1+sep*2, color=_get(profile, 'annotation_color'))
            alpha = (_get(profile, 'annotation_alpha')+1)*0.5
            drawLabel(ax, (x0+x1) * 0.5, y1 + sep, text=groupLabel, alpha=alpha)
    else:
        data = []
        y = y0
        for dur in durs:
            data.append((x0, y, x0+dur, y+height))
            y += height
        drawRects(ax, data, facecolor=color)
    return ax
