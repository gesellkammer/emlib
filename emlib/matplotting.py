import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from matplotlib import cm
import numpy as np
from emlib.lib import isiterable, pixels_to_inches
import typing as t


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
    out = default.copy()
    for key, value in kws.items():
        if key not in default:
            raise KeyError(f"Key {key} not in default profile")
        out[key] = value
    return out


_colormap = cm.cmap_d['jet']   # type: t.Callable[[float], t.Tuple[float, float, float, float]]


def _get(profile: dict, key:str, fallback:dict=defaultprofile, value=None) -> t.Any:
    if profile is not None:
        return profile.get(key, fallback.get(key, value))
    return fallback.get(key, value)


def drawLabel(ax: plt.Axes, x: float, y: float, text: str, size=None, alpha=None, profile=None) -> None:
    family = _get(profile, 'label_font')
    size = _fallback(size, profile, 'label_size')
    alpha = _fallback(alpha, profile, 'label_alpha')
    plt.text(x, y, text, ha="center", family=family, size=size, alpha=alpha)


def drawLine(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: float=None, 
             linestyle:str = 'solid', alpha: float=None, linewidth:float=None, label: str=None, 
             profile=None) -> None:
    """
    ax: plt.Axes  
        fig, ax = plt.subplots()
    
    linestyle - 'solid', 'dashed'
    label     - if given, a label with the given string is plotted next to the line
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


def _getcolor(color):
    if isinstance(color, tuple):
        return color
    return _colormap(color)
    

def _unzip(pairs):
    return zip(*pairs)


def drawConnectedLines(ax: plt.Axes, pairs: t.List[t.Tuple[float, float]], 
                       connectEdges=False, color=None, alpha:float=None, 
                       linewidth:float=None, label:str=None, linestyle:str=None, profile:dict=None) -> None:
    """
    pairs: a list of (x, y) pairs
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
    data: a list of (x0, y0, x1, y1)
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


def makeAxis(pixels: t.Tuple[int, int]=None) -> plt.Axes:
    # plt.subplots(figsize=(20, 10)) 
    if pixels is None:
        fig, ax = plt.subplots()
        return ax
    else:
        if not isinstance(pixels, tuple):
            raise TypeError(f"pixels should be of the form (x, y), got {pixels}")
        dpi = 96
        xinches = pixels_to_inches(pixels[0], dpi=dpi)
        yinches = pixels_to_inches(pixels[1], dpi=dpi)
        fig,ax = plt.subplots(figsize=(xinches, yinches), dpi=dpi)
        return ax

def _fallback(value, profile: dict, key: str):
    return value if value is not None else _get(profile, key)


def _fallbackColor(value, profile: dict, key: str) -> t.Tuple[float, float, float]:
    if value is not None:
        return _getcolor(value)
    return _getcolor(_get(profile, key))


def drawBracket(ax:plt.Axes, x0:float, y0:float, x1:float, y1:float, 
                label:str=None, color=None, linewidth:float=None, alpha:float=None, 
                profile:dict=None) -> None:
    linewidth = _fallback(linewidth, profile, 'linewidth')
    color = _fallback(color, profile, 'edgecolor')
    alpha = _fallback(alpha, profile, 'annotation_alpha')
    data = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    drawConnectedLines(ax, data, color=color, linewidth=linewidth, label=label, alpha=alpha)
    

def plotDurs(durs: t.List[float], y0=0.0, x0=0.0, height=1.0, labels:t.List[str]=None, color=None, 
             ax=None, groupLabel:str=None, profile:dict=None, stacked=False) -> plt.Axes:
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
