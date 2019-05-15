import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib import cm
import bpf4 as bpf

from emlib.typehints import *
from emlib.pitchtools import amp2db
from emlib import iterlib
import numpy as np

from .core import *
from .config import config


_colormap = cm.cmap_d['jet']   # type: Callable[[float], Tup[float, float, float, float]]


def is_step_ok(nodes:t.U[NodeList, list], start=0) -> bool:
    """
    Return True if step has been set for all nodes.
    """
    return all(node.step is not None for node in iterlib.flatten(nodes) if node.weight > 0)


def _draw_label(x:float, y:float, text:str, offset=-0.0) -> None:
    y = y + offset
    plt.text(x, y, text, ha="center", family='sans-serif', size=10, alpha=0.8)


def draw_line0(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: float,
              label: str=None, alpha: float=0.4) -> None:
    node_linewidth = config['plot.node.linewidth']
    X, Y = np.array([[x0, x1], [y0, y1]])
    line = mlines.Line2D(X, Y, lw=node_linewidth, alpha=alpha, color=_colormap(color))
    ax.add_line(line)
    if label:
        _draw_label((x0+x1)*0.5, y0, label)


def draw_line(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, plotinfo:dict,
              label: str=None) -> None:
    draw_line0(ax=ax, x0=x0, y0=y0, x1=x1, y1=y1, color=plotinfo['color'], alpha=plotinfo['alpha'],
               label=label)


db2alpha = bpf.linear(-90, 0.05, -60, 0.2, -24, 0.6, -6, 0.7)  # type: Callable[[float], float]


def amp2alpha(amp:float) -> float:
    return db2alpha(amp2db(amp))


def plotnodes(nodeseq:t.U[NodeList, list], callback=None, show=True, timerange=None) -> plt.Axes:
    """
    Plots the nodes of a NodeList as branches

    nodeseq:
        a list of Nodes, will be recursively converted to a Branch
    callback:
        a function of the form:
            callback(ax: plt.Axes, node:Node, nextnode:Opt[Node], plotinfo:Dict) -> bool
            (see the default function, draw_node_as_line, for reference)
        The callback should return False if it does not accept to draw the node,
        in which case the fallback routine is used. Anything else will assume that the
        node was drawn.
    """
    if not is_step_ok(nodeseq):
        raise ValueError("can't plot: step has not been set for all nodes")
    flatseq = list(iterlib.flatten(nodeseq))
    names = list(set(node.name for node in flatseq if node.weight > 0))
    fig, ax = plt.subplots()   # type: t.Tup[t.Any, plt.Axes]
    fallbackdraw = draw_node_default
    if callback is None:
        callback = fallbackdraw

    def getcolor(name:str) -> float:
        """
        Convert the name of a node to a value between 0-1
        """
        idx = names.index(name)
        value = idx / len(name)
        logger.debug(f"color for {name}: {value}")
        return value

    def draw_node(callback, ax: plt.Axes, node:Node, nextnode:Opt[Node]):
        plotinfo = {'alpha': amp2alpha(node.amp),
                    'color': getcolor(node.name)}
        accepted = callback(ax, node, nextnode, plotinfo)
        if accepted is False:
            fallbackdraw(ax, node, nextnode, plotinfo)

    def draw_nodes(flatseq, callback):
        allnodes = [node for node in flatseq if node.weight > 0]
        for node, nextnode in iterlib.pairwise(allnodes):
            draw_node(callback, ax, node, nextnode)
        if allnodes[-1].weight > 0:
            draw_node(callback, ax, allnodes[-1], None)

    draw_nodes(flatseq, callback)
    _plot_draw_branchlines(ax, nodeseq)
    _, maxx, miny, maxy = nodeseq_plotrange(flatseq)
    if not timerange:
        ax.set_xlim(0, maxx)
    else:
        ax.set_xlim(*timerange)
    ax.set_ylim(miny, maxy)
    if show:
        plt.show()
    return ax


def flatten_nodes(nodeseq:t.U[NodeList, t.List]) -> t.Iter[Node]:
    return nodeseq.flatview() if isinstance(nodeseq, NodeList) else iterlib.flatten(nodeseq)


def nodeseq_get_range(nodeseq) -> Tup[float, float, float, float]:
    """
    Returns (x0, x1, y0, y1)
    """
    #if isinstance(nodeseq, NodeList):
    #    return nodeseq.get_range()
    maxx = 0.0
    miny = 999999999999.0
    maxy = -miny
    for node in flatten_nodes(nodeseq):
        if node.end > maxx:
            maxx = float(node.end)
        if node.step >= 0:
            miny = min(miny, node.step, node.stepend)
            maxy = max(maxy, node.step, node.stepend)
    return (0.0, float(maxx), float(miny), float(maxy))


def nodeseq_plotrange(nodeseq, margin:U[str, float]='auto') -> Tup[float, float, float, float]:
    minx, maxx, miny, maxy = nodeseq_get_range(nodeseq)
    if margin == 'auto':
        dy = maxy - miny
        margin = max(dy * 0.2, 2)
    else:
        margin = max(margin, 2)
    return 0, maxx, miny - margin, maxy + margin


def draw_node_default(ax:plt.Axes, node:Node, nextnode:Node, plotinfo:Dict[str, Any]):
    """
    This is the default drawing callback. It assumes that the user
    has set .stepend on each node whenever a glissando is needed.
    """
    draw_line(ax, float(node.offset), node.step, float(node.end), node.stepend,
              label=node.name, plotinfo=plotinfo)
    return True  # <- we accepted to draw the node


def draw_node_gliss(ax:plt.Axes, node:Node, nextnode:Node, plotinfo:Dict[str, Any]):
    """
    This function can be passed to plotnodes as callback to handle
    'gliss' attribute in metadata. It can be used as an example
    for other callbacks
    """
    step1 = node.step
    allow_discontinuous_gliss = True
    if node.data.get('gliss', False):
        if node.end == nextnode.offset or allow_discontinuous_gliss:
            step1 = nextnode.step
    draw_line(ax, float(node.offset), node.step, float(node.end), step1,
              label=node.name, plotinfo=plotinfo)
    return True  # <- we accepted to draw the node


def draw_node_as_line(ax:plt.Axes, node:Node, nextnode:Node, plotinfo):
    """
    This is the simplest routine to draw a Node. It just draws a straight
    line, takes no account of the next node.
    """
    draw_line(ax, float(node.offset), node.step, float(node.end), node.step,
              plotinfo=plotinfo, label=node.name)
    return True  # <- we accepted to draw the node


def _plot_draw_branchlines(ax:plt.Axes, nodeseq:List[U[Node, List]]) -> None:
    """
    Traverses the nodeseq and draws lines between nodes belonging
    to the same branch and between branched children and their
    parents
    """
    def draw_branch_conn(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
        X, Y = np.array([[x0, x1], [y0, y1]])
        line = mlines.Line2D(X, Y, lw=2, alpha=0.2, color=_colormap(1))
        ax.add_line(line)

    def draw_node_conn(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
        X, Y = np.array([[x0, x1], [y0, y1]])
        line = mlines.Line2D(X, Y, lw=2, alpha=0.1, color=_colormap(1), linestyle="dashed")
        ax.add_line(line)

    def draw_lines(seq:List[U[Node, list]], offset:float, step:float):
        branch_conn_drawn = False
        for node in seq:
            if isinstance(node, list):
                draw_lines(node, offset, step)
                continue
            if node.weight <= 0:
                continue
            if not branch_conn_drawn:
                draw_branch_conn(ax, offset, step, float(node.offset), node.step)
                branch_conn_drawn = True
            else:
                if node.step != step:
                    draw_node_conn(ax, offset, step, float(node.offset), node.step)
            offset = float(node.end)
            step = node.step

    draw_lines(nodeseq, offset=0, step=0)
