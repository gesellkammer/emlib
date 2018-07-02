import logging
from quicktions import Fraction
from matplotlib import pyplot as plt
import random
import inspect
import re
from typing import NamedTuple
from emlib import iterlib
from emlib.pitch import amp2db
from emlib.lib import allequal, makereplacer
from emlib.typehints import U, Opt, List, Dict, Tup, Any, Iter, Callable
import emlib.typehints as t
from .config import APPNAME


T = t.TypeVar("T")

BranchId = Tup[int, ...]

BRANCHBEGIN = "["
BRANCHEND   = "]"
PUSHSTATE   = "<"   # reset age without creating a branch
POPSTATE    = ">"

# --------------------------------------------------------------
#
# An implementation of Lindenmayer Systems
#
# --------------------------------------------------------------


logger = logging.getLogger(APPNAME.replace(":", "."))

_reserved_opcodes = set(r'{ } [ ] + - ! @ * / % & $ > < %'.split())

# Helper functions

def _aslist(seq:Iter) -> List:
    if isinstance(seq, list):
        return seq
    return list(seq)


def _getfields(obj):
    return [attr for attr in dir(obj) if not attr.startswith("_")]


def branchid2str(branchid: BranchId) -> str:
    if branchid is None:
        return ":?"
    return ":" + ":".join(map(str, branchid))


def _ifset(x, default, unset=None):
    return default if x is unset else x


class Letter:
    def __init__(self, name: str, weight: t.Rat=1, **data) -> None:
        self.name: str = name
        self.weight = asFraction(weight)
        self.data: Dict[str, Any] = data.copy() if data else {}

    def __repr__(self):
        datastr = f", data={self.data}" if self.data else ""
        return f"Letter({self.name}, weight={self.weight}{datastr})"

    def set(self, attr, val):
        if attr == 'weight':
            self.weight = val
        elif attr == 'name':
            self.name = val
        else:
            self.data[attr] = val

    def override(self, **kws) -> 'Letter':
        """
        Returns a new Letter with values overriden from d
        """
        out = self.copy()
        for attr, val in kws.items():
            out.set(attr, val)
        return out

    def copy(self):
        return Letter(name=self.name, weight=self.weight, **self.data)


class Constant(Letter):
    def __init__(self, name:str):
        super().__init__(name, 0)

    def __repr__(self):
        return f"Constant({self.name})"


def asFraction(x: U[t.Rat, str]) -> Fraction:
    if isinstance(x, Fraction):
        return x
    return Fraction(x).limit_denominator(999999999)


def _normalize_alphabet(alphabet: dict) -> Dict[str, Letter]:
    out: Dict[str, Letter] = {}
    if isinstance(alphabet, str):
        raise ValueError("deprecated")
    if isinstance(alphabet, dict):
        for name, definition in alphabet.items():
            if isinstance(definition, Letter):
                letter = definition
            elif isinstance(definition, (int, float, Fraction)):
                letter = Letter(name=name, weight=asFraction(definition))
            elif isinstance(definition, dict):
                weight = definition.pop('weight', 1)
                letter = Letter(name=name, weight=weight, **definition)
            elif isinstance(definition, str):
                # 'A2': 'A(h=2)'    takes A as prototype, overrides its values
                lettername, params = _parse_letter(definition)
                letter = out.get(lettername)
                if not letter:
                    raise KeyError("For key {name} asked to override {definition}, but {definition} is not defined")
                if params:
                    letter = letter.override(name=name, **params)
                else:
                    letter = letter.override(name=name)
            else:
                raise TypeError("Expected a dict, a number defining the weight, or a Letter, "
                                f"got {definition} of type {type(definition)}")
            out[name] = letter
    else:
        raise TypeError("alphabet should be either:"
                        "- a dict of the form {'A': Letter(...)"
                        "- a dict of the form {'A': {'weight': ..., 'myattr':myvalue}}"
                        "- a string of the form A:2 B:1 ... <Name>:<Weight>")
    return out


class Node:
    _fields = "name weight dur branch age agerel step stepend offset amp data".split()
    __slots__ = "name weight dur branch age agerel step _stepend offset amp data".split()

    def __init__(self,
                 name:str,
                 weight: t.Rat = 1,
                 dur: Fraction = -1,
                 branch: BranchId = None,
                 age: int = 0,
                 agerel: int = 0,
                 step: float = -1,
                 offset: Fraction = 0,
                 amp: float = 1,
                 data: Dict[str, Any] = None,
                 stepend: float = -1) -> None:
        self.name = name
        self.weight = asFraction(weight)
        self.dur = asFraction(dur)
        self.branch = branch
        self.age = age
        self.agerel = agerel
        self.step = step
        self.offset = asFraction(offset)
        self.amp = amp
        self.data = data if data is not None else {}
        self._stepend = stepend

    def __repr__(self):
        n = self
        w = f"{n.weight.numerator}/{n.weight.denominator}"
        if self.stepend != self.step:
            y = f"{n.step}-{n.stepend}"
        else:
            y = str(n.step)
        d = "data=" + _dict2str(n.data) if n.data else ""
        t0 = float(n.offset)
        return f"( {n.name} {w} dur={float(n.dur):.3f} {n.branch} age={n.age} ({n.agerel}) {y} t0={t0:.3f} {n.amp} {d} )"

    def set(self, **params) -> 'Node':
        fields = self.fields()
        for attr, value in params.items():
            if attr in fields:
                setattr(self, attr, value)
            else:
                self.data[attr] = value
        return self

    @property
    def stepend(self):
        if self._stepend > 0:
            return self._stepend
        return self.step

    @stepend.setter
    def stepend(self, value):
        self._stepend = value

    def clone(self, name=None, weight=None, dur=None, branch=None, age=None, agerel=None,
              step=None, stepend=None, offset=None, amp=None, data=None) -> 'Node':
        """
        Clone this Node with overriden values.

        NB: in the case of data, the given dictionary is used to override the original. If you
            want to replace data with a new dict, just do:
            node2 = node.clone()
            node2.data = newdata
        """
        d = self.data.copy()
        if data is not None:
            d.update(data)
        return Node(name if name is not None else self.name,
                    weight  = weight if weight is not None else self.weight,
                    dur     = dur if dur is not None else self.dur,
                    branch  = branch if branch else self.branch,
                    age     = age if age is not None else self.age,
                    agerel  = agerel if agerel is not None else self.agerel,
                    step    = step if step is not None else self.step,
                    stepend = stepend if stepend is not None else self._stepend,
                    offset  = offset if offset is not None else self.offset,
                    amp     = amp if amp is not None else self.amp,
                    data    = d)

    @property
    def end(self) -> Fraction:
        return self.offset + self.dur if self.dur is not None else None

    def isrest(self) -> bool:
        return self.amp == 0

    def is_branchnode(self) -> bool:
        return self.name[0] in (BRANCHBEGIN, BRANCHEND)

    def dump(self):
        print(self.__repr__())

    def astuple(self) -> Tup:
        return (self.name, self.weight, self.dur, self.branch, self.age, self.agerel, self.step, self.stepend,
                self.offset, self.amp, self.data)

    @classmethod
    def fields(cls):
        return cls._fields


def _tokenize(s: U[str, List[str]]) -> List[str]:
    if isinstance(s, list):
        return s
    tokens = re.findall(r">|<|\[|\]|[^\(\) \[\]<>]+(?:\([^\)]+\))?", s)
    return tokens


def _flatten_subst(subst: U[str, List[U[str, Node]]]) -> List[U[str, Node]]:
    out = []
    if isinstance(subst, str):
        subst = _tokenize(subst)
    for sub in subst:
        if isinstance(sub, str):
            out.extend(_tokenize(sub))
        elif isinstance(sub, Node):
            out.append(sub)
        elif isinstance(sub, list):
            out.extend(_flatten_subst(sub))
        else:
            raise TypeError(sub)
    return out

_T = t.TypeVar("_T", bound='NodeList')

class NodeList(list):

    def __init__(self, seq, update=True) -> None:
        """
        update: if True, update node values like branch, offset
        """
        super().__init__()
        if seq:
            self.extend(seq)
            # self.extend(copy.deepcopy(list(seq)))
        if update:
            self.update()
        self._drawnode = None
        self._flatview = None

    def clone(self, root:Iter[U[Node, list]], update=True) -> 'NodeList':
        """
        Create a new NodeList with the given seq, using already set parameters
        in this NodeList

        At the moment this includes:
        * plotting callback set via set_plot_callback
        """
        out = self.__class__(root, update=update)
        out.set_plot_callback(self._drawnode)
        return out

    def get_range(self) -> Tup[Fraction, Fraction, float, float]:
        """
        Returns (x0, x1, y0 ,y1)
        """
        x0 = asFraction(0)
        x1 = asFraction(self.duration())
        y0 = float("inf")
        y1 = -y0
        for node in self.flatview():
            y0 = min(y0, node.step, node.stepend)
            y1 = max(y1, node.step, node.stepend)
        return x0, x1, y0, y1

    def _invalidate_cache(self):
        self._flatview = None

    def append(self, other):
        self._invalidate_cache()
        return super().append(other)

    def extend(self, other):
        self._invalidate_cache()
        return super().extend(other)

    def set_plot_callback(self, callback) -> None:
        """
        callback: a func. similar to the callback accepted by .plot
        """
        self._drawnode = callback

    def flat(self:_T, update=True) -> _T:
        return self.clone(list(iterlib.flatlist(self)), update=update)

    def dump(self) -> None:
        return dumpnodes(self)

    def tabulate(self, fmt='ascii') -> str:
        nodes = [node.astuple() for node in self.flatview()]
        import tabulate

        def ipython_display_html(htmlstr):
            from IPython import display
            return display.HTML(htmlstr)

        tablefmt, action = {
            'ascii': ('simple', print),
            'html': ('html', ipython_display_html),
        }.get(fmt, 'simple')
        headers = Node.fields()
        s = tabulate.tabulate(nodes, tablefmt=tablefmt, headers=headers)
        return action(s)

    def split(branch: _T, t) -> Tup[_T, _T]:
        """
        Slice branch at time t, returns left and right slices
        """
        def cb_right(n, st):
            if n.offset > t or n.weight == 0:
                return n.clone()
            if n.offset + n.dur < t:
                return n.clone(dur=0, name="r", amp=0)
            elif n.offset < t:
                return n.clone(offset=t, dur=n.dur + n.offset - t)

        def cb_left(n, st):
            if n.offset > t:
                return "SKIP"
            if n.offset < t <= n.offset + n.dur:
                return n.clone(dur=t - n.offset)
            return n.clone()

        right = branch.transform(cb_right)
        left = branch.transform(cb_left)
        return left, right

    def homogendur(self:_T, amount:float=1.0) -> _T:
        """
        amount: float between 0 and 1
            As `amount` gets closer to 1, the more homogeneous the durations become,
            instead of fitting inside the duration of the parent

        Returns a copy of this NodeList with changed durations

        Todo: add the possibility to define the curve (now it is fixed to linear)
        """
        maxdur = max(node.dur for node in self.flatview())

        def changedur(node, _):
            if node.weight == 0:
                return node
            dur = node.dur + (amount * (maxdur - node.dur))
            return node.clone(dur=dur)

        return self.transform(changedur)

    def fitdur(self:_T, dur:t.Rat, excludefunc:Callable[[Node], bool]=None) -> _T:
        """
        Return a modified version of this NodeList so that the resulting
        duration equals `dur`

        dur        : resulting duration
        excludefunc: a function (node) -> book, returning True if the given Node
                     should be excluded from the transformation
        """
        if excludefunc is None:
            return self.scaledur(dur/self.duration())
        durmut = 0
        durfix = 0
        for node in self.flatview():
            if node.weight > 0:
                if excludefunc(node):
                    durfix += node.dur
                else:
                    durmut += node.dur
        factor = (dur - durfix) / durmut
        return self.scaledur(factor, excludefunc)

    def scaledur(self:_T, factor:U[float, Fraction], excludefunc:Callable[[Node], bool]=None) -> _T:
        factor = Fraction(factor)
        if excludefunc is None:
            cback = lambda node, st: node if node.weight == 0 else node.clone(dur=node.dur*factor)
        else:
            cback = lambda node, st: node if node.weight == 0 or excludefunc(node) else node.clone(dur=node.dur*factor)
        return self.transform(cback)

    def transform(self:_T, callback, state={}, skipopcodes=False) -> _T:
        """
        Apply a list of transforms to this Branch, recursively. Returns the transformed Branch
        Can also be used to filter unwanted notes, or to break the tree at specific points

        callback:
            a function of the form (node:Node, state:dict) -> Node|"STOP"|"SKIP"|None where:
            - if a node is returned, this node replaces the original node
            - if "STOP" is returned, iteration is stopped at this point in the branch
            - if "SKIP" is returned, the node is skipped
            - if None is returned, the node is appended as is

            state has no predefined keys. You can use state to save properties (for instance,
            an opcode can save something to `state`, which is then applied to a node)

        state:
            the initial state. Here you can set any key that will be used
            later within the transform callback
        """
        return self.clone(transform(self, callback, initialstate=state, skipopcodes=skipopcodes))

    def flip_time(self, callback=None) -> 'NodeList':
        """
        Return a reversed version of self

        callback:
            (original_node, reversed_node) -> reversed_node_corrected
            The callback is given for cases where metadata (.data) needs to be modified
            when a node is reversed. It is save to modify reversed_node in place

        """
        fl = self.flat(update=False)
        dur = self.duration()
        nodes = []
        for n in fl:
            n2 = n.clone(offset=dur-n.end, step=n.stepend, stepend=n.step)
            if callback:
                n2 = callback(n, n2)
            nodes.append(n2)
        nodes.sort(key=lambda node: node.offset)
        return NodeList(nodes, update=False)

    def setstep(self:_T, start=0, base=2) -> _T:
        """
        Modify the "step" attribute of a node following the + and - tokens
        The step is depth-dependent, determined by the base param.

                         1
        stepamount = -----------
                     base**depth

        Base 2               Base 3

        depth    mul         depth  mul
            0      1             0    1
            1      0.5           1   1/3
            2      0.25          2   1/9
            3      0.125         3   1/27
        """
        def callback(node, state):
            if node.name[0] in "+-":
                tok, step = parse_token(node.name, 1)
                step *= 1 / (base ** (len(node.branch) - 1))
                if tok == '-':
                    step = -step
                state['step'] = state.get('step', 0) + step
            return node if node.weight == 0 else node.clone(step=state.get('step', 0))

        return self.transform(callback, state={'step': start})

    def makerests(self: _T, matchcallback=None) -> _T:
        """
        Converts to rest any node which is matched by matchcallback
        Returns a new NodeList

        matchcallback:
            a function of the form (node) -> bool, which takes a Node
            and returns True if this node should be a rest

        """
        if matchcallback is None:
            matchcallback = lambda node: node.name.startswith("r")

        def callback(node, _):
            if matchcallback(node):
                return node.clone(amp=0)
            return node
        return self.transform(callback)

    def flatview(self) -> Iter[Node]:
        if self._flatview is None:
            self._flatview = list(iterlib.flatlist(self))
        return self._flatview

    def duration(self):
        maxtime = 0
        for node in self.flatview():
            if node.weight > 0 and node.end > maxtime:
                maxtime = node.end
        return maxtime

    def plot(self, callback=None, timerange=None, show=True) -> plt.Axes:
        """
        plot this NodeList

        callback:
            a function of the form (axis: pyplot.Axes, node:Node, nextnode:Opt[Node]) -> accepted:bool
            If the callback accepts the request, it draws the given node on the given
            axes and returns True. Otherwise it returns False and a default routine
            is used to draw a straight line between at node.step from node.offset to node.end
            The nextnode is also given (it can be used to implement a glissando between nodes, for example)

        """
        from .plot import plotnodes
        callback = callback or self._drawnode
        return plotnodes(self, callback=callback, show=show, timerange=timerange)

    def update(self) -> None:
        """
        Updates branch, age and offset of each node according to current configuration of the tree.
        Does this INPLACE
        """

        state0 = {
            'branch': (1,),
            'age': 0,
            'offset': 0
        }
        stack = []
        branch2numsubs = {}  # type: Dict[BranchId, int]

        def new_subbranch(branch: BranchId):
            assert isinstance(branch, tuple) and all(isinstance(sub, int) for sub in branch)
            numsubs = branch2numsubs.get(branch, 1)
            branch2numsubs[branch] = numsubs + 1
            return branch + (numsubs,)

        def inner(seq:list, state:dict):
            for node in seq:
                if isinstance(node, list):
                    state['age'] += 1
                    inner(node, state)
                    state['age'] -= 1
                    continue
                name = node.name
                node.branch = state['branch']
                node.offset = state['offset']
                # node.age = state['age']
                if name == BRANCHBEGIN:
                    stack.append(state.copy())
                    state['branch'] = new_subbranch(state['branch'])
                    state['age'] = 0
                elif name == BRANCHEND:
                    if stack:
                        # if the stream was sliced irregularly, it can happen that the parity between
                        # open brackets is lost
                        state = stack.pop()
                else:
                    state['offset'] += node.dur
        inner(self, state0)

    def branched(self) -> 'Branch':
        """
        Returns a NodeList constructed as flat branches so that each
        node is either a Node or a list of Nodes representing a branch
        """
        out = Branch(self)
        out.set_plot_callback(self._drawnode)
        return out

    def isflat(self) -> bool:
        for node in self:
            if not isinstance(node, Node):
                return False
        return True

    # <<< These functions are put here in order to document possible uses of .transform >>>

    def remove(self:_T, key) -> _T:
        """
        Example

        seq2 = seq.remove(lambda node: node.age > node.data['maxage'])
        """
        def callback(node, _):
            skip = key(node)
            return "SKIP" if skip else node
        return self.transform(callback)

    def cut(self:_T, key) -> _T:
        """
        Example

        seq2 = seq.cut(lambda node: node.data['maxage'] >= 5)
        """
        def callback(node, _):
            cut = key(node)
            return "STOP" if cut else node

        return self.transform(callback)

# def rebranch(nodes) -> 'Branch':
#     out = []
#     stack = []
#     branchBeginNode = lambda: Node(BRANCHBEGIN, weight=0, dur=0)
#     branchEndNode = lambda: Node(BRANCHEND, weight=0, dur=0)
#     lastbranch = next(iterlib.flatten(nodes)).branch
#     for node in iterlib.flatten(nodes):
#         if node.branch != lastbranch:
#             if issubbranch(node.branch, lastbranch):
#                 stack.append(lastbranch)
#                 out.append(branchBeginNode())
#


class Branch(NodeList):
    """
    A Branch is a list of either Nodes or Branches
    """
    
    def __init__(self, root:Iter[U[Node, list]], update=True) -> None:
        # don't assume anything, about root, make branches again following BRANCHBEGIN/END
        root = _branched(root)
        super().__init__(root, update=update)
        if update:
            self.update()

    def flat(self, update=True) -> 'Branch':
        """ Return a flat version of this branch traversed depth-first """
        nodes = [node.clone() for node in self.flatview() if node.name not in (BRANCHBEGIN, BRANCHEND)]
        return self.clone(nodes, update=update)

    def update(self) -> None:
        """
        This should be called whenever the nodes are changes in place,
        to update offset and branch of each node
        """
        branch2numsubs = {}  # type: Dict[BranchId, int]
        state0 = {
            'branch': (1,),
            'offset': 0
        }

        def new_subbranch(branch):
            numsubs = branch2numsubs.get(branch, 1)
            branch2numsubs[branch] = numsubs + 1
            return branch + (numsubs,)

        def inner(seq, state):
            for node in seq:
                if isinstance(node, list):
                    newstate = state.copy()
                    newstate['branch'] = new_subbranch(state['branch'])
                    inner(node, newstate)
                else:
                    node.branch = state['branch']
                    node.offset = state['offset']
                    state['offset'] += node.dur
        inner(self, state0)

    def window(self,
               winsize: int,
               callback: Callable[..., U[List[Node], Node, None]],
               skipopcodes=False,
               samebranch=True
               ) -> 'Branch':
        """
        Create a new NodeList by iterating through windows of nodes

        winsize:
            the window size
        skipopcodes:
            true if nodes with weight == 0 should be skipped
        samebranch:
            if true, the callback will only be called with nodes belonging to the same branch
        callback:
            A function of the form (node0, node1, ...) -> node | nodes
            The number of arguments should match winsize. The function can:
            * modify any of the nodes:
                Return a list with the modified notes. The number of returned nodes should match winsize
            * no modification:
                Return None
            * given A, B, C → A B joined (the same would be valid for B C -> BC):
                Return [AB, C]
            * A B C should be joined:
                Return ABC
            * skip node:
                Given A, B, C → return [None, B, C]

        Example callback: join nodes with same name and same step belonging to same branch

        def callback(n0, n1):
            if n0.name == n1.name and n0.step == n1.step and n0.branch == n1.branch:
               return n0.clone(dur=node0.dur+node1.dur)
        """
        nodes = window(winsize, self, callback=callback, skipopcodes=skipopcodes,
                       samebranch=samebranch)
        return self.clone(nodes)


def _func_numargs(func):
    """
    Returns the number of args of func
    """
    return len(inspect.getfullargspec(func).args)


class _BranchTransform:
    def apply(self, branch:'Branch') -> 'Branch':
        pass

class NodeTransform(_BranchTransform):
    def __init__(self, func, state=None, skipopcodes=False):
        self.func = func
        self.state = state
        self.skipopcodes = skipopcodes

    def __iter__(self):
        return iter((self.func, self.state))

    def apply(self, branch:'Branch') -> 'Branch':
        return branch.transform(self.func, state=self.state, skipopcodes=self.skipopcodes)


class WindowTransform(_BranchTransform):
    def __init__(self, func, skipopcodes=False):
        self.func = func
        self.skipopcodes = self.skipopcodes

    def numargs(self):
        return _func_numargs(self.func)

    def apply(self, branch:'Branch') -> 'Branch':
        return branch.window(winsize=self.numargs(), callback=self.func, skipopcodes=self.skipopcodes)


def _transform_state_() -> NodeTransform:
    """
    Handle the @ operator to modify state explicitely

    This is a builtin transform and is always supported, making the @ operator a
    reserved operator
    """
    def _(node:Node, state:Dict[str, Any]):
        if node.name[0] == "@":
            tok, args, kws = parse_token_argskws(node.name)
            if args:
                raise ValueError("the @ operator only accepts keyword arguments")
            print(">>>>>>>>>>>>>>>> @", args, kws)
            state.update(kws)
        return node
    return NodeTransform(_)
_transform_state = _transform_state_()


class LSystem:
    def __init__(self, rules, axiom, alphabet={}, plotcallback=None, transforms=None, makerests=True):
        # type: (dict, str, U[str, Dict[str, U[t.Rat, t.Dict]]], Callable) -> None
        """
        rules:
            A dictionary of rules, where each item has the form:
            '<key>': '<replacement>'

            key:
                * In the most common way, this is a token (a string without spaces), like "A".
                  Each time this token is found in an axiom, it will be substituted by the
                  corresponding rule
                * For more complex cases, a callback can be passed. The callback has to have the
                  form (node, nodes, idx) -> bool, where:
                  * node: the current node to be substituted
                  * nodes: all the nodes of the axiom
                  * idx: the index within nodes which corresponds to node (node == nodes[idx])
                  For some shortcuts to define callbacks, look at `matchnode`, `matchpost`, `matchprepost`
                  A callback should return True if it matches the current node. Callback keys have
                  priority over string keys.
            rule:
                the replacement after one generation (for ex.: "A [+ B ]")
                NB: each token must be separated by a space

                A rule is either a string of opcodes (like "A", "B", "{", "+")
                or a list of opcdoes in the form ["A", "B", "[", "+"]

                Advanced Use:
                    In place of any opcode, a DelayedOpcode can be used, which can
                    return a different opcode with every evaluation
                    For example, Rnd("+", 1.5, 2.5) will return variations of "+"
                    with a different value each time (+1.5, +2.3, ...)

        axiom:
            The initial state, for. ex: "A"

        alphabet:
            Used to define special characteristics of each Letter, like the weight

            Example 1: only weight is defined
                alphabet = "A:2 B:3"
                This sets the weight of A and B to 2 and 3. The weights are used to
                distribute the duration of the parent across the children

            Example 2: metadata
                alphabet = {
                    'A': 2,
                    'B': {'weight': '1/2', 'meta': {'gliss'=True}}
                }
        transforms:
            a list of callbacks called after each generation to modify the generated nodes.
            Each callback has the form (node0, node1, ...) -> nodes, similar to the
            callback passed to the .window method. The length of the window depends
            on the number of arguments of the callback

        """
        assert isinstance(rules, dict)
        assert isinstance(axiom, (str, list, tuple))
        assert alphabet is None or isinstance(alphabet, (str, dict))
        self.rules = {var: Rule.parserule(rule) for var, rule in rules.items()}
        for key, rule in self.rules.items():
            assert isinstance(key, str) or callable(key), key
            assert isinstance(rule, Rule)
        self.axiom = _tokenize(axiom) if isinstance(axiom, str) else axiom
        self.alphabet = _generate_alphabet(self.rules)
        if alphabet:
            self.alphabet.update(_normalize_alphabet(alphabet))
        self._plotcallback = plotcallback
        self.step_transforms: List[Callable] = [_transform_state]
        self._makerests = True
        if transforms is not None:
            self.step_transforms.extend(transforms)

    def _generate(self, generation:int, dur:t.Rat=1) -> NodeList:
        """
        Generates the nodes of a given generation.

        dur: the duration assigned to the axiom

        For ex, A->AA, B->B+A with axiom AB
        A                B
        (A      A      ) (B  +  A )
        ((A  A) (A  A) ) ((B+A) (A A))
        """
        dur = asFraction(dur)
        nodes = _distribute_duration(dur, asNodes(self.axiom, self.alphabet))
        
        def apply_transforms(nodes, transforms, alphabet):
            for transf in transforms:
                if isinstance(transf, NodeTransform):
                    nodes = transform(nodes, transf.func, initialstate=transf.state)
                elif isinstance(transf, WindowTransform):
                    winsize = transf.numargs()
                    nodes = window(winsize, root=nodes, callback=transf.func, skipopcodes=False,
                                   alphabet=alphabet)
                elif callable(transf):
                    winsize = _func_numargs(transf)
                    nodes = window(winsize=winsize, root=nodes, callback=transf, skipopcodes=False,
                                   alphabet=alphabet)
                else:
                    raise ValueError(transf)
            return nodes

        for i in range(generation):
            nodes = lsys_step(self.rules, nodes, self.alphabet)
            if self.step_transforms:
                nodes = apply_transforms(nodes, self.step_transforms, self.alphabet)
        out = NodeList(nodes)
        out.set_plot_callback(self._plotcallback)
        return out

    def evolve(self, numsteps:int, dur:t.Rat=1) -> Branch:
        """
        Evolve the system, organize the generated nodes as a recursive
        branch

        numsteps: the number of steps to evolve the axiom
        axiom: if given, it overrides the axiom of this Lsystem
        dur: the duration assigned to the axiom
        """
        axiom = self.axiom
        nodelist = self._generate(generation=numsteps, dur=dur)
        if self._makerests:
            nodelist = nodelist.makerests()
        return nodelist.branched()

    def evolve_range(self, startstep, endstep, dur=1, axiom=None) -> List[Branch]:
        """
        similar to [self.evolve(n) for n in range(numsteps) but always
        feeding the generated nodes as axiom for the next step

        This makes sense only in the case of lsystem using some degree of randomness,
        where it is desirable to observe the evolution of the system throughout the
        different generations, making sure that the generation N is evolved from
        generation N-1
        """
        out = []
        numgens = endstep - startstep
        thisgen = self.evolve(startstep, dur=dur, axiom=axiom)
        out.append(thisgen)
        for N in range(startstep + 1, endstep):
            thisgen = self.evolve(axiom=thisgen, numsteps=1)
            out.append(thisgen)
        return out


def _distribute_duration(dur: Fraction, nodes: List[Node]) -> List[Node]:
    """distribute dur along nodes, based on the weight of each node"""
    assert all(isinstance(node, Node) for node in nodes)
    sumweight = sum(node.weight for node in nodes)
    newnodes = [node.clone(dur=dur * (node.weight / sumweight)) for node in nodes]
    return newnodes


def window(winsize: int,
           root: list,
           callback: Callable[..., U[Node, List[Node], None]],
           skipopcodes=False,
           alphabet=None,
           samebranch=True,
           ) -> List[Node]:
    """
    winsize:
        the window size
    skipopcodes:
        true if nodes with weight == 0 should be skipped
    callback:
        a function of the form (nodes) -> node | nodes
        An example with winsize=3:
        * no modification:
            [A B C] -> None or [A B C]. A is appended. Next: [B C D]
        * modified nodes (for example, A):
            [A B C] -> [A2 B C]. A2 is appended. Next: [B C D]
        * A B joined (the same would be valid for B C -> BC):
            [A B C] -> [AB C]. Nothing is appended. Next: [AB C D]
        * A B C should be joined:
            [A B C] -> [ABC] or simply ABC. Nothing is appended. Next: [ABC D E]

    Example: join nodes with same name and same step belonging to same branch

    def tied(n0, n1):
        if n0.name == n1.name and n0.step == n1.step and n0.branch == n1.branch:
           return n0.clone(dur=node0.dur+node1.dur)

    nodelist2 = window(2, nodelist, callback=tied)
    """
    from collections import deque
    collected = []  # type: List[Node]
    deq = deque([], maxlen=winsize)  # type: t.Deque[Node]
    EMPTY = Node("<EMPTY>", weight=0)
    flatroot = root.flatview() if isinstance(root, NodeList) else iterlib.flatlist(root)
    for node in iterlib.chain( flatroot, [EMPTY]*(winsize-1) ):
        if (skipopcodes and node.weight == 0 and node is not EMPTY) and not node.is_branchnode():
            continue
        deq.append(node)
        if len(deq) < winsize:
            continue
        if not samebranch or allequal(n.branch for n in deq):
            out = callback(*deq)
        else:
            out = None
        if isinstance(out, Node):
            deq.clear()
            deq.append(out)
        elif isinstance(out, str):
            assert alphabet is not None
            out = _str2node(out, alphabet)
            deq.append(out)
        elif isinstance(out, (list, tuple)):
            deq.clear()
            out = asNodes(out, alphabet=alphabet)
            deq.extend(out)
        elif out is None:
            pass  # deq already contains the result
        else:
            raise TypeError(f"The callback must return a seq. of Nodes, one Node or None"
                            f"Got {out} of type {type(out)} instead")
        if len(deq) == winsize:
            collected.append(deq.popleft())
    return collected


def transform(root:list, callback, initialstate:dict=None, skipopcodes=False, trimopcodes=True
              ) -> List:
    """
    Returns a new Nodes list with applied transformations.

    root:
        a possibly recursive list of Nodes (like Branch)
    callback:
        a function of the form (node, state) -> Node|"STOP"|None
        - if a node is returned, this node replaces the original node
        - if "STOP" is returned, iteration is stopped at this point in the branch
        - if "SKIP" is returned, the node is skipped
        - if None is returned, the node is appended as is
    trimopcodes:
        if True, opcodes (Nodes with weight==0) at the right of a branch will be trimmed
    bypass:
        nodes matching any of these will not be subjected to a transform and will
        be appended as is
    initialstate:
        if given, a dictionary which is set at the very beginning of the transform,
        to possibly set initial values for variables used later

    The internal structure (sublists) is preserved, so this routine
    can be called from a NodeList and a Branch. It will always return a list
    """
    state: Dict[str, Any] = initialstate.copy() if initialstate else {}

    def trimright(branch:list) -> list:
        it = reversed(branch)
        out = []
        for node in it:
            if isinstance(node, list) or node.weight > 0 or node.is_branchnode():
                out.append(node)
                out.extend(it)
        out.reverse()
        return out

    stack = []

    def inner(seq, state):
        newbranch = []  # type: List[U[Node, List]]
        for i, node in enumerate(seq):
            state['index'] = i
            if isinstance(node, list):
                sub = inner(node, state)
                if sub:
                    newbranch.append(sub)
            elif node.name in (PUSHSTATE, BRANCHBEGIN):
                stack.append(state.copy())
                newbranch.append(node)
            elif node.name in (POPSTATE, BRANCHEND):
                state.clear()
                state.update(stack.pop())
                newbranch.append(node)
            elif skipopcodes and node.weight == 0:
                newbranch.append(node)
            else:
                state['branch'] = node.branch  # TODO: track branch independently of what node.branch says
                response = callback(node, state)
                if response is None:
                    newbranch.append(node)
                elif isinstance(response, Node):
                    # add the node, skip rest of transforms
                    newbranch.append(response)
                elif response == "STOP":
                    # stop iteration of this branch (and any subbranch)
                    print("transform: STOP branch", node.branch)
                    break
                elif response == "SKIP":
                    logger.debug("skipping node: {node}")
                else:
                    raise ValueError(f"expected a Node, 'SKIP', 'STOP' or None, got {response}")
        if trimopcodes:
            newbranch = trimright(newbranch)
        return newbranch
    return inner(root, state)


def asNode(item, alphabet) -> Node:
    if isinstance(item, Node):
        return item
    elif isinstance(item, str):
        return _str2node(item, alphabet)
    raise TypeError(item)


def asNodes(seq:List[U[str, Node]], alphabet:dict) -> List[Node]:
    """
    Looks up items in seq against the defined alphabet.

    * If the item is a string, which stands for a letter, then the corresponding Node is generated
    * If the item is a Node, the Node itself is returned
    * If the item is a reserved opcode (a str), it is converted to a Node of weight 0

    At the end, a list of Nodes of the same length of the originial seq. is generated
    """
    return [asNode(item, alphabet) for item in seq]


def is_reserved_opcode(s: str):
    if s[0].isalpha():
        return False
    tok, _ = parse_token(s, evalvalue=False)
    return tok in _reserved_opcodes


def is_opcode(s: str):
    return not s[0].isalpha()


def _parse_letter(s: str) -> Tup[str, Opt[Dict]]:
    """
    For a simple letter definition, like "A", returns ("A", None)

    Given a string "Tok(foo=3, bar=1.5)", returns ("Tok", {'foo':3, 'bar':1.5})
    """
    s0 = s[0]
    if not s0.isalpha():
        raise ValueError(f"letter should begin with a-z or A-Z, got {s}")
    if "(" not in s:
        return s, None
    assert s[-1] == ")"
    letter, params = s[:-1].split("(")
    paramdict = eval(f"dict({params})")
    return letter, paramdict


def _str2node(s:str, alphabet:dict) -> Node:
    if is_opcode(s):
        return Node(s, weight=0)
    lettername, params = _parse_letter(s)
    letter = alphabet.get(lettername)
    if letter:
        return _letter2node(letter, params)
    logger.debug(f"no definition for {s} found")
    out = Node(lettername, weight=1)
    if params:
        out.set(**params)
    return out


def _as_nodes_rec(seq:list, alphabet:dict) -> List[U[Node, List]]:
    """
    Recursively convert any item in seq to a Node

    Node   -> Node
    List   -> recurse
    Letter -> Node (according to alphabet)
    str    -> interpreted as the name of a letter -> Letter -> Node
    """
    if all(isinstance(node, Node) for node in seq):
        return seq
    out = []  # type: List[U[Node, List]]
    for item in seq:
        if isinstance(item, list):
            out.append(_as_nodes_rec(item, alphabet))
        elif isinstance(item, Node):
            out.append(item)
        elif isinstance(item, str):
            node = _str2node(item, alphabet)
            out.append(node)
        else:
            raise TypeError(f"seq should be a seq. of strs or Nodes, got {item} of type {type(item)}")
    return out


def _letter2node(letter:Letter, params:Opt[dict]) -> Node:
    node = Node(letter.name, weight=letter.weight, data=letter.data)
    if params:
        node.set(**params)
    return node


def _generate_alphabet(rules:Dict[str, str]) -> Dict[str, Letter]:
    leftvars, constants = _parse_rules(rules)
    alphabet = {v:Letter(v) for v in leftvars}
    if constants:
        for cnst in constants:
            if cnst[0].isalpha():
                alphabet[cnst] = Letter(cnst)
            else:
                alphabet[cnst] = Constant(cnst)
    return alphabet


def _parse_rules(rules:Dict[str, U[str, List]]) -> Tup[t.Set[str], t.Set[str]]:
    """
    Given a dict of rules, it extracts the leftvariables and
    the constants and returnes each as a set

    * leftvariable is any letter which has a defined substitution
    * constant: a letter which has no substitution and therefore stays itself
    """
    leftvars = set([key for key in rules.keys() if isinstance(key, str)])
    # leftvars = set(list(rules.keys()))
    collected_tokens: List[str] = []
    for rule in rules.values():
        if isinstance(rule, str):
            collected_tokens.extend(_tokenize(rule))
        elif isinstance(rule, Rule):
            strtokens = [item for item in rule if isinstance(item, str)]
            if strtokens:
                collected_tokens.extend(strtokens)
    tokens = set(collected_tokens)
    is_var = lambda token: token[0] not in _reserved_opcodes
    rightvars = set(token for token in tokens if is_var(token))
    constants = tokens - leftvars - _reserved_opcodes
    if rightvars - leftvars:
        logger.debug("Variables appear in rules whithout substitution rules, making constant")
    return leftvars, constants


def _branched(seq:Iter[U[Node, list]]) -> List[U[Node, list]]:
    """
    retuns a branched version of seq, where a new list represents a branch
    The result is a list where each element is either a node or a list of nodes,
    recursively
    """
    branch = root = []   # type: List[U[Node, List]]
    stack = []
    for node in iterlib.flatlist(seq):
        branch.append(node)
        if node.name == BRANCHBEGIN:
            stack.append(branch)
            branch = []
        elif node.name == BRANCHEND:
            subbranch = branch
            branch = stack.pop()
            branch.append(subbranch)
    while stack:
        subbranch = branch
        branch = stack.pop()
        branch.append(subbranch)
    return root


def parse_token(tok:str, default=None, evalvalue=True) -> Tup[str, Opt[Any]]:
    """
    Parse a token like +10 or +(10) to ('+', 10)
    Tokens beginning with a letter are returned as is
    Tokens can have only one char, the rest is param

                 str   value
    @+10         @+     10
    @-20         @-    -20
    !0.5         !      0.5
    !(0.5, 3.4)  !      (0.5, 3.4)
    A2          A2    None
    """
    if tok[0].isalpha():
        return tok, default
    if len(tok) == 1:
        tokstr = tok
        value = default
    elif "(" in tok:   # form: tok(value)
        assert tok[-1] == ")"
        tokstr, rest = tok[:-1].split("(")
        value = eval(rest) if evalvalue else rest
    else:           # form: tokvalue, like @!!20 -> @!!, 20.0
        tokchrs = []
        restchrs = []
        it = iter(tok)
        for ch in it:
            if ch.isdigit():
                restchrs.append(ch)
                break
            tokchrs.append(ch)
        for ch in it:
            restchrs.append(ch)
        tokstr = "".join(tokchrs)
        value = float("".join(restchrs))
    return tokstr, value


def parse_token_argskws(tok:str) -> Tup[str, Tup[str, ...], Dict[str, Any]]:
    """
    Complex token parsing. Returns tokenstr, args, kws
    
    @(foo=40)           ("@", (), {'foo': 40}) 
    @(3, 4, bar="foo")  ("@", (3, 4), {"bar": "foo"})
    @3                  ("@", (3,), {})
    @(3, 4)             ("@", (3, 4), {})
    @4                  ("@", (4,), {})

    """
    def parse(*args, **kws):
        return args, kws

    if "(" not in tok:
        tokstr, value = parse_token(tok)
        return tokstr, (value,), {}
    idx = tok.index("(")
    tokstr = tok[:idx]
    rest = tok[idx+1:].strip()[:-1]
    args, kws = eval(f'parse({rest})')
    return tokstr, args, kws


def _getitem(seq, idx:int, default=None):
    """
    Return seq[idx] or default if idx is out of range
    """
    if 0 <= idx < len(seq):
        return seq[idx]
    return default


MatchFunc = Callable[[Node, List[Node], int], bool]

def matchnode(callback: Callable[[Node], bool]) -> MatchFunc:
    """
    Can be used as key in a Rule. Callback: (node:Node) -> bool
    * node is the current node being substituted.
    * callback should return True if it matches node
    """
    return lambda node, nodes, idx: callback(node)


def matchpost(callback) -> MatchFunc:
    """
    Can be used as key in a Rule.

    Callback: (current_node: Node, next_node: Node) -> bool

    * current_node: the node to be substituted
    * next_node: the next node
    * callback should return True if it matches node
    """
    def _(node, nodes, idx):
        nextnode = _getitem(nodes, idx+1)
        return callback(node, nextnode) if n1 is not None else False
    return _


def matchpre(callback) -> MatchFunc:
    """
    Callback: (current_node, previous_node) ->bool
    """
    def _(node, nodes, idx):
        prev = _getitem(nodes, idx-1)
        return callback(node, prev) if prev is not None else False
    return _


def matchprepost(callback: Callable[[Node, Node, Node], bool]) -> MatchFunc:
    """
    Can be used as key in a Rule.
    (Node, Node, Node) -> bool

    Callback: (current_node: Node, prev_node: Node, next_node: Node) -> bool

    * callback should return True if it matches node
    """
    def _(node, nodes, idx):
        pre = _getitem(nodes, idx-1)
        post = _getitem(nodes, idx+1)
        if pre is None or post is None:
            return False
        return callback(node, pre, post)
    return _


class DelayedOpcode:

    def eval_in_context(self, node, nodeseq, idx, alphabet=None) -> U[str, List[Node]]:
        return self.eval()

    def eval(self) -> U[str, List[Node]]:
        pass


class Choice(DelayedOpcode):
    def __init__(self, *pairs):
        """
        Weighted choice.

        Choice((0.7, 'A'), (0.3, 'B'))
        also accepts Choice(0.7, 'A', 0.3, 'B')

        Takes pairs of the form (weight, rule). Weights do NOT need to be normalized to 1,
        they are relative to their sum (so it is possible to use Choice((1, "A"), (3, "B")),
        which makes B 3 times more likely than A
        """
        super().__init__()
        weights = []
        tokens = []
        if isinstance(pairs[0], (int, float, Fraction)):
            weights = pairs[::2]
            tokens = pairs[1::2]
            pairs = list(zip(weights, tokens))
        else:
            for weight, token in pairs:
                weights.append(weight)
                tokens.append(token)
        self.weights = weights
        self.tokens = tokens
        self.pairs = pairs

    def eval(self) -> str:
        return random.choices(population=self.tokens, weights=self.weights)[0]

    def __repr__(self):
        return f"Choice({self.pairs})"


def _gauss_between(x0, x1):
    mean = (x0 + x1) * 0.5
    stdev = (x1 - mean) / 3
    x = random.gauss(mean, stdev)
    return x0 if x < x0 else x1 if x > x1 else x


class Rnd(DelayedOpcode):
    def __init__(self, opcode, minvalue, maxvalue, prec=-1, distr='uniform'):
        """
        prec: -1 for float64 precision, otherwise the number of digits after the comma
        """
        super().__init__()
        self.opcode = opcode
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        self.distr = distr
        funcs = {
            'uniform': random.uniform,
            'gauss': _gauss_between
        }
        func = funcs.get(distr)
        if not func:
            raise ValueError("distr can be any of f{list(funcs.keys())}")
        self.func = func  # type: Callable[[float, float], float]
        self.prec = prec

    def eval(self) -> str:
        x = self.func(self.minvalue, self.maxvalue)
        if self.prec >= 0:
            x = round(x, self.prec)
        return f"{self.opcode}{x}"

    def __repr__(self):
        return f"Rnd({self.opcode}, min={self.minvalue}, max={self.maxvalue}" \
               f", prec={self.prec}, distr={self.distr})"


class Matched(DelayedOpcode):
    def __init__(self, func: Callable[[Node], U[str, List[Node]]]):
        """
        func: (Node) -> U[str, List[Node]]

        * func will be called with the node currently being substituted
        """
        self.func = func

    def eval_in_context(self, node, nodeseq, idx, alphabet=None):
        return self.func(node)


class Rule:
    def __init__(self, items: List[U[str, DelayedOpcode]]) -> None:
        assert all(isinstance(item, (str, DelayedOpcode)) for item in items)
        self.items = items

    def evaluate(self) -> List[str]:
        """
        Evaluate this rule. Any DelayedOpcodes are evaluated, returning a list
        of opcodes, which, with the help of an alphabet, can be evaluated back
        to nodes
        """
        out = []
        for item in self.items:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, DelayedOpcode):
                itemstr = item.eval()
                items = _tokenize(itemstr)
                out.extend(items)
            else:
                raise TypeError(str(item))
        return out

    def evaluate_in_context(self, node, nodes, idx, alphabet=None):
        out = []
        for item in self.items:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, DelayedOpcode):
                subst = item.eval_in_context(node, nodes, idx, alphabet=alphabet)
                items = _flatten_subst(subst)
                out.extend(items)
            else:
                raise TypeError(str(item))
        return out

    def __repr__(self):
        return f"Rule({self.items})"

    @classmethod
    def parsestr(cls, s:str) -> 'Rule':
        """ Parse a rule defined as a string """
        assert isinstance(s, str)
        return Rule(_tokenize(s))

    @classmethod
    def parserule(cls, rule:U[str, List[U[str, DelayedOpcode]]]) -> 'Rule':
        """
        A rule can be
        * a string of the form "A B { A C }" or simply "B"
            Convert to a list of Nodes
        * a single DelayedOpcode
        * a list of strings and/or DelayedOpcodes
            each string can contain a series of opcodes
        :param rule:
        :return:
        """
        def parserec(rule):
            if isinstance(rule, DelayedOpcode):
                return [rule]
            elif isinstance(rule, str):
                return _tokenize(rule)
            elif isinstance(rule, list):
                collected = []
                for item in rule:
                    collected.extend(parserec(item))
                return collected
            else:
                raise TypeError(rule)
        return cls(parserec(rule))

    def __iter__(self) -> Iter[U[str, DelayedOpcode]]:
        return iter(self.items)


def _match_rule(ruleslist: List[Tup[U[str, MatchFunc], Rule]],
                nodes: List[Node],
                nodeidx: int) -> Opt[Rule]:
    node = nodes[nodeidx]
    assert isinstance(node, Node), node
    for key, rule in ruleslist:
        if isinstance(key, str):
            if node.name == key:
                return rule
        elif callable(key):
            matchok = key(node, nodes, nodeidx)
            if matchok:
                return rule
        else:
            raise TypeError("!!")
    return None  # no match


def lsys_step(rules:Dict[str, Rule], axiom:list, alphabet:dict) -> List[U[Node, List]]:
    """
    Replace each Node in axiom according to the replacement rules

    rules: the dict of rules as defined in LSystem
    axiom: the current state. For the first generation in an lsystem, this corresponds to the axiom
           In further generations, this corresponds to the (possibly nested) current list of nodes
    """
    nodes = _as_nodes_rec(axiom, alphabet)
    assert all(isinstance(node, Node) for node in nodes), [node for node in nodes if not isinstance(node, Node)]
    collected = []  # type: List[U[Node, List]]
    ruleslist = [(key, rule) for key, rule in rules.items()]
    ruleslist.sort(key=lambda key_rule: 10 if isinstance(key_rule[0], str) else 0)
    # nodes = list(iterlib.flatten(nodes))
    logger.debug("<<<< start matching >>>>")
    for nodeidx, node in enumerate(nodes):
        assert isinstance(node, Node)
        if node.is_branchnode():
            # branch nodes do not age, this is done to preserve their construction age
            # and account for relative age
            collected.append(node)
        else:
            rule = _match_rule(ruleslist, nodes, nodeidx)
            logger.debug(f"looking at {node} ({nodeidx}): rule {rule}")
            if not rule:
                logger.debug("                      no rule, appending")
                collected.append(node.clone(age=node.age+1))
            else:
                assert isinstance(rule, Rule)
                subst = rule.evaluate_in_context(node, nodes, nodeidx)
                logger.debug(f"matched {nodeidx} {node} -> {rule}")
                logger.debug(f"              subst: {subst}")
                substnodes = asNodes(subst, alphabet)
                dur = node.dur
                if dur > 0:
                    totalweight = sum(n.weight for n in substnodes)
                    if totalweight == 0:
                        raise ValueError("?? totalweight=0? " + str(substnodes))
                    substnodes = [n.clone(dur = dur*n.weight/totalweight,
                                          age = n.age if n.age > 0 else node.age + 1)
                                  for n in substnodes]
                collected.extend(substnodes)
    _set_branches(collected)
    return collected


def _set_branches(nodes: Iter[Node]) -> None:
    """
    Set the branch and relative age of each node, in place
    """
    stack = []
    branchage = 0
    branchnow = (1,)
    branch2numsubs = {}  # type: Dict[BranchId, int]

    def new_subbranch(branch):
        numsubs = branch2numsubs.get(branch, 1)
        branch2numsubs[branch] = numsubs + 1
        return branch + (numsubs,)

    for node in nodes:
        if node.name == BRANCHBEGIN:
            stack.append((branchnow, branchage))
            branchnow = new_subbranch(branchnow)
            branchage = node.age
        elif node.name == BRANCHEND:
            branchnow, branchage = stack.pop()
        else:
            node.agerel = node.age - branchage
            node.branch = branchnow


def _dict2str(d: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(d.items()))


def dumpnodes(nodes: Iter[Node]) -> None:
    def rec(seq, tab):
        collected = []
        for n in seq:
            if isinstance(n, list):
                collected.extend(rec(n, tab+1))
            else:
                sp = " " * (tab*3)
                t0 = str(round(float(n.offset), 3)).ljust(8)
                amp = amp2db(n.amp)
                amp = "-inf" if amp < -300 else str(amp)
                br = branchid2str(n.branch).ljust(6)
                data = _dict2str(n.data) if n.data else ""
                age = str(n.age).ljust(3)
                s = f"{sp}{n.name.ljust(5)} age={age} {br} t0={t0} dur={float(n.dur):.3f}\ty={n.step:.2f}\t{amp}\t{data}"
                collected.append(s)
        return collected
    lines = rec(nodes, 0)
    print("\n".join(lines))