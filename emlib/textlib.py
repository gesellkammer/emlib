"""
Routines for working with text
"""
from __future__ import annotations
import sys
import textwrap
import re

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


def stripLines(text: str) -> str:
    """
    Like ``str.strip`` but operates on lines as a whole.

    Removes empty lines at the beginning or end of text,
    without touching lines in between.
    """
    lines = splitAndStripLines(text)
    return "\n".join(lines)


def splitAndStripLines(text: str) -> List[str]:
    """
    Spits `text` into lines and removes empty lines at the beginning and end

    Returns the split lines in between

    Args:
        text: the text to split

    Returns:
        the list of lines
    """
    lines = text.splitlines()
    startidx, endidx = 0, 0
    for startidx, line in enumerate(lines):
        if line.strip():
            break
    for endidx, line in enumerate(reversed(lines)):
        if line.strip():
            break
    return lines[startidx:len(lines)-endidx]


def reindent(text:str, prefix:str="", stripEmptyLines=True) -> str:
    """
    Reindent a given text. Replaces the indentation with a new prefix.

    Args:
        text: the text to reindent
        prefix: the new prefix to add to each line
        stripEmptyLines: if True, remove any empty lines at the beginning
            or end of ``text``

    Returns:
        the reindented text
    """
    if stripEmptyLines:
        text = stripLines(text)
    text = textwrap.dedent(text)
    if prefix:
        text = textwrap.indent(text, prefix=prefix)
    return text


def getIndentation(code:str) -> int:
    """ get the number of spaces used to indent code """
    for line in code.splitlines():
        stripped = line.lstrip()
        if not stripped:
            # skip empty lines
            continue
        return len(line) - len(stripped)
    return 0


def joinPreservingIndentation(fragments: Sequence[str]) -> str:
    """
    Like join, but preserving indentation

    Args:
        fragments: a list of code strings

    Returns:
        the joint code

    """
    codes2 = [textwrap.dedent(code) for code in fragments if code]
    code = "\n".join(codes2)
    numspaces = getIndentation(fragments[0])
    if numspaces:
        code = textwrap.indent(code, prefix=" "*numspaces)
    return code


def fuzzymatch(pattern:str, strings:List[str]) -> List[Tuple[float, str]]:
    """
    Find possible matches to pattern in ``strings``. Returns a subseq. of
    strings sorted by best score. Only strings representing possible matches
    are returned

    Args:
        pattern: the string to search for within ``strings``
        strings: a list os possible strings

    Returns:
        a list of (score, string match)
    """
    pattern = '.*?'.join(map(re.escape, list(pattern)))

    def calculate_score(pattern, s):
        match = re.search(pattern, s)
        if match is None:
            return 0
        return 100.0 / ((1 + match.start()) * (match.end() - match.start() + 1))

    S2 = []
    for s in strings:
        score = calculate_score(pattern, s)
        if score > 0:
            S2.append((score, s))
    S2.sort(reverse=True)
    return S2


def ljust(s: str, width: int, fillchar=" ") -> str:
    """
    Like ``str.ljust``, but makes sure that the output is always the given width,
    even if s is longer than ``width``
    """
    s = s.ljust(width, fillchar)
    if len(s) > width:
        s = s[:width]
    return s


def makeReplacer(conditions: dict) -> Callable:
    """
    Create a function to replace many subtrings at once

    Args:
        conditions: a dictionary mapping a string to its replacement

    Example::

        >>> replacer = makeReplacer({"&":"&amp;", " ":"_", "(":"\\(", ")":"\\)"})
        >>> replacer("foo & (bar)")
        "foo_&amp;_\(bar\)"

    """
    rep = {re.escape(k): v for k, v in conditions.items()}
    pattern = re.compile("|".join(rep.keys()))
    return lambda txt: pattern.sub(lambda m: rep[re.escape(m.group(0))], txt)


def escapeAnsi(line: str) -> str:
    return re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]').sub('', line)


def splitInChunks(s: Union[str, bytes], maxlen: int) -> list:
    """
    Split `s` into strings of max. size `maxlen`

    Args:
        s: the str/bytes to split
        maxlen: the max. length of each substring
    """
    out = []
    idx = 0
    L = len(s)
    while idx < L:
        n = min(L-idx, maxlen)
        subs = s[idx:idx+n]
        out.append(subs)
        idx += n
    return out

