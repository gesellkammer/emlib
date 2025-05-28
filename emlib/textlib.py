"""
Routines for working with text
"""
from __future__ import annotations
import textwrap
import re

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Callable


def stripLines(text: str, which='all', splitregex=''):
    """
    Strip leading and trailing empty lines

    Args:
        text: the text to work on
        which: one of 'all', 'top', 'bottom', where all removes all
            empty lines at the top and bottom, top removes only
            leading empty lines and bottom removes only
            trailing empty lines

    Returns:
        the transformed text
    """
    lines = text.splitlines() if not splitregex else re.split(text, splitregex)
    if which == 'all':
        lines = linesStrip(lines)
    elif which == 'top':
        lines = linesStripTop(lines)
    elif which == 'bottom':
        lines = linesStripBottom(lines)
    else:
        raise ValueError(f"Expected one of 'all', 'top', 'bottom', got '{which}'")
    return '\n'.join(lines)


def linesStrip(lines: list[str]) -> list[str]:
    """
    Remove empty lines from the top and bottom

    Args:
        lines: lines already split

    Returns:
        a list of lines without any empty lines at the beginning and at the end
    """
    startidx, endidx = 0, 0
    for startidx, line in enumerate(lines):
        if line and not line.isspace():
            break
    for endidx, line in enumerate(reversed(lines)):
        if line and not line.isspace():
            break
    return lines[startidx:len(lines)-endidx]


def reindent(text: str, prefix: str) -> str:
    """
    Reindent a given text. Replaces the indentation with a new prefix.

    Args:
        text: the text to reindent
        prefix: the new prefix to add to each line

    Returns:
        the reindented text
    """
    text = textwrap.dedent(text)
    if prefix:
        text = textwrap.indent(text, prefix=prefix)
    return text


def getIndentation(code: str) -> int:
    """ get the number of spaces used to indent code """
    for line in code.splitlines():
        stripped = line.lstrip()
        if not stripped:
            # skip empty lines
            continue
        return len(line) - len(stripped)
    return 0


def matchIndentation(code: str, modelcode: str) -> str:
    """
    Indent code matching modelcode

    Args:
        code: the code to indent
        modelcode: the code to match

    Returns:
        code indented to match modelcode

    Example
    ~~~~~~~

        >>> a = "    # This is some code"
        >>> b = "        # This is some other code"
        >>> matchIndentation(a, b)
        '        # This is some code'
    """
    indentation = getIndentation(modelcode)
    code = textwrap.dedent(code)
    return textwrap.indent(code, prefix=" " * indentation)


def linesStripTop(lines: list[str]) -> list[str]:
    """
    Remove empty lines from the top

    Args:
        lines: lines already split

    Returns:
        a list of lines without any empty lines at the beginning
    """
    for i, line in enumerate(lines):
        if line and not line.isspace():
            break
    else:
        return []
    return lines[i:]


def linesStripBottom(lines: list[str], maxlines: int = 0) -> list[str]:
    """
    Strip empty lines from the end of the list

    Args:
        lines: lines already split
        maxlines: the max. number of empty lines to leave at the end

    Returns:
        a list of lines with at most `maxlines` empty lines at the end


    """
    for i, line in enumerate(reversed(lines)):
        if line and not line.isspace():
            break
    if i - maxlines > 0:
        return lines[:maxlines - i]
    return lines


def joinPreservingIndentation(fragments: Sequence[str]) -> str:
    """
    Like join, but preserving indentation

    Args:
        fragments: a list of code strings
        maxEmptyLines: if given, the max. number of empty lines between fragments

    Returns:
        the joint code

    """
    if any(not isinstance(fragment, str) for fragment in fragments):
        fragment = next(_ for _ in fragments if not isinstance(_, str))
        raise TypeError(f"Expected a string, got {fragment}")
    jointtext = "\n".join(textwrap.dedent(frag) for frag in fragments if frag)
    numspaces = getIndentation(fragments[0])
    if numspaces:
        jointtext = textwrap.indent(jointtext, prefix=" "*numspaces)
    return jointtext


def fuzzymatch(pattern: str, strings: list[str]
               ) -> list[tuple[float, str]]:
    """
    Find possible matches to pattern in ``strings``.

    This implements a **very** simple algorithm. Returns a subseq.
    of strings sorted by best score. Only strings representing
    possible matches are returned

    Args:
        pattern: the string to search for within *strings*
        strings: a list os possible strings

    Returns:
        a list of (score, string match)
    """
    pattern = '.*?'.join(map(re.escape, list(pattern)))

    def calculate_score(pattern: str, s: str) -> float:
        match = re.search(pattern, s)
        if match is None:
            return 0.
        return 100.0 / ((1 + match.start()) * (match.end() - match.start() + 1))

    matches = [(score, s) for s in strings
               if (score := calculate_score(pattern, s)) > 0]
    matches.sort(reverse=True)
    return matches


def ljust(s: str, width: int, fillchar=" ") -> str:
    """
    Like str.ljust, but ensures that the output is always the given width

    Even if s is longer than ``width``
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

    Returns:
        a function to be called to produce the given transformation

    Example
    ~~~~~~~

    Create a function to remove some unwanted characters

        >>> import emlib.textlib
        >>> replacer = emlib.textlib.makeReplacer({"[": "", "]": "", '"': '', "'": "", "{": "", "}": ""})
        >>> replacer("[foo:'{bar}']")
        foo:bar
    """
    rep = {re.escape(k): v for k, v in conditions.items()}
    pattern = re.compile("|".join(rep.keys()))
    return lambda txt: pattern.sub(lambda m: rep[re.escape(m.group(0))], txt)


def firstSentence(txt: str) -> str:
    """
    Returns the first sentence from txt

    Args:
        txt: the text to analyze

    Returns:
        the first sentence


    Example
    ~~~~~~~

        >>> firstSentence('''
        ...
        ...     This is my text. It is amazing
        ...     It continues here
        ... ''')
        "This is my text"

        >>> firstSentence('''
        ...
        ...     This is also my text
        ...     It continues here
        ... ''')
        "This is also my text"
    """
    txt = txt.strip()
    lines = txt.splitlines()
    return lines[0].split('.', maxsplit=1)[0]


def escapeAnsi(line: str) -> str:
    """
    Escape ansi codes
    """
    return re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]').sub('', line)


def splitInChunks(s: str, maxlen: int) -> list[str]:
    """
    Split `s` into strings of max. size `maxlen`

    Args:
        s: the str/bytes to split
        maxlen: the max. length of each substring

    Returns:
        a list of substrings, where each substring has a max. length
        of *maxlen*
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


def quoteIfNeeded(s: str, quote='', defaultquote='"') -> str:
    """
    Add quotation marks around `s` if needed

    Args:
        s: the string which might need quoting
        quote: which quote sign to use. If not given, it will be detected
            and if not found a default quote is used
        defaultquote: quote used when autodetection is used and no quote was
            found

    Returns:
        a string where it is ensured that it is surrounded by `quote`

    Example
    ~~~~~~~

        >>> quoteIfNeeded('test')
        "test"
        >>> quoteIfNeeded("'foo'", "'")
        'foo'

    """
    if not quote:
        s0 = s[0]
        if s0 == s[-1] and (s0 == '"' or s0 == "'"):
            return s
        else:
            return f'{defaultquote}{s}{defaultquote}'
    else:
        if s[0] == s[-1] == quote:
            return s
        return f'{quote}{s}{quote}'


_fractions = {
    (1, 3): "⅓",
    (2, 3): "⅔",
    (1, 4): "¼",
    (2, 4): "½",
    (3, 4): "¾",
    (1, 5): "⅕",
    (2, 5): "⅖",
    (3, 5): "⅗",
    (4, 5): "⅘",
    (1, 6): "⅙",
    (2, 6): "⅔",
    (3, 6): "½",
    (4, 6): "⅔",
    (5, 6): "⅚",
    (1, 7): "⅐",
    (1, 8): "⅛",
    (3, 8): "⅜",
    (4, 8): "½",
    (5, 8): "⅝",
    (6, 8): "¾",
    (7, 8): "⅞",
    (1, 9): "⅑",
    (3, 9): "⅓",
    (6, 9): "⅔",
    (1, 10): "⅒",
    (2, 10): "⅕",
    (4, 10): "⅖",
}


def unicodeFraction(numerator: int, denominator: int, simplify=True) -> str:
    if simplify:
        from fractions import Fraction
        frac = Fraction(numerator, denominator)
        numerator, denominator = frac.numerator, frac.denominator
    ufraction = _fractions.get((numerator, denominator))
    if ufraction is not None:
        return ufraction
    return f"{numerator}/{denominator}"
