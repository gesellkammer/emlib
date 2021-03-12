from .typehints import Seq, List, Tup
import textwrap
import re


def stripLines(text: str) -> str:
    """
    Like ``str.strip`` but operates on lines as a whole.
    Removes empty lines at the beginning or end of text,
    without touching lines in between.
    """
    lines = text.splitlines()
    startidx, endidx = 0, 0

    for startidx, line in enumerate(lines):
        if line.strip():
            break

    for endidx, line in enumerate(reversed(lines)):
        if line.strip():
            break

    return "\n".join(lines[startidx:len(lines)-endidx])


def reindent(text:str, prefix:str, stripEmptyLines=True) -> str:
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


def joinPreservingIndentation(fragments: Seq[str]) -> str:
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


def fuzzymatch(pattern:str, strings:List[str]) -> List[Tup[float, str]]:
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

