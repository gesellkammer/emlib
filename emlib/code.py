from __future__ import annotations
import textwrap
from emlib import misc
import emlib.typehints as t


def reindent(text:str, prefix="", stripEmptyLines=True) -> str:
    if stripEmptyLines:
        text = misc.strip_lines(text)
    text = textwrap.dedent(text)
    if prefix:
        text = textwrap.indent(text, prefix=prefix)
    return text


def getIndentation(code:str) -> int:
    """ get the number of spaces used to indent code """
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped:
            spaces = len(line) - len(stripped)
            return spaces
    return 0


def joinCode(codes: t.Iter[str]) -> str:
    """
    Like join, but preserving indentation

    Args:
        codes: a list of code strings

    Returns:

    """
    codes2 = [textwrap.dedent(code) for code in codes]
    code = "\n".join(codes2)
    numspaces = getIndentation(codes[0])
    if numspaces:
        code = textwrap.indent(code, prefix=" "*numspaces)
    return code