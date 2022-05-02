"""
Tools to parse and generate documentation

This framework was particularly conceived to document cython projects
which are badly supported with sphinx, but can be used to document
pure python projects just as well.

The main backend is mkdocs and at the moment the documentation
format supported is markdown, using docstring in google style

.. code::

    def func(a: type, b: type) -> rettype:
        '''
        Short description of func

        Longer description spanning
        multiple lines - Admonitions are allowed if
        enabled in mkdocs :

        !!! note

            My note admonition

        Args:
            * a: …
            * b: …

        Returns:
            My return value
        '''

One of the main advantages of this framework is that the documentation does not
need to be built by the host (readthedocs, for example). Used together with mkdocs
the documentation (markdown) is built locally and It can be built locally and
checked into the git repo

Projects using this framework for documentation:

- bpf4: https://github.com/gesellkammer/bpf4
  (build script: https://github.com/gesellkammer/bpf4/blob/master/docs/generatedocs.py)

- rtmidi2: https://github.com/gesellkammer/rtmidi2
  (build script: https://github.com/gesellkammer/rtmidi2/blob/master/docs/generatedocs.py)

- loristrck: https://github.com/gesellkammer/loristrck/blob/master/docs/generatedocs.py


A possible build script for a project's documentation:

.. code::

    # generatedocs.py
    import mymodule  # the module we want to document
    from emlib import doctools
    import os
    from pathlib import Path

    docsfolder = Path("docs")
    renderConfig = doctools.RenderConfig(splitName=True, fmt="markdown", docfmt="markdown")
    reference = doctools.generateDocsForModule(mymodule,
                                               renderConfig=renderConfig,
                                               exclude={'foo'},
                                               title="Reference")
    os.makedirs(docsfolder, exist_ok=True)
    open(docsfolder / "reference.md", "w").write(reference)
    index = doctools.generateMkdocsIndex(projectName="mymodule",
                                         shortDescription="mymodule does something useful")
    open(docsfolder / "index.md", "w").write(index)
    root = docsfolder.parent
    if (root/"mkdocs.yml").exists():
        os.chdir(root)
        os.system("mkdocs build")

"""
from __future__ import annotations
import os
import textwrap
import inspect
import dataclasses
import subprocess
import tempfile
from typing import Any, List, Optional as Opt, Callable, Set, Type, Dict, Tuple
import logging
from emlib import iterlib, textlib
import io
import enum
import re


logger = logging.getLogger("emlib.doctools")


@dataclasses.dataclass
class Param:
    """
    Represents a parameter in a function/method
    """
    name: str
    type: str = ''
    descr: str = ''
    default: Any = None
    hasDefault: bool = False

    def asSignatureParameter(self) -> str:
        if self.type and self.hasDefault:
            return f"{self.name}: {self.type} = {self.default}"
        elif self.hasDefault:
            return f"{self.name}={self.default}"
        elif self.type:
            return f"{self.name}: {self.type}"
        return self.name


class ObjectKind(enum.Enum):
    Class = "Class"
    Function = "Function"
    Method = "Method"
    Property = "Property"
    Attribute = "Attribute"
    Unknown = "Unknown"


@dataclasses.dataclass
class ParsedDef:
    """
    A ParsedDef is the result of parsing a function/method definition

    Attributes:
        name: the name of the object
        params: any input parameters (a list of Param)
        returns: the return param (a Param)
        shortDescr: a short description (the first line of the doc)
        longDescr: the rest of the description (without the first line)
        kind: the kind of the object (Function, Method, Property)
    """
    name: str = ''
    kind: ObjectKind = ObjectKind.Unknown
    params: List[Param] = dataclasses.field(default_factory=list)
    returns: Opt[Param] = None
    shortDescr: str = ''
    longDescr: str = ''
    embeddedSignature: str = ''

    def __post_init__(self):
        if self.longDescr:
            descr = textlib.stripLines(self.longDescr)
            descr = textwrap.dedent(descr)
            self.longDescr = descr

        if self.params is None:
            self.params = []

    def prefix(self) -> str:
        if self.kind == ObjectKind.Class:
            return "class"
        return "def"

    def generateSignature(self, includeName=True, includePrefix=False) -> str:
        """Generated the signature of this func/method based on its paremeters"""
        name = self.name
        if "." in name:
            name = self.name.split(".")[-1]
        if self.params:
            paramstrs = [p.asSignatureParameter() for p in self.params]
            paramstr = ", ".join(paramstrs)
        else:
            paramstr = ''
        if self.returns:
            returns = self.returns.type if self.returns.type else 'Any'
        else:
            returns = 'None'
        if self.kind == ObjectKind.Class:
            sig = f"({paramstr})"
        else:
            sig = f"({paramstr}) -> {returns}"
        if includeName:
            sig = name + sig
        if includePrefix:
            sig = self.prefix() + " " + sig
        return sig


@dataclasses.dataclass
class RenderConfig:
    """
    A RenderConfig determines how the documentation is rendered
    """
    maxwidth: int = 80
    """The max width of a line"""

    fmt: str = "markdown"
    """The render format"""

    docfmt: str = "markdown"
    """The format the documentation is written in"""

    splitName: bool = True
    """Show only the base name for functions/methods"""

    indentReturns: bool = True
    """Indent text in a Returns: clause"""

    attributesAreHeadings: bool = False
    """If True, attributes are rendered as headings instead of using bold"""

    includeInheritedMethods: bool = False
    """If True, add inherited methods to the documentation of each class"""


class ParseError(Exception):
    pass


def _parseProperty(p) -> ParsedDef:
    docStr = inspect.getdoc(p)
    parts = []
    if docStr is not None:
        parts.append(textwrap.dedent(docStr.strip()))
    setterDocstr = inspect.getdoc(p.fset)
    if setterDocstr is not None:
        parts.append(setterDocstr)
    if parts:
        docs = " / ".join(parts)
    else:
        docs = ""
    return ParsedDef(name=p.__name__, kind=ObjectKind.Property,
                     shortDescr=docs)


_mdreplacer = textlib.makeReplacer({'_': '\_',
                                    '*': '\*'})


def markdownEscape(s: str) -> str:
    """
    Escape s as markdown

    Example
    -------

        >>> markdownEscape("my_title_with_underscore")
        my\_title\_with\_underscore
    
    """
    return _mdreplacer(s)


def parseDef(obj) -> ParsedDef:
    """
    Parses a function/method, analyzing the signature / docstring.

    .. note::

        For a class, use :class:`~doctools.generateDocsForClass`

    Args:
        obj: the object to parse

    Returns:
        a ParsedDef

    Raises:
        ParseError if func cannot be parsed
    """
    docstr = obj.__doc__
    if inspect.isgetsetdescriptor(obj):
        # an attribute
        if docstr:
            docstr = docstr.strip()
        return ParsedDef(name=obj.__name__, shortDescr=docstr,
                         kind=ObjectKind.Attribute)
    elif inspect.isdatadescriptor(obj):
        # a property
        return _parseProperty(obj)

    docstr = obj.__doc__
    embeddedSignature = ''

    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        kind = ObjectKind.Function
    elif inspect.ismethod(obj):
        kind = ObjectKind.Method
    elif inspect.ismethoddescriptor(obj):
        # a c-defined / cython method
        kind = ObjectKind.Method
        # This removes the signature for cython function compiled
        # with embedsignature
        if docstr:
            try:
                firstline, rest = _splitFirstLine(docstr)
                basename = obj.__qualname__.split(".")[-1]
                if hasEmbeddedSignature(basename, docstr):
                    embeddedSignature, docstr = docstr.split("\n", maxsplit=1)
            except ValueError:
                pass
    else:
        raise ParseError(f"Could not parse {obj}")

    objName = obj.__qualname__
    sig: Opt[inspect.Signature]
    try:
        sig = inspect.signature(obj)
    except ValueError:
        sig = None

    try:
        docstring = parseDocstring(docstr)
    except Exception as e:
        logger.error(f"Could not parse signature for {obj}:")
        logger.error(obj.__doc__)
        raise e

    if sig and any(p for p in sig.parameters if p != 'self') and not docstring.params:
        logger.warning(f"{objName}: No parameters declared in the docstring")

    paramNameToParam: Dict[str, Param] = {p.name: p for p in docstring.params}
    shortDescr = docstring.shortDescr
    longDescr = docstring.longDescr
    params = []
    if sig:
        for paramName, param in sig.parameters.items():
            if paramName == 'self':
                params.append(Param("self"))
                continue
            hasDefault = param.default != param.empty
            paramDefault = param.default if hasDefault else None
            if (annot := param.annotation) != param.empty:
                assert isinstance(param.annotation, str), f"Use `from __future__ import annotation`, {param.annotation=} in {obj}"
                annot = annot.strip()
                if annot[0] == "'" and annot[-1] == "'":
                    annot = annot[1:-1]
                paramType = annot
            elif hasDefault and paramDefault is not None:
                paramType = type(paramDefault).__name__
            else:
                paramDef = paramNameToParam.get(paramName)
                if paramDef and paramDef.type:
                    paramType = paramDef.type
                else:
                    paramType = None
            paramDocstringDef = paramNameToParam.get(paramName, None)
            paramDescr = paramDocstringDef.descr if paramDocstringDef is not None else ''
            params.append(Param(name=paramName, type=paramType, descr=paramDescr,
                                default=paramDefault, hasDefault=hasDefault))
    elif docstring.params:
        for p in docstring.params:
            params.append(p)

    returns: Opt[Param]
    if not docstring.returns:
        if sig and sig.return_annotation != inspect._empty:
            returns = Param(name='returns', type=sig.return_annotation)
        else:
            returns = None
    else:
        blocks = docstring.returns.descr.split("\n\n")
        if len(blocks) > 1:
            returnDescr = blocks[0]
            rest = "\n".join(blocks[1:])
            longDescr += rest
        else:
            returnDescr = docstring.returns.descr
        if sig and sig.return_annotation != inspect._empty:
            returnType = sig.return_annotation
        elif docstring.returns.type:
            returnType = docstring.returns.type
        else:
            print("---- No return type: ", objName, docstring.returns)
            returnType = None
        returns = Param(name='returns', type=returnType, descr=returnDescr)

    return ParsedDef(name=objName, kind=kind, params=params, returns=returns,
                     shortDescr=shortDescr, longDescr=longDescr,
                     embeddedSignature=embeddedSignature)


def parseDocstring(docstr: str, fmt='google') -> ParsedDef:
    """
    Parses a docstring, returns a ParsedDef

    Args:
        docstr: the docstring to parse
        fmt: the format used for arguments/returns

    Returns:
        a ParsedDef
    """
    if not docstr:
        return ParsedDef()
    s0 = docstr
    docstr = textlib.stripLines(docstr)
    docstr = textwrap.dedent(docstr)
    lines = docstr.splitlines()
    line0 = lines[0].strip()
    if line0 == "Args:" or line0 == "Returns:":
        shortDescr = ""
    else:
        shortDescr = line0
        lines = lines[1:]
    descrLines = []
    context = None
    params: List[Param] = []
    returnLines: List[str] = []
    for line in lines:
        linestrip = line.strip()
        if context and not linestrip:
            context = None
            continue
        if not context:
            if linestrip == "Args:":
                context = "args"
            elif linestrip == "Returns:":
                context = "returns"
            else:
                descrLines.append(line)
        elif context == "args" or context == "param":
            if match := re.search(r"\s{2,8}(\w+)\s*:\s*\S+", line):
                #     param: descr of param
                paramName = match.group(1)
                paramDescr = line.split(":", maxsplit=1)[1].strip()
                params.append(Param(name=paramName, descr=paramDescr))
                context = "param"
            elif match := re.search(r"\s{2,8}(\w+)\s+\((.+)\)\s*:\s*\S+", line):
                #     param (type): descr of param
                paramName = match.group(1)
                paramType = match.group(2)
                paramDescr = line.split(":", maxsplit=1)[1].strip()
                params.append(Param(name=paramName, descr=paramDescr, type=paramType))
                context = "param"
            elif context == "param" and line.startswith("  "):
                assert len(params) > 0
                params[-1].descr += "\n" + line
            else:
                raise ParseError(f"Error parsing docstring {s0} at line {line}")
        elif context == "returns":
            returnLines.append(linestrip)

    longDescr = "\n".join(descrLines)
    returns: Opt[Param]
    returnLines = [l for l in returnLines
                   if l.strip()]
    if returnLines:
        if match := re.search(r"\(\s*(.*)\s*\)\s+(\w.*)", returnLines[0]):
            returnType = match.group(1)
            returnLines[0] = match.group(2)
        else:
            returnType = ""
        returns = Param(name="returns", descr=" ".join(returnLines), type=returnType)
    else:
        returns = None
    return ParsedDef(shortDescr=shortDescr,
                     longDescr=longDescr,
                     params=params,
                     returns=returns)


def markdownHeader(text: str, headernum:int, inline=True) -> str:
    """
    Create a markdown header

    Args:
        text (str): The text of the header
        headernum (int): the header number (int >= 1)
        inline (bool): if inline, the header is of the form "# header", otherwise
            the header is applied as underline

    Returns:
        the markdown text, either one line if inline, or two lines otherwise

    """
    text = markdownEscape(text)
    assert headernum >= 1
    if inline:
        return "#"*headernum + " " + text
    else:
        if headernum > 2:
            raise ValueError(f"Max. header number is 2, got {headernum}")
        headerline = ("=" if headernum == 1 else "-") * len(text)
        return "\n".join((text, headerline))



def formatSignature(name: str, signature: str, maxwidth: int=70,
                    returnNewLine=False, prefix="def") -> str:
    """
    Format a signature to align args

    This function formats a signature, possibly in multiple lines,
    so that arguments are aligned correctly and the text does not
    excede the given maxwidth

    Transforms a signature like::

        '(a: int, b: List[foo], c=200, ..., z=None) -> List[foo]'

    into::

        def func(a: int, b:List[foo], c=200, signa, b=200, ...
                 z=None
                 ) -> List[foo]:

    Args:
        name: the name of the function
        signature: the signature, as returned via ``str(inspect.signature(func))``

    Returns:
        the realigned signature (str)
    """
    signature = signature.replace("List", "list").replace("Tuple", "tuple")
    if "->" in signature:
        args, ret = signature.split("->")
    else:
        args = signature
        ret = ""
        # ret = "Any"
    args = args.strip()
    if args[0] == "(":
        args = args[1:]
    if args[-1] == ")":
        args = args[:-1]
    argparts = args.split(",")
    if "." in name:
        header = name
    else:
        header = f"{prefix} {name}("
    indent = len(header)
    lines = [header]
    lasti = len(argparts) - 1
    for i, arg in enumerate(argparts):
        arg = arg.strip()
        if ":" in arg:
            argname, rest = arg.split(":")
            if "=" in rest:
                annot, defaultval = rest.split("=")
                annot = annot.replace("'", "")
                arg = f"{argname}:{annot}={defaultval}"
            else:
                annot = rest.strip().replace("'", "")
                arg = f"{argname}: {annot}"
        if i == 0 or len(arg) + len(lines[-1]) < maxwidth:
            lines[-1] += arg
        else:
            lines.append(" "*indent + arg)
        if i < lasti:
            lines[-1] += ", "

    if ret:
        retstr = " -> " + ret.strip().replace("'", "")
    else:
        retstr = ""
    if returnNewLine or len(lines[-1]) + len(retstr) > maxwidth:
        lines.append(" "*indent + ")" + retstr)
    else:
        lines[-1] += ")" + retstr

    return "\n".join(lines)


def _mdParam(param:Param, maxwidth=70, indent=4) -> List[str]:
    s = f"* **{param.name}**"
    if param.type:
        s += f" (`{param.type}`)"
    s += ": " + param.descr
    if param.hasDefault:
        s += f" (*default*: `{param.default}`)"
    return textwrap.wrap(s, maxwidth, subsequent_indent=" "*(indent*1))


def _rstConvertNotesToMarkdown(rst: str) -> str:
    rst = rst.replace(".. note::", "!!! note")
    return rst


def _rstToMarkdown(rst: str, startLevel=1) -> str:
    rst = _rstConvertNotesToMarkdown(rst)
    rstfile = tempfile.mktemp(suffix=".rst")
    mdfile = os.path.splitext(rstfile)[0] + ".md"
    open(rstfile, "w").write(rst)
    proc = subprocess.Popen(["pandoc", "-f", "rst", "-t", "markdown", rstfile], stdout=subprocess.PIPE)
    proc.wait()
    mdstr = proc.stdout.read().decode("utf-8")
    mdstr = markdownReplaceHeadings(mdstr, startLevel=startLevel)
    return mdstr


def markdownReplaceHeadings(s: str, startLevel=1, normalize=True) -> str:
    """
    Replaces any heading of the form::

        Heading    to  # Heading
        =======

    Args:
        s: the markdown text
        startLevel: the heading start level
        normalize: if True, the highest heading in s will be forced to become
            a `startLevel` heading.

    Returns:
        the modified markdown text
    """
    lines = s.splitlines()
    out: List[str] = []
    roothnum = 100
    skip = False
    lines.append("")
    insideCode = False
    for line, nextline in iterlib.pairwise(lines):
        if line.startswith("```"):
            insideCode = not insideCode
            out.append(line)
        elif insideCode:
            out.append(line)
        elif skip:
            skip = False
        elif line.startswith("#"):
            hstr, *rest = line.split()
            hnum = len(hstr)
            if hnum < roothnum:
                roothnum = hnum
            out.append(line)
        elif line and nextline.startswith("---") or nextline.startswith("==="):
            hnum = 1 if nextline[0] == "=" else 2
            if hnum < roothnum:
                roothnum = hnum
            out.append(markdownHeader(line, hnum))
            skip = True
        else:
            out.append(line)
    if startLevel == 1 and not normalize:
        return "\n".join(out)
    # hnum     startLevel   roothnum   hnumnow
    #    1              1          1         1
    #    1              2          1         2
    #    2              1          1         2
    #    2              2          1         2
    #    2              1          2         1
    #    2              2          2         2
    #    2              3          2         3
    out2 = []
    insideCode = False
    for line in out:
        if line.startswith("```"):
            insideCode = not insideCode
        if insideCode:
            out2.append(line)
        elif line.startswith("#"):
            hstr, text = line.split(maxsplit=1)
            hnum = len(hstr)
            hnumnow = hnum - roothnum + startLevel
            if hnumnow != hnum:
                out2.append(markdownHeader(text, hnumnow))
            else:
                out2.append(line)
        else:
            out2.append(line)
    return "\n".join(out2)


def _guessDocFormat(docstring: str) -> str:
    formats = set()
    for line in docstring.splitlines():
        linestrip = line.strip()
        if "![" in linestrip:
            formats.add("markdown")
        if linestrip.startswith(".. ") or linestrip.endswith("::"):
            formats.add("rst")
        if re.search("``\w+``", linestrip) or ":meth:" in linestrip or ":class:" in linestrip:
            formats.add("rst")
        if "~~" in linestrip:
            formats.add("rst")
        if len(formats) > 1:
            raise ValueError("Ambiguous format detected")
    if not formats:
        return "markdown"
    return formats.pop()


def _renderAttributeMarkdown(parsed: ParsedDef, startLevel=0):
    if startLevel == 0:
        s = f"* **{parsed.name}**"
    else:
        s = markdownHeader(parsed.name, startLevel)

    descr = parsed.shortDescr
    if descr:
        s += ": " + descr
        lines = textwrap.wrap(s, width=80, subsequent_indent="  ")
        s = "\n".join(lines)
    return s


def _renderDocMarkdown(parsed: ParsedDef, startLevel:int, renderConfig: RenderConfig
                       ) -> str:
    if parsed.kind == ObjectKind.Attribute:
        level = startLevel if renderConfig.attributesAreHeadings else 0
        return _renderAttributeMarkdown(parsed, startLevel=level)
    arglines = []

    params = [p for p in parsed.params if p.name != 'self'] # and (p.descr or p.default or p.type)]
    if params:
        arglines.extend(["", "**Args**", ""])
        for param in params:
            arglines.extend(_mdParam(param, maxwidth=renderConfig.maxwidth))

    if parsed.returns and parsed.returns.descr and parsed.returns.descr != "None":
        arglines.append("\n**Returns**\n")
        if parsed.returns.type is not None:
            returnstr = f"(`{parsed.returns.type}`) {parsed.returns.descr}"
        else:
            returnstr = parsed.returns.descr

        if renderConfig.indentReturns:
            returnstr = textwrap.indent(returnstr, "&nbsp;&nbsp;&nbsp;&nbsp;")
        arglines.append(returnstr)

    argsstr = "\n".join(arglines)
    argsstr = textwrap.dedent(argsstr)
    if not parsed.longDescr:
        longdescr = ""
    else:
        docfmt = renderConfig.docfmt
        if docfmt == 'auto':
            docfmt = _guessDocFormat(parsed.longDescr)
        if docfmt == 'rst':
            longdescr = _rstToMarkdown(parsed.longDescr, startLevel=startLevel+1)
        elif docfmt == 'markdown':
            longdescr = parsed.longDescr
        else:
            raise ValueError(f"doc format {docfmt} not supported")
    longdescr = markdownReplaceHeadings(longdescr, startLevel=startLevel+1, normalize=True)
    componentName = parsed.name if not renderConfig.splitName else parsed.name.split(".")[-1]
    blocks = [markdownHeader(componentName, headernum=startLevel)]
    if parsed.embeddedSignature:
        fmtsig = parsed.embeddedSignature
    else:
        signature = parsed.generateSignature(includeName=False, includePrefix=False)
        fmtsig = formatSignature(componentName,
                                 signature=signature,
                                 maxwidth=renderConfig.maxwidth,
                                 prefix=parsed.prefix())
    siglines = ["```python\n"]
    siglines.append(fmtsig)
    siglines.append("\n```")
    blocks.append("\n".join(siglines))

    if parsed.shortDescr:
        blocks.append(parsed.shortDescr)

    if parsed.longDescr:
        blocks.append(longdescr)
    if argsstr:
        blocks.append(argsstr)
    s = "\n\n\n".join(blocks)
    return textwrap.dedent(s)


def renderDocumentation(parsed: ParsedDef, renderConfig: RenderConfig, startLevel:int
                        ) -> str:
    """
    Renders the parsed function / method /  property as documentation in the given format

    Args:
        parsed: the result of calling parseDef on a function or method
        startLevel: the heading level to use as root level for the documentation
        renderConfig: a RenderConfig

    Returns:
        the generated documentation as string
    """
    if renderConfig.fmt == 'markdown':
        return _renderDocMarkdown(parsed, startLevel=startLevel, renderConfig=renderConfig)
    else:
        raise ValueError(f"format {renderConfig.fmt} not supported)")


def fullname(obj, includeModule=True) -> str:
    """
    Given an object, returns its qualified name

    Args:
        obj: the object to query

    Returns:
        the full (qualified) name of obj as string
    """
    if inspect.isclass(obj):
        cls = obj
    else:
        cls = obj.__class__
    module = cls.__module__
    if module == 'builtins':
        return cls.__qualname__
    return module+'.'+cls.__qualname__ if includeModule else cls.__qualname__


def generateDocsForFunctions(funcs: List[Callable], renderConfig: RenderConfig=None,
                             title:str=None, pretext:str = None, startLevel=1
                             ) -> str:
    """
    Collects documentation for multiple functions in one string

    Args:
        funcs: the funcs to parse
        title: a title to use before the generated code
        pretext: a text between the title and the generated code
        startLevel: the heading start level
        renderConfig: a RenderConfig

    Returns:
        the generated documentation

    Raises:
        ParseError if any of the functions fails to be parsed
    """
    if renderConfig is None:
        renderConfig = RenderConfig()
    lines: List[str] = []
    sep = "\n----------\n"
    _ = lines.append
    if title:
        _(markdownHeader(title, startLevel))
        _("")

    if pretext:
        _(pretext)
        _(sep)

    lasti = len(funcs)-1
    startLevelForFuncs = startLevel
    for i, func in enumerate(funcs):
        try:
            parsed = parseDef(func)
        except ParseError:
            logger.error(f"Could not parse object {func}")
            continue
        docstr = renderDocumentation(parsed, startLevel=startLevelForFuncs,
                                     renderConfig=renderConfig)
        _(docstr)
        if i < lasti:
            _(sep)
    return "\n".join(lines)


@dataclasses.dataclass
class ClassMembers:
    """
    Gathers members defined in a class

    A member can be an attribute (a `@property`) or a method
    """
    properties: Dict[str, Any]
    """ The @properties in this class """

    methods: Dict[str, Any]
    """ The methods in this class """


def _isMethodInherited(cls, method: str, mro) -> bool:
    m = getattr(cls, method)
    if inspect.ismethod(m):
        # classmethod
        return any(getattr(base, method, None)
                   for base in mro)
    else:
        # normal method
        return any(m is getattr(base, method, None)
                   for base in mro)


def isMethodInherited(cls, method: str) -> bool:
    """
    Is this method inherited?

    Args:
        cls: the class to which this method belonds
        method: the method name

    Returns:
        True is this method is not defined in *cls* but inherited from
        a base class. If False, this method is defined in *cls* and it
        either does not exist in any base class or it overrides a parent's
        definition
    """
    mro = cls.mro()[1:]
    return _isMethodInherited(cls, method, mro)


def getClassMembers(cls, exclude: List[str] = None, inherited=True) -> ClassMembers:
    """
    Inspects cls and determines its methods and properties

    Args:
        cls: the class to inspect
        exclude: a list of regexes to exclude
        inherited: if True, include inherited attributes and methods

    Returns:
        a ClassMembers object, with attributes: properties, methods

    """
    names = dir(cls)
    memberNames = [n for n in names
                   if n[0].islower() and not n.startswith("__") and not _matchAnyRegex(exclude, n)]
    if '__init__' in names:
        initmeth = getattr(cls, '__init__')
        if initmeth is not object.__init__ and initmeth.__doc__ != 'Initialize self.  See help(type(self)) for accurate signature.':
            memberNames = ['__init__'] + memberNames
    members = [(method, getattr(cls, method)) for method in memberNames]
    properties = {m[0]: m[1] for m in members
                  if inspect.isgetsetdescriptor(m[1])}
    methods = {m[0]:m[1] for m in members
               if not inspect.isgetsetdescriptor(m[1])}
    if not inherited:
        mro = cls.mro()[1:-1]
        methods = {name: method for name, method in methods.items()
                   if not _isMethodInherited(cls, name, mro)}
        properties = {name: prop for name, prop in properties.items()
                      if not _isMethodInherited(cls, name, mro)}

    return ClassMembers(properties=properties, methods=methods)


def isExtensionClass(cls) -> bool:
    """ Returns True if cls is an extension class

    An extension class (also built-in class) is defined in c or
    cython. Its source cannot be inspected
    """
    try:
        inspect.getsourcelines(cls)
    except OSError:
        return True
    except TypeError:
        return False
    return False


def getEmbeddedSignature(objname: str, doc: str) -> Opt[str]:
    """
    Returns the docstring embedded signature

    This is present in cython generated classes with the compiler
    directive "embedsignature" enabled

    Args:
        objname: the name of the object
        doc: the docstring of the object

    Returns:
        the embedded signature or None if no embedded signature found
    """
    try:
        line0, rest = doc.split("\n", maxsplit=1)
    except ValueError:
        line0 = doc
    if re.search(fr"{objname}\(", line0):
        return line0.strip()
    return None


def hasEmbeddedSignature(objname: str, doc: str) -> bool:
    """ 
    Returns True if the docstring has an embedded signature

    This is be the case for cython generated classes with the compiler
    directive "embedsignature" enabled

    Args:
        objname: the name of the object
        doc: the docstring

    Returns:
        True if docstring has an embedded signature
    """
    return getEmbeddedSignature(objname, doc) is not None


def _splitFirstLine(s: str) -> Tuple[str, str]:
    if "\n" in s:
        l0, l1 = s.split("\n", maxsplit=1)
        return l0, l1
    return s, ''


def getModuleMembers(module, exclude: List[str] = None, classesFirst=True) -> Dict[str, Any]:
    """
    Returns a dictionary of {membername:member} in order of appearence

    The difference with inspect.getmembers(module) is that only members
    (functions, classes) actually defined in the module are included.

    Args:
        module: the module to query
        exclude: a list of regexes to exclude
        classesFirst: if True, classes are included first

    Returns:
        a dictionary of all the members defined in this module

    """
    members = inspect.getmembers(module)

    if exclude:
        members = [(name, item) for name, item in members
                   if _matchAnyRegex(exclude, name)]

    if classesFirst:
        members.sort(key=lambda item: int(inspect.isclass(item)))

    ownmembers = {name:item for name, item in members
                  if inspect.getmodule(item) is module and not name.startswith("_")}

    return ownmembers


def externalModuleMembers(module, include_private=False) -> List[str]:
    """
    Returns a list of member names which appear in dir(module) but are not
    defined there

    Args:
        module: the module to query

    Returns:
        a list of member names

    """
    members = inspect.getmembers(module)
    external = [name for name, item in members
                if inspect.getmodule(item) is not module or name.startswith("_")]
    if not include_private:
        external = [n for n in external if not n.startswith("_")]
    return external


def groupMembers(members: Dict[str, Any]) -> Tuple[dict, dict, dict]:
    """
    Sorts the members into three groups: functions, classes and modules

    Returns a tuple of three dictionaries: (functions, classes, modules)

    Args:
        module: the module to query

    Returns:
        a tuple `(funcs, classes, modules)`, where each item is a dict
        `{name:object}`

    """
    funcs = {name:item for name, item in members.items()
             if inspect.isfunction(item)}
    clss = {name:item for name, item in members.items()
            if inspect.isclass(item)}
    modules = {name:item for name, item in members.items()
               if inspect.ismodule(item)}
    return funcs, clss, modules


def sortClassesByTree(classes: List[type]) -> List[type]:
    """
    Sort classes according to their inheritance tree

    Args:
        classes: a list of classes

    Returns:
        the list of classes, sorted by inheritance
    """
    assert all(isinstance(cls, type) for cls in classes)

    seen = set()
    coll = []
    later = []

    def walk(tree):
        for entry in tree:
            if isinstance(entry, tuple):
                cls, bases = entry
                if cls == object :
                    continue
                if bases[0] != object and bases[0] not in seen:
                    later.append(cls)
                    continue
                coll.append(cls)
                seen.add(cls)
            elif isinstance(entry, list):
                walk(entry)
            else:
                raise ValueError(f"Got {entry}")

    walk(inspect.getclasstree(classes))
    coll.extend(later)
    assert all(isinstance(cls, type) for cls in coll)

    return coll


def generateDocsForModule(module, renderConfig:RenderConfig=None, exclude: List[str] = None,
                          startLevel=1, grouped=False, title:str=None,
                          includeCustomExceptions=False) -> str:
    """
    Generate documentation for the given module

    Args:
        module: the module to generate documentation for
        renderConfig: a RenderConfig
        exclude: functions/classes to exclude. A list of regexes
        startLevel: heading start level
        grouped: if True, classes / functions are grouped together. Otherwise the order
            of appearance within the source code is used
        title: if given, it will be used instead of the module name
        includeCustomExceptions: if True, include user defined exceptions into the documentation

    Returns:
        The rendered documentation as a markdown string
    """
    if renderConfig is None:
        renderConfig = RenderConfig()
    sep = "\n---------\n"
    blocks = []
    if title:
        blocks.append(markdownHeader(title, startLevel))
    else:
        blocks.append(markdownHeader(module.__name__, startLevel))
    if module.__doc__:
        doc = markdownReplaceHeadings(module.__doc__, startLevel=startLevel+1)
        blocks.append(doc)
    blocks.append(sep)
    if grouped:
        raise ValueError("grouped output not supported yet")

    membersd = getModuleMembers(module, exclude=exclude)
    classmembers = [member for name, member in membersd.items()
                    if inspect.isclass(member)]
    classmembers = sortClassesByTree(classmembers)

    funcmembers = [member for name, member in membersd.items()
                   if not inspect.isclass(member)]

    if not includeCustomExceptions and classmembers:
        classmembers = [cls for cls in classmembers
                        if Exception not in cls.mro()]

    for cls in classmembers:
        blocks.append(generateDocsForClass(cls, renderConfig=renderConfig, startLevel=startLevel+1))
        blocks.append(sep)

    for func in funcmembers:
        parsed = parseDef(func)
        doc = renderDocumentation(parsed, renderConfig=renderConfig, startLevel=startLevel+1)
        blocks.append(doc)
        blocks.append(sep)

    if blocks[-1] == sep:
        blocks.pop()
    return "\n\n".join(blocks)


def _matchAnyRegex(patterns: List[str], s: str) -> bool:
    """
    Returns True if s matches any of the regexes in patterns

    Args:
        patterns: a list of regexes (can be None)
        s: the string to match

    Returns:
        True if s matches any of the regexes in patterns
    """
    if not patterns:
        return False
    return any(re.search(patt, s) for patt in patterns)


def generateDocsForClass(cls, renderConfig:RenderConfig, exclude: List[str] = None,
                         startLevel=1) -> str:
    """
    Generate documentation for the given class

    Args:
        cls: the cls to generate documentation for
        renderConfig: a RenderConfig
        exclude: functions/classes to exclude
        startLevel: heading start level
        
    Returns:
        The rendered documentation as a markdown string
    """
    sep = "\n---------\n"
    classMembers = getClassMembers(cls, inherited=renderConfig.includeInheritedMethods)
    blocks = [markdownHeader(cls.__qualname__, startLevel)]
    mro = cls.mro()
    if len(mro) > 2:
        base = mro[1].__qualname__
        blocks.append(f' - Base Class: [{base}](#{base.lower()})')
    if cls.__doc__:
        clsdocs = textwrap.dedent(cls.__doc__)
        if isExtensionClass(cls) and hasEmbeddedSignature(cls.__qualname__, clsdocs):
            signature = getEmbeddedSignature(cls.__qualname__, clsdocs)
            _, clsdocs = _splitFirstLine(clsdocs)
            clsdocs = textwrap.dedent(clsdocs)
        else:
            signature = ''
        parsedDocs = parseDocstring(clsdocs)
        parsedDocs.embeddedSignature = signature
        rendered = renderDocumentation(parsedDocs, renderConfig=renderConfig,
                                       startLevel = startLevel+1)
        blocks.append(rendered)

    if classMembers.properties:
        blocks.append("**Attributes**")
        # blocks.append(markdownHeader("Attributes", startLevel+1))
        for prop in classMembers.properties.values():
            p = parseDef(prop)
            doc = renderDocumentation(p, renderConfig=renderConfig, startLevel=startLevel+2)
            blocks.append(doc)

    if classMembers.methods:
        if exclude:
            methods = [method for methodname, method in classMembers.methods.items()
                       if not _matchAnyRegex(exclude, methodname)]
        else:
            methods = list(classMembers.methods.values())

        blocks.append(sep)
        blocks.append("**Methods**")
        methodDocs = generateDocsForFunctions(methods,
                                              renderConfig=renderConfig,
                                              startLevel=startLevel+1)
        blocks.append(methodDocs)

    docs = "\n\n".join(blocks)
    return docs


def generateMkdocsIndex(projectName: str,
                        shortDescription: str,
                        author: str="<author name>",
                        email:str="<author email>",
                        url:str="<Project URL>",
                        longDescription:str="<Long Description>",
                        quickStart:str="<Quick Start>",
                        includeWelcome=True
                        ) -> str:
    """
    Generate the template for an index.md file suitable for mkdocs

    Args:
        projectName: the name of the project
        shortDescription: a short description (a line)
        author: the author of the project
        email: author's email
        url: url of the project (github, etc)
        longDescription: long description of the project (a paragraph)
        quickStart: an introduction to the project (multiple paragraphs)

    Returns:
        a string which could be saved to an index.md file
    """
    if includeWelcome:
        welcomeStr = f"Welcome to the **{projectName}** documentation!"
    else:
        welcomeStr = ""

    return f"""
    # {projectName}
    
    {welcomeStr}
    {shortDescription}
    
    * Author: {author}
    * email: {email}
    * home: {url}
    
    ## Description
    
    {longDescription}
    
    ----
    
    ## Installation
    
    ```bash
    pip install {projectName}
    ```
    
    ----
    
    ## Quick Start
    
    {quickStart}
    
    """
