"""
Set of misc. utilities to work with json

"""
from __future__ import annotations
import re


def _comments_replacer(match):
    s = match.group(0)
    return "" if s[0] == '/' else s
    

def remove_comments(json_like: str):
    """
    Removes C-style comments from *json_like* and returns the result.  Example::

        >>> test_json = r'''
        ... {
        ...    "foo": "bar", // This is a single-line comment
        ...    "baz": "blah" /* Multi-line
        ...    Comment */
        ... }'''
        >>> remove_comments('{"foo":"bar","baz":"blah",}')
        '{\n    "foo":"bar",\n    "baz":"blah"\n}'
    """
    comments_re = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return comments_re.sub(_comments_replacer, json_like)


def remove_trailing_commas(json_like: str):
    """
    Removes trailing commas from *json_like* and returns the result.  Example::

        >>> remove_trailing_commas('{"foo":"bar","baz":["blah",],}')
        '{"foo":"bar","baz":["blah"]}'
    """
    trailing_object_commas_re = re.compile(
        r'(,)\s*}(?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    trailing_array_commas_re = re.compile(
        r'(,)\s*\](?=([^"\\]*(\\.|"([^"\\]*\\.)*[^"\\]*"))*[^"]*$)')
    # Fix objects {} first
    objects_fixed = trailing_object_commas_re.sub("}", json_like)
    # Now fix arrays/lists [] and return the result
    return trailing_array_commas_re.sub("]", objects_fixed)


def remove_all(json_like: str):
    """
    Remove comments and trailing commas
    """
    pipe = [remove_comments, remove_trailing_commas]
    s = json_like
    for func in pipe:
        s = func(s)
    return s


def json_minify(json:str, strip_space=True) -> str:
    """
    strip comments and remove space from string

    Args:
        json: a string representing a json object
        strip_space: remove spaces

    Returns:
        the minified json
    """
    tokenizer = re.compile('"|(/\*)|(\*/)|(//)|\n|\r')
    in_string = False
    inmulticmt = False
    insinglecmt = False
    new_str = []
    from_index = 0     # from is a keyword in Python

    for match in re.finditer(tokenizer, json):
        if not inmulticmt and not insinglecmt:
            tmp2 = json[from_index:match.start()]
            if not in_string and strip_space:
                # replace only white space defined in standard
                tmp2 = re.sub('[ \t\n\r]*', '', tmp2)
            new_str.append(tmp2)

        from_index = match.end()

        if match.group() == '"' and not (inmulticmt or insinglecmt):
            escaped = re.search('(\\\\)*$', json[:match.start()])
            if not in_string or escaped is None or len(escaped.group()) % 2 == 0:
                # start of string with ", or unescaped "
                # character found to end string
                in_string = not in_string
            from_index -= 1   # include " character in next catch
        elif match.group() == '/*' and not (in_string or inmulticmt or insinglecmt):
            inmulticmt = True
        elif match.group() == '*/' and not (in_string or inmulticmt or insinglecmt):
            inmulticmt = False
        elif match.group() == '//' and not (in_string or inmulticmt or insinglecmt):
            insinglecmt = True
        elif ((match.group() == '\n' or match.group() == '\r') and not (
                in_string or inmulticmt or insinglecmt)):
            insinglecmt = False
        elif not (inmulticmt or insinglecmt) and (
                match.group() not in ['\n', '\r', ' ', '\t'] or not strip_space):
            new_str.append(match.group())

    new_str.append(json[from_index:])
    return ''.join(new_str)
