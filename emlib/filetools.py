"""
Utilities to work with files and filenames
"""
from __future__ import annotations
import datetime
import os
from typing import List, Optional as Opt
from . import misc


def modifiedDate(filename: str) -> datetime.date:
    """
    get the modified time of f as a datetime.date

    Args:
        filename: the file name for which to query the modified data
        
    Returns:
        the modification date, as a datetime.date
    """
    t = os.path.getmtime(filename)
    return datetime.date.fromtimestamp(t)


def filesBetween(files: List[str], start, end) -> List[str]:
    """
    Returns files between the given times

    Args:
        files: a list of files
        start: a tuple as would be passed to datetime.date
        end  : a tuple as would be passed to datetime.date

    Returns:
        a list of files
    """
    t0 = datetime.date(*start) if not isinstance(start, datetime.date) else start
    t1 = datetime.date(*end) if not isinstance(end, datetime.date) else end
    return [f for f in files if t0 <= modifiedDate(f) < t1]


def fixLineEndings(filename:str) -> None:
    """
    Convert any windows line endings (\r\n) to unix line endings (\n)
    """
    f = open(filename, 'rb')
    # read the beginning, look if there are CR
    data = f.read(100)
    if b'\r' in data:
        # read all in memory, this files should not be so big anyway
        data = data + f.read()
        data = data.replace(b'\r\n', b'\n')
        f.close()
        # write it back, now with LF
        open(filename, 'wb').write(data)
    else:
        f.close()


def findFile(path: str, file: str) -> Opt[str]:
    """
    Look for file recursively starting at path.

    If file is found in path or any subdir, the complete path is returned
    else None

    Args:
        path: the path to start searching
        file: the file to find

    Returns:
        the absolute path or None if the file was not found
    """
    dir_cache = set()
    for directory in os.walk(path):
        if directory[0] in dir_cache:
            continue
        dir_cache.add(directory[0])
        if file in directory[2]:
            return '/'.join((directory[0], file))
    return None


def addSuffix(filename: str, suffix: str) -> str:
    """
    Add a suffix between the name and the extension

    Args:
        filename: the filename to add a suffix to
        suffix: the suffix to add

    Returns:
        the modified filename
        
    Example
    -------

        >>> name = "test.txt"
        >>> newname = addSuffix(name, "-OLD")
        >>> newname
        test-OLD.txt
        >>> os.rename(name, newname)

    This does NOT rename the file, it merely returns the string
    """
    name, ext = os.path.splitext(filename)
    return ''.join((name, suffix, ext))


def withExtension(filename: str, extension: str) -> str:
    """
    Return a new filename where the original extension has
    been replaced with `extension`

    Args:
        filename: the filename to modify
        extension: the new extension

    Returns:
        a filename with the given extension in place of the old extension
    

    ============  ==========   =============
    filename      extension     output
    ============  ==========   =============
    foo.txt       .md           foo.md
    foo.txt       md            foo.md
    foo.bar.baz   zip           foo.bar.zip
    ============  ==========   =============
    
    """
    if not extension.startswith("."):
        extension = "." + extension
    base = os.path.splitext(filename)[0]
    return base + extension


def increaseSuffix(filename: str) -> str:
    """
    Given a filename, return a new filename with an increased suffix if 
    the filename already has a suffix, or a suffix if it hasn't

    =============  ===========
    input          output
    =============  ===========
    foo.txt        foo-01.txt
    foo-01.txt     foo-02.txt
    foo-2.txt      foo-03.txt
    =============  ===========
    
    """
    name, ext = os.path.splitext(filename)
    tokens = name.split("-")

    def increase_number(number_as_string):
        n = int(number_as_string) + 1
        s = "%0" + str(len(number_as_string)) + "d"
        return s % n

    if len(tokens) > 1:
        suffix = tokens[-1]
        if misc.asnumber(suffix) is not None:
            new_suffix = increase_number(suffix)
            new_name = name[:-len(suffix)] + new_suffix
        else:
            new_name = name + '-01'
    else:
        new_name = name + '-01'
    return new_name + ext


def normalizePath(path:str) -> str:
    """
    Convert `path` to an absolute path with user expanded
    (something that can be safely passed to a subprocess)
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))
