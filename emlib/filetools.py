from __future__ import annotations
import datetime
import os
from emlib.typehings import List, Opt


def modifiedDate(f: str) -> datetime.date:
    """
    get the modified time of f as a datetime.date
    """
    t = os.path.getmtime(f)
    return datetime.date.fromtimestamp(t)


def filesBetween(files: List[str], start, end) -> List[str]:
    """
    files: a list of files
    start: a tuple as would be passed to datetime.date
    end  : %
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
    ( /this/is/a/path/filename )

    else None
    """
    dir_cache = set()  # type: t.Set[str]
    for directory in os.walk(path):
        if directory[0] in dir_cache:
            continue
        dir_cache.add(directory[0])
        if file in directory[2]:
            return '/'.join((directory[0], file))
    return None


def addSuffix(filename: str, suffix: str) -> str:
    # type: (str, str) -> str
    """
    add a suffix between the name and the extension

    addSuffix("test.txt", "-OLD") == "test-OLD.txt"

    This does NOT rename the file, it merely returns the string
    """
    name, ext = _os.path.splitext(filename)
    return ''.join((name, suffix, ext))


def increaseSuffix(filename: str) -> str:
    """
    Given a filename, return a new filename with an increased suffix if 
    the filename already has a suffix, or a suffix if it hasn't

    input          output
    --------------------------------
    foo.txt        foo-01.txt
    foo-01.txt     foo-02.txt
    foo-2.txt      foo-03.txt
    """
    name, ext = _os.path.splitext(filename)
    tokens = name.split("-")

    def increase_number(number_as_string):
        n = int(number_as_string) + 1
        s = "%0" + str(len(number_as_string)) + "d"
        return s % n

    if len(tokens) > 1:
        suffix = tokens[-1]
        if could_be_number(suffix):
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
    return _os.path.abspath(_os.path.expanduser(path))
