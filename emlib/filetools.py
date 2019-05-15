import datetime
import os
import emlib.typehints as t


def modified_date(f: str) -> datetime.date:
    """
    get the modified time of f as a datetime.date
    """
    t = os.path.getmtime(f)
    return datetime.date.fromtimestamp(t)


def files_between(files: t.List[str], start, end) -> t.List[str]:
    """
    files: a list of files
    start: a tuple as would be passed to datetime.date
    end  : %
    """
    t0 = datetime.date(*start) if not isinstance(start, datetime.date) else start
    t1 = datetime.date(*end) if not isinstance(end, datetime.date) else end
    return [f for f in files if t0 <= modified_date(f) < t1]
