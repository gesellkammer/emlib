import datetime
import os


def modified_date(f):
    """
    get the modified time of f as a datetime.date
    """
    t = os.path.getmtime(f)
    date = datetime.date.fromtimestamp(t)
    return date


def filesbetween(files, start, end):
    """
    files: a list of files
    start: a tuple as would be passed to datetime.date
    end  : %
    """
    t0 = datetime.date(*start) if not isinstance(start, datetime.date) else start
    t1 = datetime.date(*end) if not isinstance(end, datetime.date) else end
    out = []
    for f in files:
        t = modified_date(f)
        if t0 <= t < t1:
            out.append(f)
    return out
