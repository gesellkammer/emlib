from __future__ import absolute_import as _absimport, division as _division, print_function
import csv as _csv
import os as _os
import numpy
from fractions import Fraction as _Fraction
from collections import namedtuple as _namedtuple
import re as _re
from numbers import Number as _Number
from .containers import RecordList
from typing import Sequence as Seq, List, Optional as Opt, Union


def _could_be_number(s):
    # type: (str) -> bool
    try:
        n = eval(s)
        return isinstance(n, _Number)
    except TypeError:
        return False


def _as_number_if_possible(s, accept_fractions=True):
    # type: (str, bool) -> Union[int, float, _Fraction, None]
    """try to convert 's' to a number if it is possible, return None otherwise"""
    if accept_fractions:
        if "/" in s:
            try:
                n = _Fraction("/".join(n.strip() for n in s.split("/")))
                return n
            except ValueError:
                return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None
    return None


def replace_non_alfa(s):
    # type: (str) -> str
    TRANSLATION_STRING = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !"#$%&\'__x+,__/0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'
    s = s.translate(TRANSLATION_STRING)
    s = _re.sub("[\[\]#:\(\)]", "", s)
    s.replace(" ", "_")
    return s


def _normalize_column_name(name):
    # type: (str) -> str
    name = replace_non_alfa(name)
    if name and name[0] in '0123456789':
        name = 'n' + name
    name = name.strip().rstrip('_')
    return name if name else 'untitled'


def _treat_duplicates(columns):
    # type: (Seq[str]) -> List[str]
    names = {}      # type: dict[str, int]
    new_names = []
    for column in columns:
        if column not in names:
            names[column] = 1
            new_name = column
        else:
            n = names[column]
            n += 1
            names[column] = n
            new_name = "%s_%d" % (column, n)
        new_names.append(new_name)
    return new_names


def readcsv_numpy(csvfile):
    # type: (str) -> numpy.ndarray
    """
    Read CSV into a numpy array
    """
    return numpy.genfromtxt(csvfile, names=None, delimiter=',')


class _Rows(list):
    def __init__(self, seq=None):
        super(_Rows, self).__init__()
        self._firstappend = True
        self.columns = None
        if seq:
            for elem in seq:
                self.append(elem)

    def append(self, namedtup):
        if self._firstappend:
            self.columns = namedtup._fields
        list.append(self, namedtup)


def readcsv(csvfile, columns=None, asnumber=True, prefer_fractions=False, 
            rowname=None, dialect='excel'):
    # type: (str, Opt[List[str]], bool, bool, Opt[str], str) -> RecordList
    """
    read a CSV file into a namedtuple

    if the first collumn is all text: assume these are the column names

    columns: a seq of column names, if the first row of data is not
             a list header
    rowname: override the row name specified in the CSV file (if any)
    asnumber: convert strings to numbers if they can be converted
    prefer_fractions: If True, interpret expressions like 3/4 as Fractions,
                      otherwise, as str. 
    """
    assert dialect in _csv.list_dialects()
    mode = "U"
    f = open(csvfile, mode)
    r = _csv.reader(f, dialect=dialect)
    try:
        firstrow = next(r)
    except:
        mode = mode + 'U'
        f = open(csvfile, mode + 'U')
        r = _csv.reader(f, dialect=dialect)
        firstrow = next(r)
    attributes = {}
    if firstrow[0].startswith('#'):
        # the first row contains attributes
        f.close()
        f = open(csvfile, mode)
        attribute_line = f.readline()
        attrs = attribute_line[1:].split()
        for attr in attrs:
            key, value = attr.split(':')
            attributes[key] = value
        r = _csv.reader(f, dialect=dialect)
        firstrow = next(r)
    if columns is not None:
        assert isinstance(columns, (tuple, list))
    else:
        if not any(_could_be_number(x) for x in firstrow) or firstrow[0].startswith('#'):
            columns = firstrow
        else:
            print("Can't assume column names. Pass the column names as arguments")
            raise TypeError("Number-like cells found in the first-row")
    normalized_columns = [_normalize_column_name(col) for col in columns]
    columns = _treat_duplicates(normalized_columns)
    rowname = rowname if rowname is not None else 'Row'
    Row = _namedtuple(rowname, ' '.join(columns))
    numcolumns = len(columns)
    rows = _Rows()
    for row in r:
        if asnumber:
            row = [_as_number_if_possible(cell, accept_fractions=prefer_fractions)
                   for cell in row]
        if len(row) == numcolumns:
            rows.append(Row(*row))
        else:
            row.extend([''] * (numcolumns - len(row)))
            row = row[:numcolumns]
            rows.append(Row(*row))
    return RecordList(rows)


def writecsv(namedtuples, outfile, column_names=None, write_row_name=False):
    """
    write a sequence of named tuples to outfile as CSV

    alternatively, you can also specify the column_names. in this case it
    is not necessary for the tuples to be be namedtuples
    """
    firstrow = namedtuples[0]
    isnamedtuple = hasattr(firstrow, '_fields')
    if isnamedtuple and column_names is None:
        column_names = firstrow._fields
    outfile = _os.path.splitext(outfile)[0] + '.csv'

    f = open(outfile, 'w', newline='', encoding='utf-8')
    f_write = f.write
    w = _csv.writer(f)
    if isnamedtuple and write_row_name:
        try:
            # this is a hack! where is the name of a namedtuple??
            rowname = firstrow.__doc__.split('(')[0]  
        except AttributeError:  # maybe not a namedtuple in the end
            rowname = firstrow.__class__.__name__
        line = "# rowname:%s\n" % rowname
        f_write(line)
    if column_names:
        w.writerow(column_names)
    for row in namedtuples:
        w.writerow(row)
    f.close()


def _to_number(x, accept_fractions=True):
    if _could_be_number(x):
        if '.' in x or x in ('nan', 'inf', '-inf'):
            return float(x)
        else:
            try:
                return int(x)
            except:
                try:
                    return _Fraction(x)
                except:
                    return x
    else:
        return x
