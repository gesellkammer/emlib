"""
Utilities to read and write csv files
"""
from __future__ import annotations
import csv as _csv
import os as _os
import numpy
from fractions import Fraction as _Fraction
from collections import namedtuple as _namedtuple
import re as _re
from .containers import RecordList
from typing import Sequence as Seq, List
import dataclasses
from . import misc


def _as_number_if_possible(s: str, fallback=None, accept_fractions: bool = True,
                           accept_expon=False):
    n = misc.asnumber(s, accept_fractions=accept_fractions, accept_expon=accept_expon)
    return n if n is not None else fallback


def replace_non_alpha(s: str) -> str:
    """
    Remove any non-alphanumeric characters, replace spaces with _

    Args:
        s: the string to sanitize

    Returns:
        a copy of s with all non-alphanumeric characters removed
    """
    TRANSLATION_STRING = '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !"#$%&\'__x+,__/0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff'
    s = s.translate(TRANSLATION_STRING)
    s = _re.sub("[\[\]#:\(\)]", "", s)
    s.replace(" ", "_")
    return s


def _normalize_column_name(name: str) -> str:
    name = replace_non_alpha(name)
    if name and name[0] in '0123456789':
        name = 'n' + name
    name = name.strip().rstrip('_')
    name = name.replace(" ", "")
    return name if name else 'untitled'


def _treat_duplicates(columns: Seq[str]) -> list[str]:
    names: dict[str, int] = {}
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


def readcsv_numpy(csvfile: str) -> numpy.ndarray:
    """
    Read CSV into a numpy array

    Args:
        csvfile: the file to read

    Returns:
        the contents of the file as a 2D numpy array
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


def readcsv(csvfile: str,
            columns: list[str] = None,
            asnumber: bool = True,
            accept_exponential_numbers: bool = False,
            typeconversions: dict = None,
            prefer_fractions: bool = False,
            dialect: str = 'excel',
            first_row_header=True
            ) -> RecordList:
    """
    Read a CSV file into a namedtuple

    If the first collumn is all text, assume these are the column names

    Args:
        columns: a seq of column names, if the first row of data is not
                 a list header
        asnumber: convert strings to numbers if they can be converted
        typeconversions: if given, a dict of the form {column:type}
        accept_exponential_numbers: if True, parse a string 1.5e4 as a number
        prefer_fractions: If True, interpret expressions like 3/4 as Fractions,
            otherwise, as str. 
    
    Returns:
        a :class:`~emlib.containers.RecordList`
    """
    assert dialect in _csv.list_dialects()
    mode = "U"
    f = open(csvfile, mode)
    r = _csv.reader(f, dialect=dialect)
    firstrow = next(r)
    if columns is not None:
        assert isinstance(columns, (tuple, list))
    else:
        if first_row_header and all(misc.asnumber(x) is None for x in firstrow):
            columns = firstrow
        else:
            raise TypeError("Can't infer column names. Pass the column names as arguments.")
    normalized_columns = [_normalize_column_name(col) for col in columns]
    columns = _treat_duplicates(normalized_columns)
    Row = _namedtuple('Row', ' '.join(columns))
    numcolumns = len(columns)
    rows = _Rows()
    for row in r:
        if asnumber:
            row = [_as_number_if_possible(cell, fallback=cell, accept_fractions=prefer_fractions,
                                          accept_expon=accept_exponential_numbers)
                   for cell in row]
        elif typeconversions:
            row = []
            for i, cell in enumerate(row):
                func = typeconversions.get(i)
                if func:
                    cell = func(cell)
                row.append(cell)

        if len(row) == numcolumns:
            rows.append(Row(*row))
        else:
            row.extend([''] * (numcolumns - len(row)))
            row = row[:numcolumns]
            rows.append(Row(*row))
    return RecordList(rows)


def write_records_as_csv(records: list, outfile: str) -> None:
    """
    Write the records as a csv file

    Args:
        records: a list of dataclass objects or namedtuples
            (anything with a '_fields' attribute)
        outfile: the path to save the csv file
    """
    r0 = records[0]
    if dataclasses.is_dataclass(r0):
        column_names = [field.name for field in dataclasses.fields(r0)]
        records = [dataclasses.astuple(rec) for rec in records]
    elif hasattr(r0, "_fields"):
        column_names = r0._fields
    else:
        raise TypeError("records should be a namedtuple or a dataclass")
    f = open(outfile, 'w', newline='', encoding='utf-8')
    w = _csv.writer(f)
    w.writerow(column_names)
    for record in records:
        w.writerow(record)
    f.close()


def writecsv(rows: list, outfile: str, column_names: Seq[str] = None) -> None:
    """
    write a sequence of tuples/named tuples/dataclasses to outfile as CSV

    Args:
        rows: a list of tuples (one per row), namedtuples, dataclasses, etc.
            If namedtuples/dataclasses are passed, the column named are used.
        outfile: the path of the file to write
        column_names: needed if simple tuples/lists are passed
    """
    firstrow = rows[0]
    rowsiter = rows
    if dataclasses.is_dataclass(firstrow):
        if column_names is None:
            fields = dataclasses.fields(firstrow)
            column_names = [f.name for f in fields]
        rowsiter = (dataclasses.astuple(row) for row in rows)
    elif hasattr(firstrow, '_fields'):
        if column_names is None:
            column_names = firstrow._fields
    outfile = _os.path.splitext(outfile)[0] + '.csv'
    f = open(outfile, 'w', newline='', encoding='utf-8')
    f_write = f.write
    w = _csv.writer(f)
    if column_names:
        w.writerow(column_names)
    for row in rowsiter:
        w.writerow(row)
    f.close()
