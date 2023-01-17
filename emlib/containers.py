"""
Diverse containers (IntPool, RecordList, ClassDict)
"""
from __future__ import annotations
from collections import namedtuple as _namedtuple
import dataclasses
from keyword import iskeyword as _iskeyword
from collections import OrderedDict as _OrderedDict
import itertools
import tabulate
from typing import Sequence


class FullError(Exception):
    pass


class EmptyError(Exception):
    pass


class IntPool:
    """
    A pool of intergers

    A pool will contain the integers in the range [0, capacity)

    Args:
        capacity: the capacity (size) of the pool.

    Example
    ~~~~~~~

        >>> from emlib.containers import IntPool
        >>> pool = IntPool(10)
        >>> token = pool.pop()
        >>> len(pool)
        9
        >>> pool.push(token)
        >>> len(pool)
        10
        >>> pool.push(4)
        ValueError: token 4 already in pool
    """
    def __init__(self, capacity: int, start=0):
        self.capacity = capacity
        self.pool = set(range(start, start+capacity))
        self.tokenrange = (start, start+capacity)

    def pop(self) -> int:
        if not self.pool:
            raise EmptyError("This pool is empty")
        return self.pool.pop()

    def push(self, token: int) -> None:
        if token in self.pool:
            raise ValueError(f"token {token} already in pool")
        if len(self.pool) == self.capacity:
            raise FullError()
        if not self.tokenrange[0] <= token < self.tokenrange[1]:
            raise ValueError("This token is not part of the pool")
        self.pool.add(token)

    def __contains__(self, item):
        return item in self.pool

    def __len__(self) -> int:
        return len(self.pool)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     RecordList: a list of namedtuples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class RecordList(list):
    """
    A list of namedtuples / dataclasses

    Args: 
        data: A seq of namedtuples or dataclass objects. A seq. of tuples or lists
            is also possible. In that case, fields must be given
        fields: a string as passed to namedtuple
        itemname: The name of each row (optional), overrides the name given for the namedtuples
        convert: True, data will be converted to namedtuples if they are not already


    Example
    ~~~~~~~

    .. code::

        
        # generate a RecordList of measures
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Measure:
        ...     tempo: int
        ...     timesig: tuple[int, int]
        >>> measures = [Measure(tempo, timesig) for tempo, timesig in [(60, (3, 4)), (60, (4, 4)), (72, (5, 8))]]
        >>> measurelist = RecordList(measures)
        >>> measurelist.get_column('tempo')
        [60, 60, 72]
    """

    def __init__(self,
                 data: list,
                 fields: str | Sequence[str] = '',
                 itemname=''):
        
        if not data and not fields:
            raise ValueError("data is empty, fields must be given")

        def _is_list_of_namedtuples(data) -> bool:
            return isinstance(data, list) and len(data) > 0 and hasattr(data[0], "_fields")

        if itemname:
            self.item_name = itemname
        elif _is_list_of_namedtuples(data):
            self.item_name = data[0].__class__.__name__
        else:
            self.item_name = "Row"

        self._name = "{name}s".format(name=self.item_name)

        data_already_namedtuples = _is_list_of_namedtuples(data)
        if data_already_namedtuples and fields is None:
            fieldstup = data[0]._fields
        else:
            if not fields:
                raise ValueError("A seq. of namedtuples must be given or a seq. of tuples. "
                                 "For the latter, 'fields' must be specified."
                                 f"(got {type(data[0])}")
            fieldstup = fields.split() if isinstance(fields, str) else tuple(fields)
            fieldstup = _validate_fields(fieldstup)

        list.__init__(self, data)

        if data_already_namedtuples:
            try:
                make = data[0]._make
                self._item_constructor = lambda *args: make(args)
            except AttributeError:
                self._item_constructor = None
        else:
            self._item_constructor = None
        self.columns = fieldstup

    def __repr__(self):
        return tabulate.tabulate(
            self, self.columns, disable_numparse=True, showindex=True
        )

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, showindex=True) -> str:
        return tabulate.tabulate(
            self,
            self.columns,
            tablefmt="html",
            disable_numparse=True,
            showindex=showindex,
        )

    def __getitem__(self, val) -> tuple | RecordList:
        if isinstance(val, int):
            return list.__getitem__(self, val)
        else:
            recs = list.__getitem__(self, val)
            return RecordList(recs)

    def reversed(self) -> RecordList:
        """
        return a reversed copy of self
        """
        return RecordList(list(reversed(self)), itemname=self.item_name)

    def copy(self) -> RecordList:
        """
        return a copy of self
        """
        return RecordList(self, itemname=self.item_name)

    # ######################################################
    #  columns
    # ######################################################

    def get_column(self, column: int | str) -> list:
        """
        Return a column by name or index as a list of values. 
        
        Raises ValueError if column is not found

        Args:
            column: the column to get, as index or column name

        Returns:
            a list with the values
        """
        if isinstance(column, int):
            index = column
        elif isinstance(column, str):
            try:
                index = self.columns.index(column)
            except ValueError:
                raise ValueError(f"column {column} not found")
        else:
            raise TypeError("column should be a label (str), or an index (int)")
        return [item[index] for item in self]

    def add_column(self, name: str, data, itemname: str = '', missing=None) -> RecordList:
        """
        Return a new RecordList with the added data as a column

        If len(data) < len(self), pad data with missing

        Args:
            name: the name of the new column
            data: the data of the column
            itemname: the name of each item
            missing: value to use when padding is needed

        Returns:
            the resulting RecordList 
        """
        itemname = itemname or self.item_name
        columns = tuple(self.columns) + (name,)
        padded = itertools.chain(data, itertools.repeat(missing))
        newdata = [row + (x,) for row, x in zip(self, padded)]
        r = RecordList(newdata, columns, itemname)
        return r

    def remove_column(self, colname: str) -> RecordList:
        """
        Return a new RecordList with the column removed

        Args:
            colname: the name of the column to remove

        Returns:
            the resulting RecordList
        """
        if colname not in self.columns:
            return self
        return self.get_columns([col for col in self.columns if col != colname])

    #######################################################
    # operations with other RecordLists
    #######################################################

    def merge_with(self, other: RecordList) -> RecordList:
        """
        A new list is returned with a union of the fields of self and other
        
        If there are fields in common, other prevails (similar to dict.update)
        If self and other have a different number of rows, the lowest 
        is taken.

        Args:
            other: the RecordList to merge with

        Returns:
            the merged RecordList 
        """
        if not isinstance(other, list) or not hasattr(other, "columns"):
            raise TypeError("other should be a RecordList")
        columns = list(self.columns)
        for othercol in other.columns:
            if othercol not in columns:
                columns.append(othercol)
        coldata = []
        for col in columns:
            if col in other.columns:
                coldata.append(other.get_column(col))
            else:
                coldata.append(self.get_column(col))
        return RecordList(list(zip(*coldata)), columns)

    def get_columns(self, columns: list[str]) -> RecordList:
        """
        Returns a new RecordList with the selected columns

        Args:
            columns: a list of column names

        Returns:
            the resulting RecordList
        """
        data_columns = [self.get_column(column) for column in columns]
        data = zip(*data_columns)
        constructor = _namedtuple(self.item_name, columns)
        items = [constructor(*row) for row in data]
        return RecordList(items)

    @property
    def item_constructor(self):
        if self._item_constructor is None:
            c = _namedtuple(self.item_name, self.columns)
            self._item_constructor = c
        return self._item_constructor

    def sort_by(self, column: str) -> None:
        """
        Sort this RecordList (in place) by the given column

        Args:
            column: the column name to use to sort this RecordList

        """
        self.sort(key=lambda item: getattr(item, column))

    @classmethod
    def from_csv(cls, csvfile: str) -> RecordList:
        """
        Create a new RecordList with the data in csvfile

        """
        from .csvtools import readcsv

        rows = readcsv(csvfile)
        return cls(rows)

    def to_csv(self, outfile: str) -> None:
        """
        Write the data in this RecordList as a csv file
        """
        from .csvtools import writecsv
        writecsv(self, outfile, column_names=self.columns)

    @classmethod
    def from_dataframe(cls, dataframe, itemname="row") -> RecordList:
        """
        create a RecordList from a pandas.DataFrame
        """
        columns = _validate_fields(list(dataframe.keys()))
        Row = _namedtuple(itemname, columns)
        out = []
        for i in range(dataframe.shape[0]):
            row = list(dataframe.irow(i))
            out.append(Row(*row))
        return cls(out, itemname=itemname)

    def to_dataframe(self):
        """
        create a pandas.DataFrame from this RecordList
        """
        try:
            import pandas
            return pandas.DataFrame(list(self), columns=self.columns)
        except ImportError:
            raise ImportError("pandas is needed to export to pandas.DataFrame!")


def _validate_fields(field_names: list[str]) -> list[str]:
    """
    Validate the given field names
    
    Args:
        field_names: a list of strings to be used as attributes.
    
    Example
    =======

    .. code::
    
        # Numbers are not valid identifiers
        # an object cannot have non-unique attributes
        >>> _validate_fields(["0", "field", "field"])
        ['_0', 'field', '_2']

    """
    names = list(map(str, list(field_names)))
    seen = set()
    for i, name in enumerate(names):
        if (
            not all(c.isalnum() or c == "_" for c in name)
            or _iskeyword(name)
            or not name
            or name[0].isdigit()
            or name.startswith("_")
            or name in seen
        ):
            names[i] = "field%d" % i
        seen.add(name)
    return names