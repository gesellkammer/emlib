from __future__ import annotations
from collections import namedtuple as _namedtuple
import dataclasses
from keyword import iskeyword as _iskeyword
from collections import OrderedDict as _OrderedDict
import itertools
import tabulate
from typing import Union, Sequence as Seq


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     RecordList: a list of namedtuples
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class RecordList(list):
    """
    A list of namedtuples or records 
    """

    def __init__(self, data:list, fields:Union[str, Seq[str]]=None, itemname:str=None, convert=True):
        """
        
        data     : A seq of namedtuples or dataclass objects
                   A seq. of tuples or lists is also possible. In that case,
                   fields must be given
        fields   : a string as passed to namedtuple
        itemname : The name of each row (optional), overrides the name given for the namedtuples
        convert  : True, data will be converted to namedtuples if they are not already

        Example
        ~~~~~~~

        # generate a RecordList of measures
        >>> Measure = _namedtuple("Measure", "tempo timesig")
        >>> measures = RecordList([Measure(*measure) for measure in (
                (60, (3,4)),
                (60, (4,4)),
                (72, (5,8))
        )])
        """

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

            if convert:
                if not fieldstup:
                    raise ValueError(
                        "If convert is True, data should be either a list of namedtuples, or"
                        "fields should be given"
                    )
                T = _namedtuple(self.item_name, fieldstup)
                data = [T(*row) for row in data]
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

    def __getitem__(self, val) -> Union[tuple, RecordList]:
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

    def get_column(self, column: Union[int, str]) -> list:
        """
        Return a column by name or index as a list of values. 
        
        Raises ValueError if column is not found
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

    def add_column(self, name:str, data, itemname:str=None, missing=None) -> RecordList:
        """
        return a new RecordList with the added data as a column

        If len(data) < len(self), pad data with missing 
        """
        itemname = itemname or self.item_name
        columns = tuple(self.columns) + (name,)
        padded = itertools.chain(data, itertools.repeat(missing))
        newdata = [row + (x,) for row, x in zip(self, padded)]
        r = RecordList(newdata, columns, itemname)
        return r

    def remove_column(self, colname: str) -> RecordList:
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
        self.sort(key=lambda item: getattr(item, column))

    @classmethod
    def from_csv(cls, csvfile:str) -> RecordList:
        from .csvtools import readcsv

        rows = readcsv(csvfile)
        return cls(rows)

    def to_csv(self, outfile:str) -> None:
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
    field_names: a list of strings to be used as attributes.
    
    Example
    =======
    
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


class ClassDict(object):
    def __init__(self, **kws):
        """
        ClassDict: an ordered dictionary with class-like access and lambda evaluated
        definitions

        Use normal keyword definitions to assign values, or a lambda accepting "self"
        to delay evaluation when a value depends on a previous definition (lambdas can 
        depend on lambdas also, see example --watch out for mutually dependign definitions!)

        initdict: a dictionary to be read first (optional)

        Examples
        ~~~~~~~~

        config = ClassDict(
            key1=10,
            key2=20,
            key3=lambda self:self.key1+1
            key4=lambda self:self.key3*2   # 2nd-order lambda
        )
        # now you can access members like variables:
        print(config.key1 + config.key3)
        # --> 10 + (10+1) = 21
        """
        isdelayed = lambda obj: callable(obj)
        ident = 0
        kws2 = _OrderedDict()
        for k, v in kws.items():
            if not isdelayed(v):
                kws2[k] = v
        self.__dict__.update(kws2)
        kws3 = {}
        delayed_defs = kws
        maxiterations = 10
        new_delayed_defs = None
        for i in range(maxiterations):
            new_delayed_defs = {}
            for k, v in delayed_defs.items():
                if isdelayed(v):
                    try:
                        v = v(self)
                        kws3[k] = v
                        self.__dict__[k] = v
                    except AttributeError:
                        new_delayed_defs[k] = v
            if not new_delayed_defs:
                break
            delayed_defs = new_delayed_defs
        if new_delayed_defs:
            raise ValueError("too many delayed definitions!")
        kws2.update(kws3)
        d = dict((k, v) for k, v in kws2.items() if not k.startswith("_"))
        self.__dict__.update(kws3)
        self._dict = _OrderedDict()
        self._dict.update(d)
        self._ident = ident
        self._identstr = " " * ident

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def get(self, attr, default=None):
        out0 = self.__dict__.get(attr)
        out1 = self._dict.get(attr)
        return out0 if out0 is not None else (out1 if out1 is not None else default)

    def __iter__(self):
        d = dict((k, v) for k, v in self.__dict__.items() if not k.startswith("_"))
        return d.items()

    def __repr__(self):
        out = ["{"]
        for attr, value in self:
            out.append("%s%s: %s" % (self._identstr, attr, str(value)))
        out.append("}")
        return "\n".join(out)
