"""
Simple dialogs for use at the repl
"""
from __future__ import annotations
import os
import sys
import emlib.misc
from typing import TYPE_CHECKING
import tkinter as tk
import tkinter.font
from tkinter import ttk

if TYPE_CHECKING:
    from typing import *

_DEFAULT_FONT = ("Helvetica", 11)


def _popupmsg_tk(msg, title="", buttontxt="Ok", font=_DEFAULT_FONT):
    from ttkthemes import ThemedTk
    root = ThemedTk(theme="breeze")
    root.title(title)
    s = ttk.Style()
    bg = "#f5f5f5"
    s.configure('new.TFrame', background=bg)
    ttk.Style().configure("TButton", padding=8, relief="flat",
                          background="#e0e0e0")
    frame = ttk.Frame(root, style="new.TFrame")
    dx, dy = 8, 8
    ttk.Label(frame, text=msg, font=font, background=bg).grid(column=0, row=0, padx=dx*3, pady=dy*2)
    ttk.Button(frame, text=buttontxt, command=root.destroy).grid(column=0, row=1, padx=dx, pady=dy)
    frame.grid(column=0, row=0)
    root.mainloop()


def popupMsg(msg:str, title="", buttontxt="Ok") -> None:
    """
    Open a pop-up dialog with a message
    """
    return _popupmsg_tk(msg=msg, title=title, buttontxt=buttontxt, font=_DEFAULT_FONT)


def showInfo(msg:str, title:str="Info", font=None) -> None:
    """
    Show a pop up dialog with some info
    """
    from ttkthemes import ThemedTk
    root = ThemedTk(theme="breeze")
    root.title(title)

    bg = "#f5f5f5"
    # frame = ttk.Frame(root, style=".showinfo.TFrame")
    frame = ttk.Frame(root)
    dx, dy = 8, 8
    if font is None:
        font = (_DEFAULT_FONT[0], int(_DEFAULT_FONT[1]*1.3))
    ttk.Label(frame, text="ℹ  " + msg, font=font, background=bg
              ).grid(column=0, row=0, padx=dx*2, pady=dy*2)
    ttk.Button(frame, text="Ok", command=root.destroy
               ).grid(column=0, row=1, padx=dx, pady=dy)
    frame.grid(column=0, row=0)
    root.bind("<Escape>", lambda *args: root.destroy())
    root.mainloop()


def selectFile(directory:str=None, filter="All (*.*)", title="Open file",
               backend:str=None) -> str:
    """
    Create a dialog to open a file and returns the file selected

    Args:
        filter: a string of the form "<Mask> (<glob>)". Multiple filters can be
            used, for example: ``"Image (*.png, *.jpg);; Video (*.mp4, *.mov)"``
        title: the title of the dialog
        directory: the initial directory
        backend: one of qt, tk, or None to select a default

    Returns:
        the selected filename, or an empty string if the dialog is dismissed
    """
    if _has_qt() and (backend == 'qt' or backend is None):
        return _opendialog_qt(directory=directory, filter=filter, title=title)

    from ttkthemes import ThemedTk
    from tkinter import filedialog

    filetypes = _tkParseFilter(filter)
    root = ThemedTk(theme='breeze')
    root.withdraw()

    path = filedialog.askopenfilename(initialdir=directory, title=title,
                                      filetypes=filetypes)

    root.destroy()
    return path

def _opendialog_qt(directory:str=None, filter="All (*.*)", title="Open file") -> str:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    filter = filter.replace(",", " ")
    name, mask = QtWidgets.QFileDialog.getOpenFileName(None, title, directory=directory, 
                                                       filter=filter)
    return name


@emlib.misc.runonce
def _has_qt() -> bool:
    try:
        from PyQt5 import QtWidgets
        return True
    except ImportError:
        return False


def _savedialog_qt(filter="All (*.*)", title="Save file", directory:str=None) -> str:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    filter = filter.replace(",", " ")
    name, mask = QtWidgets.QFileDialog.getSaveFileName(None, title, filter=filter, 
                                                       directory=directory)
    return name


def _tkParseFilter(filter:str) -> List[Tuple[str, str]]:
    parts = filter.split(";;")
    out = []
    for part in parts:
        part = part.strip()
        if "(" in part and part[-1] == ")":
            name, wildcard = part[:-1].split("(")
            out.append((name, wildcard))
        else:
            out.append(('', part))
    return out


def _savedialog_tk(filter="All (*.*)", title="Save file", directory:str="~") -> str:
    from ttkthemes import ThemedTk
    from tkinter import filedialog

    filetypes = _tkParseFilter(filter)
    root = ThemedTk(theme='breeze')
    root.withdraw()
    
    path = filedialog.asksaveasfilename(initialdir=directory, title=title, filetypes=filetypes)
    root.destroy()
    return path


def saveDialog(filter="All (*.*)", title="Save file", directory:str="~", backend:str=None
               ) -> str:
    """
    Open a dialog to save a file.

    Args:
        filter: a string of the form "<Mask> (<glob>)". Multiple filters can be
            used, for example: ``"Image (*.png, *.jpg);; Video (*.mp4, *.mov)"``
        title: the title of the dialog
        directory: the initial directory
        backend: one of 'qt', 'tkinter' or None to select the backend based on available
            packages
        
    Returns:
        the save filename, or an empty string if the dialog is dismissed
    """
    if not directory:
        directory = "~"
    directory = os.path.expanduser(directory)
    if _has_qt() and (backend == 'qt' or backend is None):
        return _savedialog_qt(filter=filter, title=title, directory=directory)
    else:
        return _savedialog_tk(filter=filter, title=title, directory=directory)


def _combowin_qt(options: List[str], title="Select", width=300, height=40) -> Opt[str]:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, QSortFilterProxyModel
    from PyQt5.QtWidgets import QCompleter, QComboBox

    class ExtendedComboBox(QComboBox):
        def __init__(self, parent=None):
            super(ExtendedComboBox, self).__init__(parent)
            self.dismissed = False
            self.setWindowTitle(title)

            self.setFocusPolicy(Qt.StrongFocus)
            self.setEditable(True)

            # add a filter model to filter matching items
            self.pFilterModel = QSortFilterProxyModel(self)
            self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
            self.pFilterModel.setSourceModel(self.model())

            # add a completer, which uses the filter model
            self.completer = QCompleter(self.pFilterModel, self)
            # always show all (filtered) completions
            self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
            self.setCompleter(self.completer)

            # connect signals
            self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
            self.completer.activated.connect(self.on_completer_activated)

        # on selection of an item from the completer, select the corresponding item from combobox
        def on_completer_activated(self, text):
            if text:
                index = self.findText(text)
                self.setCurrentIndex(index)
                self.activated[str].emit(self.itemText(index))

        # on model change, update the models of the filter and completer as well
        def setModel(self, model):
            super(ExtendedComboBox, self).setModel(model)
            self.pFilterModel.setSourceModel(model)
            self.completer.setModel(self.pFilterModel)

        # on model column change, update the model column of the filter and completer as well
        def setModelColumn(self, column):
            self.completer.setCompletionColumn(column)
            self.pFilterModel.setFilterKeyColumn(column)
            super(ExtendedComboBox, self).setModelColumn(column)

        def keyPressEvent(self, e):
            if e.key() == Qt.Key_Escape:
                self.dismissed = True
                self.close()
            elif e.key() == Qt.Key_Enter or e.key() == Qt.Key_Return:
                self.close()
            else:
                super().keyPressEvent(e)

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QStringListModel
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    combo = ExtendedComboBox()

    # either fill the standard model of the combobox
    # combo.addItems(options)

    # or use another model
    combo.setModel(QStringListModel(options))

    combo.resize(300, 40)
    combo.show()
    app.exec_()
    return None if combo.dismissed else combo.currentText()


def selectFromCombobox(items: List[str], title:str= '', prompt:str= 'Select',
                       promptFontSize=13, comboFontSize=11, dropdownFontSize=11
                       ) -> Optional[int]:
    """
    Select one option from a combo-box

    Args:
        items: list of options
        title: title of the window
        prompt: label to indicate what the selection is for
        promptFontSize: font size of the prompt
        comboFontSize: font size of the combo-box
        dropdownFontSize: font size of the drop-down

    Returns:
        the index of the option selected, if a selection was made, or None otherwise
    """
    from ttkthemes import ThemedTk

    # root = tk.Tk()
    root = ThemedTk(theme='breeze')

    root.resizable(False, False)
    root.title(title)

    pad = {'padx':10, 'pady':5}

    label = ttk.Label(text=prompt, font=('Arial', promptFontSize))
    label.grid(row=0, column=0, sticky=tk.W + tk.E, columnspan=2, **pad)

    selectedOption = tk.StringVar()
    width = int(max(len(o) for o in items) * 0.9)
    combobox = ttk.Combobox(root, textvariable=selectedOption, font=('Arial', comboFontSize),
                            width=width)
    combobox['values'] = items if isinstance(items, (list, tuple)) else list(items)
    combobox['state'] = 'readonly'  # normal
    combobox.current(0)
    combobox.grid(row=1, column=0, sticky=tk.W + tk.E, columnspan=2, **pad)

    root.option_add("*TCombobox*Listbox*Font", ('Arial', dropdownFontSize))

    out = None

    def ok():
        nonlocal out
        value = combobox.get()
        out = items.index(value)
        root.destroy()

    def cancel():
        root.destroy()

    ttk.Button(root, text="Cancel", command=cancel
              ).grid(row=2, column=0, sticky=tk.E, **pad)

    ttk.Button(root, text="OK", command=ok
               ).grid(row=2, column=1, sticky=tk.W, **pad)

    root.bind("<Escape>", lambda *args:cancel())
    root.bind("<Return>", lambda *args:ok())

    root.mainloop()
    return out


def selectItem(items:Sequence[str], title="Select", entryFont=('Arial', 15),
               listFont=('Arial', 12), scrollbar=True, width=400, numlines=20,
               caseSensitive=False, ensureSelection=False
               ) -> Optional[str]:
    """
    Select one item from a list

    Args:
        items: the list of options
        title: the title of the dialog
        entryFont: the font of the filter text entry (a tuple (font, size))
        listFont: the font of the list (a tuple (font, size))
        scrollbar: if True, add a scrollbar
        width: the width in pixels
        numlines: the number of lines to display at a time
        caseSensitive: if True, filtering is case sensitive
        ensureSelection: if True, raises a ValueError exception is no selection
            was done

    Returns:
        either the selected item or None
    """
    selected = selectFromList(items=items, title=title, entryFont=entryFont,
                              listFont=listFont, scrollbar=scrollbar,
                              width=width, numlines=numlines,
                              caseSensitive=caseSensitive,
                              ensureSelection=ensureSelection)
    return selected[0] if selected else None


def _tkMeasureTextWidth(font: Tuple[str, int], text: str, correctionFactor=1.1) -> int:
    tkfont = tk.font.Font(font=font)
    return int(tkfont.measure(text) * correctionFactor)


def selectFromList(items:Sequence[str], title="Select", entryFont=('Arial', 15),
                   listFont=('Arial', 12), scrollbar=True, width=400, numlines=20,
                   caseSensitive=False, ensureSelection=False
                   ) -> List[str]:
    """
    Select one or multiple items from a list

    Args:
        items: the list of options
        title: the title of the dialog
        entryFont: the font of the filter text entry (a tuple (font, size))
        listFont: the font of the list (a tuple (font, size))
        scrollbar: if True, add a scrollbar
        width: the width in pixels
        numlines: the number of lines to display at a time
        caseSensitive: if True, filtering is case sensitive
        ensureSelection: if True, raises a ValueError exception is no selection
            was done

    Returns:
        a list of selected items, or an empty list if the user aborted
        (via Escape or closing the window)
    """
    from ttkthemes import ThemedTk
    if len(items) < numlines:
        scrollbar = False
    numlines = min(numlines, len(items))
    root = ThemedTk(theme="breeze")
    root.title(title)
    root.columnconfigure(0, weight=1)

    longest = max((item for item in items), key=len)
    minwidth = _tkMeasureTextWidth(listFont, longest)
    width = max(width, minwidth)

    filterval = tk.StringVar()
    entry = ttk.Entry(root, textvariable=filterval, font=entryFont
                      )
    entry.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    treestyle = ttk.Style()
    treestyle.configure("selectFromList.Treeview", highlightthickness=0, bd=0,
                        font=listFont)
    treestyle.layout("selectFromList.Treeview", [
        ('selectFromList.Treeview.treearea', {'sticky':'nswe'})])
    tree = ttk.Treeview(root, height=numlines, show='tree', style='selectFromList.Treeview'
                        )
    tree.grid(row=1, column=0, sticky='nsew')
    tree.column("#0", minwidth=0, width=width, stretch=False)

    # adding data to the treeview
    itemids = [tree.insert('', tk.END, text=item, open=False)
               for item in items]

    id2item = {i:c for i, c in zip(itemids, items)}

    # add a scrollbar

    if scrollbar:
        scrollbarWidget = ttk.Scrollbar(root, orient=tk.VERTICAL, command=tree.yview)
        scrollbarWidget.grid(row=1, column=1, sticky='ns')
        tree.configure(yscroll=scrollbarWidget.set)
    else:
        tree.configure(yscroll=None)

    id2visible = {i:True for i in itemids}

    def applyfilter(text, caseSensitive):
        idx = 0
        if not caseSensitive:
            text = text.lower()
        for c, i in zip(items, itemids):
            if not caseSensitive:
                c = c.lower()
            if text in c:
                if not id2visible[i]:
                    tree.reattach(i, '', idx)
                    id2visible[i] = True
                idx += 1
            elif id2visible[i]:
                tree.detach(i)
                id2visible[i] = False
        ch = tree.get_children()
        if ch:
            tree.focus(ch[0])
            tree.selection_set(ch[0])

    out = [None]

    def accept(*args):
        sels = tree.selection()
        values = [id2item[sel] for sel in sels]
        out[0] = values
        root.destroy()

    def entrymove(step=1):
        tree.focus_set()
        sel = tree.selection()
        if sel:
            item = tree.next(sel[0]) if step == 1 else tree.prev(sel[0])
            if not item:
                item = sel[0]
        else:
            children = tree.get_children()
            if not children:
                return
            item = children[0 if step == 1 else -1]
        tree.selection_set(item)
        tree.focus(item)

    def entrykey(k):
        s = filterval.get()+k.char
        filterval.set(s)
        entry.icursor(len(s))
        entry.focus_set()

    def entryback():
        s = filterval.get()[:-1]
        filterval.set(s)
        entry.icursor(len(s))
        entry.focus_set()

    root.bind("<Escape>", lambda *args:root.destroy())
    root.bind("<Return>", accept)
    entry.bind("<KeyRelease>",
               lambda *args:applyfilter(filterval.get(), caseSensitive=caseSensitive))
    entry.bind("<Down>", lambda *args:entrymove(1))
    entry.bind("<Up>", lambda *args:entrymove(-1))
    for k in "abcdefghijklmnopqrstuvwxyzABCEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        tree.bind(k, entrykey)
    tree.bind("<BackSpace>", lambda *args:entryback())
    tree.bind('<Double-Button-1>', accept)

    tree.focus(itemids[0])
    tree.selection_set(itemids[0])
    entry.focus_set()

    root.mainloop()
    sel = out[0]
    if not sel and ensureSelection:
        raise ValueError("No selection was done")
    return sel if sel else []
