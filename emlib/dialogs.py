"""
Simple dialogs for use at the repl / ipython / jupyter

At the moment the better supported backend is qt5, which works
in all three major platforms.
"""
from __future__ import annotations
import os
import sys
import emlib.misc
import logging


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Tuple, Sequence, Optional

_DEFAULT_FONT = ("Helvetica", 11)

logger = logging.getLogger(__name__)

__all__ = (
    'showInfo',
    'selectFile',
    'saveDialog',
    'selectItem',
    'selectItems',
    'filters'
)


filters = {
    'All': 'All (*.*)',
    'Sound': 'Sound (*.wav, *.aiff, *.flac, *.mp3)',
    'Image': 'Image (*.jpg, *.png)'
}


@emlib.misc.runonce
def _has_qt() -> bool:
    try:
        from PyQt5 import QtWidgets
        return True
    except ImportError:
        return False



@emlib.misc.runonce
def _has_tk() -> bool:
    try:
        import tkinter
        return True
    except ImportError:
        return False


@emlib.misc.runonce
def _resolveBackend(backend: str = None):
    if sys.platform == 'darwin':
        backend = 'qt'
    elif backend is None:
        if _has_qt():
            backend = 'qt'
        elif _has_tk():
            backend = 'tk'
        else:
            raise RuntimeError("No backends available")
    if backend == 'qt' and not _has_qt():
        raise RuntimeError("pyqt5 is needed not installed. Install it via 'pip install pyqt5'")
    return backend


def showInfo(msg: str, title: str = "Info", font=None, icon: str = None, backend: str = None
             ) -> None:
    """
    Show a pop up dialog with some info

    Args:
        msg: the text to display (one line)
        title: the title of the dialog
        font: if given, a tuple (fontfamily, size)
        icon: either None or one of 'question', 'information', 'warning', 'critical'
    """
    backend = _resolveBackend(backend)
    if backend == 'qt':
        from . import _dialogsqt
        return _dialogsqt.showInfo(msg=msg, title=title, font=font, icon=icon)
    from tkinter import ttk
    from ttkthemes import ThemedTk
    root = ThemedTk(theme="breeze")
    root.title(title)

    bg = "#f5f5f5"
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


def selectFile(directory: str = None, filter="All (*.*)", title="Open file",
               backend: str = None
               ) -> str:
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
    backend = _resolveBackend(backend)

    if backend == 'qt':
        from . import _dialogsqt
        return _dialogsqt.selectFile(directory=directory, filter=filter, title=title)

    from ttkthemes import ThemedTk
    from tkinter import filedialog

    filetypes = _tkParseFilter(filter)
    if not filetypes:
        filetypes = [('All', '(*.*)')]

    root = ThemedTk(theme='breeze')
    root.withdraw()
    path = filedialog.askopenfilename(initialdir=directory, title=title,
                                      filetypes=filetypes)
    root.destroy()
    return path


def _tkParseFilter(filter: str) -> List[Tuple[str, str]]:
    # A filter has the form <name1> (<wildcard1>, <wildcard2>, ...);; name2...
    parts = filter.split(";;")
    out = []
    for part in parts:
        part = part.strip()
        if "(" in part and part[-1] == ")":
            # <name> (wildcards)
            name, wildcardstr = part[:-1].split("(")
            wildcardstr.replace(',', ' ')
            wildcards = wildcardstr.split()
            out.append((name.strip(), ' '.join(wildcards)))
        else:
            # <wildcard>
            out.append(('', part))
    return out


@emlib.misc.runonce
def _tkOk() -> bool:
    try:
        from ttkthemes import ThemedTk
        from tkinter import filedialog
        root = ThemedTk(theme='breeze')
        return True
    except:
        return False


def _saveDialogTk(filter="All (*.*)", title="Save file", directory: str = "~") -> str:
    from ttkthemes import ThemedTk
    from tkinter import filedialog
    root = ThemedTk(theme='breeze')
    filetypes = _tkParseFilter(filter)
    root.withdraw()

    path = filedialog.asksaveasfilename(initialdir=directory, title=title, filetypes=filetypes)
    root.destroy()
    return path


def saveDialog(filter="All (*.*)", title="Save file", directory: str = "~", backend: str = None
               ) -> str:
    """
    Open a dialog to save a file.

    .. note::

        At the moment macos only supports the 'qt' backend

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
    backend = _resolveBackend(backend)
    if not directory:
        directory = "~"
    directory = os.path.expanduser(directory)
    if backend == 'qt':
        from . import _dialogsqt
        return _dialogsqt.saveDialog(filter=filter, title=title, directory=directory)
    else:
        if sys.platform == 'darwin':
            raise RuntimeError("tk backend not supported in macos")
        return _saveDialogTk(filter=filter, title=title, directory=directory)


def selectItem(items: Sequence[str], title="Select", entryFont=('Arial', 15),
               listFont=('Arial', 12), scrollbar=True, width=400, numlines=20,
               caseSensitive=False, ensureSelection=False,
               backend: str = None
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
    selected = selectItems(items=items, title=title, entryFont=entryFont,
                           listFont=listFont, scrollbar=scrollbar,
                           width=width, numlines=numlines,
                           caseSensitive=caseSensitive,
                           ensureSelection=ensureSelection,
                           backend=backend)
    return selected[0] if selected else None


def selectItems(items: Sequence[str], title="Select", entryFont=('Arial', 14),
                listFont=('Arial', 12), scrollbar=True, width=400, numlines=20,
                caseSensitive=False, ensureSelection=False,
                backend: str = None
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
        backend: one of 'qt', 'tk' or None to use a default

    Returns:
        a list of selected items, or an empty list if the user aborted
        (via Escape or closing the window)
    """
    backend = _resolveBackend(backend)
    if backend == 'tk':
        return _selectFromListTk(items=items, title=title, entryFont=entryFont,
                                 listFont=listFont, scrollbar=scrollbar, width=width,
                                 numlines=numlines, caseSensitive=caseSensitive,
                                 ensureSelection=ensureSelection)
    elif backend == 'qt':
        if not _has_qt():
            raise RuntimeError("pyqt5 not installed. Install it via 'pip install pyqt5'")
        logger.info("Multiple item selection is not supported in qt at the moment")
        from . import _dialogsqt
        out = _dialogsqt.selectItem(items=items, title=title, listFont=listFont,
                                    entryFont=entryFont)
        return [out]
    else:
        raise ValueError("Backends supported: 'qt', 'tk'")


def _selectFromListTk(items: Sequence[str], title="Select", entryFont=('Arial', 15),
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
    if sys.platform == 'darwin':
        logger.error("macOS is not supported")
    import tkinter as tk
    from tkinter import ttk
    from ttkthemes import ThemedTk
    root = ThemedTk(theme="breeze")

    if len(items) < numlines:
        scrollbar = False
    numlines = min(numlines, len(items))
    root.title(title)
    root.columnconfigure(0, weight=1)

    longest = max((item for item in items), key=len)
    tkfont = tk.font.Font(font=font)
    minwidth = int(tkfont.measure(text) * correctionFactor)

    width = max(width, minwidth)

    filterval = tk.StringVar()
    entry = ttk.Entry(root, textvariable=filterval, font=entryFont)
    entry.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    treestyle = ttk.Style()
    treestyle.configure("selectFromList.Treeview", highlightthickness=0, bd=0,
                        font=listFont)
    treestyle.layout("selectFromList.Treeview", [
        ('selectFromList.Treeview.treearea', {'sticky': 'nswe'})])
    tree = ttk.Treeview(root, height=numlines, show='tree', style='selectFromList.Treeview'
                        )
    tree.grid(row=1, column=0, sticky='nsew')
    tree.column("#0", minwidth=0, width=width, stretch=False)

    # adding data to the treeview
    itemids = [tree.insert('', tk.END, text=item, open=False)
               for item in items]

    id2item = {i: c for i, c in zip(itemids, items)}

    # add a scrollbar

    if scrollbar:
        scrollbarWidget = ttk.Scrollbar(root, orient=tk.VERTICAL, command=tree.yview)
        scrollbarWidget.grid(row=1, column=1, sticky='ns')
        tree.configure(yscroll=scrollbarWidget.set)
    else:
        tree.configure(yscroll=None)

    id2visible = {i: True for i in itemids}

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

    root.bind("<Escape>", lambda *args: root.destroy())
    root.bind("<Return>", accept)
    entry.bind("<KeyRelease>",
               lambda *args: applyfilter(filterval.get(), caseSensitive=caseSensitive))
    entry.bind("<Down>", lambda *args: entrymove(1))
    entry.bind("<Up>", lambda *args: entrymove(-1))
    for k in "abcdefghijklmnopqrstuvwxyzABCEFGHIJKLMNOPQRSTUVWXYZ0123456789":
        tree.bind(k, entrykey)
    tree.bind("<BackSpace>", lambda *args: entryback())
    tree.bind('<Double-Button-1>', accept)

    tree.focus(itemids[0])
    tree.selection_set(itemids[0])
    entry.focus_set()

    root.mainloop()
    sel = out[0]
    if not sel and ensureSelection:
        raise ValueError("No selection was done")
    return sel if sel else []
