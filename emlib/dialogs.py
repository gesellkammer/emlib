"""
Simple dialogs for use at the repl
"""
from __future__ import annotations

_DEFAULT_FONT = ("Helvetica", 11)


def _popupmsg_tk(msg, title="", buttontxt="Ok", font=_DEFAULT_FONT):
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
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


def popupmsg(msg:str, title="", buttontxt="Ok") -> None:
    """
    Open a pop-up dialog with a message
    """
    return _popupmsg_tk(msg=msg, title=title, buttontxt=buttontxt, font=_DEFAULT_FONT)


def _showinfo_tk(msg:str, title:str=None) -> None:
    import tkinter as tk
    from tkinter import messagebox
    window = tk.Tk()
    window.wm_withdraw()
    messagebox.showinfo(title, msg)
    window.destroy()


def _showinfo_ttk(msg:str, title:str="Info", font=None) -> None:
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title(title)

    s = ttk.Style()
    bg = "#f5f5f5"
    ttk.Style().configure("TButton", padding=10, relief="flat",
                          background="#e0e0e0")
    s.configure('.showinfo.TFrame', background=bg)
    frame = ttk.Frame(root, style=".showinfo.TFrame")
    dx, dy = 8, 8
    if font is None:
        font = (_DEFAULT_FONT[0], int(_DEFAULT_FONT[1]*1.3))
    ttk.Label(frame, text="ℹ  " + msg, font=font, background=bg).grid(column=0, row=0, padx=dx*2,
                                                              pady=dy*2)
    ttk.Button(frame, text="Ok", command=root.destroy).grid(column=0, row=1, padx=dx,
                                                            pady=dy)
    frame.grid(column=0, row=0)
    root.bind("<Escape>", lambda *args: root.destroy())
    root.mainloop()


def showinfo(msg:str, title:str="Info") -> None:
    """
    Show a pop up dialog with some info
    """
    return _showinfo_ttk(msg=msg, title=title)


def opendialog(directory:str=None, filter="All (*.*)", title="Open file") -> str:
    """
    Create a dialog to open a file.

    Args:
        filter: a string of the form "<Mask> (<glob>)". Multiple filters can be
            used, for example: "Image (*.png, *.jpg);; Video (*.mp4, *.mov)"
        title: the title of the dialog
        directory: the initial directory

    Returns:
        the selected filename, or an empty string if the dialog is dismissed
    """
    return _opendialog_qt(directory=directory, filter=filter, title=title)


def _opendialog_qt(directory:str=None, filter="All (*.*)", title="Open file") -> str:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    name, mask = QtWidgets.QFileDialog.getOpenFileName(None, title, directory=directory, 
                                                       filter=filter)
    return name


def _savedialog_qt(filter="All (*.*)", title="Save file", directory:str=None) -> str:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    name, mask = QtWidgets.QFileDialog.getSaveFileName(None, title, filter=filter, 
                                                       directory=directory)
    return name


def savedialog(filter="All (*.*)", title="Save file", directory:str=None) -> str:
    """
    Open a dialog to save a file.

    Args:
        filter: a string of the form "<Mask> (<glob>)". Multiple filters can be
            used, for example: "Image (*.png, *.jpg);; Video (*.mp4, *.mov)"
        title: the title of the dialog
        directory: the initial directory
        
    Returns:
        the save filename, or an empty string if the dialog is dismissed
    """
    return _savedialog_qt(filename=filename, filter=filter, title=title, 
                          directory=directory)
