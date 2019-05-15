NORM_FONT = ("Helvetica", 10)


def _popupmsg_tk(msg, title="", buttontxt="Ok", font=NORM_FONT):
    import tkinter as tk
    from tkinter import ttk
    popup = tk.Tk()
    popup.wm_title(title)
    label = ttk.Label(popup, text=msg, font=font)
    label.pack(side="top", fill="x", pady=10, padx=16)
    B1 = ttk.Button(popup, text=buttontxt, command=popup.destroy)
    B1.pack(padx=5, pady=5)
    popup.mainloop()


def popupmsg(msg, title="", buttontxt="Ok"):
    return _popupmsg_tk(msg=msg, title=title, buttontxt=buttontxt, font=NORM_FONT)


def _showinfo_tk(msg, title=None):
    import tkinter as tk
    from tkinter import messagebox
    window = tk.Tk()
    window.wm_withdraw()
    messagebox.showinfo(title, msg)
    window.destroy()


def showinfo(msg, title=None):
    return _showinfo_tk(msg=msg, title=title)


def savedialog(filename='untitled', filetypes=["*"], title="Save file"):
    try:
        import toga
    except ImportError:
        raise ImportError("We depend on toga for this functionality. Install it via pip install toga")
    app = toga.App("foo", "foo.bar")
    w = toga.Window()
    return w.save_file_dialog(title=title, suggested_filename=filename, file_types=filetypes)
