from __future__ import annotations
import sys
import logging
import emlib.misc
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import *


logger = logging.getLogger(__name__)

__all__ = (
    'FilteredList',
    'selectItem',
    'selectFile',
    'saveDialog',
    'showInfo'
)


def _makeApp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        logger.debug("Making new qt QApplication")
        app = QtWidgets.QApplication([])
    else:
        logger.debug("Using existing QApplication instance")
    return app


class _FilterEdit(QtWidgets.QLineEdit):
    def __init__(self, parent: FilteredList, font:Tuple[str, int]=None):
        self.parent = parent
        super().__init__()
        if font:
            self.setFont(QtGui.QFont(*font))

    def keyPressEvent(self, keyEvent):
        k = keyEvent.key()
        if k == Qt.Key_Return:
            self.parent.accept()
        elif k in (Qt.Key_Up, Qt.Key_Down):
            self.parent.view.keyPressEvent(keyEvent)
        elif k == Qt.Key_Escape:
            self.parent.dismiss()
        else:
            # re = QRegularExpression(pattern, QRegularExpression.CaseInsensitiveOption | QRegularExpression.DotMatchesEverythingOption)
            rx = QtCore.QRegularExpression(self.text(), QtCore.QRegularExpression.CaseInsensitiveOption | QtCore.QRegularExpression.DotMatchesEverythingOption)
            self.parent.proxy_model.setFilterRegularExpression(rx)
            self.parent.proxy_model.setFilterWildcard("*" + self.text() + "*")
            super().keyPressEvent(keyEvent)


class _FilteredListView(QtWidgets.QListView):
    def __init__(self, parent: FilteredList, font:Tuple[str, int]=None):
        self.parent = parent
        super().__init__()
        if font:
            self.setFont(QtGui.QFont(*font))

    def keyPressEvent(self, keyEvent):
        k = keyEvent.key()
        if k == Qt.Key_Return:
            self.parent.accept()
        elif k == Qt.Key_Escape:
            self.parent.dismiss()
        else:
            super().keyPressEvent(keyEvent)


def _calculateTextWidth(s: str, fontfamily: str, size: int) -> float:
    f = QtGui.QFont(fontfamily, size)
    fm = QtGui.QFontMetrics(f)
    width = fm.horizontalAdvance(s)
    return width


class FilteredList(QtWidgets.QMainWindow):
    def __init__(self, items:Sequence[str], title:str,
                 listFont:Tuple[str, int]=None,
                 entryFont:Tuple[str, int]=None):
        super().__init__()
        self.out = None
        self.view = _FilteredListView(self, font=listFont)  # QListView()
        self.model = QtCore.QStringListModel(items)
        self.proxy_model = QtCore.QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setSortCaseSensitivity(Qt.CaseInsensitive)
        self.view.setModel(self.proxy_model)
        self.setWindowTitle(title)
        self.searchbar = _FilterEdit(self, font=entryFont)  # QLineEdit()
        # choose the type of search by connecting to a different slot here.
        # see https://doc.qt.io/qt-5/qsortfilterproxymodel.html#public-slots
        self.searchbar.textChanged.connect(self.proxy_model.setFilterFixedString)
        minw = _calculateTextWidth(max(items, key=lambda s:len(s)), listFont[0], listFont[1])
        self.setMinimumWidth(int(minw + 60))

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.searchbar)
        layout.addWidget(self.view)

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def accept(self):
        idx = self.view.currentIndex()
        selection = idx.data()
        if selection is None and self.proxy_model.rowCount():
            selection = self.proxy_model.index(0, 0).data()
        self.out = selection
        self.close()

    def dismiss(self):
        self.out = None
        self.close()


def selectItem(items: Sequence[str], title='Select',
               listFont:Tuple[str, int]=None,
               entryFont: Tuple[str, int]=None
               ) -> Optional[str]:
    app = _makeApp()
    w = FilteredList(items, title=title, listFont=listFont, entryFont=entryFont)
    w.show()
    app.exec_()
    return w.out


class _FilteredComboBox(QtWidgets.QComboBox):
    def __init__(self, parent=None, title:str=''):
        super().__init__(parent)
        self.dismissed = False
        if title:
            self.setWindowTitle(title)

        self.setFocusPolicy(Qt.StrongFocus)
        self.setEditable(True)

        # add a filter model to filter matching items
        self.pFilterModel = QtCore.QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QtCore.QCompleter(self.pFilterModel, self)
        # always show all (filtered) completions
        self.completer.setCompletionMode(QtCore.QCompleter.UnfilteredPopupCompletion)
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
        super().setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super().setModelColumn(column)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.dismissed = True
            self.close()
        elif e.key() == Qt.Key_Enter or e.key() == Qt.Key_Return:
            self.close()
        else:
            super().keyPressEvent(e)


def selectFromCombobox(items: List[str], title="Select", width=300, height=40
                       ) -> Optional[str]:
    app = _makeApp()
    combo = _FilteredComboBox(title=title)
    # either fill the standard model of the combobox
    # combo.addItems(options)

    # or use another model
    combo.setModel(QtCore.QStringListModel(items))

    combo.resize(width, height)
    combo.show()
    app.exec_()
    return None if combo.dismissed else combo.currentText()


def selectFile(directory:str=None, filter="All (*.*)", title="Open file") -> str:
    app = _makeApp()
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    filter = filter.replace(",", " ")
    name, mask = QtWidgets.QFileDialog.getOpenFileName(None, title, directory=directory,
                                                       filter=filter)
    return name


def saveDialog(filter="All (*.*)", title="Save file", directory:str=None) -> str:
    app = _makeApp()
    filter = filter.replace(",", " ")
    name, mask = QtWidgets.QFileDialog.getSaveFileName(None, title, filter=filter,
                                                       directory=directory)
    return name


def showInfo(msg:str, title:str='Info', font:Tuple[str,int]=None, icon:str=None) -> None:
    """
    Open a message box with a text

    Args:
        msg: the text to display (one line)
        title: the title of the dialog
        font: if given, a tuple (fontfamily, size)
        icon: either None or one of 'question', 'information', 'warning', 'critical'
    """
    app = _makeApp()
    mbox = QtWidgets.QMessageBox()
    mbox.setText(msg)
    mbox.setBaseSize(QtCore.QSize(600, 120));
    if title:
        mbox.setWindowTitle(title)
    if font:
        mbox.setFont(QtCore.QFont(*font))
    if icon:
        if icon == 'question':
            mbox.setIcon(QtWidgets.QMessageBox.Question)
        elif icon == 'information':
            mbox.setIcon(QtWidgets.QMessageBox.Information)
        elif icon == 'warning':
            mbox.setIcon(QtWidgets.QMessageBox.Warning)
        elif icon == 'critical':
            mbox.setIcon(QtWidgets.QMessageBox.Critical)
    mbox.exec_()


# init
if sys.platform == 'darwin' and emlib.misc.inside_ipython() and not emlib.misc.inside_jupyter():
    # macos needs loop integration inside ipython
    ip = get_ipython()
    if ip.active_eventloop is None:
        logger.debug("Starting ipython/qt eventloop integration")
        ip.run_line_magic('gui', 'qt')
    elif ip.active_eventloop not in ('qt', 'qt5'):
        logger.warning(
                f"IPython has an active eventloop for {ip.active_eventloop}, but "
                f"emlib.dialogs needs the qt eventloop to be able to open qt dialogs"
                f" without blocking the shell. ")
        ip.run_line_magic('gui', 'qt')

