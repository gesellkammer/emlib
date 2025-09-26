

from __future__ import annotations

import sys
from PyQt6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout,
                             QLineEdit, QListWidget, QPushButton, QListWidgetItem,
                             QLabel, QFileDialog, QMessageBox)
from PyQt6.QtCore import (Qt, QSize)
from PyQt6.QtGui import QFont
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence


__all__ = (
    'selectItem',
    'selectFile',
    'saveDialog',
    'showInfo'
)


def selectItem(items: Sequence[str],
               title='Select',
               listFont: tuple[str, int] = None,
               entryFont: tuple[str, int] = None
               ) -> str | None:
    """
    Display a searchable dialog for item selection.

    Args:
        items: Sequence of string items to choose from
        title: Window title for the dialog
        listFont: Tuple of (font_name, font_size) for the list widget
        entryFont: Tuple of (font_name, font_size) for the search entry field

    Returns:
        Selected item string, or None if canceled/escaped
    """
    class SearchableDialog(QDialog):
        def __init__(self, items: Sequence[str], title: str, listFont: tuple[str, int] = None, entryFont: tuple[str, int] = None):
            super().__init__()
            self.items = items
            self.selected_item = None
            self.listFont = listFont
            self.entryFont = entryFont
            self.setup(title)
            self.populate()

        def setup(self, title: str):
            self.setWindowTitle(title)
            self.setModal(True)
            self.resize(400, 500)

            # Main layout
            layout = QVBoxLayout()
            self.setLayout(layout)

            # Search field
            searchLabel = QLabel("Search:")
            self.searchField = QLineEdit()
            self.searchField.setPlaceholderText("Type to filter items...")
            self.searchField.textChanged.connect(self.filterItems)

            # Apply entry font if specified
            if self.entryFont:
                font = QFont(self.entryFont[0], self.entryFont[1])
                self.searchField.setFont(font)
                searchLabel.setFont(font)

            layout.addWidget(searchLabel)
            layout.addWidget(self.searchField)

            # List widget
            self.listWidget = QListWidget()
            self.listWidget.itemDoubleClicked.connect(self.accept_selection)

            # Apply list font if specified
            if self.listFont:
                font = QFont(self.listFont[0], self.listFont[1])
                self.listWidget.setFont(font)

            layout.addWidget(self.listWidget)

            # Buttons
            buttonLayout = QHBoxLayout()

            self.okButton = QPushButton("OK")
            self.okButton.clicked.connect(self.accept_selection)
            self.okButton.setEnabled(False)  # Disabled until selection is made

            cancelButton = QPushButton("Cancel")
            cancelButton.clicked.connect(self.reject)

            buttonLayout.addWidget(self.okButton)
            buttonLayout.addWidget(cancelButton)
            layout.addLayout(buttonLayout)

            # Connect selection change to enable/disable OK button
            self.listWidget.itemSelectionChanged.connect(self.on_selection_changed)

            # Set focus to search field
            self.searchField.setFocus()

        def populate(self):
            """Populate the list with all items"""
            self.listWidget.clear()
            for item in self.items:
                self.listWidget.addItem(QListWidgetItem(item))

        def filterItems(self):
            """Filter items based on search text"""
            search_text = self.searchField.text().lower()

            self.listWidget.clear()
            for item in self.items:
                if search_text in item.lower():
                    list_item = QListWidgetItem(item)
                    self.listWidget.addItem(list_item)

            # If there's exactly one item, select it
            if self.listWidget.count() == 1:
                self.listWidget.setCurrentRow(0)

        def on_selection_changed(self):
            """Enable/disable OK button based on selection"""
            hasSelection = len(self.listWidget.selectedItems()) > 0
            self.okButton.setEnabled(hasSelection)

        def accept_selection(self):
            """Accept the current selection"""
            selected = self.listWidget.selectedItems()
            if selected:
                self.selected_item = selected[0].text()
                self.accept()

        def keyPressEvent(self, event):
            """Handle key press events"""
            if event.key() == Qt.Key.Key_Escape:
                self.reject()
            elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                if self.okButton.isEnabled():
                    self.accept_selection()
            elif event.key() == Qt.Key.Key_Down:
                # If search field has focus and there are items, move to list
                if self.searchField.hasFocus() and self.listWidget.count() > 0:
                    self.listWidget.setFocus()
                    if self.listWidget.currentRow() < 0:
                        self.listWidget.setCurrentRow(0)
                else:
                    super().keyPressEvent(event)
            elif event.key() == Qt.Key.Key_Up:
                # If at top of list, move back to search field
                if (self.listWidget.hasFocus() and
                    self.listWidget.currentRow() == 0):
                    self.searchField.setFocus()
                else:
                    super().keyPressEvent(event)
            else:
                super().keyPressEvent(event)

    # Ensure we have a QApplication instance
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create and show dialog
    dialog = SearchableDialog(items, title, listFont, entryFont)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.selected_item
    else:
        return None


def selectFile(directory='', filter="All (*.*)", title="Open file") -> str:
    """
    Present a file selection dialog using PyQt6.

    Args:
        directory (str): Initial directory to open. Defaults to empty string (current directory).
        filter (str): File filter for the dialog. Defaults to "All (*.*)".
        title (str): Title of the dialog window. Defaults to "Open file".

    Returns:
        str: Path of the selected file, or empty string if canceled.
    """
    # Check if QApplication instance already exists
    app = QApplication.instance()
    appCreated = False
    if app is None:
        # Create a new QApplication if one doesn't exist
        app = QApplication(sys.argv)
        appCreated = True

    try:
        # Open the file dialog
        filePath, _ = QFileDialog.getOpenFileName(
            parent=None,
            caption=title,
            directory=directory,
            filter=filter)
        return filePath if filePath else ""
    finally:
        # Only quit the application if we created it
        if appCreated:
            app.quit()


def saveDialog(filter="All (*.*)", title="Save file", directory='') -> str:
    """
    Present a save file dialog using PyQt6.

    Args:
        filter (str): File filter string (e.g., "Text files (*.txt);;All files (*.*)")
        title (str): Dialog window title
        directory (str): Initial directory to open

    Returns:
        str: Path of the file to save, or empty string if aborted
    """
    # Create QApplication instance if one doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Open the save file dialog
    path, _ = QFileDialog.getSaveFileName(
        parent=None,
        caption=title,
        directory=directory,
        filter=filter)

    # Return the selected file path or empty string if cancelled
    return path or ""


def _makeApp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app  # type: ignore


def showInfo(msg:str, title:str='Info', font: tuple[str,int] = None, icon='') -> None:
    """
    Open a message box with a text

    Args:
        msg: the text to display (one line)
        title: the title of the dialog
        font: if given, a tuple (fontfamily, size)
        icon: either None or one of 'question', 'information', 'warning', 'critical'
    """
    if (app := QApplication.instance()) is None:
        app = QApplication([])
    mbox = QMessageBox()
    mbox.setText(msg)
    mbox.setBaseSize(QSize(600, 120))
    if title:
        mbox.setWindowTitle(title)
    if font:
        mbox.setFont(QFont(*font))
    if icon:
        if icon == 'question':
            mbox.setIcon(QMessageBox.Icon.Question)
        elif icon == 'information':
            mbox.setIcon(QMessageBox.Icon.Information)
        elif icon == 'warning':
            mbox.setIcon(QMessageBox.Icon.Warning)
        elif icon == 'critical':
            mbox.setIcon(QMessageBox.Icon.Critical)
    mbox.exec()
