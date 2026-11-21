"""Shared work library entry point for standalone and in-editor launches."""

from PyQt6.QtCore import QByteArray, QSettings

from advanced_editor.storage_dialog import StorageDialog


class OpenEditorJobDialog(StorageDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout_settings = QSettings("MangaTranslator", "WorkLibrary")
        geometry = self.layout_settings.value("geometry")
        splitter = self.layout_settings.value("splitter")
        if isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        if isinstance(splitter, QByteArray):
            self.splitter.restoreState(splitter)

    def done(self, result):
        if self.isVisible():
            self.layout_settings.setValue("geometry", self.saveGeometry())
            self.layout_settings.setValue("splitter", self.splitter.saveState())
            self.layout_settings.sync()
        super().done(result)

    @property
    def session_path(self):
        return self.open_path
