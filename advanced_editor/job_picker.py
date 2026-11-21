"""Shared work library entry point for standalone and in-editor launches."""

from advanced_editor.storage_dialog import StorageDialog


class OpenEditorJobDialog(StorageDialog):
    @property
    def session_path(self):
        return self.open_path
