import os
from types import MethodType
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow

from advanced_editor.launcher import AdvancedEditorWindow


class SaveToastTest(TestCase):
    def test_save_toast_names_the_file_and_starts_auto_hide_timer(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = QMainWindow()
        window.resize(900, 600)
        window._save_toast_timer = QTimer(window)
        window._save_toast_timer.setSingleShot(True)
        window._save_toast_timer.setInterval(2400)
        window._position_save_toast = MethodType(
            AdvancedEditorWindow._position_save_toast, window
        )
        window.show()

        AdvancedEditorWindow._show_save_toast(window, "20_translated.png 저장 완료")
        app.processEvents()

        self.assertTrue(window.save_toast.isVisible())
        self.assertEqual(window.save_toast_text.text(), "20_translated.png 저장 완료")
        self.assertTrue(window._save_toast_timer.isActive())
        self.assertEqual(window.save_toast.x(), window.width() - window.save_toast.width() - 18)
        window.close()
