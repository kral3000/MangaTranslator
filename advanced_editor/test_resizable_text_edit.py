import os
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from advanced_editor.launcher import ResizableTextEdit


class ResizableTextEditTest(TestCase):
    def test_dragging_the_corner_changes_the_height(self) -> None:
        app = QApplication.instance() or QApplication([])
        edit = ResizableTextEdit(68)
        edit.resize(300, 68)
        edit.show()
        app.processEvents()

        viewport = edit.viewport()
        start = QPoint(viewport.width() - 5, viewport.height() - 5)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, QPoint(start.x(), start.y() + 36))
        QTest.mouseRelease(
            viewport,
            Qt.MouseButton.LeftButton,
            pos=QPoint(start.x(), start.y() + 36),
        )
        app.processEvents()

        self.assertEqual(edit.height(), 104)
        edit.close()
