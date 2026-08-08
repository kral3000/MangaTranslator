import os
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from advanced_editor.launcher import (
    AutoHideStatusLabel,
    ModernComboBox,
    ResizableTextEdit,
)


class ResizableTextEditTest(TestCase):
    def test_modern_combo_paints_its_own_expansion_surface(self) -> None:
        app = QApplication.instance() or QApplication([])
        combo = ModernComboBox()
        combo.addItems(["첫 번째", "두 번째"])
        combo.resize(180, 34)
        combo.show()
        app.processEvents()

        image = combo.grab().toImage()
        self.assertNotEqual(image.pixelColor(10, 17), image.pixelColor(166, 5))
        combo.showPopup()
        app.processEvents()
        self.assertTrue(combo._popup_open)
        combo.hidePopup()
        self.assertFalse(combo._popup_open)
        combo.close()

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

    def test_empty_status_does_not_leave_a_blank_panel(self) -> None:
        app = QApplication.instance() or QApplication([])
        label = AutoHideStatusLabel("")
        app.processEvents()
        label.setText("저장했습니다.")
        self.assertFalse(label.isHidden())
        label.setText("")
        self.assertTrue(label.isHidden())
        label.close()
