import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from advanced_editor.launcher import (
    AdvancedEditorWindow,
    AutoHideStatusLabel,
    ModernComboBox,
    ResizableTextEdit,
    _decoded_font_metadata_name,
    _font_descriptor,
    _split_font_filename,
)


class ResizableTextEditTest(TestCase):
    def test_paint_stroke_schedules_exact_text_preview(self) -> None:
        scheduled = []
        window = SimpleNamespace(
            _push_history=lambda *_args: None,
            _apply_paint_stroke=lambda *_args: None,
            save_current_session=lambda **_kwargs: None,
            reload_display_image=lambda: None,
            schedule_exact_preview=lambda: scheduled.append(True),
        )

        AdvancedEditorWindow.apply_stroke(window, "paint_brush", [(1, 1)])

        self.assertEqual(scheduled, [True])

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

    def test_searchable_font_combo_filters_inside_a_taller_popup(self) -> None:
        app = QApplication.instance() or QApplication([])
        combo = ModernComboBox(searchable=True)
        combo.addItems(["감탄로드", "마포꽃섬", "Typo Crayon"])
        completer = combo.completer()
        completer.setCompletionPrefix("crayon")
        app.processEvents()

        self.assertTrue(combo.isEditable())
        self.assertTrue(combo.lineEdit().isReadOnly())
        self.assertEqual(combo.maxVisibleItems(), 18)
        self.assertEqual(completer.completionModel().rowCount(), 1)
        self.assertGreaterEqual(completer.popup().minimumWidth(), 320)
        self.assertEqual(_split_font_filename("Typo_CrayonB"), ("Typo_Crayon", "Bold"))
        self.assertEqual(_split_font_filename("HY피오피B"), ("HY피오피", "Bold"))
        self.assertEqual(_split_font_filename("HY피오피L"), ("HY피오피", "Light"))
        self.assertEqual(_split_font_filename("HY피오피M"), ("HY피오피", "Medium"))
        self.assertEqual(
            _decoded_font_metadata_name("\u00c5\u00c2-\u00c0\u00b8\u00b6\u00e4B"),
            "태-으뜸B",
        )
        QTest.mouseClick(combo.lineEdit(), Qt.MouseButton.LeftButton)
        app.processEvents()
        self.assertTrue(combo._popup_open)
        combo.hidePopup()
        combo._search_action.trigger()
        self.assertFalse(combo.lineEdit().isReadOnly())
        combo.close()

    def test_korean_filename_wins_over_internal_ascii_alias(self) -> None:
        with patch("advanced_editor.launcher.TTFont"), patch(
            "advanced_editor.launcher._font_name_record",
            side_effect=["Macb", "Regular"],
        ):
            descriptor = _font_descriptor(
                Path("미드아카시아C_Bold.ttf"),
                Path("."),
            )
        self.assertEqual(descriptor["family"], "미드아카시아C")
        self.assertEqual(descriptor["style"], "Bold")

        with patch("advanced_editor.launcher.TTFont"), patch(
            "advanced_editor.launcher._font_name_record",
            side_effect=["HY피오피L", "Regular"],
        ):
            descriptor = _font_descriptor(Path("HY피오피L.TTF"), Path("."))
        self.assertEqual(descriptor["family"], "HY피오피")
        self.assertEqual(descriptor["style"], "Light")

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

        start = QPoint(viewport.width() - 5, viewport.height() - 5)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(viewport, QPoint(start.x(), start.y() + 500))
        QTest.mouseRelease(
            viewport,
            Qt.MouseButton.LeftButton,
            pos=QPoint(start.x(), start.y() + 500),
        )
        app.processEvents()
        self.assertEqual(edit.height(), ResizableTextEdit._MAX_HEIGHT)
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
