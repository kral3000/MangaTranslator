"""Run: runtime/python.exe -m unittest advanced_editor.test_font_picker"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QComboBox

from advanced_editor.font_picker import FontPicker, TagManager, FilterManager, split_label, tag_ink


class FontPickerTest(TestCase):
    def test_fixed_dialogue_filter_excludes_tags_and_reset_shows_all(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            combo = QComboBox()
            picker = FontPicker(combo, Path(directory) / "tags.ini")
            picker.tag_definitions.append({"name": "특수", "color": "#e8edf3", "group": ""})
            picker.rebuild_tags()
            for i, tags in enumerate(([], ["일반대사"], ["특수"], ["강조"], ["효과음"], ["손글씨"], ["일반대사", "강조"])):
                combo.addItem(f"폰트{i}", str(i))
                picker.settings.setValue("tags/" + str(i), tags)
            picker.open()
            self.assertEqual(picker.group, "대사")
            self.assertEqual(picker.count_label.text(), "2개")
            self.assertEqual(picker.filter_layout.itemAt(0).widget().text(), "대사")
            self.assertEqual(picker.all_button.text(), "")
            picker.search.setText("없는 폰트")
            picker.all_button.click()
            self.assertEqual(picker.count_label.text(), "7개")
            self.assertEqual(picker.search.text(), "")
            picker.close()

    def test_custom_filters_save_reload_and_follow_tag_rename(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tags.ini"
            combo = QComboBox()
            combo.addItem("돌담", "font0")
            picker = FontPicker(combo, path)
            picker.settings.setValue("tags/font0", ["울먹"])
            picker.open()
            picker.hide()
            manager = FilterManager(picker)
            manager.entries.clear()
            manager.add_filter({"name": "감정", "tags": ["울먹", "강조"]})
            manager.add_filter({"name": "빈 필터", "tags": []})
            manager.save()
            picker.set_group("감정")
            self.assertEqual(picker.count_label.text(), "1개")
            picker.set_group("빈 필터")
            self.assertEqual(picker.count_label.text(), "0개")
            definitions = [dict(tag) for tag in picker.tag_definitions]
            rename = {tag["name"]: tag["name"] for tag in definitions}
            rename["울먹"] = "눈물"
            for tag in definitions:
                if tag["name"] == "울먹":
                    tag["name"] = "눈물"
            picker.replace_tags(definitions, rename)
            reloaded = FontPicker(combo, path)
            self.assertEqual(reloaded.filter_definitions, [{"name": "감정", "tags": ["눈물", "강조"]}, {"name": "빈 필터", "tags": []}])
            reloaded.open()
            reloaded.set_group("감정")
            self.assertEqual(reloaded.count_label.text(), "1개")
            reloaded.close()
            manager.close()
            picker.close()

    def test_tag_management_preserves_assignments_and_reloads(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tags.ini"
            combo = QComboBox()
            combo.addItem("학교안심 돌담 (3)", "font0")
            picker = FontPicker(combo, path)
            picker.settings.setValue("tags/font0", ["울먹", "남성"])
            picker.settings.setValue("tags/absent-font", ["울먹"])
            picker.open()
            picker.hide()
            manager = TagManager(picker)
            for row in range(manager.table.rowCount()):
                if manager.table.item(row, 0).text() == "울먹":
                    manager.table.item(row, 0).setText("눈물")
                    manager.set_color(manager.table.cellWidget(row, 1), "#223344")
            for row in range(manager.table.rowCount()):
                if manager.table.item(row, 0).text() == "남성":
                    manager.table.selectRow(row)
                    manager.remove_rows()
                    break
            manager.add_row()
            manager.table.item(manager.table.rowCount() - 1, 0).setText("속마음")
            manager.save()
            self.assertEqual(picker.settings.value("tags/font0", type=list), ["눈물"])
            self.assertEqual(picker.settings.value("tags/absent-font", type=list), ["눈물"])
            reloaded = FontPicker(combo, path)
            reloaded.open()
            self.assertIn("속마음", reloaded.checks)
            self.assertNotIn("남성", reloaded.checks)
            self.assertEqual(reloaded.list.property("tagColors")["눈물"], "#223344")
            reloaded.set_group("특수대사")
            self.assertEqual(reloaded.count_label.text(), "1개")
            reloaded.edit.setChecked(True)
            reloaded.checks["속마음"].setChecked(True)
            reloaded.set_group("전체")
            reloaded.tag_filter.setCurrentIndex(reloaded.tag_filter.findData("속마음"))
            self.assertEqual(reloaded.count_label.text(), "1개")
            self.assertEqual(tag_ink("#000000"), "#ffffff")
            self.assertEqual(tag_ink("#ffffff"), "#000000")
            reloaded.close()
            manager.close()
            picker.close()

    def test_grid_search_tags_selection_and_wheel(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory:
            combo = QComboBox()
            for i in range(90):
                combo.addItem(f"학교안심 돌담{i:02} (3)", f"font{i}")
            combo.show()
            picker = FontPicker(combo, Path(directory) / "tags.ini")
            picker.open()
            app.processEvents()
            a = picker.list.visualItemRect(picker.list.item(0))
            b = picker.list.visualItemRect(picker.list.item(1))
            c = picker.list.visualItemRect(picker.list.item(3))
            self.assertEqual(a.y(), b.y())
            self.assertGreater(b.x(), a.x())
            self.assertGreater(c.y(), a.y())
            self.assertEqual(picker.list.horizontalScrollBar().maximum(), 0)
            picker.resize(540, 420)
            app.processEvents()
            self.assertEqual(picker.list.visualItemRect(picker.list.item(0)).y(), picker.list.visualItemRect(picker.list.item(1)).y())
            self.assertGreater(picker.list.visualItemRect(picker.list.item(2)).y(), picker.list.visualItemRect(picker.list.item(0)).y())
            for surface in (picker.list.viewport(), picker.search, picker.edit):
                wheel = QWheelEvent(QPointF(20, 20), QPointF(20, 20), QPoint(), QPoint(0, -120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
                before = picker.list.verticalScrollBar().value()
                app.sendEvent(surface, wheel)
                self.assertGreater(picker.list.verticalScrollBar().value(), before)
            picker.search.setText("돌담07")
            self.assertEqual(picker.count_label.text(), "1개")
            self.assertEqual(picker.list.currentItem().data(Qt.ItemDataRole.UserRole), "font7")
            picker.edit.setChecked(True)
            picker.checks["울먹"].setChecked(True)
            self.assertEqual(combo.currentIndex(), 0)
            picker.edit.setChecked(False)
            QTest.keyClick(picker.search, Qt.Key.Key_Return)
            self.assertEqual(combo.currentIndex(), 7)
            self.assertFalse(picker.isVisible())
            picker.open()
            picker.set_group("특수대사")
            self.assertEqual(picker.count_label.text(), "1개")
            self.assertEqual(picker.list.currentItem().data(Qt.ItemDataRole.UserRole + 1), ["울먹"])
            picker.search.setText("no matching font")
            self.assertTrue(picker.empty.isVisible())
            picker.close()
            combo.close()
            self.assertEqual(split_label("학교안심 돌담 (3)"), ("학교안심", "돌담 (3)"))
