import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QComboBox

from advanced_editor.launcher import (
    AdvancedEditorWindow,
    _font_descriptor,
    _font_style_sort_key,
    _split_font_filename,
)
from core.text.font_manager import find_font_variants


class YoonFontVariantsTest(TestCase):
    def test_family_defaults_and_explicit_choices(self):
        app = QApplication.instance() or QApplication([])
        fonts_root = Path(__file__).resolve().parents[1] / "fonts"
        for family, default in (("윤고딕", "330"), ("윤명조", "340")):
            with self.subTest(family=family):
                paths = sorted((fonts_root / family).glob("*"))
                paths = [path for path in paths if path.suffix.lower() == ".ttf"]
                if not paths:
                    self.skipTest("Local Yoon font assets are not installed")
                items = [_font_descriptor(path, fonts_root) for path in paths]
                items.sort(key=lambda item: _font_style_sort_key(item["style"]))
                self.assertEqual({item["family"] for item in items}, {family})
                self.assertEqual([item["style"] for item in items], sorted(path.stem[-3:] for path in paths))
                self.assertEqual(_split_font_filename(family + "340"), (family, "340"))
                self.assertEqual(AdvancedEditorWindow._preferred_font_file(paths).stem, family + default)
                variants = find_font_variants(str(fonts_root / family))
                self.assertEqual(variants["regular"].stem, family + default)
                self.assertEqual(variants["bold"].stem, family + "360")
                for path in paths:
                    self.assertEqual(find_font_variants(str(path))["regular"], path.resolve())
                window = SimpleNamespace(
                    _font_catalog={family: {"items": items}},
                    font_style_combo=QComboBox(),
                    _font_combo_target=lambda value: str(Path(value).resolve()),
                    _update_font_preview=lambda: None,
                    _font_files_under=AdvancedEditorWindow._font_files_under,
                )
                selected = AdvancedEditorWindow._populate_font_style_combo(window, family)
                self.assertEqual(Path(selected).stem, family + default)
                self.assertEqual(window.font_style_combo.currentText(), default)
                explicit_path = next(path for path in paths if path.stem.endswith("350"))
                selected = AdvancedEditorWindow._populate_font_style_combo(window, family, explicit_path)
                self.assertEqual(Path(selected).stem, family + "350")
                rendered = AdvancedEditorWindow._font_file_for_render(window, {"font_dir": str(fonts_root / family)})
                self.assertEqual(rendered.stem, family + default)
                window.font_style_combo.deleteLater()
        app.processEvents()
