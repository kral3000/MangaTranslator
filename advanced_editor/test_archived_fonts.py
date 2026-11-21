import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
from PyQt6.QtWidgets import QApplication, QComboBox

from advanced_editor import launcher
from core import editor_session as editor
from core.text.font_manager import find_font_variants
from ui.utils import get_available_font_packs


class ArchivedFontsTest(TestCase):
    def test_moved_pack_keeps_rendering_but_stays_out_of_picker(self):
        source = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf"
        if not source.is_file():
            self.skipTest("Arial fixture is not installed")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fonts = root / "fonts"
            active = fonts / "OldPack"
            backup = fonts / "_BACKUP"
            archived = backup / "retired" / active.name
            active.mkdir(parents=True)
            archived.parent.mkdir(parents=True)
            original = active / "Old-Regular.ttf"
            shutil.copyfile(source, original)
            region = {
                "bbox": [0, 0, 240, 100],
                "translation": "Archived text!",
                "render": {"font_dir": str(active), "font_size": 26},
            }
            canvas = Image.new("RGBA", (240, 100), "white")
            errors = []
            with (
                patch.object(editor, "_repo_root", return_value=root),
                patch.object(launcher, "__file__", str(root / "advanced_editor/launcher.py")),
            ):
                before = editor._render_region_onto(canvas.copy(), region, 1, errors)
                before_key = editor._region_patch_cache_key("base", region)
                active.rename(archived)
                after = editor._render_region_onto(canvas.copy(), region, 1, errors)
                self.assertEqual(errors, [])
                self.assertNotEqual(before.tobytes(), canvas.tobytes())
                self.assertEqual(before.tobytes(), after.tobytes())
                self.assertNotEqual(editor._region_patch_cache_key("base", region), before_key)
                self.assertEqual(region["render"]["font_dir"], str(active))
                self.assertEqual(editor.resolve_editor_font_path(original), str(archived / original.name))
                self.assertEqual(find_font_variants(str(archived))["regular"], archived / original.name)
                self.assertEqual(get_available_font_packs(fonts)[0], [])
                self.assertIsNone(editor.resolve_tag_font_path("BN", {"BN": {"font_dir": str(active)}}))

                window = SimpleNamespace(
                    _font_files_under=launcher.AdvancedEditorWindow._font_files_under,
                    _default_font_choice=lambda: "",
                    _update_font_preview=lambda path: preview.append(path),
                    font_family_combo=QComboBox(),
                    font_style_combo=QComboBox(),
                )
                preview = []
                for method in ("_font_file_for_render", "_font_combo_target", "_rebuild_font_catalog", "_select_font_target"):
                    setattr(window, method, MethodType(getattr(launcher.AdvancedEditorWindow, method), window))
                window._rebuild_font_catalog(active)
                self.assertEqual(window._font_catalog, {})
                self.assertEqual(window._select_font_target(active), str(archived / original.name))
                self.assertEqual(preview, [str(archived / original.name)])
                self.assertEqual(window.font_style_combo.count(), 0)
        app.processEvents()

    def test_exact_archive_path_wins_and_ambiguous_names_are_not_guessed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            fonts = root / "fonts"
            relative = Path("Family") / "Regular.ttf"
            backup = fonts / "_BACKUP"
            active = fonts / relative
            exact = backup / relative
            for path in (exact, backup / "Other" / relative, active):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with patch.object(editor, "_repo_root", return_value=root):
                self.assertEqual(editor.resolve_editor_font_path(active), str(active))
                active.unlink()
                self.assertEqual(editor.resolve_editor_font_path(active), str(exact))
                exact.rename(backup / "Regular.ttf")
                self.assertEqual(editor.resolve_editor_font_path(active), str(active))
