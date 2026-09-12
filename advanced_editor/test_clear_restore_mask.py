from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from advanced_editor.launcher import AdvancedEditorWindow


class ClearRestoreMaskTest(TestCase):
    def test_clear_preserves_undo_data_and_refreshes_preview(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "restore.png"
            Image.new("L", (8, 6), 255).save(path)
            original = path.read_bytes()
            history = []
            window = SimpleNamespace(
                session={"restore_mask_path": str(path), "inpaint_mask_path": "keep.png"},
                session_path=Path(directory) / "session.json", image_size=(8, 6),
                _push_history=Mock(side_effect=lambda label: history.append(path.read_bytes())),
                status_label=Mock(), canvas=Mock(), reload_display_image=Mock(),
                schedule_exact_preview=Mock(),
            )
            with patch("advanced_editor.launcher.save_session") as save:
                AdvancedEditorWindow.clear_restore_mask(window)
                save.assert_called_once_with(window.session_path, window.session)
            self.assertEqual(history, [original])
            with Image.open(path) as image:
                self.assertIsNone(image.getbbox())
            self.assertEqual(window.session["inpaint_mask_path"], "keep.png")
            window.canvas.clear_mask_overlay_cache.assert_called_once()
            window.reload_display_image.assert_called_once()
            window.schedule_exact_preview.assert_called_once()
            window.session["restore_mask_path"] = ""
            AdvancedEditorWindow.clear_restore_mask(window)
            window._push_history.assert_called_once()
