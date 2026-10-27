from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from advanced_editor.launcher import AdvancedEditorWindow
from core.editor_package import embed_editor_package, extract_embedded_editor_package
from core.editor_session import (
    load_session,
    render_session_background,
    render_session_image,
    save_session,
)


class TextCoverTest(TestCase):
    def test_cover_is_above_text_but_ordinary_restore_and_ai_context_stay_below(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, original = root / "base.png", root / "original.png"
            Image.new("RGBA", (12, 12), "blue").save(base)
            Image.new("RGBA", (12, 12), "red").save(original)
            cover, restore = root / "cover.png", root / "restore.png"
            for path, point in ((cover, (5, 5)), (restore, (6, 6))):
                mask = Image.new("L", (12, 12))
                mask.putpixel(point, 255)
                mask.save(path)
            session = dict(
                base_image_path=str(base),
                source_image_path=str(original),
                inpainted_image_path=str(base),
                inpainted_background_text_free=True,
                text_cover_mask_path=str(cover),
                restore_mask_path=str(restore),
                regions=[{"id": "r1", "translation": "text"}],
            )
            self.assertEqual(
                render_session_background(session).getpixel((5, 5)), (0, 0, 255, 255)
            )

            def text_patch(image, *_args, **_kwargs):
                return Image.new("RGBA", image.size, "black")

            with patch(
                "core.editor_session._apply_region_patch", side_effect=text_patch
            ):
                image, errors = render_session_image(session)
            self.assertFalse(errors)
            self.assertEqual(image.getpixel((5, 5)), (255, 0, 0, 255))
            self.assertEqual(image.getpixel((6, 6)), (0, 0, 0, 255))
            self.assertEqual(image.getpixel((7, 7)), (0, 0, 0, 255))

    def test_cover_stroke_erase_and_clear_leave_restore_mask_untouched(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (12, 12), "white").save(source)
            restore = root / "restore.png"
            Image.new("L", (12, 12), 80).save(restore)
            original = restore.read_bytes()
            window = SimpleNamespace(
                session={
                    "source_image_path": str(source),
                    "assets_dir": str(root),
                    "restore_mask_path": str(restore),
                },
                session_path=root / "session.json",
                image_size=(12, 12),
                brush_shape="circle",
                brush_size_for_tool=lambda _: 1,
                status_label=Mock(),
                canvas=Mock(),
                _push_history=Mock(),
                reload_display_image=Mock(),
                schedule_exact_preview=Mock(),
            )
            window._stroke_mask = lambda points, size: (
                AdvancedEditorWindow._stroke_mask(window, points, size)
            )
            AdvancedEditorWindow._apply_restore_mask_stroke(
                window, "text_cover_brush", [(5, 5)]
            )
            cover = Path(window.session["text_cover_mask_path"])
            with Image.open(cover) as image:
                self.assertEqual(np.count_nonzero(image), 1)
            state = AdvancedEditorWindow._capture_state(
                SimpleNamespace(session=window.session, selected_index=0)
            )
            self.assertIn(str(cover), state["masks"])
            AdvancedEditorWindow._apply_restore_mask_stroke(
                window, "text_cover_eraser", [(5, 5)]
            )
            with Image.open(cover) as image:
                self.assertIsNone(image.getbbox())
            AdvancedEditorWindow._apply_restore_mask_stroke(
                window, "text_cover_brush", [(5, 5)]
            )
            AdvancedEditorWindow.clear_restore_mask(window, text_cover=True)
            with Image.open(cover) as image:
                self.assertIsNone(image.getbbox())
            self.assertEqual(restore.read_bytes(), original)
            window._push_history.assert_called_once_with("글씨 덮기 마스크 비우기")

    def test_portable_export_preserves_cover_mask(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, cover = root / "source.png", root / "cover.png"
            Image.new("RGB", (12, 12), "white").save(source)
            Image.new("L", (12, 12), 255).save(cover)
            session = {
                "source_image_path": str(source),
                "base_image_path": str(source),
                "text_cover_mask_path": str(cover),
                "regions": [],
            }
            path = root / "session.json"
            save_session(path, session)
            exported = root / "export.png"
            Image.new("RGB", (12, 12), "white").save(exported)
            embed_editor_package(exported, path)
            moved = extract_embedded_editor_package(
                exported, destination_root=root / "moved"
            )
            restored = load_session(moved)
            self.assertNotEqual(Path(restored["text_cover_mask_path"]), cover)
            self.assertEqual(
                Path(restored["text_cover_mask_path"]).read_bytes(), cover.read_bytes()
            )
