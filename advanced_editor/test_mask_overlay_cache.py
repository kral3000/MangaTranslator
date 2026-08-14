import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image
from PyQt6.QtCore import QPointF

from advanced_editor.launcher import EditorCanvas


class MaskOverlayCacheTest(TestCase):
    def test_idle_pointer_does_not_repaint_the_canvas(self) -> None:
        canvas = SimpleNamespace(
            pan_drag_start=None,
            view_to_image=Mock(return_value=QPointF(1, 1)),
            hover_image_pos=None,
            drag_mode=None,
            drag_start_bbox=None,
            tool="select",
            textbox_start=None,
            stroke_points=[],
            update=Mock(),
        )
        event = SimpleNamespace(position=Mock(return_value=QPointF(1, 1)))

        EditorCanvas.mouseMoveEvent(canvas, event)
        canvas.update.assert_not_called()

        canvas.tool = "paint_brush"
        EditorCanvas.mouseMoveEvent(canvas, event)
        canvas.update.assert_called_once_with()

    def test_unchanged_mask_is_decoded_once_and_file_change_invalidates_cache(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "mask.png"
            Image.new("L", (8, 6), 255).save(path)
            canvas = SimpleNamespace(
                window=SimpleNamespace(image_size=(8, 6)),
                _mask_overlay_cache={},
            )
            color = [0, 190, 255, 105]

            first = EditorCanvas._mask_overlay_qimage(canvas, path, color)
            with patch(
                "advanced_editor.launcher._load_mask",
                side_effect=AssertionError("unchanged mask was decoded again"),
            ):
                second = EditorCanvas._mask_overlay_qimage(canvas, path, color)
            self.assertIs(first, second)

            Image.new("L", (8, 6), 0).save(path)
            updated = path.stat().st_mtime_ns + 1_000_000_000
            os.utime(path, ns=(updated, updated))
            self.assertIsNone(EditorCanvas._mask_overlay_qimage(canvas, path, color))
