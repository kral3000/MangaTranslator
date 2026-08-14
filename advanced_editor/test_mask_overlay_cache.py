import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter

from advanced_editor.launcher import EditorCanvas


class MaskOverlayCacheTest(TestCase):
    def test_fast_preview_replaces_the_stale_region_before_drawing_text(self) -> None:
        region = {"id": "r1", "enabled": True}
        clean_image = QImage(10, 10, QImage.Format.Format_RGBA8888)
        clean_image.fill(QColor("white"))
        target_image = QImage(10, 10, QImage.Format.Format_RGBA8888)
        target_image.fill(QColor("black"))
        window = SimpleNamespace(
            image_size=(10, 10),
            fast_preview_regions=Mock(return_value=[region]),
            fast_preview_clean_qimage=Mock(return_value=clean_image),
            _fast_erase_bbox=Mock(return_value=(2, 2, 6, 6)),
            draw_region_fast_text=Mock(),
        )
        canvas = SimpleNamespace(
            window=window,
            offset=QPointF(0, 0),
            scale=1.0,
            _region_rect_view=Mock(return_value=QRectF(2, 2, 4, 4)),
        )

        painter = QPainter(target_image)
        EditorCanvas._draw_fast_text_preview(canvas, painter)
        painter.end()

        self.assertEqual(target_image.pixelColor(3, 3), QColor("white"))
        self.assertEqual(target_image.pixelColor(0, 0), QColor("black"))
        window.draw_region_fast_text.assert_called_once()

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
