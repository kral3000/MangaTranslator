from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image

from core.config import RenderingConfig
from core.text.text_renderer import render_text_skia


class EllipsisGlyphTest(TestCase):
    def test_repeated_periods_and_ellipsis_use_font_outlines(self):
        # Rectangular dots make it possible to detect a return to uniform circles.
        builder = FontBuilder(1000, isTTF=True)
        glyph_order = [".notdef", "period", "ellipsis"]
        builder.setupGlyphOrder(glyph_order)
        builder.setupCharacterMap({ord("."): "period", ord("…"): "ellipsis"})
        glyphs = {}
        for name, count in zip(glyph_order, (0, 1, 3)):
            pen = TTGlyphPen(None)
            for i in range(count):
                x = i * 340
                pen.moveTo((x, 0))
                pen.lineTo((x, 70))
                pen.lineTo((x + 200, 70))
                pen.lineTo((x + 200, 0))
                pen.closePath()
            glyphs[name] = pen.glyph()
        builder.setupGlyf(glyphs)
        builder.setupHorizontalMetrics(
            {".notdef": (400, 0), "period": (340, 0), "ellipsis": (1020, 0)}
        )
        builder.setupHorizontalHeader(ascent=800, descent=-200)
        builder.setupNameTable({"familyName": "Dot Probe", "styleName": "Regular"})
        builder.setupOS2(
            sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200
        )
        builder.setupPost()

        with TemporaryDirectory() as directory:
            path = Path(directory) / "DotProbe-Regular.ttf"
            builder.save(path)
            config = RenderingConfig(
                min_font_size=50,
                max_font_size=50,
                supersampling_factor=1,
                tight_fit_shrink_enabled=False,
            )
            for vertical in (False, True):
                for text, count in (("..", 2), ("...", 3), ("…", 3), (".....", 5)):
                    with self.subTest(text=text, vertical=vertical):
                        result = render_text_skia(
                            Image.new("RGB", (300, 300), "white"),
                            text,
                            (10, 10, 290, 290),
                            str(path),
                            config=config,
                            vertical_stack=vertical,
                            text_color_rgb=(0, 0, 0),
                            raise_on_safe_error=True,
                        )
                        ink = (np.asarray(result.convert("L")) < 128).astype(np.uint8)
                        labels, _, stats, _ = cv2.connectedComponentsWithStats(ink)
                        self.assertEqual(labels - 1, count)
                        for _, _, width, height, _ in stats[1:]:
                            along, across = (height, width) if vertical else (width, height)
                            self.assertGreater(along, 2 * across)
