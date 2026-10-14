from unittest import TestCase

import numpy as np
import skia
from PIL import Image

from core.text.drawing_engine import pil_to_skia_surface, skia_surface_to_pil


class SurfaceConversionTest(TestCase):
    def test_direct_copy_matches_image_drawing_before_and_after_transparent_drawing(self):
        rng = np.random.default_rng(32)
        pixels = rng.integers(0, 256, (73, 91, 4), dtype=np.uint8)
        # Cover every alpha value, including invisible pixels with nonzero RGB.
        pixels.reshape(-1, 4)[:256, 3] = np.arange(256, dtype=np.uint8)
        for mode in ("RGBA", "RGB", "L", "P"):
            with self.subTest(mode=mode):
                source = Image.fromarray(pixels).convert(mode)
                original_bytes = source.tobytes()
                rgba = source.convert("RGBA")
                expected = skia.Surface(*source.size)
                with expected as canvas:
                    canvas.drawImage(skia.Image.frombytes(
                        rgba.tobytes(), rgba.size, skia.kRGBA_8888_ColorType
                    ), 0, 0)
                actual = pil_to_skia_surface(source)
                for draw in (False, True):
                    if draw:
                        for surface in (expected, actual):
                            canvas = surface.getCanvas()
                            canvas.rotate(13)
                            canvas.drawRect(
                                skia.Rect.MakeXYWH(7.25, 11.5, 42.5, 28.5),
                                skia.Paint(Color=skia.ColorSetARGB(137, 91, 21, 181),
                                           AntiAlias=True),
                            )
                    np.testing.assert_array_equal(
                        skia_surface_to_pil(actual), skia_surface_to_pil(expected)
                    )
                self.assertEqual(source.tobytes(), original_bytes)
