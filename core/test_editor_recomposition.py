from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from core.editor_session import (
    _apply_region_patch,
    _region_render_signature,
    load_session,
    render_session_image,
    save_session,
)


class EditorRecompositionTest(TestCase):
    def test_patch_copy_preserves_all_channels_and_does_not_modify_inputs(self):
        rng = np.random.default_rng(42)
        pixels = rng.integers(0, 256, (4, 7, 4), dtype=np.uint8)
        mask = rng.integers(0, 2, (4, 7)).astype(bool)
        saved_pixels, saved_mask = pixels.copy(), mask.copy()
        box = (3, 2, 10, 6)
        for mode in ("RGB", "RGBA"):
            with self.subTest(mode=mode):
                source = Image.new(mode, (20, 12), (37, 91, 42))
                original = np.asarray(source).copy()
                expected = np.asarray(source.convert("RGBA")).copy()
                expected[2:6, 3:10][mask] = pixels[mask]
                errors = []
                with patch("core.editor_session._cached_region_patch",
                           return_value=((box, mask, pixels), [])):
                    actual = _apply_region_patch(source, source, "base", {}, 1, errors)
                self.assertFalse(errors)
                self.assertEqual(actual.mode, "RGBA")
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(source, original)
                np.testing.assert_array_equal(pixels, saved_pixels)
                np.testing.assert_array_equal(mask, saved_mask)

    def test_partial_neighbor_erase_does_not_leave_old_strokes_after_reload(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            clean = Image.new("RGBA", (320, 80), "white")
            clean.save(root / "base.png")
            snapshot = clean.copy()
            draw = ImageDraw.Draw(snapshot)
            draw.rectangle((30, 30, 40, 40), fill="black")
            # Saved glyph edges may differ from a fresh render (font/layout changes).
            draw.rectangle((65, 30, 115, 40), fill="black")
            draw.rectangle((150, 30, 180, 40), fill="black")
            draw.rectangle((280, 30, 290, 40), fill="blue")
            snapshot.save(root / "snapshot.png")
            regions = []
            for i, box in enumerate(
                (
                    [25, 25, 60, 45],
                    [60, 25, 120, 45],
                    [145, 25, 185, 45],
                    [270, 25, 300, 45],
                )
            ):
                region = dict(
                    id=f"r{i}",
                    enabled=True,
                    bbox=box,
                    translation="old",
                    render={"font_size": 5},
                )
                region["render_baseline"] = _region_render_signature(region)
                regions.append(region)
            session = dict(
                regions=regions,
                base_image_path=str(root / "base.png"),
                rendered_snapshot_path=str(root / "snapshot.png"),
            )

            def render(image, region, index, errors):
                if not region.get("enabled", True):
                    return image
                boxes = {
                    "r0": (30, 30, 40, 40),
                    "r1": (67, 30, 117, 40),
                    "r2": (152, 30, 182, 40),
                }
                ImageDraw.Draw(image).rectangle(boxes[region["id"]], fill="red")
                return image

            for action in ("edit", "disable", "delete", "paint"):
                with self.subTest(action=action):
                    current = deepcopy(session)
                    if action == "delete":
                        current["deleted_region_bboxes"] = [regions[0]["bbox"]]
                        current["regions"] = current["regions"][1:]
                    else:
                        current["regions"][0]["translation"] = "new"
                        current["regions"][0]["enabled"] = action != "disable"
                    expected = clean.copy()
                    if action == "paint":
                        overlay = Image.new("RGBA", clean.size)
                        ImageDraw.Draw(overlay).rectangle(
                            (85, 50, 90, 55), fill="green"
                        )
                        overlay.save(root / "paint.png")
                        current["paint_overlay_path"] = str(root / "paint.png")
                        expected.alpha_composite(overlay)
                    ImageDraw.Draw(expected).rectangle((280, 30, 290, 40), fill="blue")
                    for index, region in enumerate(current["regions"], 1):
                        if region["id"] != "r3":
                            render(expected, region, index, [])
                    save_session(root / "session.json", current)
                    with patch(
                        "core.editor_session._render_region_onto", side_effect=render
                    ):
                        first, errors = render_session_image(current)
                        reopened, reopen_errors = render_session_image(
                            load_session(root / "session.json")
                        )
                    self.assertFalse(errors + reopen_errors)
                    np.testing.assert_array_equal(first, expected)
                    np.testing.assert_array_equal(reopened, expected)
