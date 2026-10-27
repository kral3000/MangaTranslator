import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image, UnidentifiedImageError

from core import editor_session as editor


class EditorRenderCacheTest(TestCase):
    def setUp(self):
        editor._DECODED_IMAGE_CACHE.clear()
        editor._REGION_PATCH_CACHE.clear()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(editor._DECODED_IMAGE_CACHE.clear)
        self.addCleanup(editor._REGION_PATCH_CACHE.clear)
        self.root = Path(self.temp.name)

    def test_repeated_reads_decode_once_and_return_independent_images(self):
        path = self.root / "page.png"
        Image.new("RGBA", (8, 8), "red").save(path)
        with patch("core.editor_session.Image.open", wraps=Image.open) as opened:
            first = editor._read_editor_image(path)
            first.putpixel((0, 0), (0, 0, 0, 0))
            second = editor._read_editor_image(path)
            third = editor._read_editor_image(path)
        self.assertEqual(opened.call_count, 1)
        self.assertIsNot(second, third)
        self.assertEqual(second.getpixel((0, 0)), (255, 0, 0, 255))

    def test_overwrite_with_preserved_size_and_timestamp_invalidates_cache(self):
        path = self.root / "page.bmp"
        Image.new("RGB", (8, 8), "red").save(path)
        original_stat = path.stat()
        first_identity = editor._image_cache_identity(path, (8, 8))
        editor._read_editor_image(path)
        Image.new("RGB", (8, 8), "blue").save(path)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        self.assertEqual(path.stat().st_size, original_stat.st_size)
        self.assertEqual(path.stat().st_mtime_ns, original_stat.st_mtime_ns)
        self.assertEqual(editor._read_editor_image(path).getpixel((0, 0)), (0, 0, 255, 255))
        self.assertNotEqual(first_identity, editor._image_cache_identity(path, (8, 8)))

    def test_cache_is_bounded_and_mode_specific(self):
        paths = [self.root / f"{i}.png" for i in range(2)]
        for path in paths:
            Image.new("RGB", (4, 4), "red").save(path)
        # Give the second asset different content so it is a distinct cached image.
        Image.new("RGB", (4, 4), "blue").save(paths[1])
        with patch.object(editor, "_DECODED_IMAGE_CACHE_MAX_BYTES", 64):
            editor._read_editor_image(paths[0])
            editor._read_editor_image(paths[1])
            self.assertEqual(len(editor._DECODED_IMAGE_CACHE), 1)
            self.assertEqual(editor._read_editor_image(paths[0], "L").mode, "L")
            self.assertLessEqual(sum(im.width * im.height * len(im.getbands())
                                     for im in editor._DECODED_IMAGE_CACHE.values()), 64)
        path = self.root / "bad.png"
        path.write_bytes(b"invalid image")
        with self.assertRaises(UnidentifiedImageError):
            editor._read_editor_image(path)
        Image.new("RGB", (4, 4), "green").save(path)
        self.assertEqual(editor._read_editor_image(path).getpixel((0, 0)), (0, 128, 0, 255))

    def test_concurrent_readers_do_not_share_mutable_pixels(self):
        path = self.root / "page.png"
        Image.new("RGBA", (16, 16), "white").save(path)
        def read(index):
            image = editor._read_editor_image(path)
            image.putpixel((0, 0), (index, 0, 0, 255))
            return image
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(read, range(12)))
        self.assertEqual([im.getpixel((0, 0))[0] for im in results], list(range(12)))
        self.assertEqual(editor._read_editor_image(path).getpixel((0, 0)), (255, 255, 255, 255))

    def test_native_diff_preserves_rgb_threshold_bounds_and_alpha(self):
        for mode in ("RGB", "RGBA"):
            for kind in ("rgb", "alpha_only", "below_threshold"):
                with self.subTest(mode=mode, kind=kind):
                    clean = np.full((19, 23, 4), 120, dtype=np.uint8)
                    changed = clean.copy()
                    if kind == "rgb":
                        changed[4:9, 7:12, 0] += 3
                        changed[12, 15, 2] -= 3
                        changed[0, 0, 1] += 2
                    elif kind == "alpha_only":
                        changed[:, :, 3] = 255
                    else:
                        changed[4:9, 7:12, :3] += 2
                    base = Image.fromarray(clean).convert(mode)
                    rendered = Image.fromarray(changed).convert(mode)
                    a = np.asarray(base.convert("RGBA"))
                    b = np.asarray(rendered.convert("RGBA"))
                    expected_mask = np.max(np.abs(b[..., :3].astype(np.int16) - a[..., :3].astype(np.int16)), axis=2) > 2
                    with patch("core.editor_session._region_patch_cache_key", return_value=mode+kind), patch(
                        "core.editor_session._render_region_onto", return_value=rendered):
                        result, errors = editor._cached_region_patch(base, "base", {}, 1)
                    self.assertFalse(errors)
                    if not expected_mask.any():
                        self.assertIsNone(result)
                        continue
                    ys, xs = np.where(expected_mask)
                    box = (max(0, xs.min()-2), max(0, ys.min()-2), min(23, xs.max()+3), min(19, ys.max()+3))
                    self.assertEqual(result[0], box)
                    x1, y1, x2, y2 = box
                    np.testing.assert_array_equal(result[1], expected_mask[y1:y2, x1:x2])
                    np.testing.assert_array_equal(result[2], b[y1:y2, x1:x2])

    def test_owned_composite_updates_only_the_owned_accumulator(self):
        source = Image.new("RGBA", (10, 10), "white")
        background = source.copy()
        pixels = np.full((2, 2, 4), (255, 0, 0, 255), dtype=np.uint8)
        patch_value = ((2, 2, 4, 4), np.ones((2, 2), dtype=bool), pixels)
        with patch("core.editor_session._cached_region_patch", return_value=(patch_value, [])):
            result = editor._apply_region_patch(source, background, "base", {}, 1, [], copy_image=False)
        self.assertIs(result, source)
        self.assertEqual(background.getpixel((2, 2)), (255, 255, 255, 255))
        self.assertEqual(result.getpixel((2, 2)), (255, 0, 0, 255))


    def test_text_free_frame_cache_tracks_geometry_masks_background_and_output_info(self):
        base, font = self.root / "base.png", self.root / "font.ttf"
        Image.new("RGBA", (20, 20), "white").save(base)
        font.write_bytes(b"fixture")
        mask = self.root / "mask.bmp"
        Image.new("L", (20, 20), 255).save(mask)
        region = {"id": "r1", "bbox": [2, 2, 12, 12], "original_bbox": [2, 2, 12, 12],
                  "translation": "text", "render": {"font_dir": str(font)},
                  "mask_path": str(mask), "use_mask": True}
        session = {"base_image_path": str(base), "inpainted_image_path": str(base),
                   "inpainted_background_text_free": True, "regions": [region]}
        def render(image, current, index, errors):
            image.putpixel((3, 3), (20, 30, 40, 255))
            image.info["font_size"] = 17
            return image
        with patch("core.editor_session._render_region_onto", side_effect=render) as renderer:
            first, errors = editor.render_session_image(session)
            repeat, repeated_errors = editor.render_session_image(session)
            self.assertFalse(errors + repeated_errors)
            self.assertEqual(renderer.call_count, 1)
            self.assertEqual(repeat.info, first.info)
            repeat.putpixel((3, 3), (0, 0, 0, 0))
            third, _ = editor.render_session_image(session)
            self.assertEqual(third.getpixel((3, 3)), (20, 30, 40, 255))
            # Original geometry controls mask use even if the visible bbox is unchanged.
            region["original_bbox"] = [0, 0, 19, 19]
            editor.render_session_image(session)
            self.assertEqual(renderer.call_count, 2)
            stamp = mask.stat()
            Image.new("L", (20, 20), 0).save(mask)
            os.utime(mask, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            editor.render_session_image(session)
            self.assertEqual(renderer.call_count, 3)
            Image.new("RGBA", (20, 20), "blue").save(base)
            updated, _ = editor.render_session_image(session)
            self.assertEqual(renderer.call_count, 4)
            self.assertEqual(updated.getpixel((18, 18)), (0, 0, 255, 255))

    def test_text_free_render_warnings_are_not_suppressed_by_frame_cache(self):
        base = self.root / "base.png"
        Image.new("RGBA", (12, 12), "white").save(base)
        session = {"base_image_path": str(base), "inpainted_image_path": str(base),
                   "inpainted_background_text_free": True,
                   "regions": [{"id": "r1", "translation": "text"}]}
        def render(image, region, index, errors):
            errors.append("fixture warning")
            return image
        with patch("core.editor_session._render_region_onto", side_effect=render) as renderer:
            for _ in range(2):
                _, errors = editor.render_session_image(session)
                self.assertEqual(errors, ["fixture warning"])
            self.assertEqual(renderer.call_count, 2)


    def test_frame_rendered_while_mask_changes_is_not_cached_under_old_mask(self):
        base, mask, font = self.root / "base.png", self.root / "mask.bmp", self.root / "font.ttf"
        Image.new("RGBA", (12, 12), "white").save(base)
        Image.new("L", (12, 12), 255).save(mask)
        original_mask = mask.read_bytes()
        font.write_bytes(b"fixture")
        session = {"base_image_path": str(base), "inpainted_image_path": str(base),
                   "inpainted_background_text_free": True,
                   "regions": [{"id": "r1", "translation": "text", "bbox": [0, 0, 10, 10],
                                "use_mask": True, "mask_path": str(mask),
                                "render": {"font_dir": str(font)}}]}
        def render(image, region, index, errors):
            Image.new("L", (12, 12), 0).save(mask)
            image.putpixel((3, 3), (10, 20, 30, 255))
            return image
        with patch("core.editor_session._render_region_onto", side_effect=render) as renderer:
            editor.render_session_image(session)
            mask.write_bytes(original_mask)
            editor.render_session_image(session)
            self.assertEqual(renderer.call_count, 2)
