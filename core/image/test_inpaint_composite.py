import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from core.image.inpaint_composite import finish_outside_inpaint


class OutsideInpaintCompositeTest(unittest.TestCase):
    def fixture(self):
        source = np.full((90, 110, 3), 200, np.uint8)
        mask = np.zeros(source.shape[:2], bool)
        mask[20:65, 30:70] = True
        source[28:56, 27:40] = 255  # Old white outline extends beyond bbox.
        source[31:53, 32:37] = 0
        cleaned = source.copy()
        cleaned[mask] = 223
        return source, cleaned, mask

    def test_white_outline_and_flat_rectangle_are_both_removed(self):
        source, cleaned, mask = self.fixture()
        result = np.array(
            finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask]
            )
        )
        self.assertLess(np.abs(result[35:50, 27:30].astype(float) - 200).mean(), 5)
        self.assertLess(np.abs(result[35:50, 45:60].astype(float) - 200).mean(), 5)
        self.assertGreater(
            result[35:50, 32:37].min(), 190
        )  # Old black text stays erased.
        footprint = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        np.testing.assert_array_equal(result[~footprint], cleaned[~footprint])
        np.testing.assert_array_equal(source[31:53, 32:37], 0)

    def test_protected_pixels_near_outline_and_dark_linework_do_not_change(self):
        source, cleaned, mask = self.fixture()
        source[35:45, 28] = 0
        cleaned[35:45, 28] = 0
        protected = np.zeros(mask.shape, bool)
        protected[25:60, 27:30] = True
        result = np.array(
            finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask], protected
            )
        )
        np.testing.assert_array_equal(result[protected], cleaned[protected])
        np.testing.assert_array_equal(result[35:45, 28], 0)

    def test_empty_thin_irregular_and_large_masks_keep_existing_result(self):
        im = Image.new("RGB", (600, 600), (123, 130, 140))
        source = Image.new("RGB", im.size, "white")
        masks = [np.zeros((600, 600), bool) for _ in range(4)]
        masks[1][10, 10:40] = True
        masks[2][10:50:2, 10:50] = True
        masks[3][1:599, 1:599] = True
        with patch("core.image.inpaint_composite.cv2.seamlessClone") as solve:
            self.assertIs(finish_outside_inpaint(source, im, []), im)
            result = finish_outside_inpaint(source, im, masks)
        solve.assert_not_called()
        np.testing.assert_array_equal(result, im)

    def test_image_edge_rgba_remains_safe_without_a_gradient_solver(self):
        source = np.full((40, 40, 4), 200, np.uint8)
        source[:, :, 3] = 127
        cleaned = source.copy()
        cleaned[:20, :20, :3] = 225
        mask = np.zeros((40, 40), bool)
        mask[:20, :20] = True
        result = finish_outside_inpaint(
            Image.fromarray(source), Image.fromarray(cleaned), [mask]
        )
        self.assertEqual(result.mode, "RGBA")
        np.testing.assert_array_equal(np.array(result)[:, :, 3], cleaned[:, :, 3])
        np.testing.assert_array_equal(np.array(result)[~mask], cleaned[~mask])
        with patch("core.image.inpaint_composite.cv2.seamlessClone") as solve:
            result = finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask]
            )
        solve.assert_not_called()
        self.assertLess(
            np.abs(np.array(result)[:20, :20, :3].astype(float) - 200).mean(), 1
        )

    def test_white_ink_at_boundary_cannot_create_a_glow_in_black_background(self):
        source = np.zeros((100, 100, 3), np.uint8)
        for y in (30, 42, 54):
            source[y : y + 6, 19:26] = 255
        mask = np.zeros((100, 100), bool)
        mask[20:75, 20:70] = True
        cleaned = source.copy()
        cleaned[mask] = 0
        result = np.array(
            finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask]
            )
        )
        self.assertEqual(result[mask].max(), 0)
        self.assertEqual(result[30:36, 19].max(), 0)

    def test_high_contrast_and_texture_do_not_spread_into_the_generated_fill(self):
        source = np.zeros((100, 100, 3), np.uint8)
        source[:, 70:] = (
            200  # A neighboring gray object must not brighten the black fill.
        )
        mask = np.zeros((100, 100), bool)
        mask[20:80, 30:70] = True
        cleaned = source.copy()
        result = finish_outside_inpaint(
            Image.fromarray(source), Image.fromarray(cleaned), [mask]
        )
        np.testing.assert_array_equal(result, cleaned)
        source[20:80, 0:30:2] = 120
        cleaned = source.copy()
        cleaned[mask] = 4
        result = finish_outside_inpaint(
            Image.fromarray(source), Image.fromarray(cleaned), [mask]
        )
        np.testing.assert_array_equal(result, cleaned)

    def test_small_tone_correction_keeps_new_fine_linework(self):
        source = np.full((80, 100, 3), 220, np.uint8)
        mask = np.zeros((80, 100), bool)
        mask[15:65, 25:75] = True
        cleaned = source.copy()
        cleaned[mask] = 230
        cleaned[25:55, 49:51] = 0
        result = np.array(
            finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask]
            )
        )
        np.testing.assert_array_equal(result[25:55, 49:51], 0)
        np.testing.assert_array_equal(result[25:55, 40:45], 220)
        cleaned[30:45, 54:56] = 255
        result = np.array(
            finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask]
            )
        )
        np.testing.assert_array_equal(result[30:45, 54:56], 255)
        source[:] = 240
        result = np.array(
            finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask]
            )
        )
        np.testing.assert_array_equal(result[25:55, 49:51], 0)
        source[:] = 20
        cleaned[:] = 20
        cleaned[mask] = 10
        cleaned[25:55, 49:51] = 0
        result = np.array(
            finish_outside_inpaint(
                Image.fromarray(source), Image.fromarray(cleaned), [mask]
            )
        )
        np.testing.assert_array_equal(result[25:55, 49:51], 0)

    def test_unchanged_source_is_a_noop(self):
        source, _, mask = self.fixture()
        image = Image.fromarray(source)
        result = finish_outside_inpaint(image, image, [mask])
        np.testing.assert_array_equal(result, image)
