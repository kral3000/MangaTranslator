import unittest

import numpy as np

from core.image.inpainting import _klein_clipped_composite_alpha


class KleinCompositeTest(unittest.TestCase):
    def test_rectangle_preserves_outside_and_replaces_text_interior(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[4:16, 4:16] = True
        alpha = _klein_clipped_composite_alpha(mask)
        original = np.full(mask.shape, 200.0)
        original[8:12, 8:12] = 0  # Text to remove.
        generated = np.full(mask.shape, 220.0)
        blended = generated * alpha + original * (1 - alpha)
        np.testing.assert_array_equal(blended[~mask], original[~mask])
        self.assertEqual(blended[4, 10], 210)
        self.assertEqual(blended[5, 10], 220)
        self.assertTrue(np.all(blended[8:12, 8:12] == 220))

    def test_empty_thin_and_image_edge_masks(self):
        mask = np.zeros((12, 12), dtype=bool)
        self.assertFalse(_klein_clipped_composite_alpha(mask).any())
        mask[5, 2:10] = True
        np.testing.assert_array_equal(_klein_clipped_composite_alpha(mask), mask)
        full = _klein_clipped_composite_alpha(np.ones((12, 12), dtype=bool))
        self.assertEqual(full[0, 6], 0.5)
        self.assertEqual(full[6, 6], 1)


if __name__ == "__main__":
    unittest.main()
