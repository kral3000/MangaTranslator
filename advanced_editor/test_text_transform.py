import unittest

import numpy as np

from core.editor_session import _region_text_transform_matrix


class TextTransformTest(unittest.TestCase):
    def test_projective_transform_keeps_box_center_fixed(self):
        region = {
            "bbox": [10, 20, 110, 220],
            "rotation_deg": 12,
            "render": {
                "transform_scale_x": 1.3,
                "transform_scale_y": 0.8,
                "transform_skew_x_deg": 15,
                "transform_perspective_y": 0.4,
            },
        }
        matrix = _region_text_transform_matrix(region)
        center = np.array([60.0, 120.0, 1.0])
        mapped = matrix @ center
        mapped /= mapped[2]
        np.testing.assert_allclose(mapped, center, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
