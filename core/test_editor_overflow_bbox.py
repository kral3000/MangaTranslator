from unittest import TestCase

from core.editor_session import _editor_overflow_bboxes


class EditorOverflowBBoxTest(TestCase):
    def test_edge_overflow_keeps_original_box_center(self) -> None:
        region = {
            "bbox": [80, 20, 100, 60],
            "translation": "큰 글자",
            "vertical_stack": False,
            "render": {"font_size": 20},
        }

        candidates = _editor_overflow_bboxes(region, (100, 100))

        self.assertTrue(candidates)
        self.assertGreater(candidates[0][2], 100)
        for left, top, right, bottom in candidates:
            self.assertAlmostEqual((left + right) / 2.0, 90.0, delta=0.5)
            self.assertAlmostEqual((top + bottom) / 2.0, 40.0, delta=0.5)
