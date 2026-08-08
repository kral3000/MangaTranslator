from unittest import TestCase

from PyQt6.QtCore import QRectF

from advanced_editor.launcher import EditorCanvas


class RegionBadgeLayoutTest(TestCase):
    def test_badge_is_always_attached_to_the_region_top_left(self) -> None:
        region = QRectF(100, 100, 20, 120)
        badge = EditorCanvas._region_badge_rect(region, 17, 14)

        self.assertEqual(badge.left(), region.left())
        self.assertEqual(badge.bottom(), region.top() + 1)
        self.assertEqual(badge.width(), 17)
        self.assertEqual(badge.height(), 14)

    def test_selection_handles_sit_outside_each_box_corner(self) -> None:
        region = QRectF(100, 100, 80, 120)
        handles = EditorCanvas._handle_rects(region)

        self.assertLess(handles["nw"].right(), region.left())
        self.assertLess(handles["nw"].bottom(), region.top())
        self.assertGreater(handles["se"].left(), region.right())
        self.assertGreater(handles["se"].top(), region.bottom())
        self.assertTrue(all(handle.size() == handles["nw"].size() for handle in handles.values()))
