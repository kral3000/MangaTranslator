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
