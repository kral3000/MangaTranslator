from unittest import TestCase

from PyQt6.QtCore import QRectF

from advanced_editor.launcher import EditorCanvas


class RegionBadgeLayoutTest(TestCase):
    def test_badge_is_placed_outside_a_narrow_text_region(self) -> None:
        region = QRectF(100, 100, 20, 120)
        badge = EditorCanvas._choose_region_badge_rect(
            region,
            20,
            17,
            [],
            [],
            QRectF(0, 0, 400, 400),
            QRectF(50, 50, 300, 300),
        )

        self.assertFalse(badge.intersects(region))
        self.assertEqual(badge.left(), region.left())
        self.assertEqual(badge.bottom(), region.top())

    def test_badge_avoids_other_regions_and_previously_placed_badges(self) -> None:
        region = QRectF(100, 100, 40, 100)
        other_region = QRectF(108, 75, 24, 22)
        placed_badge = QRectF(110, 203, 20, 17)
        badge = EditorCanvas._choose_region_badge_rect(
            region,
            20,
            17,
            [other_region],
            [placed_badge],
            QRectF(0, 0, 400, 400),
            QRectF(50, 50, 300, 300),
        )

        self.assertFalse(badge.intersects(region))
        self.assertFalse(badge.intersects(other_region))
        self.assertFalse(badge.intersects(placed_badge))
