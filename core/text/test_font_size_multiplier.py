from unittest import TestCase

from core.config import RenderingConfig
from core.text.font_manager import font_size_multiplier
from core.text.text_renderer import _scaled_font_config


class FontSizeMultiplierTest(TestCase):
    def test_donquixote_uses_larger_layout_without_changing_saved_size(self) -> None:
        config = RenderingConfig(max_font_size=24, min_font_size=24)

        scaled = _scaled_font_config(config, "fonts/Typo_DonQuixote")

        self.assertEqual(font_size_multiplier("Typo_DonQuixote-Bold.ttf"), 1.25)
        self.assertEqual((scaled.min_font_size, scaled.max_font_size), (30, 30))
        self.assertEqual((config.min_font_size, config.max_font_size), (24, 24))
