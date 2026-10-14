import tempfile
import unittest
from pathlib import Path

from core.config import MangaTranslatorConfig, RenderingConfig, TagStyleConfig
from core.pipeline import _apply_tag_style_overrides


class TagFontFallbackTest(unittest.TestCase):
    def test_missing_tag_font_uses_main_font(self):
        base_font = str(Path(__file__).resolve().parents[1] / "fonts" / "나눔고딕")
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MangaTranslatorConfig(
                yolo_model_path="",
                rendering=RenderingConfig(
                    font_dir=base_font,
                    tag_styles={"ON": TagStyleConfig(font_dir=str(Path(temp_dir) / "missing"))},
                )
            )
            result = _apply_tag_style_overrides("ON", config, base_font, 6, 50, 1.0, 1.0)
        self.assertEqual(result[0], base_font)


if __name__ == "__main__":
    unittest.main()
