import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from core.config import (
    MangaTranslatorConfig,
    RenderingConfig,
    TagStyleConfig,
    TranslationConfig,
)
from core.editor_session import load_session, save_pipeline_edit_session
from core.pipeline import INSIDE_BUBBLE_RENDER_TAGS, _apply_tag_style_overrides
from core.services.translation import (
    _normalize_local_gemma_tag,
    call_translation_api_batch,
)
from core.tag_styles import (
    TAG_DEFINITIONS,
    normalize_tag_style_table,
    tag_styles_to_table,
)
from core.text.text_processing import extract_translation_tag


class DialogueTagsTest(unittest.TestCase):
    def test_shouting_is_recognized_as_dialogue_and_removed_before_drawing(self):
        self.assertEqual(extract_translation_tag("<BS>그만해!</BS>"), ("BS", "그만해!"))
        self.assertIn("BS", INSIDE_BUBBLE_RENDER_TAGS)
        self.assertEqual(_normalize_local_gemma_tag("SHOUTING"), "BS")
        self.assertEqual(_normalize_local_gemma_tag("BOLD"), "BE")
        self.assertEqual(_normalize_local_gemma_tag("BS"), "BS")

    def test_old_settings_seed_an_independent_shouting_style(self):
        styles = normalize_tag_style_table({"BE": {"font_pack": "emphasis", "max_font_size": 48}})
        self.assertEqual(styles["BS"]["max_font_size"], styles["BE"]["max_font_size"])
        self.assertEqual(styles["BS"]["font_pack"], "HY_POP/HY피오피B.TTF")
        styles["BS"]["font_pack"] = "shout"
        restored = normalize_tag_style_table(tag_styles_to_table(styles))
        self.assertEqual(restored["BE"]["font_pack"], "emphasis")
        self.assertEqual(restored["BS"]["font_pack"], "shout")
        self.assertEqual(list(TAG_DEFINITIONS).index("BS"), list(TAG_DEFINITIONS).index("BE") + 1)

    def test_rendering_uses_separate_fonts_and_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            emphasis = Path(directory) / "emphasis"
            shout = Path(directory) / "shout"
            emphasis.mkdir()
            shout.mkdir()
            config = MangaTranslatorConfig(yolo_model_path="", rendering=RenderingConfig(tag_styles={
                "BE": TagStyleConfig(font_dir=str(emphasis), min_font_size=10, max_font_size=30),
                "BS": TagStyleConfig(font_dir=str(shout), min_font_size=14, max_font_size=48),
            }))
            with patch("core.pipeline.find_font_variants", return_value={"regular": "font.ttf"}):
                be = _apply_tag_style_overrides("BE", config, directory, 8, 16, 1.0, 1.0)
                bs = _apply_tag_style_overrides("BS", config, directory, 8, 16, 1.0, 1.0)
            self.assertEqual(be[:3], (str(emphasis), 10, 30))
            self.assertEqual(bs[:3], (str(shout), 14, 48))

    def test_classification_prompt_and_output_work_in_both_translation_modes(self):
        saved = json.loads((Path(__file__).resolve().parents[1] / "config.public.json").read_text(encoding="utf-8"))
        for key in ("special_instructions", "batch_special_instructions"):
            instructions = saved[key]
            self.assertIn("<BS>...</BS>", instructions)
            self.assertIn("not actually shouted", instructions)
            self.assertIn("Quiet anger", instructions)
        for mode in ("one-step", "two-step"):
            with self.subTest(mode=mode):
                config = TranslationConfig(provider="Codex", model_name="test", translation_mode=mode,
                                           special_instructions=instructions)
                cache = Mock()
                cache.get_translation.return_value = (None, None)
                replies = ([json.dumps({"1": "<BS>やめろ!</BS> || <BS>그만해!</BS>"})]
                           if mode == "one-step" else
                           [json.dumps({"1": "<BS>やめろ!</BS>"}), json.dumps({"1": "<BS>그만해!</BS>"})])
                with patch("core.services.translation.get_cache", return_value=cache), patch(
                    "core.services.translation._call_llm_endpoint", side_effect=replies
                ) as endpoint:
                    result = call_translation_api_batch(config=config, images_b64=["image"],
                                                        mime_types=["image/png"], full_image_b64="",
                                                        full_image_mime_type="image/png",
                                                        bubble_metadata=[{}])
                self.assertEqual(result, ["<BS>그만해!</BS>"])
                self.assertIn("<BS>...</BS>", endpoint.call_args.args[2])

    def test_pipeline_session_retains_shouting_tag_and_plain_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (120, 120), "white")
            source = root / "source.png"
            image.save(source)
            path = save_pipeline_edit_session(
                session_path=root / "session.json", source_image_path=source,
                translated_output_path=None, base_image=image,
                sorted_bubble_data=[{"bbox": [10, 10, 110, 110]}],
                translated_texts=["<BS>그만해!</BS>"], source_ocr_texts=["やめろ!"],
                processed_bubbles_info=[], config=MangaTranslatorConfig(yolo_model_path=""),
                main_min_font=8, main_max_font=24, osb_min_font=8, osb_max_font=24,
                padding_pixels=2, osb_padding_pixels=2, osb_outline_width=0,
            )
            region = load_session(path)["regions"][0]
            self.assertEqual(region["render_tag"], "BS")
            self.assertEqual(region["translation"], "그만해!")
