import json
from unittest import TestCase
from unittest.mock import Mock, patch

from advanced_editor.backend import translate_texts
from core.config import TranslationConfig
from core.services.translation import call_translation_api_batch
from core.text.text_processing import strip_translation_terminal_periods


class TranslationTerminalPeriodsTest(TestCase):
    def test_terminal_periods_are_removed_in_all_translation_paths_only(self):
        cases = {
            "안녕.": "안녕",
            "<BN>안녕.</BN>": "<BN>안녕</BN>",
            "<OM>*내 마음은.*</OM>": "<OM>*내 마음은*</OM>",
            "「안녕。」": "「안녕」",
            "**안녕.**  ": "**안녕**  ",
            "안녕．\r\n잘 가.": "안녕\r\n잘 가",
            "3.14야.": "3.14야",
            "안녕. 잘 가.": "안녕. 잘 가",
            "안녕!": "안녕!",
            "왜?": "왜?",
            "잠깐...": "잠깐...",
            "잠깐..": "잠깐..",
            "잠깐 . . .": "잠깐 . . .",
            "잠깐…": "잠깐…",
            "<BN>.</BN>": "<BN>.</BN>",
            "<BN>...</BN>": "<BN>...</BN>",
            "[OCR FAILED]": "[OCR FAILED]",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(strip_translation_terminal_periods(text), expected)

        config = TranslationConfig(provider="Codex", model_name="test-model")
        source = "<BN>原文。</BN>"
        translated = "<BN>번역.</BN>"
        expected = "<BN>번역</BN>"
        for mode in ("one-step", "two-step", "cache"):
            with self.subTest(mode=mode):
                config.translation_mode = "one-step" if mode == "cache" else mode
                cache = Mock()
                cache.get_translation.return_value = (
                    ([translated], [source]) if mode == "cache" else (None, None)
                )
                response = json.dumps({"1": source + " || " + translated})
                replies = (
                    [response]
                    if mode != "two-step"
                    else [json.dumps({"1": source}), json.dumps({"1": translated})]
                )
                ocr = []
                with (
                    patch("core.services.translation.get_cache", return_value=cache),
                    patch(
                        "core.services.translation._call_llm_endpoint",
                        side_effect=replies,
                    ) as endpoint,
                ):
                    result = call_translation_api_batch(
                        config=config,
                        images_b64=["test-image"],
                        mime_types=["image/png"],
                        full_image_b64="",
                        full_image_mime_type="image/png",
                        bubble_metadata=[{}],
                        ocr_texts_output=ocr,
                    )
                self.assertEqual(result, [expected])
                self.assertEqual(ocr, [source])
                self.assertEqual(
                    endpoint.call_count,
                    {"one-step": 1, "two-step": 2, "cache": 0}[mode],
                )

        with (
            patch(
                "advanced_editor.backend._build_translation_config", return_value=config
            ),
            patch(
                "core.services.translation._call_llm_endpoint",
                return_value=json.dumps({"1": translated}),
            ),
        ):
            self.assertEqual(translate_texts([source], {}), [expected])
