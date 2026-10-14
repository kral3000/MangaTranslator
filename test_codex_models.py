import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.config import TranslationConfig
from core.services import translation
from ui import utils as ui_utils
from ui.settings_manager import _codex_models
from utils.endpoints import codex


class CodexModelsTest(unittest.TestCase):
    def test_catalog_controls_ui_and_preserves_selected_model(self):
        catalog = [
            {
                "model": "new-model",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "low"},
                    {"reasoningEffort": "high"},
                ],
            }
        ]
        with (
            patch.object(codex, "_catalog", catalog),
            patch.object(
                ui_utils, "_codex_models", return_value=["new-model", "chosen-model"]
            ),
            patch.object(
                ui_utils,
                "get_saved_settings",
                return_value={"provider_models": {"Codex": "chosen-model"}},
            ),
            patch.dict(ui_utils.PROVIDER_MODELS),
        ):
            self.assertEqual(
                ui_utils.get_reasoning_effort_config("Codex", "new-model"),
                (True, ["high", "low"], "low"),
            )
            self.assertEqual(
                ui_utils.initial_dynamic_fetch("Codex", "", None)["value"],
                "chosen-model",
            )

    def test_dispatch_and_two_step_ocr_share_usage_accounting(self):
        config = TranslationConfig(provider="Codex", model_name="new-model")
        usage = {}

        def reply(**kwargs):
            kwargs["usage_output"].update(
                prompt=10, thoughts=1, output=2, total=13, calls=1
            )
            return '{"1":"source text"}'

        with patch.object(
            translation, "call_codex_endpoint", side_effect=reply
        ) as endpoint:
            texts = translation._perform_llm_ocr(
                config,
                ["image"],
                ["image/png"],
                "OCR",
                "Codex",
                "Japanese",
                "rtl",
                usage_output=usage,
            )
        self.assertEqual(texts, ["source text"])
        self.assertEqual(usage["total"], 13)
        self.assertEqual(usage["calls"], 1)
        self.assertEqual(endpoint.call_args.kwargs["parts"][-1], {"text": "OCR"})

    def test_shows_new_generations_and_only_advertised_models(self) -> None:
        with tempfile.TemporaryDirectory() as codex_home:
            cache = Path(codex_home) / "models_cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "models": [
                            {"slug": "gpt-6.1-sol", "visibility": "list"},
                            {"slug": "gpt-6-luna", "visibility": "list"},
                            {"slug": "gpt-6-terra", "visibility": "list"},
                            {"slug": "gpt-6-sol", "visibility": "hide"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CODEX_HOME": codex_home}):
                self.assertEqual(
                    _codex_models(), ["gpt-6.1-sol", "gpt-6-luna", "gpt-6-terra"]
                )


if __name__ == "__main__":
    unittest.main()
