import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.config import TranslationConfig
from core.services import translation
from ui import utils as ui_utils
from ui import settings_manager
from ui.settings_manager import _codex_models
from utils.endpoints import codex


class CodexModelsTest(unittest.TestCase):
    def test_windows_native_codex_takes_priority_over_npm_shim(self):
        desktop_exe = r"C:\Codex\codex.exe"
        with (
            patch.object(codex.os, "name", "nt"),
            patch.object(codex.shutil, "which", return_value=desktop_exe) as which,
        ):
            self.assertEqual(codex._codex_command(), [desktop_exe])
        which.assert_called_once_with("codex.exe")

    def test_desktop_shortcut_finds_codex_without_shell_path(self):
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / "OpenAI/Codex/bin/build/codex.exe"
            installed.parent.mkdir(parents=True)
            installed.touch()
            with (
                patch.dict(os.environ, {"LOCALAPPDATA": directory}),
                patch.object(codex.os, "name", "nt"),
                patch.object(codex.shutil, "which", return_value=None),
                patch.object(codex, "Path", type(installed)),
            ):
                self.assertEqual(codex._codex_command(), [str(installed)])

    def test_saved_codex_model_survives_stale_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.json"
            config_file.write_text(
                json.dumps({
                    "provider": "Codex",
                    "provider_models": {"Codex": "new-model"},
                    "reasoning_effort": "high",
                }),
                encoding="utf-8",
            )
            with (
                patch.object(settings_manager, "CONFIG_FILE", config_file),
                patch.dict(settings_manager.PROVIDER_MODELS, {"Codex": ["old-model"]}),
            ):
                saved = settings_manager.get_saved_settings()
            self.assertEqual(saved["model_name"], "new-model")
            self.assertEqual(saved["reasoning_effort"], "high")

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
