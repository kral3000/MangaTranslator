from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import diffusers
import torch

from core.ml.model_manager import ModelManager, ModelType


class SmallDecoderLoaderTest(TestCase):
    @staticmethod
    def _manager() -> ModelManager:
        manager = object.__new__(ModelManager)
        manager.dtype = torch.float16
        manager.flux_cache_dir = Path("models/flux")
        manager.models = {}
        manager._flux_pipeline_memory_modes = {}
        manager.model_hf_repos = {
            ModelType.FLUX_KLEIN_4B_PIPELINE: {"repo_id": "test/klein-4b"}
        }
        manager._reuse_flux_pipeline = Mock(return_value=None)
        manager._configure_flux_pipeline_memory = Mock(
            return_value="model_cpu_offload"
        )
        return manager

    def test_uses_small_decoder_and_falls_back_when_unavailable(self) -> None:
        small_decoder = object()
        pipeline = SimpleNamespace(transformer=object(), text_encoder=object())

        for decoder_result, expected_vae in (
            (small_decoder, small_decoder),
            (OSError("offline"), None),
        ):
            with self.subTest(expected_vae=expected_vae):
                manager = self._manager()
                with (
                    patch.object(
                        diffusers.AutoencoderKLFlux2,
                        "from_pretrained",
                        side_effect=(
                            decoder_result
                            if isinstance(decoder_result, Exception)
                            else None
                        ),
                        return_value=(
                            None
                            if isinstance(decoder_result, Exception)
                            else decoder_result
                        ),
                    ),
                    patch.object(
                        diffusers.Flux2KleinPipeline,
                        "from_pretrained",
                        return_value=pipeline,
                    ) as load_pipeline,
                    patch("sdnq.common.use_torch_compile", False),
                    patch(
                        "core.ml.model_manager.torch.cuda.is_available",
                        return_value=False,
                    ),
                    patch(
                        "core.ml.model_manager.torch.xpu.is_available",
                        return_value=False,
                    ),
                ):
                    loaded = manager._load_flux_klein(
                        ModelType.FLUX_KLEIN_4B_PIPELINE,
                        "4b",
                    )

                self.assertIs(loaded, pipeline)
                options = load_pipeline.call_args.kwargs
                if expected_vae is None:
                    self.assertNotIn("vae", options)
                else:
                    self.assertIs(options["vae"], expected_vae)
