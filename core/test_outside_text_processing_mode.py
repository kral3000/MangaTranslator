import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from core import outside_text_processor as processor
from core import pipeline
from core.config import (
    MangaTranslatorConfig,
    normalize_outside_text_processing_mode,
    should_process_outside_text_tag,
)
from core.services import translation
from utils.exceptions import TranslationError


def test_outside_text_processing_modes():
    assert normalize_outside_text_processing_mode(True) == "all"
    assert normalize_outside_text_processing_mode(False) == "off"
    assert should_process_outside_text_tag("prose", "ON")
    assert should_process_outside_text_tag("prose", "OM")
    assert should_process_outside_text_tag("prose", "OD")
    assert should_process_outside_text_tag("prose", "OE")
    assert not should_process_outside_text_tag("prose", "OT")
    assert not should_process_outside_text_tag("prose", "OH")
    assert not should_process_outside_text_tag("prose", "SFX")
    assert not should_process_outside_text_tag("prose", "SFXI")
    assert not should_process_outside_text_tag("prose", "SFXS")
    assert should_process_outside_text_tag("all", "OH")
    assert not should_process_outside_text_tag("off", "ON")
    for tag in (None, "", "UNKNOWN"):
        assert not should_process_outside_text_tag("prose", tag)
    for tag in ("BN", "BE", "BS", "BT", "BW"):
        assert should_process_outside_text_tag("prose", tag)


def test_excluded_text_is_restored_before_editor_snapshot():
    from PIL import Image

    from core.pipeline import _restore_excluded_outside_text

    original = Image.new("RGB", (30, 20), "black")
    cleaned = Image.new("RGB", original.size, "white")
    items = [
        {"bbox": (0, 0, 12, 20), "is_outside_text": True},
        {"bbox": (8, 0, 22, 20), "is_outside_text": True},
        {"bbox": (20, 0, 30, 20), "is_outside_text": True},
    ]
    translations = ["<SFX>쿵</SFX>", "untagged", "<OD>안녕</OD>"]
    restored = _restore_excluded_outside_text(
        cleaned, original, items, translations, "prose"
    )
    editor_background = restored.copy()
    # Overlapping excluded boxes must not block one another's restoration.
    for x in (2, 10, 18):
        assert restored.getpixel((x, 10)) == (0, 0, 0)
        assert editor_background.getpixel((x, 10)) == (0, 0, 0)
    assert restored.getpixel((21, 10)) == (255, 255, 255)
    assert items[0]["skip_render"] and items[1]["skip_render"]
    assert not items[2].get("skip_render")


class OutsideSelectionTest(unittest.TestCase):
    def setUp(self):
        self.config = MangaTranslatorConfig(yolo_model_path="unused")
        self.config.outside_text.enabled = True
        self.config.outside_text.processing_mode = "prose"
        self.config.outside_text.enable_page_number_filtering = False
        self.config.outside_text.inpainting_method = "flux_klein_4b"
        self.config.translation.upscale_method = "none"
        self.config.translation.send_full_page_context = False
        self.config.detection.use_panel_sorting = False
        self.config.preprocessing.auto_scale = False
        self.raw = [([30, 30, 60, 60], 0.9), ([65, 30, 95, 60], 0.9)]
        rng = np.random.default_rng(1)
        self.original = Image.fromarray(
            rng.integers(25, 225, (100, 140, 3), dtype=np.uint8)
        )
        self.seen_masks = []
        self.mask_inputs = []
        owner = self

        class Detector:
            def __init__(self, **kwargs):
                pass

            def detect_outside_text(self, *args, **kwargs):
                return owner.raw.copy()

            def get_text_masks(self, *args, existing_results, **kwargs):
                owner.mask_inputs.append(existing_results)
                mask = np.zeros((100, 140), dtype=bool)
                # Deliberately over-expanded neighboring group: guard must clip SFX.
                for coords, _ in existing_results:
                    x1, y1, x2, y2 = [int(c) for c in coords]
                    mask[y1 - 8 : y2 + 8, x1 - 8 : x2 + 8] = True
                return [
                    {
                        "combined_mask": mask,
                        "bbox": {"x": 22, "y": 22, "width": 81, "height": 46},
                        "original_bbox": {"x": 30, "y": 30, "width": 65, "height": 30},
                        "mask_indices": list(range(len(existing_results))),
                    }
                ], None

        class Inpainter:
            def __init__(self, **kwargs):
                pass

            def inpaint_mask(self, image, mask, **kwargs):
                owner.seen_masks.append(mask.copy())
                # Even a bad model output must not change excluded pixels.
                return Image.new(image.mode, image.size, (250, 250, 250))

        self.detector = Detector
        self.inpainter = Inpainter

    def test_original_rules_and_legacy_restore(self):
        test_outside_text_processing_modes()
        test_excluded_text_is_restored_before_editor_snapshot()

    def test_selection_fails_closed_for_unknown_or_failed_text(self):
        items = [
            {"bbox": b, "original_bbox": b, "is_outside_text": True}
            for b in ((1, 1, 2, 2), (3, 3, 4, 4), (5, 5, 6, 6))
        ]
        selected = pipeline._select_outside_text_for_inpainting(
            items,
            ["<ON>대사</ON>", "unknown", "<OD>[Translation Error: failed]</OD>"],
            "prose",
            "Codex",
        )
        self.assertEqual(selected, {(1, 1, 2, 2)})
        self.assertEqual(
            pipeline._select_outside_text_for_inpainting(
                [items[0], items[0]],
                ["<ON>대사</ON>", "<SFX>쿵</SFX>"],
                "prose",
                "Codex",
            ),
            set(),
        )
        with self.assertRaises(TranslationError):
            pipeline._select_outside_text_for_inpainting(items, [], "prose", "Codex")

    def test_mixed_masks_and_all_fill_backends_preserve_excluded_pixels(self):
        for backend, grouped, fails in [
            ("flux_klein_4b", False, False),
            ("flux_klein_4b", True, False),
            ("flux_klein_4b", True, True),
            ("flux_klein_4b", False, True),
            ("opencv", False, False),
            ("none", False, False),
        ]:
            with self.subTest(backend=backend, grouped=grouped, fails=fails):
                self.config.outside_text.inpainting_method = backend
                self.config.outside_text.flux_group_regions = grouped
                self.seen_masks.clear()
                with (
                    patch.object(processor, "OutsideTextDetector", self.detector),
                    patch.object(processor, "FluxKleinInpainter", self.inpainter),
                    patch.object(
                        self.inpainter,
                        "inpaint_mask",
                        side_effect=RuntimeError("test failure"),
                    )
                    if fails
                    else patch.object(processor, "log_message"),
                ):
                    result, items = processor.process_outside_text(
                        self.original,
                        self.config,
                        "unused",
                        "PNG",
                        bubble_data=[],
                        pre_inpaint_data_callback=lambda _: {(65, 30, 95, 60)},
                    )
                self.assertEqual(len(items), 2)
                self.assertEqual(self.mask_inputs[-1], [self.raw[1]])
                np.testing.assert_array_equal(
                    np.asarray(result)[30:60, 30:60],
                    np.asarray(self.original)[30:60, 30:60],
                )
                self.assertTrue(items[0]["skip_render"])
                self.assertFalse(items[1].get("skip_render", False))
                for mask in self.seen_masks:
                    self.assertFalse(mask[30:60, 30:60].any())
                # Bubble cleanup and editor snapshot must preserve the same region.
                cleaned = Image.new("RGB", result.size, "white")
                restored = pipeline._restore_excluded_outside_text(
                    cleaned,
                    self.original,
                    items,
                    ["<SFX>쿵</SFX>", "<ON>대사</ON>"],
                    "prose",
                )
                np.testing.assert_array_equal(
                    np.asarray(restored)[30:60, 30:60],
                    np.asarray(self.original)[30:60, 30:60],
                )

    def test_no_selection_does_not_load_model_or_build_masks(self):
        with (
            patch.object(processor, "OutsideTextDetector", self.detector),
            patch.object(processor, "FluxKleinInpainter") as model,
        ):
            result, items = processor.process_outside_text(
                self.original,
                self.config,
                "unused",
                "PNG",
                bubble_data=[],
                pre_inpaint_data_callback=lambda _: set(),
            )
            self.assertIs(result, self.original)
            self.assertTrue(all(item["skip_render"] for item in items))
            model.assert_not_called()
            self.assertFalse(self.mask_inputs)
            with self.assertRaises(TranslationError):
                processor.process_outside_text(
                    self.original,
                    self.config,
                    "unused",
                    "PNG",
                    bubble_data=[],
                    pre_inpaint_data_callback=lambda _: None,
                )
            model.assert_not_called()

    def test_fractional_detector_boxes_keep_translated_monologue_and_skip_sfx(self):
        # YOLO returns floating coordinates; crop metadata uses integer pixels.
        self.raw = [
            ([30.25, 30.75, 60.5, 60.125], 0.9),
            ([65.75, 30.25, 95.125, 60.5], 0.9),
        ]
        self.config.translation.provider = "Codex"
        self.config.translation.translation_mode = "one-step"
        response = (
            '{"1":"<OM>どうして私ってこうなんだろ</OM> || '
            '<OM>난 왜 이 모양일까</OM>",'
            '"2":"<SFXS>ず〜ん…</SFXS> || <SFXS>추욱...</SFXS>"}'
        )
        cache = MagicMock()
        cache.get_translation.return_value = (None, None)

        def classify(items):
            texts = translation.call_translation_api_batch(
                config=self.config.translation,
                images_b64=[item["image_b64"] for item in items],
                full_image_b64="",
                mime_types=[item["mime_type"] for item in items],
                full_image_mime_type="image/png",
                bubble_metadata=items,
            )
            return pipeline._select_outside_text_for_inpainting(
                items, texts, "prose", "Codex"
            )

        with (
            patch.object(processor, "OutsideTextDetector", self.detector),
            patch.object(processor, "FluxKleinInpainter", self.inpainter),
            patch.object(translation, "get_cache", return_value=cache),
            patch.object(translation, "_call_llm_endpoint", return_value=response) as api,
            patch.object(translation, "log_message"),
        ):
            result, items = processor.process_outside_text(
                self.original, self.config, "unused", "PNG", bubble_data=[],
                pre_inpaint_data_callback=classify,
            )
        api.assert_called_once()
        self.assertFalse(items[0].get("skip_render", False))
        self.assertTrue(items[1]["skip_render"])
        self.assertEqual(self.mask_inputs[-1], [self.raw[0]])
        self.assertTrue(self.seen_masks[0][30:60, 30:60].any())
        self.assertFalse(self.seen_masks[0][30:60, 65:95].any())
        np.testing.assert_array_equal(
            np.asarray(result)[30:60, 65:95],
            np.asarray(self.original)[30:60, 65:95],
        )

    def test_full_pipeline_reuses_classification_for_both_parallel_options(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap), tempfile.TemporaryDirectory() as folder:
                self.config.translation.parallel_inpaint_translation = overlap
                path = Path(folder) / "source.png"
                self.original.save(path)

                def translate(**kwargs):
                    self.assertFalse(self.seen_masks)
                    return ["<SFX>쿵</SFX>"] * len(kwargs["bubble_metadata"])

                with (
                    patch.object(
                        pipeline, "detect_speech_bubbles", return_value=([], [])
                    ),
                    patch.object(processor, "OutsideTextDetector", self.detector),
                    patch.object(processor, "FluxKleinInpainter") as model,
                    patch.object(
                        pipeline, "call_translation_api_batch", side_effect=translate
                    ) as api,
                ):
                    result = pipeline.translate_and_render(path, self.config)
                api.assert_called_once()
                model.assert_not_called()
                np.testing.assert_array_equal(
                    np.asarray(result.convert("RGB")), np.asarray(self.original)
                )

    def test_all_and_off_modes_keep_their_meaning(self):
        self.config.outside_text.processing_mode = "all"
        with (
            patch.object(processor, "OutsideTextDetector", self.detector),
            patch.object(processor, "FluxKleinInpainter", self.inpainter),
        ):
            _, items = processor.process_outside_text(
                self.original, self.config, "unused", "PNG", bubble_data=[]
            )
        self.assertEqual(self.mask_inputs[-1], self.raw)
        self.assertFalse(any(item.get("skip_render") for item in items))
        self.assertTrue(self.seen_masks[0][30:60, 30:60].all())
        self.config.outside_text.processing_mode = "off"
        with patch.object(processor, "OutsideTextDetector") as detector:
            result, items = processor.process_outside_text(
                self.original, self.config, "unused", "PNG"
            )
        self.assertIs(result, self.original)
        self.assertEqual(items, [])
        detector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
