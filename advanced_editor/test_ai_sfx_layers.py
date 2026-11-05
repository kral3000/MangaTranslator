import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw

from advanced_editor.ai_sfx import generate_sfx_patch, prepare_request
from advanced_editor.ai_sfx_layers import (
    finish_layers,
    prepare_layers,
    repair_source,
    source_text_mask,
)
from core.ai_image_region import AI_FILE_KEYS, apply_ai_image_regions
from core.editor_package import embed_editor_package, extract_embedded_editor_package
from core.editor_session import load_session, save_session


class SeparatedLetteringTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.page = self.root / "page.png"
        Image.new("RGB", (140, 140), "white").save(self.page)

    def request(self):
        request = prepare_request(
            {"source_image_path": str(self.page)},
            {"bbox": [20, 20, 120, 120], "ai_sfx_wording": "쿵"},
            self.root / "job",
        )
        return prepare_layers(request, threading.Event(), Mock())

    def layer(self, *, mode="RGBA", outside=False):
        image = Image.new(
            mode, (1024, 1024), (0, 0, 0, 0) if mode == "RGBA" else "white"
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle((300, 300, 600, 600), fill="black")
        draw.rectangle((350, 350, 550, 550), fill="white")
        path = self.root / "generated.png"
        image.save(path)
        return {"path": str(path)}

    def test_white_outline_text_mask_excludes_thin_artwork_and_rectangle(self):
        image = Image.new("RGB", (140, 140), "white")
        draw = ImageDraw.Draw(image)
        draw.line((0, 110, 139, 110), fill="black", width=2)
        draw.rounded_rectangle((35, 30, 65, 85), radius=10, outline="black", width=5)
        draw.ellipse((80, 65, 96, 81), outline="black", width=4)
        mask = source_text_mask(image)
        self.assertEqual(mask[50, 50], 255)
        self.assertEqual(mask[73, 88], 255)
        self.assertTrue(np.all(mask[110] == 0))
        self.assertLess(np.mean(mask > 0), 0.40)
        self.assertEqual(mask[15, 15], 0)

    def test_local_repair_never_accepts_unmasked_model_changes(self):
        image = Image.new("RGB", (100, 100), (140, 140, 140))
        mask = np.zeros((100, 100), np.uint8)
        mask[35:60, 35:60] = 255
        with patch(
            "core.image.lama_inpainting.run_lama_large",
            return_value=Image.new("RGB", image.size, "blue"),
        ):
            background, _ = repair_source(image, mask)
        pixels = np.array(background)
        np.testing.assert_array_equal(pixels[mask == 0], np.array(image)[mask == 0])
        self.assertEqual(tuple(pixels[45, 45]), (0, 0, 255))

    def test_plain_background_uses_no_model_and_removes_original(self):
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).rectangle((40, 40, 55, 55), fill="black")
        mask = np.zeros((100, 100), np.uint8)
        mask[37:59, 37:59] = 255
        with patch("core.image.lama_inpainting.run_lama_large") as model:
            background, method = repair_source(image, mask)
        model.assert_not_called()
        self.assertEqual(method, "flat")
        self.assertEqual(background.getpixel((45, 45)), (255, 255, 255))

    def test_preparation_and_two_layers_keep_geometry_and_white_fill(self):
        request = self.request()
        result = finish_layers(request, self.layer())
        self.assertEqual(result["ai_image"]["box"], [20, 20, 120, 120])
        self.assertEqual(result["ai_image"]["pipeline"], "separated_v1")
        with Image.open(result["ai_lettering_path"]) as layer:
            self.assertEqual(layer.mode, "RGBA")
            self.assertEqual(layer.getpixel((0, 0))[3], 0)
            self.assertEqual(layer.getpixel((45, 45)), (255, 255, 255, 255))
        for key in AI_FILE_KEYS:
            self.assertTrue(Path(result[key]).is_file())

    def test_opaque_blank_and_oversized_layers_fail_before_application(self):
        request = self.request()
        with self.assertRaisesRegex(ValueError, "투명"):
            finish_layers(request, self.layer(mode="RGB"))
        image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        path = self.root / "blank.png"
        image.save(path)
        with self.assertRaises(ValueError):
            finish_layers(request, {"path": str(path)})
        image = Image.new("RGBA", (1024, 1024), "white")
        image.save(path)
        with self.assertRaises(ValueError):
            finish_layers(request, {"path": str(path)})

    def test_one_native_generation_and_cancellation_do_not_replace_existing_files(self):
        def generate(_prompt, _references, destination, *_args, **kwargs):
            self.assertTrue(kwargs["transparent_background"])
            path = self.layer()["path"]
            Path(destination).write_bytes(Path(path).read_bytes())
            return {"path": str(destination)}

        with patch(
            "utils.endpoints.codex_image.generate_image", side_effect=generate
        ) as generate_mock:
            result = generate_sfx_patch(
                {"source_image_path": str(self.page)},
                {"bbox": [20, 20, 120, 120]},
                self.root / "run",
                threading.Event(),
                Mock(),
            )
        self.assertEqual(generate_mock.call_count, 1)
        self.assertEqual(result["ai_image"]["pipeline"], "separated_v1")
        raw = self.page.read_bytes()
        cancel = threading.Event()
        cancel.set()
        from utils.endpoints.codex_image import GenerationCancelled

        with patch("utils.endpoints.codex_image.generate_image") as generate_mock:
            with self.assertRaises(GenerationCancelled):
                generate_sfx_patch(
                    {"source_image_path": str(self.page)},
                    {"bbox": [20, 20, 120, 120]},
                    self.root / "cancelled",
                    cancel,
                    Mock(),
                )
        generate_mock.assert_not_called()
        self.assertEqual(self.page.read_bytes(), raw)

    def test_shared_compositor_toggle_and_undo_snapshot_keep_source_and_layers(self):
        result = finish_layers(self.request(), self.layer())
        original = Image.new("RGBA", (140, 140), "white")
        session = {"regions": [{**result, "enabled": True}]}
        snapshot = copy.deepcopy(session)
        rendered = apply_ai_image_regions(original, session)
        expected = Image.open(result["ai_processed_path"]).convert("RGBA")
        np.testing.assert_array_equal(
            np.array(rendered.crop((20, 20, 120, 120))), np.array(expected)
        )
        outside = np.ones((140, 140), bool)
        outside[20:120, 20:120] = False
        np.testing.assert_array_equal(
            np.array(rendered)[outside], np.array(original)[outside]
        )
        session["regions"][0]["enabled"] = False
        np.testing.assert_array_equal(
            np.array(apply_ai_image_regions(rendered, session)), np.array(original)
        )
        np.testing.assert_array_equal(
            np.array(apply_ai_image_regions(original, snapshot)), np.array(rendered)
        )
        # The rendering path actually reads the independent layers, not a stale flattened preview.
        Image.new("RGB", (100, 100), "red").save(result["ai_processed_path"])
        np.testing.assert_array_equal(
            np.array(apply_ai_image_regions(original, snapshot)), np.array(rendered)
        )

    def test_apply_replaces_selected_text_in_place_and_undo_snapshot_keeps_original(
        self,
    ):
        from advanced_editor.launcher import AdvancedEditorWindow

        result = finish_layers(self.request(), self.layer())
        selected = {"id": "r1", "bbox": [20, 20, 120, 120], "translation": "쿵"}
        session = {"regions": [selected]}
        history = []
        draft = {"patch": result}
        window = SimpleNamespace(
            session=session,
            session_path=self.root / "session.json",
            selected_region=lambda: selected,
            _ai_sfx_draft=draft,
            _ai_sfx_draft_is_current=lambda _draft: True,
            _push_history=lambda _label: history.append(copy.deepcopy(session)),
            _mark_region_dirty=Mock(),
            refresh_region_list=Mock(),
            load_selected_region_to_form=Mock(),
            reload_display_image=Mock(),
            schedule_exact_preview=Mock(),
            _sync_ai_sfx_controls=Mock(),
            status_label=Mock(),
        )
        with patch("advanced_editor.launcher.save_session") as save:
            self.assertTrue(AdvancedEditorWindow._apply_ai_sfx_draft(window, draft))
        self.assertEqual(len(session["regions"]), 1)
        self.assertIs(session["regions"][0], selected)
        self.assertEqual(selected["ai_image"]["pipeline"], "separated_v1")
        self.assertNotIn("ai_image", history[0]["regions"][0])
        self.assertEqual(history[0]["regions"][0]["translation"], "쿵")
        save.assert_called_once()

    def test_save_reopen_and_portable_export_preserve_all_independent_layers(self):
        result = finish_layers(self.request(), self.layer())
        region = {**result, "id": "r1", "enabled": True, "bbox": [20, 20, 120, 120]}
        session = {
            "image_size": [140, 140],
            "regions": [region],
            "source_image_path": str(self.page),
            "original_snapshot_path": str(self.page),
            "base_image_path": str(self.page),
            "assets_dir": str(self.root / "assets"),
        }
        session_path = self.root / "session.json"
        save_session(session_path, session)
        reopened = load_session(session_path)
        for key in AI_FILE_KEYS:
            self.assertTrue(Path(reopened["regions"][0][key]).is_file())
        expected = apply_ai_image_regions(
            Image.open(self.page).convert("RGBA"), reopened
        )
        target = self.root / "portable.png"
        expected.save(target)
        embed_editor_package(target, session_path)
        restored_path = extract_embedded_editor_package(
            target, destination_root=self.root / "moved"
        )
        restored = json.loads(restored_path.read_text(encoding="utf-8"))
        for key in AI_FILE_KEYS:
            path = Path(restored["regions"][0][key])
            self.assertTrue(path.is_file())
            self.assertNotEqual(str(path), reopened["regions"][0][key])
        actual = apply_ai_image_regions(
            Image.open(restored["base_image_path"]).convert("RGBA"), restored
        )
        np.testing.assert_array_equal(np.array(actual), np.array(expected))
