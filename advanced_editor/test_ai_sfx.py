import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from advanced_editor.ai_sfx import finish_request, generate_sfx_patch, prepare_request
from advanced_editor.launcher import (
    AdvancedEditorWindow,
    _apply_region_format_adjustments,
)
from core.ai_image_region import apply_ai_image_regions
from core.editor_package import embed_editor_package, extract_embedded_editor_package
from core.editor_session import (
    _region_render_signature,
    add_region,
    render_session_background,
    render_session_image,
)


class AiSfxTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.original = self.root / "original.png"
        self.processed = self.root / "processed.png"
        self.mask = self.root / "mask.png"
        Image.new("RGB", (10, 12), "red").save(self.original)
        Image.new("RGB", (10, 12), "blue").save(self.processed)
        mask = np.zeros((12, 10), np.uint8)
        mask[2:10, 1:9] = 255
        Image.fromarray(mask).save(self.mask)
        self.region = {
            "id": "r1",
            "enabled": True,
            "bbox": [5, 6, 15, 18],
            "translation": "쿵",
            "ai_image": {"id": "patch1", "box": [5, 6, 15, 18]},
            "ai_original_path": str(self.original),
            "ai_processed_path": str(self.processed),
            "ai_mask_path": str(self.mask),
        }
        self.session = {"regions": [self.region]}

    def test_switching_never_modifies_surrounding_pixels_or_input(self):
        base = Image.new("RGBA", (30, 30), "white")
        result = apply_ai_image_regions(base, self.session)
        self.assertEqual(base.getpixel((7, 9)), (255, 255, 255, 255))
        self.assertEqual(result.getpixel((7, 9)), (0, 0, 255, 255))
        self.region["enabled"] = False
        restored = apply_ai_image_regions(result, self.session)
        self.assertEqual(restored.getpixel((7, 9)), (255, 0, 0, 255))
        self.assertEqual(restored.getpixel((5, 6)), (255, 255, 255, 255))
        outside = np.ones((30, 30), bool)
        outside[8:16, 6:14] = False
        np.testing.assert_array_equal(
            np.array(restored)[outside], np.array(base)[outside]
        )

    def test_revisit_after_flattened_export_still_restores_original(self):
        base = self.root / "base.png"
        saved = self.root / "saved.png"
        Image.new("RGBA", (30, 30), "white").save(base)
        apply_ai_image_regions(Image.open(base), self.session).save(saved)
        self.session.update(
            base_image_path=str(base), rendered_snapshot_path=str(saved)
        )
        self.region["render_baseline"] = _region_render_signature(self.region)
        self.region["enabled"] = False
        result, errors = render_session_image(self.session)
        self.assertEqual(errors, [])
        self.assertEqual(result.getpixel((7, 9)), (255, 0, 0, 255))
        self.region["enabled"] = True
        result, errors = render_session_image(self.session)
        self.assertEqual(errors, [])
        self.assertEqual(result.getpixel((7, 9)), (0, 0, 255, 255))

    def test_portable_export_rewrites_all_patch_paths(self):
        session_path = self.root / "edit_session.json"
        session_path.write_text(json.dumps(self.session), encoding="utf-8")
        target = self.root / "export.png"
        Image.new("RGB", (30, 30), "white").save(target)
        embed_editor_package(target, session_path)
        restored_path = extract_embedded_editor_package(
            target, destination_root=self.root / "moved"
        )
        restored = json.loads(restored_path.read_text(encoding="utf-8"))
        for key in ("ai_original_path", "ai_processed_path", "ai_mask_path"):
            moved = Path(restored["regions"][0][key])
            self.assertTrue(moved.is_file())
            self.assertNotEqual(moved, Path(self.region[key]))
            self.assertEqual(moved.read_bytes(), Path(self.region[key]).read_bytes())

    def test_stale_result_cannot_apply_to_another_region(self):
        window = SimpleNamespace(
            _ai_sfx_draft_is_current=Mock(return_value=False),
            _push_history=Mock(),
            status_label=Mock(),
        )
        self.assertFalse(
            AdvancedEditorWindow._apply_ai_sfx_draft(window, {"patch": {}})
        )
        window._push_history.assert_not_called()

    def test_image_region_cannot_move_or_duplicate(self):
        original_box = list(self.region["bbox"])
        window = SimpleNamespace(
            selected_region=lambda: self.region, _push_history=Mock()
        )
        AdvancedEditorWindow.move_selected_region(window, 10, 20)
        AdvancedEditorWindow.update_selected_bbox_from_drag(
            window, "move", original_box, None, None
        )
        AdvancedEditorWindow.duplicate_selected_region(window)
        self.assertEqual(self.region["bbox"], original_box)
        window._push_history.assert_not_called()

    def test_bulk_format_leaves_image_region_untouched(self):
        before = json.dumps(self.region, sort_keys=True)
        self.assertFalse(
            _apply_region_format_adjustments(
                self.session, self.region, 1, font_path="other.ttf", font_size_delta=12
            )
        )
        self.assertEqual(json.dumps(self.region, sort_keys=True), before)

    def test_cancelled_request_does_not_leave_export_assets(self):
        from utils.endpoints.codex_image import GenerationCancelled

        page = self.root / "page.png"
        Image.new("RGB", (100, 100), "white").save(page)
        directory = self.root / "cancelled"
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(GenerationCancelled):
            generate_sfx_patch(
                {"source_image_path": str(page)},
                {"bbox": [20, 20, 50, 60]},
                directory,
                cancelled,
                Mock(),
                current_page=Image.new("RGB", (100, 100), "gray"),
            )
        self.assertEqual(list(directory.glob("*.png")), [])
        self.assertTrue(page.is_file())

    def test_new_text_does_not_inherit_image_patch(self):
        self.session["image_size"] = [100, 100]
        with (
            patch("core.editor_session.load_session", return_value=self.session),
            patch("core.editor_session.save_session"),
            patch("core.editor_session.preview_image"),
        ):
            session, _, _ = add_region("unused.json")
        self.assertFalse(any(key.startswith("ai_") for key in session["regions"][-1]))

    def test_preparation_does_not_send_placeholder_as_exact_wording(self):
        page_path = self.root / "page.png"
        Image.new("RGB", (100, 100), "white").save(page_path)
        request = prepare_request(
            {"source_image_path": str(page_path)},
            {"bbox": [20, 20, 50, 60], "translation": "새 텍스트"},
            self.root / "request",
        )
        self.assertNotIn("exact Korean wording", request["prompt"])
        self.assertTrue(all(Path(p).is_file() for p in request["references"]))

    def test_ai_wording_is_independent_of_dialogue_and_can_be_empty(self):
        page_path = self.root / "wording_page.png"
        Image.new("RGB", (100, 100), "white").save(page_path)
        region = {
            "bbox": [20, 20, 50, 60],
            "translation": "기존 대사",
            "ai_sfx_wording": "쿵",
        }
        request = prepare_request(
            {"source_image_path": str(page_path)}, region, self.root / "wording"
        )
        self.assertIn("exact Korean wording: '쿵'", request["prompt"])
        self.assertNotIn("기존 대사", request["prompt"])
        self.assertEqual(region["translation"], "기존 대사")
        region["ai_sfx_wording"] = ""
        request = prepare_request(
            {"source_image_path": str(page_path)}, region, self.root / "empty_wording"
        )
        self.assertNotIn("exact Korean wording", request["prompt"])

    def test_changed_ai_wording_invalidates_inflight_result(self):
        before = AdvancedEditorWindow._ai_sfx_region_stamp(self.region)
        self.region["ai_sfx_wording"] = "쾅"
        self.assertNotEqual(
            before, AdvancedEditorWindow._ai_sfx_region_stamp(self.region)
        )

    def test_generated_background_drift_does_not_become_a_rectangle(self):
        root = self.root / "composite"
        root.mkdir()
        Image.new("RGB", (60, 60), "white").save(root / "original.png")
        generated = np.full((60, 60, 3), 245, np.uint8)
        generated[20:40, 25:30] = 0
        Image.fromarray(generated).save(root / "generated.png")
        result = finish_request(
            {
                "directory": str(root),
                "target": [15, 15, 45, 45],
                "box": [0, 0, 60, 60],
                "prompt": "test",
            },
            {"path": str(root / "generated.png")},
        )
        with Image.open(result["ai_processed_path"]) as image:
            self.assertEqual(image.getpixel((15, 15)), (255, 255, 255))
            self.assertEqual(image.getpixel((27, 27)), (0, 0, 0))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))

    def test_erased_letter_edges_are_not_blended_back_from_original(self):
        root = self.root / "erased_edges"
        root.mkdir()
        original = np.full((60, 60, 3), 255, np.uint8)
        original[19:41, 19:24] = 230  # Faint antialiasing around the old glyph.
        original[20:40, 20:23] = 0
        Image.fromarray(original).save(root / "original.png")
        generated = np.full_like(original, 245)  # Slight model-wide tonal drift.
        generated[22:38, 34] = 0  # A thin new Korean stroke, separate from the old one.
        Image.fromarray(generated).save(root / "generated.png")
        result = finish_request(
            {
                "directory": str(root),
                "target": [15, 15, 45, 45],
                "box": [0, 0, 60, 60],
                "prompt": "test",
            },
            {"path": str(root / "generated.png")},
        )
        with Image.open(result["ai_processed_path"]) as image:
            pixels = np.array(image)
        np.testing.assert_array_equal(pixels[19:41, 19:24], generated[19:41, 19:24])
        np.testing.assert_array_equal(pixels[22:38, 34], generated[22:38, 34])
        np.testing.assert_array_equal(pixels[0:10], original[0:10])
        self.assertEqual(tuple(pixels[15, 15]), (255, 255, 255))
        with Image.open(root / "original.png") as image:
            np.testing.assert_array_equal(np.array(image), original)

    def test_references_use_current_page_but_keep_true_original_for_restore(self):
        original = np.full((100, 100, 3), 255, np.uint8)
        original[30:50, 30:35] = 0
        page_path = self.root / "source.png"
        Image.fromarray(original).save(page_path)
        source_bytes = page_path.read_bytes()
        current = np.full_like(original, 210)
        current[:, 30:32] = (20, 40, 180)  # A manually repaired boundary.
        current[:10, :10] = (17, 83, 139)  # Outside the selection: must never be sent.
        request = prepare_request(
            {"source_image_path": str(page_path)},
            {"bbox": [25, 25, 60, 65], "ai_sfx_wording": "쿵"},
            self.root / "current_request",
            current_page=Image.fromarray(current),
        )
        box = tuple(request["box"])
        for name, expected in (
            ("working.png", np.array(Image.fromarray(current).crop(box))),
            ("original.png", np.array(Image.fromarray(original).crop(box))),
        ):
            with Image.open(Path(request["directory"]) / name) as image:
                np.testing.assert_array_equal(np.array(image), expected)
        self.assertEqual({Path(p).name for p in request["references"]}, {"reference.png", "source_reference.png"})
        self.assertFalse((Path(request["directory"]) / "page.png").exists())
        for path in request["references"]:
            with Image.open(path) as reference:
                self.assertFalse(np.any(np.all(np.array(reference) == (17, 83, 139), axis=2)))
        with Image.open(Path(request["directory"]) / "source_reference.png") as image:
            np.testing.assert_array_equal(np.array(image), original[25:65, 25:60])
        self.assertEqual(page_path.read_bytes(), source_bytes)

    def test_compositing_uses_working_background_without_reintroducing_old_boundary(self):
        root = self.root / "working_composite"
        root.mkdir()
        original = np.full((60, 60, 3), 255, np.uint8)
        original[:, 18:20] = 0  # Original line, since repaired/moved by the user.
        working = np.full_like(original, 210)
        working[:, 10:12] = 0
        Image.fromarray(original).save(root / "original.png")
        Image.fromarray(working).save(root / "working.png")
        generated = working.copy()
        generated[22:38, 32:36] = 0
        Image.fromarray(generated).save(root / "generated.png")
        result = finish_request(
            {"directory": str(root), "target": [15, 15, 45, 45],
             "box": [0, 0, 60, 60], "prompt": "test"},
            {"path": str(root / "generated.png")},
        )
        with Image.open(result["ai_processed_path"]) as image:
            self.assertEqual(image.getpixel((18, 25)), (210, 210, 210))
            self.assertEqual(image.getpixel((33, 25)), (0, 0, 0))
        with Image.open(result["ai_mask_path"]) as image:
            self.assertEqual(image.getpixel((10, 25)), 0)
        with Image.open(result["ai_original_path"]) as image:
            np.testing.assert_array_equal(np.array(image), original)

    def test_long_crops_use_square_canvas_and_restore_original_extent(self):
        from PIL import ImageDraw
        for size in ((100, 700), (700, 100)):
            with self.subTest(size=size):
                root = self.root / str(size[0])
                page = self.root / f"page_{size[0]}.png"
                Image.new("RGB", size, "white").save(page)
                region = {"bbox": [10, 10, size[0] - 10, size[1] - 10], "translation": "쿵"}
                request = prepare_request({"source_image_path": str(page)}, region, root)
                self.assertEqual(request["canvas_size"][0], request["canvas_size"][1])
                with Image.open(root / "reference.png") as reference:
                    self.assertEqual(reference.size, (1024, 1024))
                with Image.open(root / "working.png") as working:
                    expected_size = working.size
                    canvas = Image.new("RGB", tuple(request["canvas_size"]), "white")
                    content = request["content_box"]
                    canvas.paste(working, content[:2])
                x, y = request["target"][:2]
                ImageDraw.Draw(canvas).rectangle((x + content[0], y + content[1], x + content[0] + 8, y + content[1] + 15), fill="black")
                canvas.save(root / "generated.png")
                result = finish_request(request, {"path": str(root / "generated.png")})
                with Image.open(result["ai_processed_path"]) as processed:
                    self.assertEqual(processed.size, expected_size)
                    self.assertEqual(processed.getpixel((x + 3, y + 3)), (0, 0, 0))

    def test_box_overshoot_is_rejected_before_compositing(self):
        from PIL import ImageDraw
        page = self.root / "page.png"
        Image.new("RGB", (100, 300), "white").save(page)
        request = prepare_request({"source_image_path": str(page)}, {"bbox": [20, 20, 80, 280]}, self.root / "bounds")
        self.assertEqual(request["box"], [20, 20, 80, 280])
        left, top, right, bottom = request["content_box"]
        path = self.root / "bounded.png"
        canvas = Image.new("RGB", tuple(request["canvas_size"]), "white")
        ImageDraw.Draw(canvas).rectangle((left + 10, 30, left + 25, 60), fill="black")
        canvas.save(path)
        result = finish_request(request, {"path": str(path)})
        self.assertEqual(result["ai_image"]["box"], [20, 20, 80, 280])
        ImageDraw.Draw(canvas).rectangle((left - 12, 30, left - 5, 60), fill="black")
        canvas.save(path)
        with self.assertRaisesRegex(ValueError, "박스 밖"):
            finish_request(request, {"path": str(path)})

    def test_small_output_ratio_drift_is_fitted_instead_of_rejected(self):
        from PIL import ImageDraw
        page = self.root / "page.png"
        Image.new("RGB", (200, 200), "white").save(page)
        request = prepare_request({"source_image_path": str(page)}, {"bbox": [40, 40, 150, 150]}, self.root / "fit")
        canvas = Image.new("RGB", (210, 200), "white")
        ImageDraw.Draw(canvas).rectangle((70, 70, 110, 110), fill="black")
        path = self.root / "drift.png"
        canvas.save(path)
        result = finish_request(request, {"path": str(path)})
        self.assertIn("비율", result["warning"])

    def test_failed_generation_keeps_raw_image_and_report_but_success_cleans_it(self):
        from advanced_editor.ai_sfx import SfxGenerationError
        page = self.root / "page.png"
        Image.new("RGB", (100, 300), "white").save(page)
        directory = self.root / "failure"
        def generate(_prompt, _references, destination, *_args):
            Image.new("RGB", (600, 200), "black").save(destination)
            return {"path": str(destination)}
        with patch("utils.endpoints.codex_image.generate_image", side_effect=generate):
            with self.assertRaises(SfxGenerationError) as caught:
                generate_sfx_patch({"source_image_path": str(page)}, {"bbox": [5, 5, 95, 295]}, directory, threading.Event(), lambda _: None)
        self.assertTrue(caught.exception.preview["failed"])
        self.assertTrue((directory / "generated.png").is_file())
        self.assertTrue((directory / "working.png").is_file())
        report = json.loads((directory / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual(report["generated.png"], [600, 200])
        self.assertEqual(report["request"]["canvas_size"], [290, 290])
        with patch("utils.endpoints.codex_image.generate_image", side_effect=generate), patch("advanced_editor.ai_sfx.finish_request", return_value={"ok": True}):
            generate_sfx_patch({"source_image_path": str(page)}, {"bbox": [5, 5, 95, 295]}, self.root / "success", threading.Event(), lambda _: None)
        self.assertFalse((self.root / "success" / "generated.png").exists())

    def test_new_box_does_not_reuse_id_after_deletion(self):
        self.session["regions"] = [{"id": "r001"}, {"id": "r003"}]
        with (patch("core.editor_session.load_session", return_value=self.session),
              patch("core.editor_session.save_session"), patch("core.editor_session.preview_image")):
            session, region_id, _ = add_region("unused.json")
        self.assertEqual(region_id, "r004")
        self.assertEqual(len({r["id"] for r in session["regions"]}), 3)

    def test_result_open_finds_full_stamp_when_ids_collide(self):
        import copy
        import hashlib
        for kind in ("sfx", "translation"):
            with self.subTest(kind=kind):
                first = copy.deepcopy(self.region)
                if kind == "translation":
                    first.pop("ai_image", None)
                target = copy.deepcopy(first)
                target.update(bbox=[1, 2, 5, 8], source_text="target", translation="target")
                stamp = (AdvancedEditorWindow._ai_sfx_region_stamp if kind == "sfx"
                         else AdvancedEditorWindow._translation_reroll_stamp)
                background = Image.new("RGBA", (10, 12), "white")
                draft = {"session_path": str((self.root / "session.json").resolve()),
                         "stamp": stamp(target), "preview_serial": 3,
                         "background_digest": hashlib.sha256(background.tobytes()).hexdigest()}
                window = SimpleNamespace(
                    _ai_sfx_job=None, _translation_reroll_job=None,
                    _ai_feedback_notice={"kind": kind, "state": "ready", "target": "target"},
                    _ai_sfx_draft=draft, _translation_reroll_draft=draft,
                    session_path=self.root / "session.json", session={"regions": [first, target]},
                    selected_index=1, _preview_request_serial=4,
                    _ai_sfx_region_stamp=AdvancedEditorWindow._ai_sfx_region_stamp,
                    _translation_reroll_stamp=AdvancedEditorWindow._translation_reroll_stamp,
                    commit_region_form=Mock(), refresh_region_list=Mock(),
                    load_selected_region_to_form=Mock(), canvas=Mock(), status_label=Mock(),
                    _sync_ai_sfx_controls=Mock(), show_ai_sfx_preview=Mock(),
                    show_translation_reroll_preview=Mock(),
                )
                window.selected_region = lambda: window.session["regions"][window.selected_index]
                window._ai_sfx_draft_is_current = lambda draft: AdvancedEditorWindow._ai_sfx_draft_is_current(window, draft)
                window._translation_reroll_draft_is_current = lambda draft: AdvancedEditorWindow._translation_reroll_draft_is_current(window, draft)
                with patch("advanced_editor.launcher.render_session_background", return_value=background):
                    AdvancedEditorWindow._open_ai_job_result(window)
                self.assertEqual(window.selected_index, 1)
                (window.show_ai_sfx_preview if kind == "sfx" else window.show_translation_reroll_preview).assert_called_once()
                if kind == "sfx":
                    window.show_ai_sfx_preview.reset_mock()
                    with patch("advanced_editor.launcher.render_session_background", return_value=Image.new("RGBA", (10, 12), "red")):
                        AdvancedEditorWindow._open_ai_job_result(window)
                    window.show_ai_sfx_preview.assert_not_called()
                    self.assertEqual(window._ai_feedback_notice["state"], "error")

    def test_page_edit_invalidates_generation_even_if_selected_region_is_unchanged(self):
        window = SimpleNamespace(
            selected_region=lambda: self.region,
            session_path=self.root / "session.json", _preview_request_serial=3,
            _ai_sfx_region_stamp=AdvancedEditorWindow._ai_sfx_region_stamp,
        )
        draft = {"session_path": str(window.session_path.resolve()),
                 "stamp": window._ai_sfx_region_stamp(self.region), "preview_serial": 3}
        self.assertTrue(AdvancedEditorWindow._ai_sfx_draft_is_current(window, draft))
        window._preview_request_serial = 4
        self.assertFalse(AdvancedEditorWindow._ai_sfx_draft_is_current(window, draft))

    def test_start_captures_artwork_without_visible_text_before_worker_runs(self):
        display = Image.new("RGBA", (30, 30), "red")
        background = Image.new("RGBA", (30, 30), (20, 40, 180, 255))
        window = SimpleNamespace(
            _ai_sfx_job=None, _inpainting_running=False, _export_running=False,
            commit_region_form=Mock(), selected_region=lambda: self.region,
            session=self.session, session_path=self.root / "session.json",
            _ai_sfx_region_stamp=AdvancedEditorWindow._ai_sfx_region_stamp,
            _preview_request_serial=3, _task_signals=[], _ai_sfx_timer=Mock(),
            _sync_ai_sfx_controls=Mock(), display_image=display,
        )
        signals = SimpleNamespace(progress=Mock(), finished=Mock())
        with (
            patch("advanced_editor.launcher.TaskSignals", return_value=signals),
            patch("advanced_editor.launcher.render_session_background", return_value=background) as compose,
            patch("advanced_editor.launcher.threading.Thread") as thread,
            patch("advanced_editor.ai_sfx.generate_sfx_patch", return_value={}) as generate,
        ):
            AdvancedEditorWindow.start_ai_sfx_generation(window)
            display.paste("green", (0, 0, 30, 30))
            compose.assert_called_once_with(self.session)
            thread.call_args.kwargs["target"]()
        captured = generate.call_args.kwargs["current_page"]
        self.assertEqual(captured.getpixel((15, 15)), (20, 40, 180, 255))
        self.assertEqual(window._ai_sfx_job["preview_serial"], 3)
        signals.finished.emit.assert_called_once_with({"patch": {}})

    def test_unchanged_crop_needs_no_additional_original_reference(self):
        page_path = self.root / "unchanged.png"
        page = Image.new("RGB", (100, 100), "white")
        page.save(page_path)
        request = prepare_request(
            {"source_image_path": str(page_path)}, {"bbox": [20, 20, 50, 60]},
            self.root / "unchanged_request", current_page=page,
        )
        self.assertEqual(len(request["references"]), 1)
        self.assertFalse((Path(request["directory"]) / "source_reference.png").exists())
        self.assertFalse((Path(request["directory"]) / "page.png").exists())

    def test_editable_text_is_above_ai_and_restore_layers_in_both_region_orders(self):
        base = self.root / "layer_base.png"
        source = self.root / "layer_source.png"
        restore = self.root / "layer_restore.png"
        Image.new("RGBA", (30, 30), "white").save(base)
        Image.new("RGBA", (30, 30), "yellow").save(source)
        Image.new("L", (30, 30), 255).save(restore)
        text = {"id": "dialogue", "enabled": True, "bbox": [6, 8, 10, 12],
                "translation": "대사", "render": {}}

        def draw_text(image, region, index, errors):
            if region.get("ai_image") or not region.get("enabled", True):
                return image
            image.paste("black", (7, 9, 9, 11))
            return image

        for order in ([text, self.region], [self.region, text]):
            for ai_enabled in (True, False):
                for restored in (False, True):
                    with self.subTest(ai_enabled=ai_enabled, restored=restored, first=order[0]["id"]):
                        self.region["enabled"] = ai_enabled
                        session = {"regions": order, "base_image_path": str(base),
                                   "source_image_path": str(source),
                                   "restore_mask_path": str(restore) if restored else ""}
                        with patch("core.editor_session._render_region_onto", side_effect=draw_text):
                            result, errors = render_session_image(session)
                        self.assertEqual(errors, [])
                        self.assertEqual(result.getpixel((7, 9)), (0, 0, 0, 255))
                        expected = "yellow" if restored else ("blue" if ai_enabled else "red")
                        self.assertEqual(result.getpixel((9, 12)), Image.new("RGBA", (1, 1), expected).getpixel((0, 0)))

    def test_ai_context_removes_flattened_editable_text_and_keeps_brush_work(self):
        base = self.root / "context_base.png"
        saved = self.root / "context_saved.png"
        overlay = self.root / "context_paint.png"
        background = Image.new("RGBA", (100, 100), "white")
        background.save(base)
        flattened = background.copy()
        flattened.paste("black", (42, 42, 48, 48))
        flattened.paste("green", (90, 90, 96, 96))
        flattened.save(saved)
        paint = Image.new("RGBA", background.size)
        paint.paste("blue", (44, 44, 46, 46))
        paint.save(overlay)
        text = {"id": "dialogue", "enabled": True, "bbox": [40, 40, 50, 50],
                "translation": "대사", "render": {"font_size": 2}}
        text["render_baseline"] = _region_render_signature(text)
        session = {"regions": [text], "base_image_path": str(base),
                   "source_image_path": str(base), "rendered_snapshot_path": str(saved),
                   "paint_overlay_path": str(overlay)}
        with patch("core.editor_session._render_region_onto") as render_text:
            result = render_session_background(session)
        render_text.assert_not_called()
        self.assertEqual(result.getpixel((42, 42)), (255, 255, 255, 255))
        self.assertEqual(result.getpixel((44, 44)), (0, 0, 255, 255))
        self.assertEqual(result.getpixel((92, 92)), (0, 128, 0, 255))
        request = prepare_request(session, {"bbox": [20, 20, 60, 65]},
                                  self.root / "no_text_request", current_page=result)
        with Image.open(Path(request["directory"]) / "working.png") as crop:
            self.assertEqual(crop.size, (40, 45))
            self.assertEqual(crop.getpixel((22, 22)), (255, 255, 255))
            self.assertEqual(crop.getpixel((24, 24)), (0, 0, 255))
        self.assertFalse((Path(request["directory"]) / "page.png").exists())

    def test_save_keeps_text_free_brush_background_for_reopening_and_ai_context(self):
        base = self.root / "save_base.png"
        snapshot = self.root / "save_snapshot.png"
        overlay = self.root / "save_overlay.png"
        Image.new("RGBA", (100, 100), "white").save(base)
        Image.new("RGBA", (100, 100), "white").save(snapshot)
        paint = Image.new("RGBA", (100, 100))
        paint.paste("blue", (44, 44, 46, 46))
        paint.save(overlay)
        text = {"id": "dialogue", "enabled": True, "bbox": [40, 40, 50, 50],
                "translation": "대사", "render": {"font_size": 2}}
        text["render_baseline"] = _region_render_signature(text)
        session = {"regions": [text], "base_image_path": str(base),
                   "source_image_path": str(base), "rendered_snapshot_path": str(snapshot),
                   "paint_overlay_path": str(overlay), "assets_dir": str(self.root)}
        window = SimpleNamespace(session=session, session_path=self.root / "session.json",
                                 _dirty_region_ids=set())
        AdvancedEditorWindow._flatten_fast_editor_layers_after_save(window, snapshot)
        self.assertEqual(session["paint_overlay_path"], "")
        reopened = render_session_background(session)
        self.assertEqual(reopened.getpixel((44, 44)), (0, 0, 255, 255))

    def test_saved_ai_pixels_are_not_baked_into_underlying_brush_background(self):
        base = self.root / "ai_lower_base.png"
        saved = self.root / "ai_lower_saved.png"
        Image.new("RGBA", (30, 30), "white").save(base)
        apply_ai_image_regions(Image.open(base), self.session).save(saved)
        session = {**self.session, "base_image_path": str(base),
                   "rendered_snapshot_path": str(saved)}
        result = render_session_background(session, include_ai=False)
        self.assertEqual(result.getpixel((7, 9)), (255, 255, 255, 255))

    def test_patch_identity_invalidates_render_signature(self):
        before = _region_render_signature(self.region)
        self.region["ai_image"]["id"] = "patch2"
        self.assertNotEqual(before, _region_render_signature(self.region))


if __name__ == "__main__":
    unittest.main()
