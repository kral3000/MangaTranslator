import copy
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image, ImageDraw
from PyQt6.QtCore import QPoint
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from advanced_editor.ai_jobs import artwork_digest
from advanced_editor.launcher import AdvancedEditorWindow


class AiJobWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Native editor font registration is global to QApplication; restore it
        # so later standalone font-picker layout tests see their own font state.
        self.loaded_fonts = []
        add_font = QFontDatabase.addApplicationFont

        def register(path):
            font_id = add_font(path)
            if font_id >= 0:
                self.loaded_fonts.append(font_id)
            return font_id

        self.font_registration = patch.object(
            QFontDatabase, "addApplicationFont", side_effect=register
        )
        self.font_registration.start()
        self.addCleanup(self.font_registration.stop)
        self.addCleanup(self.remove_test_fonts)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source.png"
        Image.new("RGB", (600, 400), "white").save(source)
        font = str(Path("fonts/나눔고딕").resolve())
        regions = [
            {
                "id": str(i),
                "bbox": box,
                "source_text": "ありがとう",
                "translation": "고마워",
                "font_manual": True,
                "enabled": True,
                "render": {"font_dir": font, "font_size": 18},
            }
            for i, box in enumerate(
                ([40, 40, 140, 140], [240, 40, 340, 140], [40, 240, 140, 340])
            )
        ]
        self.session = {
            "image_size": [600, 400],
            "regions": regions,
            "source_image_path": str(source),
            "original_snapshot_path": str(source),
            "base_image_path": str(source),
            "rendered_snapshot_path": str(source),
            "assets_dir": str(self.root / "assets"),
        }
        self.path = self.root / "page.json"
        self.path.write_text(json.dumps(self.session), encoding="utf-8")
        self.settings = patch(
            "advanced_editor.launcher._load_editor_settings",
            return_value={"show_help": False},
        )
        self.save_settings = patch("advanced_editor.launcher._save_editor_settings")
        self.settings.start()
        self.save_settings_mock = self.save_settings.start()
        self.addCleanup(self.settings.stop)
        self.addCleanup(self.save_settings.stop)
        self.window = AdvancedEditorWindow(str(self.path))
        self.window.resize(1360, 900)
        self.window.show()
        self.gates = [threading.Event(), threading.Event()]
        self.addCleanup(self.close_window)
        self.app.processEvents()

    def test_window_layout_survives_reopen(self):
        self.window.resize(self.window.minimumWidth(), self.window.minimumHeight())
        self.window.splitter.setSizes([475, 640, 0])
        self.app.processEvents()
        size = self.window.size()
        inspector_width = self.window.splitter.sizes()[0]
        self.window.close()
        reopened = AdvancedEditorWindow(str(self.path))
        self.addCleanup(reopened.close)
        reopened.show()
        self.app.processEvents()
        self.assertEqual(reopened.size(), size)
        self.assertEqual(reopened.splitter.sizes()[0], inspector_width)
        reopened.showMaximized()
        reopened.close()
        maximized = AdvancedEditorWindow(str(self.path))
        self.addCleanup(maximized.close)
        self.assertTrue(maximized._start_maximized)

    def remove_test_fonts(self):
        for font_id in self.loaded_fonts:
            QFontDatabase.removeApplicationFont(font_id)

    def close_window(self):
        for gate in self.gates:
            gate.set()
        self.window.close()
        self.app.processEvents()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate())

    def patch_result(
        self, session, region, directory, cancel, progress, *, current_page
    ):
        index = int(region["id"])
        progress("test image request")
        self.gates[index].wait(4)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        box = tuple(round(v) for v in region["bbox"])
        before = current_page.crop(box).convert("RGB")
        layer = Image.new("RGBA", before.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rectangle((30, 30, 50, 50), fill="black")
        processed = Image.alpha_composite(before.convert("RGBA"), layer)
        files = {}
        for key, image in (
            ("ai_original_path", before),
            ("ai_processed_path", processed),
            ("ai_mask_path", Image.new("L", before.size, 255)),
            ("ai_background_path", before),
            ("ai_lettering_path", layer),
            ("ai_erase_mask_path", Image.new("L", before.size, 0)),
        ):
            path = directory / (key + ".png")
            image.save(path)
            files[key] = str(path)
        return {
            **files,
            "ai_image": {
                "id": uuid4().hex,
                "box": list(box),
                "pipeline": "separated_v1",
                "boundary_preserved": True,
            },
        }

    def test_persisted_result_reuses_original_pixels_and_rechecks_edits(self):
        from advanced_editor.lettering_history import read_record, stored_draft
        from core.editor_session import render_session_image

        window = self.window
        window.set_selected_index(0)
        with patch("advanced_editor.ai_sfx.generate_sfx_patch", side_effect=self.patch_result) as generate:
            self.gates[0].set()
            self.assertTrue(window.start_ai_sfx_generation())
            self.wait_for(lambda: any(j["state"] == "ready" for j in window._ai_jobs.jobs.values()))
            job = next(iter(window._ai_jobs.jobs.values()))
            patch_data = job["draft"]["patch"]
            directory = Path(patch_data["ai_processed_path"]).parent
            recovered = stored_draft(directory, read_record(directory), self.path)
            window._ai_jobs.jobs.clear()
            self.assertTrue(window._focus_ai_job_target(recovered, "sfx"))
            self.assertTrue(window._ai_sfx_draft_is_current(recovered))
            edited = copy.deepcopy(window.session)
            edited["regions"][0].update(patch_data, enabled=True)
            expected = render_session_image(edited)[0]
            state_before = copy.deepcopy(window.session)
            self.assertTrue(window._apply_ai_sfx_draft(recovered))
            self.assertEqual(window.display_image.tobytes(), expected.tobytes())
            self.assertNotEqual(window.session["regions"][0]["ai_processed_path"], str(directory / "ai_processed_path.png"))
            window.session = state_before
            window.session["regions"][0]["translation"] = "새 문구"
            self.assertFalse(window._ai_sfx_draft_is_current(recovered))
            generate.assert_called_once()

    def test_parallel_images_translation_preview_apply_and_undo_use_one_editor(self):
        w = self.window
        w.set_selected_index(0)
        before = (
            w.source_edit.mapTo(w, QPoint()),
            w.translation_edit.mapTo(w, QPoint()),
            w.ai_sfx_button.mapTo(w, QPoint()),
        )
        with (
            patch(
                "advanced_editor.ai_sfx.generate_sfx_patch",
                side_effect=self.patch_result,
            ),
            patch(
                "advanced_editor.translation_reroll.reroll_translation",
                return_value={"text": "정말 고마워", "usage": {}},
            ),
        ):
            self.assertTrue(w.start_ai_sfx_generation())
            self.assertTrue(w.translation_reroll_button.isEnabled())
            w.set_selected_index(1)
            self.assertTrue(w.ai_sfx_button.isEnabled())
            self.assertTrue(w.start_ai_sfx_generation())
            w.set_selected_index(2)
            self.assertTrue(w.translation_reroll_button.isEnabled())
            self.assertTrue(w.start_translation_reroll())
            self.wait_for(
                lambda: any(
                    j["kind"] == "translation" and j["state"] == "ready"
                    for j in w._ai_jobs.jobs.values()
                )
            )
            translation = next(
                j for j in w._ai_jobs.jobs.values() if j["kind"] == "translation"
            )
            self.assertEqual(
                sum(j["state"] == "running" for j in w._ai_jobs.jobs.values()), 2
            )
            w._open_ai_job_result(translation["id"])
            self.assertTrue(w._ai_previews[translation["id"]].isVisible())
            self.assertTrue(
                w._apply_translation_reroll(translation["draft"], "정말 고마워")
            )
            self.assertTrue(w.session["regions"][2]["font_manual"])
            self.assertEqual(
                w.session["regions"][2]["render"]["font_dir"],
                self.session["regions"][2]["render"]["font_dir"],
            )
            self.gates[0].set()
            self.wait_for(
                lambda: any(
                    j["kind"] == "sfx"
                    and j["stamp"][0] == "0"
                    and j["state"] == "ready"
                    for j in w._ai_jobs.jobs.values()
                )
            )
            first = next(
                j
                for j in w._ai_jobs.jobs.values()
                if j["kind"] == "sfx" and j["stamp"][0] == "0"
            )
            self.assertTrue(w._apply_ai_sfx_draft(first["draft"]))
            self.gates[1].set()
            self.wait_for(
                lambda: all(j["state"] != "running" for j in w._ai_jobs.jobs.values())
            )
            second = next(
                j
                for j in w._ai_jobs.jobs.values()
                if j["kind"] == "sfx" and j["stamp"][0] == "1"
            )
            w.set_selected_index(1)
            self.assertTrue(w._ai_sfx_draft_is_current(second["draft"]))
            self.assertTrue(w._apply_ai_sfx_draft(second["draft"]))
        self.assertTrue(all(j["state"] == "applied" for j in w._ai_jobs.jobs.values()))
        self.assertEqual(len(w.session["regions"]), 3)
        w.undo()
        self.assertNotIn("ai_image", w.session["regions"][1])
        self.assertIn("ai_image", w.session["regions"][0])
        w.redo()
        self.assertIn("ai_image", w.session["regions"][1])
        self.app.processEvents()
        after = (
            w.source_edit.mapTo(w, QPoint()),
            w.translation_edit.mapTo(w, QPoint()),
            w.ai_sfx_button.mapTo(w, QPoint()),
        )
        self.assertEqual(before, after)

    def test_results_return_to_their_page_without_blocking_other_pages(self):
        w = self.window
        second_path = self.root / "second.json"
        second_path.write_text(json.dumps(self.session), encoding="utf-8")
        with patch(
            "advanced_editor.ai_sfx.generate_sfx_patch", side_effect=self.patch_result
        ):
            w.set_selected_index(0)
            self.assertTrue(w.start_ai_sfx_generation())
            w._switch_editor_session(second_path)
            w.set_selected_index(0)
            self.assertTrue(w.start_ai_sfx_generation())
            self.assertEqual(len(w._ai_jobs.jobs), 2)
            self.gates[0].set()
            self.wait_for(
                lambda: all(j["state"] == "ready" for j in w._ai_jobs.jobs.values())
            )
            jobs = list(w._ai_jobs.jobs.values())
            for job, path in zip(jobs, (self.path, second_path)):
                w._open_ai_job_result(job["id"])
                self.assertEqual(w.session_path.resolve(), path.resolve())
                self.assertTrue(w._ai_previews[job["id"]].isVisible())
                self.assertTrue(w._apply_ai_sfx_draft(job["draft"]))
                w._ai_previews[job["id"]].close()
                self.assertIn("ai_image", w.session["regions"][0])
            self.assertTrue(all(j["state"] == "applied" for j in jobs))

    def test_only_target_artwork_invalidates_an_inflight_result(self):
        w = self.window
        w.set_selected_index(0)
        page = Image.new("RGB", (600, 400), "white")
        draft = {
            "session_path": str(w.session_path.resolve()),
            "stamp": w._ai_sfx_region_stamp(w.selected_region()),
            "background_box": [40, 40, 140, 140],
            "background_digest": artwork_digest(page, [40, 40, 140, 140]),
        }
        with patch(
            "advanced_editor.launcher.render_session_background", return_value=page
        ):
            w._preview_request_serial += 99
            self.assertTrue(w._ai_sfx_draft_is_current(draft))
            page.putpixel((300, 300), (0, 0, 0))
            self.assertTrue(w._ai_sfx_draft_is_current(draft))
            page.putpixel((80, 80), (0, 0, 0))
            self.assertFalse(w._ai_sfx_draft_is_current(draft))

    def test_concurrency_setting_is_saved_without_changing_other_preferences(self):
        w = self.window
        previous = copy.deepcopy(w.editor_settings)
        w._set_ai_concurrency(4)
        self.assertEqual(w._ai_jobs.concurrency, 4)
        self.assertEqual(w.editor_settings, {**previous, "ai_sfx_concurrency": 4})
        self.save_settings_mock.assert_called()
