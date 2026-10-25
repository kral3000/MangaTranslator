import os
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image

from advanced_editor.launcher import (
    AdvancedEditorWindow,
    _apply_region_format_adjustments,
)
from core.config import MangaTranslatorConfig, RenderingConfig, TagStyleConfig
from core.editor_session import (
    apply_region_tag_font,
    load_session,
    save_pipeline_edit_session,
)


class TagFontSyncTest(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.font_a = self.root / "a.ttf"
        self.font_b = self.root / "b.ttf"
        self.font_a.write_bytes(b"font fixture")
        self.font_b.write_bytes(b"font fixture")
        self.styles = {"BE": {"font_dir": str(self.font_a)}, "BS": {"font_dir": str(self.font_b)},
                       "SFX": {"font_dir": str(self.font_b)}}
        self.region = {"id": "r001", "render_tag": "BE", "font_manual": False,
                       "translation": "강조", "bbox": [0, 0, 80, 80],
                       "render": {"font_dir": str(self.font_a), "font_size": 24, "line_spacing_mult": 1.1}}

    def test_automatic_font_changes_without_touching_other_formatting(self):
        self.region["render_tag"] = "BS"
        self.assertTrue(apply_region_tag_font(self.region, self.styles))
        self.assertEqual(self.region["render"], {"font_dir": str(self.font_b), "font_size": 24,
                                               "line_spacing_mult": 1.1})
        self.assertFalse(apply_region_tag_font(self.region, self.styles))

    def test_manual_legacy_ai_and_missing_font_are_preserved(self):
        for kind in ("manual", "legacy", "ai", "missing"):
            with self.subTest(kind=kind):
                region = deepcopy(self.region)
                region["render_tag"] = "BS"
                if kind == "manual":
                    region["font_manual"] = True
                elif kind == "legacy":
                    region.pop("font_manual")
                elif kind == "ai":
                    region["ai_image"] = {"id": "generated"}
                else:
                    region["render_tag"] = "UNKNOWN"
                before = deepcopy(region)
                self.assertFalse(apply_region_tag_font(region, self.styles))
                self.assertEqual(region, before)
        self.font_b.unlink()
        self.region["render_tag"] = "BS"
        self.assertFalse(apply_region_tag_font(self.region, self.styles))
        self.assertEqual(self.region["render"]["font_dir"], str(self.font_a))

    def test_pipeline_snapshot_uses_tag_font_even_with_missing_or_stale_metadata(self):
        source = self.root / "source.png"
        image = Image.new("RGB", (100, 100), "white")
        image.save(source)
        config = MangaTranslatorConfig(yolo_model_path="", rendering=RenderingConfig(
            font_dir=str(self.font_a), tag_styles={"SFX": TagStyleConfig(font_dir=str(self.font_b))}))
        for metadata in ({}, {"font_dir": str(self.font_a), "font_size": 22}):
            with self.subTest(metadata=metadata):
                path = save_pipeline_edit_session(
                    session_path=self.root / "session.json", source_image_path=source,
                    translated_output_path=None, base_image=image,
                    sorted_bubble_data=[{"bbox": [10, 10, 80, 80], "is_outside_text": True,
                                         "_editor_render_meta": metadata}],
                    translated_texts=["<SFX>덥석</SFX>"], source_ocr_texts=["<SFX>ガッ</SFX>"],
                    processed_bubbles_info=[], config=config, main_min_font=8, main_max_font=30,
                    osb_min_font=8, osb_max_font=30, padding_pixels=2, osb_padding_pixels=2,
                    osb_outline_width=0,
                )
                region = load_session(path)["regions"][0]
                self.assertEqual(region["render"]["font_dir"], str(self.font_b))
                self.assertFalse(region["font_manual"])
                if metadata:
                    self.assertEqual(region["render"]["font_size"], 22)

    def window(self, regions):
        return SimpleNamespace(
            session={"regions": regions}, session_path=self.root / "session.json",
            _push_history=Mock(), _mark_region_dirty=Mock(), refresh_region_list=Mock(),
            load_selected_region_to_form=Mock(), schedule_exact_preview=Mock(), status_label=Mock(),
            _clean_editor_translation=AdvancedEditorWindow._clean_editor_translation,
            selected_region=lambda: regions[0],
        )

    def test_real_editor_translation_apply_updates_auto_only(self):
        manual, legacy = deepcopy(self.region), deepcopy(self.region)
        manual["font_manual"] = True
        legacy.pop("font_manual")
        window = self.window([self.region, manual, legacy])
        with patch("ui.settings_manager.get_saved_settings", return_value={"tag_styles": self.styles}):
            AdvancedEditorWindow._apply_editor_translations(
                window, [0, 1, 2], ["<BS>외침!</BS>"] * 3, settings={})
        self.assertEqual(self.region["render"]["font_dir"], str(self.font_b))
        self.assertEqual(manual["render"]["font_dir"], str(self.font_a))
        self.assertEqual(legacy["render"]["font_dir"], str(self.font_a))
        reopened = load_session(window.session_path)["regions"]
        self.assertFalse(reopened[0]["font_manual"])
        self.assertTrue(reopened[1]["font_manual"])
        self.assertNotIn("font_manual", reopened[2])

    def test_bulk_font_selection_is_manual_even_when_choosing_same_font(self):
        self.assertTrue(_apply_region_format_adjustments(
            {}, self.region, 1, font_path=str(self.font_a)))
        self.assertTrue(self.region["font_manual"])
        self.region["render_tag"] = "BS"
        self.assertFalse(apply_region_tag_font(self.region, self.styles))

    def test_explicit_tag_reset_reenables_automatic_fonts_with_undo(self):
        self.region["font_manual"] = True
        self.region["render_tag"] = "BS"
        window = self.window([self.region])
        with patch("ui.settings_manager.get_saved_settings", return_value={"tag_styles": {
            "BS": {"font_pack": str(self.font_b)}}}):
            AdvancedEditorWindow.use_selected_tag_font(window)
        self.assertFalse(self.region["font_manual"])
        self.assertEqual(self.region["render"]["font_dir"], str(self.font_b))
        window._push_history.assert_called_once_with("태그 기본 글꼴 적용")
        self.assertFalse(load_session(window.session_path)["regions"][0]["font_manual"])

    def test_missing_reset_font_does_not_unlock_manual_font(self):
        self.region["font_manual"] = True
        window = self.window([self.region])
        with patch("ui.settings_manager.get_saved_settings", return_value={"tag_styles": {}}):
            AdvancedEditorWindow.use_selected_tag_font(window)
        self.assertTrue(self.region["font_manual"])
        window._push_history.assert_not_called()
