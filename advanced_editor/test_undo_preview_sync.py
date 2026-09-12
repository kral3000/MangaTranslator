from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from advanced_editor.launcher import (
    AdvancedEditorWindow,
    _apply_region_format_adjustments,
)


class UndoPreviewSyncTest(TestCase):
    def test_duplicate_textbox_keeps_style_without_sharing_erase_mask(self) -> None:
        source = {
            "id": "r001",
            "label": "1. 원본",
            "translation": "복제할 글자",
            "bbox": [10, 20, 40, 60],
            "original_bbox": [10, 20, 40, 60],
            "mask_path": "shared-mask.png",
            "use_mask": True,
            "render_baseline": {"font_size": 12},
            "render": {"font_size": 18, "outline_width": 2.0},
        }
        window = SimpleNamespace(
            session={"regions": [source], "editor_dirty_region_ids": []},
            session_path=Path("session.json"),
            selected_index=0,
            image_size=(100, 100),
            _dirty_region_ids=set(),
            selected_region=lambda: window.session["regions"][window.selected_index],
            commit_region_form=Mock(),
            _push_history=Mock(),
            refresh_region_list=Mock(),
            load_selected_region_to_form=Mock(),
            canvas=SimpleNamespace(update=Mock()),
            schedule_exact_preview=Mock(),
            status_label=SimpleNamespace(setText=Mock()),
        )
        window._mark_region_dirty = lambda region: AdvancedEditorWindow._mark_region_dirty(
            window, region
        )

        with patch("advanced_editor.launcher.save_session"):
            AdvancedEditorWindow.duplicate_selected_region(window)

        duplicate = window.session["regions"][1]
        self.assertEqual(duplicate["bbox"], [26.0, 36.0, 56.0, 76.0])
        self.assertEqual(duplicate["render"], source["render"])
        self.assertIsNot(duplicate["render"], source["render"])
        self.assertIsNone(duplicate["mask_path"])
        self.assertFalse(duplicate["use_mask"])
        self.assertNotIn("render_baseline", duplicate)
        self.assertEqual(window.selected_index, 1)
        window._push_history.assert_called_once_with("텍스트 박스 복제")

    def test_bulk_format_adjustments_apply_relative_values_and_limits(self) -> None:
        session = {"regions": []}
        region = {
            "id": "region-1",
            "vertical_stack": False,
            "render": {
                "font_dir": "old.ttf",
                "font_size": 20,
                "line_spacing_mult": 1.0,
                "padding_pixels": 5.0,
                "outline_width": 1.0,
            },
        }
        session["regions"].append(region)

        changed = _apply_region_format_adjustments(
            session,
            region,
            1,
            font_path="new.ttf",
            font_size_delta=-3,
            line_spacing_delta=0.2,
            padding_delta=-10.0,
            outline_delta=30.0,
        )

        self.assertTrue(changed)
        self.assertEqual(region["render"]["font_dir"], "new.ttf")
        self.assertEqual(region["render"]["font_size"], 17)
        self.assertEqual(region["render"]["line_spacing_mult"], 1.2)
        self.assertEqual(region["render"]["vertical_line_spacing_mult"], 1.2)
        self.assertEqual(region["render"]["padding_pixels"], 0.0)
        self.assertEqual(region["render"]["outline_width"], 20.0)

    def test_history_snapshots_restore_exact_compressed_assets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            mask_path = root / "mask.png"
            overlay_path = root / "overlay.png"
            Image.new("L", (8, 6), 255).save(mask_path)
            Image.new("RGBA", (8, 6), (10, 20, 30, 40)).save(overlay_path)
            original_mask = mask_path.read_bytes()
            original_overlay = overlay_path.read_bytes()
            window = SimpleNamespace(
                session={
                    "regions": [{"mask_path": str(mask_path)}],
                    "paint_overlay_path": str(overlay_path),
                    "editor_dirty_region_ids": [],
                },
                session_path=root / "session.json",
                image_size=(8, 6),
                selected_index=0,
                reload_display_image=Mock(),
                refresh_region_list=Mock(),
                load_selected_region_to_form=Mock(),
                schedule_exact_preview=Mock(),
            )

            state = AdvancedEditorWindow._capture_state(window)
            self.assertIsInstance(state["masks"][str(mask_path)], bytes)
            self.assertIsInstance(state["overlay"], bytes)

            Image.new("L", (8, 6), 0).save(mask_path)
            Image.new("RGBA", (8, 6), (0, 0, 0, 0)).save(overlay_path)
            with patch("advanced_editor.launcher.save_session"):
                AdvancedEditorWindow._restore_state(window, state)

            self.assertEqual(mask_path.read_bytes(), original_mask)
            self.assertEqual(overlay_path.read_bytes(), original_overlay)

    def test_restore_state_invalidates_and_reschedules_exact_preview(self) -> None:
        window = SimpleNamespace(
            session={},
            session_path=Path("session.json"),
            selected_index=None,
            reload_display_image=Mock(),
            refresh_region_list=Mock(),
            load_selected_region_to_form=Mock(),
            schedule_exact_preview=Mock(),
        )
        state = {
            "session": {"regions": [], "editor_dirty_region_ids": []},
            "masks": {},
            "overlay_path": None,
            "overlay": None,
            "selected_index": None,
        }

        with patch("advanced_editor.launcher.save_session"):
            AdvancedEditorWindow._restore_state(window, state)

        window.schedule_exact_preview.assert_called_once_with()

    def test_preview_background_removes_stale_text_from_dirty_regions(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            background_path = Path(temporary_directory) / "translated.png"
            Image.new("RGBA", (8, 6), (255, 255, 255, 255)).save(background_path)
            overlaid = Image.new("RGBA", (8, 6), (10, 20, 30, 255))
            cleaned = Image.new("RGBA", (8, 6), (40, 50, 60, 255))
            window = SimpleNamespace(
                _dirty_region_ids={"r001"},
                session={"regions": [{"id": "r001"}]},
                _editor_background_source=Mock(
                    return_value=(background_path, "translated")
                ),
                _apply_paint_overlay_to=Mock(return_value=overlaid),
                _paste_clean_regions_for_fast_edit=Mock(return_value=cleaned),
            )
            window._region_changed_for_fast_preview = lambda region: (
                AdvancedEditorWindow._region_changed_for_fast_preview(window, region)
            )

            result = AdvancedEditorWindow._preview_background_image(window)

            self.assertIs(result, cleaned)
            window._paste_clean_regions_for_fast_edit.assert_called_once_with(
                overlaid,
                include_dirty_regions=True,
            )

    def test_changed_region_without_dirty_id_is_still_cleaned_before_preview(self) -> None:
        region = {
            "id": "r011",
            "bbox": [10, 20, 110, 120],
            "translation": "현재 문장",
            "render": {"text_alignment": "center"},
            "render_baseline": {
                "bbox": [10, 20, 110, 120],
                "translation": "이전 문장",
                "render": {"text_alignment": "center"},
            },
        }
        window = SimpleNamespace(_dirty_region_ids=set())

        self.assertTrue(
            AdvancedEditorWindow._region_changed_for_fast_preview(window, region)
        )

    def test_named_history_undo_and_redo_restore_one_action_at_a_time(self) -> None:
        window = SimpleNamespace(
            history=[],
            redo_stack=[],
            _capture_state=Mock(
                side_effect=[
                    {"session": {"value": 0}},
                    {"session": {"value": 1}},
                    {"session": {"value": 0}},
                ]
            ),
            _restore_state=Mock(),
            _refresh_history_menu=Mock(),
            status_label=SimpleNamespace(setText=Mock()),
        )

        AdvancedEditorWindow._push_history(window, "텍스트 수정")
        AdvancedEditorWindow.undo_steps(window, 1)
        self.assertEqual(window._restore_state.call_args.args[0]["session"]["value"], 0)
        self.assertEqual(window.redo_stack[-1]["history_label"], "텍스트 수정")

        AdvancedEditorWindow.redo(window)
        self.assertEqual(window._restore_state.call_args.args[0]["session"]["value"], 1)
        self.assertEqual(window.history[-1]["history_label"], "텍스트 수정")

    def test_bulk_history_undo_and_redo_keep_related_sessions_together(self) -> None:
        before_related = {"other.json": {"value": "before"}}
        after_related = {"other.json": {"value": "after"}}
        window = SimpleNamespace(
            history=[
                {
                    "session": {"value": "before"},
                    "related_sessions": before_related,
                    "history_label": "묶음 일괄 서식 변경",
                }
            ],
            redo_stack=[],
            _capture_state=Mock(
                side_effect=[
                    {"session": {"value": "after"}},
                    {"session": {"value": "before"}},
                ]
            ),
            _capture_related_sessions=Mock(
                side_effect=[after_related, before_related]
            ),
            _restore_state=Mock(),
            _refresh_history_menu=Mock(),
            status_label=SimpleNamespace(setText=Mock()),
        )

        AdvancedEditorWindow.undo_steps(window, 1)
        self.assertEqual(
            window._restore_state.call_args.args[0]["related_sessions"],
            before_related,
        )

        AdvancedEditorWindow.redo(window)
        self.assertEqual(
            window._restore_state.call_args.args[0]["related_sessions"],
            after_related,
        )
        self.assertEqual(window.history[-1]["related_sessions"], before_related)

    def test_restore_related_sessions_uses_only_linked_json_allowlist(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            current = root / "current.json"
            sibling = root / "sibling.json"
            outside = root / "outside.json"
            for path in (current, sibling, outside):
                path.write_text("{}", encoding="utf-8")
            window = SimpleNamespace(
                session={},
                session_path=current,
                related_session_paths=[current, sibling],
                selected_index=None,
                reload_display_image=Mock(),
                refresh_region_list=Mock(),
                load_selected_region_to_form=Mock(),
                schedule_exact_preview=Mock(),
            )
            state = {
                "session": {"regions": [], "editor_dirty_region_ids": []},
                "masks": {},
                "overlay_path": None,
                "overlay": None,
                "selected_index": None,
                "related_sessions": {
                    str(sibling): {"value": "allowed"},
                    str(outside): {"value": "blocked"},
                },
            }

            with patch("advanced_editor.launcher.save_session") as save:
                AdvancedEditorWindow._restore_state(window, state)

            saved_paths = [Path(call.args[0]).resolve() for call in save.call_args_list]
            self.assertIn(sibling.resolve(), saved_paths)
            self.assertIn(current.resolve(), saved_paths)
            self.assertNotIn(outside.resolve(), saved_paths)
