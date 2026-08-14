from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from PIL import Image

from advanced_editor.launcher import AdvancedEditorWindow


class UndoPreviewSyncTest(TestCase):
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

    def test_named_history_undo_and_redo_restore_one_action_at_a_time(self) -> None:
        window = SimpleNamespace(
            history=[],
            redo_stack=[],
            _capture_state=Mock(side_effect=[
                {"session": {"value": 0}},
                {"session": {"value": 1}},
                {"session": {"value": 0}},
            ]),
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
