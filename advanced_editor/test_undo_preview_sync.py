from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from advanced_editor.launcher import AdvancedEditorWindow


class UndoPreviewSyncTest(TestCase):
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
