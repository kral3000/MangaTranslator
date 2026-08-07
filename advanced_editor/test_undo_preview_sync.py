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
