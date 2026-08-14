from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from advanced_editor.launcher import AdvancedEditorWindow


class RenderStatusTest(TestCase):
    def test_status_copy_and_color_state_follow_render_progress(self) -> None:
        style = SimpleNamespace(unpolish=Mock(), polish=Mock())
        dot = SimpleNamespace(
            setProperty=Mock(),
            style=Mock(return_value=style),
            update=Mock(),
        )
        window = SimpleNamespace(
            render_status_label=SimpleNamespace(setText=Mock()),
            render_status_widget=SimpleNamespace(setToolTip=Mock()),
            render_status_dot=dot,
        )

        AdvancedEditorWindow._set_render_status(window, "running")

        window.render_status_label.setText.assert_called_once_with("렌더링 중")
        dot.setProperty.assert_called_once_with("renderState", "running")
        self.assertIn(
            "실제 출력과 같은 방식",
            window.render_status_widget.setToolTip.call_args.args[0],
        )

    def test_edit_marks_the_exact_preview_as_pending_immediately(self) -> None:
        window = SimpleNamespace(
            _preview_request_serial=3,
            _preview_running=False,
            _preview_pending=False,
            _preview_timer=SimpleNamespace(start=Mock()),
            _set_render_status=Mock(),
        )

        AdvancedEditorWindow.schedule_exact_preview(window)

        self.assertEqual(window._preview_request_serial, 4)
        window._set_render_status.assert_called_once_with("pending")
        window._preview_timer.start.assert_called_once_with()
