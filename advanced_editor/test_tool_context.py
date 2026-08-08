from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from advanced_editor.launcher import AdvancedEditorWindow


class ToolContextTest(TestCase):
    def test_alignment_buttons_keep_the_existing_session_values(self) -> None:
        window = SimpleNamespace(
            alignment_buttons={
                "left": SimpleNamespace(isChecked=Mock(return_value=False)),
                "center": SimpleNamespace(isChecked=Mock(return_value=False)),
                "right": SimpleNamespace(isChecked=Mock(return_value=True)),
            }
        )

        self.assertEqual(AdvancedEditorWindow._current_alignment(window), "right")

    def test_inpaint_tool_opens_its_contextual_settings(self) -> None:
        window = SimpleNamespace(
            inspector_sections=SimpleNamespace(setCurrentIndex=Mock()),
            settings_tabs=SimpleNamespace(setCurrentIndex=Mock()),
            context_tool_label=SimpleNamespace(setText=Mock()),
            context_hint_label=SimpleNamespace(setText=Mock()),
            paint_color_toolbar_button=SimpleNamespace(setVisible=Mock()),
        )

        AdvancedEditorWindow._sync_tool_context(window, "inpaint_brush")

        window.inspector_sections.setCurrentIndex.assert_called_once_with(1)
        window.settings_tabs.setCurrentIndex.assert_called_once_with(1)
        window.context_tool_label.setText.assert_called_once_with("인페인팅 브러시")
        window.paint_color_toolbar_button.setVisible.assert_called_once_with(False)
