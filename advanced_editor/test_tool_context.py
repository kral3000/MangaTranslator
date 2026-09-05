from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from advanced_editor.launcher import AdvancedEditorWindow, SPEECH_BALLOON_REPAIR_PROMPT


class ToolContextTest(TestCase):
    def test_text_style_marker_drives_pressed_button_feedback(self) -> None:
        self.assertEqual(AdvancedEditorWindow._text_style_marker("**굵게**"), "**")
        self.assertEqual(AdvancedEditorWindow._text_style_marker("*기울임*"), "*")
        self.assertEqual(
            AdvancedEditorWindow._text_style_marker("***둘 다***"), "***"
        )
        self.assertEqual(AdvancedEditorWindow._text_style_marker("일반"), "")

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
            paint_color_toolbar_button=SimpleNamespace(setVisible=Mock()),
        )

        AdvancedEditorWindow._sync_tool_context(window, "inpaint_brush")

        window.inspector_sections.setCurrentIndex.assert_called_once_with(1)
        window.settings_tabs.setCurrentIndex.assert_called_once_with(1)
        window.paint_color_toolbar_button.setVisible.assert_called_once_with(False)

    def test_inpaint_help_only_describes_the_selected_method(self) -> None:
        window = SimpleNamespace(
            inpaint_method_combo=SimpleNamespace(
                currentData=Mock(return_value="lama_large")
            ),
            inpaint_method_help=Mock(),
            _inpaint_option_widgets={},
        )

        AdvancedEditorWindow._update_inpaint_option_visibility(window)

        description = window.inpaint_method_help.setText.call_args.args[0]
        self.assertIn("중간 수준", description)
        self.assertNotIn("OpenCV", description)
        self.assertNotIn("Flux", description)

    def test_inpaint_prompt_presets_switch_and_show_the_active_choice(self) -> None:
        prompt_edit = SimpleNamespace(
            value="custom",
            setPlainText=lambda value: setattr(prompt_edit, "value", value),
            toPlainText=lambda: prompt_edit.value,
        )
        buttons = {"일반 인페인트": Mock(), "말풍선 복구": Mock()}
        window = SimpleNamespace(
            inpaint_prompt_edit=prompt_edit,
            inpaint_prompt_presets={
                "일반 인페인트": "general",
                "말풍선 복구": SPEECH_BALLOON_REPAIR_PROMPT,
            },
            inpaint_prompt_preset_buttons=buttons,
            _sync_inpaint_prompt_preset_buttons=lambda: AdvancedEditorWindow._sync_inpaint_prompt_preset_buttons(window),
            _store_inpaint_controls=Mock(),
            status_label=SimpleNamespace(setText=Mock()),
        )

        AdvancedEditorWindow._apply_inpaint_prompt_preset(window, "말풍선 복구")

        self.assertEqual(prompt_edit.value, SPEECH_BALLOON_REPAIR_PROMPT)
        buttons["일반 인페인트"].setChecked.assert_called_with(False)
        buttons["말풍선 복구"].setChecked.assert_called_with(True)
        window._store_inpaint_controls.assert_called_once()

    def test_paint_brush_opens_color_settings_and_toolbar_swatch(self) -> None:
        window = SimpleNamespace(
            inspector_sections=SimpleNamespace(setCurrentIndex=Mock()),
            settings_tabs=SimpleNamespace(setCurrentIndex=Mock()),
            paint_color_toolbar_button=SimpleNamespace(setVisible=Mock()),
        )

        AdvancedEditorWindow._sync_tool_context(window, "paint_brush")

        window.inspector_sections.setCurrentIndex.assert_called_once_with(1)
        window.settings_tabs.setCurrentIndex.assert_called_once_with(3)
        window.paint_color_toolbar_button.setVisible.assert_called_once_with(True)

    def test_square_brush_mask_has_square_corners(self) -> None:
        window = SimpleNamespace(
            image_size=(30, 30),
            brush_size=10,
            brush_shape="square",
        )

        mask = AdvancedEditorWindow._stroke_mask(window, [(15, 15)])

        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask[10, 10], 255)
        self.assertEqual(mask[9, 9], 0)

    def test_circle_brush_mask_keeps_square_corner_clear(self) -> None:
        window = SimpleNamespace(
            image_size=(30, 30),
            brush_size=10,
            brush_shape="circle",
        )

        mask = AdvancedEditorWindow._stroke_mask(window, [(15, 15)])

        self.assertEqual(mask[10, 10], 0)
        self.assertEqual(mask[15, 15], 255)

    def test_inpainting_progress_exposes_running_and_complete_states(self) -> None:
        state_label = Mock()
        window = SimpleNamespace(
            inpaint_progress_bar=Mock(),
            inpaint_progress_state=state_label,
            inpaint_progress_detail=Mock(),
            inpaint_run_button=Mock(),
            inpaint_progress_timer=Mock(),
            _inpainting_elapsed_seconds=99,
        )

        AdvancedEditorWindow._set_inpainting_progress(window, "running", "처리 중")
        window.inpaint_progress_bar.setRange.assert_called_with(0, 0)
        window.inpaint_run_button.setEnabled.assert_called_with(False)
        window.inpaint_progress_timer.start.assert_called_once()

        AdvancedEditorWindow._set_inpainting_progress(window, "complete", "완료")
        window.inpaint_progress_bar.setValue.assert_called_with(100)
        window.inpaint_run_button.setEnabled.assert_called_with(True)
        window.inpaint_progress_timer.stop.assert_called_once()
