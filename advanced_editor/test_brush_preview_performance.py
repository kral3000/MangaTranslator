from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from advanced_editor.launcher import AdvancedEditorWindow


class BrushPreviewPerformanceTest(TestCase):
    def test_layered_stroke_saves_then_schedules_one_immediate_worker_preview(self):
        events = []
        window = SimpleNamespace(
            session={"regions": [{"ai_image": {"id": "existing"}}]},
            _push_history=Mock(),
            _apply_paint_stroke=lambda *_args: events.append("stroke"),
            save_current_session=lambda **_kwargs: events.append("save"),
            reload_display_image=Mock(), canvas=Mock(), _preview_timer=Mock(),
            schedule_exact_preview=lambda: events.append("schedule"),
        )
        AdvancedEditorWindow.apply_stroke(window, "paint_brush", [(1, 1)])
        self.assertEqual(events, ["stroke", "save", "schedule"])
        window.reload_display_image.assert_not_called()
        window._preview_timer.start.assert_called_once_with(0)
        window.canvas.clear_mask_overlay_cache.assert_called_once()
        window.canvas.update.assert_called_once()

    def test_plain_page_keeps_existing_immediate_background_preview(self):
        window = SimpleNamespace(
            session={"regions": []}, _push_history=Mock(), _apply_paint_stroke=Mock(),
            save_current_session=Mock(), reload_display_image=Mock(),
            schedule_exact_preview=Mock(),
        )
        AdvancedEditorWindow.apply_stroke(window, "paint_brush", [(1, 1)])
        window.reload_display_image.assert_called_once()
        window.schedule_exact_preview.assert_called_once()
