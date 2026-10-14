from datetime import datetime
from unittest import TestCase
from unittest.mock import patch

import gradio as gr

from ui.cleanup import build_cleanup_panel


class CleanupPanelTest(TestCase):
    def test_custom_filter_and_post_cleanup_refresh_share_the_same_dates(self):
        records = [
            dict(
                id=str(day),
                label=f"작업 {day}",
                paths=[],
                modified=datetime.fromisoformat(
                    f"2026-09-{day:02d}T12:00:00+09:00"
                ).timestamp(),
            )
            for day in (1, 2)
        ]
        with gr.Blocks(analytics_enabled=False) as app:
            build_cleanup_panel("sessions")
        functions = {
            event.fn.__name__: event.fn for event in app.fns.values() if event.fn
        }
        filtered = functions["filter_list"](records, -1, "", "2026-09-01", "2026-09-01")
        self.assertEqual(filtered[0]["choices"], [("작업 1", "1")])
        self.assertTrue(filtered[-1]["visible"])
        selected = functions["select_visible"](
            records, -1, "", "2026-09-01", "2026-09-01"
        )
        self.assertEqual(selected[0], ["1"])
        with (
            patch("ui.cleanup.execute_cleanup", return_value=(["1"], [])),
            patch(
                "ui.callbacks._discover_existing_editor_sessions",
                return_value=([], 0, records),
            ),
        ):
            refreshed = functions["remove_selected"](
                {"selected": ["1"]}, ["1"], "output", -1, "", "2026-09-01", "2026-09-01"
            )
        self.assertEqual(refreshed[2]["choices"], [("작업 1", "1")])
        self.assertIn("휴지통으로 1개", refreshed[-1])
