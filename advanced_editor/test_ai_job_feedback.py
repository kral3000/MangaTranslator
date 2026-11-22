import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtTest import QSignalSpy
from PyQt6.QtWidgets import QApplication

from advanced_editor.ai_job_feedback import AiJobFeedback


class AiJobFeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widget = AiJobFeedback()
        self.widget.resize(280, 30)
        self.widget.show()
        self.addCleanup(self.widget.close)
        self.addCleanup(self.widget.popup.close)

    def job(self, index, state):
        return {
            "id": str(index),
            "target": f"AI 식자 · 04 · {index}번 영역",
            "session_path": "page.json",
            "state": state,
            "stage": "이미지 생성 중" if state == "running" else "결과 준비됨",
            "started": time.monotonic() - 10,
            "submitted": time.monotonic() - 12,
            "finished": None,
            "cancel": threading.Event(),
            **({"draft": {}} if state == "ready" else {}),
        }

    def pump(self):
        for _ in range(4):
            self.app.processEvents()

    def test_running_and_completed_results_have_independent_actions(self):
        running, ready = self.job(1, "running"), self.job(2, "ready")
        ready["draft"] = {"text": "result"}
        self.widget.set_jobs({"1": running, "2": ready}, 4, ready=True)
        self.widget._show_popup()
        self.pump()
        stop, result = (
            QSignalSpy(self.widget.jobStopRequested),
            QSignalSpy(self.widget.jobResultRequested),
        )
        self.widget.rows["1"].action.click()
        self.widget.rows["2"].action.click()
        self.assertEqual(list(stop[0]), ["1"])
        self.assertEqual(list(result[0]), ["2"])
        self.assertTrue(self.widget.rows["2"].action.isEnabled())
        self.assertEqual(self.widget.concurrency_spin.value(), 4)

    def test_narrow_summary_keeps_scrollable_jobs_and_selected_result_accessible(self):
        jobs = {str(i): self.job(i, "running") for i in range(8)}
        self.widget.resize(80, 30)
        self.widget.set_jobs(jobs, 2, ready=True)
        self.pump()
        self.widget.summary_button.click()
        self.pump()
        self.assertTrue(self.widget.popup.isVisible())
        self.assertGreater(self.widget.scroll.verticalScrollBar().maximum(), 0)
        self.assertTrue(self.widget.selection_result_button.isEnabled())
        result = QSignalSpy(self.widget.resultRequested)
        self.widget.selection_result_button.click()
        self.pump()
        self.assertEqual(len(result), 1)
        self.assertFalse(self.widget.popup.isVisible())

    def test_summary_keeps_statuses_readable_at_reduced_width(self):
        self.widget.set_jobs(
            {
                state: self.job(index, state)
                for index, state in enumerate(("running", "queued", "ready"))
            },
            2,
            ready=True,
        )
        # Exercise the intermediate state with the platform's actual glyph metrics.
        width = self.widget.summary_stats.sizeHint().width() + 40
        self.widget.resize(width, 30)
        self.pump()
        self.assertFalse(self.widget.result_button.isVisible())
        self.assertEqual(
            [badge.text() for _, badge in self.widget.summary_badges if badge.isVisible()],
            ["진행 1", "대기 1", "결과 1"],
        )
        self.assertIn("결과 1", self.widget.summary_button.accessibleName())
        self.widget.resize(80, 30)
        self.pump()
        self.assertTrue(self.widget.summary_title.isVisible())
        self.assertFalse(self.widget.summary_stats.isVisible())

    def test_setting_and_clear_signals_do_not_change_toolbar_geometry(self):
        before = self.widget.sizeHint()
        limit, cleared = (
            QSignalSpy(self.widget.concurrencyChanged),
            QSignalSpy(self.widget.clearRequested),
        )
        self.widget.set_jobs({"1": self.job(1, "applied")}, 2)
        self.widget._show_popup()
        self.pump()
        self.widget.concurrency_spin.setValue(6)
        self.widget.clear_button.click()
        self.assertEqual(list(limit[0]), [6])
        self.assertEqual(len(cleared), 1)
        self.assertEqual(before, self.widget.sizeHint())
        self.widget.set_jobs({}, 6)
        self.assertEqual(before, self.widget.sizeHint())
        self.assertTrue(self.widget.empty.isVisible())
