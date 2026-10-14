import os
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
        self.widget.resize(370, 30)
        self.widget.show()
        self.addCleanup(self.widget.close)
        self.addCleanup(self.widget.popup.close)
        self.pump()

    def pump(self):
        for _ in range(6):
            self.app.processEvents()

    def test_busy_cancel_and_result_states_use_separate_controls(self):
        w = self.widget
        w.set_state(
            title="AI 식자 · 04 · 10번 영역",
            detail="이미지 생성 중 · 34초",
            busy=True,
            state="busy",
        )
        self.pump()
        self.assertTrue(w.stop_button.isVisible())
        self.assertTrue(w.stop_button.icon().isNull())
        self.assertFalse(w.result_button.isEnabled())
        spy = QSignalSpy(w.stopRequested)
        w.stop_button.click()
        self.assertEqual(len(spy), 1)
        w.set_state(title="AI 식자", detail="중단 요청됨", busy=True, cancelled=True)
        self.assertFalse(w.stop_button.isEnabled())
        w.set_state(
            title="AI 식자",
            detail="결과 준비됨",
            ready=True,
            state="ready",
            result_text="비교·적용",
        )
        self.pump()
        self.assertFalse(w.stop_button.isVisible())
        self.assertTrue(w.result_button.isEnabled())

    def test_narrow_disclosure_preserves_full_controls_and_expands_again(self):
        w = self.widget
        w.set_state(
            title="재번역 · 아주 긴 페이지 이름 · 10번 영역",
            detail="번역 응답 대기 · 6초",
            busy=True,
            compact_text="번역 6초",
        )
        w.resize(80, 30)
        self.pump()
        self.assertTrue(w.compact_button.isVisible())
        self.assertFalse(w.content.isVisible())
        w.compact_button.click()
        self.pump()
        self.assertTrue(w.popup.isVisible())
        self.assertTrue(w.stop_button.isVisible())
        self.assertEqual(w.stop_button.text(), "중단")
        w.popup.hide()
        w.resize(370, 30)
        self.pump()
        self.assertFalse(w.compact_button.isVisible())
        self.assertTrue(w.content.isVisible())
        self.assertIn("번역 응답 대기", w.detail_label.text())

    def test_idle_saved_result_opens_directly_and_dismiss_is_a_distinct_signal(self):
        w = self.widget
        w.resize(80, 30)
        w.set_state(ready=True)
        self.pump()
        result = QSignalSpy(w.resultRequested)
        w.compact_button.click()
        self.pump()
        self.assertEqual(len(result), 1)
        self.assertFalse(w.popup.isVisible())
        w.resize(370, 30)
        w.set_state(title="재번역", detail="오류", state="error")
        self.pump()
        dismissed = QSignalSpy(w.dismissed)
        w.dismiss_button.click()
        self.assertEqual(len(dismissed), 1)


if __name__ == "__main__":
    unittest.main()
