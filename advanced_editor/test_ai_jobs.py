import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from advanced_editor.ai_jobs import AiJobQueue, artwork_digest


class AiJobQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.queue = AiJobQueue(concurrency=2)
        self.gates = []
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.queue.close()
        for event in self.gates:
            event.set()
        self.app.processEvents()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(5)
        self.assertTrue(predicate())

    def submit(self, index, kind="sfx"):
        started, release = threading.Event(), threading.Event()
        self.gates.append(release)

        def work(cancel, progress):
            progress("stage-" + str(index))
            started.set()
            release.wait(3)
            return {"text": str(index)}

        metadata = {
            "session_path": "page.json",
            "stamp": (str(index),),
            "target": str(index),
        }
        job_id = self.queue.submit(kind, metadata, work)
        return job_id, started, release

    def test_image_limit_queue_and_translation_run_independently(self):
        a, b, c = [self.submit(i) for i in range(3)]
        t = self.submit(4, "translation")
        self.assertTrue(a[1].wait(1))
        self.assertTrue(b[1].wait(1))
        self.assertFalse(c[1].is_set())
        self.assertTrue(t[1].wait(1))
        a[2].set()
        self.wait_for(c[1].is_set)
        b[2].set()
        c[2].set()
        t[2].set()
        self.wait_for(
            lambda: all(job["state"] == "ready" for job in self.queue.jobs.values())
        )
        self.assertEqual(
            [self.queue.jobs[item[0]]["result"]["text"] for item in (a, b, c, t)],
            ["0", "1", "2", "4"],
        )

    def test_changing_limit_never_cancels_running_jobs(self):
        self.queue.set_concurrency(1)
        jobs = [self.submit(i) for i in range(4)]
        self.queue.set_concurrency(3)
        self.assertEqual(
            sum(job["state"] == "running" for job in self.queue.jobs.values()), 3
        )
        self.queue.set_concurrency(1)
        self.assertFalse(
            any(job["cancel"].is_set() for job in self.queue.jobs.values())
        )
        jobs[0][2].set()
        jobs[1][2].set()
        self.wait_for(
            lambda: (
                self.queue.jobs[jobs[0][0]]["state"] == "ready"
                and self.queue.jobs[jobs[1][0]]["state"] == "ready"
            )
        )
        self.assertFalse(jobs[3][1].is_set())
        jobs[2][2].set()
        self.wait_for(jobs[3][1].is_set)

    def test_queued_and_running_cancellation_do_not_lose_other_results(self):
        self.queue.set_concurrency(1)
        active = self.submit(1)
        queued = self.submit(2)
        self.queue.cancel(queued[0])
        self.assertEqual(self.queue.jobs[queued[0]]["state"], "cancelled")
        self.assertFalse(queued[1].is_set())
        self.queue.cancel(active[0])
        active[2].set()
        self.wait_for(lambda: self.queue.jobs[active[0]]["state"] == "cancelled")
        ready = self.submit(3)
        ready[2].set()
        self.wait_for(lambda: self.queue.jobs[ready[0]]["state"] == "ready")
        self.queue.clear_finished()
        self.assertEqual(list(self.queue.jobs), [ready[0]])
        self.queue.mark_applied(ready[0])
        self.queue.clear_finished()
        self.assertFalse(self.queue.jobs)

    def test_duplicate_version_is_blocked_and_shutdown_cancels_all_workers(self):
        job = self.submit(1)
        duplicate = self.queue.submit(
            "sfx", {"session_path": "page.json", "stamp": ("1",)}, lambda *_: {}
        )
        self.assertIsNone(duplicate)
        self.queue.close()
        self.assertTrue(self.queue.jobs[job[0]]["cancel"].is_set())
        self.assertIsNone(
            self.queue.submit(
                "translation",
                {"session_path": "page.json", "stamp": ("x",)},
                lambda *_: {},
            )
        )

    def test_progress_and_errors_belong_to_their_own_job(self):
        good = self.submit(1)
        bad = self.queue.submit(
            "sfx",
            {"session_path": "page.json", "stamp": ("bad",), "target": "bad"},
            lambda *_: (_ for _ in ()).throw(ValueError("native failure")),
        )
        self.wait_for(lambda: self.queue.jobs[bad]["state"] == "error")
        self.wait_for(lambda: self.queue.jobs[good[0]]["stage"] == "stage-1")
        self.assertEqual(self.queue.jobs[bad]["result"]["error"], "native failure")
        self.assertEqual(self.queue.jobs[good[0]]["state"], "running")

    def test_invalid_saved_limit_and_crop_digest(self):
        other = AiJobQueue(concurrency="bad")
        self.addCleanup(other.close)
        self.assertEqual(other.concurrency, 2)
        image = Image.new("RGB", (100, 100), "white")
        box = (10, 10, 30, 30)
        original = artwork_digest(image, box)
        image.putpixel((90, 90), (0, 0, 0))
        self.assertEqual(artwork_digest(image, box), original)
        image.putpixel((20, 20), (0, 0, 0))
        self.assertNotEqual(artwork_digest(image, box), original)
