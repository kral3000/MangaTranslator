"""Qt-owned job queue; workers only return immutable results and progress events."""

import hashlib
import threading
import time
from uuid import uuid4

from PyQt6.QtCore import QObject, Qt, pyqtSignal

PENDING = {"queued", "running"}


def artwork_digest(image, box):
    return hashlib.sha256(
        repr((image.size, tuple(box))).encode()
        + image.convert("RGB").crop(tuple(box)).tobytes()
    ).hexdigest()


class AiJobQueue(QObject):
    changed = pyqtSignal()
    completed = pyqtSignal(str)
    _event = pyqtSignal(str, str, object)

    def __init__(self, parent=None, concurrency=2):
        super().__init__(parent)
        self.jobs = {}
        try:
            self.concurrency = max(1, min(16, int(concurrency)))
        except (TypeError, ValueError):
            self.concurrency = 2
        self.closed = False
        self._event.connect(self._receive, Qt.ConnectionType.QueuedConnection)

    def active_for(self, kind, metadata):
        return any(
            job["kind"] == kind
            and job["state"] in PENDING
            and job["session_path"] == metadata["session_path"]
            and job["stamp"] == metadata["stamp"]
            for job in self.jobs.values()
        )

    def submit(self, kind, metadata, worker):
        if self.closed or self.active_for(kind, metadata):
            return None
        job_id = uuid4().hex
        self.jobs[job_id] = {
            **metadata,
            "id": job_id,
            "job_id": job_id,
            "kind": kind,
            "state": "queued",
            "stage": "순서 대기",
            "submitted": time.monotonic(),
            "started": None,
            "finished": None,
            "cancel": threading.Event(),
            "worker": worker,
        }
        self._launch_pending()
        self.changed.emit()
        return job_id

    def set_concurrency(self, value):
        self.concurrency = max(1, min(16, int(value)))
        self._launch_pending()
        self.changed.emit()

    def _launch_pending(self):
        if self.closed:
            return
        for job in list(self.jobs.values()):
            if job["state"] != "queued":
                continue
            limit = self.concurrency if job["kind"] == "sfx" else 1
            running = sum(
                item["kind"] == job["kind"] and item["state"] == "running"
                for item in self.jobs.values()
            )
            if running >= limit:
                continue
            job["state"], job["started"], job["stage"] = (
                "running",
                time.monotonic(),
                "요청 준비",
            )
            worker = job.pop("worker")
            threading.Thread(
                target=self._run, args=(job["id"], worker, job["cancel"]), daemon=True
            ).start()

    def _run(self, job_id, worker, cancel):
        def progress(message):
            try:
                self._event.emit(job_id, "progress", str(message))
            except RuntimeError:
                pass  # Window was closed while a native request was finishing.

        try:
            result = worker(cancel, progress)
            if not isinstance(result, dict):
                result = {"error": "AI 응답을 읽지 못했습니다."}
        except Exception as exc:
            result = {
                "error": str(exc),
                "failure_preview": getattr(exc, "preview", None),
            }
        try:
            self._event.emit(job_id, "finished", result)
        except RuntimeError:
            pass

    def _receive(self, job_id, event, payload):
        if self.closed or job_id not in self.jobs:
            return
        job = self.jobs[job_id]
        if event == "progress":
            if job["state"] == "running" and not job["cancel"].is_set():
                job["stage"] = payload
        else:
            job["finished"] = time.monotonic()
            job["result"] = payload
            if job["cancel"].is_set() or payload.get("cancelled"):
                job["state"], job["stage"] = "cancelled", "중단됨 · 기존 내용 유지"
            elif payload.get("error"):
                job["state"], job["stage"] = "error", str(payload["error"])
            else:
                job["state"], job["stage"] = "ready", "결과 준비됨 · 비교 후 적용"
            self.completed.emit(job_id)
            self._launch_pending()
        self.changed.emit()

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job is None or job["state"] not in PENDING:
            return
        job["cancel"].set()
        if job["state"] == "queued":
            job.pop("worker", None)
            job["state"], job["stage"] = "cancelled", "대기 취소됨"
            job["finished"] = time.monotonic()
        else:
            job["stage"] = "중단 요청됨 · 응답 정리 중"
        self.changed.emit()

    def mark_applied(self, job_id):
        if job_id in self.jobs:
            self.jobs[job_id]["state"] = "applied"
            self.jobs[job_id]["stage"] = "적용·저장됨"
            self.changed.emit()

    def clear_finished(self):
        self.jobs = {
            key: job
            for key, job in self.jobs.items()
            if job["state"] not in {"applied", "error", "cancelled"}
        }
        self.changed.emit()

    def close(self):
        self.closed = True
        for job in self.jobs.values():
            job["cancel"].set()
            job.pop("worker", None)
