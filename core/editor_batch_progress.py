"""Publish completed batch pages without rewriting editable session files."""

import ctypes
import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4


def _process_exists(pid):
    if pid == os.getpid():
        return True
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class EditorBatchProgress:
    def __init__(self, repo, title, output_dir, sessions):
        self.repo = repo.resolve()
        self.path = self.repo / "editor_sessions" / "batch_progress" / f"{uuid4().hex}.json"
        self.completed = set()
        self.data = {
            "title": title,
            "output_dir": str(output_dir.resolve()),
            "pid": os.getpid(),
            "active": True,
            "sessions": [path.relative_to(self.repo).as_posix() for path in sessions],
            "ready": [],
        }

    def publish(self):
        self.data["modified"] = time.time()
        pending = self.path.with_suffix(".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            pending.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
            pending.replace(self.path)
            return True
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not publish editor batch progress: %s", exc)
            return False

    def page_ready(self, session_path):
        from core.editor_session import _asset_paths

        if not session_path.is_file():
            return
        assets, _, _ = _asset_paths(session_path)
        if not (assets / "rendered_snapshot.png").is_file():
            return
        self.completed.add(session_path.relative_to(self.repo).as_posix())
        self.data["ready"] = [path for path in dict.fromkeys(self.data["sessions"]) if path in self.completed]
        self.publish()

    def finish(self):
        self.data["active"] = False
        self.publish()


def read_editor_batches(repo, containing=None):
    repo = repo.resolve()
    folder = repo / "editor_sessions" / "batch_progress"
    for path in folder.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            expected = [repo / value for value in data["sessions"]]
            if not all(p.is_relative_to(repo / "editor_sessions") and ".." not in p.parts for p in expected):
                continue
            ready = set(data["ready"])
            if containing is not None and containing not in ready:
                continue
            yield {
                "key": path.stem,
                "title": str(data["title"]),
                "output_dir": str(data["output_dir"]),
                "modified": float(data["modified"]),
                "active": bool(data["active"] and int(data["pid"]) > 0 and _process_exists(int(data["pid"]))),
                "expected": expected,
                "paths": [
                    p for p in dict.fromkeys(expected)
                    if p.relative_to(repo).as_posix() in ready and p.is_file()
                ],
            }
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            continue


def find_editor_batch(repo, session_path):
    if not session_path.is_relative_to(repo):
        return None
    return max(
        (
            batch for batch in read_editor_batches(repo, containing=session_path.relative_to(repo).as_posix())
            if session_path in batch["paths"]
        ),
        key=lambda batch: batch["modified"],
        default=None,
    )
