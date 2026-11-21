"""Share a storage scan across dialogs, including an unfinished size calculation."""

import threading
import time
from functools import lru_cache

from PyQt6.QtCore import QObject, pyqtSignal

from advanced_editor import lettering_history
from advanced_editor.storage_records import job_size, list_storage, session_rows
from core import editor_paths
from core.editor_session_catalog import (
    discover_existing_editor_sessions,
    scan_session_files,
)


@lru_cache(maxsize=12)
def inventory(repo, output, kind, scope):
    return StorageInventory(repo, output, kind, scope)


class StorageInventory(QObject):
    updated = pyqtSignal(str, object)
    delivery = pyqtSignal(int, str, object)

    def __init__(self, repo, output, kind, scope):
        super().__init__()
        self.repo, self.output, self.kind, self.scope = repo, output, kind, scope
        self.rows = None
        self.sizes = {}
        self.loaded = 0
        self.revision = None
        self.running = False
        self.invalid_count = 0
        self.partial = False
        self.want_sizes = False
        self.sizing = False
        self.version = 0
        self.delivery.connect(self._receive)

    def load(self, jobs=None, force=False):
        revision = None
        if self.kind == "history":
            revision = (
                lettering_history.history_revision,
                editor_paths.session_revision,
            )
            if self.scope:
                stamps = []
                for path in self.scope:
                    try:
                        stamps.append(path.stat().st_mtime_ns)
                    except OSError:
                        stamps.append(None)
                revision = (revision, tuple(stamps))
        elif self.kind == "sessions" and jobs is None:
            revision = self.revision
        elif jobs is not None:
            revision = tuple(
                (j["id"], j["modified"], j["page_count"], j.get("translating"))
                for j in jobs
            )

        if (
            not force
            and revision == self.revision
            and (
                self.running
                or (self.rows is not None and time.monotonic() - self.loaded < 90)
            )
        ):
            return
        self.version += 1
        version = self.version
        self.running = True
        self.sizing = False
        self.revision = revision
        seeded = (
            session_rows(jobs) if self.kind == "sessions" and jobs is not None else None
        )
        if seeded is not None:
            self._receive(version, "rows", seeded)

        def scan():
            try:
                rows = seeded
                if rows is None:
                    if self.kind == "sessions":
                        files = scan_session_files(self.repo)
                        if self.rows is None:
                            _, _, recent = discover_existing_editor_sessions(
                                self.repo,
                                files=files,
                                recent_files=96,
                                include_members=True,
                                include_active=True,
                            )
                            self.delivery.emit(version, "recent", session_rows(recent))
                        if version != self.version:
                            return
                        _, invalid_count, jobs_found = discover_existing_editor_sessions(
                            self.repo,
                            files=files,
                            include_members=True,
                            include_active=True,
                        )
                        self.delivery.emit(version, "invalid_count", invalid_count)
                        rows = session_rows(jobs_found)
                    else:
                        rows = list_storage(
                            self.repo, self.kind, self.output, self.scope, jobs=jobs
                        )
                    self.delivery.emit(version, "rows", rows)
                self.delivery.emit(version, "done", None)
            except Exception as exc:
                self.delivery.emit(version, "error", str(exc))

        threading.Thread(target=scan, daemon=True).start()

    def request_sizes(self):
        self.want_sizes = True
        if self.kind != "sessions" or self.running or self.sizing or not self.rows:
            return
        pending = [row for row in self.rows if row["id"] not in self.sizes]
        if not pending:
            return
        version = self.version
        self.sizing = True

        def calculate():
            try:
                batch = {}
                for row in pending:
                    if version != self.version:
                        return
                    batch[row["id"]] = job_size(row, self.repo)
                    if len(batch) >= 8:
                        self.delivery.emit(version, "sizes", batch)
                        batch = {}
                if batch:
                    self.delivery.emit(version, "sizes", batch)
            finally:
                self.delivery.emit(version, "sizes_done", None)

        threading.Thread(target=calculate, daemon=True).start()

    def _receive(self, version, mode, value):
        if version != self.version:
            return
        if mode == "invalid_count":
            self.invalid_count = value
        elif mode in {"rows", "recent"}:
            self.partial = mode == "recent"
            self.rows, self.sizes = value, {}
            self.loaded = time.monotonic()
        elif mode == "sizes":
            self.sizes.update(value)
        elif mode == "sizes_done":
            self.sizing = False
        elif mode in {"error", "done"}:
            self.running = False
            if mode == "error":
                self.loaded = 0
            elif self.want_sizes:
                self.request_sizes()
        self.updated.emit(mode, value)
