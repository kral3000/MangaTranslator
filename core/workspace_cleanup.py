"""Reviewable, recoverable cleanup of generated editor and translation files."""

import ctypes
import json
import os
import re
import stat
import threading
import time
from datetime import datetime, timedelta, timezone
from datetime import time as daytime
from functools import lru_cache, wraps
from pathlib import Path

from core.editor_session import _resolve_stored_path, related_edit_session_paths

_LOCK = threading.RLock()
_ACTIVE_JOBS = 0
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def workspace_job(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global _ACTIVE_JOBS
        with _LOCK:
            _ACTIVE_JOBS += 1
        try:
            return fn(*args, **kwargs)
        finally:
            with _LOCK:
                _ACTIVE_JOBS -= 1

    return wrapped


def _is_link(path):
    info = path.lstat()
    return bool(getattr(info, "st_file_attributes", 0) & 0x400) or stat.S_ISLNK(
        info.st_mode
    )


def _plain_path(path, roots):
    path = Path(path).absolute()
    roots = [Path(root).resolve() for root in roots]
    resolved = path.resolve()
    root = next(
        (root for root in roots if resolved != root and resolved.is_relative_to(root)),
        None,
    )
    if root is None:
        raise ValueError("정리 범위를 벗어난 경로입니다.")
    for item in (path, *path.parents):
        if item.exists() and _is_link(item):
            raise ValueError("연결된 폴더나 파일은 정리할 수 없습니다.")
        if item == root:
            break
    return resolved


def output_records(root):
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    records = []
    for path in root.iterdir():
        single = path.is_file() and bool(
            re.fullmatch(
                r".+_translated_\d{8}_\d{6}(?:_plain)?\.(?:png|jpe?g|webp|bmp|tiff?)",
                path.name,
                re.I,
            )
        )
        batch = path.is_dir() and bool(re.fullmatch(r"\d{8}_\d{6}", path.name))
        if not (single or batch):
            continue
        try:
            _plain_path(path, [root])
            state, _bytes = _tree_state(path, [root])
            images = [
                Path(p)
                for p, _mtime, _size, is_dir in state
                if not is_dir and Path(p).suffix.lower() in _IMAGE_EXTENSIONS
            ]
            if not images or (
                batch and not any("_translated" in p.stem for p in images)
            ):
                continue
            modified = max(mtime for _p, mtime, _size, _is_dir in state) / 1_000_000_000
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(modified))
            records.append(
                dict(
                    id=str(path),
                    label=f"{path.name}  ·  이미지 {len(images)}개  ·  {stamp}",
                    paths=[str(path)],
                    modified=modified,
                )
            )
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: (-item["modified"], item["label"]))


def filtered_choices(records, days, query, now=None, *, start_date=None, end_date=None):
    days = int(days or 0)
    cutoff = (time.time() if now is None else now) - max(days, 0) * 86400
    start, end = None, None
    if days == -1:
        # Calendar days follow the Korean UI, independent of the server timezone.
        korean_time = timezone(timedelta(hours=9))
        try:
            if start_date:
                day = datetime.fromisoformat(str(start_date)).date()
                start = datetime.combine(day, daytime.min, korean_time).timestamp()
            if end_date:
                day = datetime.fromisoformat(str(end_date)).date() + timedelta(days=1)
                end = datetime.combine(day, daytime.min, korean_time).timestamp()
        except (ValueError, OverflowError) as exc:
            raise ValueError("수정일을 올바른 날짜로 선택해 주세요.") from exc
        if start is not None and end is not None and start >= end:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
    query = str(query or "").strip().casefold()
    return [
        (item["label"], item["id"])
        for item in records
        if (days <= 0 or item["modified"] < cutoff)
        and (start is None or item["modified"] >= start)
        and (end is None or item["modified"] < end)
        and query in item["label"].casefold()
    ]


@lru_cache(maxsize=4096)
def _session_files(path, modified_ns, size):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid session")
    keys = (
        "assets_dir",
        "edited_image_path",
        "source_image_path",
        "overwrite_target_path",
        "base_image_path",
        "original_snapshot_path",
        "rendered_snapshot_path",
        "inpainted_image_path",
        "inpaint_source_path",
        "inpaint_mask_path",
        "paint_overlay_path",
        "restore_mask_path",
        "translated_image_path",
    )
    return {key: data.get(key) for key in keys}


def _metadata(repo):
    records = {}
    for root in (repo / "editor_sessions", repo / "editor_imports"):
        if not root.is_dir() or _is_link(root):
            continue
        candidates = []
        for parent, folders, files in os.walk(root, followlinks=False):
            kept = []
            for folder in folders:
                try:
                    if not _is_link(Path(parent) / folder):
                        kept.append(folder)
                except (OSError, ValueError):
                    pass
            folders[:] = kept
            candidates.extend(
                Path(parent) / name
                for name in files
                if name.casefold().endswith("edit_session.json")
            )
        for path in candidates:
            try:
                if _is_link(path):
                    continue
                path = path.resolve()
                stat = path.stat()
                records[path] = _session_files(path, stat.st_mtime_ns, stat.st_size)
            except (OSError, ValueError):
                continue
    return records


def _active_editor_sessions(repo):
    log = repo / "output" / "advanced_editor_logs" / "advanced_editor_lifecycle.log"
    if os.name != "nt" or not log.is_file():
        return []
    launched = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"PID=(\d+) (.*)", line)
        if not match:
            continue
        pid, event = int(match[1]), match[2]
        if event.startswith("launch started session="):
            launched[pid] = Path(event.split("session=", 1)[1])
        elif event.startswith(("application quitting", "event loop exited")):
            launched.pop(pid, None)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    active = []
    for pid, path in launched.items():
        handle = kernel.OpenProcess(0x100000, False, pid)
        if handle:
            try:
                if kernel.WaitForSingleObject(handle, 0) == 258:
                    active.extend(related_edit_session_paths(path))
            finally:
                kernel.CloseHandle(handle)
        elif ctypes.get_last_error() == 5:
            active.extend(related_edit_session_paths(path))
    return active


def _tree_state(path, roots):
    path = _plain_path(path, roots)
    nodes = [path]
    if path.is_dir():
        for parent, folders, files in os.walk(path, followlinks=False):
            # Validate each directory before os.walk can descend into it.
            nodes.extend(
                _plain_path(Path(parent) / name, roots) for name in (*folders, *files)
            )
    result, size = [], 0
    for node in sorted(nodes):
        node = _plain_path(node, roots)
        stat = node.stat()
        result.append((str(node), stat.st_mtime_ns, stat.st_size, node.is_dir()))
        if node.is_file():
            size += stat.st_size
    return result, size


def prepare_cleanup(kind, records, selected, repo, output_root):
    if kind not in {"sessions", "outputs"}:
        raise ValueError("알 수 없는 정리 대상입니다.")
    if _ACTIVE_JOBS:
        raise ValueError("번역이 진행 중입니다. 완료한 뒤 정리해 주세요.")
    wanted = set(selected or [])
    chosen = [record for record in records if record["id"] in wanted]
    if not chosen or len(chosen) != len(wanted):
        raise ValueError("정리할 작업을 목록에서 선택해 주세요.")
    repo, output_root = Path(repo).resolve(), Path(output_root).resolve()
    roots = (
        [repo / "editor_sessions", repo / "editor_imports"]
        if kind == "sessions"
        else [output_root]
    )
    metadata = _metadata(repo)
    active = set(_active_editor_sessions(repo))
    selected_paths = {Path(p).resolve() for record in chosen for p in record["paths"]}
    for path in selected_paths:
        _plain_path(path, roots)
    if kind == "sessions" and active & selected_paths:
        raise ValueError(
            "열려 있는 편집 작업이 포함되어 있습니다. 편집기를 닫은 뒤 정리해 주세요."
        )
    if kind == "outputs":
        for session in active:
            for value in metadata.get(session, {}).values():
                linked = _resolve_stored_path(session, value) if value else None
                if linked and any(
                    linked == p or linked.is_relative_to(p) for p in selected_paths
                ):
                    raise ValueError(
                        "편집기에서 사용하는 결과물이 포함되어 있습니다. 편집기를 닫아 주세요."
                    )
    refs, sources, resolved_refs = {}, set(), {}
    for session, data in metadata.items():
        for key, value in data.items():
            if not value:
                continue
            stored = Path(str(value))
            cache_key = (None if stored.is_absolute() else session.parent, str(stored))
            if cache_key not in resolved_refs:
                resolved = _resolve_stored_path(session, value)
                resolved_refs[cache_key] = resolved.resolve() if resolved else None
            linked = resolved_refs[cache_key]
            if linked:
                ancestors = (linked, *linked.parents)
                for ancestor in ancestors:
                    refs.setdefault(ancestor, set()).add(session)
                if key == "source_image_path":
                    sources.update(ancestors)
    paths = set(selected_paths)
    if kind == "sessions":
        for session in selected_paths:
            data = metadata.get(session)
            if data is None:
                raise ValueError(
                    "세션이 변경되었거나 읽을 수 없습니다. 목록을 새로고침해 주세요."
                )
            for key in ("assets_dir", "edited_image_path"):
                value = data.get(key)
                candidate = _resolve_stored_path(session, value) if value else None
                if candidate is None or not candidate.exists():
                    continue
                expected = (
                    candidate.name == session.stem
                    if key == "assets_dir"
                    else (candidate.stem == session.stem + "_edited")
                )
                if not expected:
                    continue
                try:
                    candidate = _plain_path(candidate, roots)
                except ValueError:
                    continue
                if refs.get(candidate, set()) - selected_paths:
                    continue
                paths.add(candidate)
    paths = sorted(p for p in paths if not any(parent in paths for parent in p.parents))
    if sources.intersection(paths):
        raise ValueError("원본 이미지가 포함된 경로는 정리할 수 없습니다.")
    fingerprints, size = {}, 0
    for path in paths:
        state, bytes_ = _tree_state(path, roots)
        fingerprints[str(path)] = state
        size += bytes_
    return dict(
        kind=kind,
        records=chosen,
        selected=list(wanted),
        repo=str(repo),
        output_root=str(output_root),
        fingerprints=fingerprints,
        bytes=size,
        count=len(chosen),
        created=time.time(),
    )


def _trash(path):
    from PyQt6.QtCore import QFile

    ok, _destination = QFile.moveToTrash(str(path))
    if not ok:
        raise OSError(f"휴지통으로 보내지 못했습니다: {path.name}")


def execute_cleanup(plan):
    with _LOCK:
        if not plan or time.time() - plan["created"] > 300:
            raise ValueError(
                "삭제 확인이 만료되었습니다. 선택 정리를 다시 눌러 주세요."
            )
        current = prepare_cleanup(
            plan["kind"],
            plan["records"],
            plan["selected"],
            plan["repo"],
            plan["output_root"],
        )
        if current["fingerprints"] != plan["fingerprints"]:
            raise ValueError(
                "확인 후 파일이 변경되었습니다. 목록을 새로고침하고 다시 확인해 주세요."
            )
        completed, errors = [], []
        # Each target is reversible independently; never fall back to permanent deletion.
        for value in plan["fingerprints"]:
            try:
                _trash(Path(value))
                completed.append(value)
            except OSError as exc:
                errors.append(str(exc))
        return completed, errors
