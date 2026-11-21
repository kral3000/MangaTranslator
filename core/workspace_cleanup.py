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

from core.editor_paths import _resolve_stored_path, related_edit_session_paths

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
                    preview_path=str(min(images)),
                    paths=[str(path)],
                    size=_bytes,
                    modified=modified,
                )
            )
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: (-item["modified"], item["label"]))


def maintenance_records(repo):
    """Only reproducible browser caches and dated app logs have a cleanup contract."""
    root = Path(repo).resolve() / "output"
    result = []
    cutoff = time.time() - 7 * 86400
    for folder, prefix, title in (
        ("logs", r"MangaTranslatorKo", "번역기 실행 로그"),
        ("advanced_editor_logs", r"advanced_editor_(?:stdout|stderr)", "고급 편집기 실행 로그"),
    ):
        parent = root / folder
        if not parent.is_dir() or _is_link(parent):
            continue
        logs = []
        for path in parent.glob("*.log"):
            if re.fullmatch(prefix + r"_\d{8}_\d{6}\.log", path.name) and not _is_link(path) and path.is_file():
                info = path.stat()
                logs.append((path, info))
        # Keep the newest launch (including the matching stdout/stderr pair).
        newest = max((p.stem[-15:] for p, _ in logs), default="")
        old = [(p, info) for p, info in logs if info.st_mtime < cutoff and p.stem[-15:] != newest]
        if old:
            result.append(dict(id=str(parent), kind="maintenance", paths=[str(p) for p, _ in old],
                               title=title, state=f"{len(old)}개 로그", modified=max(st.st_mtime for _, st in old),
                               size=sum(st.st_size for _, st in old), label=title + " · 7일 지난 실행 로그"))
    base = root / "webview2-desktop" / "EBWebView" / "Default"
    for folder, title in (("Cache", "브라우저 페이지 캐시"), ("Code Cache", "브라우저 코드 캐시")):
        path = base / folder
        if not path.is_dir():
            continue
        try:
            _plain_path(path, [root])
            total = 0
            latest = path.stat().st_mtime
            for directory, folders, files in os.walk(path, followlinks=False):
                current = Path(directory)
                folders[:] = [d for d in folders if not _is_link(current / d)]
                for name in files:
                    f = current / name
                    if not _is_link(f):
                        stat = f.stat()
                        total += stat.st_size
                        latest = max(latest, stat.st_mtime)
            if total:
                result.append(dict(id=str(path), kind="maintenance", paths=[str(path)], title=title,
                                   state="재생성 가능", modified=latest, size=total, label=title))
        except (OSError, ValueError):
            continue
    return result


def _browser_cache_in_use(repo):
    import psutil

    profile = os.path.normcase(str(repo / "output" / "webview2-desktop"))
    for proc in psutil.process_iter(["name"]):
        if (proc.info["name"] or "").casefold() != "msedgewebview2.exe":
            continue
        try:
            for arg in proc.cmdline():
                if profile in os.path.normcase(arg):
                    return True
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            return True
    return False


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
        "source_archive_path",
        "overwrite_target_path",
        "base_image_path",
        "original_snapshot_path",
        "rendered_snapshot_path",
        "inpainted_image_path",
        "inpaint_source_path",
        "inpaint_mask_path",
        "paint_overlay_path",
        "text_cover_mask_path",
        "restore_mask_path",
        "translated_image_path",
    )
    result = {key: data.get(key) for key in keys}
    for index, region in enumerate(data.get("regions") or []):
        if not isinstance(region, dict):
            continue
        for key, value in region.items():
            if key == "mask_path" or (key.startswith("ai_") and key.endswith("_path")):
                result[f"region/{index}/{key}"] = value
    return result


def _metadata(repo, plain_files):
    """Index ordinary managed paths while checking for links during this review."""
    records = {}
    for root in (repo / "editor_sessions", repo / "editor_imports"):
        if not root.is_dir() or _is_link(root):
            continue
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            info = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        if getattr(info, "st_file_attributes", 0) & 0x400 or stat.S_ISLNK(info.st_mode):
                            continue
                        plain_files.add(os.path.normcase(entry.path))
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(Path(entry.path))
                        elif entry.name.casefold().endswith("edit_session.json"):
                            path = Path(entry.path)
                            try:
                                records[path] = _session_files(path, info.st_mtime_ns, info.st_size)
                            except FileNotFoundError:
                                continue
                            except (ValueError, OSError) as exc:
                                raise ValueError(f"사용 중인 파일을 확인하려면 이 작업을 읽을 수 있어야 합니다: {path}") from exc
            except FileNotFoundError:
                continue
    return records


def _active_editor_sessions(repo):
    log = repo / "output" / "advanced_editor_logs" / "advanced_editor_lifecycle.log"
    if os.name != "nt" or not log.is_file():
        return []
    launched = {}
    queued_pages = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"PID=(\d+) (.*)", line)
        if not match:
            continue
        pid, event = int(match[1]), match[2]
        if event.startswith("launch started session="):
            launched[pid] = Path(event.split("session=", 1)[1])
        elif event.startswith("AI job uses session="):
            queued_pages.setdefault(pid, set()).add(Path(event.split("session=", 1)[1]))
        elif event.startswith(("application quitting", "event loop exited")):
            queued_pages.pop(pid, None)
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
                    active.extend(queued_pages.get(pid, ()))
            finally:
                kernel.CloseHandle(handle)
        elif ctypes.get_last_error() == 5:
            active.extend(related_edit_session_paths(path))
            active.extend(queued_pages.get(pid, ()))
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
    if kind not in {"sessions", "outputs", "history", "downloads", "maintenance"}:
        raise ValueError("알 수 없는 정리 대상입니다.")
    if _ACTIVE_JOBS:
        raise ValueError("번역이 진행 중입니다. 완료한 뒤 정리해 주세요.")
    wanted = set(selected or [])
    chosen = [record for record in records if record["id"] in wanted]
    if not chosen or len(chosen) != len(wanted):
        raise ValueError("정리할 작업을 목록에서 선택해 주세요.")
    repo, output_root = Path(repo).resolve(), Path(output_root).resolve()
    session_roots = [repo / "editor_sessions", repo / "editor_imports"]
    roots = {"maintenance": [repo / "output"], "sessions": session_roots, "outputs": [output_root],
             "downloads": [repo / "editor_sessions" / "batch_download_zips"],
             "history": [repo / "editor_history", *session_roots]}[kind]
    if kind == "maintenance":
        eligible = {r["id"]: set(r["paths"]) for r in maintenance_records(repo)}
        for row in chosen:
            if not row["paths"] or not set(row["paths"]).issubset(eligible.get(row["id"], set())):
                raise ValueError("정리 대상이 바뀌었습니다. 새로고침해 주세요.")
        if any(Path(row["id"]).parent.name == "Default" for row in chosen) and _browser_cache_in_use(repo):
            raise ValueError("캐시가 사용 중이거나 사용 여부를 확인할 수 없습니다. 번역기 창을 닫고 다시 시도해 주세요.")
    plain_files = set()
    metadata = _metadata(repo, plain_files)
    active = set(_active_editor_sessions(repo))
    selected_paths = {_plain_path(p, roots) for record in chosen for p in record["paths"]}
    from core.editor_batch_progress import read_editor_batches
    running = [batch for batch in read_editor_batches(repo) if batch["active"]]
    active.update(path for batch in running for path in batch["expected"])
    if kind == "outputs" and any(Path(batch["output_dir"]).resolve() in selected_paths for batch in running):
        raise ValueError("번역 중인 작업은 정리할 수 없습니다.")
    if kind == "downloads" and any(
        p.name != "successful_translations.zip" or p.parent.parent != roots[0]
        or not re.fullmatch(r"\d{8}_\d{6}_[0-9a-f]{6}", p.parent.name)
        for p in selected_paths
    ):
        raise ValueError("관리되는 다운로드 ZIP만 정리할 수 있습니다.")
    if kind == "history":
        from advanced_editor.lettering_history import read_record, state_label
        for item in chosen:
            path = Path(item["id"])
            record = read_record(path)
            if (not re.fullmatch(r"ai_sfx_[0-9a-f]{32}", path.name) or item["paths"] != [str(path)]
                    or item.get("state") in {"적용 중", "확인 필요", "생성 중"}
                    or state_label(record) == "생성 중"
                    or not (record or (path / "failure.json").is_file() or (path / "generation.json").is_file())):
                raise ValueError("사용 중이거나 확인되지 않은 식자 기록입니다.")
            image_id = ((record.get("patch") or {}).get("ai_image") or {}).get("id")
            if image_id and any(
                Path(str(value)).name == f"ai_{image_id}_processed_path.png"
                for data in metadata.values() for key, value in data.items()
                if value and key.endswith("/ai_processed_path")
            ):
                raise ValueError("적용 중인 식자 기록은 정리할 수 없습니다.")
            owner = record.get("session_path") or item.get("session")
            if (owner and Path(owner).resolve() in active) or (not owner and active):
                raise ValueError("열린 작업의 식자 기록은 편집기를 닫은 뒤 정리해 주세요.")
            from core.editor_batch_progress import _process_exists
            if record.get("pid") and _process_exists(int(record["pid"])):
                raise ValueError("이 식자를 생성한 편집기를 닫은 뒤 정리해 주세요.")
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
    targets = set(selected_paths)
    if kind == "sessions":
        for page in selected_paths:
            for key in ("assets_dir", "edited_image_path"):
                target = _resolve_stored_path(page, metadata.get(page, {}).get(key))
                if target:
                    targets.add(target.resolve())
    refs, sources, resolved_refs = {}, set(), {}
    for session, data in metadata.items():
        for key, value in data.items():
            if not value:
                continue
            stored = Path(str(value))
            cache_key = (None if stored.is_absolute() else session.parent, str(stored))
            if cache_key not in resolved_refs:
                if stored.is_absolute() and os.path.normcase(str(stored)) in plain_files:
                    resolved_refs[cache_key] = stored
                else:
                    resolved = _resolve_stored_path(session, value)
                    resolved_refs[cache_key] = resolved.resolve() if resolved else None
            linked = resolved_refs[cache_key]
            if linked:
                ancestors = (linked, *linked.parents)
                for ancestor in targets.intersection(ancestors):
                    refs.setdefault(ancestor, set()).add(session)
                    if key in {"source_image_path", "source_archive_path"}:
                        sources.add(ancestor)
    if kind in {"history", "downloads", "maintenance"} and any(refs.get(path) for path in selected_paths):
        raise ValueError("저장된 편집 작업에서 사용 중인 파일입니다.")
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
