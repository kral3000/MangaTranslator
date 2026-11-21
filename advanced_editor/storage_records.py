"""Read the existing session catalogue and enumerate storage only on demand."""

import hashlib
import os
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from advanced_editor.lettering_history import read_record, state_label, stored_draft
from core.editor_paths import _resolve_stored_path, editor_session_context_label
from core.editor_session_catalog import (
    _read_editor_session_summary,
    discover_existing_editor_sessions,
)
from core.workspace_cleanup import (
    _is_link,
    _session_files,
    maintenance_records,
    output_records,
)


def storage_size(paths):
    total = 0
    pending = list(dict.fromkeys(Path(p) for p in paths))
    while pending:
        path = pending.pop()
        try:
            if _is_link(path):
                continue
            if path.is_dir():
                pending.extend(path.iterdir())
            else:
                total += path.stat().st_size
        except OSError:
            continue
    return total


def session_assets(path):
    stat = path.stat()
    data = _session_files(path, stat.st_mtime_ns, stat.st_size)
    assets = _resolve_stored_path(path, data.get("assets_dir"))
    edited = _resolve_stored_path(path, data.get("edited_image_path"))
    return data, assets, edited


def job_size(row, repo):
    paths = set()
    for value in row["paths"]:
        page = Path(value)
        paths.add(page)
        try:
            _, assets, edited = session_assets(page)
        except (OSError, ValueError):
            continue
        for location in (assets, edited):
            if location and any(
                location.is_relative_to(repo / r)
                for r in ("editor_sessions", "editor_imports")
            ):
                paths.add(location)
    return storage_size(paths)


def _history_rows(repo, jobs, scope):
    owners, applied, folders = {}, {}, set()
    session_map = {}
    for job in jobs:
        for value in job["paths"]:
            page = Path(value)
            if scope and page not in scope:
                continue
            session_map[page] = job
            try:
                data, assets, _ = session_assets(page)
                if assets and not _is_link(assets):
                    owners[assets] = page
                    if scope:
                        folders.update(assets.glob("ai_sfx_*"))
                for key, original in data.items():
                    if not key.endswith("/ai_processed_path"):
                        continue
                    output = _resolve_stored_path(page, original)
                    if output and output.is_file():
                        prefix = key.rsplit("/", 1)[0]
                        index = int(prefix.split("/")[1])
                        before = _resolve_stored_path(
                            page, data.get(prefix + "/ai_original_path")
                        )
                        applied[output] = (page, index, before)
            except (OSError, ValueError):
                continue
    history = repo / "editor_history"
    if scope:
        for page in scope:
            if page.is_relative_to(repo):
                key = hashlib.sha256(
                    page.relative_to(repo).as_posix().casefold().encode()
                ).hexdigest()[:24]
                folders.update((history / key).glob("ai_sfx_*"))
    else:
        for root in (history, repo / "editor_sessions", repo / "editor_imports"):
            for parent, dirs, _ in os.walk(root, followlinks=False):
                for name in dirs[:]:
                    child = Path(parent) / name
                    if _is_link(child):
                        dirs.remove(name)
                    elif name.startswith("ai_sfx_"):
                        folders.add(child)
                        dirs.remove(name)
    rows = []
    for folder in folders:
        if not folder.is_dir() or _is_link(folder):
            continue
        record = read_record(folder)
        page = owners.get(folder.parent)
        if record.get("session_path"):
            page = _resolve_stored_path(folder, record["session_path"])
        if scope and page not in scope:
            continue
        generated = folder / "generated.png"
        output = folder / "processed.png"
        before = folder / "original.png"
        state = state_label(record)
        if not record:
            if (folder / "failure.json").is_file():
                state = "실패"
            elif page and (folder / "generation.json").is_file():
                state = "이전 생성"
        if not output.is_file():
            output = generated
        if state == "실패" and (folder / "reference.png").is_file():
            before = folder / "reference.png"
        draft = None
        if page in session_map and record.get("patch"):
            try:
                draft = stored_draft(folder, record, page)
            except (TypeError, ValueError, KeyError):
                pass
        if draft:
            output = Path(draft["patch"]["ai_processed_path"])
            before = Path(draft["patch"]["ai_original_path"])
        match = applied.pop(output, None)
        if draft and not match and page in session_map:
            image_id = draft["patch"]["ai_image"].get("id")
            if image_id:
                match = applied.pop(
                    next(
                        (
                            p
                            for p, (owner, _, _) in applied.items()
                            if owner == page
                            and p.name == f"ai_{image_id}_processed_path.png"
                        ),
                        None,
                    ),
                    None,
                )
        if match:
            page, index, _ = match
            state = "적용 중"
        else:
            index = record.get("region_index")
        title = session_map.get(page, {}).get("title", "")
        page_label = (
            page.stem.removesuffix("_translated_edit_session").removesuffix(
                "_edit_session"
            )
            if page
            else folder.parent.name
        )
        page_label = f"{page_label}페이지" if page_label.isdigit() else page_label
        label = " · ".join(
            filter(
                None,
                (
                    title,
                    page_label,
                    f"{index + 1}번 영역" if isinstance(index, int) else "",
                ),
            )
        )
        rows.append(
            dict(
                kind="history",
                id=str(folder),
                paths=[str(folder)],
                title=label,
                group_id=session_map.get(page, {}).get("id", ""),
                group_title=title or "연결되지 않은 기록",
                detail=page_label
                + (f" · {index + 1}번 영역" if isinstance(index, int) else ""),
                state=state,
                session=str(page) if page else "",
                modified=folder.stat().st_mtime,
                size=storage_size([folder]),
                before=before,
                after=output,
                draft=draft,
                error=record.get("error", ""),
            )
        )
    for output, (page, index, before) in applied.items():
        page_label = page.stem.removesuffix("_edit_session").removesuffix("_translated")
        if page_label.isdigit():
            page_label += "페이지"
        rows.append(
            dict(
                kind="history",
                id=str(output),
                paths=[str(output)],
                title=f"{session_map[page]['title']} · {page.stem.split('_translated')[0]} · {index + 1}번 영역",
                state="적용 중",
                group_id=session_map[page]["id"],
                group_title=session_map[page]["title"],
                detail=f"{page_label} · {index + 1}번 영역",
                session=str(page),
                modified=output.stat().st_mtime,
                size=output.stat().st_size,
                before=before,
                after=output,
            )
        )
    return rows


def _download_status(archive, output):
    """Check the current output folder without unpacking or reading any page images."""
    directory = output / archive.parent.name.rsplit("_", 1)[0]
    try:
        with ZipFile(archive) as z:
            members = [
                Path(entry.filename) for entry in z.infolist() if not entry.is_dir()
            ]
        if not members or any(p.anchor or ".." in p.parts for p in members):
            return "출력본 확인 필요"
        present = sum((directory / p).is_file() for p in members)
    except (OSError, BadZipFile, ValueError):
        return "출력본 확인 필요"
    return f"출력본 {present}/{len(members)}장"


def list_storage(repo, kind, output, scope=None, *, jobs=None):
    if kind == "maintenance":
        return maintenance_records(repo)
    if scope and kind == "history":
        jobs = []
        for page in sorted(scope):
            if page.is_file():
                stat = page.stat()
                data = _read_editor_session_summary(
                    page, stat.st_mtime_ns, stat.st_size
                )
                jobs.append(
                    dict(
                        id=str(min(scope)),
                        title=editor_session_context_label(data) or page.parent.name,
                        paths=[str(page)],
                    )
                )
        return sorted(
            _history_rows(repo, jobs, scope), key=lambda row: -row["modified"]
        )
    if kind in {"history", "sessions"}:
        if jobs is None:
            _, _, jobs = discover_existing_editor_sessions(
                repo, include_members=True, include_active=True
            )
        if kind == "sessions":
            return session_rows(jobs)
        return sorted(
            _history_rows(repo, jobs, scope), key=lambda row: -row["modified"]
        )
    rows = [
        dict(
            item, title=item["label"].split("  ·")[0], state="번역 결과", kind="outputs"
        )
        for item in output_records(output)
    ]
    for path in (repo / "editor_sessions" / "batch_download_zips").glob(
        "*/successful_translations.zip"
    ):
        if path.is_file() and not _is_link(path.parent):
            st = path.stat()
            rows.append(
                dict(
                    kind="downloads",
                    id=str(path),
                    paths=[str(path)],
                    title=f"{path.parent.name} · 다운로드 ZIP",
                    label=f"{path.parent.name} · 다운로드 ZIP · 설정된 출력 폴더에서 같은 이름으로 찾은 이미지 수",
                    state=_download_status(path, output),
                    size=st.st_size,
                    modified=st.st_mtime,
                )
            )
    return sorted(rows, key=lambda row: -row["modified"])


def session_rows(jobs):
    return [
        dict(
            job,
            kind="sessions",
            state="번역 중" if job.get("translating") else f"{job['page_count']}장",
        )
        for job in jobs
    ]
