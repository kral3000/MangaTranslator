"""Saved editor job discovery shared by the native editor and WebUI."""

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@lru_cache(maxsize=4096)
def _read_editor_session_summary(path: Path, modified_ns: int, size: int) -> Dict[str, Any]:
    """Cache only list metadata; a file revision invalidates its entry."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid session")
    summary = {
        key: data.get(key)
        for key in (
            "source_display_path", "translated_image_path", "source_image_path",
            "editor_session_group", "editor_character_profile_name",
            "editor_source_folder_name",
        )
    }
    summary["region_count"] = len(data.get("regions") or [])
    return summary


def discover_existing_editor_sessions(
    repo_root: Optional[Path] = None,
    *, include_members: bool = False, recent_files: Optional[int] = None,
) -> tuple[List[Tuple[str, str]], int] | tuple[List[Tuple[str, str]], int, List[Dict[str, Any]]]:
    from core.editor_session import (
        editor_session_context_label,
        related_edit_session_paths,
    )

    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    session_paths: Dict[str, Path] = {}
    for folder_name in ("editor_sessions", "editor_imports"):
        root = repo_root / folder_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if path.name.casefold().endswith("edit_session.json"):
                resolved = path.resolve()
                session_paths.setdefault(str(resolved).casefold(), resolved)

    sessions: Dict[str, tuple[Path, Dict[str, Any], float, str]] = {}
    invalid_count = 0
    paths = list(session_paths.values())
    if recent_files is not None:
        def modified(path: Path) -> int:
            try:
                return path.stat().st_mtime_ns
            except OSError:
                return 0

        paths = sorted(paths, key=modified, reverse=True)[:recent_files]
    for path in paths:
        try:
            stat = path.stat()
            data = _read_editor_session_summary(path, stat.st_mtime_ns, stat.st_size)
            display_name = str(data.get("source_display_path") or "").strip()
            if not display_name:
                source = str(
                    data.get("translated_image_path")
                    or data.get("source_image_path")
                    or ""
                ).strip()
                display_name = Path(source).name if source else path.stem
            modified = stat.st_mtime
            sessions[str(path).casefold()] = (path, data, modified, display_name)
        except Exception:
            invalid_count += 1

    def group_member_key(value: Any) -> str:
        parts = Path(str(value)).parts
        for index, part in enumerate(parts):
            if part.casefold() in {"editor_sessions", "editor_imports"}:
                parts = parts[index:]
                break
        return "/".join(part.casefold() for part in parts)

    groups: Dict[tuple[str, ...], List[Path]] = {}
    group_positions: Dict[tuple[str, ...], Dict[str, int]] = {}
    related_cache: Dict[tuple[str, ...], List[Path]] = {}
    for path, data, _modified, _display_name in sessions.values():
        stored_group = tuple(
            group_member_key(value)
            for value in data.get("editor_session_group", []) or []
        )
        if stored_group:
            groups.setdefault(stored_group, []).append(path)
            group_positions.setdefault(
                stored_group,
                {member: index for index, member in enumerate(stored_group)},
            )
            continue
        # Legacy imports share one directory; discovering it per page rereads
        # every member for sorting, turning a large import into quadratic I/O.
        collection = path.parent if path.parent.name.startswith("imported_") else path
        cache_key = (str(collection).casefold(),)
        if cache_key in related_cache:
            continue
        related_cache[cache_key] = [
            candidate
            for candidate in related_edit_session_paths(path, data)
            if str(candidate).casefold() in sessions
        ]
        related = related_cache[cache_key]
        key = tuple(str(candidate).casefold() for candidate in related) or (
            str(path).casefold(),
        )
        groups.setdefault(key, related or [path])

    choices: List[Tuple[float, str, str]] = []
    cleanup_records = {}
    for key, paths in groups.items():
        positions = group_positions.get(key)
        if positions:
            paths = sorted(
                paths,
                key=lambda item: positions.get(group_member_key(item), len(positions)),
            )
        records = [sessions[str(path).casefold()] for path in paths]
        modified = max(record[2] for record in records)
        modified_text = time.strftime("%Y-%m-%d %H:%M", time.localtime(modified))
        if recent_files is not None and positions and len(positions) > 1:
            label = f"{records[0][3]}  ·  여러 장  ·  {modified_text}"
        elif len(records) > 1:
            first_name = records[0][3]
            last_name = records[-1][3]
            range_text = (
                first_name
                if first_name == last_name
                else f"{first_name} → {last_name}"
            )
            label = f"{range_text}  ·  묶음 {len(records)}장  ·  {modified_text}"
        else:
            path, data, _modified, display_name = records[0]
            label = (
                f"{display_name}  ·  영역 {data['region_count']}개"
                f"  ·  {modified_text}"
            )
        context = next(
            (
                value
                for record in records
                if (value := editor_session_context_label(record[1]))
            ),
            "",
        )
        if context:
            label += f"  ·  〔{context}〕"
        choices.append((modified, label, str(paths[0])))
        if include_members:
            cleanup_records[str(paths[0])] = dict(
                id=str(paths[0]), label=label,
                paths=[str(path) for path in paths], modified=modified,
            )

    choices.sort(key=lambda item: (-item[0], item[1].casefold()))
    result = [(label, path) for _modified, label, path in choices]
    if include_members:
        return result, invalid_count, [cleanup_records[path] for _label, path in result]
    return result, invalid_count
