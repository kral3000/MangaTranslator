"""File discovery and relocation for saved editor sessions."""

import base64
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

_UPLOADED_RELATIVE_PATH_PREFIX = "__MT_RELATIVE_PATH__"


session_revision = 0


def _is_temporary_upload_path(path: Path) -> bool:
    try:
        resolved = Path(path).resolve()
    except Exception:
        resolved = Path(path)
    lowered_parts = {part.lower() for part in resolved.parts}
    if "gradio" in lowered_parts:
        return True
    return resolved.name.startswith(_UPLOADED_RELATIVE_PATH_PREFIX)


def _decode_uploaded_relative_path(file_name: str) -> Optional[Path]:
    if not file_name.startswith(_UPLOADED_RELATIVE_PATH_PREFIX):
        return None
    suffix = Path(file_name).suffix
    encoded_end = -len(suffix) if suffix else None
    encoded = file_name[len(_UPLOADED_RELATIVE_PATH_PREFIX) : encoded_end]
    try:
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    clean_parts = [
        part
        for part in PurePosixPath(decoded.replace("\\", "/")).parts
        if part not in ("", ".", "..", "/")
    ]
    return Path(*clean_parts) if clean_parts else None


def _resolve_stored_path(session_path: Path, value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    stored = Path(text)
    if stored.is_absolute() and stored.exists():
        return stored
    repo_root = Path(__file__).resolve().parents[1]
    for parent in (session_path.parent, *session_path.parents):
        if parent.name.casefold() in {"editor_sessions", "editor_imports"}:
            repo_root = parent.parent
            break
    relocated = None
    if stored.is_absolute():
        parts = list(stored.parts)
        for index, part in enumerate(parts):
            if part.casefold() in {"editor_sessions", "editor_imports", "output"}:
                relocated = repo_root.joinpath(*parts[index:])
                break
    candidates = (
        relocated,
        repo_root / stored,
        session_path.parent / stored,
        session_path.parent / stored.name,
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return (repo_root / stored).resolve()


def _natural_sort_key(value: Any) -> Tuple[Tuple[int, Any], ...]:
    text = str(value or "").replace("\\", "/").casefold()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", text)
    )


def _stored_image_sort_text(
    session_path: Path,
    value: Any,
    collection_root: Optional[Path],
) -> str:
    image_path = _resolve_stored_path(session_path, value)
    if image_path is None:
        return ""

    decoded_relative = _decode_uploaded_relative_path(image_path.name)
    if decoded_relative is not None:
        return str(decoded_relative).replace("\\", "/")
    if _is_temporary_upload_path(image_path):
        return image_path.name

    if collection_root is not None:
        try:
            return str(image_path.resolve().relative_to(collection_root)).replace(
                "\\", "/"
            )
        except Exception:
            pass
    return str(image_path).replace("\\", "/")


def _editor_session_sort_text(
    path: Path,
    collection_root: Optional[Path] = None,
) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        display_path = str(session.get("source_display_path") or "").strip()
        if display_path:
            return display_path.replace("\\", "/").strip("/")

        for key in (
            "source_image_path",
            "translated_image_path",
            "overwrite_target_path",
        ):
            sort_text = _stored_image_sort_text(path, session.get(key), collection_root)
            if sort_text:
                return sort_text
    except Exception:
        pass

    if collection_root is not None:
        try:
            return str(path.resolve().relative_to(collection_root)).replace("\\", "/")
        except Exception:
            pass
    return path.name


def editor_session_context_label(session: Dict[str, Any]) -> str:
    profile = str(session.get("editor_character_profile_name") or "").strip()
    folder = str(session.get("editor_source_folder_name") or "").strip()
    if not folder:
        display_path = (
            str(session.get("source_display_path") or "")
            .replace("\\", "/")
            .strip("/")
        )
        display_parent = PurePosixPath(display_path).parent
        if display_path and str(display_parent) not in {"", "."}:
            folder = display_parent.name
    return " · ".join(dict.fromkeys(value for value in (profile, folder) if value))


def related_edit_session_paths(
    session_path: Any,
    session: Optional[Dict[str, Any]] = None,
    *,
    include_batch: bool = True,
) -> List[Path]:
    """Return linked batch/import sessions, with a directory fallback for older runs."""
    current = Path(str(session_path or "")).resolve()
    if include_batch:
        batch = editor_batch_for_session(current)
        if batch is not None:
            return batch["paths"]
    if session is None:
        try:
            with open(current, "r", encoding="utf-8") as f:
                session = json.load(f)
        except Exception:
            session = {}

    candidates: List[Path] = []
    collection_root: Optional[Path] = None
    for value in session.get("editor_session_group", []) or []:
        resolved = _resolve_stored_path(current, value)
        if resolved is not None and resolved.is_file():
            candidates.append(resolved.resolve())

    if not candidates:
        if current.parent.name.startswith("imported_"):
            collection_root = current.parent
        else:
            for parent in (current.parent, *current.parents):
                if (
                    re.fullmatch(r"\d{8}_\d{6}", parent.name)
                    and parent.parent.name.casefold() == "output"
                ):
                    collection_root = parent
                    break
        if collection_root is not None:
            candidates = list(collection_root.rglob("*_edit_session.json"))

    candidates.append(current)
    unique: Dict[str, Path] = {}
    positions: Dict[str, int] = {}
    for index, path in enumerate(candidates):
        resolved = path.resolve()
        if resolved.is_file():
            key = str(resolved).casefold()
            unique.setdefault(key, resolved)
            positions.setdefault(key, index)

    def session_sort_key(path: Path):
        path_key = str(path.resolve()).casefold()
        return (
            _natural_sort_key(_editor_session_sort_text(path, collection_root)),
            positions.get(path_key, 0),
            _natural_sort_key(path.name),
        )

    ordered = sorted(unique.values(), key=session_sort_key)
    if current in ordered:
        return ordered
    return [current, *ordered]


def editor_batch_for_session(session_path: Path):
    from .editor_batch_progress import find_editor_batch

    root = next((parent for parent in session_path.parents if parent.name == "editor_sessions"), None)
    if root is not None:
        return find_editor_batch(root.parent, session_path)
    return None
