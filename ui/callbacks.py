import json
import os
import re
import subprocess
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

import runtime_logging
from core.config import MangaTranslatorConfig, normalize_outside_text_processing_mode
from core.pipeline import (
    BATCH_EMBEDDED_OUTPUT_DIR_NAME,
    BATCH_PLAIN_OUTPUT_DIR_NAME,
    editor_plain_copy_path,
)
from core.tag_styles import tag_styles_to_table
from core.validation import (
    normalize_zip_file_input,
    validate_mutually_exclusive_modes,
    validate_zip_file,
)
from core.workspace_cleanup import workspace_job
from utils.exceptions import CancellationError, ValidationError
from utils.token_usage import format_usage

from . import character_profiles, logic, settings_manager, utils
from .cancellation import CancellationManager
from .korean import ko
from .ui_models import (
    UICleaningSettings,
    UIConfigState,
    UIDetectionSettings,
    UIGeneralSettings,
    UIGoogleRateLimitSettings,
    UIOutputSettings,
    UIOutsideTextSettings,
    UIRenderingSettings,
    UITranslationLLMSettings,
    UITranslationProviderSettings,
    flux_offload_flags,
    map_ui_to_backend_config,
)

ERROR_PREFIX = "오류: "
SUCCESS_PREFIX = "완료: "
IMAGE_EDITOR_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_MANA_DIRECTORY = Path(r"C:\Users\dongw\Documents\Mana")

CANCELLATION_MANAGER: Optional[CancellationManager] = None


def handle_runtime_log_refresh() -> str:
    return runtime_logging.read_log_tail()


def handle_runtime_log_toggle(is_open: bool):
    next_open = not bool(is_open)
    return (
        gr.update(visible=next_open),
        next_open,
        gr.update(active=next_open),
        gr.update(value="로그 닫기" if next_open else "실행 로그"),
        runtime_logging.read_log_tail() if next_open else gr.update(),
    )


def handle_open_runtime_log_directory() -> None:
    try:
        directory = runtime_logging.open_log_directory()
        gr.Info(f"로그 폴더를 열었습니다: {directory}")
    except OSError as exc:
        gr.Warning(f"로그 폴더를 열 수 없습니다: {exc}")


def _clean_error_message(message: Any) -> str:
    """Normalize error text for display and avoid duplicate prefixes/quotes."""
    try:
        text = str(message).strip()
    except Exception:
        text = ""

    if (len(text) >= 2) and (
        (text[0] == text[-1] == "'") or (text[0] == text[-1] == '"')
    ):
        text = text[1:-1].strip()

    text = ko(text)

    if text.startswith(ERROR_PREFIX):
        return text
    return f"{ERROR_PREFIX}{text}"


def _character_profile_updates(
    store: dict[str, Any], selected: str, *, editor_visible: bool, message: str
) -> tuple[Any, ...]:
    choices = character_profiles.profile_choices(store)
    entries = store["profiles"].get(selected, "")
    line_count = len([line for line in entries.splitlines() if line.strip()])
    status = message or (
        f"선택한 사전의 이름 표기 {line_count}개가 번역 지침에 자동 적용됩니다."
        if selected
        else "작품을 선택하면 저장된 이름 표기가 번역 지침에 자동 적용됩니다."
    )
    return (
        gr.update(choices=choices, value=selected),
        gr.update(choices=choices, value=selected),
        gr.update(value=selected),
        gr.update(value=selected),
        gr.update(value=entries),
        gr.update(value=entries),
        gr.update(visible=editor_visible),
        gr.update(visible=editor_visible),
        status,
        status,
    )


def handle_character_profile_select(name: str) -> tuple[Any, ...]:
    store, _entries = character_profiles.select_profile(name)
    selected = store["active"]
    return _character_profile_updates(
        store, selected, editor_visible=bool(selected), message=""
    )


def handle_new_character_profile() -> tuple[Any, ...]:
    store, _entries = character_profiles.select_profile("")
    choices = character_profiles.profile_choices(store)
    message = "새 작품 이름과 이름 표기를 입력한 뒤 ‘사전 저장’을 누르세요."
    return (
        gr.update(choices=choices, value=""),
        gr.update(choices=choices, value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=True),
        gr.update(visible=True),
        message,
        message,
    )


def handle_save_character_profile(
    original_name: str, name: str, entries: str
) -> tuple[Any, ...]:
    try:
        store = character_profiles.save_profile(original_name, name, entries)
    except ValueError as exc:
        gr.Warning(str(exc))
        return tuple(gr.update() for _ in range(10))
    selected = store["active"]
    return _character_profile_updates(
        store,
        selected,
        editor_visible=True,
        message=f"‘{selected}’ 등장인물 사전을 저장했습니다.",
    )


def handle_delete_character_profile(name: str) -> tuple[Any, ...]:
    selected = str(name or "").strip()
    if not selected:
        gr.Warning("삭제할 작품을 먼저 선택해 주세요.")
        return tuple(gr.update() for _ in range(10))
    store = character_profiles.delete_profile(selected)
    return _character_profile_updates(
        store,
        "",
        editor_visible=False,
        message=f"‘{selected}’ 등장인물 사전을 삭제했습니다.",
    )


def _status_update(message: str) -> gr.update:
    """Create a gr.update() for the status textbox with calculated line count.

    Accounts for both explicit newlines and text wrapping (~120 chars per line).
    """
    message = ko(message)
    line_count = 0
    for line in message.split("\n"):
        # Each segment takes at least 1 line, plus additional lines for wrapping
        line_count += max(1, (len(line) + 119) // 120)
    return gr.update(value=message, lines=line_count)


def _download_file_update(path: Any = None) -> gr.update:
    return gr.update(value=str(path) if path else None, visible=True)


def _editor_button_update(path: Any = None) -> gr.update:
    return gr.update(visible=bool(path), interactive=bool(path))


def _plain_download_path(
    save_path: Optional[Path],
    backend_config: MangaTranslatorConfig,
) -> Optional[Path]:
    if not save_path:
        return None
    if not (
        bool(getattr(backend_config.output, "embed_editor_package", True))
        and bool(getattr(backend_config.output, "save_plain_image_copy", False))
    ):
        return None
    candidate = editor_plain_copy_path(save_path)
    return candidate if candidate.is_file() else None


def handle_browse_output_directory(current_value: str) -> gr.update:
    """Open a native folder picker and return the chosen absolute output path."""
    try:
        selected = _select_folder_with_explorer(current_value, "저장 폴더 선택")
        if selected:
            return gr.update(value=selected)
    except Exception as exc:
        gr.Warning(f"폴더 선택 창을 열 수 없습니다: {exc}")
    return gr.update(value=current_value)


def _select_folder_with_explorer(current_value: Any, title: str) -> str:
    return _select_path_with_explorer(current_value, title, "folder")


def _select_path_with_explorer(current_value: Any, title: str, mode: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    current_text = os.path.expandvars(str(current_value or "").strip())
    initial = Path(current_text).expanduser() if current_text else repo_root
    if not initial.is_absolute():
        initial = repo_root / initial
    if initial.is_file():
        initial = initial.parent
    if not initial.exists():
        initial = initial.parent if initial.parent.exists() else repo_root

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "advanced_editor.folder_picker",
            str(initial),
            title,
            mode,
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        timeout=3600,
    )
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(error_text)
    return (
        completed.stdout.strip().splitlines()[-1].strip()
        if completed.stdout.strip()
        else ""
    )


def handle_browse_batch_folder() -> tuple[gr.update, gr.update]:
    try:
        selected = _select_folder_with_explorer(
            DEFAULT_MANA_DIRECTORY, "번역할 만화 폴더 선택"
        )
        if selected:
            return (
                gr.update(value=selected),
                gr.update(value=None, label=f"선택한 폴더: {Path(selected).name}"),
            )
    except Exception as exc:
        gr.Warning(f"폴더 선택 창을 열 수 없습니다: {exc}")
    return gr.update(), gr.update()


def handle_browse_batch_zip() -> tuple[gr.update, gr.update]:
    try:
        selected = _select_path_with_explorer(
            DEFAULT_MANA_DIRECTORY, "번역할 ZIP 압축파일 선택", "zip"
        )
        if selected:
            return (
                gr.update(value=selected),
                gr.update(value=None, label=f"선택한 ZIP: {Path(selected).name}"),
            )
    except Exception as exc:
        gr.Warning(f"ZIP 선택 창을 열 수 없습니다: {exc}")
    return gr.update(), gr.update()


def _resolve_native_batch_inputs(
    input_files: Any,
    input_zip: Any,
    folder_path: Any,
    zip_path: Any,
) -> tuple[Any, Any]:
    folder_text = str(folder_path or "").strip()
    if folder_text:
        folder = Path(folder_text).expanduser().resolve()
        if not folder.is_dir():
            raise ValidationError(f"선택한 폴더를 찾을 수 없습니다: {folder}")
        input_files = sorted(
            (
                str(path)
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EDITOR_EXTENSIONS
            ),
            key=str.casefold,
        )
        if not input_files:
            raise ValidationError(f"선택한 폴더에 지원되는 이미지가 없습니다: {folder}")
    if str(zip_path or "").strip():
        input_zip = str(zip_path).strip()
    return input_files, input_zip


def _remember_image_editor_source_path(value: Any) -> str:
    path_text = str(value or "").strip()
    settings = settings_manager.get_saved_settings()
    if settings.get("image_editor_source_path") != path_text:
        settings["image_editor_source_path"] = path_text
        settings_manager.save_config(settings)
    return path_text


def handle_remember_image_editor_source_path(value: Any) -> None:
    _remember_image_editor_source_path(value)


def handle_browse_image_editor_folder(current_value: str) -> gr.update:
    """Open the native Windows Explorer folder picker for image import."""
    try:
        selected = _select_folder_with_explorer(
            current_value, "새 이미지 폴더 가져오기"
        )
        if selected:
            _remember_image_editor_source_path(selected)
            return gr.update(value=selected)
    except Exception as exc:
        gr.Warning(f"폴더 선택 창을 열 수 없습니다: {exc}")
    return gr.update(value=current_value)


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


def _discover_existing_editor_sessions(
    repo_root: Optional[Path] = None,
    *, include_members: bool = False,
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
    for path in session_paths.values():
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
        if len(records) > 1:
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


def handle_refresh_existing_editor_sessions() -> Tuple[gr.update, gr.update, gr.update]:
    choices, invalid_count = _discover_existing_editor_sessions()
    selected = choices[0][1] if choices else None
    if choices:
        message = f"저장된 편집 작업 {len(choices)}개를 찾았습니다. 여러 장 작업은 한 묶음으로 엽니다."
        if invalid_count:
            message += f" 읽을 수 없는 세션 {invalid_count}개는 제외했습니다."
    else:
        message = "저장된 편집 세션을 찾지 못했습니다. 새 이미지 폴더를 가져올 수 있습니다."
    return (
        gr.update(
            choices=choices,
            value=selected,
            visible=True,
            interactive=bool(choices),
        ),
        _editor_button_update(selected),
        _status_update(message),
    )


def handle_output_path_explorer_select(
    selected_path: Any,
    current_value: str,
) -> gr.update:
    """Apply a FileExplorer selection to the output directory textbox."""
    try:
        if isinstance(selected_path, (list, tuple)):
            selected = selected_path[0] if selected_path else ""
        else:
            selected = selected_path
        if not selected:
            return gr.update(value=current_value)

        path = Path(str(selected)).expanduser()
        if path.is_file():
            path = path.parent
        elif not path.exists() and path.suffix:
            path = path.parent
        return gr.update(value=str(path.resolve()))
    except Exception as exc:
        gr.Warning(f"선택한 경로를 적용할 수 없습니다: {exc}")
        return gr.update(value=current_value)


def _editor_session_choices_from_outputs(output_files: List[str]) -> List[Tuple[str, str]]:
    try:
        from core.editor_package import has_embedded_editor_package
        from core.editor_session import (
            default_edit_session_path,
            legacy_edit_session_path,
        )
    except Exception:
        return []

    choices: List[Tuple[str, str]] = []
    for file_path in output_files or []:
        output_path = Path(file_path)
        session_path = default_edit_session_path(output_path)
        if not session_path.exists():
            legacy_path = legacy_edit_session_path(output_path)
            if legacy_path.exists():
                session_path = legacy_path
        if session_path.exists():
            choices.append((output_path.name, str(session_path)))
        elif output_path.is_file() and has_embedded_editor_package(output_path):
            choices.append((output_path.name, str(output_path)))
    return choices


def _as_path_list(value: Any) -> List[Path]:
    if not value:
        return []
    raw_items = value if isinstance(value, (list, tuple)) else [value]
    paths: List[Path] = []
    for item in raw_items:
        candidate = getattr(item, "name", item)
        if candidate:
            paths.append(Path(str(candidate)))
    return paths


def _natural_path_key(path: Path) -> tuple:
    text = str(path).casefold()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", text)
    )


def _collect_image_editor_paths(
    image_files: Any,
    folder_files: Any,
    zip_file: Any,
) -> tuple[List[Path], Optional[Any], int, Dict[str, str]]:
    collected: List[Path] = []
    collected.extend(_as_path_list(image_files))
    collected.extend(_as_path_list(folder_files))

    zip_temp_dir_obj = None
    zip_count = 0
    zip_members: Dict[str, str] = {}
    if zip_file:
        zip_path_str = normalize_zip_file_input(zip_file)
        zip_path = validate_zip_file(zip_path_str)
        extracted_root, zip_temp_dir_obj = logic.extract_zip_to_temp(zip_path)
        zip_images = [
            path
            for path in extracted_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EDITOR_EXTENSIONS
        ]
        zip_count = len(zip_images)
        collected.extend(zip_images)
        zip_members.update(
            {
                str(path.resolve()).casefold(): path.relative_to(extracted_root).as_posix()
                for path in zip_images
            }
        )

    image_paths: List[Path] = []
    seen: set[str] = set()
    for path in sorted(collected, key=_natural_path_key):
        resolved = path.resolve()
        candidates = (
            sorted(
                (
                    child
                    for child in resolved.rglob("*")
                    if child.is_file()
                    and child.suffix.lower() in IMAGE_EDITOR_EXTENSIONS
                ),
                key=_natural_path_key,
            )
            if resolved.is_dir()
            else [resolved]
        )
        for candidate in candidates:
            key = str(candidate).casefold()
            if (
                key not in seen
                and candidate.is_file()
                and candidate.suffix.lower() in IMAGE_EDITOR_EXTENSIONS
            ):
                seen.add(key)
                image_paths.append(candidate)

    return image_paths, zip_temp_dir_obj, zip_count, zip_members


def _source_relative_name(path: Path, zip_members: Dict[str, str]) -> Path:
    member = zip_members.get(str(path.resolve()).casefold())
    if member:
        return Path(member)
    decoded = logic._decode_uploaded_relative_path(path.name)
    return decoded if decoded is not None else Path(path.name)


def _resolve_linked_folder_file(root: Path, relative_name: Path) -> Path:
    clean_parts = [
        part for part in relative_name.parts if part not in ("", ".", "..")
    ]
    relative_name = Path(*clean_parts) if clean_parts else Path(relative_name.name)
    candidates: List[Path] = [root / relative_name]
    if relative_name.parts and relative_name.parts[0].casefold() == root.name.casefold():
        candidates.append(root.parent / relative_name)
        if len(relative_name.parts) > 1:
            candidates.append(root / Path(*relative_name.parts[1:]))
    candidates.append(root / relative_name.name)

    unique_matches: Dict[str, Path] = {}
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            unique_matches.setdefault(str(resolved).casefold(), resolved)
    if not unique_matches:
        recursive_matches = [
            candidate.resolve()
            for candidate in root.rglob(relative_name.name)
            if candidate.is_file()
        ]
        for candidate in recursive_matches:
            unique_matches.setdefault(str(candidate).casefold(), candidate)
    if len(unique_matches) == 1:
        return next(iter(unique_matches.values()))
    if not unique_matches:
        raise FileNotFoundError(
            f"연결한 원본 폴더에서 파일을 찾을 수 없습니다: {relative_name.as_posix()}"
        )
    raise ValueError(
        f"연결한 원본 폴더에 같은 이름의 파일이 여러 개 있습니다: {relative_name.name}"
    )


def _resolve_image_editor_source_references(
    image_paths: List[Path],
    linked_source_path: Any,
    zip_members: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    linked_text = os.path.expandvars(str(linked_source_path or "").strip())
    if not linked_text:
        return {}
    linked = Path(linked_text).expanduser().resolve()
    if not linked.exists():
        raise FileNotFoundError(f"연결한 원본 위치를 찾을 수 없습니다: {linked}")

    references: Dict[str, Dict[str, str]] = {}
    if linked.is_file() and linked.suffix.lower() == ".zip":
        with zipfile.ZipFile(linked, "r") as archive:
            archive_names = {
                name.replace("\\", "/").lstrip("/"): name
                for name in archive.namelist()
                if not name.endswith("/")
            }
        for image_path in image_paths:
            relative_name = _source_relative_name(image_path, zip_members)
            normalized = relative_name.as_posix().lstrip("/")
            matched = archive_names.get(normalized)
            if matched is None:
                suffix_matches = [
                    original
                    for normalized_name, original in archive_names.items()
                    if normalized_name.casefold().endswith(
                        f"/{normalized.casefold()}"
                    )
                    or Path(normalized_name).name.casefold()
                    == relative_name.name.casefold()
                ]
                if len(suffix_matches) == 1:
                    matched = suffix_matches[0]
            if matched is None:
                raise FileNotFoundError(
                    f"연결한 원본 ZIP에서 파일을 찾을 수 없습니다: {normalized}"
                )
            references[str(image_path.resolve()).casefold()] = {
                "kind": "zip",
                "archive_path": str(linked),
                "member_path": matched.replace("\\", "/"),
            }
        return references

    if linked.is_file():
        if len(image_paths) != 1 or linked.suffix.lower() not in IMAGE_EDITOR_EXTENSIONS:
            raise ValueError(
                "원본 이미지 파일 하나를 연결할 때는 편집할 이미지도 한 장만 업로드해 주세요."
            )
        references[str(image_paths[0].resolve()).casefold()] = {
            "kind": "file",
            "path": str(linked),
        }
        return references

    for image_path in image_paths:
        relative_name = _source_relative_name(image_path, zip_members)
        target = _resolve_linked_folder_file(linked, relative_name)
        references[str(image_path.resolve()).casefold()] = {
            "kind": "file",
            "path": str(target),
        }
    return references


def _looks_like_empty_folder_placeholder(value: Any) -> bool:
    """Detect the 0-byte pseudo-file Gradio can emit for an unprocessed folder drop."""
    for path in _as_path_list(value):
        try:
            resolved = path.resolve()
            if (
                resolved.is_file()
                and resolved.stat().st_size == 0
                and not resolved.suffix
            ):
                return True
        except Exception:
            continue
    return False


def _image_editor_choice_label(source_path: Path, session_path: Path) -> str:
    try:
        with open(session_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        display = str(data.get("source_display_path") or "").strip()
        if display:
            return display
    except Exception:
        pass
    try:
        decoded = logic._decode_uploaded_relative_path(source_path.name)
        if decoded is not None:
            return str(decoded).replace("\\", "/")
    except Exception:
        pass
    return source_path.name


def _build_ui_state_from_args(args: tuple, is_batch: bool) -> UIConfigState:
    """Build UIConfigState from UI component arguments, handling single vs batch mode differences."""
    (
        confidence,
        conjoined_confidence,
        panel_confidence,
        seg_model_val,
        bubble_detector_model_val,
        conjoined_detection_checkbox_val,
        osb_text_verification_checkbox_val,
        use_panel_sorting_checkbox_val,
        thresholding_value,
        use_otsu_threshold,
        inpaint_colored_bubbles,
        colored_bubble_inpainting_size,
        colored_bubble_inpainting_precision,
        colored_bubble_mask_kernel_size,
        colored_bubble_mask_dilation_offset,
        roi_shrink_px,
        provider_selector,
        google_api_key,
        openai_api_key,
        anthropic_api_key,
        xai_api_key,
        deepseek_api_key,
        zai_api_key,
        moonshot_api_key,
        openrouter_api_key,
        openai_compatible_url_input,
        openai_compatible_api_key_input,
        config_model_name,
        temperature,
        top_p,
        top_k,
        max_tokens,
        config_reading_direction,
        config_translation_mode,
        ocr_method_val,
        input_language,
        output_language,
        font_dropdown,
        emphasis_font_pack_val,
        max_font_size,
        emphasis_max_font_size,
        min_font_size,
        line_spacing_mult,
        emphasis_line_spacing_mult,
        vertical_line_spacing_mult,
        vertical_emphasis_line_spacing_mult,
        vertical_font_size_mult,
        vertical_max_chars,
        vertical_emphasis_max_chars,
        tag_style_table_val,
        use_subpixel_rendering,
        font_hinting,
        use_ligatures,
        inside_light_text_color,
        inside_light_outline_color,
        inside_light_outline_width,
        inside_dark_text_color,
        inside_dark_outline_color,
        inside_dark_outline_width,
        outside_light_text_color,
        outside_light_outline_color,
        outside_light_outline_width,
        outside_dark_text_color,
        outside_dark_outline_color,
        outside_dark_outline_width,
        output_format,
        jpeg_quality,
        png_compression,
        output_directory,
        disable_auto_save,
        embed_editor_package_val,
        save_plain_image_copy_val,
        verbose,
        cleaning_only_toggle,
        upscaling_only_val,
        test_mode_toggle,
        enable_web_search_val,
        enable_code_execution_val,
        image_detail_val,
        media_resolution_val,
        media_resolution_bubbles_val,
        media_resolution_context_val,
        reasoning_effort_val,
        effort_val,
        verbosity_val,
        send_full_page_context_val,
        parallel_inpaint_translation_val,
        whiteout_conjoined_bubbles_val,
        upscale_method_val,
        bubble_min_side_pixels_val,
        context_image_max_side_pixels_val,
        osb_min_side_pixels_val,
        hyphenate_before_scaling_val,
        detach_trailing_punctuation_val,
        auto_vertical_text_val,
        hyphen_penalty_val,
        hyphenation_min_word_length_val,
        badness_exponent_val,
        padding_horizontal_pixels_val,
        padding_vertical_pixels_val,
        supersampling_factor_val,
        outside_text_enabled_val,
        outside_text_seed_val,
        outside_text_inpainting_method_val,
        outside_text_kontext_backend_val,
        outside_text_flux_offload_mode_val,
        outside_text_flux_num_inference_steps_val,
        outside_text_flux_kontext_prompt_val,
        outside_text_flux_klein_prompt_val,
        outside_text_flux_luminance_correction_val,
        outside_text_flux_upscale_small_crops_val,
        outside_text_flux_group_regions_val,
        outside_text_flux_residual_diff_threshold_val,
        outside_text_osb_residual_render_guard_action_val,
        outside_text_osb_confidence_val,
        outside_text_enable_page_number_filtering_val,
        outside_text_page_filter_margin_threshold_val,
        outside_text_page_filter_min_area_ratio_val,
        outside_text_huggingface_token_val,
        outside_text_osb_font_pack_val,
        outside_text_osb_emphasis_font_pack_val,
        outside_text_osb_max_font_size_val,
        outside_text_osb_emphasis_max_font_size_val,
        outside_text_osb_min_font_size_val,
        outside_text_osb_use_ligatures_val,
        outside_text_osb_outline_width_val,
        outside_text_osb_padding_pixels_val,
        outside_text_osb_line_spacing_val,
        outside_text_osb_vertical_line_spacing_val,
        outside_text_osb_auto_vertical_val,
        outside_text_osb_vertical_aspect_ratio_val,
        outside_text_osb_vertical_max_chars_val,
        outside_text_osb_tall_box_expand_aspect_ratio_val,
        outside_text_osb_tall_box_expand_width_percent_val,
        outside_text_osb_rotation_max_degrees_val,
        outside_text_osb_use_subpixel_rendering_val,
        outside_text_osb_font_hinting_val,
        outside_text_bbox_expansion_percent_val,
        outside_text_osb_render_expansion_narrow_multiplier_val,
        outside_text_osb_render_expansion_aspect_ratio_threshold_val,
        outside_text_osb_render_expansion_tiny_multiplier_val,
        outside_text_osb_render_expansion_area_threshold_percent_val,
        outside_text_text_box_proximity_ratio_val,
        image_upscale_mode_val,
        image_upscale_factor_val,
        image_upscale_model_val,
        auto_scale_val,
        batch_input_language,
        batch_output_language,
        batch_font_dropdown,
        character_profile_name,
        special_instructions_val,
        batch_special_instructions_val,
        batch_parallel_requests_val,
        batch_retry_failed_once_val,
        batch_previous_context_image_count_val,
        batch_previous_context_text_count_val,
        batch_previous_context_text_mode_val,
    ) = args

    final_input_language = batch_input_language if is_batch else input_language
    final_output_language = batch_output_language if is_batch else output_language
    final_font_pack = batch_font_dropdown if is_batch else font_dropdown
    base_special_instructions = (
        batch_special_instructions_val if is_batch else special_instructions_val
    )
    final_special_instructions = character_profiles.merge_with_special_instructions(
        character_profile_name, base_special_instructions
    )

    flux_cpu_offload, flux_low_vram = flux_offload_flags(
        outside_text_flux_offload_mode_val
    )
    outside_text_processing_mode = normalize_outside_text_processing_mode(
        outside_text_enabled_val
    )

    return UIConfigState(
        detection=UIDetectionSettings(
            confidence=confidence,
            conjoined_confidence=conjoined_confidence,
            panel_confidence=panel_confidence,
            seg_model=seg_model_val,
            bubble_detector_model=bubble_detector_model_val,
            conjoined_detection=conjoined_detection_checkbox_val,
            use_osb_text_verification=osb_text_verification_checkbox_val,
            use_panel_sorting=use_panel_sorting_checkbox_val,
        ),
        cleaning=UICleaningSettings(
            thresholding_value=thresholding_value,
            use_otsu_threshold=use_otsu_threshold,
            roi_shrink_px=int(roi_shrink_px),
            inpaint_colored_bubbles=inpaint_colored_bubbles,
            colored_bubble_inpainting_size=int(
                colored_bubble_inpainting_size
            ),
            colored_bubble_inpainting_precision=(
                colored_bubble_inpainting_precision
            ),
            colored_bubble_mask_kernel_size=int(
                colored_bubble_mask_kernel_size
            ),
            colored_bubble_mask_dilation_offset=int(
                colored_bubble_mask_dilation_offset
            ),
        ),
        outside_text=UIOutsideTextSettings(
            enabled=outside_text_processing_mode != "off",
            processing_mode=outside_text_processing_mode,
            seed=int(outside_text_seed_val),
            huggingface_token=outside_text_huggingface_token_val,
            inpainting_method=outside_text_inpainting_method_val,
            kontext_backend=outside_text_kontext_backend_val,
            flux_cpu_offload=flux_cpu_offload,
            flux_low_vram=flux_low_vram,
            flux_num_inference_steps=int(outside_text_flux_num_inference_steps_val),
            flux_kontext_prompt=outside_text_flux_kontext_prompt_val,
            flux_klein_prompt=outside_text_flux_klein_prompt_val,
            flux_luminance_correction=outside_text_flux_luminance_correction_val,
            flux_upscale_small_crops=outside_text_flux_upscale_small_crops_val,
            flux_group_regions=outside_text_flux_group_regions_val,
            flux_residual_diff_threshold=float(
                outside_text_flux_residual_diff_threshold_val
            ),
            osb_residual_render_guard_action=outside_text_osb_residual_render_guard_action_val,
            osb_confidence=float(outside_text_osb_confidence_val),
            enable_page_number_filtering=outside_text_enable_page_number_filtering_val,
            page_filter_margin_threshold=float(
                outside_text_page_filter_margin_threshold_val
            ),
            page_filter_min_area_ratio=float(
                outside_text_page_filter_min_area_ratio_val
            ),
            osb_font_dir=outside_text_osb_font_pack_val,
            osb_emphasis_font_dir=outside_text_osb_emphasis_font_pack_val,
            osb_max_font_size=int(outside_text_osb_max_font_size_val),
            osb_emphasis_max_font_size=int(
                outside_text_osb_emphasis_max_font_size_val
            ),
            osb_min_font_size=int(outside_text_osb_min_font_size_val),
            osb_use_ligatures=outside_text_osb_use_ligatures_val,
            osb_outline_width=float(outside_text_osb_outline_width_val),
            osb_padding_pixels=float(outside_text_osb_padding_pixels_val),
            osb_line_spacing=float(outside_text_osb_line_spacing_val),
            osb_vertical_line_spacing=float(
                outside_text_osb_vertical_line_spacing_val
            ),
            osb_auto_vertical=outside_text_osb_auto_vertical_val,
            osb_vertical_aspect_ratio=float(
                outside_text_osb_vertical_aspect_ratio_val
            ),
            osb_vertical_max_chars=int(outside_text_osb_vertical_max_chars_val),
            osb_tall_box_expand_aspect_ratio=float(
                outside_text_osb_tall_box_expand_aspect_ratio_val
            ),
            osb_tall_box_expand_width_percent=float(
                outside_text_osb_tall_box_expand_width_percent_val
            ),
            osb_rotation_max_degrees=float(
                outside_text_osb_rotation_max_degrees_val
            ),
            osb_use_subpixel_rendering=outside_text_osb_use_subpixel_rendering_val,
            osb_font_hinting=outside_text_osb_font_hinting_val,
            bbox_expansion_percent=float(outside_text_bbox_expansion_percent_val),
            osb_render_expansion_narrow_multiplier=float(
                outside_text_osb_render_expansion_narrow_multiplier_val
            ),
            osb_render_expansion_tiny_multiplier=float(
                outside_text_osb_render_expansion_tiny_multiplier_val
            ),
            osb_render_expansion_aspect_ratio_threshold=float(
                outside_text_osb_render_expansion_aspect_ratio_threshold_val
            ),
            osb_render_expansion_area_ratio_threshold=float(
                outside_text_osb_render_expansion_area_threshold_percent_val
            )
            / 100.0,
            text_box_proximity_ratio=float(outside_text_text_box_proximity_ratio_val),
        ),
        provider_settings=UITranslationProviderSettings(
            provider=provider_selector,
            google_api_key=google_api_key,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            xai_api_key=xai_api_key,
            deepseek_api_key=deepseek_api_key,
            zai_api_key=zai_api_key,
            moonshot_api_key=moonshot_api_key,
            openrouter_api_key=openrouter_api_key,
            openai_compatible_url=openai_compatible_url_input,
            openai_compatible_api_key=openai_compatible_api_key_input,
        ),
        llm_settings=UITranslationLLMSettings(
            model_name=config_model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reading_direction=config_reading_direction,
            translation_mode=config_translation_mode,
            ocr_method=ocr_method_val,
            send_full_page_context=send_full_page_context_val,
            parallel_inpaint_translation=parallel_inpaint_translation_val,
            whiteout_conjoined_bubbles=whiteout_conjoined_bubbles_val,
            upscale_method=upscale_method_val,
            bubble_min_side_pixels=bubble_min_side_pixels_val,
            context_image_max_side_pixels=context_image_max_side_pixels_val,
            osb_min_side_pixels=osb_min_side_pixels_val,
            special_instructions=final_special_instructions,
        ),
        rendering=UIRenderingSettings(
            emphasis_font_dir=emphasis_font_pack_val,
            max_font_size=max_font_size,
            emphasis_max_font_size=emphasis_max_font_size,
            min_font_size=min_font_size,
            line_spacing_mult=line_spacing_mult,
            emphasis_line_spacing_mult=emphasis_line_spacing_mult,
            vertical_line_spacing_mult=vertical_line_spacing_mult,
            vertical_emphasis_line_spacing_mult=vertical_emphasis_line_spacing_mult,
            vertical_font_size_mult=vertical_font_size_mult,
            vertical_max_chars=int(vertical_max_chars),
            vertical_emphasis_max_chars=int(vertical_emphasis_max_chars),
            tag_style_table=tag_style_table_val,
            use_subpixel_rendering=use_subpixel_rendering,
            font_hinting=font_hinting,
            use_ligatures=use_ligatures,
            inside_light_text_color=inside_light_text_color,
            inside_light_outline_color=inside_light_outline_color,
            inside_light_outline_width=float(inside_light_outline_width),
            inside_dark_text_color=inside_dark_text_color,
            inside_dark_outline_color=inside_dark_outline_color,
            inside_dark_outline_width=float(inside_dark_outline_width),
            outside_light_text_color=outside_light_text_color,
            outside_light_outline_color=outside_light_outline_color,
            outside_light_outline_width=float(outside_light_outline_width),
            outside_dark_text_color=outside_dark_text_color,
            outside_dark_outline_color=outside_dark_outline_color,
            outside_dark_outline_width=float(outside_dark_outline_width),
            hyphenate_before_scaling=hyphenate_before_scaling_val,
            detach_trailing_punctuation=detach_trailing_punctuation_val,
            auto_vertical_text=auto_vertical_text_val,
            hyphen_penalty=hyphen_penalty_val,
            hyphenation_min_word_length=hyphenation_min_word_length_val,
            badness_exponent=badness_exponent_val,
            padding_pixels=max(
                float(padding_horizontal_pixels_val),
                float(padding_vertical_pixels_val),
            ),
            padding_horizontal_pixels=float(padding_horizontal_pixels_val),
            padding_vertical_pixels=float(padding_vertical_pixels_val),
            supersampling_factor=int(supersampling_factor_val),
        ),
        output=UIOutputSettings(
            output_directory=output_directory,
            disable_auto_save=bool(disable_auto_save),
            embed_editor_package=bool(embed_editor_package_val),
            save_plain_image_copy=bool(save_plain_image_copy_val),
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
            image_upscale_mode=image_upscale_mode_val,
            image_upscale_factor=image_upscale_factor_val,
            image_upscale_model=image_upscale_model_val,
        ),
        general=UIGeneralSettings(
            verbose=verbose,
            cleaning_only=cleaning_only_toggle,
            upscaling_only=upscaling_only_val,
            test_mode=test_mode_toggle,
            enable_web_search=enable_web_search_val,
            enable_code_execution=enable_code_execution_val,
            image_detail=image_detail_val,
            media_resolution=media_resolution_val,
            media_resolution_bubbles=media_resolution_bubbles_val,
            media_resolution_context=media_resolution_context_val,
            reasoning_effort=reasoning_effort_val,
            effort=effort_val,
            verbosity=verbosity_val,
            auto_scale=auto_scale_val,
        ),
        input_language=final_input_language,
        output_language=final_output_language,
        font_pack=final_font_pack,
        batch_input_language=batch_input_language,
        batch_output_language=batch_output_language,
        batch_font_pack=batch_font_dropdown,
        batch_parallel_requests=int(batch_parallel_requests_val),
        batch_retry_failed_once=bool(batch_retry_failed_once_val),
        batch_previous_context_image_count=int(batch_previous_context_image_count_val),
        batch_previous_context_text_count=int(batch_previous_context_text_count_val),
        batch_previous_context_text_mode=str(batch_previous_context_text_mode_val),
        character_profile_name=str(character_profile_name or "").strip(),
    )


def _validate_ui_state(ui_state: UIConfigState) -> None:
    """Validate UI state including modes and API keys. Raises gr.Error on validation failure."""
    try:
        validate_mutually_exclusive_modes(
            ui_state.general.cleaning_only,
            ui_state.general.upscaling_only,
            ui_state.general.test_mode,
        )
    except ValidationError as e:
        raise gr.Error(f"{ERROR_PREFIX}{str(e)}")

    # Skip API key validation if in cleaning-only, upscaling-only, or test mode
    if (
        not ui_state.general.cleaning_only
        and not ui_state.general.upscaling_only
        and not ui_state.general.test_mode
    ):
        api_key_to_validate = ""
        provider_selector = ui_state.provider_settings.provider
        if provider_selector == "Google":
            api_key_to_validate = ui_state.provider_settings.google_api_key
        elif provider_selector == "OpenAI":
            api_key_to_validate = ui_state.provider_settings.openai_api_key
        elif provider_selector == "Anthropic":
            api_key_to_validate = ui_state.provider_settings.anthropic_api_key
        elif provider_selector == "xAI":
            api_key_to_validate = ui_state.provider_settings.xai_api_key
        elif provider_selector == "DeepSeek":
            api_key_to_validate = ui_state.provider_settings.deepseek_api_key
        elif provider_selector == "Z.ai":
            api_key_to_validate = ui_state.provider_settings.zai_api_key
        elif provider_selector == "Moonshot AI":
            api_key_to_validate = ui_state.provider_settings.moonshot_api_key
        elif provider_selector == "OpenRouter":
            api_key_to_validate = ui_state.provider_settings.openrouter_api_key
        elif provider_selector == "OpenAI-Compatible":
            api_key_to_validate = ui_state.provider_settings.openai_compatible_api_key

        api_valid, api_msg = utils.validate_api_key(
            api_key_to_validate, provider_selector
        )
        if not api_valid and not (
            provider_selector == "OpenAI-Compatible" and not api_key_to_validate
        ):
            raise gr.Error(f"{ERROR_PREFIX}{api_msg}")

    if (
        ui_state.provider_settings.provider == "OpenAI-Compatible"
        and not ui_state.general.cleaning_only
        and not ui_state.general.upscaling_only
        and not ui_state.general.test_mode
    ):
        if not ui_state.provider_settings.openai_compatible_url:
            raise gr.Error(f"{ERROR_PREFIX}OpenAI-Compatible URL is required.")
        if not ui_state.provider_settings.openai_compatible_url.startswith(
            ("http://", "https://")
        ):
            raise gr.Error(
                f"{ERROR_PREFIX}Invalid OpenAI-Compatible URL format. Must start with http:// or https://",
            )

    if ui_state.outside_text.enabled:
        hf_valid, hf_msg = utils.validate_huggingface_token(
            ui_state.outside_text.huggingface_token
        )
        if not hf_valid:
            raise gr.Error(f"{ERROR_PREFIX}{hf_msg}")


def _format_single_success_message(
    result_image: Image.Image,
    backend_config: MangaTranslatorConfig,
    font_dir_path: Path,
    save_path: Optional[Path],
    processing_time: float,
    token_usage: Optional[Dict[str, int]] = None,
) -> str:
    """Formats the success message for single image translation."""
    width, height = result_image.size
    provider = backend_config.translation.provider
    model_name = backend_config.translation.model_name
    thinking_status_str = utils.format_thinking_status(
        provider,
        model_name,
        backend_config.translation.reasoning_effort,
    )
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Google":
        llm_params_str += f", Top-K={top_k_val}"
    elif provider == "Anthropic":
        llm_params_str += f", Top-K={top_k_val}"
        param_notes = " (Temp clamped <= 1.0)"
    elif provider == "OpenAI":
        param_notes = " (Top-K N/A)"
    elif provider == "OpenRouter":
        is_openai_model = "openai/" in model_name
        is_anthropic_model = "anthropic/" in model_name
        if is_openai_model or is_anthropic_model:
            param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
        else:
            llm_params_str += f", Top-K={top_k_val}"
    elif provider == "OpenAI-Compatible":
        llm_params_str += f", Top-K={top_k_val}"
    llm_params_str += param_notes

    if backend_config.cleaning_only:
        processing_mode_str = "Cleaning Only"
    elif backend_config.upscaling_only:
        processing_mode_str = "Upscaling Only"
    else:
        processing_mode_str = "Translation"

    if backend_config.test_mode:
        model_display = "<Test Mode>"
    elif backend_config.cleaning_only:
        model_display = "<Cleaning-only Mode>"
    elif backend_config.upscaling_only:
        model_display = "<Upscaling-only Mode>"
    else:
        model_display = f"{model_name}{thinking_status_str}"

    msg_parts = [
        f"{SUCCESS_PREFIX}{processing_mode_str} completed!\n",
        f"• Image Size: {width}x{height} pixels\n",
        f"• Outside Text Detection: {'Enabled' if backend_config.outside_text.enabled else 'Disabled'}\n",
    ]

    if backend_config.outside_text.enabled and backend_config.outside_text.osb_font_dir:
        osb_font_pack_name = Path(backend_config.outside_text.osb_font_dir).name
        msg_parts.append(f"• OSB Font Pack: {osb_font_pack_name}\n")

    if (
        backend_config.outside_text.enabled
        and backend_config.outside_text.osb_emphasis_font_dir
    ):
        osb_emphasis_font_name = Path(
            backend_config.outside_text.osb_emphasis_font_dir
        ).name
        msg_parts.append(f"• OSB Emphasis Font Pack: {osb_emphasis_font_name}\n")

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
        )
        previous_context_count = backend_config.translation.previous_context_image_count
        if previous_context_count:
            msg_parts.append(f"• Previous Context Images: {previous_context_count}\n")

    msg_parts.append(
        f"• Upscale Method: {backend_config.translation.upscale_method.title()}\n"
    )

    msg_parts.extend(
        [
            f"• Provider: {provider}\n",
            f"• Model: {model_display}\n",
            f"• Source Language: {backend_config.translation.input_language}\n",
            f"• Target Language: {backend_config.translation.output_language}\n",
            f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
            f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        ]
    )

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(f"{llm_params_str}\n")

    usage_str = format_usage(token_usage)
    if usage_str:
        msg_parts.append(f"• Tokens: {usage_str}\n")

    msg_parts.append(f"• Font Pack: {font_dir_path.name}\n")
    if backend_config.rendering.emphasis_font_dir:
        emphasis_font_name = Path(backend_config.rendering.emphasis_font_dir).name
        msg_parts.append(f"• Emphasis Font Pack: {emphasis_font_name}\n")

    if (
        getattr(backend_config, "preprocessing", None)
        and backend_config.preprocessing.enabled
    ):
        msg_parts.append(
            f"• Initial Image Upscaling: {backend_config.preprocessing.factor}x\n"
        )

    if backend_config.output.upscale_final_image:
        msg_parts.append(
            f"• Final Image Upscaling: {backend_config.output.image_upscale_factor}x\n"
        )

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output Format: png\n")
        msg_parts.append(
            f"• PNG Compression: {backend_config.output.png_compression}\n"
        )
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output Format: jpeg\n")
        msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
    else:
        try:
            ext = save_path.suffix.lower() if save_path else ""
        except Exception:
            ext = ""
        if ext in {".jpg", ".jpeg"}:
            msg_parts.append("• Output Format: auto → jpeg\n")
            msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
        elif ext == ".png":
            msg_parts.append("• Output Format: auto → png\n")
            msg_parts.append(
                f"• PNG Compression: {backend_config.output.png_compression}\n"
            )
        elif ext == ".webp":
            msg_parts.append("• Output Format: auto → webp\n")
        else:
            msg_parts.append("• Output Format: auto\n")

    msg_parts.append(f"• Processing Time: {processing_time:.2f} seconds\n")
    if save_path:
        msg_parts.append(f"• Saved To: {save_path}")
        plain_copy_path = _plain_download_path(save_path, backend_config)
        if plain_copy_path:
            msg_parts.append(f"\n• Plain Copy: {plain_copy_path}")
    else:
        msg_parts.append(
            "• 자동 저장: 꺼짐 - 지정 저장 폴더 저장을 건너뛰었습니다. 고급 편집기 세션은 유지됩니다."
        )
    return ko("".join(msg_parts))


def _format_batch_success_message(
    results: Dict[str, Any],
    backend_config: MangaTranslatorConfig,
    font_dir_path: Path,
) -> str:
    """Formats the success message for batch processing."""
    success_count = results["success_count"]
    error_count = results["error_count"]
    total_images = int(results.get("total_count") or (success_count + error_count))
    processing_time = results["processing_time"]
    output_path = results["output_path"]
    seconds_per_image = processing_time / success_count if success_count > 0 else 0
    was_cancelled = bool(results.get("cancelled", False))
    auto_save_disabled = bool(results.get("auto_save_disabled", False))
    cancel_message = str(results.get("cancel_message") or "").strip()

    provider = backend_config.translation.provider
    model_name = backend_config.translation.model_name
    thinking_status_str = utils.format_thinking_status(
        provider,
        model_name,
        backend_config.translation.reasoning_effort,
    )
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Google":
        llm_params_str += f", Top-K={top_k_val}"
    elif provider == "Anthropic":
        llm_params_str += f", Top-K={top_k_val}"
        param_notes = " (Temp clamped <= 1.0)"
    elif provider == "OpenAI":
        param_notes = " (Top-K N/A)"
    elif provider == "OpenRouter":
        is_openai_model = "openai/" in model_name or model_name.startswith("gpt-")
        is_anthropic_model = "anthropic/" in model_name or model_name.startswith(
            "claude-"
        )
        if is_openai_model or is_anthropic_model:
            param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
        else:
            llm_params_str += f", Top-K={top_k_val}"
    elif provider == "OpenAI-Compatible":
        param_notes = " (Top-K N/A)"
    llm_params_str += param_notes

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        llm_params_str = (
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
            + llm_params_str
        )

    error_summary = ""
    if error_count > 0:
        error_summary = f"\n• Warning: {error_count} image(s) failed to process."

    if backend_config.cleaning_only:
        processing_mode_str = "Cleaning Only"
    elif backend_config.upscaling_only:
        processing_mode_str = "Upscaling Only"
    else:
        processing_mode_str = "Translation"

    if was_cancelled and error_count > 0:
        error_summary = (
            f"\n??Warning: {error_count} image(s) failed or were not processed."
        )
    if was_cancelled and cancel_message:
        error_summary += f"\n??Cancel Reason: {cancel_message}"

    retry_attempted = int(results.get("retry_attempted_count", 0) or 0)
    retry_summary = ""
    if retry_attempted:
        retry_summary = (
            f"??Failed File Retry: {results.get('retry_success_count', 0)} "
            f"succeeded, {results.get('retry_failed_count', 0)} still failed\n"
        )

    # Determine model display based on mode
    if backend_config.test_mode:
        model_display = "<Test Mode>"
    elif backend_config.cleaning_only:
        model_display = "<Cleaning-only Mode>"
    elif backend_config.upscaling_only:
        model_display = "<Upscaling-only Mode>"
    else:
        model_display = f"{model_name}{thinking_status_str}"

    completion_word = "cancelled" if was_cancelled else "completed"
    msg_parts = [
        f"{SUCCESS_PREFIX}Batch {processing_mode_str.lower()} {completion_word}!\n",
        f"• Outside Text Detection: {'Enabled' if backend_config.outside_text.enabled else 'Disabled'}\n",
    ]

    if backend_config.outside_text.enabled and backend_config.outside_text.osb_font_dir:
        osb_font_pack_name = Path(backend_config.outside_text.osb_font_dir).name
        msg_parts.append(f"• OSB Font Pack: {osb_font_pack_name}\n")

    if (
        backend_config.outside_text.enabled
        and backend_config.outside_text.osb_emphasis_font_dir
    ):
        osb_emphasis_font_name = Path(
            backend_config.outside_text.osb_emphasis_font_dir
        ).name
        msg_parts.append(f"• OSB Emphasis Font Pack: {osb_emphasis_font_name}\n")

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
        )
        previous_context_count = backend_config.translation.previous_context_image_count
        if previous_context_count:
            msg_parts.append(f"• Previous Context Images: {previous_context_count}\n")

    msg_parts.append(
        f"• Upscale Method: {backend_config.translation.upscale_method.title()}\n"
    )

    msg_parts.extend(
        [
            f"• Provider: {provider}\n",
            f"• Model: {model_display}\n",
            f"• Source Language: {backend_config.translation.input_language}\n",
            f"• Target Language: {backend_config.translation.output_language}\n",
            f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
            f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        ]
    )

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(f"{llm_params_str}\n")

    msg_parts.append(f"• Font Pack: {font_dir_path.name}\n")
    if backend_config.rendering.emphasis_font_dir:
        emphasis_font_name = Path(backend_config.rendering.emphasis_font_dir).name
        msg_parts.append(f"• Emphasis Font Pack: {emphasis_font_name}\n")

    if (
        getattr(backend_config, "preprocessing", None)
        and backend_config.preprocessing.enabled
    ):
        msg_parts.append(
            f"• Initial Image Upscaling: {backend_config.preprocessing.factor}x\n"
        )

    if backend_config.output.upscale_final_image:
        msg_parts.append(
            f"• Final Image Upscaling: {backend_config.output.image_upscale_factor}x\n"
        )

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output Format: png\n")
        msg_parts.append(
            f"• PNG Compression: {backend_config.output.png_compression}\n"
        )
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output Format: jpeg\n")
        msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
    else:
        msg_parts.append("• Output Format: auto\n")
        try:
            out_dir = Path(output_path)
            if out_dir.exists() and out_dir.is_dir():
                exts = {p.suffix.lower() for p in out_dir.glob("*.*") if p.is_file()}
                only_jpeg = exts and all(e in {".jpg", ".jpeg"} for e in exts)
                only_png = exts == {".png"}
                if only_png:
                    msg_parts.append(
                        f"• PNG Compression: {backend_config.output.png_compression}\n"
                    )
                elif only_jpeg:
                    msg_parts.append(
                        f"• JPEG Quality: {backend_config.output.jpeg_quality}\n"
                    )
                # If mixed or unknown, omit specific settings to avoid showing both
        except Exception:
            pass

    msg_parts.extend(
        [
            f"• Successful Translations: {success_count}/{total_images}{error_summary}\n",
            f"• Total Processing Time: {processing_time:.2f} seconds ({seconds_per_image:.2f} seconds/image)\n",
        ]
    )
    if auto_save_disabled:
        msg_parts.append(
            "• 자동 저장: 꺼짐 - 지정 저장 폴더 저장은 건너뛰고, 결과 ZIP은 다운로드용으로 생성했습니다. "
            "화면 목록과 고급 편집기 세션은 유지됩니다."
        )
    else:
        msg_parts.append(f"• Saved To: {output_path}")
        if (
            bool(getattr(backend_config.output, "embed_editor_package", True))
            and bool(getattr(backend_config.output, "save_plain_image_copy", False))
        ):
            msg_parts.append(
                f"\n• 편집 정보 포함 폴더: {Path(output_path) / BATCH_EMBEDDED_OUTPUT_DIR_NAME}"
            )
            msg_parts.append(
                f"\n• 일반 이미지 폴더: {Path(output_path) / BATCH_PLAIN_OUTPUT_DIR_NAME}"
            )
    if retry_summary:
        msg_parts.append(retry_summary)
    if results.get("success_zip_path"):
        msg_parts.append(f"\n??Successful Files ZIP: {results['success_zip_path']}")
    if results.get("failed_zip_path"):
        msg_parts.append(f"\n??Failed Files ZIP: {results['failed_zip_path']}")
    return ko("".join(msg_parts))


@workspace_job
def handle_translate_click(
    *args: Any,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
) -> Tuple[Optional[Image.Image], gr.update, gr.update, gr.update, str, gr.update]:
    """Callback for the 'Translate' button click. Uses dataclasses for config."""
    input_image_path = args[0]
    start_time = time.time()
    global CANCELLATION_MANAGER
    CANCELLATION_MANAGER = CancellationManager()
    try:
        img_valid, img_msg = utils.validate_image(input_image_path)
        if not img_valid:
            raise gr.Error(f"{ERROR_PREFIX}{img_msg}")

        ui_state = _build_ui_state_from_args(args[1:], is_batch=False)

        _validate_ui_state(ui_state)

        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=False,
        )
        selected_font_pack_name = ui_state.font_pack

        token_usage: Dict[str, int] = {}
        result_image, save_path, edit_session_path = logic.translate_manga_logic(
            image=input_image_path,
            config=backend_config,
            selected_font_pack_name=selected_font_pack_name,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            output_base_dir=ui_state.output.output_directory,
            disable_auto_save=ui_state.output.disable_auto_save,
            cancellation_manager=CANCELLATION_MANAGER,
            usage_out=token_usage,
        )

        # Re-get font path for success message (logic already validated it)
        font_dir_path = (
            fonts_base_dir / selected_font_pack_name
            if selected_font_pack_name
            else Path(".")
        )

        processing_time = time.time() - start_time
        status_msg = _format_single_success_message(
            result_image,
            backend_config,
            font_dir_path,
            save_path,
            processing_time,
            token_usage,
        )
        if edit_session_path:
            status_msg = f"{status_msg}\n편집 세션: {edit_session_path}"
        plain_copy_path = _plain_download_path(save_path, backend_config)

        return (
            result_image.copy(),
            _status_update(status_msg),
            _download_file_update(save_path),
            _download_file_update(plain_copy_path),
            str(edit_session_path or ""),
            _editor_button_update(edit_session_path),
        )

    except gr.Error as e:
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return (
            None,
            _status_update(cleaned),
            _download_file_update(),
            _download_file_update(),
            "",
            _editor_button_update(),
        )
    except CancellationError as e:
        message = str(e) or "Translation cancelled by user."
        return (
            None,
            _status_update(message),
            _download_file_update(),
            _download_file_update(),
            "",
            _editor_button_update(),
        )
    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError) as e:
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return (
            None,
            _status_update(cleaned),
            _download_file_update(),
            _download_file_update(),
            "",
            _editor_button_update(),
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        cleaned = _clean_error_message(f"An unexpected error occurred: {str(e)}")
        gr.Error(cleaned)
        return (
            None,
            _status_update(cleaned),
            _download_file_update(),
            _download_file_update(),
            "",
            _editor_button_update(),
        )


def handle_create_image_editor_sessions(
    folder_path: str,
    font_pack: str,
    *,
    fonts_base_dir: Path,
) -> Tuple[gr.update, gr.update, gr.update]:
    try:
        selected_folder = Path(
            os.path.expandvars(str(folder_path or "").strip())
        ).expanduser()
        if not selected_folder.is_absolute():
            selected_folder = Path(__file__).resolve().parents[1] / selected_folder
        selected_folder = selected_folder.resolve()
        if not selected_folder.is_dir():
            raise ValueError("먼저 '편집할 폴더 선택'을 눌러 이미지 폴더를 선택해 주세요.")

        if not font_pack:
            available_fonts = sorted(
                path.name
                for path in fonts_base_dir.iterdir()
                if path.is_dir()
            )
            font_pack = available_fonts[0] if available_fonts else ""
        if not font_pack:
            raise ValueError("텍스트 렌더링에 사용할 글꼴 폴더가 없습니다.")

        paths = sorted(
            (
                path.resolve()
                for path in selected_folder.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EDITOR_EXTENSIONS
            ),
            key=_natural_path_key,
        )
        if not paths:
            raise ValueError(
                f"선택한 폴더에서 편집할 이미지 파일을 찾지 못했습니다: {selected_folder}"
            )

        from core.editor_session import create_image_edit_sessions

        source_references = {
            str(path).casefold(): {
                "kind": "file",
                "path": str(path),
                "display_path": str(path.relative_to(selected_folder)).replace(
                    "\\", "/"
                ),
            }
            for path in paths
        }
        _remember_image_editor_source_path(selected_folder)
        created = create_image_edit_sessions(
            paths,
            font_dir=fonts_base_dir / font_pack,
            source_references=source_references,
        )
        choices = [
            (
                f"{index}. {_image_editor_choice_label(source_path, session_path)}",
                str(session_path),
            )
            for index, (source_path, session_path) in enumerate(created, start=1)
        ]
        selected = choices[0][1]
        status = (
            f"폴더에서 이미지 {len(choices)}장을 불러왔습니다.\n"
            f"원본 폴더: {selected_folder}\n"
            "목록에서 이미지를 선택한 뒤 '고급 편집기 열기'를 누르세요. "
            "편집기의 원본 덮어쓰기는 이 폴더의 실제 파일에 적용됩니다."
        )
        return (
            gr.update(choices=choices, value=selected, visible=True, interactive=True),
            _editor_button_update(selected),
            _status_update(status),
        )
    except Exception as exc:
        cleaned = _clean_error_message(exc)
        gr.Error(cleaned)
        return (
            gr.update(choices=[], value=None, visible=False),
            _editor_button_update(),
            _status_update(cleaned),
        )


def handle_launch_advanced_editor(session_path: str) -> gr.update:
    try:
        if not session_path:
            raise ValueError("열 수 있는 편집 세션이 없습니다.")
        repo_root = Path(__file__).resolve().parents[1]
        raw_path = Path(session_path)
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"편집 세션을 찾을 수 없습니다: {path}")

        log_dir = repo_root / "output" / "advanced_editor_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stdout_path = log_dir / f"advanced_editor_stdout_{stamp}.log"
        stderr_path = log_dir / f"advanced_editor_stderr_{stamp}.log"
        lifecycle_path = log_dir / "advanced_editor_lifecycle.log"
        lifecycle_offset = lifecycle_path.stat().st_size if lifecycle_path.exists() else 0
        pythonw_path = Path(sys.executable).with_name("pythonw.exe")
        python_exe = str(pythonw_path if pythonw_path.exists() else Path(sys.executable))

        creationflags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP

        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

        with open(stdout_path, "w", encoding="utf-8") as stdout_file, open(
            stderr_path, "w", encoding="utf-8"
        ) as stderr_file:
            process = subprocess.Popen(
                [python_exe, "-m", "advanced_editor.launcher", str(path)],
                cwd=str(repo_root),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=False,
                creationflags=creationflags,
                env=env,
            )

        window_ready = False
        ready_marker = f"PID={process.pid} window shown"
        for _ in range(150):
            time.sleep(0.2)
            if process.poll() is not None:
                break
            try:
                with open(lifecycle_path, "r", encoding="utf-8", errors="replace") as lifecycle_file:
                    lifecycle_file.seek(lifecycle_offset)
                    if ready_marker in lifecycle_file.read():
                        window_ready = True
                        break
            except (FileNotFoundError, OSError):
                pass

        if window_ready:
            time.sleep(0.5)

        if process.poll() is not None:
            stderr_text = ""
            try:
                stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                stderr_text = ""
            stdout_text = ""
            try:
                stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                stdout_text = ""
            details = stderr_text or stdout_text or "프로세스 로그가 기록되지 않았습니다."
            raise RuntimeError(
                "고급 편집기가 실행 직후 종료되었습니다.\n"
                f"종료 코드: {process.returncode}\n"
                f"stderr: {stderr_path}\n"
                f"stdout: {stdout_path}\n"
                f"실행 기록: {lifecycle_path}\n"
                f"{details}"
            )

        launch_state = "열렸습니다" if window_ready else "실행되었으며 불러오는 중입니다"
        return _status_update(
            f"고급 편집기가 {launch_state}. PID: {process.pid}\n"
            f"편집 세션: {path}\n"
            f"오류 로그: {stderr_path}\n"
            f"실행 기록: {lifecycle_path}"
        )
    except Exception as e:
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return _status_update(cleaned)


def _editor_blank_outputs(message: str = "") -> Tuple[Any, ...]:
    return (
        None,
        gr.update(choices=[], value=None),
        "",
        "",
        0,
        0,
        1,
        1,
        16,
        8,
        0.0,
        False,
        "#000000",
        True,
        _status_update(message),
    )


def _editor_form_outputs(session_path: str, region_id: Optional[str]) -> Tuple[Any, ...]:
    from core.editor_session import get_region_form_values

    (
        source_text,
        translation,
        x,
        y,
        width,
        height,
        max_font_size,
        min_font_size,
        rotation,
        vertical_stack,
        font_color,
        enabled,
        preview,
    ) = get_region_form_values(session_path, region_id)
    return (
        preview,
        source_text,
        translation,
        x,
        y,
        width,
        height,
        max_font_size,
        min_font_size,
        rotation,
        vertical_stack,
        font_color,
        enabled,
    )


def handle_editor_load(session_path: str) -> Tuple[Any, ...]:
    """Load the post-translation editor session into the editor tab."""
    try:
        if not session_path:
            return _editor_blank_outputs("No translated editor session is available yet.")
        from core.editor_session import load_session, region_choices

        session = load_session(session_path)
        choices = region_choices(session)
        if not choices:
            return _editor_blank_outputs("This editor session has no editable regions.")
        selected_id = choices[0][1]
        (
            preview,
            source_text,
            translation,
            x,
            y,
            width,
            height,
            max_font_size,
            min_font_size,
            rotation,
            vertical_stack,
            font_color,
            enabled,
        ) = _editor_form_outputs(session_path, selected_id)
        return (
            preview,
            gr.update(choices=choices, value=selected_id),
            source_text,
            translation,
            x,
            y,
            width,
            height,
            max_font_size,
            min_font_size,
            rotation,
            vertical_stack,
            font_color,
            enabled,
            _status_update(f"Loaded editor session: {session_path}"),
        )
    except Exception as e:
        cleaned = _clean_error_message(e)
        return _editor_blank_outputs(cleaned)


def handle_editor_region_change(session_path: str, region_id: str) -> Tuple[Any, ...]:
    try:
        if not session_path or not region_id:
            return (
                None,
                "",
                "",
                0,
                0,
                1,
                1,
                16,
                8,
                0.0,
                False,
                "#000000",
                True,
                _status_update("Select an editable region."),
            )
        return (*_editor_form_outputs(session_path, region_id), _status_update(""))
    except Exception as e:
        cleaned = _clean_error_message(e)
        return (
            None,
            "",
            "",
            0,
            0,
            1,
            1,
            16,
            8,
            0.0,
            False,
            "#000000",
            True,
            _status_update(cleaned),
        )


def handle_editor_save_region(
    session_path: str,
    region_id: str,
    translation: str,
    x: Any,
    y: Any,
    width: Any,
    height: Any,
    max_font_size: Any,
    min_font_size: Any,
    rotation_deg: Any,
    vertical_stack: bool,
    font_color: str,
    enabled: bool,
) -> Tuple[Any, str]:
    try:
        if not session_path or not region_id:
            raise ValueError("No editor region is selected.")
        from core.editor_session import update_region_from_form

        _, preview = update_region_from_form(
            session_path,
            region_id,
            translation,
            x,
            y,
            width,
            height,
            max_font_size,
            min_font_size,
            rotation_deg,
            vertical_stack,
            font_color,
            enabled,
        )
        return preview, _status_update("Region changes saved.")
    except Exception as e:
        cleaned = _clean_error_message(e)
        return None, _status_update(cleaned)


def handle_editor_add_region(session_path: str) -> Tuple[Any, ...]:
    try:
        if not session_path:
            return _editor_blank_outputs("No translated editor session is available yet.")
        from core.editor_session import add_region, region_choices

        session, selected_id, preview = add_region(session_path)
        choices = region_choices(session)
        (
            _preview,
            source_text,
            translation,
            x,
            y,
            width,
            height,
            max_font_size,
            min_font_size,
            rotation,
            vertical_stack,
            font_color,
            enabled,
        ) = _editor_form_outputs(session_path, selected_id)
        return (
            preview,
            gr.update(choices=choices, value=selected_id),
            source_text,
            translation,
            x,
            y,
            width,
            height,
            max_font_size,
            min_font_size,
            rotation,
            vertical_stack,
            font_color,
            enabled,
            _status_update("Added a new text region."),
        )
    except Exception as e:
        cleaned = _clean_error_message(e)
        return _editor_blank_outputs(cleaned)


def handle_editor_delete_region(session_path: str, region_id: str) -> Tuple[Any, ...]:
    try:
        if not session_path or not region_id:
            return _editor_blank_outputs("No editor region is selected.")
        from core.editor_session import delete_region, region_choices

        session, selected_id, preview = delete_region(session_path, region_id)
        choices = region_choices(session)
        if not selected_id:
            return (
                preview,
                gr.update(choices=[], value=None),
                "",
                "",
                0,
                0,
                1,
                1,
                16,
                8,
                0.0,
                False,
                "#000000",
                True,
                _status_update("Deleted region. No regions remain."),
            )
        (
            _preview,
            source_text,
            translation,
            x,
            y,
            width,
            height,
            max_font_size,
            min_font_size,
            rotation,
            vertical_stack,
            font_color,
            enabled,
        ) = _editor_form_outputs(session_path, selected_id)
        return (
            preview,
            gr.update(choices=choices, value=selected_id),
            source_text,
            translation,
            x,
            y,
            width,
            height,
            max_font_size,
            min_font_size,
            rotation,
            vertical_stack,
            font_color,
            enabled,
            _status_update("Deleted region."),
        )
    except Exception as e:
        cleaned = _clean_error_message(e)
        return _editor_blank_outputs(cleaned)


def handle_editor_rerender(
    session_path: str,
    region_id: str,
    translation: str,
    x: Any,
    y: Any,
    width: Any,
    height: Any,
    max_font_size: Any,
    min_font_size: Any,
    rotation_deg: Any,
    vertical_stack: bool,
    font_color: str,
    enabled: bool,
) -> Tuple[Any, gr.update, gr.update, Any]:
    try:
        if not session_path:
            raise ValueError("No translated editor session is available yet.")
        if region_id:
            from core.editor_session import update_region_from_form

            update_region_from_form(
                session_path,
                region_id,
                translation,
                x,
                y,
                width,
                height,
                max_font_size,
                min_font_size,
                rotation_deg,
                vertical_stack,
                font_color,
                enabled,
            )
        from core.editor_session import rerender_session

        image, output_path, errors = rerender_session(session_path)
        message = f"Edited image saved: {output_path}"
        if errors:
            message += "\n" + "\n".join(errors[:8])
            if len(errors) > 8:
                message += f"\n...and {len(errors) - 8} more render warning(s)."
        return image, _status_update(message), _download_file_update(output_path), image.copy()
    except Exception as e:
        cleaned = _clean_error_message(e)
        return None, _status_update(cleaned), _download_file_update(), None


@workspace_job
def handle_batch_click(
    *args: Any,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[str], str, Optional[str], Optional[str], gr.update, gr.update]:
    """Callback for the 'Start Batch Translating' button click. Uses dataclasses."""
    input_files = args[0]
    input_zip = args[1] if len(args) > 1 else None
    native_folder_path = args[2] if len(args) > 2 else None
    native_zip_path = args[3] if len(args) > 3 else None
    progress(0, desc=ko("Starting batch process..."))
    global CANCELLATION_MANAGER
    CANCELLATION_MANAGER = CancellationManager()
    try:
        input_files, input_zip = _resolve_native_batch_inputs(
            input_files,
            input_zip,
            native_folder_path,
            native_zip_path,
        )
        zip_file_path = None
        if input_zip:
            try:
                zip_path_str = normalize_zip_file_input(input_zip)
                zip_file_path = validate_zip_file(zip_path_str)
            except (ValidationError, FileNotFoundError) as e:
                raise gr.Error(f"{ERROR_PREFIX}{str(e)}")

        if input_files:
            if not isinstance(input_files, list):
                raise gr.Error(
                    f"{ERROR_PREFIX}Invalid input format. Expected a list of files."
                )

        if not zip_file_path and not input_files:
            raise gr.Error(
                f"{ERROR_PREFIX}Please upload images, a folder, or a ZIP archive."
            )

        if zip_file_path and input_files:
            input_to_process = {
                "zip": str(zip_file_path),
                "files": input_files,
            }
        elif zip_file_path:
            input_to_process = str(zip_file_path)
        else:
            input_to_process = input_files

        ui_state = _build_ui_state_from_args(args[4:], is_batch=True)

        _validate_ui_state(ui_state)

        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=True,
        )
        selected_font_pack_name = ui_state.batch_font_pack

        results = logic.process_batch_logic(
            input_dir_or_files=input_to_process,
            config=backend_config,
            selected_font_pack_name=selected_font_pack_name,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            output_base_dir=ui_state.output.output_directory,
            disable_auto_save=ui_state.output.disable_auto_save,
            gradio_progress=progress,
            cancellation_manager=CANCELLATION_MANAGER,
        )

        output_path = results["output_path"]
        gallery_images = []
        if output_path.exists():
            # Use rglob to recursively find all images when structure is preserved
            processed_files = list(output_path.rglob("*.*"))
            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            processed_files = [
                f
                for f in processed_files
                if f.is_file() and f.suffix.lower() in image_extensions
                and not any(part.endswith("_edit_session_assets") for part in f.parts)
                and "_edit_session" not in f.stem
            ]
            processed_files.sort(  # Sort naturally (e.g., page_1, page_2, page_10)
                key=lambda x: tuple(
                    int(part) if part.isdigit() else part
                    for part in re.split(r"(\d+)", x.stem)
                )
            )
            gallery_images = [str(file_path) for file_path in processed_files]
        editor_session_choices = _editor_session_choices_from_outputs(gallery_images)

        # Re-get font path for success message (logic already validated it)
        font_dir_path = (
            fonts_base_dir / selected_font_pack_name
            if selected_font_pack_name
            else Path(".")
        )

        status_msg = _format_batch_success_message(
            results, backend_config, font_dir_path
        )
        success_zip_path = results.get("success_zip_path")
        failed_zip_path = results.get("failed_zip_path")
        progress(
            1.0,
            desc=(
                ko("Batch cancelled; auto-save is off.")
                if results.get("auto_save_disabled") and results.get("cancelled")
                else ko("Batch complete; auto-save is off.")
                if results.get("auto_save_disabled")
                else ko("Batch cancelled; download ZIPs are ready.")
                if results.get("cancelled")
                else ko("Batch complete!")
            ),
        )
        return (
            gallery_images,
            _status_update(status_msg),
            _download_file_update(success_zip_path),
            _download_file_update(failed_zip_path),
            gr.update(
                choices=editor_session_choices,
                value=editor_session_choices[0][1] if editor_session_choices else None,
                visible=bool(editor_session_choices),
            ),
            _editor_button_update(editor_session_choices[0][1] if editor_session_choices else None),
        )

    except gr.Error as e:
        progress(1.0, desc=ko("Error occurred"))
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return None, _status_update(cleaned), _download_file_update(), _download_file_update(), gr.update(choices=[], value=None, visible=False), _editor_button_update()
    except CancellationError as e:
        message = str(e) or "Batch process cancelled by user."
        progress(1.0, desc=ko("Cancelled"))
        return (
            None,
            _status_update(message),
            _download_file_update(),
            _download_file_update(),
            gr.update(choices=[], value=None, visible=False),
            _editor_button_update(),
        )
    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError) as e:
        progress(1.0, desc=ko("Error occurred"))
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return None, _status_update(cleaned), _download_file_update(), _download_file_update(), gr.update(choices=[], value=None, visible=False), _editor_button_update()
    except Exception as e:
        progress(1.0, desc=ko("Error occurred"))
        import traceback

        traceback.print_exc()
        cleaned = _clean_error_message(
            f"An unexpected error occurred during batch processing: {str(e)}"
        )
        gr.Error(cleaned)
        return None, _status_update(cleaned), _download_file_update(), _download_file_update(), gr.update(choices=[], value=None, visible=False), _editor_button_update()


def handle_save_config_click(*args: Any) -> str:
    """Callback for the 'Save Config' button. Uses dataclasses."""
    (
        conf,
        conjoined_conf,
        panel_conf,
        seg_model,
        bubble_detector_model,
        conjoined_detection,
        osb_text_verification,
        use_panel_sorting,
        rd,
        thresholding_val,
        otsu,
        inpaint_colored_bubbles,
        colored_bubble_inpainting_size,
        colored_bubble_inpainting_precision,
        colored_bubble_mask_kernel_size,
        colored_bubble_mask_dilation_offset,
        roi_shrink_px,
        prov,
        gem_key,
        google_rate_limit_enabled_val,
        google_request_interval_seconds_val,
        google_requests_per_minute_val,
        google_burst_requests_val,
        google_burst_cooldown_seconds_val,
        google_429_cooldown_seconds_val,
        google_max_retries_val,
        google_429_hourly_limit_val,
        google_429_hourly_action_val,
        google_429_pause_minutes_val,
        oai_key,
        ant_key,
        xai_key,
        deepseek_key,
        zai_key,
        moonshot_key,
        or_key,
        comp_url,
        comp_key,
        model,
        temp,
        tp,
        tk,
        max_tokens,
        trans_mode,
        ocr_method_val,
        max_fs,
        emphasis_max_fs,
        min_fs,
        ls,
        emphasis_ls,
        vertical_ls,
        vertical_emphasis_ls,
        vertical_font_size_mult,
        vertical_max_chars,
        vertical_emphasis_max_chars,
        tag_style_table_val,
        subpix,
        hint,
        liga,
        inside_light_text_color,
        inside_light_outline_color,
        inside_light_outline_width,
        inside_dark_text_color,
        inside_dark_outline_color,
        inside_dark_outline_width,
        outside_light_text_color,
        outside_light_outline_color,
        outside_light_outline_width,
        outside_dark_text_color,
        outside_dark_outline_color,
        outside_dark_outline_width,
        out_fmt,
        jq,
        pngc,
        output_directory_val,
        disable_auto_save_val,
        embed_editor_package_val,
        save_plain_image_copy_val,
        verb,
        cleaning_only_val,
        upscaling_only_val,
        test_mode_val,
        s_in_lang,
        s_out_lang,
        s_font,
        emphasis_font_pack_val,
        b_in_lang,
        b_out_lang,
        b_font,
        enable_web_search_val,
        enable_code_execution_val,
        image_detail_val,
        media_resolution_val,
        media_resolution_bubbles_val,
        media_resolution_context_val,
        reasoning_effort_val,
        effort_val,
        verbosity_val,
        send_full_page_context_val,
        parallel_inpaint_translation_val,
        whiteout_conjoined_bubbles_val,
        upscale_method_val,
        bubble_min_side_pixels_val,
        context_image_max_side_pixels_val,
        osb_min_side_pixels_val,
        hyphenate_before_scaling_val,
        detach_trailing_punctuation_val,
        auto_vertical_text_val,
        special_instructions_val,
        batch_special_instructions_val,
        hyphen_penalty_val,
        hyphenation_min_word_length_val,
        badness_exponent_val,
        padding_horizontal_pixels_val,
        padding_vertical_pixels_val,
        supersampling_factor_val,
        outside_text_enabled_val,
        outside_text_seed_val,
        outside_text_inpainting_method_val,
        outside_text_kontext_backend_val,
        outside_text_flux_offload_mode_val,
        outside_text_flux_num_inference_steps_val,
        outside_text_flux_kontext_prompt_val,
        outside_text_flux_klein_prompt_val,
        outside_text_flux_luminance_correction_val,
        outside_text_flux_upscale_small_crops_val,
        outside_text_flux_group_regions_val,
        outside_text_flux_residual_diff_threshold_val,
        outside_text_osb_residual_render_guard_action_val,
        outside_text_osb_confidence_val,
        outside_text_enable_page_number_filtering_val,
        outside_text_page_filter_margin_threshold_val,
        outside_text_page_filter_min_area_ratio_val,
        outside_text_huggingface_token_val,
        outside_text_osb_font_pack_val,
        outside_text_osb_emphasis_font_pack_val,
        outside_text_osb_max_font_size_val,
        outside_text_osb_emphasis_max_font_size_val,
        outside_text_osb_min_font_size_val,
        outside_text_osb_use_ligatures_val,
        outside_text_osb_outline_width_val,
        outside_text_osb_padding_pixels_val,
        outside_text_osb_line_spacing_val,
        outside_text_osb_vertical_line_spacing_val,
        outside_text_osb_auto_vertical_val,
        outside_text_osb_vertical_aspect_ratio_val,
        outside_text_osb_vertical_max_chars_val,
        outside_text_osb_tall_box_expand_aspect_ratio_val,
        outside_text_osb_tall_box_expand_width_percent_val,
        outside_text_osb_rotation_max_degrees_val,
        outside_text_osb_use_subpixel_rendering_val,
        outside_text_osb_font_hinting_val,
        outside_text_bbox_expansion_percent_val,
        outside_text_osb_render_expansion_narrow_multiplier_val,
        outside_text_osb_render_expansion_aspect_ratio_threshold_val,
        outside_text_osb_render_expansion_tiny_multiplier_val,
        outside_text_osb_render_expansion_area_threshold_percent_val,
        outside_text_text_box_proximity_ratio_val,
        image_upscale_mode_val,
        image_upscale_factor_val,
        image_upscale_model_val,
        auto_scale_val,
        batch_parallel_requests_val,
        batch_retry_failed_once_val,
        batch_previous_context_image_count_val,
        batch_previous_context_text_count_val,
        batch_previous_context_text_mode_val,
    ) = args
    flux_cpu_offload, flux_low_vram = flux_offload_flags(
        outside_text_flux_offload_mode_val
    )
    outside_text_processing_mode = normalize_outside_text_processing_mode(
        outside_text_enabled_val
    )

    ui_state = UIConfigState(
        detection=UIDetectionSettings(
            confidence=conf,
            conjoined_confidence=conjoined_conf,
            panel_confidence=panel_conf,
            seg_model=seg_model,
            bubble_detector_model=bubble_detector_model,
            conjoined_detection=conjoined_detection,
            use_osb_text_verification=osb_text_verification,
            use_panel_sorting=use_panel_sorting,
        ),
        cleaning=UICleaningSettings(
            thresholding_value=thresholding_val,
            use_otsu_threshold=otsu,
            inpaint_colored_bubbles=inpaint_colored_bubbles,
            colored_bubble_inpainting_size=int(
                colored_bubble_inpainting_size
            ),
            colored_bubble_inpainting_precision=(
                colored_bubble_inpainting_precision
            ),
            colored_bubble_mask_kernel_size=int(
                colored_bubble_mask_kernel_size
            ),
            colored_bubble_mask_dilation_offset=int(
                colored_bubble_mask_dilation_offset
            ),
            roi_shrink_px=int(max(0, min(8, roi_shrink_px))),
        ),
        outside_text=UIOutsideTextSettings(
            enabled=outside_text_processing_mode != "off",
            processing_mode=outside_text_processing_mode,
            seed=int(outside_text_seed_val),
            huggingface_token=outside_text_huggingface_token_val,
            inpainting_method=outside_text_inpainting_method_val,
            kontext_backend=outside_text_kontext_backend_val,
            flux_cpu_offload=flux_cpu_offload,
            flux_low_vram=flux_low_vram,
            flux_num_inference_steps=int(outside_text_flux_num_inference_steps_val),
            flux_kontext_prompt=outside_text_flux_kontext_prompt_val,
            flux_klein_prompt=outside_text_flux_klein_prompt_val,
            flux_luminance_correction=outside_text_flux_luminance_correction_val,
            flux_upscale_small_crops=outside_text_flux_upscale_small_crops_val,
            flux_group_regions=outside_text_flux_group_regions_val,
            flux_residual_diff_threshold=float(
                outside_text_flux_residual_diff_threshold_val
            ),
            osb_residual_render_guard_action=outside_text_osb_residual_render_guard_action_val,
            osb_confidence=float(outside_text_osb_confidence_val),
            enable_page_number_filtering=outside_text_enable_page_number_filtering_val,
            page_filter_margin_threshold=float(
                outside_text_page_filter_margin_threshold_val
            ),
            page_filter_min_area_ratio=float(
                outside_text_page_filter_min_area_ratio_val
            ),
            osb_font_dir=outside_text_osb_font_pack_val,
            osb_emphasis_font_dir=outside_text_osb_emphasis_font_pack_val,
            osb_max_font_size=int(outside_text_osb_max_font_size_val),
            osb_emphasis_max_font_size=int(
                outside_text_osb_emphasis_max_font_size_val
            ),
            osb_min_font_size=int(outside_text_osb_min_font_size_val),
            osb_use_ligatures=outside_text_osb_use_ligatures_val,
            osb_outline_width=float(outside_text_osb_outline_width_val),
            osb_padding_pixels=float(outside_text_osb_padding_pixels_val),
            osb_line_spacing=float(outside_text_osb_line_spacing_val),
            osb_vertical_line_spacing=float(
                outside_text_osb_vertical_line_spacing_val
            ),
            osb_auto_vertical=outside_text_osb_auto_vertical_val,
            osb_vertical_aspect_ratio=float(
                outside_text_osb_vertical_aspect_ratio_val
            ),
            osb_vertical_max_chars=int(outside_text_osb_vertical_max_chars_val),
            osb_tall_box_expand_aspect_ratio=float(
                outside_text_osb_tall_box_expand_aspect_ratio_val
            ),
            osb_tall_box_expand_width_percent=float(
                outside_text_osb_tall_box_expand_width_percent_val
            ),
            osb_rotation_max_degrees=float(
                outside_text_osb_rotation_max_degrees_val
            ),
            osb_use_subpixel_rendering=outside_text_osb_use_subpixel_rendering_val,
            osb_font_hinting=outside_text_osb_font_hinting_val,
            bbox_expansion_percent=float(outside_text_bbox_expansion_percent_val),
            osb_render_expansion_narrow_multiplier=float(
                outside_text_osb_render_expansion_narrow_multiplier_val
            ),
            osb_render_expansion_tiny_multiplier=float(
                outside_text_osb_render_expansion_tiny_multiplier_val
            ),
            osb_render_expansion_aspect_ratio_threshold=float(
                outside_text_osb_render_expansion_aspect_ratio_threshold_val
            ),
            osb_render_expansion_area_ratio_threshold=float(
                outside_text_osb_render_expansion_area_threshold_percent_val
            )
            / 100.0,
            text_box_proximity_ratio=float(outside_text_text_box_proximity_ratio_val),
        ),
        provider_settings=UITranslationProviderSettings(
            provider=prov,
            google_api_key=gem_key,
            openai_api_key=oai_key,
            anthropic_api_key=ant_key,
            xai_api_key=xai_key,
            deepseek_api_key=deepseek_key,
            zai_api_key=zai_key,
            moonshot_api_key=moonshot_key,
            openrouter_api_key=or_key,
            openai_compatible_url=comp_url,
            openai_compatible_api_key=comp_key,
        ),
        google_rate_limit=UIGoogleRateLimitSettings(
            enabled=bool(google_rate_limit_enabled_val),
            request_interval_seconds=max(
                0.0, float(google_request_interval_seconds_val or 0.0)
            ),
            requests_per_minute=max(
                0, int(float(google_requests_per_minute_val or 0))
            ),
            burst_requests=max(0, int(float(google_burst_requests_val or 0))),
            burst_cooldown_seconds=max(
                0.0, float(google_burst_cooldown_seconds_val or 0.0)
            ),
            cooldown_429_seconds=max(
                0.0, float(google_429_cooldown_seconds_val or 0.0)
            ),
            max_retries=max(0, int(float(google_max_retries_val or 0))),
            hourly_429_limit=max(
                0, int(float(google_429_hourly_limit_val or 0))
            ),
            hourly_429_action=(
                str(google_429_hourly_action_val or "stop").strip().lower()
                if str(google_429_hourly_action_val or "stop").strip().lower()
                in {"stop", "pause"}
                else "stop"
            ),
            pause_429_minutes=max(
                0.0, float(google_429_pause_minutes_val or 0.0)
            ),
        ),
        llm_settings=UITranslationLLMSettings(
            model_name=model,
            temperature=temp,
            top_p=tp,
            top_k=tk,
            max_tokens=max_tokens,
            reading_direction=rd,
            translation_mode=trans_mode,
            ocr_method=ocr_method_val,
            send_full_page_context=send_full_page_context_val,
            parallel_inpaint_translation=parallel_inpaint_translation_val,
            whiteout_conjoined_bubbles=whiteout_conjoined_bubbles_val,
            upscale_method=upscale_method_val,
            bubble_min_side_pixels=bubble_min_side_pixels_val,
            context_image_max_side_pixels=context_image_max_side_pixels_val,
            osb_min_side_pixels=osb_min_side_pixels_val,
            special_instructions=special_instructions_val,
        ),
        rendering=UIRenderingSettings(
            emphasis_font_dir=emphasis_font_pack_val,
            max_font_size=max_fs,
            emphasis_max_font_size=emphasis_max_fs,
            min_font_size=min_fs,
            line_spacing_mult=ls,
            emphasis_line_spacing_mult=emphasis_ls,
            vertical_line_spacing_mult=vertical_ls,
            vertical_emphasis_line_spacing_mult=vertical_emphasis_ls,
            vertical_font_size_mult=vertical_font_size_mult,
            vertical_max_chars=int(vertical_max_chars),
            vertical_emphasis_max_chars=int(vertical_emphasis_max_chars),
            tag_style_table=tag_style_table_val,
            use_subpixel_rendering=subpix,
            font_hinting=hint,
            use_ligatures=liga,
            inside_light_text_color=inside_light_text_color,
            inside_light_outline_color=inside_light_outline_color,
            inside_light_outline_width=float(inside_light_outline_width),
            inside_dark_text_color=inside_dark_text_color,
            inside_dark_outline_color=inside_dark_outline_color,
            inside_dark_outline_width=float(inside_dark_outline_width),
            outside_light_text_color=outside_light_text_color,
            outside_light_outline_color=outside_light_outline_color,
            outside_light_outline_width=float(outside_light_outline_width),
            outside_dark_text_color=outside_dark_text_color,
            outside_dark_outline_color=outside_dark_outline_color,
            outside_dark_outline_width=float(outside_dark_outline_width),
            hyphenate_before_scaling=hyphenate_before_scaling_val,
            detach_trailing_punctuation=detach_trailing_punctuation_val,
            auto_vertical_text=auto_vertical_text_val,
            hyphen_penalty=hyphen_penalty_val,
            hyphenation_min_word_length=hyphenation_min_word_length_val,
            badness_exponent=badness_exponent_val,
            padding_pixels=max(
                float(padding_horizontal_pixels_val),
                float(padding_vertical_pixels_val),
            ),
            padding_horizontal_pixels=float(padding_horizontal_pixels_val),
            padding_vertical_pixels=float(padding_vertical_pixels_val),
            supersampling_factor=int(supersampling_factor_val),
        ),
        output=UIOutputSettings(
            output_directory=output_directory_val,
            disable_auto_save=bool(disable_auto_save_val),
            embed_editor_package=bool(embed_editor_package_val),
            save_plain_image_copy=bool(save_plain_image_copy_val),
            output_format=out_fmt,
            jpeg_quality=jq,
            png_compression=pngc,
            image_upscale_mode=image_upscale_mode_val,
            image_upscale_factor=image_upscale_factor_val,
            image_upscale_model=image_upscale_model_val,
        ),
        general=UIGeneralSettings(
            verbose=verb,
            cleaning_only=cleaning_only_val,
            upscaling_only=upscaling_only_val,
            test_mode=test_mode_val,
            enable_web_search=enable_web_search_val,
            enable_code_execution=enable_code_execution_val,
            image_detail=image_detail_val,
            media_resolution=media_resolution_val,
            media_resolution_bubbles=media_resolution_bubbles_val,
            media_resolution_context=media_resolution_context_val,
            reasoning_effort=reasoning_effort_val,
            effort=effort_val,
            verbosity=verbosity_val,
            auto_scale=auto_scale_val,
        ),
        input_language=s_in_lang,
        output_language=s_out_lang,
        font_pack=s_font,
        batch_input_language=b_in_lang,
        batch_output_language=b_out_lang,
        batch_font_pack=b_font,
        batch_special_instructions=batch_special_instructions_val,
        batch_parallel_requests=int(batch_parallel_requests_val),
        batch_retry_failed_once=bool(batch_retry_failed_once_val),
        batch_previous_context_image_count=int(batch_previous_context_image_count_val),
        batch_previous_context_text_count=int(batch_previous_context_text_count_val),
        batch_previous_context_text_mode=str(batch_previous_context_text_mode_val),
    )

    # Convert UI state to dictionary for saving
    settings_dict = ui_state.to_save_dict()
    # Persist send_full_page_context from the single UI value (used for both single and batch flows).
    settings_dict["send_full_page_context"] = bool(send_full_page_context_val)
    settings_dict["parallel_inpaint_translation"] = bool(
        parallel_inpaint_translation_val
    )
    message = settings_manager.save_config(settings_dict)
    return ko(message)


def handle_reset_defaults_click(fonts_base_dir: Path) -> List[gr.update]:
    """Callback for the 'Reset Defaults' button. Uses dataclasses."""

    default_settings_dict = settings_manager.reset_to_defaults()
    default_ui_state = UIConfigState.from_dict(default_settings_dict)

    available_fonts, _ = utils.get_available_font_packs(fonts_base_dir)
    reset_single_font = default_ui_state.font_pack
    if reset_single_font not in available_fonts:
        reset_single_font = available_fonts[0] if available_fonts else None
    default_ui_state.font_pack = reset_single_font

    batch_reset_font = default_ui_state.batch_font_pack
    if batch_reset_font not in available_fonts:
        batch_reset_font = available_fonts[0] if available_fonts else None
    default_ui_state.batch_font_pack = batch_reset_font

    default_provider = default_ui_state.provider_settings.provider
    default_model_name = default_ui_state.llm_settings.model_name
    if not default_model_name:
        default_model_name = settings_manager.DEFAULT_SETTINGS["provider_models"].get(
            default_provider
        )
        default_ui_state.llm_settings.model_name = default_model_name

    if default_provider == "OpenRouter" or default_provider == "OpenAI-Compatible":
        default_models_choices = [default_model_name] if default_model_name else []
    else:
        default_models_choices = settings_manager.PROVIDER_MODELS.get(
            default_provider, []
        )

    gemini_visible = default_provider == "Google"
    openai_visible = default_provider == "OpenAI"
    anthropic_visible = default_provider == "Anthropic"
    xai_visible = default_provider == "xAI"
    deepseek_visible = default_provider == "DeepSeek"
    zai_visible = default_provider == "Z.ai"
    moonshot_visible = default_provider == "Moonshot AI"
    openrouter_visible = default_provider == "OpenRouter"
    compatible_visible = default_provider == "OpenAI-Compatible"
    (
        temp_update,
        top_p_update,
        top_k_update,
        _,  # max_tokens_update - unused (using saved default instead)
        enable_web_search_update,
        enable_code_execution_update,
        image_detail_update,
        media_resolution_update,
        media_resolution_bubbles_update,
        media_resolution_context_update,
        reasoning_effort_update,
        effort_update,
        verbosity_update,
    ) = utils.update_params_for_model(
        default_provider, default_model_name, default_ui_state.llm_settings.temperature
    )
    temp_val = temp_update.get("value", default_ui_state.llm_settings.temperature)
    temp_max = temp_update.get("maximum", 2.0)
    top_p_interactive = top_p_update.get("interactive", True)
    top_k_interactive = top_k_update.get("interactive", True)
    top_k_val = top_k_update.get("value", default_ui_state.llm_settings.top_k)
    is_reasoning = utils.is_reasoning_model(default_provider, default_model_name)
    max_tokens_val = 16384 if is_reasoning else 4096
    enable_web_search_visible = enable_web_search_update.get("visible", False)
    enable_code_execution_visible = enable_code_execution_update.get("visible", False)
    image_detail_visible = image_detail_update.get("visible", False)
    media_resolution_visible = media_resolution_update.get("visible", False)
    media_resolution_bubbles_visible = media_resolution_bubbles_update.get(
        "visible", False
    )
    media_resolution_context_visible = media_resolution_context_update.get(
        "visible", False
    )
    reasoning_visible = reasoning_effort_update.get("visible", False)
    reasoning_effort_val = reasoning_effort_update.get(
        "value", default_ui_state.general.reasoning_effort
    )
    effort_visible = effort_update.get("visible", False)
    effort_val = effort_update.get("value", default_ui_state.general.effort)
    verbosity_visible = verbosity_update.get("visible", False)
    verbosity_val = verbosity_update.get("value", default_ui_state.general.verbosity)

    return [
        default_ui_state.detection.confidence,
        default_ui_state.detection.conjoined_confidence,
        gr.update(
            value=default_ui_state.detection.panel_confidence,
            interactive=default_ui_state.detection.use_panel_sorting,
        ),
        default_ui_state.detection.seg_model,
        default_ui_state.detection.bubble_detector_model,
        default_ui_state.detection.conjoined_detection,
        default_ui_state.detection.use_osb_text_verification,
        default_ui_state.detection.use_panel_sorting,
        default_ui_state.llm_settings.reading_direction,
        default_ui_state.cleaning.thresholding_value,
        default_ui_state.cleaning.use_otsu_threshold,
        default_ui_state.cleaning.inpaint_colored_bubbles,
        gr.update(
            value=default_ui_state.cleaning.colored_bubble_inpainting_size,
            interactive=default_ui_state.cleaning.inpaint_colored_bubbles,
        ),
        gr.update(
            value=default_ui_state.cleaning.colored_bubble_inpainting_precision,
            interactive=default_ui_state.cleaning.inpaint_colored_bubbles,
        ),
        gr.update(
            value=default_ui_state.cleaning.colored_bubble_mask_kernel_size,
            interactive=default_ui_state.cleaning.inpaint_colored_bubbles,
        ),
        gr.update(
            value=default_ui_state.cleaning.colored_bubble_mask_dilation_offset,
            interactive=default_ui_state.cleaning.inpaint_colored_bubbles,
        ),
        default_ui_state.cleaning.roi_shrink_px,
        gr.update(value=default_provider),
        gr.update(
            value=default_ui_state.provider_settings.google_api_key,
            visible=gemini_visible,
        ),
        gr.update(value=default_ui_state.google_rate_limit.enabled),
        gr.update(
            value=default_ui_state.google_rate_limit.request_interval_seconds
        ),
        gr.update(value=default_ui_state.google_rate_limit.requests_per_minute),
        gr.update(value=default_ui_state.google_rate_limit.burst_requests),
        gr.update(
            value=default_ui_state.google_rate_limit.burst_cooldown_seconds
        ),
        gr.update(value=default_ui_state.google_rate_limit.cooldown_429_seconds),
        gr.update(value=default_ui_state.google_rate_limit.max_retries),
        gr.update(value=default_ui_state.google_rate_limit.hourly_429_limit),
        gr.update(value=default_ui_state.google_rate_limit.hourly_429_action),
        gr.update(value=default_ui_state.google_rate_limit.pause_429_minutes),
        gr.update(
            value=default_ui_state.provider_settings.openai_api_key,
            visible=openai_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.anthropic_api_key,
            visible=anthropic_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.xai_api_key, visible=xai_visible
        ),
        gr.update(
            value=default_ui_state.provider_settings.deepseek_api_key,
            visible=deepseek_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.zai_api_key,
            visible=zai_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.moonshot_api_key,
            visible=moonshot_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openrouter_api_key,
            visible=openrouter_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_compatible_url,
            visible=compatible_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_compatible_api_key,
            visible=compatible_visible,
        ),
        gr.update(choices=default_models_choices, value=default_model_name),
        gr.update(value=temp_val, maximum=temp_max),
        gr.update(
            value=default_ui_state.llm_settings.top_p, interactive=top_p_interactive
        ),
        gr.update(value=top_k_val, interactive=top_k_interactive),
        gr.update(value=max_tokens_val),
        gr.update(value=default_ui_state.llm_settings.translation_mode),
        gr.update(value=default_ui_state.llm_settings.ocr_method),
        default_ui_state.rendering.max_font_size,
        default_ui_state.rendering.emphasis_max_font_size,
        default_ui_state.rendering.min_font_size,
        default_ui_state.rendering.line_spacing_mult,
        default_ui_state.rendering.emphasis_line_spacing_mult,
        default_ui_state.rendering.vertical_line_spacing_mult,
        default_ui_state.rendering.vertical_emphasis_line_spacing_mult,
        default_ui_state.rendering.vertical_font_size_mult,
        default_ui_state.rendering.vertical_max_chars,
        default_ui_state.rendering.vertical_emphasis_max_chars,
        gr.update(value=tag_styles_to_table(default_ui_state.rendering.tag_style_table)),
        default_ui_state.rendering.use_subpixel_rendering,
        default_ui_state.rendering.font_hinting,
        default_ui_state.rendering.use_ligatures,
        default_ui_state.rendering.inside_light_text_color,
        default_ui_state.rendering.inside_light_outline_color,
        default_ui_state.rendering.inside_light_outline_width,
        default_ui_state.rendering.inside_dark_text_color,
        default_ui_state.rendering.inside_dark_outline_color,
        default_ui_state.rendering.inside_dark_outline_width,
        default_ui_state.rendering.outside_light_text_color,
        default_ui_state.rendering.outside_light_outline_color,
        default_ui_state.rendering.outside_light_outline_width,
        default_ui_state.rendering.outside_dark_text_color,
        default_ui_state.rendering.outside_dark_outline_color,
        default_ui_state.rendering.outside_dark_outline_width,
        default_ui_state.output.output_format,
        default_ui_state.output.jpeg_quality,
        default_ui_state.output.png_compression,
        default_ui_state.output.output_directory,
        default_ui_state.output.disable_auto_save,
        default_ui_state.output.embed_editor_package,
        default_ui_state.output.save_plain_image_copy,
        default_ui_state.general.verbose,
        default_ui_state.general.cleaning_only,
        default_ui_state.general.upscaling_only,
        default_ui_state.general.test_mode,
        default_ui_state.input_language,
        default_ui_state.output_language,
        gr.update(value=default_ui_state.font_pack),
        gr.update(value=default_ui_state.rendering.emphasis_font_dir),
        default_ui_state.batch_input_language,
        default_ui_state.batch_output_language,
        gr.update(value=default_ui_state.batch_font_pack),
        gr.update(
            value=default_ui_state.general.enable_web_search,
            visible=enable_web_search_visible,
        ),
        gr.update(
            value=default_ui_state.general.enable_code_execution,
            visible=enable_code_execution_visible,
        ),
        gr.update(
            value=default_ui_state.general.image_detail,
            visible=image_detail_visible,
            choices=image_detail_update.get("choices"),
            info=image_detail_update.get("info"),
        ),
        gr.update(
            value=default_ui_state.general.media_resolution,
            visible=media_resolution_visible,
        ),
        gr.update(
            value=default_ui_state.general.media_resolution_bubbles,
            visible=media_resolution_bubbles_visible,
        ),
        gr.update(
            value=default_ui_state.general.media_resolution_context,
            visible=media_resolution_context_visible,
        ),
        gr.update(value=reasoning_effort_val, visible=reasoning_visible),
        gr.update(value=effort_val, visible=effort_visible),
        gr.update(value=verbosity_val, visible=verbosity_visible),
        ko("Settings reset to defaults (API keys preserved)."),
        gr.update(value=default_ui_state.llm_settings.send_full_page_context),
        gr.update(value=default_ui_state.llm_settings.parallel_inpaint_translation),
        gr.update(value=default_ui_state.llm_settings.whiteout_conjoined_bubbles),
        gr.update(value=default_ui_state.llm_settings.upscale_method),
        default_ui_state.llm_settings.bubble_min_side_pixels,
        default_ui_state.llm_settings.context_image_max_side_pixels,
        default_ui_state.llm_settings.osb_min_side_pixels,
        gr.update(value=default_ui_state.rendering.hyphenate_before_scaling),
        gr.update(value=default_ui_state.rendering.detach_trailing_punctuation),
        gr.update(value=default_ui_state.rendering.auto_vertical_text),
        default_ui_state.rendering.padding_horizontal_pixels,
        default_ui_state.rendering.padding_vertical_pixels,
        default_ui_state.llm_settings.special_instructions or "",
        default_ui_state.batch_special_instructions or "",
        default_ui_state.outside_text.enabled,
        default_ui_state.outside_text.seed,
        gr.update(value=default_ui_state.outside_text.inpainting_method),
        gr.update(value=default_ui_state.outside_text.kontext_backend),
        default_ui_state.outside_text.flux_low_vram,
        default_ui_state.outside_text.flux_num_inference_steps,
        default_ui_state.outside_text.flux_kontext_prompt,
        default_ui_state.outside_text.flux_klein_prompt,
        default_ui_state.outside_text.flux_luminance_correction,
        default_ui_state.outside_text.flux_upscale_small_crops,
        gr.update(value=default_ui_state.outside_text.flux_group_regions),
        default_ui_state.outside_text.flux_residual_diff_threshold,
        default_ui_state.outside_text.osb_confidence,
        gr.update(value=default_ui_state.outside_text.enable_page_number_filtering),
        gr.update(
            value=default_ui_state.outside_text.page_filter_margin_threshold,
            interactive=default_ui_state.outside_text.enable_page_number_filtering,
        ),
        gr.update(
            value=default_ui_state.outside_text.page_filter_min_area_ratio,
            interactive=default_ui_state.outside_text.enable_page_number_filtering,
        ),
        default_ui_state.outside_text.huggingface_token,
        gr.update(value=default_ui_state.outside_text.osb_font_dir),
        gr.update(value=default_ui_state.outside_text.osb_emphasis_font_dir),
        default_ui_state.outside_text.osb_max_font_size,
        default_ui_state.outside_text.osb_emphasis_max_font_size,
        default_ui_state.outside_text.osb_min_font_size,
        default_ui_state.outside_text.osb_use_ligatures,
        default_ui_state.outside_text.osb_outline_width,
        default_ui_state.outside_text.osb_padding_pixels,
        default_ui_state.outside_text.osb_line_spacing,
        default_ui_state.outside_text.osb_vertical_line_spacing,
        default_ui_state.outside_text.osb_auto_vertical,
        default_ui_state.outside_text.osb_vertical_aspect_ratio,
        default_ui_state.outside_text.osb_vertical_max_chars,
        default_ui_state.outside_text.osb_tall_box_expand_aspect_ratio,
        default_ui_state.outside_text.osb_tall_box_expand_width_percent,
        default_ui_state.outside_text.osb_rotation_max_degrees,
        default_ui_state.outside_text.osb_use_subpixel_rendering,
        default_ui_state.outside_text.osb_font_hinting,
        default_ui_state.outside_text.bbox_expansion_percent,
        default_ui_state.outside_text.osb_render_expansion_narrow_multiplier,
        default_ui_state.outside_text.osb_render_expansion_aspect_ratio_threshold,
        default_ui_state.outside_text.osb_render_expansion_tiny_multiplier,
        default_ui_state.outside_text.osb_render_expansion_area_ratio_threshold * 100.0,
        default_ui_state.outside_text.text_box_proximity_ratio,
        gr.update(value=default_ui_state.output.image_upscale_mode),
        gr.update(
            value=default_ui_state.output.image_upscale_factor,
            interactive=default_ui_state.output.image_upscale_mode != "off",
        ),
        gr.update(
            value=default_ui_state.output.image_upscale_model,
            interactive=default_ui_state.output.image_upscale_mode != "off",
        ),
        default_ui_state.general.auto_scale,
        default_ui_state.batch_parallel_requests,
        default_ui_state.batch_retry_failed_once,
        default_ui_state.batch_previous_context_image_count,
        default_ui_state.batch_previous_context_text_count,
        default_ui_state.batch_previous_context_text_mode,
    ]


def handle_provider_change(provider: str, current_temp: float, ocr_method: str = "LLM"):
    """Handles changes in the provider selector."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()
    return utils.update_translation_ui(provider, current_temp, ocr_method)


def handle_output_format_change(output_format_value: str):
    """Handles changes in the output format radio button."""
    is_jpeg = output_format_value == "jpeg"
    is_png = output_format_value == "png"

    jpeg_interactive = not is_png
    png_interactive = not is_jpeg

    return gr.update(interactive=jpeg_interactive), gr.update(
        interactive=png_interactive
    )


def handle_refresh_resources_click(fonts_base_dir: Path):
    """Callback for the 'Refresh Models / Fonts' button."""
    return utils.refresh_models_and_fonts(fonts_base_dir)


def handle_apply_tag_style_font(table_value, tag_name: str, font_pack: str):
    """Apply a selected font pack to one tag row in the tag style table."""
    rows = tag_styles_to_table(table_value)
    normalized_tag = str(tag_name or "").upper().strip().strip("<>")
    selected_font = str(font_pack or "").strip()

    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        row_tag = str(row[0]).upper().strip().strip("<>")
        if row_tag == normalized_tag:
            while len(row) < 3:
                row.append("")
            row[2] = selected_font
            break

    return gr.update(value=rows)


def handle_unload_models_click():
    """Callback for the 'Unload Models' button."""
    try:
        from core.ml.model_manager import get_model_manager

        model_manager = get_model_manager()
        model_manager.unload_all()
        gr.Info("All models unloaded from memory successfully.")
    except Exception as e:
        gr.Error(f"Error unloading models: {str(e)}")


def handle_model_change(provider: str, model_name: Optional[str], current_temp: float):
    """Handles changes in the model name dropdown."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()
    return utils.update_params_for_model(provider, model_name, current_temp)


def handle_reasoning_effort_change(
    provider: str, model_name: Optional[str], reasoning_effort: Optional[str]
):
    """Updates temp/top_p slider interactivity when reasoning effort changes."""
    temp_ok, top_p_ok = utils.get_sampling_interactivity_for_effort(
        provider, model_name, reasoning_effort
    )
    return gr.update(interactive=temp_ok), gr.update(interactive=top_p_ok)


def handle_app_load(provider: str, url: str, key: Optional[str]):
    """Callback for the app.load event to fetch dynamic models."""
    return utils.initial_dynamic_fetch(provider, url, key)


def update_process_buttons(
    processing: bool, button_text_processing: str, button_text_idle: str
):
    """Toggle visibility and interactivity of processing buttons."""
    global CANCELLATION_MANAGER
    if not processing:
        CANCELLATION_MANAGER = None
    return (
        gr.update(
            interactive=not processing,
            value=button_text_processing if processing else button_text_idle,
        ),
        gr.update(visible=not processing),  # Clear button
        gr.update(visible=processing),  # Cancel button
        # Also update the other tab's buttons to prevent concurrent runs
        gr.update(interactive=not processing),
        gr.update(visible=not processing),
        gr.update(visible=processing),
    )


def cancel_process():
    """Signal that the current process should be cancelled."""
    if CANCELLATION_MANAGER:
        CANCELLATION_MANAGER.cancel()


def handle_thresholding_change(use_otsu_threshold: bool):
    """Handles changes in the Otsu thresholding checkbox to enable/disable thresholding_value slider."""
    return gr.update(interactive=not use_otsu_threshold)


def handle_hyphenation_change(hyphenate_before_scaling: bool):
    """Handles changes in the hyphenation checkbox to enable/disable hyphenation-related sliders."""
    return gr.update(interactive=hyphenate_before_scaling), gr.update(
        interactive=hyphenate_before_scaling
    )


def handle_cleaning_only_change(cleaning_only: bool):
    """Handles changes in the cleaning-only checkbox to disable upscaling-only and test mode if enabled."""
    return (
        gr.update(
            interactive=not cleaning_only, value=False if cleaning_only else None
        ),
        gr.update(
            interactive=not cleaning_only, value=False if cleaning_only else None
        ),
    )


def handle_upscaling_only_change(upscaling_only: bool):
    """Handles changes in the upscaling-only checkbox to disable cleaning-only and test mode if enabled."""
    return (
        gr.update(
            interactive=not upscaling_only, value=False if upscaling_only else None
        ),
        gr.update(
            interactive=not upscaling_only, value=False if upscaling_only else None
        ),
    )


def handle_test_mode_change(test_mode: bool):
    """Handles changes in the test mode checkbox to disable cleaning-only and upscaling-only if enabled."""
    return (
        gr.update(interactive=not test_mode, value=False if test_mode else None),
        gr.update(interactive=not test_mode, value=False if test_mode else None),
    )


def handle_conjoined_detection_change(_conjoined_detection: bool):
    """Handles changes in the conjoined detection checkbox to clear SAM cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_sam_cache()
    return gr.update(interactive=_conjoined_detection)


def handle_confidence_threshold_change(_confidence: float):
    """Handles changes in confidence threshold settings to clear YOLO cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_yolo_cache()
    return None


def handle_luminance_correction_change(_enabled: bool):
    """Handles changes in the flux luminance correction checkbox to clear inpaint cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_inpaint_cache()
    return None


def handle_ocr_method_change(
    ocr_method: str,
    input_language: str,
    original_language_state: str,
    batch_input_language: str,
    batch_original_language_state: str,
    provider: str,
    current_model: Optional[str],
    openai_compatible_url: str,
    openai_compatible_api_key: Optional[str],
):
    """Handles changes in OCR method selection."""
    import gradio as gr

    from core.caching import get_cache

    from . import layout, utils

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()

    updates = []

    # Filter provider selector based on OCR method
    available_providers = utils.get_available_providers(ocr_method)
    current_provider = (
        provider
        if provider in available_providers
        else available_providers[0] if available_providers else "Google"
    )
    provider_selector_update = gr.update(
        choices=available_providers, value=current_provider
    )
    updates.append(provider_selector_update)

    if ocr_method == "manga-ocr":
        if input_language != "Japanese":
            saved_language = input_language
        else:
            saved_language = original_language_state

        updates.append(
            gr.update(value="Japanese", choices=["Japanese"], interactive=False)
        )
        updates.append(saved_language)

        if batch_input_language != "Japanese":
            batch_saved_language = batch_input_language
        else:
            batch_saved_language = batch_original_language_state

        updates.append(
            gr.update(value="Japanese", choices=["Japanese"], interactive=False)
        )
        updates.append(batch_saved_language)

        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(value=0, interactive=False))
        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update())
        # Disable code execution checkbox (Gemini Flash only, disabled in text-only mode)
        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(interactive=False))
        # Disable media resolution dropdowns (no images sent to LLM in text-only mode)
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))

        # Trigger model list refresh for providers with dynamic model lists
        if current_provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="manga-ocr", current_model=current_model
            )
            updates.append(model_update)
        elif current_provider == "OpenAI-Compatible":
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif current_provider == "Z.ai":
            # For manga-ocr mode, show all Z.ai models (text-only models work)
            models = settings_manager.PROVIDER_MODELS.get("Z.ai", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Moonshot AI":
            # For manga-ocr mode, show all Moonshot AI models (text-only models work)
            models = settings_manager.PROVIDER_MODELS.get("Moonshot AI", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Moonshot AI")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        else:
            updates.append(gr.update())
    elif ocr_method == "paddleocr-vl":
        # Restrict to PaddleOCR-VL-1.5 supported languages; keep current if supported
        paddle_langs = layout.PADDLE_OCR_VL_LANGUAGES
        if input_language in paddle_langs:
            saved_language = original_language_state
            selected_lang = input_language
        else:
            saved_language = input_language
            selected_lang = (
                "Japanese" if "Japanese" in paddle_langs else paddle_langs[0]
            )

        updates.append(
            gr.update(value=selected_lang, choices=paddle_langs, interactive=True)
        )
        updates.append(saved_language)

        if batch_input_language in paddle_langs:
            batch_saved_language = batch_original_language_state
            batch_selected_lang = batch_input_language
        else:
            batch_saved_language = batch_input_language
            batch_selected_lang = (
                "Japanese" if "Japanese" in paddle_langs else paddle_langs[0]
            )

        updates.append(
            gr.update(value=batch_selected_lang, choices=paddle_langs, interactive=True)
        )
        updates.append(batch_saved_language)

        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(value=0, interactive=False))
        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update())
        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))

        # Model list refresh — same as manga-ocr (enables text-only providers)
        if current_provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="manga-ocr", current_model=current_model
            )
            updates.append(model_update)
        elif current_provider == "OpenAI-Compatible":
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif current_provider == "Z.ai":
            models = settings_manager.PROVIDER_MODELS.get("Z.ai", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Moonshot AI":
            models = settings_manager.PROVIDER_MODELS.get("Moonshot AI", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Moonshot AI")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        else:
            updates.append(gr.update())
    else:
        restored_language = (
            original_language_state if original_language_state else input_language
        )
        updates.append(
            gr.update(
                value=restored_language,
                choices=layout.SOURCE_LANGUAGES,
                interactive=True,
            )
        )
        updates.append(original_language_state)

        batch_restored_language = (
            batch_original_language_state
            if batch_original_language_state
            else batch_input_language
        )
        updates.append(
            gr.update(
                value=batch_restored_language,
                choices=layout.SOURCE_LANGUAGES,
                interactive=True,
            )
        )
        updates.append(batch_original_language_state)

        saved_settings = settings_manager.get_saved_settings()
        restored_send_full_page_context = saved_settings.get(
            "send_full_page_context", True
        )
        updates.append(
            gr.update(value=restored_send_full_page_context, interactive=True)
        )
        restored_previous_context_count = saved_settings.get(
            "batch_previous_context_image_count", 0
        )
        updates.append(
            gr.update(
                value=(
                    restored_previous_context_count
                    if restored_send_full_page_context
                    else 0
                ),
                interactive=restored_send_full_page_context,
            )
        )
        restored_parallel = saved_settings.get("parallel_inpaint_translation", False)
        updates.append(gr.update(value=restored_parallel, interactive=True))
        restored_whiteout = saved_settings.get("whiteout_conjoined_bubbles", True)
        updates.append(gr.update(value=restored_whiteout, interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))

        # Trigger model list refresh for providers with dynamic or filtered model lists
        if current_provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="LLM", current_model=current_model
            )
            updates.append(model_update)
        elif current_provider == "OpenAI-Compatible":
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif current_provider == "Z.ai":
            # For LLM OCR mode, only show Z.ai vision models
            models = [
                m for m in settings_manager.PROVIDER_MODELS.get("Z.ai", []) if "v" in m
            ]
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Moonshot AI":
            # For LLM OCR mode, only show Moonshot vision models
            all_models = settings_manager.PROVIDER_MODELS.get("Moonshot AI", [])
            models = [m for m in all_models if "kimi-k2." in m.lower()]
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Moonshot AI")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        else:
            updates.append(gr.update())

    updates.append(current_provider)

    return updates


def handle_translation_mode_change(translation_mode: str, current_ocr_method: str):
    """Handles changes in translation mode to enable/disable OCR method selection."""
    import gradio as gr

    if translation_mode == "one-step":
        if current_ocr_method in ("manga-ocr", "paddleocr-vl"):
            return gr.update(value="LLM", interactive=False)
        else:
            return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)
