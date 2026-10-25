from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import shutil
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from core.ai_image_region import AI_FILE_KEYS
from core.config import MangaTranslatorConfig, RenderingConfig
from core.image.image_utils import save_image_with_compression, upscale_image
from core.text.text_processing import extract_translation_tag
from core.text.text_renderer import render_text_skia
from utils.logging import log_message

SESSION_VERSION = 1
INVALID_TEXT_MARKERS = {
    "[OCR FAILED]",
    "[Empty response / no content]",
}
FONT_PATH_KEYS = {
    "font_dir",
    "emphasis_font_dir",
    "fallback_font_dir",
    "bold_font_dir",
    "italic_font_dir",
    "bold_italic_font_dir",
}
_UPLOADED_RELATIVE_PATH_PREFIX = "__MT_RELATIVE_PATH__"
_REGION_PATCH_CACHE_MAX_ITEMS = 96
_REGION_PATCH_CACHE: "OrderedDict[str, Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]]" = OrderedDict()
_REGION_PATCH_CACHE_LOCK = threading.Lock()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_temporary_upload_path(path: Path) -> bool:
    try:
        resolved = Path(path).resolve()
    except Exception:
        resolved = Path(path)
    lowered_parts = {part.lower() for part in resolved.parts}
    if "gradio" in lowered_parts:
        return True
    return resolved.name.startswith(_UPLOADED_RELATIVE_PATH_PREFIX)


def _resolve_repo_relative_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    repo_root = _repo_root()
    parts = list(path.parts)
    lower_parts = [part.casefold() for part in parts]

    def foreign_repo_candidate() -> Optional[str]:
        for marker in ("mangatranslator",):
            if marker in lower_parts:
                marker_index = lower_parts.index(marker)
                if marker_index + 1 < len(parts):
                    candidate = (repo_root / Path(*parts[marker_index + 1 :])).resolve()
                    if candidate.exists():
                        return str(candidate)
        for marker in ("fonts", "models", "editor_sessions", "editor_imports"):
            if marker in lower_parts:
                marker_index = lower_parts.index(marker)
                candidate = (repo_root / Path(*parts[marker_index:])).resolve()
                if candidate.exists():
                    return str(candidate)
        return None

    if path.is_absolute():
        if path.exists():
            return str(path)
        candidate = foreign_repo_candidate()
        if candidate:
            return candidate
        return str(path)
    candidate = foreign_repo_candidate()
    if candidate:
        return candidate
    repo_candidate = (repo_root / path).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    return text


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


def _path_contains_font(path: Path) -> bool:
    try:
        if any(part.casefold() == "_backup" for part in path.parts):
            return False
        if path.is_file():
            return path.suffix.lower() in {".ttf", ".otf", ".ttc"}
        if path.is_dir():
            return any(
                item.suffix.lower() in {".ttf", ".otf", ".ttc"}
                for item in path.rglob("*")
                if item.is_file()
                and all(part.casefold() != "_backup" for part in item.parts)
            )
    except Exception:
        return False
    return False


def _default_editor_font_path() -> str:
    fonts_root = _repo_root() / "fonts"
    try:
        from ui import settings_manager

        saved_settings = settings_manager.get_saved_settings()
    except Exception:
        saved_settings = {}
    for key in (
        "font_pack",
        "batch_font_pack",
        "outside_text_osb_font_pack",
        "outside_text_osb_emphasis_font_pack",
    ):
        value = str(saved_settings.get(key) or "").strip()
        if not value:
            continue
        candidates = []
        candidate = Path(value)
        candidates.append(candidate)
        if not candidate.is_absolute():
            candidates.append(fonts_root / value)
        for path in candidates:
            try:
                resolved = path.expanduser().resolve()
            except Exception:
                resolved = path
            if _path_contains_font(resolved):
                return str(resolved)

    preferred = (
        fonts_root / "나눔고딕",
        fonts_root / "Nanum_Gothic",
        fonts_root / "Gowun_Dodum",
        fonts_root / "Noto_Serif_KR",
        fonts_root / "Komika Hand",
    )
    for path in preferred:
        if _path_contains_font(path):
            return str(path.resolve())
    if fonts_root.exists():
        for path in sorted(fonts_root.iterdir(), key=lambda item: item.name.casefold()):
            if _path_contains_font(path):
                return str(path.resolve())
    return ""


def _ensure_render_font_path(render: Dict[str, Any], *, persist: bool = True) -> str:
    _resolve_render_font_paths(render)
    for key in FONT_PATH_KEYS - {"font_dir"}:
        value = str(render.get(key) or "").strip()
        if value and not _path_contains_font(Path(value)):
            render[key] = None
    font_dir = str(render.get("font_dir") or "").strip()
    if font_dir and _path_contains_font(Path(font_dir)):
        return font_dir
    fallback = _default_editor_font_path()
    if fallback and persist:
        render["font_dir"] = fallback
    return fallback


def resolve_tag_font_path(render_tag: Any, tag_styles: Dict[str, Any]) -> Optional[str]:
    tag = str(render_tag or "").strip().upper()
    style = tag_styles.get(tag)
    value = (
        style.get("font_dir") or style.get("font_pack")
        if isinstance(style, dict)
        else getattr(style, "font_dir", None)
    )
    if not value:
        return None
    target = Path(_resolve_repo_relative_path(value))
    if not target.is_absolute():
        target = _repo_root() / "fonts" / target
    return str(target.resolve()) if _path_contains_font(target) else None


def apply_region_tag_font(region: Dict[str, Any], tag_styles: Dict[str, Any]) -> bool:
    """Sync only explicitly automatic fonts; legacy and manual choices stay intact."""
    if region.get("font_manual") is not False or region.get("ai_image"):
        return False
    target = resolve_tag_font_path(region.get("render_tag"), tag_styles)
    if target is None:
        return False
    render = region.setdefault("render", {})
    if str(render.get("font_dir") or "") == target:
        return False
    render["font_dir"] = target
    return True


def _resolve_render_font_paths(render: Dict[str, Any]) -> Dict[str, Any]:
    for key in FONT_PATH_KEYS:
        if key in render and render.get(key):
            render[key] = _resolve_repo_relative_path(render[key])
    return render


def default_edit_session_path(output_path: Path) -> Path:
    output_path = Path(output_path).resolve()
    path_key = str(output_path).casefold()
    digest = hashlib.sha256(path_key.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    parent_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", output_path.parent.name).strip(" ._")
    parent_stem = parent_stem or "output"
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", output_path.stem).strip(" ._")
    stem = stem or "image"
    return (
        _repo_root()
        / "editor_sessions"
        / "translation_sessions"
        / f"{parent_stem}_{digest}"
        / f"{stem}_edit_session.json"
    )


def legacy_edit_session_path(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}_edit_session.json")


def _asset_paths(session_path: Path) -> Tuple[Path, Path, Path]:
    session_path = session_path.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    parent_name = session_path.parent.name or "output"
    assets_dir = repo_root / "editor_sessions" / parent_name / session_path.stem
    base_image_path = assets_dir / "base_cleaned.png"
    edited_image_path = session_path.with_name(f"{session_path.stem}_edited.png")
    return assets_dir, base_image_path, edited_image_path


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


def _normalize_session_paths(
    session_path: Path,
    session: Dict[str, Any],
    *,
    persist_font_fallback: bool = False,
) -> Dict[str, Any]:
    path_keys = (
        "source_image_path",
        "overwrite_target_path",
        "original_snapshot_path",
        "translated_image_path",
        "rendered_snapshot_path",
        "base_image_path",
        "inpainted_image_path",
        "inpaint_source_path",
        "inpaint_mask_path",
        "edited_image_path",
        "assets_dir",
        "paint_overlay_path",
        "restore_mask_path",
        "text_cover_mask_path",
    )
    for key in path_keys:
        resolved = _resolve_stored_path(session_path, session.get(key))
        if resolved is not None:
            session[key] = str(resolved)
    for region in session.get("regions", []):
        resolved = _resolve_stored_path(session_path, region.get("mask_path"))
        if resolved is not None:
            region["mask_path"] = str(resolved)
        render = region.get("render")
        if isinstance(render, dict):
            _ensure_render_font_path(render, persist=persist_font_fallback)
        baseline = region.get("render_baseline")
        if isinstance(baseline, dict):
            baseline_render = baseline.get("render")
            if isinstance(baseline_render, dict):
                _ensure_render_font_path(
                    baseline_render,
                    persist=persist_font_fallback,
                )
    return session


def repair_session_assets(
    session_path: Any,
    session: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    path = Path(str(session_path or "")).resolve()
    if session is None:
        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)
    session = _normalize_session_paths(path, session)
    warnings: List[str] = []
    assets_dir, default_base_path, default_edited_path = _asset_paths(path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    session["assets_dir"] = str(assets_dir)

    original_snapshot = assets_dir / "source_original.png"
    source_path = _resolve_stored_path(path, session.get("original_snapshot_path"))
    if source_path is None or not source_path.exists():
        source_path = _resolve_stored_path(path, session.get("source_image_path"))
    if source_path is not None and source_path.exists() and not original_snapshot.exists():
        try:
            with Image.open(source_path) as image:
                image.convert("RGBA").save(original_snapshot)
        except Exception as exc:
            warnings.append(f"Could not preserve original image: {exc}")
    if original_snapshot.exists():
        session["original_snapshot_path"] = str(original_snapshot.resolve())

    rendered_snapshot = assets_dir / "rendered_snapshot.png"
    rendered_source = _resolve_stored_path(path, session.get("rendered_snapshot_path"))
    if rendered_source is None or not rendered_source.exists():
        rendered_source = _resolve_stored_path(path, session.get("translated_image_path"))
    if (
        rendered_source is not None
        and rendered_source.exists()
        and not rendered_snapshot.exists()
    ):
        try:
            with Image.open(rendered_source) as image:
                image.convert("RGBA").save(rendered_snapshot)
        except Exception as exc:
            warnings.append(f"Could not preserve rendered image: {exc}")
    if rendered_source is not None and rendered_source.exists():
        session["rendered_snapshot_path"] = str(rendered_source.resolve())
    elif rendered_snapshot.exists():
        session["rendered_snapshot_path"] = str(rendered_snapshot.resolve())

    base_path = _resolve_stored_path(path, session.get("base_image_path"))
    if (
        base_path is not None
        and base_path.exists()
        and base_path.resolve() != default_base_path.resolve()
    ):
        try:
            with Image.open(base_path) as image:
                image.convert("RGBA").save(default_base_path)
            base_path = default_base_path
        except Exception as exc:
            warnings.append(f"Could not preserve cleaned editor base: {exc}")
    if base_path is None or not base_path.exists():
        fallback_keys = (
            "rendered_snapshot_path",
            "translated_image_path",
            "original_snapshot_path",
            "source_image_path",
        )
        fallback = next(
            (
                candidate
                for key in fallback_keys
                if (candidate := _resolve_stored_path(path, session.get(key))) is not None
                and candidate.exists()
            ),
            None,
        )
        if fallback is not None:
            with Image.open(fallback) as image:
                image.convert("RGBA").save(default_base_path)
            base_path = default_base_path
            warning = (
                "The cleaned editor base was missing and was recovered from the best available image."
            )
            warnings.append(warning)
            session["asset_recovery_warning"] = warning
    if base_path is not None and base_path.exists():
        session["base_image_path"] = str(base_path.resolve())

    for index, region in enumerate(session.get("regions", []), start=1):
        for key in AI_FILE_KEYS:
            stored = _resolve_stored_path(path, region.get(key))
            if stored is not None and stored.is_file():
                image_id = re.sub(r"[^a-zA-Z0-9_-]", "", str((region.get("ai_image") or {}).get("id") or region.get("id")))[:80]
                target = assets_dir / f"ai_{image_id}_{key[3:]}.png"
                try:
                    if stored.resolve() != target.resolve() and not target.is_file():
                        shutil.copy2(stored, target)
                    region[key] = str(target.resolve())
                except OSError as exc:
                    warnings.append(f"Could not preserve AI lettering {index}: {exc}")
                    region[key] = str(stored.resolve())
        mask_path = _resolve_stored_path(path, region.get("mask_path"))
        if mask_path is None or not mask_path.exists():
            continue
        target_mask = assets_dir / f"{region.get('id') or f'r{index:03d}'}_mask.png"
        if mask_path.resolve() != target_mask.resolve():
            try:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.ndim > 2:
                        mask = mask[..., 0]
                    cv2.imwrite(str(target_mask), mask)
                    mask_path = target_mask
            except Exception as exc:
                warnings.append(f"Could not preserve mask {index}: {exc}")
        region["mask_path"] = str(mask_path.resolve())

    edited_path = _resolve_stored_path(path, session.get("edited_image_path"))
    session["edited_image_path"] = str((edited_path or default_edited_path).resolve())
    save_session(path, session)
    return session, warnings


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _mask_digest(mask_path: Any) -> str:
    path = Path(str(mask_path or ""))
    if not path.exists() or not path.is_file():
        return ""
    try:
        import hashlib

        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return ""


def _region_render_signature(region: Dict[str, Any]) -> Dict[str, Any]:
    render = deepcopy(region.get("render", {}))
    if isinstance(render, dict):
        _ensure_render_font_path(render)
    signature = {
        "enabled": bool(region.get("enabled", True)),
        "type": region.get("type", "region"),
        "translation": str(region.get("translation") or ""),
        "bbox": _bbox_to_list(region.get("bbox")),
        "rotation_deg": float(region.get("rotation_deg") or 0.0),
        "vertical_stack": bool(region.get("vertical_stack", False)),
        "resolved_orientation": str(region.get("resolved_orientation") or ""),
        "font_color": region.get("font_color") or "#000000",
        "text_color_rgb": _jsonable(region.get("text_color_rgb")),
        "bubble_color_bgr": _jsonable(region.get("bubble_color_bgr")),
        "use_mask": bool(region.get("use_mask", False)),
        "mask_digest": _mask_digest(region.get("mask_path")) if region.get("use_mask") else "",
        "render": _jsonable(render),
    }
    if region.get("ai_image"):
        signature["ai_image"] = {key: _jsonable(region["ai_image"].get(key)) for key in ("id", "box")}
    return signature


def _auto_vertical_candidate(region: Dict[str, Any]) -> bool:
    if bool(region.get("vertical_stack", False)):
        return False
    render = region.get("render")
    if not isinstance(render, dict) or not bool(render.get("auto_vertical_text", False)):
        return False
    try:
        x1, y1, x2, y2 = _bbox_to_list(region.get("bbox"))
        width = max(1.0, float(x2) - float(x1))
        height = max(1.0, float(y2) - float(y1))
    except Exception:
        return False
    if height / width < 1.7:
        return False
    text = str(region.get("translation") or "")
    _tag, text = extract_translation_tag(text)
    text = re.sub(r"(\*{1,3})(.*?)\1", r"\2", text)
    limit_chars = sum(1 for ch in re.sub(r"\s+", " ", text.strip()) if ch.isalnum())
    if limit_chars <= 0:
        return False
    try:
        max_chars = int(render.get("vertical_max_chars") or 4)
    except (TypeError, ValueError):
        max_chars = 4
    return limit_chars <= max(0, max_chars)


def _orientation_candidate_region(
    region: Dict[str, Any],
    orientation: str,
) -> Dict[str, Any]:
    candidate = deepcopy(region)
    candidate["vertical_stack"] = orientation == "vertical"
    candidate["resolved_orientation"] = orientation
    render = candidate.setdefault("render", {})
    if isinstance(render, dict):
        render["auto_vertical_text"] = False
        render["compare_forced_vertical_layout"] = False
        render["force_compare_vertical_text"] = False
    return candidate


def _score_orientation_against_rendered_snapshot(
    session: Dict[str, Any],
    region: Dict[str, Any],
    index: int,
    orientation: str,
) -> Optional[float]:
    target_path = Path(
        session.get("rendered_snapshot_path")
        or session.get("translated_image_path")
        or ""
    )
    if not target_path.is_file():
        return None
    try:
        base_path = _editor_base_image_path(session)
        with Image.open(base_path) as base_image:
            clean_base = base_image.convert("RGBA")
        with Image.open(target_path) as target_image:
            target = target_image.convert("RGBA")
        if target.size != clean_base.size:
            target = target.resize(clean_base.size, Image.Resampling.LANCZOS)

        candidate_region = _orientation_candidate_region(region, orientation)
        errors: List[str] = []
        rendered = _render_region_onto(
            clean_base.copy(),
            candidate_region,
            index,
            errors,
        )
        if errors:
            return None

        left, top, right, bottom = _expanded_union_bbox(region, clean_base.size)
        if right <= left or bottom <= top:
            return None

        base_crop = np.asarray(clean_base.crop((left, top, right, bottom)).convert("RGBA"), dtype=np.int16)
        target_crop = np.asarray(target.crop((left, top, right, bottom)).convert("RGBA"), dtype=np.int16)
        rendered_crop = np.asarray(rendered.crop((left, top, right, bottom)).convert("RGBA"), dtype=np.int16)

        target_delta = np.max(np.abs(target_crop[..., :3] - base_crop[..., :3]), axis=2)
        rendered_delta = np.max(np.abs(rendered_crop[..., :3] - base_crop[..., :3]), axis=2)
        mask = np.logical_or(target_delta > 2, rendered_delta > 2)
        if np.any(mask):
            mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
            mask_u8 = cv2.dilate(mask_u8, np.ones((3, 3), dtype=np.uint8), iterations=1)
            mask = mask_u8 > 0
        else:
            mask = np.ones(target_delta.shape, dtype=bool)

        diff = np.abs(rendered_crop[..., :3] - target_crop[..., :3])
        return float(diff[mask].mean()) if np.any(mask) else float(diff.mean())
    except Exception:
        return None


def _infer_orientation_from_rendered_snapshot(
    session: Dict[str, Any],
    region: Dict[str, Any],
    index: int,
) -> str:
    horizontal_score = _score_orientation_against_rendered_snapshot(
        session, region, index, "horizontal"
    )
    vertical_score = _score_orientation_against_rendered_snapshot(
        session, region, index, "vertical"
    )
    if horizontal_score is None or vertical_score is None:
        return ""
    # Require a small margin so nearly-identical punctuation-only boxes do not flap.
    margin = max(1.0, min(horizontal_score, vertical_score) * 0.03)
    if vertical_score + margin < horizontal_score:
        return "vertical"
    if horizontal_score + margin < vertical_score:
        return "horizontal"
    return ""


def _repair_auto_vertical_regions(session: Dict[str, Any]) -> bool:
    changed = False
    dirty_region_ids = {
        str(region_id)
        for region_id in session.get("editor_dirty_region_ids", []) or []
        if str(region_id)
    }
    for index, region in enumerate(session.get("regions", []) or [], start=1):
        region_id = str(region.get("id") or "")
        region_is_dirty = region_id in dirty_region_ids
        orientation = str(region.get("resolved_orientation") or "").lower()
        render = region.get("render")
        should_infer_from_snapshot = (
            not region_is_dirty
            and (
                orientation not in {"vertical", "horizontal"}
                or (
                    isinstance(render, dict)
                    and bool(render.get("auto_vertical_text", False))
                )
            )
        )
        if should_infer_from_snapshot:
            orientation = _resolve_auto_vertical_orientation(session, region, index)
        if orientation in {"vertical", "horizontal"}:
            if region.get("resolved_orientation") != orientation:
                region["resolved_orientation"] = orientation
                changed = True
            baseline = region.get("render_baseline")
            if (
                not region_is_dirty
                and isinstance(baseline, dict)
                and baseline.get("resolved_orientation") != orientation
            ):
                baseline["resolved_orientation"] = orientation
                changed = True
            target_vertical = orientation == "vertical"
            if bool(region.get("vertical_stack", False)) != target_vertical:
                region["vertical_stack"] = target_vertical
                changed = True
            render = region.get("render")
            if isinstance(render, dict):
                for key, value in {
                    "auto_vertical_text": False,
                    "compare_forced_vertical_layout": False,
                    "force_compare_vertical_text": False,
                }.items():
                    if render.get(key) != value:
                        render[key] = value
                        changed = True
            baseline = region.get("render_baseline")
            if not region_is_dirty and isinstance(baseline, dict):
                if bool(baseline.get("vertical_stack", False)) != target_vertical:
                    baseline["vertical_stack"] = target_vertical
                    changed = True
                baseline_render = baseline.get("render")
                if isinstance(baseline_render, dict):
                    for key, value in {
                        "auto_vertical_text": False,
                        "compare_forced_vertical_layout": False,
                        "force_compare_vertical_text": False,
                    }.items():
                        if baseline_render.get(key) != value:
                            baseline_render[key] = value
                            changed = True
    return changed


def _resolve_auto_vertical_orientation(
    session: Dict[str, Any],
    region: Dict[str, Any],
    index: int,
) -> str:
    if bool(region.get("vertical_stack", False)):
        return "vertical"
    render = region.get("render")
    if not isinstance(render, dict) or not bool(render.get("auto_vertical_text", False)):
        return "horizontal"
    text = str(region.get("translation") or "")
    if not text.strip():
        return "horizontal"
    snapshot_orientation = _infer_orientation_from_rendered_snapshot(
        session, region, index
    )
    if snapshot_orientation:
        return snapshot_orientation
    font_dir = _ensure_render_font_path(render, persist=False)
    if not font_dir:
        return ""
    try:
        base_path = _editor_base_image_path(session)
        with Image.open(base_path) as base_image:
            image = base_image.convert("RGBA")
        mask = (
            _load_mask(region.get("mask_path"))
            if bool(region.get("use_mask")) and not _region_geometry_changed(region)
            else None
        )
        result = render_text_skia(
            pil_image=image,
            text=text,
            bbox=_safe_int_bbox(region.get("bbox"), image.size),
            font_dir=font_dir,
            cleaned_mask=mask,
            bubble_color_bgr=_bgr_tuple(region.get("bubble_color_bgr"), (255, 255, 255)),
            config=_render_config_from_region(region),
            verbose=False,
            bubble_id=str(index),
            rotation_deg=float(region.get("rotation_deg") or 0.0),
            vertical_stack=False,
            text_color_rgb=_normalize_rgb(region.get("font_color")),
            raise_on_safe_error=False,
            layout_only=True,
            auto_vertical_ignore_word_count=True,
        )
        orientation = str(result.info.get("orientation") or "").lower()
        if orientation in {"vertical", "horizontal"}:
            return orientation
    except Exception:
        pass
    return ""


def _repair_copied_added_region_baseline(region: Dict[str, Any]) -> bool:
    if str(region.get("type") or "") not in {"added_text", "ocr_text"}:
        return False
    baseline = region.get("render_baseline")
    if not isinstance(baseline, dict):
        return False
    if str(baseline.get("type") or "") in {"added_text", "ocr_text"}:
        return False
    region.pop("render_baseline", None)
    return True


def _ensure_region_baselines(session: Dict[str, Any]) -> bool:
    changed = False
    dirty_region_ids = {
        str(region_id)
        for region_id in session.get("editor_dirty_region_ids", []) or []
        if str(region_id)
    }
    for region in session.get("regions", []):
        render = region.get("render")
        if isinstance(render, dict):
            _ensure_render_font_path(render, persist=False)
        if _repair_copied_added_region_baseline(region):
            changed = True
            region_id = str(region.get("id") or "")
            if region_id:
                dirty_region_ids.add(region_id)
        if not isinstance(region.get("render_baseline"), dict):
            region["render_baseline"] = _region_render_signature(region)
            changed = True
    if sorted(dirty_region_ids) != list(session.get("editor_dirty_region_ids", []) or []):
        session["editor_dirty_region_ids"] = sorted(dirty_region_ids)
        changed = True
    return changed


def _region_is_baseline(region: Dict[str, Any]) -> bool:
    baseline = region.get("render_baseline")
    if not isinstance(baseline, dict):
        return True
    current = _region_render_signature(region)
    return _jsonable(current) == _jsonable(baseline)


def _bbox_to_list(value: Any) -> List[float]:
    try:
        vals = [float(v) for v in value]
        if len(vals) == 4:
            return vals
    except Exception:
        pass
    return [0.0, 0.0, 1.0, 1.0]


def _safe_int_bbox(bbox: Any, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = _bbox_to_list(bbox)
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    return left, top, right, bottom


def _region_text_transform(
    region: Dict[str, Any], *, apply_font_compensation: bool = False
) -> Tuple[float, float, float, float]:
    """Return the editable text-layer scale, skew and perspective values."""
    render = region.get("render", {})
    if not isinstance(render, dict):
        render = {}

    def number(key: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(render.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    font_name = Path(str(render.get("font_dir") or "")).stem.casefold()
    width_factor = 0.9 if apply_font_compensation and font_name.startswith("hy피오피") else 1.0
    return (
        number("transform_scale_x", 1.0, 0.1, 4.0) * width_factor,
        number("transform_scale_y", 1.0, 0.1, 4.0),
        number("transform_skew_x_deg", 0.0, -60.0, 60.0),
        number("transform_perspective_y", 0.0, -0.8, 0.8),
    )


def _region_text_transform_matrix(
    region: Dict[str, Any],
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox or tuple(_bbox_to_list(region.get("bbox")))
    height = max(1.0, float(y2) - float(y1))
    center_x = (float(x1) + float(x2)) / 2.0
    center_y = (float(y1) + float(y2)) / 2.0
    scale_x, scale_y, skew_deg, perspective_y = _region_text_transform(region, apply_font_compensation=True)
    rotation = math.radians(float(region.get("rotation_deg") or 0.0))
    cosine, sine = math.cos(rotation), math.sin(rotation)
    translate_to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.array(
        [
            [scale_x, math.tan(math.radians(skew_deg)), 0.0],
            [0.0, scale_y, 0.0],
            [0.0, perspective_y / height, 1.0],
        ],
        dtype=np.float64,
    )
    rotate = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translate_back = np.array(
        [[1.0, 0.0, center_x], [0.0, 1.0, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return translate_back @ rotate @ transform @ translate_to_origin


def _warp_rgba_text_layer(layer: Image.Image, matrix: np.ndarray) -> Image.Image:
    rgba = np.asarray(layer.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    premultiplied = np.concatenate((rgba[..., :3] * alpha, rgba[..., 3:4]), axis=2)
    warped = cv2.warpPerspective(
        premultiplied,
        matrix,
        layer.size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    warped_alpha = warped[..., 3:4] / 255.0
    rgb = np.zeros_like(warped[..., :3])
    np.divide(warped[..., :3], warped_alpha, out=rgb, where=warped_alpha > 1e-6)
    result = np.concatenate((rgb, warped[..., 3:4]), axis=2)
    return Image.fromarray(np.clip(np.rint(result), 0, 255).astype(np.uint8), "RGBA")


def _baseline_bbox(region: Dict[str, Any]) -> Optional[List[float]]:
    baseline = region.get("render_baseline")
    if not isinstance(baseline, dict):
        return None
    try:
        return _bbox_to_list(baseline.get("bbox"))
    except Exception:
        return None


def _expanded_union_bbox(
    region: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    boxes = [_bbox_to_list(region.get("bbox"))]
    baseline_box = _baseline_bbox(region)
    if baseline_box is not None:
        boxes.append(baseline_box)
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    render = region.get("render", {})
    max_font = 24.0
    if isinstance(render, dict):
        try:
            max_font = float(
                render.get("font_size")
                or render.get("max_font_size")
                or max_font
            )
        except Exception:
            max_font = 24.0
    pad = max(16.0, max_font * 4.0, width * 0.50, height * 0.50)
    return _safe_int_bbox([x1 - pad, y1 - pad, x2 + pad, y2 + pad], image_size)


def _region_geometry_changed(region: Dict[str, Any], tolerance: float = 1.0) -> bool:
    baseline_box = _baseline_bbox(region)
    if baseline_box is None:
        return False
    current_box = _bbox_to_list(region.get("bbox"))
    return any(
        abs(float(current) - float(baseline)) > tolerance
        for current, baseline in zip(current_box, baseline_box)
    )


def _baseline_region(region: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    baseline = region.get("render_baseline")
    if not isinstance(baseline, dict):
        return None
    restored = deepcopy(region)
    for key in (
        "enabled",
        "type",
        "translation",
        "bbox",
        "rotation_deg",
        "vertical_stack",
        "font_color",
        "text_color_rgb",
        "bubble_color_bgr",
        "use_mask",
        "render",
    ):
        if key in baseline:
            restored[key] = deepcopy(baseline[key])
    return restored


def _image_cache_identity(path: Path, image_size: Tuple[int, int]) -> str:
    try:
        stat = path.stat()
        return (
            f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|"
            f"{image_size[0]}x{image_size[1]}"
        )
    except Exception:
        return f"{path}|{image_size[0]}x{image_size[1]}"


def _region_patch_cache_key(base_identity: str, region: Dict[str, Any]) -> str:
    signature = _region_render_signature(region)
    return base_identity + "|" + json.dumps(
        _jsonable(signature),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _cached_region_patch(
    clean_base: Image.Image,
    base_identity: str,
    region: Dict[str, Any],
    index: int,
) -> Tuple[
    Optional[Tuple[Tuple[int, int, int, int], np.ndarray, np.ndarray]],
    List[str],
]:
    cache_key = _region_patch_cache_key(base_identity, region)
    with _REGION_PATCH_CACHE_LOCK:
        cached = _REGION_PATCH_CACHE.get(cache_key)
        if cached is not None:
            _REGION_PATCH_CACHE.move_to_end(cache_key)
            box, mask, pixels = cached
            return (box, mask.copy(), pixels.copy()), []

    errors: List[str] = []
    rendered = _render_region_onto(
        clean_base.copy(),
        deepcopy(region),
        index,
        errors,
    )
    if errors:
        return None, errors
    clean_arr = np.asarray(clean_base.convert("RGBA"), dtype=np.uint8)
    rendered_arr = np.asarray(rendered.convert("RGBA"), dtype=np.uint8)
    diff = np.max(
        np.abs(
            rendered_arr[..., :3].astype(np.int16)
            - clean_arr[..., :3].astype(np.int16)
        ),
        axis=2,
    )
    changed = diff > 2
    if not np.any(changed):
        return None, []
    ys, xs = np.where(changed)
    left = max(0, int(xs.min()) - 2)
    top = max(0, int(ys.min()) - 2)
    right = min(rendered_arr.shape[1], int(xs.max()) + 3)
    bottom = min(rendered_arr.shape[0], int(ys.max()) + 3)
    patch = (
        (left, top, right, bottom),
        changed[top:bottom, left:right].copy(),
        rendered_arr[top:bottom, left:right].copy(),
    )
    with _REGION_PATCH_CACHE_LOCK:
        _REGION_PATCH_CACHE[cache_key] = patch
        _REGION_PATCH_CACHE.move_to_end(cache_key)
        while len(_REGION_PATCH_CACHE) > _REGION_PATCH_CACHE_MAX_ITEMS:
            _REGION_PATCH_CACHE.popitem(last=False)
    box, mask, pixels = patch
    return (box, mask.copy(), pixels.copy()), []


def _is_invalid_translation(text: str, provider: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    provider_prefixes = (f"[{provider}:", f"[{provider}-") if provider else ()
    return (
        value.startswith("API Error")
        or value.startswith("[Translation Error")
        or value.startswith(provider_prefixes)
        or value in INVALID_TEXT_MARKERS
    )


def _save_mask_asset(mask: Any, assets_dir: Path, region_id: str) -> Optional[str]:
    if mask is None:
        return None
    try:
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)
        if not np.any(mask_np):
            return None
        mask_path = assets_dir / f"{region_id}_mask.png"
        cv2.imwrite(str(mask_path), mask_np)
        return str(mask_path)
    except Exception:
        return None


def _load_mask(mask_path: Optional[str]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    path = Path(mask_path)
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = mask[..., 0]
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _normalize_rgb(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) == 7:
            try:
                return (
                    int(text[1:3], 16),
                    int(text[3:5], 16),
                    int(text[5:7], 16),
                )
            except ValueError:
                return None
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(int(max(0, min(255, round(float(v))))) for v in value[:3])  # type: ignore[return-value]
        except Exception:
            return None
    return None


def _rgb_to_hex(value: Any, fallback: str = "#000000") -> str:
    rgb = _normalize_rgb(value)
    if rgb is None:
        return fallback
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _bgr_tuple(value: Any, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(int(max(0, min(255, round(float(v))))) for v in value[:3])  # type: ignore[return-value]
        except Exception:
            return fallback
    return fallback


def _snapshot_region_config(
    config: MangaTranslatorConfig,
    *,
    is_outside_text: bool,
    font_dir: str,
    min_font_size: int,
    max_font_size: int,
    line_spacing: float,
    vertical_line_spacing: Optional[float],
    padding_pixels: float,
    padding_horizontal_pixels: Optional[float],
    padding_vertical_pixels: Optional[float],
    outline_width: float,
) -> Dict[str, Any]:
    padding_x = padding_pixels if padding_horizontal_pixels is None else padding_horizontal_pixels
    padding_y = padding_pixels if padding_vertical_pixels is None else padding_vertical_pixels
    return {
        "font_dir": font_dir,
        "min_font_size": int(min_font_size),
        "max_font_size": int(max_font_size),
        "emphasis_font_dir": (
            config.outside_text.osb_emphasis_font_dir
            if is_outside_text
            else config.rendering.emphasis_font_dir
        ),
        "emphasis_max_font_size": (
            config.outside_text.osb_emphasis_max_font_size
            if is_outside_text
            else config.rendering.emphasis_max_font_size
        ),
        "line_spacing_mult": float(line_spacing),
        "emphasis_line_spacing_mult": float(line_spacing),
        "vertical_line_spacing_mult": vertical_line_spacing,
        "vertical_emphasis_line_spacing_mult": vertical_line_spacing,
        "vertical_font_size_mult": (
            1.0 if is_outside_text else float(config.rendering.vertical_font_size_mult)
        ),
        "vertical_max_chars": (
            int(config.outside_text.osb_vertical_max_chars)
            if is_outside_text
            else int(config.rendering.vertical_max_chars)
        ),
        "use_subpixel_rendering": (
            bool(config.outside_text.osb_use_subpixel_rendering)
            if is_outside_text
            else bool(config.rendering.use_subpixel_rendering)
        ),
        "font_hinting": (
            config.outside_text.osb_font_hinting
            if is_outside_text
            else config.rendering.font_hinting
        ),
        "use_ligatures": (
            bool(config.outside_text.osb_use_ligatures)
            if is_outside_text
            else bool(config.rendering.use_ligatures)
        ),
        "hyphenate_before_scaling": bool(config.rendering.hyphenate_before_scaling),
        "hyphen_penalty": float(config.rendering.hyphen_penalty),
        "hyphenation_min_word_length": int(config.rendering.hyphenation_min_word_length),
        "badness_exponent": float(config.rendering.badness_exponent),
        "padding_pixels": float(padding_pixels),
        "padding_horizontal_pixels": float(padding_x),
        "padding_vertical_pixels": float(padding_y),
        "outline_width": float(outline_width),
        "outline_color_rgb": None,
        "text_alignment": "center",
        "supersampling_factor": int(config.rendering.supersampling_factor),
        "detach_trailing_punctuation": bool(config.rendering.detach_trailing_punctuation),
        "auto_vertical_text": (
            bool(config.outside_text.osb_auto_vertical)
            if is_outside_text
            else bool(config.rendering.auto_vertical_text)
        ),
        "compare_forced_vertical_layout": False,
        "force_vertical_fallback_only": False,
        "tight_fit_shrink_enabled": bool(config.rendering.tight_fit_shrink_enabled),
        "tight_fit_shrink_min_font_size": int(
            config.rendering.tight_fit_shrink_min_font_size
        ),
        "tight_fit_shrink_width_ratio": float(config.rendering.tight_fit_shrink_width_ratio),
        "tight_fit_shrink_height_ratio": float(config.rendering.tight_fit_shrink_height_ratio),
        "tight_fit_shrink_factor": float(config.rendering.tight_fit_shrink_factor),
    }


def save_pipeline_edit_session(
    *,
    session_path: Path,
    source_image_path: Path,
    translated_output_path: Optional[Path],
    base_image: Image.Image,
    sorted_bubble_data: List[Dict[str, Any]],
    translated_texts: List[str],
    source_ocr_texts: List[str],
    processed_bubbles_info: List[Dict[str, Any]],
    config: MangaTranslatorConfig,
    main_min_font: int,
    main_max_font: int,
    osb_min_font: int,
    osb_max_font: int,
    padding_pixels: float,
    osb_padding_pixels: float,
    osb_outline_width: float,
    verbose: bool = False,
    padding_horizontal_pixels: Optional[float] = None,
    padding_vertical_pixels: Optional[float] = None,
) -> Optional[Path]:
    if len(translated_texts) != len(sorted_bubble_data):
        return None

    session_path = Path(session_path).resolve()
    assets_dir, base_image_path, edited_image_path = _asset_paths(session_path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    session_path.parent.mkdir(parents=True, exist_ok=True)

    base_to_save = base_image.convert("RGBA") if base_image.mode == "P" else base_image
    base_to_save.save(base_image_path)
    original_snapshot_path = assets_dir / "source_original.png"
    source_path = Path(source_image_path).resolve()
    if source_path.exists():
        try:
            with Image.open(source_path) as source_image:
                source_image.convert("RGBA").save(original_snapshot_path)
        except Exception as exc:
            log_message(
                f"Warning: failed to preserve editor source image: {exc}",
                always_print=True,
            )
    rendered_snapshot_path = assets_dir / "rendered_snapshot.png"

    render_info_by_bbox: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for info in processed_bubbles_info or []:
        try:
            key = tuple(int(round(float(v))) for v in info.get("bbox", ()))
        except Exception:
            continue
        if len(key) == 4:
            render_info_by_bbox[key] = info

    regions: List[Dict[str, Any]] = []
    provider = getattr(config.translation, "provider", "")
    inside_padding_x = (
        padding_pixels if padding_horizontal_pixels is None else padding_horizontal_pixels
    )
    inside_padding_y = (
        padding_pixels if padding_vertical_pixels is None else padding_vertical_pixels
    )
    for index, bubble in enumerate(sorted_bubble_data):
        if bubble.get("skip_render"):
            continue
        raw_text = translated_texts[index] if index < len(translated_texts) else ""
        render_tag, text = extract_translation_tag(raw_text)
        disabled = _is_invalid_translation(text, provider)
        is_outside_text = bool(bubble.get("is_outside_text", False))
        if is_outside_text and text:
            text = text.upper()

        source_bbox = _bbox_to_list(bubble.get("bbox", [0, 0, 1, 1]))
        editor_meta = (
            bubble.get("_editor_render_meta")
            if isinstance(bubble.get("_editor_render_meta"), dict)
            else {}
        )
        editor_render_config = (
            editor_meta.get("render_config")
            if isinstance(editor_meta.get("render_config"), dict)
            else {}
        )
        bbox = _bbox_to_list(editor_meta.get("bbox") or source_bbox)
        bbox_key = tuple(int(round(float(v))) for v in source_bbox)
        render_info = render_info_by_bbox.get(bbox_key, {})

        if is_outside_text:
            font_dir = (
                config.outside_text.osb_font_dir
                if config.outside_text.osb_font_dir
                else config.rendering.font_dir
            )
            min_font = osb_min_font
            max_font = osb_max_font
            line_spacing = float(config.outside_text.osb_line_spacing)
            vertical_line_spacing = float(
                getattr(config.outside_text, "osb_vertical_line_spacing", line_spacing)
            )
            bubble_color_bgr = (50, 50, 50) if bubble.get("is_dark_text", True) else (255, 255, 255)
            text_color_rgb = _normalize_rgb(bubble.get("text_color_rgb"))
            mask = None
            rotation_deg = float(bubble.get("rotation_deg", 0.0) or 0.0)
            vertical_stack = bool(bubble.get("vertical_stack", False))
            outline_width = osb_outline_width
            region_padding = osb_padding_pixels
            region_padding_x = osb_padding_pixels
            region_padding_y = osb_padding_pixels
        else:
            font_dir = config.rendering.font_dir
            min_font = main_min_font
            max_font = main_max_font
            line_spacing = float(config.rendering.line_spacing_mult)
            vertical_line_spacing = config.rendering.vertical_line_spacing_mult
            bubble_color_bgr = _bgr_tuple(render_info.get("color"), (255, 255, 255))
            text_color_bgr = render_info.get("text_color_bgr")
            text_color_rgb = _normalize_rgb(bubble.get("text_color_rgb"))
            if text_color_rgb is None:
                text_color_rgb = (
                    (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])
                    if isinstance(text_color_bgr, (list, tuple)) and len(text_color_bgr) >= 3
                    else None
                )
            mask = bubble.get("_editor_layout_mask")
            if mask is None:
                mask = render_info.get("mask")
            if mask is None:
                mask = bubble.get("sam_mask")
            rotation_deg = 0.0
            vertical_stack = False
            outline_width = config.rendering.outline_width
            region_padding = padding_pixels
            region_padding_x = inside_padding_x
            region_padding_y = inside_padding_y

        meta_font_dir = str(editor_meta.get("font_dir") or "").strip()
        if meta_font_dir:
            font_dir = meta_font_dir
        tag_font_dir = resolve_tag_font_path(render_tag, config.rendering.tag_styles)
        if tag_font_dir:
            font_dir = tag_font_dir
        if "rotation_deg" in editor_meta:
            try:
                rotation_deg = float(editor_meta.get("rotation_deg") or 0.0)
            except (TypeError, ValueError):
                pass
        if "vertical_stack" in editor_meta:
            vertical_stack = bool(editor_meta.get("vertical_stack"))

        region_id = f"r{len(regions) + 1:03d}"
        mask_path = _save_mask_asset(mask, assets_dir, region_id)
        region_config = _snapshot_region_config(
            config,
            is_outside_text=is_outside_text,
            font_dir=str(font_dir),
            min_font_size=min_font,
            max_font_size=max_font,
            line_spacing=line_spacing,
            vertical_line_spacing=vertical_line_spacing,
            padding_pixels=region_padding,
            padding_horizontal_pixels=region_padding_x,
            padding_vertical_pixels=region_padding_y,
            outline_width=outline_width,
        )
        for key, value in editor_render_config.items():
            if key in RenderingConfig.__dataclass_fields__ and key != "font_dir":
                region_config[key] = value
        editor_font_size = editor_meta.get("font_size")
        if editor_font_size is None:
            editor_font_size = editor_render_config.get("font_size")
        try:
            editor_font_size = int(round(float(editor_font_size)))
        except (TypeError, ValueError):
            editor_font_size = 0
        if editor_font_size > 0:
            _set_fixed_font_size_fields(region_config, editor_font_size)
        # The pipeline renderer already resolved the final orientation and font
        # size. Re-running auto-orientation while finalizing the editor session
        # is redundant, can be expensive, and emits false "text too large"
        # messages for rejected comparison candidates.
        region_config["auto_vertical_text"] = False
        region_config["compare_forced_vertical_layout"] = False
        region_config["force_compare_vertical_text"] = False
        region_config["font_dir"] = str(font_dir)
        regions.append(
            {
                "id": region_id,
                "label": f"{len(regions) + 1}. {'OSB' if is_outside_text else 'Bubble'}",
                "enabled": not disabled,
                "type": "outside_text" if is_outside_text else "speech_bubble",
                "source_text": source_ocr_texts[index] if index < len(source_ocr_texts) else "",
                "translation": text,
                "render_tag": render_tag or "",
                "font_manual": False,
                "bbox": bbox,
                "original_bbox": _jsonable(bubble.get("original_bbox") or bubble.get("bbox")),
                "rotation_deg": rotation_deg,
                "vertical_stack": vertical_stack,
                "resolved_orientation": "vertical" if vertical_stack else "horizontal",
                "font_color": _rgb_to_hex(text_color_rgb, "#000000"),
                "text_color_rgb": _jsonable(text_color_rgb),
                "bubble_color_bgr": _jsonable(bubble_color_bgr),
                "mask_path": mask_path,
                "use_mask": bool(mask_path and not is_outside_text),
                "render": region_config,
            }
        )

    if not regions:
        return None

    data = {
        "version": SESSION_VERSION,
        "editor_character_profile_name": config.editor_character_profile_name,
        "editor_source_folder_name": config.editor_source_folder_name,
        "source_image_path": str(source_path),
        "overwrite_target_path": (
            str(Path(translated_output_path).resolve()) if translated_output_path else str(source_path)
        ),
        "original_snapshot_path": (
            str(original_snapshot_path.resolve()) if original_snapshot_path.exists() else ""
        ),
        "translated_image_path": (
            str(Path(translated_output_path).resolve()) if translated_output_path else ""
        ),
        "rendered_snapshot_path": str(rendered_snapshot_path.resolve()),
        "base_image_path": str(base_image_path.resolve()),
        "inpainted_image_path": "",
        "inpaint_source_path": "",
        "inpainted_background_text_free": False,
        "inpaint_mask_path": "",
        "edited_image_path": str(edited_image_path.resolve()),
        "assets_dir": str(assets_dir.resolve()),
        "paint_overlay_path": "",
        "restore_mask_path": "",
        "image_size": list(base_image.size),
        "output": {
            "jpeg_quality": int(config.output.jpeg_quality),
            "png_compression": int(config.output.png_compression),
            "output_format": config.output.output_format,
            "upscale_final_image": bool(config.output.upscale_final_image),
            "image_upscale_factor": float(config.output.image_upscale_factor),
            "image_upscale_model": config.output.image_upscale_model,
        },
        "regions": regions,
    }

    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, ensure_ascii=False)

    log_message(f"Saved editor session: {session_path}", verbose=verbose)
    return session_path


def finalize_pipeline_edit_session(
    session_path: Any,
    rendered_image: Image.Image,
    *,
    verbose: bool = False,
) -> Optional[Path]:
    path = Path(str(session_path or "")).resolve()
    if not path.exists():
        return None
    session = load_session(path)
    assets_dir, _base_image_path, _edited_image_path = _asset_paths(path)
    assets_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = assets_dir / "rendered_snapshot.png"
    rendered_image.convert("RGBA").save(snapshot_path)
    session["rendered_snapshot_path"] = str(snapshot_path.resolve())
    session["assets_dir"] = str(assets_dir.resolve())
    save_session(path, session)
    log_message(f"Saved editor rendered snapshot: {snapshot_path}", verbose=verbose)
    return snapshot_path


def create_image_edit_sessions(
    image_paths: List[Any],
    *,
    font_dir: Any,
    source_references: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Tuple[Path, Path]]:
    """Create standalone editor sessions without running the translation pipeline."""
    paths = [Path(str(value)).resolve() for value in image_paths or [] if value]
    if not paths:
        raise ValueError("편집할 이미지를 한 장 이상 선택해 주세요.")

    # 글꼴 선택은 폰트팩 폴더이거나 그 안의 개별 글꼴 파일 하나다.
    selected_font_dir = Path(str(font_dir or "")).resolve()
    if not selected_font_dir.is_dir() and not selected_font_dir.is_file():
        raise FileNotFoundError(f"선택한 글꼴을 찾을 수 없습니다: {selected_font_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    run_name = f"imported_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    session_dir = repo_root / "editor_imports" / run_name
    session_dir.mkdir(parents=True, exist_ok=True)
    source_references = source_references or {}

    created: List[Tuple[Path, Path]] = []
    for index, source_path in enumerate(paths, start=1):
        if not source_path.is_file():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {source_path}")

        try:
            from core.editor_package import extract_embedded_editor_package

            embedded_session_path = extract_embedded_editor_package(
                source_path,
                destination_root=session_dir / "embedded",
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to load embedded editor data from image: {source_path.name} ({exc})"
            ) from exc
        if embedded_session_path is not None:
            created.append((source_path, embedded_session_path.resolve()))
            continue

        source_reference = source_references.get(
            str(source_path.resolve()).casefold(), {}
        )
        reference_kind = str(source_reference.get("kind") or "").strip().lower()
        linked_file_path = str(source_reference.get("path") or "").strip()
        linked_display_path = str(
            source_reference.get("display_path") or ""
        ).replace("\\", "/").strip("/")
        linked_archive_path = str(
            source_reference.get("archive_path") or ""
        ).strip()
        linked_archive_member = str(
            source_reference.get("member_path") or ""
        ).replace("\\", "/").strip("/")

        decoded_relative_path = _decode_uploaded_relative_path(source_path.name)
        source_display_path = (
            linked_display_path
            if linked_display_path
            else linked_archive_member
            if reference_kind == "zip" and linked_archive_member
            else
            str(decoded_relative_path).replace("\\", "/")
            if decoded_relative_path is not None
            else source_path.name
        )
        if reference_kind == "file" and linked_file_path:
            overwrite_target_path = str(Path(linked_file_path).resolve())
            source_overwrite_available = Path(overwrite_target_path).is_file()
        elif (
            reference_kind == "zip"
            and linked_archive_path
            and linked_archive_member
        ):
            overwrite_target_path = ""
            source_overwrite_available = Path(linked_archive_path).is_file()
        else:
            overwrite_target_path = str(source_path)
            source_overwrite_available = not _is_temporary_upload_path(source_path)
        display_stem = Path(source_display_path).stem or source_path.stem
        safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", display_stem).strip(" ._")
        safe_stem = safe_stem or f"image_{index:03d}"
        session_path = session_dir / f"{index:03d}_{safe_stem}_edit_session.json"
        assets_dir, base_image_path, edited_image_path = _asset_paths(session_path)
        assets_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(source_path) as loaded:
                image = ImageOps.exif_transpose(loaded).convert("RGBA")
        except Exception as exc:
            raise ValueError(f"이미지를 열 수 없습니다: {source_path.name} ({exc})") from exc

        original_snapshot_path = assets_dir / "source_original.png"
        image.save(original_snapshot_path)
        image.save(base_image_path)

        default_render = {
            "font_dir": str(selected_font_dir),
            "font_size": 24,
            "min_font_size": 24,
            "max_font_size": 24,
            "emphasis_max_font_size": 24,
            "line_spacing_mult": 1.0,
            "padding_pixels": 5.0,
            "padding_horizontal_pixels": 5.0,
            "padding_vertical_pixels": 5.0,
            "outline_width": 0.0,
            "outline_color_rgb": None,
            "text_alignment": "center",
            "supersampling_factor": 4,
            "auto_vertical_text": False,
        }
        data = {
            "version": SESSION_VERSION,
            "standalone_image_session": True,
            "source_image_path": str(source_path),
            "overwrite_target_path": overwrite_target_path,
            "source_overwrite_available": source_overwrite_available,
            "source_display_path": source_display_path,
            "source_link_kind": reference_kind,
            "source_archive_path": (
                str(Path(linked_archive_path).resolve())
                if reference_kind == "zip" and linked_archive_path
                else ""
            ),
            "source_archive_member": (
                linked_archive_member if reference_kind == "zip" else ""
            ),
            "original_snapshot_path": str(original_snapshot_path.resolve()),
            "translated_image_path": "",
            "rendered_snapshot_path": "",
            "base_image_path": str(base_image_path.resolve()),
            "inpainted_image_path": "",
            "inpaint_source_path": "",
            "inpainted_background_text_free": False,
            "inpaint_mask_path": "",
            "edited_image_path": str(edited_image_path.resolve()),
            "assets_dir": str(assets_dir.resolve()),
            "paint_overlay_path": "",
            "restore_mask_path": "",
            "image_size": list(image.size),
            "default_render": default_render,
            "output": {
                "jpeg_quality": 95,
                "png_compression": 2,
                "output_format": "png",
                "upscale_final_image": False,
                "image_upscale_factor": 2.0,
                "image_upscale_model": "model_lite",
            },
            "regions": [],
        }
        save_session(session_path, data)
        created.append((source_path, session_path.resolve()))

    link_edit_session_group([session_path for _source_path, session_path in created])
    return created


def load_session(session_path: Any) -> Dict[str, Any]:
    path = Path(str(session_path or "")).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Editor session not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        session = json.load(f)
    session, _warnings = repair_session_assets(path, session)
    changed = _ensure_region_baselines(session)
    if _repair_auto_vertical_regions(session):
        changed = True
    if _ensure_region_baselines(session):
        changed = True
    if changed:
        save_session(path, session)
    return session


def save_session(session_path: Any, session: Dict[str, Any]) -> None:
    path = Path(str(session_path or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(session), f, indent=2, ensure_ascii=False)


def link_edit_session_group(session_paths: List[Any]) -> List[Path]:
    """Persist an ordered set of editor sessions so one window can navigate them."""
    resolved: List[Path] = []
    seen = set()
    for value in session_paths or []:
        path = Path(str(value or "")).resolve()
        key = str(path).casefold()
        if path.is_file() and key not in seen:
            seen.add(key)
            resolved.append(path)
    if not resolved:
        return []

    group = [str(path) for path in resolved]
    for index, path in enumerate(resolved):
        try:
            with open(path, "r", encoding="utf-8") as f:
                session = json.load(f)
            session["editor_session_group"] = group
            session["editor_session_group_index"] = index
            save_session(path, session)
        except Exception as exc:
            log_message(
                f"Warning: failed to link editor session group for {path}: {exc}",
                always_print=True,
            )
    return resolved


def related_edit_session_paths(
    session_path: Any,
    session: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """Return linked batch/import sessions, with a directory fallback for older runs."""
    current = Path(str(session_path or "")).resolve()
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


def region_choices(session: Dict[str, Any]) -> List[Tuple[str, str]]:
    choices = []
    for index, region in enumerate(session.get("regions", []), start=1):
        text = str(region.get("translation") or "").replace("\n", " ").strip()
        if len(text) > 32:
            text = f"{text[:32]}..."
        label = f"{index}. {region.get('type', 'region')} | {text or '(empty)'}"
        choices.append((label, region.get("id", f"r{index:03d}")))
    return choices


def find_region(session: Dict[str, Any], region_id: Any) -> Optional[Dict[str, Any]]:
    for region in session.get("regions", []):
        if region.get("id") == region_id:
            return region
    return None


def preview_image(session: Dict[str, Any], selected_region_id: Optional[str] = None) -> Image.Image:
    image_path = (
        session.get("edited_image_path")
        if session.get("edited_image_path") and Path(session.get("edited_image_path")).exists()
        else session.get("translated_image_path")
    )
    if not image_path or not Path(image_path).exists():
        image_path = session.get("base_image_path")
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for index, region in enumerate(session.get("regions", []), start=1):
        bbox = _safe_int_bbox(region.get("bbox"), (width, height))
        selected = region.get("id") == selected_region_id
        color = (33, 150, 243, 255) if selected else (255, 193, 7, 220)
        if not region.get("enabled", True):
            color = (158, 158, 158, 180)
        draw.rectangle(bbox, outline=color, width=4 if selected else 2)
        label = str(index)
        text_bbox = draw.textbbox((bbox[0], bbox[1]), label, font=font)
        pad = 3
        draw.rectangle(
            (
                text_bbox[0] - pad,
                text_bbox[1] - pad,
                text_bbox[2] + pad,
                text_bbox[3] + pad,
            ),
            fill=(0, 0, 0, 180),
        )
        draw.text((bbox[0], bbox[1]), label, fill=(255, 255, 255, 255), font=font)
    return image


def get_region_form_values(
    session_path: Any, region_id: Any
) -> Tuple[Any, ...]:
    session = load_session(session_path)
    region = find_region(session, region_id)
    if region is None:
        return ("", "", 0, 0, 1, 1, 16, 8, 0.0, False, "#000000", True, None)
    x1, y1, x2, y2 = _bbox_to_list(region.get("bbox"))
    render = region.get("render", {})
    return (
        region.get("source_text", ""),
        region.get("translation", ""),
        int(round(x1)),
        int(round(y1)),
        int(round(x2 - x1)),
        int(round(y2 - y1)),
        int(render.get("max_font_size") or 16),
        int(render.get("min_font_size") or 8),
        float(region.get("rotation_deg") or 0.0),
        bool(region.get("vertical_stack", False)),
        region.get("font_color") or "#000000",
        bool(region.get("enabled", True)),
        preview_image(session, str(region_id)),
    )


def update_region_from_form(
    session_path: Any,
    region_id: Any,
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
) -> Tuple[Dict[str, Any], Image.Image]:
    session = load_session(session_path)
    region = find_region(session, region_id)
    if region is None:
        raise ValueError("No editor region is selected.")
    left = float(x or 0)
    top = float(y or 0)
    box_width = max(1.0, float(width or 1))
    box_height = max(1.0, float(height or 1))
    region["translation"] = str(translation or "")
    region["bbox"] = [left, top, left + box_width, top + box_height]
    region["rotation_deg"] = float(rotation_deg or 0.0)
    region["vertical_stack"] = bool(vertical_stack)
    region["font_color"] = font_color or "#000000"
    region["text_color_rgb"] = list(_normalize_rgb(region["font_color"]) or (0, 0, 0))
    region["enabled"] = bool(enabled)
    render = region.setdefault("render", {})
    render["max_font_size"] = int(max(1, round(float(max_font_size or 16))))
    render["min_font_size"] = int(max(1, round(float(min_font_size or 8))))
    save_session(session_path, session)
    return session, preview_image(session, str(region_id))


def add_region(session_path: Any) -> Tuple[Dict[str, Any], str, Image.Image]:
    session = load_session(session_path)
    width, height = session.get("image_size", [800, 1200])
    box_w = max(80, int(width * 0.22))
    box_h = max(40, int(height * 0.08))
    left = max(0, int((width - box_w) / 2))
    top = max(0, int((height - box_h) / 2))
    new_index = len(session.get("regions", [])) + 1
    if session.get("regions"):
        template = deepcopy(session["regions"][-1])
    else:
        template = {"render": deepcopy(session.get("default_render", {}))}
    template.pop("render_baseline", None)
    for key in list(template):
        if str(key).startswith(("_editor_", "ai_")):
            template.pop(key, None)
    existing_ids = {str(region.get("id") or "") for region in session.get("regions", [])}
    next_number = new_index
    while f"r{next_number:03d}" in existing_ids:
        next_number += 1
    region_id = f"r{next_number:03d}"
    template.update(
        {
            "id": region_id,
            "label": f"{new_index}. 추가한 텍스트",
            "enabled": True,
            "type": "added_text",
            "source_text": "",
            "translation": "새 텍스트",
            "bbox": [left, top, left + box_w, top + box_h],
            "original_bbox": [left, top, left + box_w, top + box_h],
            "rotation_deg": 0.0,
            "vertical_stack": False,
            "font_color": "#000000",
            "text_color_rgb": [0, 0, 0],
            "bubble_color_bgr": [255, 255, 255],
            "mask_path": None,
            "use_mask": False,
        }
    )
    render = template.setdefault("render", {})
    render.setdefault("font_dir", "")
    default_font_size = int(render.get("font_size") or render.get("max_font_size") or 24)
    _set_fixed_font_size_fields(render, default_font_size)
    session.setdefault("regions", []).append(template)
    save_session(session_path, session)
    return session, region_id, preview_image(session, region_id)


def delete_region(session_path: Any, region_id: Any) -> Tuple[Dict[str, Any], Optional[str], Image.Image]:
    session = load_session(session_path)
    regions = session.get("regions", [])
    session["regions"] = [region for region in regions if region.get("id") != region_id]
    next_id = session["regions"][0]["id"] if session["regions"] else None
    save_session(session_path, session)
    return session, next_id, preview_image(session, next_id)


def _render_config_from_region(region: Dict[str, Any]) -> RenderingConfig:
    data = dict(region.get("render", {}))
    _resolve_render_font_paths(data)
    fixed_font_size = data.pop("font_size", None)
    try:
        fixed_font_size = int(round(float(fixed_font_size)))
    except (TypeError, ValueError):
        fixed_font_size = 0
    if fixed_font_size > 0:
        data["min_font_size"] = fixed_font_size
        data["max_font_size"] = fixed_font_size
        data["emphasis_max_font_size"] = fixed_font_size
        data["vertical_font_size_mult"] = 1.0
        data["tight_fit_shrink_enabled"] = False
    if bool(region.get("vertical_stack", False)):
        data["auto_vertical_text"] = False
        data["compare_forced_vertical_layout"] = False
    allowed = RenderingConfig.__dataclass_fields__.keys()
    filtered = {key: data[key] for key in allowed if key in data}
    return RenderingConfig(**filtered)


def _set_fixed_font_size_fields(render: Dict[str, Any], font_size: int) -> int:
    size = max(1, int(round(float(font_size))))
    render["font_size"] = size
    render["min_font_size"] = size
    render["max_font_size"] = size
    render["emphasis_max_font_size"] = size
    render["vertical_font_size_mult"] = 1.0
    render["tight_fit_shrink_enabled"] = False
    return size


def set_region_font_size(
    region: Dict[str, Any],
    font_size: int,
    *,
    update_baseline: bool = False,
) -> int:
    render = region.setdefault("render", {})
    size = _set_fixed_font_size_fields(render, font_size)
    if update_baseline:
        baseline = region.get("render_baseline")
        if isinstance(baseline, dict):
            baseline_render = baseline.get("render")
            if isinstance(baseline_render, dict):
                _set_fixed_font_size_fields(baseline_render, size)
    return size


def resolve_region_font_size(
    session: Dict[str, Any],
    region: Dict[str, Any],
    index: int,
) -> int:
    render = region.setdefault("render", {})
    try:
        fixed_font_size = int(round(float(render.get("font_size") or 0)))
    except (TypeError, ValueError):
        fixed_font_size = 0
    if fixed_font_size > 0:
        return set_region_font_size(region, fixed_font_size)

    try:
        min_font_size = int(round(float(render.get("min_font_size") or 0)))
        max_font_size = int(round(float(render.get("max_font_size") or 0)))
    except (TypeError, ValueError):
        min_font_size = 0
        max_font_size = 0
    if min_font_size > 0 and min_font_size == max_font_size:
        return set_region_font_size(
            region,
            min_font_size,
            update_baseline=True,
        )

    text = str(region.get("translation") or "")
    font_dir = _ensure_render_font_path(render, persist=False)
    if text.strip() and font_dir:
        try:
            base_path = _editor_base_image_path(session)
            with Image.open(base_path) as base_image:
                image = base_image.convert("RGBA")
            mask = (
                _load_mask(region.get("mask_path"))
                if bool(region.get("use_mask")) and not _region_geometry_changed(region)
                else None
            )
            result = render_text_skia(
                pil_image=image,
                text=text,
                bbox=_safe_int_bbox(region.get("bbox"), image.size),
                font_dir=font_dir,
                cleaned_mask=mask,
                bubble_color_bgr=_bgr_tuple(
                    region.get("bubble_color_bgr"),
                    (255, 255, 255),
                ),
                config=_render_config_from_region(region),
                verbose=False,
                bubble_id=str(index),
                rotation_deg=float(region.get("rotation_deg") or 0.0),
                vertical_stack=bool(region.get("vertical_stack", False)),
                text_color_rgb=_normalize_rgb(region.get("font_color")),
                raise_on_safe_error=False,
                layout_only=True,
                auto_vertical_ignore_word_count=True,
            )
            resolved = int(round(float(result.info.get("font_size") or 0)))
            if resolved > 0:
                return set_region_font_size(
                    region,
                    resolved,
                    update_baseline=True,
                )
        except Exception:
            pass

    fallback = max_font_size if max_font_size > 0 else 24
    return set_region_font_size(
        region,
        fallback,
        update_baseline=True,
    )


def _editor_base_image_path(session: Dict[str, Any]) -> Path:
    for key in ("inpainted_image_path", "inpaint_source_path", "base_image_path"):
        value = session.get(key)
        if value and Path(value).is_file():
            return Path(value)
    raise FileNotFoundError(f"Editor base image not found: {session.get('base_image_path')}")


def _apply_paint_overlay(image: Image.Image, session: Dict[str, Any]) -> Image.Image:
    paint_overlay_path = session.get("paint_overlay_path")
    if not paint_overlay_path or not Path(paint_overlay_path).is_file():
        return image
    try:
        overlay = Image.open(paint_overlay_path).convert("RGBA")
        if overlay.size != image.size:
            overlay = overlay.resize(image.size, Image.Resampling.LANCZOS)
        image = image.copy()
        image.alpha_composite(overlay)
    except Exception as exc:
        log_message(
            f"Warning: failed to apply editor paint overlay: {exc}",
            always_print=True,
        )
    return image


def _apply_original_restore(
    image: Image.Image, session: Dict[str, Any], *, mask_key: str = "restore_mask_path"
) -> Image.Image:
    mask_path = Path(session.get(mask_key) or "")
    if not mask_path.is_file():
        return image
    source_path = Path(
        session.get("original_snapshot_path")
        or session.get("source_image_path")
        or ""
    )
    if not source_path.is_file():
        return image
    try:
        original = Image.open(source_path).convert("RGBA")
        if original.size != image.size:
            original = original.resize(image.size, Image.Resampling.LANCZOS)
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        return Image.composite(original, image.convert("RGBA"), mask)
    except Exception as exc:
        log_message(
            f"Warning: failed to apply original restore mask: {exc}",
            always_print=True,
        )
        return image


def _editor_overflow_bboxes(
    region: Dict[str, Any],
    image_size: Tuple[int, int],
) -> List[Tuple[int, int, int, int]]:
    render = region.get("render", {})
    if not isinstance(render, dict):
        return []
    try:
        font_size = float(
            render.get("font_size")
            or render.get("max_font_size")
            or 0
        )
    except (TypeError, ValueError):
        return []
    if font_size <= 0:
        return []

    x1, y1, x2, y2 = _bbox_to_list(region.get("bbox"))
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    text = str(region.get("translation") or "")
    visible_chars = max(1, len(re.sub(r"\s+", "", text)))
    text_span = font_size * max(1.0, math.sqrt(float(visible_chars)))
    vertical = bool(region.get("vertical_stack", False))

    dimensions: List[Tuple[float, float]] = []
    for expansion in (0.75, 1.5, 3.0):
        dimensions.append(
            (
                max(box_width * (1.0 + expansion * 0.45), box_width + font_size * expansion * 2.0),
                max(box_height * (1.0 + expansion * 0.45), box_height + font_size * expansion * 2.0),
            )
        )
    if vertical:
        dimensions.append(
            (
                max(box_width * 2.5, text_span),
                max(box_height * 2.5, text_span * 1.8),
            )
        )
    else:
        dimensions.append(
            (
                max(box_width * 2.5, text_span * 1.8),
                max(box_height * 2.5, text_span),
            )
        )

    candidates: List[Tuple[int, int, int, int]] = []
    original = _safe_int_bbox([x1, y1, x2, y2], image_size)
    for width, height in dimensions:
        candidate = tuple(
            int(round(value))
            for value in (
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            )
        )
        if candidate != original and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _render_region_onto(
    image: Image.Image,
    region: Dict[str, Any],
    index: int,
    errors: List[str],
) -> Image.Image:
    if region.get("ai_image") or not region.get("enabled", True):
        return image
    text = str(region.get("translation") or "")
    if not text.strip():
        return image
    render = region.get("render", {})
    if isinstance(render, dict):
        font_dir = _ensure_render_font_path(render, persist=False)
    else:
        font_dir = ""
    if not font_dir:
        errors.append(f"Region {index}: missing font path")
        return image
    config = _render_config_from_region(region)
    bbox = _safe_int_bbox(region.get("bbox"), image.size)
    bubble_color_bgr = _bgr_tuple(
        region.get("bubble_color_bgr"),
        (255, 255, 255),
    )
    rotation_deg = float(region.get("rotation_deg") or 0.0)
    vertical_stack = bool(region.get("vertical_stack", False))
    text_color_rgb = _normalize_rgb(region.get("font_color"))

    scale_x, scale_y, skew_deg, perspective_y = _region_text_transform(region, apply_font_compensation=True)
    use_text_warp = (
        abs(scale_x - 1.0) > 1e-4
        or abs(scale_y - 1.0) > 1e-4
        or abs(skew_deg) > 1e-4
        or abs(perspective_y) > 1e-4
    )

    def render_at(
        target_bbox: Tuple[int, int, int, int],
        target_mask: Optional[np.ndarray],
    ) -> Image.Image:
        target_image = Image.new("RGBA", image.size, (0, 0, 0, 0)) if use_text_warp else image
        result = render_text_skia(
            pil_image=target_image,
            text=text,
            bbox=target_bbox,
            font_dir=font_dir,
            cleaned_mask=target_mask,
            bubble_color_bgr=bubble_color_bgr,
            config=config,
            verbose=False,
            bubble_id=str(index),
            rotation_deg=0.0 if use_text_warp else rotation_deg,
            vertical_stack=vertical_stack,
            text_color_rgb=text_color_rgb,
            raise_on_safe_error=False,
            auto_vertical_ignore_word_count=True,
        )
        if not use_text_warp:
            return result
        warped = _warp_rgba_text_layer(
            result,
            _region_text_transform_matrix(region, target_bbox),
        )
        composited = Image.alpha_composite(image.convert("RGBA"), warped)
        composited.info.update(result.info)
        return composited

    try:
        use_mask = bool(region.get("use_mask")) and not _region_geometry_changed(region)
        mask = _load_mask(region.get("mask_path")) if use_mask else None
        return render_at(bbox, mask)
    except Exception as initial_exc:
        for overflow_bbox in _editor_overflow_bboxes(region, image.size):
            try:
                result = render_at(overflow_bbox, None)
                result.info["editor_overflow_bbox"] = overflow_bbox
                return result
            except Exception:
                continue
        errors.append(f"Region {index}: {initial_exc}")
        return image


def _apply_region_patch(
    image: Image.Image,
    clean_base: Image.Image,
    base_identity: str,
    region: Dict[str, Any],
    index: int,
    errors: List[str],
) -> Image.Image:
    if region.get("ai_image"):
        return image
    patch, patch_errors = _cached_region_patch(
        clean_base,
        base_identity,
        region,
        index,
    )
    errors.extend(patch_errors)
    if patch is None:
        return image
    box, patch_mask, pixels = patch
    result = image.convert("RGBA")
    result.paste(
        Image.fromarray(pixels), box,
        Image.fromarray(patch_mask.astype(np.uint8) * 255),
    )
    return result


def _boxes_intersect(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _expand_int_box(
    box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    padding: int = 8,
) -> Tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = box
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _render_session_text_image(
    session: Dict[str, Any], *, include_text: bool = True,
) -> Tuple[Image.Image, List[str]]:
    """Render an editor session in memory using the exact export renderer."""
    _ensure_region_baselines(session)
    base_path = _editor_base_image_path(session)
    deleted_boxes = list(session.get("deleted_region_bboxes", []) or [])
    inpainted_path = Path(session.get("inpainted_image_path") or "")
    text_free_inpainted_background = bool(
        session.get("inpainted_background_text_free")
        and inpainted_path.is_file()
    )
    background_path = inpainted_path
    if not background_path.is_file():
        background_path = Path(session.get("rendered_snapshot_path") or "")
    if not background_path.is_file():
        background_path = Path(session.get("translated_image_path") or "")
    dirty_region_ids = {
        str(region_id)
        for region_id in session.get("editor_dirty_region_ids", []) or []
        if str(region_id)
    }
    changed_region_indexes = [
        index
        for index, region in enumerate(session.get("regions", []), start=1)
        if (not include_text and not region.get("ai_image"))
        or str(region.get("id") or "") in dirty_region_ids or not _region_is_baseline(region)
    ]
    if background_path.is_file():
        image = Image.open(background_path).convert("RGBA")
        image = _apply_paint_overlay(image, session)
        if text_free_inpainted_background:
            errors: List[str] = []
            if include_text:
                for index, region in enumerate(session.get("regions", []), start=1):
                    image = _render_region_onto(image, region, index, errors)
            return _apply_original_restore(image, session), errors
        if not changed_region_indexes and not deleted_boxes:
            return _apply_original_restore(image, session), []
        clean_base = Image.open(base_path).convert("RGBA")
        if clean_base.size != image.size:
            clean_base = clean_base.resize(image.size, Image.Resampling.LANCZOS)
        base_identity = _image_cache_identity(base_path, image.size)
        clean_base = _apply_paint_overlay(clean_base, session)
        overlay_path = Path(session.get("paint_overlay_path") or "")
        if overlay_path.is_file():
            base_identity += "|" + _image_cache_identity(overlay_path, image.size)
        erase_boxes: List[Tuple[int, int, int, int]] = []
        for box in deleted_boxes:
            try:
                erase_box = _safe_int_bbox(box, image.size)
            except Exception:
                continue
            expanded_erase_box = _expand_int_box(erase_box, image.size)
            erase_boxes.append(expanded_erase_box)
        for index in changed_region_indexes:
            region = session.get("regions", [])[index - 1]
            baseline_region = _baseline_region(region)
            if baseline_region is not None:
                try:
                    erase_boxes.append(
                        _expanded_union_bbox(region, image.size)
                    )
                except Exception:
                    pass
        affected_indexes = set(changed_region_indexes)
        # Erasing part of a neighbor requires clearing its entire old text before
        # redrawing. Include subsequent neighbors before applying any text patches.
        # ponytail: repeated scans; use spatial indexing only for very dense pages.
        while include_text:
            added = False
            for index, region in enumerate(session.get("regions", []), start=1):
                if index in affected_indexes or not region.get("enabled", True):
                    continue
                region_box = _safe_int_bbox(region.get("bbox"), image.size)
                if any(_boxes_intersect(region_box, box) for box in erase_boxes):
                    affected_indexes.add(index)
                    erase_boxes.append(_expanded_union_bbox(region, image.size))
                    added = True
            if not added:
                break
        for erase_box in erase_boxes:
            image.paste(clean_base.crop(erase_box), erase_box)
        errors: List[str] = []
        for index, region in enumerate(session.get("regions", []), start=1):
            if not include_text or index not in affected_indexes:
                continue
            image = _apply_region_patch(
                image,
                clean_base,
                base_identity,
                region,
                index,
                errors,
            )
        return _apply_original_restore(image, session), errors

    image = Image.open(base_path).convert("RGBA")
    image = _apply_paint_overlay(image, session)
    errors: List[str] = []

    if include_text:
        for index, region in enumerate(session.get("regions", []), start=1):
            image = _render_region_onto(image, region, index, errors)

    return _apply_original_restore(image, session), errors


def render_session_background(
    session: Dict[str, Any], *, include_ai: bool = True, errors: Optional[List[str]] = None,
) -> Image.Image:
    """Compose current artwork and brush layers without editable text."""
    from core.ai_image_region import apply_ai_image_regions

    image, background_errors = _render_session_text_image(session, include_text=False)
    if errors is not None:
        errors.extend(background_errors)
    if include_ai:
        image = apply_ai_image_regions(image, session, errors)
    elif any(region.get("ai_image") for region in session.get("regions", [])):
        # A saved flattened page can contain AI pixels; keep the baked artwork independent.
        clean_base = Image.open(_editor_base_image_path(session)).convert("RGBA")
        if clean_base.size != image.size:
            clean_base = clean_base.resize(image.size, Image.Resampling.LANCZOS)
        clean_base = _apply_paint_overlay(clean_base, session)
        for region in session.get("regions", []):
            if not region.get("ai_image"):
                continue
            box = tuple(int(v) for v in region["ai_image"]["box"])
            with Image.open(region["ai_mask_path"]) as opened:
                mask = opened.convert("L").point(lambda value: 255 if value else 0)
            image.paste(clean_base.crop(box), box, mask)
    return _apply_original_restore(image, session)


def render_session_image(session: Dict[str, Any]) -> Tuple[Image.Image, List[str]]:
    layered = any(region.get("ai_image") for region in session.get("regions", [])) or Path(
        session.get("restore_mask_path") or ""
    ).is_file()
    layered = layered or Path(session.get("text_cover_mask_path") or "").is_file()
    if not layered:
        return _render_session_text_image(session)
    errors: List[str] = []
    background = render_session_background(session, errors=errors)
    image = background.copy()
    # Cache against actual lower-layer pixels, including AI toggles and brush changes.
    base_identity = "layered|" + hashlib.sha256(background.tobytes()).hexdigest()
    for index, region in enumerate(session.get("regions", []), start=1):
        image = _apply_region_patch(image, background, base_identity, region, index, errors)
    # Intentional occlusion is a separate top layer; ordinary restore stays below text.
    return _apply_original_restore(image, session, mask_key="text_cover_mask_path"), errors


def rerender_session(session_path: Any) -> Tuple[Image.Image, Path, List[str]]:
    session = load_session(session_path)
    image, errors = render_session_image(session)
    edited_path = Path(session.get("edited_image_path") or "")
    if not edited_path:
        edited_path = Path(session_path).with_name(f"{Path(session_path).stem}_edited.png")
        session["edited_image_path"] = str(edited_path)
    edited_path.parent.mkdir(parents=True, exist_ok=True)
    output_cfg = session.get("output", {})
    if output_cfg.get("upscale_final_image"):
        image = upscale_image(
            image,
            float(output_cfg.get("image_upscale_factor", 2.0)),
            model_type=output_cfg.get("image_upscale_model", "model_lite"),
            verbose=False,
        )
    if edited_path.suffix.lower() in {".jpg", ".jpeg"}:
        image_to_save = image.convert("RGB")
    else:
        image_to_save = image
    save_image_with_compression(
        image_to_save,
        edited_path,
        jpeg_quality=int(output_cfg.get("jpeg_quality", 95)),
        png_compression=int(output_cfg.get("png_compression", 2)),
        verbose=False,
    )
    save_session(session_path, session)
    return image_to_save.copy(), edited_path, errors
