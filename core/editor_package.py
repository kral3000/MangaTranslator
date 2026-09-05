from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from uuid import uuid4

from utils.logging import log_message


PACKAGE_VERSION = 1
TRAILER_MAGIC = b"\nMANGATRANSLATOR_EDITOR_PACKAGE_V1\n"
TRAILER_LENGTH_SIZE = 8
TRAILER_DIGEST_SIZE = 32
TRAILER_FOOTER_SIZE = (
    TRAILER_LENGTH_SIZE + TRAILER_DIGEST_SIZE + len(TRAILER_MAGIC)
)

SUPPORTED_EMBED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SESSION_FILE_KEYS = (
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
    "paint_overlay_path",
    "restore_mask_path",
)
REGION_FILE_KEYS = ("mask_path",)
OUTPUT_REFERENCE_KEYS = (
    "translated_image_path",
    "overwrite_target_path",
    "edited_image_path",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clean_name(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text).strip(" ._")
    return text or fallback


def _safe_archive_name(path: Path, index: int) -> str:
    name = _clean_name(path.name, f"asset_{index:04d}")
    return f"assets/{index:04d}_{name}"


def _split_trailer(data: bytes) -> Tuple[bytes, Optional[bytes]]:
    if len(data) < TRAILER_FOOTER_SIZE or not data.endswith(TRAILER_MAGIC):
        return data, None

    magic_start = len(data) - len(TRAILER_MAGIC)
    digest_start = magic_start - TRAILER_DIGEST_SIZE
    length_start = digest_start - TRAILER_LENGTH_SIZE
    if length_start < 0:
        return data, None

    payload_length = struct.unpack("<Q", data[length_start:digest_start])[0]
    payload_start = length_start - payload_length
    if payload_start < 0:
        return data, None

    payload = data[payload_start:length_start]
    expected_digest = data[digest_start:magic_start]
    if hashlib.sha256(payload).digest() != expected_digest:
        return data, None
    return data[:payload_start], payload


def strip_embedded_editor_package(image_path: Any) -> bool:
    path = Path(str(image_path or "")).resolve()
    data = path.read_bytes()
    stripped, payload = _split_trailer(data)
    if payload is None:
        return False
    _atomic_write(path, stripped)
    return True


def read_embedded_editor_package(image_path: Any) -> Optional[bytes]:
    path = Path(str(image_path or "")).resolve()
    if not path.is_file():
        return None
    _stripped, payload = _split_trailer(path.read_bytes())
    return payload


def has_embedded_editor_package(image_path: Any) -> bool:
    return read_embedded_editor_package(image_path) is not None


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_from_value(value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return Path(text).expanduser().resolve()
    except Exception:
        return None


def _add_file_candidate(
    files: Dict[str, Path],
    value: Any,
    *,
    skip_paths: Iterable[Path],
) -> None:
    path = _path_from_value(value)
    if path is None or not path.is_file():
        return
    if path in skip_paths:
        return
    files.setdefault(str(path).casefold(), path)


def _collect_asset_files(
    session: Dict[str, Any],
    *,
    image_path: Optional[Path],
) -> Iterable[Path]:
    files: Dict[str, Path] = {}
    skip_paths = {image_path} if image_path is not None else set()
    for key in OUTPUT_REFERENCE_KEYS:
        path = _path_from_value(session.get(key))
        if path is not None:
            skip_paths.add(path)

    for key in SESSION_FILE_KEYS:
        _add_file_candidate(files, session.get(key), skip_paths=skip_paths)

    assets_dir = _path_from_value(session.get("assets_dir"))
    if assets_dir is not None and assets_dir.is_dir():
        for path in assets_dir.rglob("*"):
            if path.is_file():
                _add_file_candidate(files, path, skip_paths=skip_paths)

    for region in session.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        for key in REGION_FILE_KEYS:
            _add_file_candidate(files, region.get(key), skip_paths=skip_paths)

    return [files[key] for key in sorted(files)]


def build_editor_package(session_path: Any, *, image_path: Any = None) -> bytes:
    session_file = Path(str(session_path or "")).resolve()
    if not session_file.is_file():
        raise FileNotFoundError(f"Editor session not found: {session_file}")

    embedded_image_path = (
        Path(str(image_path)).resolve() if image_path is not None else None
    )
    with open(session_file, "r", encoding="utf-8") as handle:
        session = json.load(handle)

    package_session = deepcopy(session)
    package_session["embedded_editor_package"] = {
        "version": PACKAGE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_image_name": embedded_image_path.name if embedded_image_path else "",
    }

    asset_map = []
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, path in enumerate(
            _collect_asset_files(package_session, image_path=embedded_image_path),
            start=1,
        ):
            archive_name = _safe_archive_name(path, index)
            archive.write(path, archive_name)
            asset_map.append(
                {
                    "original": str(path),
                    "archive": archive_name,
                }
            )

        manifest = {
            "version": PACKAGE_VERSION,
            "session_name": session_file.name,
            "asset_map": asset_map,
        }
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "edit_session.json",
            json.dumps(package_session, ensure_ascii=False, indent=2),
        )
    return output.getvalue()


def embed_editor_package(
    image_path: Any,
    session_path: Any,
    *,
    verbose: bool = False,
) -> bool:
    image_file = Path(str(image_path or "")).resolve()
    if image_file.suffix.lower() not in SUPPORTED_EMBED_IMAGE_SUFFIXES:
        return False
    if not image_file.is_file():
        raise FileNotFoundError(f"Image not found: {image_file}")

    package = build_editor_package(session_path, image_path=image_file)
    image_data, _old_package = _split_trailer(image_file.read_bytes())
    footer = (
        struct.pack("<Q", len(package))
        + hashlib.sha256(package).digest()
        + TRAILER_MAGIC
    )
    _atomic_write(image_file, image_data + package + footer)
    log_message(
        f"Embedded editor package into image: {image_file}",
        verbose=verbose,
    )
    return True


def _safe_extract_zip(package: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(package), "r") as archive:
        for info in archive.infolist():
            member = Path(info.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe package member: {info.filename}")
            target = (destination / member).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:
                raise ValueError(f"Unsafe package member: {info.filename}") from exc
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, open(target, "wb") as output:
                output.write(source.read())


def _rewrite_session_paths(
    session: Dict[str, Any],
    *,
    replacements: Dict[str, str],
    image_path: Path,
    session_path: Path,
    assets_dir: Path,
) -> Dict[str, Any]:
    def rewrite_value(value: Any) -> Any:
        path = _path_from_value(value)
        if path is None:
            return value
        return replacements.get(str(path).casefold(), value)

    for key in SESSION_FILE_KEYS:
        if key in session:
            session[key] = rewrite_value(session.get(key))

    for region in session.get("regions", []) or []:
        if not isinstance(region, dict):
            continue
        for key in REGION_FILE_KEYS:
            if key in region:
                region[key] = rewrite_value(region.get(key))

    image_text = str(image_path)
    original_snapshot = Path(str(session.get("original_snapshot_path") or ""))
    session["assets_dir"] = str(assets_dir)
    session["translated_image_path"] = image_text
    session["overwrite_target_path"] = image_text
    session["source_overwrite_available"] = True
    if not str(session.get("source_display_path") or "").strip():
        session["source_display_path"] = image_path.name
    if not Path(str(session.get("source_image_path") or "")).is_file():
        if original_snapshot.is_file():
            session["source_image_path"] = str(original_snapshot)
        else:
            session["source_image_path"] = image_text
    edited_suffix = (
        image_path.suffix
        if image_path.suffix.lower() in SUPPORTED_EMBED_IMAGE_SUFFIXES
        else ".png"
    )
    session["edited_image_path"] = str(
        session_path.with_name(f"{image_path.stem}_edited{edited_suffix}").resolve()
    )
    session["editor_session_group"] = [str(session_path)]
    session["editor_session_group_index"] = 0
    package_info = dict(session.get("embedded_editor_package") or {})
    package_info.update(
        {
            "source_image_path": image_text,
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    session["embedded_editor_package"] = package_info
    return session


def extract_embedded_editor_package(
    image_path: Any,
    *,
    destination_root: Any = None,
) -> Optional[Path]:
    image_file = Path(str(image_path or "")).resolve()
    package = read_embedded_editor_package(image_file)
    if package is None:
        return None

    stripped_image, _payload = _split_trailer(image_file.read_bytes())
    digest = hashlib.sha256(stripped_image + package).hexdigest()[:16]
    root = (
        Path(str(destination_root)).resolve()
        if destination_root is not None
        else _repo_root() / "editor_sessions" / "embedded"
    )
    session_dir = root / f"{_clean_name(image_file.stem, 'image')}_{digest}"
    _safe_extract_zip(package, session_dir)

    manifest_path = session_dir / "manifest.json"
    session_path = session_dir / "edit_session.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("version", 0)) != PACKAGE_VERSION:
        raise ValueError("Unsupported embedded editor package version")
    with open(session_path, "r", encoding="utf-8") as handle:
        session = json.load(handle)

    replacements: Dict[str, str] = {}
    for item in manifest.get("asset_map", []) or []:
        if not isinstance(item, dict):
            continue
        original = _path_from_value(item.get("original"))
        archive_name = str(item.get("archive") or "").replace("\\", "/").strip("/")
        if original is None or not archive_name:
            continue
        replacement = (session_dir / archive_name).resolve()
        replacements[str(original).casefold()] = str(replacement)

    assets_dir = (session_dir / "assets").resolve()
    session = _rewrite_session_paths(
        session,
        replacements=replacements,
        image_path=image_file,
        session_path=session_path.resolve(),
        assets_dir=assets_dir,
    )
    with open(session_path, "w", encoding="utf-8") as handle:
        json.dump(session, handle, ensure_ascii=False, indent=2)
    return session_path.resolve()


def resolve_editor_session_input(input_path: Any) -> Path:
    path = Path(str(input_path or "")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Editor input not found: {path}")
    if path.suffix.lower() == ".json":
        return path
    session_path = extract_embedded_editor_package(path)
    if session_path is None:
        raise FileNotFoundError(
            "No embedded MangaTranslator editor package was found in this image."
        )
    return session_path
