"""Record generated lettering separately from the artwork needed for export."""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

from core.editor_batch_progress import _process_exists

history_revision = 0

REPO = Path(__file__).resolve().parents[1]
PATCH_FILES = (
    "ai_original_path",
    "ai_processed_path",
    "ai_mask_path",
    "ai_background_path",
    "ai_lettering_path",
    "ai_erase_mask_path",
)


def attempt_directory(session, assets, repo=REPO):
    session = Path(session).resolve()
    if session.is_relative_to(repo):
        key = hashlib.sha256(
            session.relative_to(repo).as_posix().casefold().encode()
        ).hexdigest()[:24]
        assets = repo / "editor_history" / key
    return Path(assets) / ("ai_sfx_" + uuid4().hex)


def read_record(directory):
    try:
        value = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_record(directory, metadata, state, *, patch=None, error=""):
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(metadata, state=state, pid=os.getpid(), error=error)
    if patch:
        record["patch"] = {
            k: v for k, v in patch.items() if k in {"ai_image", "warning"}
        }
        for key in PATCH_FILES:
            if patch.get(key):
                record["patch"][key] = Path(patch[key]).name
    pending = directory / "history.tmp"
    pending.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    pending.replace(directory / "history.json")
    global history_revision
    history_revision += 1


def stored_draft(directory, record, session):
    patch = {
        k: v
        for k, v in (record.get("patch") or {}).items()
        if k in (*PATCH_FILES, "ai_image", "warning")
    }
    stamp = record.get("stamp")
    if record.get("state") != "ready" or not isinstance(stamp, list) or len(stamp) != 6:
        return None
    if not all(patch.get(key) for key in PATCH_FILES[:3]) or not isinstance(
        patch.get("ai_image"), dict
    ):
        return None
    if not re.fullmatch(r"[a-fA-F0-9]{32}", str(patch["ai_image"].get("id", ""))):
        return None
    if len(record.get("background_box") or []) != 4 or not record.get(
        "background_digest"
    ):
        return None
    for key in PATCH_FILES:
        if key not in patch:
            continue
        name = patch[key]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.lower().endswith(".png")
        ):
            return None
        path = (directory / name).resolve()
        if path.parent != directory.resolve() or not path.is_file():
            return None
        patch[key] = str(path)
    stamp = list(stamp)
    stamp[1] = tuple(stamp[1])
    return dict(
        session_path=str(session),
        stamp=tuple(stamp),
        patch=patch,
        region_index=record["region_index"],
        background_box=record["background_box"],
        background_digest=record["background_digest"],
    )


def state_label(record):
    state = record.get("state")
    if state == "running":
        return "생성 중" if _process_exists(int(record.get("pid") or 0)) else "중단됨"
    return {"ready": "미적용", "failed": "실패", "cancelled": "중단됨"}.get(
        state, "확인 필요"
    )


def preserve_applied_files(patch, assets):
    """Install immutable session copies before publishing any region paths."""
    result = dict(patch)
    image_id = patch["ai_image"]["id"]
    if not re.fullmatch(r"[a-fA-F0-9]{32}", str(image_id)):
        raise ValueError("식자 결과 식별자를 확인할 수 없습니다.")
    assets.mkdir(parents=True, exist_ok=True)
    for key in PATCH_FILES:
        if not patch.get(key):
            continue
        target = assets / f"ai_{image_id}_{key[3:]}.png"
        if not target.is_file():
            pending = target.with_suffix("." + uuid4().hex + ".tmp")
            try:
                shutil.copyfile(patch[key], pending)
                pending.replace(target)
            finally:
                pending.unlink(missing_ok=True)
        result[key] = str(target.resolve())
    return result
