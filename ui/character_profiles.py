from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROFILE_FILE = (
    Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
    / "MangaTranslator"
    / "character_name_profiles.json"
)

CHARACTER_NAME_RULE = (
    "0. Character Name\n"
    "If you find the names below in the text, use these translations."
)


def load_profiles() -> dict[str, Any]:
    empty = {"active": "", "profiles": {}}
    try:
        data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return empty

    raw_profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
    profiles = {
        str(name).strip(): str(entries).strip()
        for name, entries in raw_profiles.items()
        if str(name).strip()
    } if isinstance(raw_profiles, dict) else {}
    active = str(data.get("active", "")).strip() if isinstance(data, dict) else ""
    return {"active": active if active in profiles else "", "profiles": profiles}


def save_profiles(data: dict[str, Any]) -> None:
    PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path = PROFILE_FILE.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(PROFILE_FILE)


def profile_choices(data: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    store = data or load_profiles()
    names = sorted(store["profiles"], key=str.casefold)
    return [("사용 안 함", ""), *((name, name) for name in names)]


def select_profile(name: str) -> tuple[dict[str, Any], str]:
    store = load_profiles()
    selected = str(name or "").strip()
    if selected not in store["profiles"]:
        selected = ""
    store["active"] = selected
    save_profiles(store)
    return store, store["profiles"].get(selected, "")


def save_profile(original_name: str, name: str, entries: str) -> dict[str, Any]:
    store = load_profiles()
    original = str(original_name or "").strip()
    target = str(name or "").strip()
    content = str(entries or "").strip()
    if not target:
        raise ValueError("작품 이름을 입력해 주세요.")
    if not content:
        raise ValueError("일본어 이름과 사용할 한국어 표기를 입력해 주세요.")
    if target != original and target in store["profiles"]:
        raise ValueError("같은 작품 이름의 사전이 이미 있습니다.")
    if original and original != target:
        store["profiles"].pop(original, None)
    store["profiles"][target] = content
    store["active"] = target
    save_profiles(store)
    return store


def delete_profile(name: str) -> dict[str, Any]:
    store = load_profiles()
    target = str(name or "").strip()
    if target:
        store["profiles"].pop(target, None)
    store["active"] = ""
    save_profiles(store)
    return store


def merge_with_special_instructions(profile_name: str, instructions: str) -> str:
    store = load_profiles()
    entries = store["profiles"].get(str(profile_name or "").strip(), "").strip()
    base = str(instructions or "").strip()
    if not entries:
        return base
    character_section = f"{CHARACTER_NAME_RULE}\n{entries}"
    return f"{character_section}\n\n{base}" if base else character_section
