"""Font fallback helpers for the Skia/HarfBuzz renderer.

The renderer normally uses one selected font family.  Some handwriting fonts are
missing CJK punctuation such as 【】, so this module finds a fallback font and
splits only the unsupported characters into fallback-font runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.text.font_manager import get_font_cmap
from utils.logging import log_message

FALLBACK_FONT_KEY = "fallback"
STYLE_MARKER_CHARS = {"*"}
WHITESPACE_CHARS = {" ", "\t", "\n", "\r"}
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc"}

# Prefer broad CJK/Korean fonts before decorative fonts.  The sort is only a
# priority hint; every candidate is still verified by cmap before use.
PRIORITY_NAME_HINTS = [
    "notosanscjkkr",
    "notosanscjk",
    "notosanskr",
    "sourcehansansk",
    "sourcehansanskr",
    "sourcehansans",
    "notoserifcjkkr",
    "notoserifcjk",
    "notoserifkr",
    "sourcehanserifk",
    "sourcehanserifkr",
    "sourcehanserif",
    "nanumgothic",
    "nanummyeongjo",
    "malgun",
    "droidsansfallback",
    "arialuni",
    "arialunicode",
    "unifont",
]

_fallback_font_cache: Dict[Tuple[Tuple[int, ...], Tuple[str, ...]], Optional[Path]] = {}


def _normalize_name(path: Path) -> str:
    return "".join(ch for ch in path.stem.lower() if ch.isalnum())


def _candidate_score(path: Path) -> Tuple[int, int, str]:
    name = _normalize_name(path)
    for index, hint in enumerate(PRIORITY_NAME_HINTS):
        if hint in name:
            return (index, len(path.name), path.name.lower())
    return (len(PRIORITY_NAME_HINTS), len(path.name), path.name.lower())


def _iter_font_files_in_dir(directory: Path, recursive: bool) -> Iterable[Path]:
    if not directory or not directory.exists() or not directory.is_dir():
        return
    try:
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        for path in iterator:
            if path.is_file() and path.suffix.lower() in FONT_EXTENSIONS:
                yield path
    except Exception:
        return


def _system_font_dirs() -> List[Path]:
    dirs: List[Path] = []
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(windir) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs.extend(
            [
                Path("/System/Library/Fonts"),
                Path("/Library/Fonts"),
                Path.home() / "Library" / "Fonts",
            ]
        )
    else:
        dirs.extend(
            [
                Path("/usr/share/fonts"),
                Path("/usr/local/share/fonts"),
                Path.home() / ".local" / "share" / "fonts",
                Path.home() / ".fonts",
            ]
        )
    return dirs


def _fallback_search_dirs(primary_font_dir: Optional[str]) -> List[Tuple[Path, bool]]:
    """Return (directory, recursive) pairs to search for fallback fonts."""
    dirs: List[Tuple[Path, bool]] = []

    env_dir = os.environ.get("MANGA_TRANSLATOR_FALLBACK_FONT_DIR")
    if env_dir:
        dirs.append((Path(env_dir), True))

    if primary_font_dir:
        font_dir = Path(primary_font_dir).resolve()
        dirs.append((font_dir, False))
        parent = font_dir.parent
        # Font packs usually live below ./fonts/<pack-name>.  Searching the
        # parent recursively lets the user simply drop Noto/SourceHan in any
        # font pack folder without adding a UI setting.
        dirs.append((parent, True))
        for name in ("_fallback", "fallback", "Fallback", "NotoSansCJK", "SourceHan"):
            dirs.append((parent / name, True))

    for directory in _system_font_dirs():
        dirs.append((directory, True))

    # De-duplicate while preserving order.  Recursive=True wins if the same dir
    # appears twice.
    merged: Dict[Path, bool] = {}
    for directory, recursive in dirs:
        try:
            resolved = directory.resolve()
        except Exception:
            resolved = directory
        merged[resolved] = merged.get(resolved, False) or recursive
    return list(merged.items())


def _all_supported_by(path: Path, needed_codepoints: Set[int]) -> bool:
    cmap = get_font_cmap(str(path))
    return bool(cmap) and needed_codepoints.issubset(cmap)


def find_fallback_font_for_codepoints(
    needed_codepoints: Set[int],
    primary_font_path: Optional[str] = None,
    primary_font_dir: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Path]:
    """Find one fallback font that supports every requested codepoint."""
    if not needed_codepoints:
        return None

    env_file = os.environ.get("MANGA_TRANSLATOR_FALLBACK_FONT")
    if env_file:
        env_path = Path(env_file)
        if env_path.exists() and env_path.suffix.lower() in FONT_EXTENSIONS:
            if _all_supported_by(env_path, needed_codepoints):
                log_message(
                    f"Using fallback font from MANGA_TRANSLATOR_FALLBACK_FONT: {env_path.name}",
                    verbose=verbose,
                )
                return env_path
            log_message(
                f"Fallback font env file does not support required glyphs: {env_path.name}",
                verbose=verbose,
            )

    search_dirs = _fallback_search_dirs(primary_font_dir)
    cache_key = (
        tuple(sorted(needed_codepoints)),
        tuple(str(directory) + ("/**" if recursive else "") for directory, recursive in search_dirs),
    )
    if cache_key in _fallback_font_cache:
        return _fallback_font_cache[cache_key]

    primary_resolved: Optional[Path] = None
    if primary_font_path:
        try:
            primary_resolved = Path(primary_font_path).resolve()
        except Exception:
            primary_resolved = Path(primary_font_path)

    candidates: List[Path] = []
    seen: Set[Path] = set()
    for directory, recursive in search_dirs:
        for path in _iter_font_files_in_dir(directory, recursive):
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if primary_resolved is not None and resolved == primary_resolved:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(resolved)

    candidates.sort(key=_candidate_score)

    for path in candidates:
        try:
            if _all_supported_by(path, needed_codepoints):
                _fallback_font_cache[cache_key] = path
                log_message(
                    "Using font fallback for unsupported glyphs: "
                    f"{path.name} ({path.parent})",
                    always_print=True,
                )
                return path
        except Exception:
            continue

    _fallback_font_cache[cache_key] = None
    chars = "".join(chr(cp) for cp in sorted(needed_codepoints))
    log_message(
        f"No fallback font found for unsupported glyphs: {chars}",
        always_print=True,
    )
    return None


def missing_codepoints_for_text(text: str, primary_cmap: Set[int]) -> Set[int]:
    """Codepoints unsupported by the primary font, ignoring markers/space."""
    if not text or not primary_cmap:
        return set()
    missing: Set[int] = set()
    for char in text:
        if char in STYLE_MARKER_CHARS or char in WHITESPACE_CHARS:
            continue
        cp = ord(char)
        if cp not in primary_cmap:
            missing.add(cp)
    return missing


def sanitize_text_with_font_fallback(
    text: str,
    font_cmaps: Dict[str, Set[int]],
    verbose: bool = False,
) -> str:
    """Remove only characters unsupported by every loaded render font."""
    if not text:
        return text

    cmaps = [cmap for cmap in font_cmaps.values() if cmap]
    if not cmaps:
        return text

    removed_chars: List[str] = []
    sanitized_chars: List[str] = []
    for char in text:
        if char in STYLE_MARKER_CHARS or char in WHITESPACE_CHARS:
            sanitized_chars.append(char)
            continue
        cp = ord(char)
        if any(cp in cmap for cmap in cmaps):
            sanitized_chars.append(char)
        else:
            removed_chars.append(char)

    if removed_chars:
        unique_removed = sorted(set(removed_chars), key=lambda c: ord(c))
        char_descriptions = [f"'{c}' (U+{ord(c):04X})" for c in unique_removed[:10]]
        if len(unique_removed) > 10:
            char_descriptions.append(f"... and {len(unique_removed) - 10} more")
        log_message(
            f"Removed {len(removed_chars)} unsupported character(s) after fallback search: "
            f"{', '.join(char_descriptions)}",
            always_print=True,
        )

    return "".join(sanitized_chars)


def _candidate_font_keys(preferred_key: str, font_cmaps: Dict[str, Set[int]]) -> List[str]:
    keys: List[str] = []

    def add(key: str) -> None:
        if key and key in font_cmaps and key not in keys:
            keys.append(key)

    add(preferred_key)
    if preferred_key != "regular":
        add("regular")
    add(FALLBACK_FONT_KEY)
    return keys


def pick_font_key_for_char(
    char: str,
    preferred_key: str,
    font_cmaps: Optional[Dict[str, Set[int]]],
) -> str:
    """Pick style/regular/fallback for one character."""
    if not font_cmaps or char.isspace():
        return preferred_key

    cp = ord(char)
    for key in _candidate_font_keys(preferred_key, font_cmaps):
        cmap = font_cmaps.get(key)
        if not cmap or cp in cmap:
            return key
    return preferred_key


def split_text_by_font_support(
    text: str,
    preferred_key: str,
    font_cmaps: Optional[Dict[str, Set[int]]],
) -> List[Tuple[str, str]]:
    """Split text into runs that should be shaped with the same font key."""
    if not text:
        return []
    if not font_cmaps:
        return [(text, preferred_key)]

    runs: List[Tuple[str, str]] = []
    current_text: List[str] = []
    current_key: Optional[str] = None

    for char in text:
        key = pick_font_key_for_char(char, preferred_key, font_cmaps)
        if current_key is None:
            current_key = key
            current_text = [char]
        elif key == current_key:
            current_text.append(char)
        else:
            runs.append(("".join(current_text), current_key))
            current_key = key
            current_text = [char]

    if current_text and current_key is not None:
        runs.append(("".join(current_text), current_key))
    return runs
