from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List, Mapping

TAG_DEFINITIONS = OrderedDict([
    ("BN", "말풍선: 일반 대화"),
    ("BE", "말풍선: 굵게/크게/외침/강한 감정 대화"),
    ("BT", "말풍선: 생각/속마음/내면 독백"),
    ("BW", "말풍선: 작은 목소리/속삭임 대사"),
    ("ON", "말풍선 밖: 내레이션/설명/회상 내레이션"),
    ("OT", "말풍선 밖: 제목/챕터명/장소명/날짜 표시"),
    ("OM", "말풍선 밖: 독백/감정 문장"),
    ("OD", "말풍선 밖: 일반 대화/평범한 대사"),
    ("OE", "말풍선 밖: 큰 글씨/시각적 강조 문구"),
    ("OH", "말풍선 밖: 종이 위 손글씨"),
    ("SFX", "효과음: 일반 효과음"),
    ("SFXI", "효과음: 충격/폭발/전투/충돌/큰 소리"),
    ("SFXS", "효과음: 귀여움/부드러움/잔잔함/코믹 효과음"),
])

TAG_STYLE_COLUMNS = [
    "태그",
    "설명",
    "글꼴",
    "최소 글꼴 크기",
    "최대 글꼴 크기",
    "줄 간격 배율",
    "세로쓰기 줄 간격 배율",
    "세로쓰기 최대 글자 수",
]

_NUMERIC_DEFAULTS = {
    "min_font_size": 8,
    "max_font_size": 16,
    "line_spacing_mult": 1.0,
    "vertical_line_spacing_mult": 1.0,
    "vertical_max_chars": 4,
}


def default_tag_style_rows() -> List[List[Any]]:
    return [
        [
            tag,
            description,
            "",
            _NUMERIC_DEFAULTS["min_font_size"],
            _NUMERIC_DEFAULTS["max_font_size"],
            _NUMERIC_DEFAULTS["line_spacing_mult"],
            _NUMERIC_DEFAULTS["vertical_line_spacing_mult"],
            _NUMERIC_DEFAULTS["vertical_max_chars"],
        ]
        for tag, description in TAG_DEFINITIONS.items()
    ]


def _rows_from_any(value: Any) -> List[List[Any]]:
    if value is None:
        return []

    # Gradio Dataframe commonly returns a pandas.DataFrame.
    if hasattr(value, "values") and hasattr(value, "columns"):
        try:
            return value.values.tolist()
        except Exception:
            pass

    if isinstance(value, Mapping):
        # Either {"BN": {...}} or {"data": [[...]]} style.
        if "data" in value and isinstance(value.get("data"), list):
            return value.get("data") or []
        rows: List[List[Any]] = []
        for tag, cfg in value.items():
            tag_name = str(tag).upper().strip().strip("<>")
            if tag_name not in TAG_DEFINITIONS:
                continue
            cfg = cfg if isinstance(cfg, Mapping) else {}
            rows.append([
                tag_name,
                TAG_DEFINITIONS[tag_name],
                cfg.get("font_pack", cfg.get("font_dir", cfg.get("글꼴", ""))) or "",
                cfg.get("min_font_size", cfg.get("최소 글꼴 크기", _NUMERIC_DEFAULTS["min_font_size"])),
                cfg.get("max_font_size", cfg.get("최대 글꼴 크기", _NUMERIC_DEFAULTS["max_font_size"])),
                cfg.get("line_spacing_mult", cfg.get("줄 간격 배율", _NUMERIC_DEFAULTS["line_spacing_mult"])),
                cfg.get("vertical_line_spacing_mult", cfg.get("세로쓰기 줄 간격 배율", _NUMERIC_DEFAULTS["vertical_line_spacing_mult"])),
                cfg.get("vertical_max_chars", cfg.get("세로쓰기 최대 글자 수", _NUMERIC_DEFAULTS["vertical_max_chars"])),
            ])
        return rows

    if isinstance(value, list):
        if value and isinstance(value[0], Mapping):
            rows = []
            for item in value:
                tag_name = str(item.get("tag", item.get("Tag", item.get("태그", "")))).upper().strip().strip("<>")
                if tag_name not in TAG_DEFINITIONS:
                    continue
                rows.append([
                    tag_name,
                    TAG_DEFINITIONS[tag_name],
                    item.get("font_pack", item.get("Font Pack", item.get("글꼴", ""))) or "",
                    item.get("min_font_size", item.get("Min Font Size", item.get("최소 글꼴 크기", _NUMERIC_DEFAULTS["min_font_size"]))),
                    item.get("max_font_size", item.get("Max Font Size", item.get("최대 글꼴 크기", _NUMERIC_DEFAULTS["max_font_size"]))),
                    item.get("line_spacing_mult", item.get("Line Spacing", item.get("줄 간격 배율", _NUMERIC_DEFAULTS["line_spacing_mult"]))),
                    item.get("vertical_line_spacing_mult", item.get("Vertical Line Spacing", item.get("세로쓰기 줄 간격 배율", _NUMERIC_DEFAULTS["vertical_line_spacing_mult"]))),
                    item.get("vertical_max_chars", item.get("Vertical Max Chars", item.get("세로쓰기 최대 글자 수", _NUMERIC_DEFAULTS["vertical_max_chars"]))),
                ])
            return rows
        return value

    return []


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def normalize_tag_style_table(value: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize a UI table/dict into a stable serializable tag style dict."""
    rows = _rows_from_any(value)
    result: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if not isinstance(row, (list, tuple)) or not row:
            continue
        tag_name = str(row[0]).upper().strip().strip("<>")
        if tag_name not in TAG_DEFINITIONS:
            continue
        font_pack = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""
        result[tag_name] = {
            "font_pack": font_pack,
            "min_font_size": _safe_int(row[3] if len(row) > 3 else None, _NUMERIC_DEFAULTS["min_font_size"], 1, 256),
            "max_font_size": _safe_int(row[4] if len(row) > 4 else None, _NUMERIC_DEFAULTS["max_font_size"], 1, 256),
            "line_spacing_mult": _safe_float(row[5] if len(row) > 5 else None, _NUMERIC_DEFAULTS["line_spacing_mult"], 0.1, 5.0),
            "vertical_line_spacing_mult": _safe_float(row[6] if len(row) > 6 else None, _NUMERIC_DEFAULTS["vertical_line_spacing_mult"], 0.1, 5.0),
            "vertical_max_chars": _safe_int(row[7] if len(row) > 7 else None, _NUMERIC_DEFAULTS["vertical_max_chars"], 0, 200),
        }
        if result[tag_name]["max_font_size"] < result[tag_name]["min_font_size"]:
            result[tag_name]["max_font_size"] = result[tag_name]["min_font_size"]

    for tag_name in TAG_DEFINITIONS:
        result.setdefault(tag_name, {
            "font_pack": "",
            "min_font_size": _NUMERIC_DEFAULTS["min_font_size"],
            "max_font_size": _NUMERIC_DEFAULTS["max_font_size"],
            "line_spacing_mult": _NUMERIC_DEFAULTS["line_spacing_mult"],
            "vertical_line_spacing_mult": _NUMERIC_DEFAULTS["vertical_line_spacing_mult"],
            "vertical_max_chars": _NUMERIC_DEFAULTS["vertical_max_chars"],
        })

    return result


def tag_styles_to_table(value: Any) -> List[List[Any]]:
    styles = normalize_tag_style_table(value)
    rows: List[List[Any]] = []
    for tag_name, description in TAG_DEFINITIONS.items():
        cfg = styles.get(tag_name, {})
        rows.append([
            tag_name,
            description,
            cfg.get("font_pack", "") or "",
            cfg.get("min_font_size", _NUMERIC_DEFAULTS["min_font_size"]),
            cfg.get("max_font_size", _NUMERIC_DEFAULTS["max_font_size"]),
            cfg.get("line_spacing_mult", _NUMERIC_DEFAULTS["line_spacing_mult"]),
            cfg.get("vertical_line_spacing_mult", _NUMERIC_DEFAULTS["vertical_line_spacing_mult"]),
            cfg.get("vertical_max_chars", _NUMERIC_DEFAULTS["vertical_max_chars"]),
        ])
    return rows
