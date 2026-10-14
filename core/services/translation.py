import base64
import html
import json
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from core.caching import get_cache
from core.config import TranslationConfig, calculate_reasoning_budget
from core.image.image_utils import cv2_to_pil, pil_to_cv2, process_bubble_image_cached
from core.image.ocr_detection import (
    extract_text_with_manga_ocr,
    extract_text_with_paddle_ocr_vl,
)
from core.text.text_processing import TRANSLATION_TAGS, strip_translation_terminal_periods
from utils.token_usage import accumulate_usage
from utils.endpoints import (
    call_anthropic_endpoint,
    call_codex_endpoint,
    call_deepseek_endpoint,
    call_gemini_endpoint,
    call_moonshot_endpoint,
    call_openai_compatible_endpoint,
    call_openai_endpoint,
    call_openrouter_endpoint,
    call_xai_endpoint,
    call_zai_endpoint,
    openrouter_is_reasoning_model,
)
from utils.exceptions import TranslationError
from utils.logging import log_message
from utils.model_metadata import (
    get_gpt5_generation,
    get_max_tokens_cap,
    is_46_model,
    is_anthropic_reasoning_model,
    is_deepseek_reasoning_model,
    is_gemini_3_model,
    is_gemini_25_flash_model,
    is_gemini_25_pro_model,
    is_gemma_model,
    is_google_reasoning_model,
    is_gpt5_chat_variant,
    is_gpt5_series,
    is_openai_compatible_reasoning_model,
    is_openai_model_family,
)
from utils.model_metadata import is_openai_reasoning_model as _is_openai_reasoning_meta
from utils.model_metadata import (
    is_opus_45_model,
    is_opus_47_model,
    is_rosetta_model,
    is_xai_reasoning_model,
    is_zai_reasoning_model,
    supports_openai_original_image_detail,
    supports_xai_reasoning_parameter,
)

TRANSLATION_PATTERN = re.compile(
    r'^\s*(\d+)\s*:\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*:|\s*$)',
    re.MULTILINE | re.DOTALL,
)
EMBEDDED_ITEM_LABEL_VERSION = "embedded-item-label-v1"
ITEM_LABEL_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "DejaVuSans-Bold.ttf",
)
LOCAL_GEMMA_RESPONSE_TAGS = {
    "BN",
    "BE",
    "BT",
    "BW",
    "ON",
    "OT",
    "OM",
    "OD",
    "OE",
    "SFX",
    "SFXI",
    "SFXS",
}
LOCAL_GEMMA_TAG_ALIASES = {
    "NORMAL": "BN",
    "NORMAL_DIALOGUE": "BN",
    "DIALOGUE": "BN",
    "SPEECH": "BN",
    "SPEECH_BUBBLE": "BN",
    "EMPHASIS": "BE",
    "EMPHASIZED": "BE",
    "BOLD": "BE",
    "SHOUT": "BE",
    "SHOUTING": "BE",
    "THOUGHT": "BT",
    "THOUGHTS": "BT",
    "INNER_MONOLOGUE": "BT",
    "WHISPER": "BW",
    "WHISPERED": "BW",
    "NARRATION": "ON",
    "TITLE": "OT",
    "MONOLOGUE": "OM",
    "SOUND_EFFECT": "SFX",
    "SOUNDEFFECT": "SFX",
    "IMPACT": "SFXI",
    "SOFT_SFX": "SFXS",
}
LOCAL_GEMMA_TAG_WRAPPER_PATTERN = re.compile(
    r"<\s*([A-Za-z0-9_-]+)\s*>\s*(.*?)\s*<\s*/\s*\1\s*>",
    re.DOTALL,
)
CONJOINED_WHITEOUT_RIGHT_NEIGHBOR_MIN_SEPARATION_RATIO = 0.25
CONJOINED_WHITEOUT_RIGHT_NEIGHBOR_MIN_SEPARATION_PX = 18.0
CONJOINED_WHITEOUT_RIGHT_NEIGHBOR_MIN_OVERLAP_RATIO = 0.10
CONJOINED_WHITEOUT_RESCUE_MAX_PROTECTED_DARK_PIXELS = 500
CONJOINED_WHITEOUT_RESCUE_MAX_PROTECTED_DARK_RATIO = 0.035
CONJOINED_WHITEOUT_NEIGHBOR_SIDE_MARGIN_PX = 4.0
CONJOINED_WHITEOUT_NEIGHBOR_SIDE_MARGIN_RATIO = 0.03
CONJOINED_WHITEOUT_TEXT_LINE_ASSIGN_MARGIN_PX = 8.0
CONJOINED_WHITEOUT_TEXT_LINE_ASSIGN_MARGIN_RATIO = 0.04
CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS = 12
CONJOINED_WHITEOUT_TEXT_LINE_MIN_NEIGHBOR_OVERLAP_RATIO = 0.18
CONJOINED_WHITEOUT_TEXT_LINE_MAX_DARK_AREA_RATIO = 0.20
CONJOINED_WHITEOUT_TEXT_LINE_MAX_OWN_PROTECT_RATIO = 0.45
CONJOINED_WHITEOUT_CURRENT_TEXT_ASSIGN_MARGIN = 0.05
CONJOINED_WHITEOUT_AXIS_ASSIGN_MARGIN = 0.10
CONJOINED_WHITEOUT_AXIS_MAX_DARK_AREA_RATIO = 0.35
CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO = 0.30
CONJOINED_WHITEOUT_BORDERLINE_MIN_CHILD_OVERLAP_RATIO = 0.27
CONJOINED_WHITEOUT_BORDERLINE_MIN_OWN_PROTECT_RATIO = 0.90
CONJOINED_WHITEOUT_BORDERLINE_MIN_LINE_ASPECT_RATIO = 1.35
CONJOINED_WHITEOUT_BORDERLINE_MIN_NEIGHBOR_MASK_RATIO = 0.85
CONJOINED_WHITEOUT_BORDERLINE_MAX_OWN_PROTECT_RATIO = 0.10
CONJOINED_WHITEOUT_BORDERLINE_MAX_COMPONENT_AREA_RATIO = 0.04
CONJOINED_WHITEOUT_BORDERLINE_MAX_COMPONENT_DIMENSION_RATIO = 0.40
CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_MIN_SEPARATION_RATIO = 0.35
CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_ASSIGN_MARGIN = 0.05
CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_MAX_DARK_AREA_RATIO = 0.70
CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_MAX_COMPONENT_HEIGHT_RATIO = 0.60
CONJOINED_WHITEOUT_SIDE_RESCUE_ASSIGN_MARGIN = 0.03
CONJOINED_WHITEOUT_SIDE_RESCUE_MIN_NEIGHBOR_SIDE_RATIO = 0.55
CONJOINED_WHITEOUT_SIDE_RESCUE_MIN_NEIGHBOR_OVERLAP_RATIO = 0.06
CONJOINED_WHITEOUT_SIDE_RESCUE_MAX_DARK_AREA_RATIO = 0.12
CONJOINED_WHITEOUT_SIDE_RESCUE_MAX_OWN_RATIO = 0.90
CONJOINED_WHITEOUT_TEXT_BOX_RESCUE_ASSIGN_MARGIN = 0.03
CONJOINED_WHITEOUT_TEXT_BOX_RESCUE_MAX_DARK_AREA_RATIO = 0.08
CONJOINED_WHITEOUT_TEXT_BOX_RESCUE_MAX_DIMENSION_RATIO = 0.75


def _cv_white_value(image: np.ndarray):
    """Return a white fill value compatible with grayscale, BGR, or BGRA arrays."""
    if image.ndim == 2:
        return 255
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 4:
            return (255, 255, 255, 255)
        if channels == 3:
            return (255, 255, 255)
        return tuple([255] * channels)
    return 255


def _strip_response_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _extract_json_candidate(text: str) -> Optional[str]:
    cleaned = _strip_response_code_fence(text)
    if not cleaned:
        return None
    if cleaned[0] in "{[":
        return cleaned

    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
    if not starts:
        return None

    start = min(starts)
    opening = cleaned[start]
    closing = "}" if opening == "{" else "]"
    end = cleaned.rfind(closing)
    if end <= start:
        return None
    return cleaned[start : end + 1].strip()


def _coerce_response_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        source_keys = ("ocr", "transcription", "transcribed", "source")
        translation_keys = ("translation", "translated", "target")
        source_value = next((value[key] for key in source_keys if key in value), None)
        translation_value = next(
            (value[key] for key in translation_keys if key in value),
            None,
        )
        if source_value is not None and translation_value is not None:
            return (
                f"{_coerce_response_value(source_value)} || "
                f"{_coerce_response_value(translation_value)}"
            ).strip()
        for key in (
            "translation",
            "translated",
            "text",
            "ocr",
            "transcription",
            "value",
        ):
            if key in value:
                return _coerce_response_value(value[key])
        return json.dumps(value, ensure_ascii=False).strip()
    if isinstance(value, list):
        return " ".join(_coerce_response_value(item) for item in value).strip()
    return str(value).strip()


def _is_local_gemma_model_name(model_name: Optional[str]) -> bool:
    """Recognize Gemma names used by llama.cpp, LM Studio, and Ollama."""
    if not model_name:
        return False
    return bool(
        re.search(
            r"(?:^|[/\\:_.\s-])gemma(?:$|[/\\:_.\s-]|\d)",
            str(model_name).lower(),
        )
    )


def _normalize_local_gemma_markup(value: Any) -> str:
    cleaned = html.unescape(str(value or ""))
    cleaned = cleaned.replace("\uff1c", "<").replace("\uff1e", ">").strip()
    cleaned = _strip_response_code_fence(cleaned)
    if len(cleaned) >= 2 and cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _normalize_local_gemma_tag(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = _normalize_local_gemma_markup(value)
    cleaned = re.sub(r"[*_`#]", "", cleaned).strip()
    match = re.fullmatch(
        r"\s*(?:<\s*/?\s*)?([A-Za-z0-9]+)(?:\s*>)?\s*",
        cleaned,
    )
    if match:
        tag = match.group(1).upper()
        if tag in LOCAL_GEMMA_RESPONSE_TAGS:
            return tag
        alias = LOCAL_GEMMA_TAG_ALIASES.get(tag)
        if alias:
            return alias

    alias_key = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_").upper()
    alias = LOCAL_GEMMA_TAG_ALIASES.get(alias_key)
    if alias:
        return alias

    for tag in sorted(LOCAL_GEMMA_RESPONSE_TAGS, key=len, reverse=True):
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(tag)}(?![A-Za-z0-9])",
            cleaned,
            re.IGNORECASE,
        ):
            return tag
    return None


def _first_local_gemma_tagged_block(text: str) -> Optional[tuple[str, str]]:
    cleaned = _normalize_local_gemma_markup(text)
    for match in LOCAL_GEMMA_TAG_WRAPPER_PATTERN.finditer(cleaned):
        tag = match.group(1).upper()
        if tag in LOCAL_GEMMA_RESPONSE_TAGS:
            return tag, match.group(2).strip()
    return None


def _clean_local_gemma_translation(
    text: Any,
    tag_hint: Optional[str] = None,
) -> str:
    cleaned = _normalize_local_gemma_markup(text)
    if not cleaned:
        return ""

    tagged = _first_local_gemma_tagged_block(cleaned)
    if tagged is not None:
        tag, inner = tagged
        return f"<{tag}>{inner}</{tag}>"

    opening = re.match(
        r"^\s*<\s*([A-Za-z0-9_-]+)\s*>\s*(.*?)"
        r"(?:<\s*/\s*[A-Za-z0-9_-]+\s*>)?\s*$",
        cleaned,
        re.DOTALL,
    )
    if opening:
        tag = opening.group(1).upper()
        normalized_tag = _normalize_local_gemma_tag(tag)
        if normalized_tag:
            return (
                f"<{normalized_tag}>{opening.group(2).strip()}"
                f"</{normalized_tag}>"
            )

    generic_wrapper = re.fullmatch(
        r"\s*<\s*([A-Za-z][A-Za-z0-9_-]*)\s*>\s*(.*?)"
        r"\s*<\s*/\s*\1\s*>\s*",
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if generic_wrapper:
        return generic_wrapper.group(2).strip()

    prefix = re.match(
        r"^\s*([A-Za-z0-9]+)\s*[:\uff1a-]\s*(.+?)\s*$",
        cleaned,
        re.DOTALL,
    )
    if prefix:
        prefix_tag = _normalize_local_gemma_tag(prefix.group(1))
        if prefix_tag:
            return f"<{prefix_tag}>{prefix.group(2).strip()}</{prefix_tag}>"

    tag = _normalize_local_gemma_tag(tag_hint or "")
    if tag:
        return f"<{tag}>{cleaned}</{tag}>"
    return cleaned


def _coerce_local_gemma_response_value(value: Any) -> str:
    """Normalize structured response variants emitted by local Gemma servers."""
    if isinstance(value, dict):
        normalized = {
            str(key).strip().lower().replace(" ", "_"): item
            for key, item in value.items()
        }
        source_keys = (
            "ocr",
            "ocr_text",
            "transcription",
            "transcribed",
            "source",
            "source_text",
            "original",
            "original_text",
            "\uc6d0\ubb38",
            "\uc77c\ubcf8\uc5b4",
            "\u539f\u6587",
            "\u30bd\u30fc\u30b9",
        )
        translation_keys = (
            "translation",
            "translated",
            "translated_text",
            "target",
            "target_text",
            "\ubc88\uc5ed",
            "\ubc88\uc5ed\ubb38",
            "\ubc88\uc5ed_\uacb0\uacfc",
            "\u7ffb\u8a33",
            "\u8a33\u6587",
        )
        tag_keys = (
            "tag",
            "type",
            "classification",
            "style",
            "render_tag",
            "\ud0dc\uadf8",
            "\ubd84\ub958",
            "\u30bf\u30b0",
            "\u5206\u985e",
        )

        source = next(
            (normalized[key] for key in source_keys if key in normalized),
            None,
        )
        translation = next(
            (normalized[key] for key in translation_keys if key in normalized),
            None,
        )
        tag_value = next(
            (normalized[key] for key in tag_keys if key in normalized),
            None,
        )
        tag = _normalize_local_gemma_tag(tag_value)

        if source is not None and translation is not None:
            source_text = _coerce_response_value(source)
            translation_text = _clean_local_gemma_translation(
                _coerce_response_value(translation),
                tag,
            )
            return f"{source_text} || {translation_text}".strip()
        if translation is not None:
            return _clean_local_gemma_translation(
                _coerce_response_value(translation),
                tag,
            )

    if isinstance(value, list) and len(value) == 3:
        tag = _normalize_local_gemma_tag(value[1])
        if tag:
            source_text = _coerce_response_value(value[0])
            translation_text = _clean_local_gemma_translation(
                _coerce_response_value(value[2]),
                tag,
            )
            return f"{source_text} || {translation_text}".strip()

    return _coerce_response_value(value)


def _local_gemma_label_kind(label: str) -> Optional[str]:
    source_labels = {
        "original",
        "original text",
        "source",
        "source text",
        "ocr",
        "ocr text",
        "transcription",
        "\uc6d0\ubb38",
        "\uc77c\ubcf8\uc5b4",
        "\u539f\u6587",
        "\u30bd\u30fc\u30b9",
    }
    tag_labels = {
        "tag",
        "type",
        "classification",
        "\ud0dc\uadf8",
        "\ubd84\ub958",
        "\u30bf\u30b0",
        "\u5206\u985e",
    }
    translation_labels = {
        "translation",
        "translated text",
        "target",
        "target text",
        "\ubc88\uc5ed",
        "\ubc88\uc5ed\ubb38",
        "\ubc88\uc5ed \uacb0\uacfc",
        "\u7ffb\u8a33",
        "\u8a33\u6587",
    }
    cleaned = _normalize_local_gemma_markup(label)
    cleaned = re.sub(r"^[\s>*#_-]+|[\s*#_-]+$", "", cleaned)
    cleaned = cleaned.lower().strip()
    if cleaned in source_labels:
        return "source"
    if cleaned in tag_labels:
        return "tag"
    if cleaned in translation_labels:
        return "translation"
    return None


def _strip_local_gemma_markdown_edge(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^(?:\*\*|__|`)+\s*", "", cleaned)
    cleaned = re.sub(r"\s*(?:\*\*|__|`)+$", "", cleaned)
    return cleaned.strip()


def _local_gemma_labeled_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current_field: Optional[str] = None

    for raw_line in _normalize_local_gemma_markup(text).splitlines():
        line = raw_line.strip()
        match = re.match(
            r"^\s*(?:[-*]\s*)?([^:\uff1a]+)[:\uff1a]\s*(.*)$",
            line,
        )
        if match:
            current_field = _local_gemma_label_kind(match.group(1))
            if current_field is not None:
                sections.setdefault(current_field, [])
                content = _strip_local_gemma_markdown_edge(match.group(2))
                if content:
                    sections[current_field].append(content)
                continue
            current_field = None
            continue

        heading_field = _local_gemma_label_kind(line)
        if heading_field is not None:
            current_field = heading_field
            sections.setdefault(current_field, [])
            continue

        if current_field is not None and line:
            sections[current_field].append(
                _strip_local_gemma_markdown_edge(line)
            )

    return sections


def _split_local_gemma_labeled_value(text: str) -> Optional[tuple[str, str]]:
    sections = _local_gemma_labeled_sections(text)

    source = "\n".join(sections.get("source", [])).strip()
    translation = "\n".join(sections.get("translation", [])).strip()
    if not source or not translation:
        return None

    tag = _normalize_local_gemma_tag(" ".join(sections.get("tag", [])))
    return source, _clean_local_gemma_translation(translation, tag)


def _split_local_gemma_one_step_value(value: Any) -> tuple[str, str]:
    text = _normalize_local_gemma_markup(
        _coerce_local_gemma_response_value(value)
    )
    if "||" in text:
        source, translation = text.split("||", 1)
        source_tag = _first_local_gemma_tagged_block(source)
        tag_hint = source_tag[0] if source_tag is not None else None
        return source.strip(), _clean_local_gemma_translation(
            translation,
            tag_hint,
        )

    labeled = _split_local_gemma_labeled_value(text)
    if labeled is not None:
        return labeled

    for delimiter in (r"\s+/\s+", r"\s+\|\s+"):
        parts = re.split(delimiter, text, maxsplit=2)
        if len(parts) == 3:
            tag = _normalize_local_gemma_tag(parts[1])
            if tag:
                return parts[0].strip(), _clean_local_gemma_translation(
                    parts[2],
                    tag,
                )

    tagged_blocks = [
        (tag.upper(), inner.strip())
        for tag, inner in LOCAL_GEMMA_TAG_WRAPPER_PATTERN.findall(text)
        if tag.upper() in LOCAL_GEMMA_RESPONSE_TAGS
    ]
    if len(tagged_blocks) >= 2:
        source = tagged_blocks[0][1]
        translation_tag, translation = tagged_blocks[-1]
        return source, _clean_local_gemma_translation(
            translation,
            translation_tag,
        )

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index in range(1, len(lines) - 1):
        tag = _normalize_local_gemma_tag(lines[index])
        if tag:
            source = "\n".join(lines[:index]).strip()
            translation = "\n".join(lines[index + 1 :]).strip()
            if source and translation:
                return source, _clean_local_gemma_translation(
                    translation,
                    tag,
                )

    return "[OCR FAILED]", _clean_local_gemma_translation(text)


def _unwrap_local_gemma_response_container(parsed: Any) -> Any:
    current = parsed
    for _ in range(4):
        if not isinstance(current, dict):
            break
        if _lookup_indexed_value(current, 1) is not None:
            break
        nested = None
        for key in ("translations", "items", "results", "data", "output"):
            if key in current and isinstance(current[key], (dict, list)):
                nested = current[key]
                break
        if nested is None:
            break
        current = nested
    return current


def _parse_local_gemma_json_candidate(response_text: str) -> Any:
    candidate = _extract_json_candidate(response_text)
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return _unwrap_local_gemma_response_container(parsed)


def _lookup_local_gemma_response_value(
    parsed: Any,
    index: int,
    total_elements: int,
) -> Any:
    value = _lookup_indexed_value(parsed, index)
    if value is not None:
        return value
    if index != 1 or total_elements != 1:
        return None

    if isinstance(parsed, dict):
        normalized_keys = {
            str(key).strip().lower().replace(" ", "_")
            for key in parsed.keys()
        }
        item_keys = {
            "ocr",
            "ocr_text",
            "source",
            "source_text",
            "original",
            "original_text",
            "translation",
            "translated",
            "translated_text",
            "target",
            "target_text",
            "tag",
            "classification",
            "\uc6d0\ubb38",
            "\ubc88\uc5ed",
            "\ud0dc\uadf8",
            "\u539f\u6587",
            "\u7ffb\u8a33",
            "\u30bf\u30b0",
        }
        if normalized_keys & item_keys:
            return parsed

    if (
        isinstance(parsed, list)
        and len(parsed) == 3
        and _normalize_local_gemma_tag(parsed[1])
    ):
        return parsed
    return None


def _parse_local_gemma_one_step_response(
    response_text: Optional[str],
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> List[tuple[str, str]]:
    parsed = _parse_local_gemma_json_candidate(response_text or "")
    if isinstance(parsed, (dict, list)):
        results = []
        for index in range(1, total_elements + 1):
            value = _lookup_local_gemma_response_value(
                parsed,
                index,
                total_elements,
            )
            if value is None:
                value = f"[{provider}: Missing item {index}]"
            results.append(_split_local_gemma_one_step_value(value))
        return results

    if total_elements == 1 and response_text:
        single = _split_local_gemma_one_step_value(response_text)
        if single[1] and not single[1].startswith(f"[{provider}:"):
            return [single]

    raw_lines = _parse_llm_response_unified(
        response_text,
        total_elements,
        provider,
        debug,
    )
    return [_split_local_gemma_one_step_value(line) for line in raw_lines]


def _local_gemma_translation_only_value(value: Any) -> str:
    text = _normalize_local_gemma_markup(
        _coerce_local_gemma_response_value(value)
    )
    if "||" in text:
        _source, translation = text.split("||", 1)
        return _clean_local_gemma_translation(translation)

    sections = _local_gemma_labeled_sections(text)
    translation = "\n".join(sections.get("translation", [])).strip()
    if translation:
        tag = _normalize_local_gemma_tag(" ".join(sections.get("tag", [])))
        return _clean_local_gemma_translation(translation, tag)

    for delimiter in (r"\s+/\s+", r"\s+\|\s+"):
        parts = re.split(delimiter, text, maxsplit=2)
        if len(parts) == 3:
            tag = _normalize_local_gemma_tag(parts[1])
            if tag:
                return _clean_local_gemma_translation(parts[2], tag)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        tag = _normalize_local_gemma_tag(lines[0])
        if tag:
            return _clean_local_gemma_translation("\n".join(lines[1:]), tag)
    return _clean_local_gemma_translation(text)


def _parse_local_gemma_translation_response(
    response_text: Optional[str],
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> List[str]:
    parsed = _parse_local_gemma_json_candidate(response_text or "")
    if isinstance(parsed, (dict, list)):
        results = []
        for index in range(1, total_elements + 1):
            value = _lookup_local_gemma_response_value(
                parsed,
                index,
                total_elements,
            )
            if value is None:
                value = f"[{provider}: Missing item {index}]"
            results.append(_local_gemma_translation_only_value(value))
        return results

    if total_elements == 1 and response_text:
        single = _local_gemma_translation_only_value(response_text)
        if single and not single.startswith(f"[{provider}:"):
            return [single]

    raw_lines = _parse_llm_response_unified(
        response_text,
        total_elements,
        provider,
        debug,
    )
    return [_local_gemma_translation_only_value(line) for line in raw_lines]


def _lookup_indexed_value(parsed: Any, index: int) -> Any:
    if isinstance(parsed, dict):
        keys = (
            str(index),
            f"{index}",
            f"item_{index}",
            f"Item {index}",
            f"ITEM_{index}",
        )
        for key in keys:
            if key in parsed:
                return parsed[key]
    elif isinstance(parsed, list) and 1 <= index <= len(parsed):
        return parsed[index - 1]
    return None


def _parse_json_indexed_response(
    response_text: Optional[str],
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> Optional[List[str]]:
    if not response_text:
        return None

    candidate = _extract_json_candidate(response_text)
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, (dict, list)):
        return None

    result = []
    parsed_count = 0
    for i in range(1, total_elements + 1):
        value = _lookup_indexed_value(parsed, i)
        if value is None:
            result.append(f"[{provider}: Missing item {i}]")
        else:
            parsed_count += 1
            result.append(_coerce_response_value(value))

    log_message(
        f"Parsed {parsed_count} items from JSON response (expected {total_elements})",
        verbose=debug,
    )
    return result


def _format_indexed_json(values: List[str]) -> str:
    return json.dumps(
        {str(i + 1): value for i, value in enumerate(values)},
        ensure_ascii=False,
        indent=2,
    )


def _build_system_prompt_ocr(
    input_language: Optional[str],
    reading_direction: str,
) -> str:
    lang_label = f"{input_language} " if input_language else ""
    direction = (
        "right-to-left"
        if (reading_direction or "rtl").lower() == "rtl"
        else "left-to-right"
    )

    return f"""
## ROLE
You are an expert manga OCR transcriber.

## OBJECTIVE
Your sole purpose is to accurately transcribe the original text from a series of provided images. You must not translate, interpret, or add commentary.

## CORE RULES
- **Reading Context:** The image crops are presented in a {direction} reading order. Do not reorder them.
- **Transcription Policy:** Preserve all original punctuation, ellipses, and casing. Collapse multi-line text into a single line, separated by a single space.
- **Ignore Policy:** Ignore all non-text visual elements (borders, tails, watermarks, etc.).
- **Language Focus:** Transcribe only the original {lang_label}text.
- **Ruby/Furigana Policy:** When small phonetic characters are attached to larger base text, preserve the relationship inline as `BASE{{RUBY}}`, for example `三嶋{{みしま}}さん！`. Keep both in the same JSON item and never emit ruby as a separate item. Do not invent a reading; if the ruby is unreadable, transcribe only the base text.
- **Quotes:** Do not wrap the transcribed text in quotation marks unless they are explicitly present in the image.
- **Edge Cases:**
  - If an image contains standalone periods/ellipses, you must return it exactly as it appears.
  - If text is indecipherable, you must return the exact token: `[OCR FAILED]`.

## OUTPUT SCHEMA
- Return a single JSON object only. Do not use Markdown fences, numbered lists, section headers, explanations, or extra text.
- The JSON keys must be numeric strings matching the input image order: "1", "2", "3", ...
- The JSON value for each key must be the transcribed {lang_label}text for that same input image only.
- Example: {{"1":"<text for image 1>","2":"<text for image 2>"}}
"""  # noqa


def _format_previous_context_prompt_note(
    previous_context_image_count: int,
    previous_context_text_count: int,
    image_order: str,
) -> str:
    has_images = previous_context_image_count > 0
    has_text = previous_context_text_count > 0

    if has_images and has_text:
        return (
            f" {previous_context_image_count} previous source page image(s) are "
            "attached as visual reference, and transcribed text from "
            f"{previous_context_text_count} previous source page(s) is provided "
            "in `## PREVIOUS PAGE TRANSCRIPTS`. Image order: "
            f"{image_order}. Use this previous-page context only as narrative "
            "reference; do not transcribe, translate, or renumber previous-page "
            "material."
        )

    if has_images:
        return (
            f" {previous_context_image_count} previous source page image(s) "
            f"are attached as reference. Image order: {image_order}."
        )

    if has_text:
        return (
            f" Transcribed text from {previous_context_text_count} previous "
            "source page(s) is provided in `## PREVIOUS PAGE TRANSCRIPTS` "
            "as narrative reference only — do not translate or renumber it."
        )

    return ""


def _build_system_prompt_translation(
    output_language: str,
    mode: str,
    reading_direction: str,
    full_page_context: bool = False,
    previous_context_image_count: int = 0,
    previous_context_text_count: int = 0,
) -> str:
    direction = (
        "right-to-left"
        if (reading_direction or "rtl").lower() == "rtl"
        else "left-to-right"
    )
    input_type = "transcriptions" if mode == "two-step" else "image crops"

    cohesion_visual = (
        " Refer to the full-page image only for ambiguous context; never copy, "
        "renumber, or translate text from the full-page context itself."
        if full_page_context
        else ""
    )

    if mode == "two-step":
        edge_cases = """- **Edge Cases:**
  - If an input line contains standalone periods/ellipses, you must return it exactly as it appears.
  - If an input line is the exact token `[OCR FAILED]`, you must output it unchanged."""
    else:
        edge_cases = """- **Edge Cases:**
  - If an image contains standalone periods/ellipses, you must return it exactly as it appears.
  - If text is indecipherable, you must return the exact token: `[OCR FAILED]`."""

    previous_context_rule = ""
    if previous_context_image_count > 0 and previous_context_text_count > 0:
        previous_context_rule = """
- **Previous Page Context:** Earlier source-page images and transcripts are visual/narrative context only; do not transcribe, translate, number, or count them. Use them to maintain consistency:
  - **Proper Nouns:** Keep character names, place names, organizations, technique/skill/title names, honorifics, and stylized terms consistent with established usage.
  - **Character Voice:** Preserve each character's established voice, register, and pronoun choices.
  - **Referents:** Disambiguate callbacks, ongoing beats, or unclear references using prior visuals and dialogue."""  # noqa
    elif previous_context_image_count > 0:
        previous_context_rule = """
- **Previous Page Reference:** Earlier source pages are visual/narrative context only — do not transcribe, translate, number, or count them. Use them to maintain consistency:
  - **Proper Nouns:** Keep character names, place names, organizations, technique/skill/title names, honorifics, and stylized terms spelled exactly as they appeared previously.
  - **Character Voice:** Preserve each character's established voice, register, and pronoun choices.
  - **Referents:** Disambiguate callbacks, ongoing beats, or unclear references using prior context."""  # noqa
    elif previous_context_text_count > 0:
        previous_context_rule = """
- **Previous Page Transcripts:** Earlier source-page transcribed text is provided as narrative context only — do not translate, number, or count it. Use it to maintain consistency:
  - **Proper Nouns:** Keep character names, place names, organizations, technique/skill/title names, honorifics, and stylized terms aligned with their established usage.
  - **Character Voice:** Preserve each character's established voice, register, and pronoun choices.
  - **Referents:** Disambiguate callbacks, ongoing beats, or unclear references using prior dialogue."""  # noqa

    if mode == "two-step":
        ruby_rule = """
- **Ruby/Furigana:** OCR annotations use `BASE{RUBY}`. Unless an explicit user-provided character-name mapping specifies another target spelling, treat RUBY as the author's authoritative intended reading. For person names and other proper nouns, transliterate RUBY into the target language; never guess a competing reading from BASE. Remove the annotation and braces from the translation. For ordinary vocabulary or wordplay, use both BASE and RUBY to translate the intended expression naturally."""  # noqa
    else:
        ruby_rule = """
- **Ruby/Furigana:** When small phonetic ruby is attached to larger base text in an image, preserve it in the transcription as `BASE{RUBY}`. Unless an explicit user-provided character-name mapping specifies another target spelling, treat RUBY as the author's authoritative intended reading. For person names and other proper nouns, transliterate RUBY into the target language; never guess a competing reading from BASE. Remove the annotation and braces from the translation, and never output ruby as a separate item."""  # noqa

    core_rules = f"""
## CORE RULES
- **Reading Context:** The {input_type} are presented in a {direction} reading order. Do not reorder them.
- **Item Boundary:** Each output JSON value must translate only the corresponding input item with the same numeric key. Never split, merge, move, continue, borrow, or redistribute text across keys. If one item is a continuation or an incomplete sentence, translate only that item's own text.
- **Cohesion:** Treat the input items as a continuous narrative only for tone, context, and word choice. Do not change item boundaries.{cohesion_visual}
- **Fidelity:** Focus on intent; translate functionally rather than literally.
- **Conciseness:** Keep translations idiomatic and concise.
- **Punctuation:** Replace ellipses (e.g., "…") with consecutive periods (e.g., "...").
- **Sentence Endings:** Do not end translated dialogue, monologue, narration, or sound effects with a single period (`.`, `。`, or `．`). Preserve question marks, exclamation marks, ellipses, decimal points, and punctuation-only items. This applies only to translations; preserve the original punctuation in OCR transcriptions.
- **Quotes:** Do not wrap the translated text in quotation marks unless they are explicitly present in the source text.
{ruby_rule}
- **Text Types:**
  - **Spoken Dialogue/Internal Monologue:** Translate naturally, matching the character's personality.
  - **Narration:** Translate neutrally without special styling.
  - **Audible SFX:** Translate physical sounds (Giongo) as standard onomatopoeia.
  - **Mimetic FX:** Translate atmospheric text (Gitaigo) or silent actions as descriptive verbs or adjectives. Do not add a period at the end.
{edge_cases}{previous_context_rule}
"""  # noqa

    shared_components = f"""
## ROLE
You are a professional manga localization translator and editor.

## OBJECTIVE
Your goal is to produce natural-sounding, high-quality translations in {output_language} that are faithful to the original source's meaning, tone.


{core_rules}
"""  # noqa

    if mode == "one-step":
        output_schema = f"""
## OUTPUT SCHEMA
- Return a single valid JSON object only. Do not use Markdown fences, numbered lists, section headers, explanations, or extra text.
- The JSON keys must be numeric strings matching the NUMBERED INPUT ITEM crop order: "1", "2", "3", ...
- Context/reference images are not input items. Never create JSON keys for full-page or previous-page context images.
- Every NUMBERED INPUT ITEM crop must have exactly one JSON key, and every value must belong only to that same crop.
- For each value, provide both transcription and translation in this exact string format:
  `<transcribed text> || <translated {output_language} text>`
- Example: {{"1":"<transcribed image 1> || <translated image 1>","2":"<transcribed image 2> || <translated image 2>"}}
"""
    elif mode == "two-step":
        output_schema = f"""
## OUTPUT SCHEMA
- Return a single valid JSON object only. Do not use Markdown fences, numbered lists, section headers, explanations, or extra text.
- The JSON keys must be numeric strings matching the input text order: "1", "2", "3", ...
- Every input text must have exactly one JSON key, and every value must translate only that same input text.
- Example: {{"1":"<translated text 1>","2":"<translated text 2>"}}
"""  # noqa
    else:
        raise ValueError(
            f"Invalid mode '{mode}' specified for translation system prompt."
        )

    return shared_components + output_schema


def _is_reasoning_model_google(model_name: str) -> bool:
    """Check if a Google model is reasoning-capable."""
    return is_google_reasoning_model(model_name)


def _is_reasoning_model_openai(model_name: str) -> bool:
    """Check if an OpenAI model is reasoning-capable."""
    return _is_openai_reasoning_meta(model_name)


def _is_reasoning_model_anthropic(model_name: str) -> bool:
    """Check if an Anthropic model is reasoning-capable."""
    return is_anthropic_reasoning_model(model_name)


def _add_media_resolution_to_part(
    part: Dict[str, Any],
    media_resolution_ui: str,
) -> Dict[str, Any]:
    """
    Add media_resolution to an inline_data part.

    Args:
        part: Part dictionary with inline_data
        media_resolution_ui: UI format media resolution ("auto"/"high"/"medium"/"low")

    Returns:
        Part dictionary with media_resolution added
    """
    if "inline_data" not in part:
        return part

    media_resolution_mapping = {
        "auto": "MEDIA_RESOLUTION_UNSPECIFIED",
        "high": "MEDIA_RESOLUTION_HIGH",
        "medium": "MEDIA_RESOLUTION_MEDIUM",
        "low": "MEDIA_RESOLUTION_LOW",
    }
    backend_media_resolution = media_resolution_mapping.get(
        media_resolution_ui.lower(), "MEDIA_RESOLUTION_UNSPECIFIED"
    )

    result = part.copy()
    result["media_resolution"] = {"level": backend_media_resolution}
    return result



def _make_inline_image_part(
    image_b64: str,
    mime_type: str,
    media_resolution_ui: str,
    supports_per_part_res: bool,
) -> Dict[str, Any]:
    """Create an inline image part with optional provider media resolution."""
    part = {
        "inline_data": {
            "mime_type": mime_type or "image/jpeg",
            "data": image_b64 or "",
        }
    }
    if supports_per_part_res:
        part = _add_media_resolution_to_part(part, media_resolution_ui)
    return part


def _load_item_label_font(font_size: int) -> ImageFont.ImageFont:
    for font_path in ITEM_LABEL_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (OSError, ValueError):
            continue
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def _embed_input_item_label(
    image_b64: str,
    mime_type: str,
    item_index: int,
    total_items: int,
) -> tuple[str, str]:
    """Add a visible item-number header without covering source artwork."""
    try:
        image_data = base64.b64decode(image_b64)
        source = ImageOps.exif_transpose(Image.open(BytesIO(image_data))).convert("RGB")
    except Exception as exc:
        log_message(
            f"Could not embed INPUT ITEM {item_index} label: {exc}",
            always_print=True,
        )
        return image_b64, mime_type or "image/jpeg"

    short_side = max(1, min(source.width, source.height))
    header_height = max(32, min(64, int(round(short_side * 0.25))))
    label = f"ITEM {item_index:02d}/{total_items:02d}"
    font_size = max(14, int(round(header_height * 0.58)))
    font = _load_item_label_font(font_size)

    canvas = Image.new("RGB", (source.width, source.height + header_height), (10, 10, 10))
    canvas.paste(source, (0, header_height))
    draw = ImageDraw.Draw(canvas)

    max_text_width = max(1, source.width - 12)
    while font_size > 10:
        text_box = draw.textbbox((0, 0), label, font=font)
        if text_box[2] - text_box[0] <= max_text_width:
            break
        font_size -= 1
        font = _load_item_label_font(font_size)

    text_box = draw.textbbox((0, 0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_x = max(4, (source.width - text_width) // 2)
    text_y = max(1, (header_height - text_height) // 2 - text_box[1])
    draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)
    draw.line((0, header_height - 2, source.width, header_height - 2), fill=(0, 220, 255), width=2)

    output = BytesIO()
    output_mime = (mime_type or "image/jpeg").lower()
    if output_mime == "image/png":
        canvas.save(output, format="PNG", optimize=False, compress_level=2)
    else:
        output_mime = "image/jpeg"
        canvas.save(output, format="JPEG", quality=95, subsampling=0)
    return base64.b64encode(output.getvalue()).decode("ascii"), output_mime


def _build_labeled_one_step_parts(
    images_b64: List[str],
    mime_types: List[str],
    full_image_b64: str,
    full_image_mime_type: str,
    previous_context_images: List[Dict[str, str]],
    config: TranslationConfig,
    supports_per_part_res: bool,
    include_context_images: bool = True,
) -> List[Dict[str, Any]]:
    """Build one-step vision parts with explicit non-counted context labels.

    The provider receives one full-page context image plus N text crops.  Without
    labels, multimodal models may count the context image as item N+1 and return
    an extra JSON key.  These text labels make the contract explicit: only images
    labelled INPUT ITEM are OCR/translation targets.
    """
    total_elements = len(images_b64)
    parts: List[Dict[str, Any]] = []

    if include_context_images and config.send_full_page_context and full_image_b64:
        parts.append(
            {
                "text": (
                    "REFERENCE_CONTEXT_CURRENT_PAGE_IMAGE. "
                    "This full-page manga page is VISUAL/NARRATIVE CONTEXT ONLY. "
                    "Do NOT OCR it. Do NOT translate text from it. "
                    "Do NOT count it as an input item. Do NOT create any JSON key for it."
                )
            }
        )
        parts.append(
            _make_inline_image_part(
                full_image_b64,
                full_image_mime_type,
                config.media_resolution_context,
                supports_per_part_res,
            )
        )

    if include_context_images:
        for idx, image in enumerate(previous_context_images, start=1):
            parts.append(
                {
                    "text": (
                        f"REFERENCE_PREVIOUS_PAGE_CONTEXT_IMAGE {idx}. "
                        "This previous page is context only. "
                        "Do NOT OCR it, translate it, count it, or create any JSON key for it."
                    )
                }
            )
            parts.append(
                _make_inline_image_part(
                    image.get("data", ""),
                    image.get("mime_type", "image/jpeg"),
                    config.media_resolution_context,
                    supports_per_part_res,
                )
            )

    parts.append(
        {
            "text": (
                f"BEGIN NUMBERED INPUT CROPS. There are exactly {total_elements} "
                "input item images below. Return exactly these JSON keys: "
                + ", ".join(f'"{i}"' for i in range(1, total_elements + 1))
                + "."
            )
        }
    )

    for i, img_b64 in enumerate(images_b64, start=1):
        mime_type = mime_types[i - 1] if (i - 1) < len(mime_types) else "image/jpeg"
        labeled_image_b64, labeled_mime_type = _embed_input_item_label(
            img_b64,
            mime_type,
            i,
            total_elements,
        )
        parts.append(
            {
                "text": (
                    f"INPUT ITEM {i} OF {total_elements}. "
                    f"This crop is the ONLY source for JSON key \"{i}\". "
                    f"The black header inside the image reads ITEM {i:02d}/{total_elements:02d}; "
                    "it is an item identifier, not manga text. Never transcribe or translate "
                    "the header. Transcribe and translate only the artwork below the cyan line "
                    "for this key."
                )
            }
        )
        parts.append(
            _make_inline_image_part(
                labeled_image_b64,
                labeled_mime_type,
                config.media_resolution_bubbles,
                supports_per_part_res,
            )
        )

    parts.append(
        {
            "text": (
                f"END NUMBERED INPUT CROPS. Output must contain exactly {total_elements} "
                "numeric JSON keys and no extra keys for context images."
            )
        }
    )
    return parts


def _get_out_of_range_json_keys(
    response_text: Optional[str],
    total_elements: int,
) -> List[int]:
    """Return numeric JSON keys outside 1..total_elements, if response is JSON."""
    if not response_text:
        return []
    candidate = _extract_json_candidate(response_text)
    if not candidate:
        return []
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return []

    keys: List[int] = []
    if isinstance(parsed, dict):
        for key in parsed.keys():
            try:
                keys.append(int(str(key).strip()))
            except (TypeError, ValueError):
                continue
    elif isinstance(parsed, list):
        keys = list(range(1, len(parsed) + 1))

    return sorted(k for k in keys if k < 1 or k > total_elements)



def _build_generation_config(
    provider: str,
    model_name: str,
    config: TranslationConfig,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Build provider-specific generation config dictionary.

    Centralizes logic for:
    - Base parameters (temperature, top_p, top_k)
    - Provider-specific parameter names and constraints
    - Reasoning model detection and token limits
    - Special features (thinking, reasoning_effort, etc.)

    Args:
        provider: Provider name (Google, OpenAI, Anthropic, xAI, OpenRouter, OpenAI-Compatible)
        model_name: Model identifier
        config: TranslationConfig with all settings
        debug: Whether to log debug messages

    Returns:
        Dictionary with generation config parameters for the specific provider
    """
    temperature = config.temperature
    top_p = config.top_p
    top_k = config.top_k

    def normalize_image_detail() -> str:
        image_detail = (config.image_detail or "auto").lower()
        if image_detail not in ("auto", "original", "high", "low"):
            image_detail = "auto"
        if image_detail == "original" and not supports_openai_original_image_detail(
            model_name
        ):
            image_detail = "high"
        return image_detail

    if config.max_tokens is not None:
        max_tokens_value = config.max_tokens
    else:
        is_reasoning = False
        if provider == "Google":
            is_reasoning = _is_reasoning_model_google(model_name)
        elif provider == "OpenAI":
            is_reasoning = _is_reasoning_model_openai(model_name)
        elif provider == "Anthropic":
            is_reasoning = _is_reasoning_model_anthropic(model_name)
        elif provider == "xAI":
            is_reasoning = is_xai_reasoning_model(model_name)
        elif provider == "OpenRouter":
            is_reasoning = openrouter_is_reasoning_model(model_name, debug)
        elif provider == "OpenAI-Compatible":
            is_reasoning = is_openai_compatible_reasoning_model(model_name)
        elif provider == "Codex":
            is_reasoning = True
        elif provider == "DeepSeek":
            is_reasoning = is_deepseek_reasoning_model(model_name)
        elif provider == "Z.ai":
            is_reasoning = is_zai_reasoning_model(model_name)
        elif provider == "Moonshot AI":
            lm = (model_name or "").lower()
            is_reasoning = "kimi-k2." in lm
        max_tokens_value = 16384 if is_reasoning else 4096

    max_tokens_cap = get_max_tokens_cap(provider, model_name)
    if max_tokens_cap is not None and max_tokens_value > max_tokens_cap:
        max_tokens_value = max_tokens_cap

    if provider == "Google":
        is_gemini_3 = is_gemini_3_model(model_name)
        is_gemma = is_gemma_model(model_name)
        generation_config = {
            "temperature": temperature,
            "topP": top_p,
            "topK": top_k,
            "maxOutputTokens": max_tokens_value,
        }
        if not is_gemini_3:
            media_resolution_mapping = {
                "auto": "MEDIA_RESOLUTION_UNSPECIFIED",
                "high": "MEDIA_RESOLUTION_HIGH",
                "medium": "MEDIA_RESOLUTION_MEDIUM",
                "low": "MEDIA_RESOLUTION_LOW",
            }
            backend_media_resolution = media_resolution_mapping.get(
                config.media_resolution.lower(), "MEDIA_RESOLUTION_UNSPECIFIED"
            )
            generation_config["media_resolution"] = backend_media_resolution
        if is_gemini_3 or is_gemma:
            reasoning_effort = config.reasoning_effort or "high"
            generation_config["thinkingConfig"] = {"thinkingLevel": reasoning_effort}
            log_message(
                f"Using reasoning effort '{reasoning_effort}' for {model_name}",
                verbose=debug,
            )
        elif _is_reasoning_model_google(model_name) and not is_gemini_3:
            reasoning_effort = config.reasoning_effort or "auto"
            is_flash = is_gemini_25_flash_model(model_name)
            is_pro = is_gemini_25_pro_model(model_name)
            if reasoning_effort == "none":
                if is_flash:
                    generation_config["thinkingConfig"] = {"thinkingBudget": 0}
                    log_message(f"Disabled reasoning for {model_name}", verbose=debug)
                elif is_pro:
                    generation_config["thinkingConfig"] = {"thinkingBudget": 128}
                    log_message(
                        f"Using 'none' reasoning effort (thinkingBudget: 128) for {model_name}",
                        verbose=debug,
                    )
                else:
                    log_message(
                        f"Warning: 'none' not supported for {model_name}, using 'auto'",
                        verbose=debug,
                    )
            elif reasoning_effort == "auto":
                log_message(
                    f"Using auto reasoning allocation for {model_name}", verbose=debug
                )
            else:
                thinking_budget = calculate_reasoning_budget(
                    max_tokens_value, reasoning_effort
                )
                generation_config["thinkingConfig"] = {
                    "thinkingBudget": thinking_budget
                }
                log_message(
                    f"Using reasoning effort '{reasoning_effort}' (budget: {thinking_budget} tokens) for {model_name}",
                    verbose=debug,
                )
        return generation_config

    elif provider == "OpenAI":
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens_value,
        }  # top_k not supported by OpenAI
        generation_config["image_detail"] = normalize_image_detail()
        if config.reasoning_effort:
            gen = get_gpt5_generation(model_name)
            is_chat = is_gpt5_chat_variant(model_name)
            xhigh_capable = gen in ("5.2", "5.3", "5.4", "5.5")
            effort = config.reasoning_effort
            if effort == "xhigh" and not xhigh_capable:
                effort = "high"
            none_capable = gen is not None and gen != "5"
            if not is_chat and (none_capable or effort != "none"):
                generation_config["reasoning_effort"] = effort
        if is_gpt5_series(model_name) and not is_gpt5_chat_variant(model_name):
            generation_config["verbosity"] = config.verbosity or "low"
        return generation_config

    elif provider == "Anthropic":
        is_reasoning = _is_reasoning_model_anthropic(model_name)
        is_opus_45 = is_opus_45_model(model_name)
        is_46 = is_46_model(model_name)
        is_47 = is_opus_47_model(model_name)
        clamped_temp = min(temperature, 1.0)  # Anthropic caps at 1.0
        generation_config = {
            "temperature": clamped_temp,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }
        if is_reasoning:
            reasoning_effort = config.reasoning_effort or "none"
            generation_config["reasoning_effort"] = reasoning_effort
            # Claude 4.6/4.7 models use adaptive thinking; older models use budget-based
            if is_46 or is_47:
                if reasoning_effort == "auto":
                    generation_config["thinking_type"] = "adaptive"
            else:
                if reasoning_effort != "none":
                    generation_config["thinking_type"] = "enabled"
        if (is_opus_45 or is_46 or is_47) and config.effort:
            generation_config["effort"] = config.effort
            generation_config["is_46_model"] = is_46 or is_47
            generation_config["is_47_model"] = is_47
        return generation_config

    elif provider == "xAI":
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens_value,
            "media_resolution": config.media_resolution,
        }
        if config.reasoning_effort:
            generation_config["reasoning_effort"] = config.reasoning_effort
        return generation_config

    elif provider == "DeepSeek":
        is_reasoning = is_deepseek_reasoning_model(model_name)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens_value,
        }
        if is_reasoning:
            reasoning_effort = config.reasoning_effort or "high"
            thinking_type = "enabled" if reasoning_effort != "none" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
            if thinking_type == "enabled":
                generation_config["reasoning_effort"] = reasoning_effort
        return generation_config

    elif provider == "Z.ai":
        is_reasoning = is_zai_reasoning_model(model_name)
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens_value,
        }
        if is_reasoning:
            # Z.ai uses thinking parameter with {"type": "enabled"} or {"type": "disabled"}
            # Map reasoning_effort: "auto" -> enabled, "none" -> disabled
            reasoning_effort = config.reasoning_effort or "auto"
            thinking_type = "enabled" if reasoning_effort != "none" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
        return generation_config

    elif provider == "Moonshot AI":
        lm = (model_name or "").lower()
        is_reasoning = "kimi-k2." in lm

        generation_config = {
            "temperature": min(temperature, 1.0),  # Moonshot caps at 1.0
            "top_p": top_p,
            "max_tokens": max_tokens_value,
        }

        if is_reasoning:
            # Map reasoning_effort: "high" -> enabled, "none" -> disabled
            reasoning_effort = config.reasoning_effort or "high"
            thinking_type = "enabled" if reasoning_effort == "high" else "disabled"
            generation_config["thinking"] = {"type": thinking_type}
        return generation_config

    elif provider == "OpenRouter":
        model_lower = (model_name or "").lower()
        is_openai_model = is_openai_model_family(model_name)
        is_anthropic_model = "anthropic/" in model_lower or model_lower.startswith(
            "claude-"
        )
        is_grok_model = "grok-4" in model_lower
        is_gemini_3 = is_gemini_3_model(model_name)

        generation_config = {
            "temperature": temperature,
            "top_p": top_p if not is_anthropic_model else None,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }
        if is_openai_model:
            generation_config["image_detail"] = normalize_image_detail()

        is_openai_reasoning = is_openai_model and (
            "gpt-5" in model_lower
            or "o1" in model_lower
            or "o3" in model_lower
            or "o4-mini" in model_lower
        )
        is_gpt5_model = is_openai_model and is_gpt5_series(model_name)
        is_gpt5_1 = is_openai_model and "gpt-5.1" in model_lower
        is_gpt5 = is_openai_model and "gpt-5" in model_lower and not is_gpt5_1
        is_anthropic_reasoning = is_anthropic_reasoning_model(model_name)
        # OpenRouter Grok metadata omit explicit reasoning tags in name
        is_grok_reasoning = is_grok_model and "non-reasoning" not in model_lower

        is_opus_45 = is_opus_45_model(model_name)
        is_46 = is_46_model(model_name)
        is_47 = is_opus_47_model(model_name)
        generation_config["_metadata"] = {
            "is_openai_model": is_openai_model,
            "is_anthropic_model": is_anthropic_model,
            "is_grok_model": is_grok_model,
            "is_gemini_3": is_gemini_3,
            "is_google_model": "google/" in model_lower or "gemini" in model_lower,
            "is_openai_reasoning": is_openai_reasoning,
            "is_anthropic_reasoning": is_anthropic_reasoning,
            "is_grok_reasoning": is_grok_reasoning,
            "is_gpt5_1": is_gpt5_1,
            "is_gpt5": is_gpt5,
            "is_gpt5_model": is_gpt5_model,
            "is_opus_45": is_opus_45,
            "is_46_model": is_46 or is_47,
            "is_47_model": is_47,
        }

        if is_openai_reasoning or is_anthropic_reasoning or is_grok_reasoning:
            if is_anthropic_reasoning:
                reasoning_effort = config.reasoning_effort or "none"
                generation_config["reasoning_effort"] = reasoning_effort
            elif is_gpt5_1:
                generation_config["reasoning_effort"] = config.reasoning_effort
            elif config.reasoning_effort and config.reasoning_effort != "none":
                generation_config["reasoning_effort"] = config.reasoning_effort
        elif "gemini" in model_lower or "google/" in model_lower:
            if config.reasoning_effort:
                generation_config["reasoning_effort"] = config.reasoning_effort

        # Opus 4.5/4.6/4.7, Sonnet 4.6 effort parameter
        if (is_opus_45 or is_46 or is_47) and config.effort:
            generation_config["effort"] = config.effort

        # GPT-5 series verbosity
        if is_gpt5_model and not is_gpt5_chat_variant(model_name):
            generation_config["verbosity"] = config.verbosity or "low"

        return generation_config

    elif provider == "OpenAI-Compatible":
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens_value,
        }

    elif provider == "Codex":
        return {"reasoning_effort": config.reasoning_effort}

    else:
        raise TranslationError(f"Unknown provider for generation config: {provider}")


def _call_llm_endpoint(
    config: TranslationConfig,
    parts: List[Dict[str, Any]],
    prompt_text: str,
    debug: bool = False,
    system_prompt: Optional[str] = None,
    usage_output: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    """Internal helper to dispatch API calls based on provider.

    ``usage_output`` is an optional running total; providers that report token
    counts add this call's usage to it. Providers that do not report usage
    simply leave it untouched.
    """
    provider = config.provider
    model_name = config.model_name
    api_parts = parts + [{"text": prompt_text}]
    call_usage: Dict[str, int] = {}

    try:
        if provider == "Google":
            api_key = config.google_api_key
            if not api_key:
                raise TranslationError("Google API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_gemini_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
                usage_output=call_usage,
            )
        elif provider == "OpenAI":
            api_key = config.openai_api_key
            if not api_key:
                raise TranslationError("OpenAI API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_openai_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "Anthropic":
            api_key = config.anthropic_api_key
            if not api_key:
                raise TranslationError("Anthropic API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_anthropic_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "xAI":
            api_key = config.xai_api_key
            if not api_key:
                raise TranslationError("xAI API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_xai_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "DeepSeek":
            api_key = config.deepseek_api_key
            if not api_key:
                raise TranslationError("DeepSeek API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_deepseek_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                usage_output=call_usage,
            )
        elif provider == "Z.ai":
            api_key = config.zai_api_key
            if not api_key:
                raise TranslationError("Z.ai API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_zai_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "Moonshot AI":
            api_key = config.moonshot_api_key
            if not api_key:
                raise TranslationError("Moonshot API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_moonshot_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "OpenRouter":
            api_key = config.openrouter_api_key
            if not api_key:
                raise TranslationError("OpenRouter API key is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_openrouter_endpoint(
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
                enable_web_search=config.enable_web_search,
            )
        elif provider == "OpenAI-Compatible":
            base_url = config.openai_compatible_url
            api_key = config.openai_compatible_api_key  # Optional
            if not base_url:
                raise TranslationError("OpenAI-Compatible URL is missing.")
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_openai_compatible_endpoint(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                parts=api_parts,
                generation_config=generation_config,
                system_prompt=system_prompt,
                debug=debug,
            )
        elif provider == "Codex":
            generation_config = _build_generation_config(
                provider, model_name, config, debug
            )
            return call_codex_endpoint(
                model_name=model_name,
                parts=api_parts,
                system_prompt=system_prompt,
                debug=debug,
                reasoning_effort=generation_config.get("reasoning_effort"),
                usage_output=call_usage,
            )
        else:
            raise TranslationError(
                f"Unknown translation provider specified: {provider}"
            )

    except (ValueError, RuntimeError):
        raise
    finally:
        # Every provider branch returns directly, so fold the counters in here
        # to cover all of them (and partial usage on a failed call).
        accumulate_usage(usage_output, call_usage)


def _parse_llm_response_unified(
    response_text: Optional[str],
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> List[str]:
    """Parse an indexed LLM response.

    JSON is preferred because it preserves item boundaries more reliably than
    free-form numbered text. The numbered-list parser remains as a compatibility
    fallback for older cached/model responses.
    """
    if response_text is None:
        log_message(f"API call failed: {provider} returned None", always_print=True)
        raise TranslationError(f"{provider}: API failed (returned None)")
    elif response_text == "":
        log_message(f"API call returned empty response: {provider}", always_print=True)
        raise TranslationError(f"{provider}: Empty response")

    try:
        log_message(
            f"Parsing {provider} unified response: {len(response_text)} chars",
            verbose=debug,
        )
        log_message(f"Raw response:\n---\n{response_text}\n---", always_print=True)

        json_result = _parse_json_indexed_response(
            response_text,
            total_elements,
            provider,
            debug,
        )
        if json_result is not None:
            return json_result

        # Pattern matches "1: text" or "1. text" or "1 text" etc.
        pattern = re.compile(
            r'^\s*(\d+)\s*[:.]\s*"?\s*(.*?)\s*"?\s*(?=\s*\n\s*\d+\s*[:.]|\s*$)',
            re.MULTILINE | re.DOTALL,
        )

        matches = pattern.findall(response_text)
        result_dict = {}

        for num_str, text in matches:
            try:
                num = int(num_str)
                if 1 <= num <= total_elements:
                    result_dict[num] = text.strip()
            except ValueError:
                continue

        final_list = []
        for i in range(1, total_elements + 1):
            if i in result_dict:
                final_list.append(result_dict[i])
            else:
                final_list.append(f"[{provider}: Missing item {i}]")

        log_message(
            f"Parsed {len(result_dict)} items from unified response (expected {total_elements})",
            verbose=debug,
        )
        return final_list

    except Exception as e:
        log_message(
            f"Failed to parse {provider} unified response: {str(e)}",
            always_print=True,
        )
        return [f"[{provider}: Parse error]"] * total_elements


def _prepare_images_for_ocr(
    images_b64: List[str], verbose: bool = False
) -> List[Optional[Image.Image]]:
    """Prepare base64-encoded images for OCR by decoding and converting to RGB.

    Args:
        images_b64: List of base64-encoded image strings
        verbose: Whether to print verbose logging

    Returns:
        List of PIL Images (or None for decode failures), all in RGB mode
    """
    pil_images = []
    for img_b64 in images_b64:
        try:
            image_data = base64.b64decode(img_b64)
            pil_img = Image.open(BytesIO(image_data))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
        except Exception as e:
            log_message(
                f"Failed to decode image for manga-ocr: {e}",
                always_print=True,
            )
            pil_images.append(None)
    return pil_images


def _format_ocr_results(
    extracted_texts: List[str],
    bubble_metadata: List[Dict[str, Any]],
) -> None:
    """Format and log OCR results.

    Args:
        extracted_texts: List of extracted text strings
        bubble_metadata: List of metadata dicts for text elements
        verbose: Whether to print verbose logging
    """
    log_lines = []

    for i, text in enumerate(extracted_texts):
        metadata = bubble_metadata[i] if i < len(bubble_metadata) else {}
        is_osb = metadata.get("is_outside_text", False)
        prefix = f"{i + 1}"
        type_label = "[OSB]" if is_osb else "[Bubble]"

        log_lines.append(f"{prefix}: {type_label} {text}")

    if log_lines:
        log_message(
            f"Raw OCR output:\n---\n{chr(10).join(log_lines)}\n---",
            always_print=True,
        )


def _check_ocr_failure(texts: List[str], provider: Optional[str] = None) -> bool:
    """Check if all OCR results indicate failure.

    Args:
        texts: List of extracted text strings
        provider: Optional provider name for LLM OCR failure detection

    Returns:
        True if all texts indicate failure, False otherwise
    """
    if not texts:
        return True

    if provider:
        for text in texts:
            if f"[{provider}-OCR:" not in text:
                return False
        return True
    else:
        return all(text == "[OCR FAILED]" for text in texts)


def _format_previous_context_texts(
    previous_context_texts: Optional[List[List[str]]],
) -> str:
    """Format previous-page OCR transcripts as a labeled context block.

    Pages are listed oldest-to-newest (matching the previous-image convention).
    Empty/failed entries are omitted, and the section is suppressed entirely
    when no usable transcripts remain.
    """
    if not previous_context_texts:
        return ""

    page_blocks = []
    for page_index, page_texts in enumerate(previous_context_texts, start=1):
        if not page_texts:
            continue
        lines = []
        for idx, text in enumerate(page_texts, start=1):
            cleaned = (text or "").strip()
            if not cleaned or cleaned == "[OCR FAILED]":
                continue
            lines.append(f"{idx}: {cleaned}")
        if not lines:
            continue
        page_blocks.append(f"### Previous Page {page_index}\n" + "\n".join(lines))

    if not page_blocks:
        return ""

    return (
        "\n## PREVIOUS PAGE TRANSCRIPTS\n"
        "Listed oldest-to-newest. These are reference only — do not translate or renumber.\n"
        + "\n\n".join(page_blocks)
        + "\n"
    )


def _format_special_instructions(config: TranslationConfig) -> str:
    """Format user's special instructions section for prompts.

    Args:
        config: TranslationConfig with special_instructions

    Returns:
        Formatted special instructions string (empty if none)
    """
    if config.special_instructions and config.special_instructions.strip():
        return f"""

## SPECIAL INSTRUCTIONS
{config.special_instructions.strip()}
"""
    return ""


# Special Instructions are shared by both translation modes, so a user's
# one-step output-format line (`<TAG>source</TAG> || <TAG>translation</TAG>`)
# also reaches the two-step translation call, where the source text has already
# been transcribed. Override just the format while leaving the user's tag
# definitions intact.
_TWO_STEP_TAG_FORMAT_OVERRIDE = """

## OUTPUT FORMAT OVERRIDE (TWO-STEP MODE)
- The source text was already transcribed for you. Never repeat or re-transcribe it.
- If the instructions above describe a `source || translation` output format, ignore that part only; all other instructions above (including tag definitions and how to choose a tag) still apply in full.
- Each JSON value must contain the translation only, wrapped in exactly one classification tag: `<TAG>translated text</TAG>`
- Never emit the `||` separator, and never emit more than one tag pair per value.
"""  # noqa


_LEADING_TAG_RE = re.compile(r"^\s*<\s*([A-Za-z0-9]+)\s*>")


def _normalize_two_step_tagged_value(value: Any) -> str:
    """Repair a two-step value that echoed the one-step `source || translation` form.

    The prompt override tells the model to emit `<TAG>translation</TAG>`, but
    models routinely fall back to the format spelled out in Special
    Instructions. Mirrors the one-step split so the tag survives either way.
    """
    text = str(value if value is not None else "").strip()
    if "||" not in text:
        return text

    leading_tag: Optional[str] = None
    leading_match = _LEADING_TAG_RE.match(text)
    if leading_match and leading_match.group(1).upper() in TRANSLATION_TAGS:
        leading_tag = leading_match.group(1).upper()

    translation = text.split("||", 1)[1].strip()
    if not translation:
        return text

    # Already wrapped by the model (the common case) — nothing more to do.
    if _LEADING_TAG_RE.match(translation):
        return translation

    # Only the source half carried the tag; re-apply it so per-tag styling works.
    if leading_tag:
        return f"<{leading_tag}>{translation}</{leading_tag}>"
    return translation


def _build_rosetta_instruction(
    output_language: str,
    special_instructions: Optional[str] = None,
) -> str:
    """Build instruction prompt for YanoljaNEXT Rosetta translation models.

    Follows the Rosetta chat template format: concise instruction with target
    language, context, tone, optional glossary, and output format.
    Special instructions are mapped to the Glossary field.
    """
    instruction = (
        f"Translate the user's text to {output_language}. "
        f"Keep the JSON structure and keys.\n"
        f"Context: Manga dialogue, sound effects, and narration.\n"
        f"Tone: Natural-sounding manga localization"
    )

    if special_instructions and special_instructions.strip():
        glossary_lines = []
        for line in special_instructions.strip().splitlines():
            line = line.strip()
            if line:
                entry = line if line.startswith("- ") else f"- {line}"
                glossary_lines.append(entry)
        if glossary_lines:
            instruction += "\nGlossary:\n" + "\n".join(glossary_lines)

    instruction += (
        "\nOutput format: JSON\n"
        "Provide the final translation immediately without any other text."
    )
    return instruction


def _build_rosetta_source_prompt(extracted_texts: List[str]) -> str:
    """Format OCR texts as JSON for Rosetta models.

    Returns a JSON object with string keys "1", "2", etc.
    """
    data = {str(i + 1): text for i, text in enumerate(extracted_texts)}
    return json.dumps(data, ensure_ascii=False)


def _parse_rosetta_response(
    response_text: Optional[str],
    total_elements: int,
    provider: str,
    debug: bool = False,
) -> List[str]:
    """Parse JSON response from a Rosetta model.

    Falls back to _parse_llm_response_unified if JSON parsing fails.
    """
    if response_text is None:
        raise TranslationError(f"{provider}: API failed (returned None)")
    if response_text == "":
        raise TranslationError(f"{provider}: Empty response")

    log_message(f"Raw response:\n---\n{response_text}\n---", always_print=True)

    # Strip markdown code fences if present
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            result = []
            for i in range(1, total_elements + 1):
                val = parsed.get(str(i))
                if val is not None:
                    result.append(str(val).strip())
                else:
                    result.append(f"[{provider}: Missing item {i}]")
            log_message(
                f"Parsed {len(parsed)} items from Rosetta JSON (expected {total_elements})",
                verbose=debug,
            )
            return result
    except (json.JSONDecodeError, TypeError):
        log_message(
            "Rosetta response is not valid JSON, falling back to numbered-list parser",
            verbose=debug,
        )

    return _parse_llm_response_unified(response_text, total_elements, provider, debug)


def _perform_manga_ocr(
    images_b64: List[str],
    bubble_metadata: List[Dict[str, Any]],
    debug: bool = False,
) -> List[str]:
    """Perform OCR using manga-ocr model.

    Args:
        images_b64: List of base64-encoded images
        bubble_metadata: List of metadata dicts for text elements
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    log_message("Using manga-ocr for text extraction", verbose=debug)

    cache = get_cache()
    cache_key = cache.get_manga_ocr_cache_key(images_b64, total_elements)
    cached_ocr = cache.get_manga_ocr_result(cache_key)
    if cached_ocr is not None:
        if len(cached_ocr) == total_elements:
            log_message("Using cached manga-ocr results", verbose=debug)
            return cached_ocr
        log_message("Discarding manga-ocr cache due to length mismatch", verbose=debug)

    pil_images = _prepare_images_for_ocr(images_b64, verbose=debug)
    extracted_texts = extract_text_with_manga_ocr(pil_images, verbose=debug)

    formatted_texts = []
    for i, text in enumerate(extracted_texts):
        if text == "[OCR FAILED]" or not text:
            formatted_texts.append(text if text else "[OCR FAILED]")
        else:
            formatted_texts.append(text)

    extracted_texts = formatted_texts

    _format_ocr_results(extracted_texts, bubble_metadata)

    if len(extracted_texts) != total_elements:
        msg = (
            f"Warning: extracted_texts length ({len(extracted_texts)}) "
            f"doesn't match total_elements ({total_elements})"
        )
        log_message(msg, always_print=True)
        while len(extracted_texts) < total_elements:
            extracted_texts.append("[OCR FAILED]")
        extracted_texts = extracted_texts[:total_elements]

    if not extracted_texts:
        log_message("manga-ocr returned empty results", verbose=debug)
        failure_results = ["[OCR FAILED]"] * total_elements
        cache.set_manga_ocr_result(cache_key, failure_results, debug)
        return failure_results

    if _check_ocr_failure(extracted_texts):
        log_message("manga-ocr returned only failures", verbose=debug)
        cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
        return extracted_texts

    cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
    return extracted_texts


def _perform_paddle_ocr_vl(
    images_b64: List[str],
    bubble_metadata: List[Dict[str, Any]],
    debug: bool = False,
) -> List[str]:
    """Perform OCR using PaddleOCR-VL-1.5 model.

    Args:
        images_b64: List of base64-encoded images
        bubble_metadata: List of metadata dicts for text elements
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    log_message("Using PaddleOCR-VL-1.5 for text extraction", verbose=debug)

    cache = get_cache()
    cache_key = cache.get_manga_ocr_cache_key(
        images_b64, total_elements, prefix="pocr_"
    )
    cached_ocr = cache.get_manga_ocr_result(cache_key)
    if cached_ocr is not None:
        if len(cached_ocr) == total_elements:
            log_message("Using cached PaddleOCR-VL-1.5 results", verbose=debug)
            return cached_ocr
        log_message(
            "Discarding PaddleOCR-VL-1.5 cache due to length mismatch", verbose=debug
        )

    pil_images = _prepare_images_for_ocr(images_b64, verbose=debug)
    extracted_texts = extract_text_with_paddle_ocr_vl(pil_images, verbose=debug)

    formatted_texts = []
    for i, text in enumerate(extracted_texts):
        if text == "[OCR FAILED]" or not text:
            formatted_texts.append(text if text else "[OCR FAILED]")
        else:
            # Collapse newlines hallucinated by PaddleOCR-VL
            formatted_texts.append(" ".join(text.split()))

    extracted_texts = formatted_texts

    _format_ocr_results(extracted_texts, bubble_metadata)

    if len(extracted_texts) != total_elements:
        msg = (
            f"Warning: extracted_texts length ({len(extracted_texts)}) "
            f"doesn't match total_elements ({total_elements})"
        )
        log_message(msg, always_print=True)
        while len(extracted_texts) < total_elements:
            extracted_texts.append("[OCR FAILED]")
        extracted_texts = extracted_texts[:total_elements]

    if not extracted_texts:
        log_message("PaddleOCR-VL-1.5 returned empty results", verbose=debug)
        failure_results = ["[OCR FAILED]"] * total_elements
        cache.set_manga_ocr_result(cache_key, failure_results, debug)
        return failure_results

    if _check_ocr_failure(extracted_texts):
        log_message("PaddleOCR-VL-1.5 returned only failures", verbose=debug)
        cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
        return extracted_texts

    cache.set_manga_ocr_result(cache_key, extracted_texts, debug)
    return extracted_texts


def _perform_llm_ocr(
    config: TranslationConfig,
    images_b64: List[str],
    mime_types: List[str],
    ocr_prompt: str,
    provider: str,
    input_language: Optional[str],
    reading_direction: str,
    debug: bool = False,
    usage_output: Optional[Dict[str, int]] = None,
) -> List[str]:
    """Perform OCR using vision LLM.

    Args:
        config: TranslationConfig
        images_b64: List of base64-encoded images
        mime_types: List of MIME types for each image
        ocr_prompt: OCR prompt text
        provider: Provider name
        input_language: Input language
        reading_direction: Reading direction
        debug: Whether to print verbose logging

    Returns:
        List of extracted text strings, or early return with failure list
    """
    total_elements = len(images_b64)
    ocr_parts = []
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        supports_per_part_res = (
            provider == "Google" and is_gemini_3_model(config.model_name)
        ) or provider == "xAI"
        if supports_per_part_res:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles
            )
        ocr_parts.append(bubble_part)

    ocr_system = _build_system_prompt_ocr(input_language, reading_direction)
    ocr_response_text = _call_llm_endpoint(
        config,
        ocr_parts,
        ocr_prompt,
        debug,
        system_prompt=ocr_system,
        usage_output=usage_output,
    )
    extracted_texts = _parse_llm_response_unified(
        ocr_response_text,
        total_elements,
        provider + "-OCR",
        debug,
    )

    if extracted_texts is None:
        log_message("OCR API call failed", always_print=True)
        return [f"[{provider}: OCR failed]"] * total_elements

    if _check_ocr_failure(extracted_texts, provider):
        log_message("OCR returned only placeholders", verbose=debug)
        return extracted_texts

    return extracted_texts


def call_translation_api_batch(
    config: TranslationConfig,
    images_b64: List[str],
    full_image_b64: str,
    mime_types: List[str],
    full_image_mime_type: str,
    bubble_metadata: List[Dict[str, Any]],
    previous_context_images: Optional[List[Dict[str, str]]] = None,
    previous_context_texts: Optional[List[List[str]]] = None,
    ocr_texts_output: Optional[List[str]] = None,
    usage_output: Optional[Dict[str, int]] = None,
    debug: bool = False,
) -> List[str]:
    """
    Generates prompts and calls the appropriate LLM API endpoint based on the provider and mode
    specified in the configuration, translating text from speech bubbles and outside-bubble text.

    Supports "one-step" (OCR+Translate) and "two-step" (OCR then Translate) modes.

    Args:
        config (TranslationConfig): Configuration object.
        images_b64 (list): List of base64 encoded images of all text elements, in reading order.
        full_image_b64 (str): Base64 encoded image of the full manga page.
        mime_types (List[str]): List of MIME types for each text element image.
        full_image_mime_type (str): MIME type of the full page image.
        bubble_metadata (List[Dict]): List of metadata dicts with 'is_outside_text' flags for each image.
        previous_context_images: Previous source page images, oldest-to-newest, as reference only.
        previous_context_texts: Per-previous-page OCR transcripts (oldest-to-newest) included as
            narrative reference only. Each inner list contains the OCR strings for that page in reading order.
        ocr_texts_output: Optional mutable list. When provided, OCR transcripts (source-language text) for
            the current page's bubbles are appended in reading order so callers can propagate them as
            previous-page text context for subsequent calls.
        debug (bool): Whether to print debugging information.

    Returns:
        list: List of translated strings (potentially with style markers), one for each input text element.
              Returns placeholder messages on errors or empty responses.

    Raises:
        ValueError: If required config (API key, provider, URL) is missing or invalid.
        RuntimeError: If an API call fails irrecoverably after retries (raised by endpoint functions).
    """
    provider = config.provider
    input_language = config.input_language
    output_language = config.output_language
    reading_direction = config.reading_direction
    translation_mode = config.translation_mode
    previous_context_images = previous_context_images or []
    if not config.send_full_page_context or config.ocr_method != "LLM":
        previous_context_images = []
    previous_context_image_count = len(previous_context_images)
    # Filter out empty pages (no usable OCR) and trim to configured cap so the
    # request order matches the prompt order regardless of upstream history gaps.
    cleaned_previous_texts: List[List[str]] = []
    configured_text_count = int(getattr(config, "previous_context_text_count", 0) or 0)
    if previous_context_texts and configured_text_count > 0:
        for page_texts in previous_context_texts:
            usable = [
                (t or "").strip()
                for t in (page_texts or [])
                if (t or "").strip() and (t or "").strip() != "[OCR FAILED]"
            ]
            if usable:
                cleaned_previous_texts.append(usable)
        cleaned_previous_texts = cleaned_previous_texts[-configured_text_count:]
    previous_context_text_count = len(cleaned_previous_texts)
    previous_text_section = _format_previous_context_texts(cleaned_previous_texts)

    # Include conditional bubble hints
    total_elements = len(images_b64)
    dialogue_indices = [
        i + 1
        for i, meta in enumerate(bubble_metadata)
        if not meta.get("is_outside_text", False)
    ]
    osb_indices = [
        i + 1
        for i, meta in enumerate(bubble_metadata)
        if meta.get("is_outside_text", False)
    ]

    hints = []
    if dialogue_indices:
        dialogue_list_str = ", ".join(map(str, dialogue_indices))
        hints.append(
            f"Items [{dialogue_list_str}] are INSIDE speech bubbles; "
            "use inside-bubble or SFX tag rules when tags are requested."
            if provider == "Codex"
            else f"Items [{dialogue_list_str}] contain spoken dialogue."
        )
    if osb_indices:
        osb_list_str = ", ".join(map(str, osb_indices))
        hints.append(
            f"Items [{osb_list_str}] are OUTSIDE speech bubbles; "
            "use outside-bubble or SFX tag rules when tags are requested. "
            "Never use inside-bubble dialogue tags for these items."
            if provider == "Codex"
            else f"Items [{osb_list_str}] contain sound effects, mimetic effects, narration, or internal monologues."
        )

    context_hints = ""
    if hints:
        context_hints = "\nNote: " + " ".join(hints) + " Translate them accordingly."

    cache = get_cache()
    cache_key = cache.get_translation_cache_key(
        images_b64,
        full_image_b64,
        config,
        previous_context_images=previous_context_images,
        previous_context_texts=cleaned_previous_texts,
    )
    cached_translation, cached_ocr_texts = cache.get_translation(cache_key)
    if cached_translation is not None:
        log_message("  - Using cached translation", verbose=debug)
        if ocr_texts_output is not None and cached_ocr_texts is not None:
            ocr_texts_output.extend(cached_ocr_texts)
        return [strip_translation_terminal_periods(text) for text in cached_translation]

    model_name = config.model_name
    is_local_gemma = (
        provider == "OpenAI-Compatible"
        and _is_local_gemma_model_name(model_name)
    )
    is_gemini_3 = provider == "Google" and is_gemini_3_model(model_name)
    supports_per_part_res = is_gemini_3 or provider == "xAI"

    base_parts = []
    for i, img_b64 in enumerate(images_b64):
        mime_type = mime_types[i] if i < len(mime_types) else "image/jpeg"
        bubble_part = {"inline_data": {"mime_type": mime_type, "data": img_b64}}
        if supports_per_part_res:
            bubble_part = _add_media_resolution_to_part(
                bubble_part, config.media_resolution_bubbles
            )
        base_parts.append(bubble_part)

    if config.send_full_page_context and full_image_b64:
        context_part = {
            "inline_data": {
                "mime_type": full_image_mime_type,
                "data": full_image_b64,
            }
        }
        if supports_per_part_res:
            context_part = _add_media_resolution_to_part(
                context_part, config.media_resolution_context
            )
        base_parts.append(context_part)

    for image in previous_context_images:
        previous_part = {
            "inline_data": {
                "mime_type": image.get("mime_type", "image/jpeg"),
                "data": image.get("data", ""),
            }
        }
        if supports_per_part_res:
            previous_part = _add_media_resolution_to_part(
                previous_part, config.media_resolution_context
            )
        base_parts.append(previous_part)

    try:
        if translation_mode == "two-step":
            special_instructions_section = _format_special_instructions(config)

            ocr_prompt = f"""
## CONTEXT
You have been provided with {total_elements} individual text images from a manga page.

## TASK
Apply your OCR transcription rules to each image provided.{special_instructions_section}
"""  # noqa

            log_message("Starting OCR step", verbose=debug)

            if config.ocr_method == "manga-ocr":
                extracted_texts = _perform_manga_ocr(
                    images_b64,
                    bubble_metadata,
                    debug,
                )
            elif config.ocr_method == "paddleocr-vl":
                extracted_texts = _perform_paddle_ocr_vl(
                    images_b64,
                    bubble_metadata,
                    debug,
                )
            else:
                extracted_texts = _perform_llm_ocr(
                    config,
                    images_b64,
                    mime_types,
                    ocr_prompt,
                    provider,
                    input_language,
                    reading_direction,
                    debug,
                    usage_output=usage_output,
                )

            log_message("Starting translation step", verbose=debug)

            formatted_texts = []
            ocr_failed_indices = set()
            for i, text in enumerate(extracted_texts):
                if f"[{provider}-OCR:" in text or text == "[OCR FAILED]":
                    formatted_texts.append("[OCR FAILED]")
                    ocr_failed_indices.add(i)
                else:
                    formatted_texts.append(text)

            ocr_input_section = (
                "\n## INPUT DATA\n"
                + _format_indexed_json(formatted_texts)
                + "\n"
            )

            full_page_context = (
                "A full-page image is also provided for visual and narrative context."
                if (
                    config.ocr_method not in ("manga-ocr", "paddleocr-vl")
                    and config.send_full_page_context
                    and full_image_b64
                )
                else ""
            )
            previous_page_context = _format_previous_context_prompt_note(
                previous_context_image_count,
                previous_context_text_count,
                (
                    "current full page first (when present), then previous "
                    "source pages oldest-to-newest"
                ),
            )

            special_instructions_section = _format_special_instructions(config)

            translation_prompt = f"""
## CONTEXT
You have been provided with a list of {total_elements} transcribed text segments from a manga page. {full_page_context}{previous_page_context}
{context_hints}
{previous_text_section}
{ocr_input_section}

## TASK
Apply your translation rules to the text in the `## INPUT DATA` section. 
The target language is {output_language}. Use the appropriate translation approach for each text type.{special_instructions_section}
"""  # noqa
            if is_local_gemma:
                translation_prompt += """

## LOCAL GEMMA FORMAT REQUIREMENT
Return one JSON string value per input key. Each value must contain only the
tagged translation, for example: `<BN>translated text</BN>`.
Do not repeat the source text and do not add labels or explanations.
"""
            elif special_instructions_section:
                # Rosetta replaces this prompt entirely further down, so the
                # override only ever reaches the standard two-step path.
                translation_prompt += _TWO_STEP_TAG_FORMAT_OVERRIDE

            translation_parts = []
            if (
                config.ocr_method not in ("manga-ocr", "paddleocr-vl")
                and config.send_full_page_context
                and full_image_b64
            ):
                context_part = {
                    "inline_data": {
                        "mime_type": full_image_mime_type,
                        "data": full_image_b64,
                    }
                }
                if supports_per_part_res:
                    context_part = _add_media_resolution_to_part(
                        context_part, config.media_resolution_context
                    )
                translation_parts.append(context_part)

            for image in previous_context_images:
                previous_part = {
                    "inline_data": {
                        "mime_type": image.get("mime_type", "image/jpeg"),
                        "data": image.get("data", ""),
                    }
                }
                if supports_per_part_res:
                    previous_part = _add_media_resolution_to_part(
                        previous_part, config.media_resolution_context
                    )
                translation_parts.append(previous_part)

            use_rosetta = is_rosetta_model(model_name)
            if use_rosetta:
                log_message(
                    "YanoljaNEXT Rosetta model detected — using Rosetta prompt format",
                    always_print=True,
                )
                translation_system = _build_rosetta_instruction(
                    output_language,
                    config.special_instructions,
                )
                translation_prompt = _build_rosetta_source_prompt(formatted_texts)
                translation_parts = []  # text-only model, no image parts
            else:
                translation_system = _build_system_prompt_translation(
                    output_language,
                    mode="two-step",
                    reading_direction=reading_direction,
                    full_page_context=(
                        config.send_full_page_context and bool(full_image_b64)
                    ),
                    previous_context_image_count=previous_context_image_count,
                    previous_context_text_count=previous_context_text_count,
                )
            translation_response_text = _call_llm_endpoint(
                config,
                translation_parts,
                translation_prompt,
                debug,
                system_prompt=translation_system,
                usage_output=usage_output,
            )
            if use_rosetta:
                final_translations = _parse_rosetta_response(
                    translation_response_text,
                    total_elements,
                    provider + "-Translate",
                    debug,
                )
            elif is_local_gemma:
                final_translations = _parse_local_gemma_translation_response(
                    translation_response_text,
                    total_elements,
                    provider + "-Translate",
                    debug,
                )
            else:
                final_translations = _parse_llm_response_unified(
                    translation_response_text,
                    total_elements,
                    provider + "-Translate",
                    debug,
                )
                if final_translations is not None:
                    repaired = [
                        _normalize_two_step_tagged_value(line)
                        for line in final_translations
                    ]
                    repaired_count = sum(
                        1
                        for before, after in zip(final_translations, repaired)
                        if before != after
                    )
                    if repaired_count:
                        log_message(
                            f"Repaired {repaired_count} two-step value(s) that echoed "
                            "the one-step 'source || translation' format",
                            verbose=debug,
                        )
                    final_translations = repaired

            if final_translations is None:
                log_message("Translation API call failed", always_print=True)
                combined_results = []
                for i in range(total_elements):
                    if i in ocr_failed_indices:
                        combined_results.append(f"[{provider}: OCR Failed]")
                    else:
                        combined_results.append(f"[{provider}: Translation failed]")
                if ocr_texts_output is not None:
                    ocr_texts_output.extend(extracted_texts)
                return combined_results

            combined_results = []
            for i in range(total_elements):
                if i in ocr_failed_indices:
                    if final_translations[i] == "[OCR FAILED]":
                        combined_results.append("[OCR FAILED]")
                    else:
                        log_message(
                            f"Element {i + 1}: LLM ignored OCR failure instruction",
                            verbose=debug,
                        )
                        combined_results.append("[OCR FAILED]")
                else:
                    combined_results.append(strip_translation_terminal_periods(final_translations[i]))

            cache.set_translation(
                cache_key, combined_results, ocr_texts=extracted_texts
            )
            if ocr_texts_output is not None:
                ocr_texts_output.extend(extracted_texts)
            return combined_results

        elif translation_mode == "one-step":
            log_message("Starting one-step translation", verbose=debug)

            full_page_context = (
                "A full-page image labelled REFERENCE_CONTEXT_CURRENT_PAGE_IMAGE "
                "is also provided for visual and narrative context only. It is not "
                "one of the numbered input crops and must not receive a JSON key."
                if config.send_full_page_context and full_image_b64
                else ""
            )
            previous_page_context = _format_previous_context_prompt_note(
                previous_context_image_count,
                previous_context_text_count,
                (
                    "explicitly labelled context/reference images first, then "
                    "numbered INPUT ITEM crops in reading order"
                ),
            )

            special_instructions_section = _format_special_instructions(config)

            one_step_prompt = f"""
## CONTEXT
You have been provided with {total_elements} individual text images from a manga page. {full_page_context}{previous_page_context}
{context_hints}
{previous_text_section}
## TASK
For each image, you must perform two steps:
1.  **Transcribe:** Extract the original text exactly as it appears.
2.  **Translate:** Translate the text you just transcribed into {output_language}.{special_instructions_section}
"""  # noqa
            if is_local_gemma:
                one_step_prompt += """

## LOCAL GEMMA FORMAT REQUIREMENT
Return each JSON value as one string using exactly:
`<TAG>source OCR</TAG> || <TAG>translated text</TAG>`
The `||` delimiter is mandatory. Do not put OCR, tag, and translation in
separate fields or on separate labeled lines.
Source-language characters are allowed only before `||`. Any instruction that
forbids Japanese or Chinese characters applies only to the translated text
after `||`.
"""

            one_step_system = _build_system_prompt_translation(
                output_language,
                mode="one-step",
                reading_direction=reading_direction,
                full_page_context=(
                    config.send_full_page_context and bool(full_image_b64)
                ),
                previous_context_image_count=previous_context_image_count,
                previous_context_text_count=previous_context_text_count,
            )
            one_step_parts = _build_labeled_one_step_parts(
                images_b64,
                mime_types,
                full_image_b64,
                full_image_mime_type,
                previous_context_images,
                config,
                supports_per_part_res,
                include_context_images=True,
            )
            if config.send_full_page_context and full_image_b64:
                log_message(
                    "One-step request includes labelled full-page context; "
                    "only labelled INPUT ITEM crops are countable.",
                    verbose=debug,
                )

            response_text = _call_llm_endpoint(
                config,
                one_step_parts,
                one_step_prompt,
                debug,
                system_prompt=one_step_system,
                usage_output=usage_output,
            )

            extra_keys = _get_out_of_range_json_keys(response_text, total_elements)
            if extra_keys:
                log_message(
                    "One-step response created JSON key(s) for non-input context "
                    f"image(s): {extra_keys}. Retrying once with crop images only "
                    "to prevent misaligned rendering.",
                    always_print=True,
                )
                crop_only_parts = _build_labeled_one_step_parts(
                    images_b64,
                    mime_types,
                    "",
                    full_image_mime_type,
                    [],
                    config,
                    supports_per_part_res,
                    include_context_images=False,
                )
                retry_prompt = (
                    one_step_prompt
                    + "\n\n## CRITICAL RETRY NOTE\n"
                    + f"The previous response contained out-of-range key(s) {extra_keys}. "
                    + f"Return exactly {total_elements} keys, from \"1\" to "
                    + f"\"{total_elements}\". Do not create any other keys."
                )
                response_text = _call_llm_endpoint(
                    config,
                    crop_only_parts,
                    retry_prompt,
                    debug,
                    system_prompt=one_step_system,
                    usage_output=usage_output,
                )

            translations = []
            ocr_texts = []
            if is_local_gemma:
                parsed_values = _parse_local_gemma_one_step_response(
                    response_text,
                    total_elements,
                    provider,
                    debug,
                )
                for ocr_text, translation in parsed_values:
                    ocr_texts.append(ocr_text)
                    translations.append(translation)
            else:
                # Preserve the existing parser for Google/Vertex and all other
                # providers. The compatibility fallback is local Gemma only.
                raw_lines = _parse_llm_response_unified(
                    response_text, total_elements, provider, debug
                )
                for line in raw_lines:
                    if "||" in line:
                        parts = line.split("||", 1)
                        ocr_texts.append(parts[0].strip())
                        translations.append(parts[1].strip())
                    else:
                        # Model violated the format — keep the line as the translation
                        # and mark OCR as failed so it isn't reused as prior context.
                        ocr_texts.append("[OCR FAILED]")
                        translations.append(line)

            translations = [strip_translation_terminal_periods(text) for text in translations]
            cache.set_translation(cache_key, translations, ocr_texts=ocr_texts)
            if ocr_texts_output is not None:
                ocr_texts_output.extend(ocr_texts)
            return translations
        else:
            raise TranslationError(
                f"Unknown translation_mode specified in config: {translation_mode}"
            )
    except TranslationError:
        raise
    except (ValueError, RuntimeError) as e:
        log_message(f"Translation error: {e}", always_print=True)
        return [f"[Translation Error: {e}]"] * total_elements


def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _find_conjoined_neighbor_mask(
    mask_lookup: Dict[tuple[int, int, int, int], Any],
    mask_entries: List[tuple[tuple[int, int, int, int], Any]],
    bbox: tuple[int, int, int, int],
) -> Any:
    neighbor_mask = mask_lookup.get(bbox)
    if neighbor_mask is not None:
        return neighbor_mask

    best_mask = None
    best_iou = 0.0
    for candidate_bbox, candidate_mask in mask_entries:
        if candidate_mask is None:
            continue
        iou = _bbox_iou(candidate_bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best_mask = candidate_mask

    return best_mask if best_iou >= 0.35 else None


def _normalize_bool_mask(mask: Any) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    if mask_arr.ndim != 2:
        return None
    return mask_arr > 0


def _right_neighbor_bbox_whiteout_region(
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Return a conservative rescue region for a side-by-side conjoined neighbor.

    Split masks sometimes assign the neighbor's last vertical glyph to the left
    child, so the normal own-mask protection keeps it visible in the LLM crop.
    The common right-neighbor case is always allowed when sufficiently separated.
    The left-neighbor case is allowed only for low-overlap child boxes, where it
    is unlikely to cut into the current bubble's own vertical text.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    crop_h, crop_w = crop_shape

    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if (
        smaller_child_area <= 0
        or pair_overlap_area / smaller_child_area
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None
    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    dx = nb_cx - cur_cx
    dy = nb_cy - cur_cy

    min_separation = max(
        CONJOINED_WHITEOUT_RIGHT_NEIGHBOR_MIN_SEPARATION_PX,
        min(cur_w, nb_w) * CONJOINED_WHITEOUT_RIGHT_NEIGHBOR_MIN_SEPARATION_RATIO,
    )
    if abs(dx) < min_separation:
        return None
    if dx >= 0 and abs(dx) < abs(dy) * 0.75:
        return None
    if dx < 0 and abs(dx) < abs(dy) * 0.35:
        return None

    ix1 = max(crop_x1, nb_x1)
    iy1 = max(crop_y1, nb_y1)
    ix2 = min(crop_x2, nb_x2)
    iy2 = min(crop_y2, nb_y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None

    overlap_area = (ix2 - ix1) * (iy2 - iy1)
    min_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if (
        min_child_area <= 0
        or overlap_area / min_child_area
        < CONJOINED_WHITEOUT_RIGHT_NEIGHBOR_MIN_OVERLAP_RATIO
    ):
        return None
    if (
        dx < 0
        and overlap_area / min_child_area
        > CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None

    mask = np.zeros((crop_h, crop_w), dtype=bool)
    mask[iy1 - crop_y1 : iy2 - crop_y1, ix1 - crop_x1 : ix2 - crop_x1] = True
    return mask


def _neighbor_side_mask_for_crop(
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Return the half-plane of the crop that confidently belongs to neighbor."""
    crop_x1, crop_y1, _, _ = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    crop_h, crop_w = crop_shape
    if crop_h <= 0 or crop_w <= 0:
        return None

    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    vx = nb_cx - cur_cx
    vy = nb_cy - cur_cy
    length = float((vx * vx + vy * vy) ** 0.5)
    if length < 1.0:
        return None

    ux = vx / length
    uy = vy / length
    mid_x = (cur_cx + nb_cx) / 2.0
    mid_y = (cur_cy + nb_cy) / 2.0
    min_side = max(
        1.0,
        min(
            max(1, cur_x2 - cur_x1),
            max(1, cur_y2 - cur_y1),
            max(1, nb_x2 - nb_x1),
            max(1, nb_y2 - nb_y1),
        ),
    )
    margin = max(
        CONJOINED_WHITEOUT_NEIGHBOR_SIDE_MARGIN_PX,
        min_side * CONJOINED_WHITEOUT_NEIGHBOR_SIDE_MARGIN_RATIO,
    )

    ys, xs = np.ogrid[:crop_h, :crop_w]
    global_x = xs.astype(np.float32) + float(crop_x1)
    global_y = ys.astype(np.float32) + float(crop_y1)
    signed_distance = (global_x - mid_x) * ux + (global_y - mid_y) * uy
    return signed_distance >= margin


def _is_low_overlap_left_neighbor(
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
) -> bool:
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    dx = nb_cx - cur_cx
    dy = nb_cy - cur_cy
    if dx >= 0 or abs(dx) < abs(dy) * 0.35:
        return False

    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    return (
        smaller_child_area > 0
        and pair_overlap_area / smaller_child_area
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    )


def _low_overlap_left_neighbor_side_mask(
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    crop_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    if not _is_low_overlap_left_neighbor(current_bbox, neighbor_bbox):
        return None

    crop_x1, _, _, _ = crop_bbox
    cur_x1, _, cur_x2, _ = current_bbox
    nb_x1, _, nb_x2, _ = neighbor_bbox
    crop_h, crop_w = crop_shape
    cur_cx = (cur_x1 + cur_x2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    mid_x = (cur_cx + nb_cx) / 2.0
    margin = max(2.0, min(max(1, cur_x2 - cur_x1), max(1, nb_x2 - nb_x1)) * 0.03)

    _, xs = np.ogrid[:crop_h, :crop_w]
    global_x = xs.astype(np.float32) + float(crop_x1)
    return np.broadcast_to(global_x <= (mid_x + margin), (crop_h, crop_w)).copy()


def _neighbor_assigned_text_line_mask(
    gray: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    own_protect: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Find full text-line fragments that are clearly owned by the neighbor.

    Split masks can put only the tail character of a vertical line into the
    neighbor mask. Assigning by individual pixels then either erases the tail
    from its real bubble or leaves the same tail visible in the sibling crop.
    This groups nearby dark glyphs into text-line candidates and assigns each
    whole candidate by centroid, so ambiguous edge glyphs follow their line.
    """
    crop_h, crop_w = gray.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return None

    dark = gray < 210
    if not np.any(dark):
        return None

    crop_x1, crop_y1, _, _ = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(
        max(1, cur_x2 - cur_x1) * max(1, cur_y2 - cur_y1),
        max(1, nb_x2 - nb_x1) * max(1, nb_y2 - nb_y1),
    )
    high_overlap = (
        smaller_child_area > 0
        and pair_overlap_area / smaller_child_area
        > CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    )
    min_side = max(
        1.0,
        min(
            max(1, cur_x2 - cur_x1),
            max(1, cur_y2 - cur_y1),
            max(1, nb_x2 - nb_x1),
            max(1, nb_y2 - nb_y1),
        ),
    )
    assign_margin = max(
        CONJOINED_WHITEOUT_TEXT_LINE_ASSIGN_MARGIN_PX,
        min_side * CONJOINED_WHITEOUT_TEXT_LINE_ASSIGN_MARGIN_RATIO,
    )

    result = np.zeros((crop_h, crop_w), dtype=bool)
    dark_u8 = dark.astype(np.uint8)
    kernels = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 25)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)),
    )
    crop_area = max(1, crop_h * crop_w)

    for kernel in kernels:
        grouped = cv2.dilate(dark_u8, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            grouped, connectivity=8
        )
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area <= 0:
                continue
            if area > crop_area * 0.45:
                continue
            if (w > crop_w * 0.75 and h < max(10, crop_h * 0.10)) or (
                h > crop_h * 0.85 and w < max(8, crop_w * 0.04)
            ):
                continue

            component_region = labels == label
            component_dark = dark & component_region
            dark_count = int(np.count_nonzero(component_dark))
            if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
                continue
            if dark_count > crop_area * CONJOINED_WHITEOUT_TEXT_LINE_MAX_DARK_AREA_RATIO:
                continue
            if (
                high_overlap
                and own_protect is not None
                and own_protect.shape == component_dark.shape
            ):
                own_ratio = float(np.count_nonzero(component_dark & own_protect)) / max(
                    1, dark_count
                )
                if own_ratio > CONJOINED_WHITEOUT_TEXT_LINE_MAX_OWN_PROTECT_RATIO:
                    continue

            ys, xs = np.where(component_dark)
            if (
                high_overlap
                and ys.size
                and (
                    int(ys.max()) - int(ys.min()) + 1
                    > crop_h
                    * CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_MAX_COMPONENT_HEIGHT_RATIO
                )
            ):
                continue
            global_xs = xs + crop_x1
            global_ys = ys + crop_y1
            inside_neighbor = (
                (global_xs >= nb_x1)
                & (global_xs < nb_x2)
                & (global_ys >= nb_y1)
                & (global_ys < nb_y2)
            )
            neighbor_overlap_ratio = float(np.count_nonzero(inside_neighbor)) / max(
                1, dark_count
            )
            if (
                neighbor_overlap_ratio
                < CONJOINED_WHITEOUT_TEXT_LINE_MIN_NEIGHBOR_OVERLAP_RATIO
            ):
                continue

            gx = float(np.mean(global_xs))
            gy = float(np.mean(global_ys))
            dist_current = ((gx - cur_cx) ** 2 + (gy - cur_cy) ** 2) ** 0.5
            dist_neighbor = ((gx - nb_cx) ** 2 + (gy - nb_cy) ** 2) ** 0.5
            if dist_neighbor + assign_margin >= dist_current:
                continue

            result |= cv2.dilate(
                component_dark.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ) > 0

    return result if np.any(result) else None


def _neighbor_overlap_axis_text_mask(
    gray: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    own_protect: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Recover neighbor glyph lines misassigned by a conjoined split mask.

    The check is deliberately limited to the current crop's overlap with the
    neighbor bbox. Vertical text lines are assigned on the horizontal axis and
    horizontal lines on the vertical axis, using bbox-normalized distances.
    This handles diagonal child boxes without relaxing protection elsewhere.
    """
    crop_h, crop_w = gray.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return None

    crop_x1, crop_y1, _, _ = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if (
        smaller_child_area <= 0
        or pair_overlap_area / smaller_child_area
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None
    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if smaller_child_area <= 0:
        return None
    pair_overlap_ratio = pair_overlap_area / smaller_child_area
    high_overlap = (
        pair_overlap_ratio > CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    )
    if high_overlap:
        min_side_w = max(1, min(cur_w, nb_w))
        if (
            abs(nb_cx - cur_cx)
            < min_side_w
            * CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_MIN_SEPARATION_RATIO
        ):
            return None

    ys_grid, xs_grid = np.ogrid[:crop_h, :crop_w]
    global_x = xs_grid + crop_x1
    global_y = ys_grid + crop_y1
    overlap_roi = (
        (global_x >= nb_x1)
        & (global_x < nb_x2)
        & (global_y >= nb_y1)
        & (global_y < nb_y2)
    )
    overlap_area = int(np.count_nonzero(overlap_roi))
    if overlap_area <= 0:
        return None

    dark = (gray < 210) & overlap_roi
    if not np.any(dark):
        return None

    result = np.zeros((crop_h, crop_w), dtype=bool)
    dark_u8 = dark.astype(np.uint8)
    if high_overlap:
        oriented_kernels = (
            ("vertical", cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))),
            ("vertical", cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))),
        )
        axis_assign_margin = CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_ASSIGN_MARGIN
        max_dark_area_ratio = (
            CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_MAX_DARK_AREA_RATIO
        )
    else:
        oriented_kernels = (
            ("vertical", cv2.getStructuringElement(cv2.MORPH_RECT, (7, 25))),
            ("horizontal", cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))),
        )
        axis_assign_margin = CONJOINED_WHITEOUT_AXIS_ASSIGN_MARGIN
        max_dark_area_ratio = CONJOINED_WHITEOUT_AXIS_MAX_DARK_AREA_RATIO

    for orientation, kernel in oriented_kernels:
        grouped = cv2.dilate(dark_u8, kernel, iterations=1)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            grouped, connectivity=8
        )
        for label in range(1, num_labels):
            component_dark = dark & (labels == label)
            dark_count = int(np.count_nonzero(component_dark))
            if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
                continue
            if dark_count > overlap_area * max_dark_area_ratio:
                continue
            if (
                high_overlap
                and own_protect is not None
                and own_protect.shape == component_dark.shape
            ):
                own_ratio = float(np.count_nonzero(component_dark & own_protect)) / max(
                    1, dark_count
                )
                if own_ratio > CONJOINED_WHITEOUT_TEXT_LINE_MAX_OWN_PROTECT_RATIO:
                    continue

            component_ys, component_xs = np.where(component_dark)
            if (
                high_overlap
                and component_ys.size
                and (
                    int(component_ys.max()) - int(component_ys.min()) + 1
                    > crop_h
                    * CONJOINED_WHITEOUT_AXIS_HIGH_OVERLAP_MAX_COMPONENT_HEIGHT_RATIO
                )
            ):
                continue
            center_x = float(np.mean(component_xs + crop_x1))
            center_y = float(np.mean(component_ys + crop_y1))
            if high_overlap:
                if nb_cx > cur_cx and center_x <= cur_cx:
                    continue
                if nb_cx < cur_cx and center_x >= cur_cx:
                    continue
                current_distance = abs(center_x - cur_cx) / cur_w
                neighbor_distance = abs(center_x - nb_cx) / nb_w
            elif orientation == "vertical":
                current_distance = abs(center_x - cur_cx) / cur_w
                neighbor_distance = abs(center_x - nb_cx) / nb_w
            else:
                current_distance = abs(center_y - cur_cy) / cur_h
                neighbor_distance = abs(center_y - nb_cy) / nb_h

            if neighbor_distance + axis_assign_margin >= current_distance:
                continue

            result |= cv2.dilate(
                component_dark.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ) > 0

    result &= overlap_roi
    return result if np.any(result) else None


def _neighbor_side_text_rescue_mask(
    gray: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    neighbor_side_mask: Optional[np.ndarray] = None,
    own_protect: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Recover neighbor text tails swallowed by the current split mask.

    This intentionally only handles high-overlap, side-by-side child bubbles.
    Some YOLO split masks include the neighbor's last vertical glyph in the
    current child mask; the broad own-mask protection then preserves fragments
    like a trailing kana. We only rescue compact glyph lines that sit on the
    neighbor-facing side and are clearly closer to the neighbor child center.
    """
    crop_h, crop_w = gray.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return None

    crop_x1, crop_y1, _, _ = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if (
        smaller_child_area <= 0
        or pair_overlap_area / smaller_child_area
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None

    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    dx = nb_cx - cur_cx
    dy = nb_cy - cur_cy
    if abs(dx) < max(1.0, abs(dy) * 0.50):
        return None

    ys_grid, xs_grid = np.ogrid[:crop_h, :crop_w]
    global_x = xs_grid + crop_x1
    global_y = ys_grid + crop_y1
    overlap_roi = (
        (global_x >= nb_x1)
        & (global_x < nb_x2)
        & (global_y >= nb_y1)
        & (global_y < nb_y2)
    )
    if not np.any(overlap_roi):
        return None

    if neighbor_side_mask is not None and neighbor_side_mask.shape == gray.shape:
        rescue_roi = neighbor_side_mask | overlap_roi
    else:
        rescue_roi = overlap_roi

    dark = gray < 210
    seed_dark = dark & rescue_roi
    if not np.any(seed_dark):
        return None

    result = np.zeros((crop_h, crop_w), dtype=bool)
    dark_u8 = seed_dark.astype(np.uint8)
    kernels = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 21)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
    )
    crop_area = max(1, crop_h * crop_w)

    for kernel in kernels:
        grouped = cv2.dilate(dark_u8, kernel, iterations=1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            grouped, connectivity=8
        )
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area <= 0:
                continue
            if (
                h > crop_h * 0.85
                and w < max(8, crop_w * 0.05)
            ) or (
                w > crop_w * 0.80
                and h < max(10, crop_h * 0.10)
            ):
                continue
            if w > h * 1.8 and h < crop_h * 0.25:
                continue

            component_dark = seed_dark & (labels == label)
            dark_count = int(np.count_nonzero(component_dark))
            if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
                continue
            if dark_count > crop_area * CONJOINED_WHITEOUT_SIDE_RESCUE_MAX_DARK_AREA_RATIO:
                continue
            fill_ratio = dark_count / max(1, int(w * h))
            if (
                (x <= 1 or x + w >= crop_w - 1)
                and h > crop_h * 0.50
                and fill_ratio < 0.08
            ):
                continue
            if h > crop_h * 0.70 and fill_ratio > 0.45:
                continue

            side_ratio = float(np.count_nonzero(component_dark & rescue_roi)) / max(
                1, dark_count
            )
            if side_ratio < CONJOINED_WHITEOUT_SIDE_RESCUE_MIN_NEIGHBOR_SIDE_RATIO:
                continue
            neighbor_overlap_ratio = float(
                np.count_nonzero(component_dark & overlap_roi)
            ) / max(1, dark_count)
            if (
                neighbor_overlap_ratio
                < CONJOINED_WHITEOUT_SIDE_RESCUE_MIN_NEIGHBOR_OVERLAP_RATIO
            ):
                continue

            component_ys, component_xs = np.where(component_dark)
            center_x = float(np.mean(component_xs + crop_x1))
            center_y = float(np.mean(component_ys + crop_y1))
            if dx > 0 and center_x <= cur_cx:
                continue
            if dx < 0 and center_x >= cur_cx:
                continue
            current_distance = abs(center_x - cur_cx) / cur_w
            neighbor_distance = abs(center_x - nb_cx) / nb_w
            if (
                neighbor_distance + CONJOINED_WHITEOUT_SIDE_RESCUE_ASSIGN_MARGIN
                >= current_distance
            ):
                continue

            if own_protect is not None and own_protect.shape == component_dark.shape:
                own_ratio = float(np.count_nonzero(component_dark & own_protect)) / max(
                    1, dark_count
                )
                if (
                    own_ratio > CONJOINED_WHITEOUT_SIDE_RESCUE_MAX_OWN_RATIO
                    and neighbor_distance
                    + CONJOINED_WHITEOUT_SIDE_RESCUE_ASSIGN_MARGIN
                    >= current_distance * 0.80
                ):
                    continue

            result |= cv2.dilate(
                component_dark.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ) > 0

    result &= rescue_roi
    return result if np.any(result) else None


def _current_assigned_text_protect_mask(
    gray: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    own_protect: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Protect dark text components that are more likely owned by current crop.

    The neighbor whiteout region can still touch the current bubble when the
    split masks overlap heavily. Before painting pixels white, group visible
    glyph fragments and keep components whose centroid or mask overlap clearly
    belongs to the current child.
    """
    crop_h, crop_w = gray.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return None

    dark = gray < 210
    if not np.any(dark):
        return None

    crop_x1, crop_y1, _, _ = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if (
        smaller_child_area <= 0
        or pair_overlap_area / smaller_child_area
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None
    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0

    result = np.zeros((crop_h, crop_w), dtype=bool)
    dark_u8 = dark.astype(np.uint8)
    if abs(nb_cx - cur_cx) >= abs(nb_cy - cur_cy) * 0.75:
        kernels = (
            ("vertical", cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))),
        )
    else:
        kernels = (
            ("horizontal", cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))),
        )
    crop_area = max(1, crop_h * crop_w)

    for orientation, kernel in kernels:
        grouped = cv2.dilate(dark_u8, kernel, iterations=1)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            grouped, connectivity=8
        )
        for label in range(1, num_labels):
            component_dark = dark & (labels == label)
            dark_count = int(np.count_nonzero(component_dark))
            if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
                continue
            if dark_count > crop_area * CONJOINED_WHITEOUT_TEXT_LINE_MAX_DARK_AREA_RATIO:
                continue

            if own_protect is None or own_protect.shape != component_dark.shape:
                continue
            own_ratio = float(np.count_nonzero(component_dark & own_protect)) / max(
                1, dark_count
            )
            if own_ratio <= CONJOINED_WHITEOUT_TEXT_LINE_MAX_OWN_PROTECT_RATIO:
                continue

            component_ys, component_xs = np.where(component_dark)
            center_x = float(np.mean(component_xs + crop_x1))
            center_y = float(np.mean(component_ys + crop_y1))
            if orientation == "vertical":
                current_distance = abs(center_x - cur_cx) / cur_w
                neighbor_distance = abs(center_x - nb_cx) / nb_w
            else:
                current_distance = abs(center_y - cur_cy) / cur_h
                neighbor_distance = abs(center_y - nb_cy) / nb_h
            if (
                current_distance + CONJOINED_WHITEOUT_CURRENT_TEXT_ASSIGN_MARGIN
                >= neighbor_distance
            ):
                continue

            result |= cv2.dilate(
                component_dark.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ) > 0

    return result if np.any(result) else None


def _borderline_current_text_protect_mask(
    gray: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    own_protect: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Protect strongly owned text lines near the high-overlap cutoff.

    This intentionally covers only a narrow overlap band. It prevents a
    neighboring child from erasing the current bubble's edge glyphs without
    relaxing the general whiteout rules used by less-overlapping bubbles.
    """
    crop_h, crop_w = gray.shape[:2]
    if (
        crop_h <= 0
        or crop_w <= 0
        or own_protect is None
        or own_protect.shape != gray.shape[:2]
    ):
        return None

    crop_x1, crop_y1, _, _ = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if smaller_child_area <= 0:
        return None
    pair_overlap_ratio = pair_overlap_area / smaller_child_area
    if not (
        CONJOINED_WHITEOUT_BORDERLINE_MIN_CHILD_OVERLAP_RATIO
        <= pair_overlap_ratio
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None

    dark = gray < 210
    if not np.any(dark):
        return None

    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    kernels = (
        ("vertical", cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))),
        ("horizontal", cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))),
    )
    crop_area = max(1, crop_h * crop_w)
    result = np.zeros((crop_h, crop_w), dtype=bool)
    dark_u8 = dark.astype(np.uint8)

    for orientation, kernel in kernels:
        grouped = cv2.dilate(dark_u8, kernel, iterations=1)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            grouped, connectivity=8
        )
        for label in range(1, num_labels):
            component_dark = dark & (labels == label)
            dark_count = int(np.count_nonzero(component_dark))
            if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
                continue
            if dark_count > crop_area * CONJOINED_WHITEOUT_TEXT_LINE_MAX_DARK_AREA_RATIO:
                continue

            own_ratio = float(np.count_nonzero(component_dark & own_protect)) / max(
                1, dark_count
            )
            if own_ratio < CONJOINED_WHITEOUT_BORDERLINE_MIN_OWN_PROTECT_RATIO:
                continue

            component_ys, component_xs = np.where(component_dark)
            component_w = int(component_xs.max() - component_xs.min() + 1)
            component_h = int(component_ys.max() - component_ys.min() + 1)
            if orientation == "vertical":
                if (
                    component_h
                    < component_w
                    * CONJOINED_WHITEOUT_BORDERLINE_MIN_LINE_ASPECT_RATIO
                ):
                    continue
            elif (
                component_w
                < component_h
                * CONJOINED_WHITEOUT_BORDERLINE_MIN_LINE_ASPECT_RATIO
            ):
                continue

            center_x = float(np.mean(component_xs + crop_x1))
            center_y = float(np.mean(component_ys + crop_y1))
            if orientation == "vertical":
                current_distance = abs(center_x - cur_cx) / cur_w
                neighbor_distance = abs(center_x - nb_cx) / nb_w
            else:
                current_distance = abs(center_y - cur_cy) / cur_h
                neighbor_distance = abs(center_y - nb_cy) / nb_h
            if (
                current_distance + CONJOINED_WHITEOUT_CURRENT_TEXT_ASSIGN_MARGIN
                >= neighbor_distance
            ):
                continue

            result |= cv2.dilate(
                component_dark.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ) > 0

    return result if np.any(result) else None


def _borderline_neighbor_owned_text_mask(
    gray: np.ndarray,
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    neighbor_region: Optional[np.ndarray],
    own_protect: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Recover small glyphs strongly owned by the neighbor segmentation mask."""
    crop_h, crop_w = gray.shape[:2]
    if (
        crop_h <= 0
        or crop_w <= 0
        or neighbor_region is None
        or neighbor_region.shape != gray.shape[:2]
        or own_protect is None
        or own_protect.shape != gray.shape[:2]
    ):
        return None

    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if smaller_child_area <= 0:
        return None
    pair_overlap_ratio = pair_overlap_area / smaller_child_area
    if not (
        CONJOINED_WHITEOUT_BORDERLINE_MIN_CHILD_OVERLAP_RATIO
        <= pair_overlap_ratio
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None

    dark = gray < 210
    if not np.any(dark):
        return None
    grouped = cv2.dilate(
        dark.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        grouped, connectivity=8
    )
    crop_area = max(1, crop_h * crop_w)
    result = np.zeros((crop_h, crop_w), dtype=bool)

    for label in range(1, num_labels):
        component_dark = dark & (labels == label)
        dark_count = int(np.count_nonzero(component_dark))
        if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
            continue
        if (
            dark_count
            > crop_area * CONJOINED_WHITEOUT_BORDERLINE_MAX_COMPONENT_AREA_RATIO
        ):
            continue

        component_ys, component_xs = np.where(component_dark)
        component_w = int(component_xs.max() - component_xs.min() + 1)
        component_h = int(component_ys.max() - component_ys.min() + 1)
        if (
            component_w
            > crop_w * CONJOINED_WHITEOUT_BORDERLINE_MAX_COMPONENT_DIMENSION_RATIO
            or component_h
            > crop_h * CONJOINED_WHITEOUT_BORDERLINE_MAX_COMPONENT_DIMENSION_RATIO
        ):
            continue

        neighbor_ratio = float(
            np.count_nonzero(component_dark & neighbor_region)
        ) / max(1, dark_count)
        if (
            neighbor_ratio
            < CONJOINED_WHITEOUT_BORDERLINE_MIN_NEIGHBOR_MASK_RATIO
        ):
            continue
        own_ratio = float(np.count_nonzero(component_dark & own_protect)) / max(
            1, dark_count
        )
        if own_ratio > CONJOINED_WHITEOUT_BORDERLINE_MAX_OWN_PROTECT_RATIO:
            continue

        neighbor_owned_dark = component_dark & neighbor_region & ~own_protect
        result |= cv2.dilate(
            neighbor_owned_dark.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ) > 0

    return result if np.any(result) else None


def _neighbor_text_box_rescue_mask(
    gray: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    current_text_boxes: Optional[List[Any]],
    neighbor_text_boxes: Optional[List[Any]],
    neighbor_region: Optional[np.ndarray],
    own_mask_crop: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Recover high-overlap neighbor glyphs confirmed by OCR text boxes."""
    crop_h, crop_w = gray.shape[:2]
    if (
        crop_h <= 0
        or crop_w <= 0
        or not current_text_boxes
        or not neighbor_text_boxes
    ):
        return None

    crop_x1, crop_y1, _, _ = crop_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = current_bbox
    nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
    cur_w = max(1, cur_x2 - cur_x1)
    cur_h = max(1, cur_y2 - cur_y1)
    nb_w = max(1, nb_x2 - nb_x1)
    nb_h = max(1, nb_y2 - nb_y1)
    pair_overlap_w = max(0, min(cur_x2, nb_x2) - max(cur_x1, nb_x1))
    pair_overlap_h = max(0, min(cur_y2, nb_y2) - max(cur_y1, nb_y1))
    pair_overlap_area = pair_overlap_w * pair_overlap_h
    smaller_child_area = min(cur_w * cur_h, nb_w * nb_h)
    if (
        smaller_child_area <= 0
        or pair_overlap_area / smaller_child_area
        <= CONJOINED_WHITEOUT_AXIS_MAX_CHILD_OVERLAP_RATIO
    ):
        return None

    def boxes_to_mask(boxes: List[Any]) -> np.ndarray:
        result = np.zeros((crop_h, crop_w), dtype=bool)
        for box in boxes:
            try:
                box_x1, box_y1, box_x2, box_y2 = [
                    int(round(float(value))) for value in box
                ]
            except (TypeError, ValueError):
                continue
            local_x1 = max(0, min(crop_w, box_x1 - crop_x1))
            local_y1 = max(0, min(crop_h, box_y1 - crop_y1))
            local_x2 = max(0, min(crop_w, box_x2 - crop_x1))
            local_y2 = max(0, min(crop_h, box_y2 - crop_y1))
            if local_x2 > local_x1 and local_y2 > local_y1:
                result[local_y1:local_y2, local_x1:local_x2] = True
        return result

    current_text_roi = boxes_to_mask(current_text_boxes)
    neighbor_text_roi = boxes_to_mask(neighbor_text_boxes)
    if not np.any(neighbor_text_roi):
        return None

    ys_grid, xs_grid = np.ogrid[:crop_h, :crop_w]
    global_x = xs_grid + crop_x1
    global_y = ys_grid + crop_y1
    overlap_roi = (
        (global_x >= nb_x1)
        & (global_x < nb_x2)
        & (global_y >= nb_y1)
        & (global_y < nb_y2)
    )
    candidate_dark = (
        (gray < 210)
        & overlap_roi
        & neighbor_text_roi
        & ~current_text_roi
    )
    if not np.any(candidate_dark):
        return None

    cur_cx = (cur_x1 + cur_x2) / 2.0
    cur_cy = (cur_y1 + cur_y2) / 2.0
    nb_cx = (nb_x1 + nb_x2) / 2.0
    nb_cy = (nb_y1 + nb_y2) / 2.0
    crop_area = max(1, crop_h * crop_w)
    result = np.zeros((crop_h, crop_w), dtype=bool)
    if (
        neighbor_region is not None
        and neighbor_region.shape == gray.shape
        and own_mask_crop is not None
        and own_mask_crop.shape == gray.shape
    ):
        exclusive_seed = (
            (gray < 210)
            & overlap_roi
            & neighbor_text_roi
            & neighbor_region
            & ~own_mask_crop
        )
        if np.any(exclusive_seed):
            grouped = cv2.dilate(
                exclusive_seed.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19)),
                iterations=1,
            )
            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
                grouped, connectivity=8
            )
            for label in range(1, num_labels):
                component_dark = exclusive_seed & (labels == label)
                dark_count = int(np.count_nonzero(component_dark))
                if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
                    continue
                component_ys, component_xs = np.where(component_dark)
                component_w = int(component_xs.max() - component_xs.min() + 1)
                component_h = int(component_ys.max() - component_ys.min() + 1)
                if int(component_ys.max()) >= crop_h - 2:
                    continue
                if component_h < component_w * 1.20:
                    continue
                fill_ratio = dark_count / max(1, component_w * component_h)
                if fill_ratio < 0.08:
                    continue
                result |= cv2.dilate(
                    component_dark.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                ) > 0
            result &= neighbor_region & ~own_mask_crop

    candidate_u8 = candidate_dark.astype(np.uint8)
    oriented_kernels = (
        ("vertical", cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))),
    )

    for orientation, kernel in oriented_kernels:
        grouped = cv2.dilate(candidate_u8, kernel, iterations=1)
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            grouped, connectivity=8
        )
        for label in range(1, num_labels):
            component_dark = candidate_dark & (labels == label)
            dark_count = int(np.count_nonzero(component_dark))
            if dark_count < CONJOINED_WHITEOUT_TEXT_LINE_MIN_DARK_PIXELS:
                continue
            if (
                dark_count
                > crop_area
                * CONJOINED_WHITEOUT_TEXT_BOX_RESCUE_MAX_DARK_AREA_RATIO
            ):
                continue

            component_ys, component_xs = np.where(component_dark)
            component_w = int(component_xs.max() - component_xs.min() + 1)
            component_h = int(component_ys.max() - component_ys.min() + 1)
            if (
                component_w
                > crop_w * CONJOINED_WHITEOUT_TEXT_BOX_RESCUE_MAX_DIMENSION_RATIO
                or component_h
                > crop_h * CONJOINED_WHITEOUT_TEXT_BOX_RESCUE_MAX_DIMENSION_RATIO
            ):
                continue
            fill_ratio = dark_count / max(1, component_w * component_h)
            if fill_ratio < 0.08:
                continue
            if orientation == "vertical" and component_h < component_w * 1.20:
                continue
            center_x = float(np.mean(component_xs + crop_x1))
            center_y = float(np.mean(component_ys + crop_y1))
            current_distance = abs(center_x - cur_cx) / cur_w
            neighbor_distance = abs(center_x - nb_cx) / nb_w
            if (
                neighbor_distance
                + CONJOINED_WHITEOUT_TEXT_BOX_RESCUE_ASSIGN_MARGIN
                >= current_distance
            ):
                continue

            result |= cv2.dilate(
                component_dark.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ) > 0

    result &= overlap_roi & ~current_text_roi
    return result if np.any(result) else None


def _whiteout_neighbor_text_pixels(
    crop_cv: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    current_bbox: tuple[int, int, int, int],
    neighbor_bbox: tuple[int, int, int, int],
    neighbor_mask: Any,
    own_mask_crop: Optional[np.ndarray],
    current_text_boxes: Optional[List[Any]] = None,
    neighbor_text_boxes: Optional[List[Any]] = None,
) -> int:
    """Hide neighboring conjoined-bubble text in an LLM crop.

    The old behavior painted the neighbor mask directly. That fails when the
    split mask misses text and can also erase parts of the current bubble. This
    version builds a neighbor region from mask or bbox, protects the current
    bubble mask, then whites out only text-like dark pixels plus a small halo.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    crop_h, crop_w = crop_cv.shape[:2]
    if crop_h <= 0 or crop_w <= 0:
        return 0

    region_mask = None
    segmented_neighbor_region = None
    normalized_neighbor = _normalize_bool_mask(neighbor_mask)
    if normalized_neighbor is not None:
        region_mask = normalized_neighbor[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        segmented_neighbor_region = region_mask.copy()

    if region_mask is None or not np.any(region_mask):
        nb_x1, nb_y1, nb_x2, nb_y2 = neighbor_bbox
        ix1 = max(crop_x1, nb_x1)
        iy1 = max(crop_y1, nb_y1)
        ix2 = min(crop_x2, nb_x2)
        iy2 = min(crop_y2, nb_y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        region_mask = np.zeros((crop_h, crop_w), dtype=bool)
        region_mask[iy1 - crop_y1 : iy2 - crop_y1, ix1 - crop_x1 : ix2 - crop_x1] = True

    rescue_region = _right_neighbor_bbox_whiteout_region(
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        (crop_h, crop_w),
    )
    force_low_overlap_left_rescue = (
        rescue_region is not None
        and _is_low_overlap_left_neighbor(current_bbox, neighbor_bbox)
    )
    low_overlap_left_side_mask = _low_overlap_left_neighbor_side_mask(
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        (crop_h, crop_w),
    )
    neighbor_side_mask = _neighbor_side_mask_for_crop(
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        (crop_h, crop_w),
    )

    if crop_cv.ndim == 2:
        gray = crop_cv
    elif crop_cv.shape[2] == 4:
        gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(crop_cv, cv2.COLOR_BGR2GRAY)

    own_protect = None
    if own_mask_crop is not None and own_mask_crop.shape == region_mask.shape:
        own_protect = cv2.dilate(
            own_mask_crop.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ) > 0

    neighbor_text_line_mask = _neighbor_assigned_text_line_mask(
        gray,
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        own_protect=own_protect,
    )
    neighbor_axis_text_mask = _neighbor_overlap_axis_text_mask(
        gray,
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        own_protect=own_protect,
    )
    neighbor_side_rescue_mask = _neighbor_side_text_rescue_mask(
        gray,
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        neighbor_side_mask=neighbor_side_mask,
        own_protect=own_protect,
    )
    if neighbor_axis_text_mask is not None:
        if neighbor_text_line_mask is None:
            neighbor_text_line_mask = neighbor_axis_text_mask
        else:
            neighbor_text_line_mask |= neighbor_axis_text_mask
    if neighbor_side_rescue_mask is not None:
        if neighbor_text_line_mask is None:
            neighbor_text_line_mask = neighbor_side_rescue_mask
        else:
            neighbor_text_line_mask |= neighbor_side_rescue_mask

    if own_protect is not None:
        region_mask &= ~own_protect
        if rescue_region is not None:
            protected_dark = (gray < 235) & rescue_region & own_protect
            protected_dark_count = int(np.count_nonzero(protected_dark))
            protected_dark_ratio = protected_dark_count / max(
                1, int(np.count_nonzero(rescue_region))
            )
            if force_low_overlap_left_rescue or (
                protected_dark_count
                <= CONJOINED_WHITEOUT_RESCUE_MAX_PROTECTED_DARK_PIXELS
                and protected_dark_ratio
                <= CONJOINED_WHITEOUT_RESCUE_MAX_PROTECTED_DARK_RATIO
            ):
                region_mask |= rescue_region
    elif rescue_region is not None:
        region_mask |= rescue_region

    if neighbor_side_mask is not None and neighbor_side_mask.shape == region_mask.shape:
        if (
            neighbor_text_line_mask is not None
            and neighbor_text_line_mask.shape == region_mask.shape
        ):
            region_mask = (region_mask & neighbor_side_mask) | neighbor_text_line_mask
        else:
            region_mask &= neighbor_side_mask
        if force_low_overlap_left_rescue:
            region_mask |= neighbor_side_mask
    if (
        low_overlap_left_side_mask is not None
        and low_overlap_left_side_mask.shape == region_mask.shape
    ):
        region_mask |= low_overlap_left_side_mask
    elif (
        neighbor_text_line_mask is not None
        and neighbor_text_line_mask.shape == region_mask.shape
    ):
        region_mask |= neighbor_text_line_mask

    current_text_protect = _current_assigned_text_protect_mask(
        gray,
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        own_protect=own_protect,
    )
    if (
        current_text_protect is not None
        and current_text_protect.shape == region_mask.shape
    ):
        region_mask &= ~current_text_protect
        if (
            neighbor_side_rescue_mask is not None
            and neighbor_side_rescue_mask.shape == region_mask.shape
        ):
            region_mask |= neighbor_side_rescue_mask

    borderline_current_text_protect = _borderline_current_text_protect_mask(
        gray,
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        own_protect=own_protect,
    )
    if (
        borderline_current_text_protect is not None
        and borderline_current_text_protect.shape == region_mask.shape
    ):
        region_mask &= ~borderline_current_text_protect

    borderline_neighbor_text = _borderline_neighbor_owned_text_mask(
        gray,
        current_bbox,
        neighbor_bbox,
        neighbor_region=segmented_neighbor_region,
        own_protect=own_protect,
    )
    if (
        borderline_neighbor_text is not None
        and borderline_neighbor_text.shape == region_mask.shape
    ):
        region_mask |= borderline_neighbor_text

    neighbor_text_box_rescue = _neighbor_text_box_rescue_mask(
        gray,
        crop_bbox,
        current_bbox,
        neighbor_bbox,
        current_text_boxes=current_text_boxes,
        neighbor_text_boxes=neighbor_text_boxes,
        neighbor_region=segmented_neighbor_region,
        own_mask_crop=own_mask_crop,
    )
    if (
        neighbor_text_box_rescue is not None
        and neighbor_text_box_rescue.shape == region_mask.shape
    ):
        region_mask |= neighbor_text_box_rescue

    if not np.any(region_mask):
        return 0

    text_like = (gray < 235) & region_mask
    if not np.any(text_like):
        return 0

    whiteout_mask = cv2.dilate(
        text_like.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    whiteout_mask &= region_mask

    if crop_cv.ndim == 2:
        crop_cv[whiteout_mask] = 255
    else:
        crop_cv[whiteout_mask] = _cv_white_value(crop_cv)

    return int(np.count_nonzero(whiteout_mask))


def prepare_bubble_images_for_translation(
    bubble_data: List[Dict[str, Any]],
    original_cv_image: np.ndarray,
    upscale_model: Any,
    device: Any,
    mime_type: str,
    bubble_min_side_pixels: int,
    upscale_method: str = "model_lite",
    whiteout_conjoined_bubbles: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Prepare bubble images for translation by cropping, upscaling, color matching, and encoding.

    This function processes each speech bubble to prepare it for the translation API:
    1. Crops the bubble from the original image
    2. Upscales the bubble to meet minimum size requirements (based on upscale_method)
    3. Matches colors to preserve visual consistency (only for model upscaling)
    4. Encodes the processed bubble as base64 for API transmission

    Args:
        bubble_data: List of bubble detection dicts with 'bbox' keys
        original_cv_image: OpenCV image array of the original image
        upscale_model: Loaded upscaling model
        device: PyTorch device for model inference
        mime_type: MIME type for image encoding
        upscale_method: Method for upscaling - "model", "lanczos", or "none"
        verbose: Whether to print detailed logging

    Returns:
        List of bubble dicts with added 'image_b64' and 'mime_type' keys
        (immutable approach - returns new list without mutating input)
    """
    cv2_ext = ".png" if mime_type == "image/png" else ".jpg"

    prepared_bubbles = []

    mask_lookup = {}
    mask_entries = []
    bubble_lookup = {}
    for b in bubble_data:
        b_bbox = tuple(int(round(v)) for v in b["bbox"])
        b_mask = b.get("sam_mask")
        mask_lookup[b_bbox] = b_mask
        mask_entries.append((b_bbox, b_mask))
        bubble_lookup[b_bbox] = b

    if upscale_method == "model":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with 2x-AnimeSharpV4_RCAN",
            always_print=True,
        )
    elif upscale_method == "model_lite":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with 2x-AnimeSharpV4_Fast_RCAN_PU (Lite)",
            always_print=True,
        )
    elif upscale_method == "lanczos":
        log_message(
            f"Upscaling {len(bubble_data)} bubble images with LANCZOS",
            always_print=True,
        )
    else:
        log_message(
            f"Processing {len(bubble_data)} bubble images without upscaling",
            always_print=True,
        )

    for bubble in bubble_data:
        prepared_bubble = bubble.copy()
        x1, y1, x2, y2 = bubble["bbox"]

        # Use the tight bbox of the mask
        _mask = bubble.get("sam_mask")
        _ma = None
        if _mask is not None:
            _ma = np.asarray(_mask)
            if _ma.ndim == 3:
                _ma = _ma[..., 0]
            if _ma.ndim == 2:
                _rows, _cols = np.where(_ma > 0)
                if _rows.size and _cols.size:
                    mx1, my1 = int(_cols.min()), int(_rows.min())
                    mx2, my2 = int(_cols.max()) + 1, int(_rows.max()) + 1
                    x1 = min(x1, mx1)
                    y1 = min(y1, my1)
                    x2 = max(x2, mx2)
                    y2 = max(y2, my2)

        bubble_image_cv = original_cv_image[y1:y2, x1:x2].copy()

        # White-out conjoined neighbor text regions visible in this crop
        neighbor_bboxes = bubble.get("conjoined_neighbor_bboxes")
        if whiteout_conjoined_bubbles and neighbor_bboxes:
            own_mask_crop = (
                _ma[y1:y2, x1:x2] > 0 if (_ma is not None and _ma.ndim == 2) else None
            )

            for nb in neighbor_bboxes:
                nb_tuple = tuple(int(round(v)) for v in nb)
                neighbor_bubble = bubble_lookup.get(nb_tuple)
                neighbor_mask = _find_conjoined_neighbor_mask(
                    mask_lookup, mask_entries, nb_tuple
                )
                whitened_px = _whiteout_neighbor_text_pixels(
                    bubble_image_cv,
                    crop_bbox=(x1, y1, x2, y2),
                    current_bbox=tuple(int(round(v)) for v in bubble["bbox"]),
                    neighbor_bbox=nb_tuple,
                    neighbor_mask=neighbor_mask,
                    own_mask_crop=own_mask_crop,
                    current_text_boxes=bubble.get("text_boxes"),
                    neighbor_text_boxes=(
                        neighbor_bubble.get("text_boxes")
                        if neighbor_bubble is not None
                        else None
                    ),
                )
                if whitened_px:
                    log_message(
                        f"White-out conjoined neighbor text for bubble {bubble.get('bbox')}: "
                        f"{whitened_px} px",
                        verbose=verbose,
                    )

        exclusion_bboxes = bubble.get("translation_exclusion_bboxes") or []
        if exclusion_bboxes:
            total_excluded_px = 0
            for exclusion_bbox in exclusion_bboxes:
                try:
                    ex1, ey1, ex2, ey2 = [
                        int(round(float(v))) for v in exclusion_bbox
                    ]
                except (TypeError, ValueError):
                    continue
                lx1 = max(0, min(bubble_image_cv.shape[1], ex1 - x1))
                ly1 = max(0, min(bubble_image_cv.shape[0], ey1 - y1))
                lx2 = max(0, min(bubble_image_cv.shape[1], ex2 - x1))
                ly2 = max(0, min(bubble_image_cv.shape[0], ey2 - y1))
                if lx2 <= lx1 or ly2 <= ly1:
                    continue
                bubble_image_cv[ly1:ly2, lx1:lx2] = _cv_white_value(bubble_image_cv)
                total_excluded_px += int((lx2 - lx1) * (ly2 - ly1))
            if total_excluded_px:
                log_message(
                    "White-out separated edge text for bubble "
                    f"{bubble.get('bbox')}: {total_excluded_px} px",
                    verbose=verbose,
                )

        bubble_image_pil = cv2_to_pil(bubble_image_cv)

        if upscale_method == "model" or upscale_method == "model_lite":
            final_bubble_pil = process_bubble_image_cached(
                bubble_image_pil,
                upscale_model,
                device,
                bubble_min_side_pixels,
                "min",
                upscale_method,
                verbose,
            )
        elif upscale_method == "lanczos":
            w, h = bubble_image_pil.size
            min_side = min(w, h)
            if min_side < bubble_min_side_pixels:
                scale_factor = bubble_min_side_pixels / min_side
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                resized_bubble = bubble_image_pil.resize((new_w, new_h), Image.LANCZOS)
            else:
                resized_bubble = bubble_image_pil
            final_bubble_pil = resized_bubble
        else:
            final_bubble_pil = bubble_image_pil

        final_bubble_cv = pil_to_cv2(final_bubble_pil)

        try:
            is_success, buffer = cv2.imencode(cv2_ext, final_bubble_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                prepared_bubble["image_b64"] = image_b64
                prepared_bubble["mime_type"] = mime_type
                log_message(
                    f"Bubble {x1},{y1} ({final_bubble_pil.size[0]}x{final_bubble_pil.size[1]})",
                    verbose=verbose,
                )
            else:
                log_message(
                    f"Failed to encode bubble {bubble['bbox']}", verbose=verbose
                )
                prepared_bubble["image_b64"] = None
        except Exception as e:
            log_message(f"Error encoding bubble {bubble['bbox']}: {e}", verbose=verbose)
            prepared_bubble["image_b64"] = None

        prepared_bubbles.append(prepared_bubble)

    return prepared_bubbles
