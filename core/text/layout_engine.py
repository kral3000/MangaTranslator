import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import skia
import uharfbuzz as hb

from core.text.text_processing import (
    NO_SPACE_BEFORE_MARKER,
    STYLE_PATTERN,
    attach_korean_postposition_particles,
    find_optimal_breaks_dp,
    glue_korean_postposition_particles,
    hyphenation_token_length,
    is_detached_trailing_punctuation,
    parse_styled_segments,
    restore_explicit_spaces,
    split_hangul_word_for_wrapping,
    strip_no_space_before_marker,
    tokenize_styled_text,
    try_hyphenate_word,
)
from utils.exceptions import RenderingError
from utils.logging import log_message
from core.text.font_fallback import pick_font_key_for_char, split_text_by_font_support

# Epsilon to guard rounding when converting from HarfBuzz 26.6 fixed-point.
VISUAL_WIDTH_EPSILON = 0.0
HB_26_6_SCALE_FACTOR = 64.0
VERTICAL_ADVANCE_TRACKING = 0.90
VERTICAL_BASE_GAP_RATIO = 0.08
VERTICAL_BASE_GAP_MIN = 1.5
VERTICAL_BASE_GAP_MAX = 4.0
VERTICAL_EXTRA_GAP_SQRT_RATIO = 3.0
VERTICAL_EXTRA_GAP_MIN = 8.0
VERTICAL_EXTRA_GAP_MAX = 18.0
WRAPPED_SYMBOL_RELAX_FONT_SIZE_THRESHOLD = 18
COLLISION_VERY_SLIGHT_RATIO = 0.06
COLLISION_SLIGHT_RATIO = 0.10
COLLISION_VERY_SLIGHT_MAX_CORNERS = 1
COLLISION_SLIGHT_MAX_CORNERS = 2
COLLISION_SHIFT_BASE_MAX_PX = 14.0
COLLISION_SHIFT_MAX_SIDE_RATIO = 0.12
SHORT_TEXT_MAX_2_LINES_CHAR_LIMIT = 6
SHORT_TEXT_MAX_2_LINES = 2
SHORT_WORD_MAX_2_LINES_CHAR_LIMIT = 6
VERTICAL_COLUMN_GAP_RATIO = 0.35
VERTICAL_COLUMN_GAP_MIN = 4.0
VERTICAL_COLUMN_GAP_MAX = 16.0
VERTICAL_GROUPED_PUNCTUATION = set(".,;:!?…。．！？｡")
VERTICAL_ATTACHED_PUNCTUATION = set("!?！？,")
VERTICAL_GROUPED_ROTATED_GLYPHS = set("~～〜")
DASH_LINE_CHARS = set("-‐‑‒–—―−─━ーｰㅡ")
# Glyphs that should be drawn sideways inside a vertical OSB column.
# This includes bracket-like punctuation so strings such as "(신체강화)"
# render as top/bottom parentheses instead of awkward side parentheses.
VERTICAL_VERTICAL_BRACKET_GLYPHS = set("()（）[]［］{}｛｝【】「」『』《》〈〉〔〕〖〗〘〙〚〛")
VERTICAL_ROTATED_GLYPHS = DASH_LINE_CHARS | set("~～〜") | VERTICAL_VERTICAL_BRACKET_GLYPHS
VERTICAL_ROTATION_DEGREES = 90.0
VERTICAL_SPACE_TOKEN = "\ue002"
VERTICAL_CENTERED_DOT_TOKEN = "\ue007"
VERTICAL_WORD_SPACE_RATIO = 0.24
VERTICAL_WORD_SPACE_MIN = 2.0
VERTICAL_WORD_SPACE_MAX = 7.0
VERTICAL_DOT_ELLIPSIS_CHARS = {".", "\uff0e"}
VERTICAL_ELLIPSIS_CHARS = {"\u2025", "\u2026", "\u22ef", "\ufe19", "\u22ee"}
VERTICAL_SENTENCE_PERIOD_CHARS = {"."}
VERTICAL_TRAILING_CLOSERS = set(
    ")]}）］｝」』》〉】〕〗〙〛\"'\u2019\u201d"
)
VERTICAL_QUOTE_OPEN_SINGLE_TOKEN = "\ue003"
VERTICAL_QUOTE_CLOSE_SINGLE_TOKEN = "\ue004"
VERTICAL_QUOTE_OPEN_DOUBLE_TOKEN = "\ue005"
VERTICAL_QUOTE_CLOSE_DOUBLE_TOKEN = "\ue006"
VERTICAL_QUOTE_DISPLAY_GLYPHS = {
    VERTICAL_QUOTE_OPEN_SINGLE_TOKEN: "\u300c",
    VERTICAL_QUOTE_CLOSE_SINGLE_TOKEN: "\u300d",
    VERTICAL_QUOTE_OPEN_DOUBLE_TOKEN: "\u300e",
    VERTICAL_QUOTE_CLOSE_DOUBLE_TOKEN: "\u300f",
}


def _normalize_layout_scale(layout_scale: Optional[float]) -> float:
    try:
        value = float(layout_scale)
    except (TypeError, ValueError):
        return 1.0
    return value if value > 0.0 else 1.0


def _scale_layout_px(value: float, layout_scale: Optional[float]) -> float:
    return float(value) * _normalize_layout_scale(layout_scale)
VERTICAL_QUOTE_TOKEN_INFO = {
    VERTICAL_QUOTE_OPEN_SINGLE_TOKEN: ("open", "single"),
    VERTICAL_QUOTE_CLOSE_SINGLE_TOKEN: ("close", "single"),
    VERTICAL_QUOTE_OPEN_DOUBLE_TOKEN: ("open", "double"),
    VERTICAL_QUOTE_CLOSE_DOUBLE_TOKEN: ("close", "double"),
}


def _plain_layout_text(text: str) -> str:
    value = str(text or "").replace(NO_SPACE_BEFORE_MARKER, "")
    return "".join(segment_text for segment_text, _style in parse_styled_segments(value))


def _visible_layout_char_count(text: str) -> int:
    return sum(1 for ch in _plain_layout_text(text) if not ch.isspace())


def _short_text_max_lines(text: str, vertical_stack: bool) -> Optional[int]:
    if vertical_stack:
        return None
    visible_chars = _visible_layout_char_count(text)
    if 0 < visible_chars <= SHORT_TEXT_MAX_2_LINES_CHAR_LIMIT:
        return SHORT_TEXT_MAX_2_LINES
    return None


def _short_word_line_limit_spans(text: str) -> List[Tuple[int, int]]:
    plain_text = _plain_layout_text(text)
    spans: List[Tuple[int, int]] = []
    compact_pos = 0
    run_start: Optional[int] = None
    run_visible_chars = 0

    def finish_run() -> None:
        nonlocal run_start
        nonlocal run_visible_chars
        if (
            run_start is not None
            and 0 < run_visible_chars <= SHORT_WORD_MAX_2_LINES_CHAR_LIMIT
        ):
            spans.append((run_start, run_start + run_visible_chars))
        run_start = None
        run_visible_chars = 0

    for ch in plain_text:
        if ch.isspace():
            finish_run()
            continue
        if run_start is None:
            run_start = compact_pos
        run_visible_chars += 1
        compact_pos += 1

    finish_run()
    return spans


def _short_words_fit_line_limit(
    source_text: str,
    rendered_lines: List[str],
) -> bool:
    spans = _short_word_line_limit_spans(source_text)
    if not spans:
        return True

    line_ranges: List[Tuple[int, int]] = []
    compact_pos = 0
    for line_text in rendered_lines:
        line_chars = _visible_layout_char_count(line_text)
        line_ranges.append((compact_pos, compact_pos + line_chars))
        compact_pos += line_chars

    for start, end in spans:
        touched_lines = 0
        for line_start, line_end in line_ranges:
            if start < line_end and end > line_start:
                touched_lines += 1
                if touched_lines > SHORT_TEXT_MAX_2_LINES:
                    return False
    return True


VERTICAL_QUOTE_OPEN_GLYPHS = {
    "'": VERTICAL_QUOTE_OPEN_SINGLE_TOKEN,
    '"': VERTICAL_QUOTE_OPEN_DOUBLE_TOKEN,
    "\u2018": VERTICAL_QUOTE_OPEN_SINGLE_TOKEN,
    "\u201c": VERTICAL_QUOTE_OPEN_DOUBLE_TOKEN,
    "\u300c": VERTICAL_QUOTE_OPEN_SINGLE_TOKEN,
    "\u300e": VERTICAL_QUOTE_OPEN_DOUBLE_TOKEN,
}
VERTICAL_QUOTE_CLOSE_GLYPHS = {
    "'": VERTICAL_QUOTE_CLOSE_SINGLE_TOKEN,
    '"': VERTICAL_QUOTE_CLOSE_DOUBLE_TOKEN,
    "\u2019": VERTICAL_QUOTE_CLOSE_SINGLE_TOKEN,
    "\u201d": VERTICAL_QUOTE_CLOSE_DOUBLE_TOKEN,
    "\u300d": VERTICAL_QUOTE_CLOSE_SINGLE_TOKEN,
    "\u300f": VERTICAL_QUOTE_CLOSE_DOUBLE_TOKEN,
}
VERTICAL_SYMMETRIC_QUOTE_GLYPHS = {"'", '"'}


def _nearest_non_space(text: str, index: int, step: int) -> str:
    cursor = index + step
    while 0 <= cursor < len(text):
        ch = text[cursor]
        if not _is_separator_or_space(ch):
            return ch
        cursor += step
    return ""


def _vertical_quote_glyph(
    ch: str, text: str, index: int, quote_stack: List[str]
) -> str:
    """Use corner brackets for quotes in Korean/Japanese vertical text."""
    if ch not in VERTICAL_QUOTE_OPEN_GLYPHS and ch not in VERTICAL_QUOTE_CLOSE_GLYPHS:
        return ch

    if ch not in VERTICAL_SYMMETRIC_QUOTE_GLYPHS:
        return VERTICAL_QUOTE_CLOSE_GLYPHS.get(ch, VERTICAL_QUOTE_OPEN_GLYPHS.get(ch, ch))

    prev_ch = _nearest_non_space(text, index, -1)
    next_ch = _nearest_non_space(text, index, 1)
    if (
        ch == "'"
        and prev_ch
        and next_ch
        and prev_ch.isascii()
        and next_ch.isascii()
        and prev_ch.isalnum()
        and next_ch.isalnum()
    ):
        return ch

    if quote_stack and quote_stack[-1] == ch:
        quote_stack.pop()
        return VERTICAL_QUOTE_CLOSE_GLYPHS[ch]

    if prev_ch and not next_ch:
        return VERTICAL_QUOTE_CLOSE_GLYPHS[ch]

    quote_stack.append(ch)
    return VERTICAL_QUOTE_OPEN_GLYPHS[ch]


def _normalize_vertical_quote_text(text: str) -> str:
    if not text:
        return text

    quote_stack: List[str] = []
    normalized = []
    for index, ch in enumerate(text):
        normalized.append(_vertical_quote_glyph(ch, text, index, quote_stack))
    return "".join(normalized)


def _is_vertical_quote_token(text: str) -> bool:
    return text in VERTICAL_QUOTE_TOKEN_INFO


def _can_attach_vertical_quote(unit: Dict) -> bool:
    return not (
        unit.get("vertical_space")
        or unit.get("draw_vertical_quote")
        or unit.get("draw_vertical_ellipsis")
        or unit.get("draw_vertical_dash")
    )


def _vertical_quote_payload(unit: Dict) -> Dict:
    return {
        "quote_role": unit.get("quote_role", "open"),
        "quote_kind": unit.get("quote_kind", "single"),
        "quote_width": float(unit.get("width", 1.0)),
        "quote_height": float(unit.get("height", 1.0)),
        "quote_stroke_width": float(unit.get("quote_stroke_width", 1.0)),
        "quote_text": unit.get("text_with_markers", ""),
    }


def _attach_vertical_quote_units(units: List[Dict]) -> List[Dict]:
    if not units:
        return units

    result: List[Dict] = []
    pending_open_quotes: List[Dict] = []

    def attach_to_previous(payload: Dict) -> bool:
        for target in reversed(result):
            if _can_attach_vertical_quote(target):
                target.setdefault("attached_vertical_quotes", []).append(payload)
                return True
        return False

    for unit in units:
        if unit.get("draw_vertical_quote"):
            payload = _vertical_quote_payload(unit)
            if payload.get("quote_role") == "close":
                if not attach_to_previous(payload):
                    pending_open_quotes.append(payload)
            else:
                pending_open_quotes.append(payload)
            continue

        if pending_open_quotes and _can_attach_vertical_quote(unit):
            unit.setdefault("attached_vertical_quotes", []).extend(pending_open_quotes)
            pending_open_quotes = []
        result.append(unit)

    if pending_open_quotes:
        for payload in pending_open_quotes:
            attach_to_previous(payload)

    return result


def _should_rotate_vertical_unit(text: str) -> bool:
    return bool(text) and all(ch in VERTICAL_ROTATED_GLYPHS for ch in text)


def _should_group_rotated_vertical_unit(text: str) -> bool:
    return bool(text) and all(ch in VERTICAL_GROUPED_ROTATED_GLYPHS for ch in text)


def _is_vertical_digit_char(ch: str) -> bool:
    try:
        return unicodedata.category(ch) == "Nd"
    except Exception:
        return False


def _is_vertical_digit_text(text: str) -> bool:
    return 0 < len(text) <= 3 and all(_is_vertical_digit_char(ch) for ch in text)


def _is_attached_vertical_punctuation_text(text: str) -> bool:
    return (
        bool(text)
        and len(text) <= 2
        and all(ch in VERTICAL_ATTACHED_PUNCTUATION for ch in text)
    )


def _is_dash_line_text(text: str) -> bool:
    return len(text) >= 2 and all(ch in DASH_LINE_CHARS for ch in text)


def _is_vertical_ellipsis_text(text: str) -> bool:
    if not text:
        return False
    if text in VERTICAL_ELLIPSIS_CHARS:
        return True
    if all(ch in VERTICAL_DOT_ELLIPSIS_CHARS for ch in text):
        return len(text) >= 2
    ellipsis_source_chars = VERTICAL_DOT_ELLIPSIS_CHARS | VERTICAL_ELLIPSIS_CHARS
    return all(ch in ellipsis_source_chars for ch in text) and any(
        ch in VERTICAL_ELLIPSIS_CHARS for ch in text
    )


def _vertical_ellipsis_dot_count(text: str) -> int:
    count = 0
    for ch in text:
        if ch in VERTICAL_DOT_ELLIPSIS_CHARS:
            count += 1
        elif ch == "\u2025":
            count += 2
        elif ch in VERTICAL_ELLIPSIS_CHARS:
            count += 3
    return count


def _strip_vertical_terminal_period(text: str) -> str:
    """Drop a single ASCII sentence-ending period only for vertical text layout."""
    if not text:
        return text

    end = len(text)
    while end > 0 and text[end - 1].isspace():
        end -= 1
    if end <= 0:
        return text

    period_end = end
    while period_end > 0 and text[period_end - 1] in VERTICAL_TRAILING_CLOSERS:
        period_end -= 1

    period_index = period_end - 1
    if period_index < 0 or text[period_index] not in VERTICAL_SENTENCE_PERIOD_CHARS:
        return text

    # Preserve ellipses and repeated dot effects such as "..." or "。。".
    if (
        period_index > 0
        and text[period_index - 1]
        in (VERTICAL_SENTENCE_PERIOD_CHARS | VERTICAL_ELLIPSIS_CHARS)
    ):
        return text

    return text[:period_index] + text[period_index + 1 :]


def _append_vertical_unit(
    units: List[Tuple[str, str]], text: str, style_name: str
) -> None:
    if len(text) == 1 and text in VERTICAL_DOT_ELLIPSIS_CHARS:
        text = VERTICAL_CENTERED_DOT_TOKEN
    units.append((text, style_name))


def _vertical_gap_size(
    font_size: int,
    line_spacing_mult: float,
    layout_scale: Optional[float] = 1.0,
) -> float:
    spacing_mult = max(0.01, float(line_spacing_mult or 1.0))
    base_gap_min = _scale_layout_px(VERTICAL_BASE_GAP_MIN, layout_scale)
    base_gap_max = _scale_layout_px(VERTICAL_BASE_GAP_MAX, layout_scale)
    extra_gap_min = _scale_layout_px(VERTICAL_EXTRA_GAP_MIN, layout_scale)
    extra_gap_max = _scale_layout_px(VERTICAL_EXTRA_GAP_MAX, layout_scale)
    base_gap = min(
        base_gap_max,
        max(base_gap_min, float(font_size) * VERTICAL_BASE_GAP_RATIO),
    )
    if spacing_mult <= 1.0:
        # At the WebUI minimum (0.5), remove almost all extra gap.
        gap = base_gap * max(0.0, min(1.0, (spacing_mult - 0.5) / 0.5))
    else:
        extra_gap_unit = min(
            extra_gap_max,
            max(
                extra_gap_min,
                (float(font_size) ** 0.5) * VERTICAL_EXTRA_GAP_SQRT_RATIO,
            ),
        )
        gap = base_gap + (spacing_mult - 1.0) * extra_gap_unit
    return max(0.0, gap)


def shape_line(
    text_line: str, hb_font: hb.Font, features: Dict[str, bool]
) -> Tuple[List[hb.GlyphInfo], List[hb.GlyphPosition], str]:
    """Shapes a line of text with HarfBuzz.

    Returns:
        Tuple of (glyph_infos, glyph_positions, direction) where direction
        is "ltr" or "rtl" as detected by HarfBuzz.

    Raises:
        RenderingError: If HarfBuzz shaping fails
    """
    hb_buffer = hb.Buffer()
    hb_buffer.add_str(text_line)
    hb_buffer.guess_segment_properties()
    direction = str(hb_buffer.direction)
    try:
        hb.shape(hb_font, hb_buffer, features)
        return hb_buffer.glyph_infos, hb_buffer.glyph_positions, direction
    except Exception as e:
        log_message(f"HarfBuzz shaping failed: {e}", always_print=True)
        raise RenderingError("HarfBuzz text shaping failed") from e


def calculate_line_width(positions: List[hb.GlyphPosition]) -> float:
    """Calculate visual width using advances and first/last x_offset."""
    if not positions:
        return 0.0
    HB_26_6_SCALE_FACTOR = 64.0

    total_advance_fixed = sum(pos.x_advance for pos in positions)
    first_offset_fixed = positions[0].x_offset
    last_offset_fixed = positions[-1].x_offset

    visual_width_fixed = total_advance_fixed + (last_offset_fixed - first_offset_fixed)
    visual_width = float(visual_width_fixed / HB_26_6_SCALE_FACTOR)
    return visual_width + VISUAL_WIDTH_EPSILON


def calculate_styled_line_width(
    line_with_markers: str,
    font_size: int,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features: Dict[str, bool],
    font_cmaps: Optional[Dict[str, set]] = None,
) -> float:
    """Calculate the width of a line that may contain style markers.

    Uses the appropriate HarfBuzz faces per style segment, falling back to
    the 'regular' face if a style-specific face is missing.
    """
    if not line_with_markers:
        return 0.0

    segments = parse_styled_segments(line_with_markers)
    if not segments:
        return 0.0

    regular_face = loaded_hb_faces.get("regular")
    if regular_face is None:
        return 0.0

    total_advance_fixed_all = 0
    first_offset_fixed_global: Optional[int] = None
    last_offset_fixed_global: Optional[int] = None

    for segment_text, style_name in segments:
        for run_text, font_key in split_text_by_font_support(
            segment_text, style_name, font_cmaps
        ):
            hb_face_to_use = loaded_hb_faces.get(font_key)
            if hb_face_to_use is None and style_name in (
                "regular",
                "italic",
                "bold",
                "bold_italic",
            ):
                hb_face_to_use = loaded_hb_faces.get(style_name)
            hb_face_to_use = hb_face_to_use or regular_face

            hb_font_segment = hb.Font(hb_face_to_use)
            hb_font_segment.ptem = float(font_size)
            # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
            hb_scale = int(font_size * 64)
            hb_font_segment.scale = (hb_scale, hb_scale)

            _, positions, _ = shape_line(run_text, hb_font_segment, features)
            if not positions:
                continue

            total_advance_fixed_all += sum(pos.x_advance for pos in positions)
            if first_offset_fixed_global is None:
                first_offset_fixed_global = positions[0].x_offset
            last_offset_fixed_global = positions[-1].x_offset

    if total_advance_fixed_all == 0 and first_offset_fixed_global is None:
        return 0.0

    HB_26_6_SCALE_FACTOR = 64.0
    offset_delta_fixed = 0
    if first_offset_fixed_global is not None and last_offset_fixed_global is not None:
        offset_delta_fixed = last_offset_fixed_global - first_offset_fixed_global

    visual_width_fixed_all = total_advance_fixed_all + offset_delta_fixed
    visual_width_all = float(visual_width_fixed_all / HB_26_6_SCALE_FACTOR)
    return visual_width_all + VISUAL_WIDTH_EPSILON


def _is_separator_or_space(ch: str) -> bool:
    try:
        category = unicodedata.category(ch)
    except Exception:
        return ch.isspace()
    return ch.isspace() or (len(category) > 0 and category[0] == "Z")


def _append_vertical_space_unit(
    units: List[Tuple[str, str]], style_name: str, count: int = 1
) -> None:
    """Append the requested number of visible gaps for vertical text layout.

    Normal vertical unit generation used to drop all whitespace.  That made
    Korean OSB vertical text such as "평화로운 농장에" render as if it were
    "평화로운농장에".  Keep whitespace as a non-drawing layout unit so word
    boundaries and user-entered repeated spaces remain visible.
    """
    for _ in range(max(0, int(count))):
        units.append((VERTICAL_SPACE_TOKEN, style_name))


def _iter_vertical_units(text: str) -> List[Tuple[str, str]]:
    units: List[Tuple[str, str]] = []
    pending_space_count = 0
    pending_space_style = "regular"
    styled_segments = [
        (_normalize_vertical_quote_text(segment_text), style_name)
        for segment_text, style_name in parse_styled_segments(text)
    ]

    for idx in range(len(styled_segments) - 1, -1, -1):
        segment_text, style_name = styled_segments[idx]
        if not segment_text.strip():
            continue
        stripped_text = _strip_vertical_terminal_period(segment_text)
        if stripped_text != segment_text:
            styled_segments[idx] = (stripped_text, style_name)
        break

    for segment_text, style_name in styled_segments:
        current = ""
        current_is_punctuation = False
        current_is_grouped_rotated = False
        current_is_dash_line = False
        current_is_digit = False
        current_digit_overflow = False
        for ch in segment_text:
            if _is_separator_or_space(ch):
                if current:
                    _append_vertical_unit(units, current, style_name)
                    current = ""
                    current_is_punctuation = False
                    current_is_grouped_rotated = False
                    current_is_dash_line = False
                    current_is_digit = False
                    current_digit_overflow = False
                if units:
                    pending_space_count += 1
                    pending_space_style = style_name
                continue

            if pending_space_count:
                _append_vertical_space_unit(
                    units,
                    pending_space_style,
                    pending_space_count,
                )
                pending_space_count = 0

            if unicodedata.combining(ch) and current:
                current += ch
                continue
            ch_is_digit = _is_vertical_digit_char(ch)
            if current and current_is_digit and ch_is_digit:
                candidate = current + ch
                if not current_digit_overflow and _is_vertical_digit_text(candidate):
                    current = candidate
                else:
                    for digit_ch in current:
                        _append_vertical_unit(units, digit_ch, style_name)
                    current = ch
                    current_digit_overflow = True
                current_is_punctuation = False
                current_is_grouped_rotated = False
                current_is_dash_line = False
                current_is_digit = True
                continue
            if ch_is_digit:
                if current:
                    _append_vertical_unit(units, current, style_name)
                current = ch
                current_is_punctuation = False
                current_is_grouped_rotated = False
                current_is_dash_line = False
                current_is_digit = True
                current_digit_overflow = False
                continue
            ch_is_dash_line = ch in DASH_LINE_CHARS
            if current and current_is_dash_line and ch_is_dash_line:
                current += ch
                continue
            if ch_is_dash_line:
                if current:
                    _append_vertical_unit(units, current, style_name)
                current = ch
                current_is_punctuation = False
                current_is_grouped_rotated = False
                current_is_dash_line = True
                current_is_digit = False
                current_digit_overflow = False
                continue
            ch_is_punctuation = ch in VERTICAL_GROUPED_PUNCTUATION
            ch_is_grouped_rotated = ch in VERTICAL_GROUPED_ROTATED_GLYPHS
            if current and current_is_punctuation and ch_is_punctuation:
                candidate = current + ch
                if _is_vertical_ellipsis_text(candidate):
                    current = candidate
                elif (
                    len(candidate) <= 2
                    and all(p in VERTICAL_ATTACHED_PUNCTUATION for p in candidate)
                ):
                    current = candidate
                else:
                    _append_vertical_unit(units, current, style_name)
                    current = ch
                continue
            if (
                current
                and current_is_grouped_rotated
                and ch_is_grouped_rotated
                and _should_group_rotated_vertical_unit(current + ch)
            ):
                current += ch
                continue
            if current:
                _append_vertical_unit(units, current, style_name)
            current = ch
            current_is_punctuation = ch_is_punctuation
            current_is_grouped_rotated = ch_is_grouped_rotated
            current_is_dash_line = False
            current_is_digit = False
            current_digit_overflow = False
        if current:
            _append_vertical_unit(units, current, style_name)

    return units


def _measure_vertical_unit(
    text: str,
    style_name: str,
    font_size: int,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features: Dict[str, bool],
    font_cmaps: Optional[Dict[str, set]] = None,
    layout_scale: Optional[float] = 1.0,
) -> Optional[Dict]:
    if text == VERTICAL_SPACE_TOKEN:
        space_min = _scale_layout_px(VERTICAL_WORD_SPACE_MIN, layout_scale)
        space_max = _scale_layout_px(VERTICAL_WORD_SPACE_MAX, layout_scale)
        space_height = min(
            space_max,
            max(space_min, float(font_size) * VERTICAL_WORD_SPACE_RATIO),
        )
        space_width = max(1.0, float(font_size) * 0.08)
        return {
            "text_with_markers": "",
            "style": style_name,
            "width": space_width,
            "height": space_height,
            "advance_height": space_height,
            "baseline_offset_y": space_height / 2.0,
            "left": -space_width / 2.0,
            "right": space_width / 2.0,
            "top": -space_height / 2.0,
            "bottom": space_height / 2.0,
            "vertical_space": True,
        }

    if _is_vertical_ellipsis_text(text):
        dot_count = _vertical_ellipsis_dot_count(text)
        dot_radius = max(0.95, float(font_size) * 0.065)
        dot_gap = max(1.25, float(font_size) * 0.15)
        dot_step = dot_radius * 2.0 + dot_gap
        visual_width = dot_radius * 2.0
        visual_height = dot_radius * 2.0 * dot_count + dot_gap * (dot_count - 1)
        return {
            "text_with_markers": text,
            "style": style_name,
            "width": visual_width,
            "height": visual_height,
            "advance_height": visual_height,
            "baseline_offset_y": visual_height / 2.0,
            "left": -visual_width / 2.0,
            "right": visual_width / 2.0,
            "top": -visual_height / 2.0,
            "bottom": visual_height / 2.0,
            "draw_vertical_ellipsis": True,
            "ellipsis_dot_count": dot_count,
            "ellipsis_dot_radius": dot_radius,
            "ellipsis_dot_step": dot_step,
        }

    if text == VERTICAL_CENTERED_DOT_TOKEN:
        dot_radius = max(0.85, float(font_size) * 0.07)
        visual_width = dot_radius * 2.0
        visual_height = max(dot_radius * 2.0, float(font_size) * 0.28)
        return {
            "text_with_markers": ".",
            "style": style_name,
            "width": visual_width,
            "height": visual_height,
            "advance_height": visual_height,
            "baseline_offset_y": visual_height / 2.0,
            "left": -visual_width / 2.0,
            "right": visual_width / 2.0,
            "top": -visual_height / 2.0,
            "bottom": visual_height / 2.0,
            "draw_vertical_centered_dot": True,
            "centered_dot_radius": dot_radius,
        }

    if _is_dash_line_text(text):
        dash_thickness = max(1.2, float(font_size) * 0.075)
        dash_length = max(float(font_size) * 0.95, float(font_size) * 0.78 * len(text))
        return {
            "text_with_markers": text,
            "style": style_name,
            "width": dash_thickness,
            "height": dash_length,
            "advance_height": dash_length,
            "baseline_offset_y": dash_length / 2.0,
            "left": -dash_thickness / 2.0,
            "right": dash_thickness / 2.0,
            "top": -dash_length / 2.0,
            "bottom": dash_length / 2.0,
            "draw_vertical_dash": True,
            "dash_thickness": dash_thickness,
            "dash_length": dash_length,
        }

    if _is_vertical_quote_token(text):
        quote_role, quote_kind = VERTICAL_QUOTE_TOKEN_INFO[text]
        quote_cell_width = max(7.0, float(font_size) * 0.58)
        quote_cell_height = max(6.8, float(font_size) * 0.59)
        quote_draw_width = max(4.5, float(font_size) * 0.36)
        quote_draw_height = max(5.2, float(font_size) * 0.42)
        quote_stroke_width = max(1.15, float(font_size) * 0.062)
        quote_padding_y = max(0.32, float(font_size) * 0.028)
        quote_x_offset = float(font_size) * (-0.23 if quote_role == "open" else 0.23)
        quote_y_offset = (
            -((quote_cell_height - quote_draw_height) / 2.0) + quote_padding_y
            if quote_role == "open"
            else ((quote_cell_height - quote_draw_height) / 2.0) - quote_padding_y
        )
        return {
            "text_with_markers": VERTICAL_QUOTE_DISPLAY_GLYPHS[text],
            "style": style_name,
            "width": quote_cell_width,
            "height": quote_cell_height,
            "advance_height": quote_cell_height,
            "baseline_offset_y": quote_cell_height / 2.0,
            "left": -quote_cell_width / 2.0,
            "right": quote_cell_width / 2.0,
            "top": -quote_cell_height / 2.0,
            "bottom": quote_cell_height / 2.0,
            "draw_vertical_quote": True,
            "quote_role": quote_role,
            "quote_kind": quote_kind,
            "quote_draw_width": quote_draw_width,
            "quote_draw_height": quote_draw_height,
            "quote_stroke_width": quote_stroke_width,
            "quote_x_offset": quote_x_offset,
            "quote_y_offset": quote_y_offset,
        }

    preferred_font_key = style_name
    if text:
        preferred_font_key = pick_font_key_for_char(text[0], style_name, font_cmaps)

    hb_face = (
        loaded_hb_faces.get(preferred_font_key)
        or loaded_hb_faces.get(style_name)
        or loaded_hb_faces.get("regular")
    )
    if hb_face is None:
        return None

    hb_font = hb.Font(hb_face)
    hb_font.ptem = float(font_size)
    hb_scale = int(font_size * 64)
    hb_font.scale = (hb_scale, hb_scale)

    infos, positions, _ = shape_line(text, hb_font, features)
    if not infos or not positions:
        return None

    cursor_x = 0
    left = float("inf")
    right = float("-inf")
    top = float("inf")
    bottom = float("-inf")
    vertical_advances = []
    first_vertical_origin_y: Optional[float] = None

    for info, pos in zip(infos, positions):
        extents = hb_font.get_glyph_extents(info.codepoint)
        origin_x = cursor_x + pos.x_offset
        origin_y = -pos.y_offset

        if extents:
            glyph_x1 = origin_x + extents.x_bearing
            glyph_x2 = glyph_x1 + extents.width
            glyph_y1 = origin_y - extents.y_bearing
            glyph_y2 = origin_y - (extents.y_bearing + extents.height)
        else:
            glyph_x1 = origin_x
            glyph_x2 = origin_x + pos.x_advance
            glyph_y1 = origin_y - font_size * 64
            glyph_y2 = origin_y

        left = min(left, glyph_x1, glyph_x2)
        right = max(right, glyph_x1, glyph_x2)
        top = min(top, glyph_y1, glyph_y2)
        bottom = max(bottom, glyph_y1, glyph_y2)
        cursor_x += pos.x_advance

        vertical_advance = hb_font.get_glyph_v_advance(info.codepoint)
        if vertical_advance:
            vertical_advances.append(abs(vertical_advance) / HB_26_6_SCALE_FACTOR)

        if first_vertical_origin_y is None:
            vertical_origin = hb_font.get_glyph_v_origin(info.codepoint)
            if vertical_origin is not None:
                _, vertical_origin_y = vertical_origin
                first_vertical_origin_y = vertical_origin_y / HB_26_6_SCALE_FACTOR

    if left == float("inf") or right == float("-inf"):
        return None

    original_left = left / HB_26_6_SCALE_FACTOR
    original_right = right / HB_26_6_SCALE_FACTOR
    original_top = top / HB_26_6_SCALE_FACTOR
    original_bottom = bottom / HB_26_6_SCALE_FACTOR
    advance_width = max(1.0, cursor_x / HB_26_6_SCALE_FACTOR)
    visual_width = max(1.0, original_right - original_left)
    visual_height = max(1.0, original_bottom - original_top)
    center_offset = (original_left + original_right) / 2.0
    advance_height = max(vertical_advances) if vertical_advances else 0.0
    if advance_height <= 0:
        advance_height = max(visual_height, float(font_size))

    rotate_in_vertical = _should_rotate_vertical_unit(text)
    if rotate_in_vertical:
        rotated_width = visual_height
        rotated_height = visual_width
        return {
            "text_with_markers": text,
            "style": style_name,
            "font_key": preferred_font_key,
            "width": rotated_width,
            "height": rotated_height,
            "advance_height": max(advance_height, rotated_height),
            "baseline_offset_y": rotated_height / 2.0,
            "left": -rotated_width / 2.0,
            "right": rotated_width / 2.0,
            "top": -rotated_height / 2.0,
            "bottom": rotated_height / 2.0,
            "rotate_in_vertical": True,
            "rotation_degrees": VERTICAL_ROTATION_DEGREES,
            "advance_width": advance_width,
            "original_left": original_left,
            "original_right": original_right,
            "original_top": original_top,
            "original_bottom": original_bottom,
        }

    if first_vertical_origin_y is None:
        first_vertical_origin_y = -original_top + max(
            0.0, (advance_height - visual_height) / 2.0
        )

    return {
        "text_with_markers": text,
        "style": style_name,
        "font_key": preferred_font_key,
        "width": visual_width,
        "height": visual_height,
        "advance_height": advance_height,
        "baseline_offset_y": first_vertical_origin_y,
        "left": original_left,
        "right": original_right,
        "top": original_top,
        "bottom": original_bottom,
        "advance_width": advance_width,
        "center_offset": center_offset,
        "center_on_column": (
            _is_attached_vertical_punctuation_text(text)
            or _is_vertical_digit_text(text)
        ),
        "horizontal_in_vertical": _is_vertical_digit_text(text),
    }


def _build_vertical_layout(
    text: str,
    font_size: int,
    max_render_width: float,
    max_render_height: float,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    line_spacing_mult: float,
    metrics,
    font_cmaps: Optional[Dict[str, set]] = None,
    column_gap_mult: float = 1.0,
    layout_scale: Optional[float] = 1.0,
) -> Optional[Dict]:
    # In vertical OSB layout, explicit newlines are hard column breaks.
    # Example: "A\nB" becomes two vertical columns instead of one continuous
    # stack. Columns flow right-to-left, which matches Japanese-style vertical
    # manga text.
    raw_columns = text.split("\n") if "\n" in text else [text]
    column_layouts: List[Dict] = []
    gap = _vertical_gap_size(font_size, line_spacing_mult, layout_scale)

    for raw_column in raw_columns:
        column_text = raw_column.strip()
        if not column_text:
            continue

        column_units = []
        for unit_text, style_name in _iter_vertical_units(column_text):
            unit = _measure_vertical_unit(
                unit_text,
                style_name,
                font_size,
                loaded_hb_faces,
                features_to_enable,
                font_cmaps,
                layout_scale,
            )
            if unit is None:
                continue
            column_units.append(unit)

        if not column_units:
            continue

        center_offsets = [
            float(unit["center_offset"])
            for unit in column_units
            if not unit.get("rotate_in_vertical")
            and not unit.get("draw_vertical_ellipsis")
            and not unit.get("draw_vertical_centered_dot")
            and not unit.get("draw_vertical_dash")
            and not unit.get("draw_vertical_quote")
            and not unit.get("vertical_space")
            and not unit.get("center_on_column")
            and "center_offset" in unit
        ]
        shared_center_offset = (
            float(np.median(center_offsets)) if center_offsets else None
        )
        if shared_center_offset is not None:
            for unit in column_units:
                if (
                    unit.get("rotate_in_vertical")
                    or unit.get("draw_vertical_ellipsis")
                    or unit.get("draw_vertical_centered_dot")
                    or unit.get("draw_vertical_dash")
                    or unit.get("draw_vertical_quote")
                    or unit.get("vertical_space")
                    or unit.get("center_on_column")
                ):
                    continue
                unit["alignment_center_offset"] = shared_center_offset
                aligned_left = float(unit.get("left", 0.0)) - shared_center_offset
                aligned_right = float(
                    unit.get("right", unit.get("width", 0.0))
                ) - shared_center_offset
                unit["width"] = max(
                    float(unit.get("width", 1.0)),
                    2.0 * max(abs(aligned_left), abs(aligned_right), 0.5),
                )
        for unit in column_units:
            if not unit.get("center_on_column") or "center_offset" not in unit:
                continue
            center_offset = float(unit["center_offset"])
            unit["alignment_center_offset"] = center_offset
            aligned_left = float(unit.get("left", 0.0)) - center_offset
            aligned_right = float(unit.get("right", unit.get("width", 0.0))) - center_offset
            unit["width"] = max(
                float(unit.get("width", 1.0)),
                2.0 * max(abs(aligned_left), abs(aligned_right), 0.5),
            )

        current_y = 0.0
        last_index = len(column_units) - 1
        for index, unit in enumerate(column_units):
            is_vertical_space = bool(unit.get("vertical_space"))
            unit["origin_y"] = current_y - unit["top"]
            next_unit = column_units[index + 1] if index != last_index else None
            quote_adjacent = bool(
                unit.get("draw_vertical_quote")
                or (next_unit is not None and next_unit.get("draw_vertical_quote"))
            )
            gap_after = gap * (0.22 if quote_adjacent else 1.0)
            unit["advance_height"] = unit["height"] + (
                0.0
                if (
                    index == last_index
                    or is_vertical_space
                )
                else gap_after
            )
            current_y += unit["height"]
            if (
                index != last_index
                and not is_vertical_space
            ):
                current_y += gap_after

        column_layouts.append(
            {
                "units": column_units,
                "height": max(1.0, current_y),
                "width": max(unit["width"] for unit in column_units),
            }
        )

    if not column_layouts:
        return None

    block_height = max(column["height"] for column in column_layouts)
    try:
        gap_mult = max(0.1, float(column_gap_mult or 1.0))
    except (TypeError, ValueError):
        gap_mult = 1.0
    column_gap = min(
        _scale_layout_px(VERTICAL_COLUMN_GAP_MAX, layout_scale),
        max(
            _scale_layout_px(VERTICAL_COLUMN_GAP_MIN, layout_scale),
            float(font_size) * VERTICAL_COLUMN_GAP_RATIO,
        ),
    ) * gap_mult
    block_width = sum(column["width"] for column in column_layouts) + column_gap * (
        len(column_layouts) - 1
    )

    if block_width > max_render_width or block_height > max_render_height:
        return None

    # Layout columns as a block centered on target_center_x. Column 0 is the
    # rightmost column, then each explicit newline moves one column to the left.
    lines_data_at_size: List[Dict] = []
    right_edge = block_width / 2.0
    cursor_right = right_edge
    for column in column_layouts:
        column_width = column["width"]
        column_center_x = cursor_right - column_width / 2.0
        column_y_offset = max(0.0, (block_height - column["height"]) / 2.0)
        for unit in column["units"]:
            unit["origin_x"] = column_center_x
            unit["origin_y"] += column_y_offset
            lines_data_at_size.append(unit)
        cursor_right -= column_width + column_gap

    return {
        "lines": lines_data_at_size,
        "metrics": metrics,
        "max_line_width": block_width,
        "line_height": 0.0,
        "block_height": block_height,
        "block_width": block_width,
        "orientation": "vertical",
    }


def check_fit(
    font_size: int,
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    line_spacing_mult: float,
    hyphenate_before_scaling: bool,
    hyphen_penalty: float,
    hyphenation_min_word_length: int,
    badness_exponent: float,
    word_width_cache: Optional[Dict[Tuple[str, int], float]] = None,
    verbose: bool = False,
    detach_trailing_punctuation: bool = True,
    vertical_stack: bool = False,
    font_cmaps: Optional[Dict[str, set]] = None,
    vertical_column_gap_mult: float = 1.0,
    protect_short_wrapped_symbols: bool = True,
    layout_scale: Optional[float] = 1.0,
) -> Optional[Dict]:
    """Check if text fits within the given dimensions at the specified font size.

    Args:
        font_size: Font size to test
        text: Text to wrap and measure
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for shaping
        regular_typeface: Skia typeface for metrics
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before scaling
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        word_width_cache: Optional cache for word widths
        verbose: Whether to print detailed logs

    Returns:
        Dict containing fit data if successful, None if doesn't fit
    """
    try:
        hb_font = hb.Font(regular_hb_face)
        hb_font.ptem = float(font_size)

        # Standard HarfBuzz scaling: font_size * 64 (for 26.6 fixed point coordinates)
        hb_scale = int(font_size * 64)
        hb_font.scale = (hb_scale, hb_scale)

        skia_font_test = skia.Font(regular_typeface, font_size)
        try:
            metrics = skia_font_test.getMetrics()
            single_line_height = (
                -metrics.fAscent + metrics.fDescent + metrics.fLeading
            ) * line_spacing_mult
            if single_line_height <= 0:
                single_line_height = font_size * 1.2 * line_spacing_mult
        except Exception as e:
            if verbose:
                log_message(
                    f"Font metrics unavailable at size {font_size}: {e}",
                    verbose=verbose,
                )
            single_line_height = font_size * 1.2 * line_spacing_mult

        short_text_max_lines = _short_text_max_lines(text, vertical_stack)

        if vertical_stack:
            return _build_vertical_layout(
                text,
                font_size,
                max_render_width,
                max_render_height,
                loaded_hb_faces,
                features_to_enable,
                line_spacing_mult,
                metrics,
                font_cmaps,
                vertical_column_gap_mult,
                layout_scale,
            )

        # Respect explicit newlines as hard line breaks
        if "\n" in text:
            explicit_lines = text.split("\n")
            current_max_line_width = 0.0
            lines_data_at_size = []
            for line_text in explicit_lines:
                width = calculate_styled_line_width(
                    line_text, font_size, loaded_hb_faces, features_to_enable, font_cmaps
                )
                if not line_text or width <= max_render_width:
                    wrapped_line_data = [
                        {"text_with_markers": line_text, "width": width}
                    ]
                else:
                    wrapped_fit = check_fit(
                        font_size,
                        line_text,
                        max_render_width,
                        max_render_height,
                        regular_hb_face,
                        regular_typeface,
                        loaded_hb_faces,
                        features_to_enable,
                        line_spacing_mult,
                        hyphenate_before_scaling,
                        hyphen_penalty,
                        hyphenation_min_word_length,
                        badness_exponent,
                        word_width_cache,
                        verbose,
                        detach_trailing_punctuation,
                        False,
                        font_cmaps,
                        vertical_column_gap_mult,
                        protect_short_wrapped_symbols,
                        layout_scale,
                    )
                    if wrapped_fit is None:
                        return None
                    wrapped_line_data = wrapped_fit["lines"]

                lines_data_at_size.extend(wrapped_line_data)
                for line_data in wrapped_line_data:
                    current_max_line_width = max(
                        current_max_line_width,
                        float(line_data.get("width", 0.0)),
                    )

            if (
                short_text_max_lines is not None
                and len(lines_data_at_size) > short_text_max_lines
            ):
                return None
            if not _short_words_fit_line_limit(
                text,
                [line.get("text_with_markers", "") for line in lines_data_at_size],
            ):
                return None

            total_block_height = (-metrics.fAscent + metrics.fDescent) + (
                len(lines_data_at_size) - 1
            ) * single_line_height

            if (
                current_max_line_width <= max_render_width
                and total_block_height <= max_render_height
            ):
                return {
                    "lines": lines_data_at_size,
                    "metrics": metrics,
                    "max_line_width": current_max_line_width,
                    "line_height": single_line_height,
                    "block_height": total_block_height,
                }
            return None

        tokens: List[Tuple[str, bool]] = tokenize_styled_text(
            text, detach_trailing_punctuation
        )
        augmented_tokens: List[str] = []

        if hyphenate_before_scaling:
            for token_text, is_styled in tokens:
                marker = ""
                core_text = token_text

                if is_styled:
                    styled_match = STYLE_PATTERN.match(token_text)
                    if not styled_match:
                        augmented_tokens.append(token_text)
                        continue
                    marker = styled_match.group(1)
                    core_text = styled_match.group(2)

                core_word_length = hyphenation_token_length(core_text)

                if core_word_length >= hyphenation_min_word_length:
                    word_width = calculate_styled_line_width(
                        token_text, font_size, loaded_hb_faces, features_to_enable, font_cmaps
                    )

                    if word_width > max_render_width:

                        def wrap_part(part: str) -> str:
                            no_space_before = part.startswith(NO_SPACE_BEFORE_MARKER)
                            clean_part = strip_no_space_before_marker(part)
                            wrapped = (
                                f"{marker}{clean_part}{marker}"
                                if marker
                                else clean_part
                            )
                            if no_space_before:
                                return f"{NO_SPACE_BEFORE_MARKER}{wrapped}"
                            return wrapped

                        def width_test_func(part: str) -> bool:
                            wrapped = wrap_part(part)
                            w = calculate_styled_line_width(
                                wrapped, font_size, loaded_hb_faces, features_to_enable, font_cmaps
                            )
                            return w <= max_render_width

                        split_parts = split_hangul_word_for_wrapping(core_text)
                        if split_parts is None:
                            split_parts = try_hyphenate_word(
                                core_text,
                                hyphenation_min_word_length,
                                width_test_func,
                            )

                        if split_parts:
                            augmented_tokens.extend(wrap_part(p) for p in split_parts)
                        else:
                            augmented_tokens.append(token_text)
                    else:
                        augmented_tokens.append(token_text)
                else:
                    augmented_tokens.append(token_text)
        else:
            augmented_tokens = [t for t, _ in tokens]

        try:
            GLUE_TRAILING_PUNCT_RE = re.compile(r"^[,.;:!?…]+$")
            GLUE_CLOSERS_RE = re.compile(r"^[\)\]\}\>\u2019\u201D\'\"]+$")

            def _glue_trailing_punctuation(
                tokens_list: List[str], _detach: bool = True
            ) -> List[str]:
                glued: List[str] = []
                for tok in tokens_list:
                    clean_tok = strip_no_space_before_marker(tok)
                    match = STYLE_PATTERN.match(clean_tok)
                    content = match.group(2) if match else clean_tok

                    if _detach and is_detached_trailing_punctuation(content):
                        glued.append(tok)
                        continue

                    if glued and (
                        GLUE_TRAILING_PUNCT_RE.match(content)
                        or GLUE_CLOSERS_RE.match(content)
                    ):
                        glued[-1] = glued[-1] + clean_tok
                    else:
                        glued.append(tok)
                return glued

            augmented_tokens = _glue_trailing_punctuation(
                augmented_tokens, detach_trailing_punctuation
            )
            augmented_tokens = glue_korean_postposition_particles(augmented_tokens)
        except Exception:
            pass

        def word_width_func(word: str) -> float:
            if word_width_cache is not None:
                cached_key = (word, font_size)
                if cached_key in word_width_cache:
                    return word_width_cache[cached_key]

            clean_word = restore_explicit_spaces(
                strip_no_space_before_marker(word)
            )
            width_val = calculate_styled_line_width(
                clean_word, font_size, loaded_hb_faces, features_to_enable, font_cmaps
            )

            if word_width_cache is not None:
                word_width_cache[(word, font_size)] = width_val

            return width_val

        space_width = calculate_styled_line_width(
            " ", font_size, loaded_hb_faces, features_to_enable, font_cmaps
        )

        wrapped_lines_text = find_optimal_breaks_dp(
            augmented_tokens,
            max_render_width,
            word_width_func,
            space_width,
            badness_exponent,
            hyphen_penalty,
            detach_trailing_punctuation,
            protect_short_wrapped_symbols,
        )

        if not wrapped_lines_text:
            return None
        if (
            short_text_max_lines is not None
            and len(wrapped_lines_text) > short_text_max_lines
        ):
            return None
        if not _short_words_fit_line_limit(text, wrapped_lines_text):
            return None

        current_max_line_width = 0
        lines_data_at_size = []
        for line_text_with_markers in wrapped_lines_text:
            line_text_with_markers = restore_explicit_spaces(
                line_text_with_markers
            )
            width = calculate_styled_line_width(
                line_text_with_markers, font_size, loaded_hb_faces, features_to_enable, font_cmaps
            )
            lines_data_at_size.append(
                {"text_with_markers": line_text_with_markers, "width": width}
            )
            current_max_line_width = max(current_max_line_width, width)

        total_block_height = (-metrics.fAscent + metrics.fDescent) + (
            len(wrapped_lines_text) - 1
        ) * single_line_height

        if verbose:
            log_message(
                f"Size {font_size}: {current_max_line_width:.0f}x{total_block_height:.0f} "
                f"(max {max_render_width:.0f}x{max_render_height:.0f})",
                verbose=verbose,
            )

        if (
            current_max_line_width <= max_render_width
            and total_block_height <= max_render_height
        ):
            if verbose:
                log_message(f"Size {font_size} fits", verbose=verbose)
            return {
                "lines": lines_data_at_size,
                "metrics": metrics,
                "max_line_width": current_max_line_width,
                "line_height": single_line_height,
                "block_height": total_block_height,
            }

        return None

    except Exception as e:
        if verbose:
            log_message(f"Fit check failed at size {font_size}: {e}", verbose=verbose)
        return None


def _check_collision(
    lines_data: List[Dict],
    box_top_left: Tuple[int, int],
    cleaned_mask: np.ndarray,
    line_height: float,
    render_size: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
) -> bool:
    """
    Check if any text pixel overlaps with background (0) in mask.

    Args:
        lines_data: List of dictionaries containing line width and text.
        box_top_left: (x, y) coordinates of the bounding box top-left corner.
        cleaned_mask: Binary mask of the bubble (0=background, 255=bubble).
        line_height: Height of a single line of text.
        render_size: (width, height) of the render box.

    Returns:
        True if collision detected, False otherwise.
    """
    offset_x, offset_y = offset
    box_x = float(box_top_left[0]) + float(offset_x)
    box_y = float(box_top_left[1]) + float(offset_y)
    mask_h, mask_w = cleaned_mask.shape
    max_w, max_h = render_size

    total_text_height = len(lines_data) * line_height
    start_y = box_y + (max_h - total_text_height) / 2

    current_y = start_y
    for line in lines_data:
        line_w = line["width"]
        line_x = box_x + (max_w - line_w) / 2

        y1, y2 = int(current_y), int(current_y + line_height)
        x1, x2 = int(line_x), int(line_x + line_w)

        points_to_check = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

        for px, py in points_to_check:
            px = max(0, min(px, mask_w - 1))
            py = max(0, min(py, mask_h - 1))

            if cleaned_mask[py, px] == 0:
                return True

        current_y += line_height

    return False


def _collision_details(
    lines_data: List[Dict],
    box_top_left: Tuple[int, int],
    cleaned_mask: np.ndarray,
    line_height: float,
    render_size: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
) -> List[Dict]:
    """Return line-corner collisions using the same geometry as _check_collision."""
    offset_x, offset_y = offset
    box_x = float(box_top_left[0]) + float(offset_x)
    box_y = float(box_top_left[1]) + float(offset_y)
    mask_h, mask_w = cleaned_mask.shape
    max_w, max_h = render_size

    total_text_height = len(lines_data) * line_height
    start_y = box_y + (max_h - total_text_height) / 2

    collisions: List[Dict] = []
    current_y = start_y
    for line_index, line in enumerate(lines_data):
        line_w = line["width"]
        line_x = box_x + (max_w - line_w) / 2

        y1, y2 = int(current_y), int(current_y + line_height)
        x1, x2 = int(line_x), int(line_x + line_w)

        points_to_check = [
            ("LT", x1, y1),
            ("RT", x2, y1),
            ("LB", x1, y2),
            ("RB", x2, y2),
        ]

        for corner, px, py in points_to_check:
            clamped_x = max(0, min(px, mask_w - 1))
            clamped_y = max(0, min(py, mask_h - 1))
            if cleaned_mask[clamped_y, clamped_x] == 0:
                collisions.append(
                    {
                        "line_index": line_index,
                        "line_text": line.get("text_with_markers", ""),
                        "corner": corner,
                        "x": px,
                        "y": py,
                    }
                )

        current_y += line_height

    return collisions


def _collision_stats(
    lines_data: List[Dict],
    box_top_left: Tuple[int, int],
    cleaned_mask: np.ndarray,
    line_height: float,
    render_size: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
) -> Dict:
    """Measure how much the line rectangles spill outside the bubble mask."""
    offset_x, offset_y = offset
    box_x = float(box_top_left[0]) + float(offset_x)
    box_y = float(box_top_left[1]) + float(offset_y)
    mask_h, mask_w = cleaned_mask.shape
    max_w, max_h = render_size

    total_text_height = len(lines_data) * line_height
    start_y = box_y + (max_h - total_text_height) / 2

    outside_pixels = 0
    total_pixels = 0
    current_y = start_y
    for line in lines_data:
        line_w = float(line["width"])
        line_x = box_x + (max_w - line_w) / 2

        y1, y2 = int(current_y), int(current_y + line_height)
        x1, x2 = int(line_x), int(line_x + line_w)

        clipped_x1 = max(0, min(x1, mask_w))
        clipped_x2 = max(0, min(x2 + 1, mask_w))
        clipped_y1 = max(0, min(y1, mask_h))
        clipped_y2 = max(0, min(y2 + 1, mask_h))

        rect_width = max(0, x2 - x1 + 1)
        rect_height = max(0, y2 - y1 + 1)
        rect_pixels = rect_width * rect_height
        total_pixels += rect_pixels

        if rect_pixels <= 0:
            current_y += line_height
            continue

        clipped_pixels = max(0, clipped_x2 - clipped_x1) * max(0, clipped_y2 - clipped_y1)
        outside_pixels += rect_pixels - clipped_pixels
        if clipped_pixels > 0:
            region = cleaned_mask[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
            outside_pixels += int(np.count_nonzero(region == 0))

        current_y += line_height

    corner_collisions = _collision_details(
        lines_data,
        box_top_left,
        cleaned_mask,
        line_height,
        render_size,
        offset,
    )
    ratio = (outside_pixels / total_pixels) if total_pixels > 0 else 1.0
    return {
        "has_collision": bool(corner_collisions),
        "corner_count": len(corner_collisions),
        "outside_pixels": outside_pixels,
        "total_pixels": total_pixels,
        "outside_ratio": ratio,
        "corner_collisions": corner_collisions,
    }


def _collision_is_small(stats: Dict, ratio_limit: float, corner_limit: int) -> bool:
    return (
        bool(stats.get("has_collision"))
        and int(stats.get("corner_count", 0)) <= int(corner_limit)
        and float(stats.get("outside_ratio", 1.0)) <= float(ratio_limit)
    )


def _find_collision_shift(
    lines_data: List[Dict],
    box_top_left: Tuple[int, int],
    cleaned_mask: np.ndarray,
    line_height: float,
    render_size: Tuple[float, float],
    layout_scale: Optional[float] = 1.0,
) -> Optional[Tuple[float, float]]:
    """Find a small center offset that clears minor mask-corner collisions."""
    max_w, max_h = render_size
    scale = _normalize_layout_scale(layout_scale)
    max_shift = int(
        round(
            min(
                COLLISION_SHIFT_BASE_MAX_PX * scale,
                max(1.0, float(max_w)) * COLLISION_SHIFT_MAX_SIDE_RATIO,
                max(1.0, float(max_h)) * COLLISION_SHIFT_MAX_SIDE_RATIO,
            )
        )
    )
    max_shift = max(1, max_shift)

    candidates: List[Tuple[int, int]] = []
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            candidates.append((dx, dy))
    candidates.sort(key=lambda item: (abs(item[0]) + abs(item[1]), abs(item[1]), abs(item[0])))

    best_offset: Optional[Tuple[float, float]] = None
    best_score: Optional[Tuple[float, int, int]] = None
    for dx, dy in candidates:
        stats = _collision_stats(
            lines_data,
            box_top_left,
            cleaned_mask,
            line_height,
            render_size,
            (float(dx), float(dy)),
        )
        if stats["has_collision"]:
            continue

        score = (
            float(stats.get("outside_ratio", 1.0)),
            abs(dx) + abs(dy),
            abs(dy),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_offset = (float(dx), float(dy))

    return best_offset


def find_optimal_layout(
    text: str,
    max_render_width: float,
    max_render_height: float,
    regular_hb_face: hb.Face,
    regular_typeface: skia.Typeface,
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    features_to_enable: Dict[str, bool],
    min_font_size: int = 8,
    max_font_size: int = 16,
    line_spacing_mult: float = 1.0,
    hyphenate_before_scaling: bool = True,
    hyphen_penalty: float = 1000.0,
    hyphenation_min_word_length: int = 8,
    badness_exponent: float = 3.0,
    verbose: bool = False,
    bubble_id: Optional[str] = None,
    cleaned_mask: Optional[np.ndarray] = None,
    box_top_left: Optional[Tuple[int, int]] = None,
    detach_trailing_punctuation: bool = True,
    vertical_stack: bool = False,
    font_cmaps: Optional[Dict[str, set]] = None,
    vertical_column_gap_mult: float = 1.0,
    layout_scale: Optional[float] = 1.0,
    wrapped_symbol_relax_font_size_threshold: Optional[int] = None,
) -> Dict:
    """Find the optimal font size and layout for text within given dimensions.

    Uses binary search to find the largest font size that fits.

    Args:
        text: Text to layout
        max_render_width: Maximum allowed width
        max_render_height: Maximum allowed height
        regular_hb_face: HarfBuzz face for the regular font
        regular_typeface: Skia typeface for the regular font
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        features_to_enable: HarfBuzz features to enable
        min_font_size: Minimum font size to try
        max_font_size: Maximum font size to try
        line_spacing_mult: Line spacing multiplier
        hyphenate_before_scaling: Whether to hyphenate before reducing font size
        hyphen_penalty: Penalty for hyphenated lines
        hyphenation_min_word_length: Minimum word length for hyphenation
        badness_exponent: Exponent for line breaking badness calculation
        verbose: Whether to print detailed logs
        bubble_id: Optional identifier for the bubble (for logging purposes)
        cleaned_mask: Optional binary mask of the bubble for collision detection
        box_top_left: Optional (x, y) coordinates of the bounding box top-left corner

    Returns:
        Dictionary containing layout data (font_size, lines, metrics, etc.)

    Raises:
        RenderingError: If text doesn't fit at minimum font size or layout fails
    """
    # Preserve user-entered spacing and explicit newlines. The wrapping
    # tokenizer protects consecutive spaces while it computes line breaks.
    text = attach_korean_postposition_particles(text)
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not clean_text:
        raise RenderingError("Empty text cannot be laid out")

    best_fit_size = -1
    best_fit_lines_data = None
    best_fit_metrics = None
    best_fit_max_line_width = float("inf")
    best_fit_line_height = 0.0
    best_fit_block_height = 0.0

    word_width_cache: Dict[Tuple[str, int], float] = {}
    try:
        wrapped_symbol_relax_threshold = int(
            wrapped_symbol_relax_font_size_threshold
            if wrapped_symbol_relax_font_size_threshold is not None
            else WRAPPED_SYMBOL_RELAX_FONT_SIZE_THRESHOLD
        )
    except (TypeError, ValueError):
        wrapped_symbol_relax_threshold = WRAPPED_SYMBOL_RELAX_FONT_SIZE_THRESHOLD
    wrapped_symbol_relax_threshold = max(1, wrapped_symbol_relax_threshold)

    low = min_font_size
    high = max_font_size

    def _accept_collision_layout(
        font_size: int,
        fit_data: Dict,
        ratio_limit: float,
        corner_limit: int,
        allow_shift: bool,
    ) -> Optional[Tuple[Dict, Tuple[float, float]]]:
        if cleaned_mask is None or box_top_left is None:
            return fit_data, (0.0, 0.0)

        stats = _collision_stats(
            fit_data["lines"],
            box_top_left,
            cleaned_mask,
            fit_data["line_height"],
            (max_render_width, max_render_height),
        )
        if not stats["has_collision"]:
            return fit_data, (0.0, 0.0)

        if allow_shift and _collision_is_small(stats, ratio_limit, corner_limit):
            shift = _find_collision_shift(
                fit_data["lines"],
                box_top_left,
                cleaned_mask,
                fit_data["line_height"],
                (max_render_width, max_render_height),
                layout_scale,
            )
            if shift is not None:
                log_message(
                    "Resolved minor layout collision by shifting text "
                    f"at size {font_size} by ({shift[0]:.0f}, {shift[1]:.0f}) "
                    f"(ratio={stats['outside_ratio']:.3f}, corners={stats['corner_count']})",
                    verbose=verbose,
                )
                return fit_data, shift

        return None

    if cleaned_mask is not None and box_top_left is not None and not vertical_stack:
        for mid in range(high, low - 1, -1):
            log_message(f"Testing size {mid}", verbose=verbose)

            width_attempts = [float(max_render_width)]
            if mid == high:
                width_attempts.append(float(max_render_width) * 0.90)

            for attempt_index, current_width_attempt in enumerate(width_attempts):
                fit_data = check_fit(
                    mid,
                    clean_text,
                    current_width_attempt,
                    max_render_height,
                    regular_hb_face,
                    regular_typeface,
                    loaded_hb_faces,
                    features_to_enable,
                    line_spacing_mult,
                    hyphenate_before_scaling,
                    hyphen_penalty,
                    hyphenation_min_word_length,
                    badness_exponent,
                    word_width_cache,
                    verbose,
                    detach_trailing_punctuation,
                    vertical_stack,
                    font_cmaps,
                    vertical_column_gap_mult,
                    layout_scale=layout_scale,
                )

                if fit_data is None:
                    break

                if mid == high and attempt_index == 0:
                    ratio_limit = COLLISION_VERY_SLIGHT_RATIO
                    corner_limit = COLLISION_VERY_SLIGHT_MAX_CORNERS
                else:
                    ratio_limit = COLLISION_SLIGHT_RATIO
                    corner_limit = COLLISION_SLIGHT_MAX_CORNERS

                accepted = _accept_collision_layout(
                    mid,
                    fit_data,
                    ratio_limit,
                    corner_limit,
                    True,
                )
                if accepted is not None:
                    accepted_fit, accepted_offset = accepted
                    if mid < max_font_size:
                        bubble_desc = f"bubble {bubble_id}" if bubble_id else "bubble"
                        log_message(
                            f"Shrinking text in {bubble_desc} to size {mid}",
                            verbose=verbose,
                        )
                    return {
                        "font_size": mid,
                        "lines": accepted_fit["lines"],
                        "metrics": accepted_fit["metrics"],
                        "max_line_width": accepted_fit["max_line_width"],
                        "line_height": accepted_fit["line_height"],
                        "orientation": "horizontal",
                        "block_height": accepted_fit.get("block_height", 0.0),
                        "position_offset_x": accepted_offset[0],
                        "position_offset_y": accepted_offset[1],
                    }

                if mid == high and attempt_index == 0:
                    stats = _collision_stats(
                        fit_data["lines"],
                        box_top_left,
                        cleaned_mask,
                        fit_data["line_height"],
                        (max_render_width, max_render_height),
                    )
                    log_message(
                        "Layout collision too large at current width; "
                        f"trying 90% render width once "
                        f"(size={mid}, ratio={stats['outside_ratio']:.3f}, "
                        f"corners={stats['corner_count']})",
                        verbose=verbose,
                    )

        log_message(
            f"Text too large for bubble at min size {min_font_size}: '{clean_text[:30]}'",
            always_print=True,
        )
        raise RenderingError(
            f"Text too large for bubble at minimum font size {min_font_size}"
        )

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            break

        log_message(f"Testing size {mid}", verbose=verbose)

        succeeded_at_current_size = False
        current_width_attempt = max_render_width
        max_squeezes = 3 if cleaned_mask is not None else 1

        for _ in range(max_squeezes):
            fit_data = check_fit(
                mid,
                clean_text,
                current_width_attempt,
                max_render_height,
                regular_hb_face,
                regular_typeface,
                loaded_hb_faces,
                features_to_enable,
                line_spacing_mult,
                hyphenate_before_scaling,
                hyphen_penalty,
                hyphenation_min_word_length,
                badness_exponent,
                word_width_cache,
                verbose,
                detach_trailing_punctuation,
                vertical_stack,
                font_cmaps,
                vertical_column_gap_mult,
                layout_scale=layout_scale,
            )

            if fit_data is None:
                # Squeezing narrower won't help (only makes it taller)
                break

            if (
                cleaned_mask is not None
                and box_top_left is not None
                and not vertical_stack
            ):
                has_collision = _check_collision(
                    fit_data["lines"],
                    box_top_left,
                    cleaned_mask,
                    fit_data["line_height"],
                    (current_width_attempt, max_render_height),
                )

                if not has_collision:
                    best_fit_size = mid
                    best_fit_lines_data = fit_data["lines"]
                    best_fit_metrics = fit_data["metrics"]
                    best_fit_max_line_width = fit_data["max_line_width"]
                    best_fit_line_height = fit_data["line_height"]
                    best_fit_block_height = fit_data.get("block_height", 0.0)

                    succeeded_at_current_size = True
                    break
                else:
                    if verbose:
                        log_message(
                            f"Collision at size {mid} width {current_width_attempt:.0f}, squeezing...",
                            verbose=verbose,
                        )
                    current_width_attempt *= 0.90
                    continue
            else:
                best_fit_size = mid
                best_fit_lines_data = fit_data["lines"]
                best_fit_metrics = fit_data["metrics"]
                best_fit_max_line_width = fit_data["max_line_width"]
                best_fit_line_height = fit_data["line_height"]
                best_fit_block_height = fit_data.get("block_height", 0.0)
                succeeded_at_current_size = True
                break

        if succeeded_at_current_size:
            low = mid + 1
        else:
            high = mid - 1

    if best_fit_size == -1:
        log_message(
            f"Text too large for bubble at min size {min_font_size}: '{clean_text[:30]}'",
            always_print=True,
        )
        raise RenderingError(
            f"Text too large for bubble at minimum font size {min_font_size}"
        )

    if (
        not vertical_stack
        and cleaned_mask is not None
        and best_fit_size <= wrapped_symbol_relax_threshold
    ):
        relaxed_best_fit_size = -1
        relaxed_best_fit_lines_data = None
        relaxed_best_fit_metrics = None
        relaxed_best_fit_max_line_width = float("inf")
        relaxed_best_fit_line_height = 0.0
        relaxed_best_fit_block_height = 0.0

        low = min_font_size
        high = max_font_size
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                break

            succeeded_at_current_size = False
            current_width_attempt = max_render_width
            max_squeezes = 3 if cleaned_mask is not None else 1

            for _ in range(max_squeezes):
                fit_data = check_fit(
                    mid,
                    clean_text,
                    current_width_attempt,
                    max_render_height,
                    regular_hb_face,
                    regular_typeface,
                    loaded_hb_faces,
                    features_to_enable,
                    line_spacing_mult,
                    hyphenate_before_scaling,
                    hyphen_penalty,
                    hyphenation_min_word_length,
                    badness_exponent,
                    word_width_cache,
                    verbose,
                    detach_trailing_punctuation,
                    vertical_stack,
                    font_cmaps,
                    vertical_column_gap_mult,
                    False,
                    layout_scale=layout_scale,
                )

                if fit_data is None:
                    break

                if cleaned_mask is not None and box_top_left is not None:
                    has_collision = _check_collision(
                        fit_data["lines"],
                        box_top_left,
                        cleaned_mask,
                        fit_data["line_height"],
                        (current_width_attempt, max_render_height),
                    )
                    if has_collision:
                        current_width_attempt *= 0.90
                        continue

                relaxed_best_fit_size = mid
                relaxed_best_fit_lines_data = fit_data["lines"]
                relaxed_best_fit_metrics = fit_data["metrics"]
                relaxed_best_fit_max_line_width = fit_data["max_line_width"]
                relaxed_best_fit_line_height = fit_data["line_height"]
                relaxed_best_fit_block_height = fit_data.get("block_height", 0.0)
                succeeded_at_current_size = True
                break

            if succeeded_at_current_size:
                low = mid + 1
            else:
                high = mid - 1

        if relaxed_best_fit_size > best_fit_size:
            log_message(
                "Relaxed wrapped-symbol keep-together after small font "
                f"({best_fit_size} <= {wrapped_symbol_relax_threshold}); "
                f"using size {relaxed_best_fit_size}",
                verbose=verbose,
            )
            best_fit_size = relaxed_best_fit_size
            best_fit_lines_data = relaxed_best_fit_lines_data
            best_fit_metrics = relaxed_best_fit_metrics
            best_fit_max_line_width = relaxed_best_fit_max_line_width
            best_fit_line_height = relaxed_best_fit_line_height
            best_fit_block_height = relaxed_best_fit_block_height

    if best_fit_size < max_font_size:
        bubble_desc = f"bubble {bubble_id}" if bubble_id else "bubble"
        log_message(
            f"Shrinking text in {bubble_desc} to size {best_fit_size}",
            verbose=verbose,
        )

    return {
        "font_size": best_fit_size,
        "lines": best_fit_lines_data,
        "metrics": best_fit_metrics,
        "max_line_width": best_fit_max_line_width,
        "line_height": best_fit_line_height,
        "orientation": "vertical" if vertical_stack else "horizontal",
        "block_height": best_fit_block_height,
        "position_offset_x": 0.0,
        "position_offset_y": 0.0,
    }
