import re
from typing import Optional, Tuple

import numpy as np
import skia
from PIL import Image

from core.config import RenderingConfig
from core.image.image_utils import calculate_centroid_expansion_box
from core.text.drawing_engine import (
    draw_layout,
    load_font_resources,
    pil_to_skia_surface,
    skia_surface_to_pil,
)
from core.text.font_manager import (
    find_font_variants,
    get_font_cmap,
    get_font_features,
)
from core.text.font_fallback import (
    FALLBACK_FONT_KEY,
    find_fallback_font_for_codepoints,
    missing_codepoints_for_text,
    sanitize_text_with_font_fallback,
)
from core.text.layout_engine import find_optimal_layout
from core.text.text_processing import (
    NO_SPACE_BEFORE_MARKER,
    attach_korean_postposition_particles,
    is_rtl_script,
    parse_styled_segments,
    strip_no_space_before_marker,
)
from utils.exceptions import FontError, ImageProcessingError, RenderingError
from utils.logging import log_message

GRAYSCALE_MIDPOINT = 128  # Threshold for determining text color
FALLBACK_PADDING_RATIO = 0.08  # 8% padding ratio when safe area calculation fails
SMALL_TEXT_ZERO_PADDING_MAX_FONT_SIZE = 10
SMALL_TEXT_ZERO_PADDING_MAX_CHARS = 10
COLLAPSED_SMALL_BBOX_FALLBACK_MAX_AREA = 8000
COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SIDE = 140
COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SHORT_SIDE = 90
COLLAPSED_SMALL_BBOX_FALLBACK_MAX_AREA_RATIO = 0.30
COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SIDE_RATIO = 0.55
CONSTRICTED_TALL_SHORT_BBOX_FALLBACK_MIN_ASPECT = 1.5
CONSTRICTED_TALL_SHORT_BBOX_FALLBACK_MAX_CHARS = 15
CONSTRICTED_TALL_SHORT_BBOX_FALLBACK_SCALE = 0.90
AUTO_VERTICAL_MIN_ASPECT_RATIO = 1.7
AUTO_VERTICAL_MAX_CHARS = 4
# Japanese source text has no word spacing, so a vertical line there is always
# "one word".  Korean output does have spacing, which made this gate reject
# vertical layout for almost every translated line -- even in column-shaped
# bubbles where vertical is clearly right.  Allow a few words.
AUTO_VERTICAL_MAX_WORDS = 3
AUTO_VERTICAL_MAX_INTERNAL_SPACES = 1
AUTO_VERTICAL_MAX_HORIZONTAL_FILL = 0.45
AUTO_VERTICAL_MIN_FILL_GAIN = 0.20
AUTO_VERTICAL_MIN_FONT_SIZE_GAIN = 1.40
TALL_BUBBLE_VERTICAL_MIN_ASPECT_RATIO = 3.0
TALL_BUBBLE_VERTICAL_MAX_HORIZONTAL_FILL = 0.75
TALL_BUBBLE_VERTICAL_MIN_FILL_GAIN = 0.10
TALL_BUBBLE_VERTICAL_MIN_FONT_SIZE_RATIO = 1.0
EMPHASIS_STYLES = {"bold", "bold_italic"}


def _render_padding_pair(config: RenderingConfig) -> tuple[float, float]:
    try:
        fallback = max(0.0, float(getattr(config, "padding_pixels", 5.0) or 0.0))
    except (TypeError, ValueError):
        fallback = 5.0
    try:
        padding_x = (
            fallback
            if getattr(config, "padding_horizontal_pixels", None) is None
            else max(0.0, float(getattr(config, "padding_horizontal_pixels")))
        )
    except (TypeError, ValueError):
        padding_x = fallback
    try:
        padding_y = (
            fallback
            if getattr(config, "padding_vertical_pixels", None) is None
            else max(0.0, float(getattr(config, "padding_vertical_pixels")))
        )
    except (TypeError, ValueError):
        padding_y = fallback
    return padding_x, padding_y


def _plain_text_for_layout_policy(text: str) -> str:
    return "".join(segment_text for segment_text, _ in parse_styled_segments(text))


def _style_segments_for_text(text: str) -> list[tuple[str, str]]:
    return [(seg_text, style) for seg_text, style in parse_styled_segments(text) if seg_text]


def _count_visible_vertical_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace() and ch not in {"*", "_", "`"})


def _count_auto_vertical_limit_chars(text: str) -> int:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    return sum(1 for ch in value if ch.isalnum())


def _count_small_text_padding_retry_chars(text: str) -> int:
    plain_text = strip_no_space_before_marker(_plain_text_for_layout_policy(text))
    return sum(1 for ch in plain_text if not ch.isspace())


def _should_retry_zero_padding_for_small_text(
    layout_data: dict,
    text: str,
    config: Optional[RenderingConfig] = None,
) -> bool:
    try:
        font_size = float(layout_data.get("font_size") or 0.0)
    except (TypeError, ValueError):
        return False
    try:
        max_font_size = float(
            getattr(
                config,
                "small_text_zero_padding_max_font_size",
                SMALL_TEXT_ZERO_PADDING_MAX_FONT_SIZE,
            )
            or SMALL_TEXT_ZERO_PADDING_MAX_FONT_SIZE
        )
    except (TypeError, ValueError):
        max_font_size = float(SMALL_TEXT_ZERO_PADDING_MAX_FONT_SIZE)
    if font_size <= 0.0 or font_size > max_font_size:
        return False

    char_count = _count_small_text_padding_retry_chars(text)
    try:
        max_chars = int(
            getattr(
                config,
                "small_text_zero_padding_max_chars",
                SMALL_TEXT_ZERO_PADDING_MAX_CHARS,
            )
            or SMALL_TEXT_ZERO_PADDING_MAX_CHARS
        )
    except (TypeError, ValueError):
        max_chars = SMALL_TEXT_ZERO_PADDING_MAX_CHARS
    return 0 < char_count <= max(1, max_chars)


def _should_retry_zero_padding_after_layout_failure(
    text: str,
    config: RenderingConfig,
) -> bool:
    try:
        min_font_size = float(config.min_font_size)
    except (TypeError, ValueError):
        return False
    try:
        max_font_size = float(
            getattr(
                config,
                "small_text_zero_padding_max_font_size",
                SMALL_TEXT_ZERO_PADDING_MAX_FONT_SIZE,
            )
            or SMALL_TEXT_ZERO_PADDING_MAX_FONT_SIZE
        )
    except (TypeError, ValueError):
        max_font_size = float(SMALL_TEXT_ZERO_PADDING_MAX_FONT_SIZE)
    if min_font_size > max_font_size:
        return False

    char_count = _count_small_text_padding_retry_chars(text)
    try:
        max_chars = int(
            getattr(
                config,
                "small_text_zero_padding_max_chars",
                SMALL_TEXT_ZERO_PADDING_MAX_CHARS,
            )
            or SMALL_TEXT_ZERO_PADDING_MAX_CHARS
        )
    except (TypeError, ValueError):
        max_chars = SMALL_TEXT_ZERO_PADDING_MAX_CHARS
    return 0 < char_count <= max(1, max_chars)


def _safe_area_collapsed_for_small_bbox(
    bbox_width: float,
    bbox_height: float,
    safe_width: float,
    safe_height: float,
    config: Optional[RenderingConfig] = None,
) -> bool:
    if bbox_width <= 0 or bbox_height <= 0 or safe_width <= 0 or safe_height <= 0:
        return False

    bbox_area = float(bbox_width) * float(bbox_height)
    max_area = float(
        getattr(
            config,
            "collapsed_small_bbox_fallback_max_area",
            COLLAPSED_SMALL_BBOX_FALLBACK_MAX_AREA,
        )
        or COLLAPSED_SMALL_BBOX_FALLBACK_MAX_AREA
    )
    max_side = float(
        getattr(
            config,
            "collapsed_small_bbox_fallback_max_side",
            COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SIDE,
        )
        or COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SIDE
    )
    max_short_side = float(
        getattr(
            config,
            "collapsed_small_bbox_fallback_max_short_side",
            COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SHORT_SIDE,
        )
        or COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SHORT_SIDE
    )
    if bbox_area > max_area:
        return False
    if max(bbox_width, bbox_height) > max_side:
        return False
    if min(bbox_width, bbox_height) > max_short_side:
        return False

    safe_area_ratio = (float(safe_width) * float(safe_height)) / bbox_area
    safe_side_ratio = min(
        float(safe_width) / float(bbox_width),
        float(safe_height) / float(bbox_height),
    )
    return (
        safe_area_ratio <= COLLAPSED_SMALL_BBOX_FALLBACK_MAX_AREA_RATIO
        and safe_side_ratio <= COLLAPSED_SMALL_BBOX_FALLBACK_MAX_SIDE_RATIO
    )


def _should_use_constricted_tall_short_bbox_fallback(
    moved_to_pole: bool,
    bbox_width: float,
    bbox_height: float,
    text: str,
) -> bool:
    if not moved_to_pole or bbox_width <= 0.0 or bbox_height <= 0.0:
        return False
    if bbox_height < bbox_width * CONSTRICTED_TALL_SHORT_BBOX_FALLBACK_MIN_ASPECT:
        return False

    char_count = _count_small_text_padding_retry_chars(text)
    return 0 < char_count <= CONSTRICTED_TALL_SHORT_BBOX_FALLBACK_MAX_CHARS


def _split_forced_vertical_columns(text: str, max_columns: int) -> str:
    value = (
        str(text or "")
        .replace("\\r\\n", " ")
        .replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    value = " ".join(line.strip() for line in value.splitlines()).strip()
    if max_columns <= 1 or _count_visible_vertical_chars(value) <= 1:
        return value

    best_split = None
    best_score = None
    for split_pos in range(1, len(value)):
        left = value[:split_pos].strip()
        right = value[split_pos:].strip()
        if not left or not right:
            continue

        left_count = _count_visible_vertical_chars(left)
        right_count = _count_visible_vertical_chars(right)
        if left_count <= 0 or right_count <= 0:
            continue

        prev_char = value[split_pos - 1]
        next_char = value[split_pos]
        boundary_penalty = 0 if prev_char.isspace() or next_char.isspace() else 4
        edge_penalty = 1 if left_count == 1 or right_count == 1 else 0
        score = abs(left_count - right_count) + boundary_penalty + edge_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_split = (left, right)

    if best_split is None:
        return value
    return f"{best_split[0]}\n{best_split[1]}"


def _collapse_llm_line_breaks_for_vertical(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(line.strip() for line in normalized.split("\n")).strip()


def _style_sensitive_missing_codepoints(
    text: str,
    font_cmaps: dict[str, set],
) -> set[int]:
    """Find glyphs missing from the actual style that will render each segment.

    The old logic used the union of every font variant in the pack. That caused
    false negatives when, for example, the bold/emphasis font contained 【】 but
    the regular font did not. In that case no fallback font was loaded, and
    regular text silently lost those glyphs. This helper checks each styled
    segment against the cmap of the style that will actually render it.
    """
    segments = _style_segments_for_text(text)
    if not segments:
        segments = [(text, 'regular')]

    regular_cmap = font_cmaps.get('regular') or set()
    missing: set[int] = set()
    for seg_text, style in segments:
        cmap = font_cmaps.get(style) or regular_cmap
        missing.update(missing_codepoints_for_text(seg_text, cmap or set()))
    return missing


def _effective_max_font_size(text: str, config: RenderingConfig) -> int:
    segments = _style_segments_for_text(text)
    if not segments:
        return int(config.max_font_size)

    has_emphasis = any(style in EMPHASIS_STYLES for _, style in segments)
    if not has_emphasis:
        return int(config.max_font_size)

    emphasis_max = getattr(config, "emphasis_max_font_size", None)
    if emphasis_max is None:
        emphasis_max = config.max_font_size

    max_regular = int(config.max_font_size)
    max_emphasis = int(emphasis_max)
    only_emphasis = all(style in EMPHASIS_STYLES for _, style in segments)
    effective_max = max_emphasis if only_emphasis else min(max_regular, max_emphasis)
    return max(int(config.min_font_size), effective_max)


def _effective_line_spacing_mult(text: str, config: RenderingConfig) -> float:
    segments = _style_segments_for_text(text)
    regular_spacing = float(getattr(config, "line_spacing_mult", 1.0) or 1.0)
    if not segments:
        return regular_spacing

    only_emphasis = all(style in EMPHASIS_STYLES for _, style in segments)
    if not only_emphasis:
        return regular_spacing

    emphasis_spacing = getattr(config, "emphasis_line_spacing_mult", None)
    if emphasis_spacing is None:
        return regular_spacing
    try:
        emphasis_spacing = float(emphasis_spacing)
    except (TypeError, ValueError):
        return regular_spacing
    return emphasis_spacing if emphasis_spacing > 0 else regular_spacing


def _effective_vertical_line_spacing_mult(
    text: str, config: RenderingConfig, fallback_spacing: float
) -> float:
    """Return vertical-only line spacing, with a separate value for fully emphasized text."""
    vertical_spacing = getattr(config, "vertical_line_spacing_mult", None)
    if vertical_spacing is None:
        vertical_spacing = fallback_spacing
    try:
        vertical_spacing = float(vertical_spacing)
    except (TypeError, ValueError):
        vertical_spacing = fallback_spacing
    if vertical_spacing <= 0:
        vertical_spacing = fallback_spacing

    segments = _style_segments_for_text(text)
    only_emphasis = bool(segments) and all(style in EMPHASIS_STYLES for _, style in segments)
    if not only_emphasis:
        return vertical_spacing

    emphasis_spacing = getattr(config, "vertical_emphasis_line_spacing_mult", None)
    if emphasis_spacing is None:
        return vertical_spacing
    try:
        emphasis_spacing = float(emphasis_spacing)
    except (TypeError, ValueError):
        return vertical_spacing
    return emphasis_spacing if emphasis_spacing > 0 else vertical_spacing


def _effective_vertical_max_font_size(base_max_font_size: int, config: RenderingConfig) -> int:
    """Apply the vertical-only font-size multiplier to the max size search range."""
    try:
        mult = float(getattr(config, "vertical_font_size_mult", 1.0) or 1.0)
    except (TypeError, ValueError):
        mult = 1.0
    if mult <= 0:
        mult = 1.0
    scaled = int(round(float(base_max_font_size) * mult))
    scaled = max(int(config.min_font_size), scaled)
    return min(256, scaled)


def _effective_auto_vertical_max_chars(text: str, config: Optional[RenderingConfig]) -> int:
    if config is None:
        return AUTO_VERTICAL_MAX_CHARS

    try:
        regular_max = int(getattr(config, "vertical_max_chars", AUTO_VERTICAL_MAX_CHARS))
    except (TypeError, ValueError):
        regular_max = AUTO_VERTICAL_MAX_CHARS
    regular_max = max(0, regular_max)

    segments = _style_segments_for_text(text)
    only_emphasis = bool(segments) and all(style in EMPHASIS_STYLES for _, style in segments)
    if not only_emphasis:
        return regular_max

    emphasis_max = getattr(config, "vertical_emphasis_max_chars", None)
    if emphasis_max is None:
        return regular_max
    try:
        emphasis_max = int(emphasis_max)
    except (TypeError, ValueError):
        return regular_max
    return max(0, emphasis_max)


def _first_available_variant(font_variants: dict):
    for style in ("regular", "bold", "bold_italic", "italic"):
        path = font_variants.get(style)
        if path:
            return path
    return None


def _apply_emphasis_font_variants(
    font_variants: dict,
    emphasis_font_dir: Optional[str],
    verbose: bool = False,
) -> None:
    if not emphasis_font_dir:
        return

    emphasis_variants = dict(find_font_variants(emphasis_font_dir, verbose=verbose))
    emphasis_regular = _first_available_variant(emphasis_variants)
    if not emphasis_regular:
        return

    emphasis_bold = emphasis_variants.get("bold") or emphasis_regular
    emphasis_bold_italic = (
        emphasis_variants.get("bold_italic")
        or emphasis_variants.get("bold")
        or emphasis_variants.get("italic")
        or emphasis_regular
    )

    font_variants["bold"] = emphasis_bold
    font_variants["bold_italic"] = emphasis_bold_italic


def _effective_auto_vertical_min_aspect_ratio(config: Optional[RenderingConfig]) -> float:
    """Per-tag box aspect ratio required before vertical layout is considered.

    Set from Config > Tag Styles so dialogue and sound effects can differ: Korean
    dialogue should stay horizontal unless the bubble is a genuine column, while
    sound effects follow the original artwork and read fine vertically at milder
    ratios.  Falls back to the module default when unset.
    """
    if config is None:
        return AUTO_VERTICAL_MIN_ASPECT_RATIO
    try:
        value = getattr(config, "vertical_min_aspect_ratio", None)
        if value is None:
            return AUTO_VERTICAL_MIN_ASPECT_RATIO
        ratio = float(value)
    except (TypeError, ValueError):
        return AUTO_VERTICAL_MIN_ASPECT_RATIO
    return ratio if ratio > 0 else AUTO_VERTICAL_MIN_ASPECT_RATIO


def _should_try_auto_vertical_text(
    text: str,
    max_render_width: float,
    max_render_height: float,
    config: Optional[RenderingConfig] = None,
    ignore_word_count: bool = False,
) -> bool:
    if max_render_width <= 0 or max_render_height <= 0:
        return False
    if max_render_height / max_render_width < _effective_auto_vertical_min_aspect_ratio(config):
        return False

    plain_text = _plain_text_for_layout_policy(text).strip()
    if not plain_text or is_rtl_script(plain_text):
        return False

    limit_chars = _count_auto_vertical_limit_chars(plain_text)
    max_chars = _effective_auto_vertical_max_chars(text, config)
    if ignore_word_count:
        space_count = sum(1 for char in plain_text if char.isspace())
        return (
            0 < limit_chars <= max_chars
            and space_count <= AUTO_VERTICAL_MAX_INTERNAL_SPACES
        )

    word_count = len(plain_text.split())
    return 0 < limit_chars <= max_chars and word_count <= AUTO_VERTICAL_MAX_WORDS


def _should_use_tall_bubble_vertical_policy(
    text: str,
    bubble_width: float,
    bubble_height: float,
    max_render_width: float,
    max_render_height: float,
    config: RenderingConfig,
    ignore_word_count: bool = False,
) -> bool:
    if bubble_width <= 0 or bubble_height <= 0:
        return False
    if bubble_height / bubble_width < TALL_BUBBLE_VERTICAL_MIN_ASPECT_RATIO:
        return False
    return bool(
        config.auto_vertical_text
        and _should_try_auto_vertical_text(
            text,
            max_render_width,
            max_render_height,
            config,
            ignore_word_count=ignore_word_count,
        )
    )


def _auto_vertical_layout_is_better(
    horizontal_layout: dict,
    vertical_layout: dict,
    max_render_height: float,
    max_horizontal_fill: float = AUTO_VERTICAL_MAX_HORIZONTAL_FILL,
    min_fill_gain: float = AUTO_VERTICAL_MIN_FILL_GAIN,
    min_font_size_ratio: float = 1.0,
    vertical_max_render_height: Optional[float] = None,
    ignore_min_font_size_gain: bool = False,
) -> bool:
    if max_render_height <= 0:
        return False
    if vertical_max_render_height is None:
        vertical_max_render_height = max_render_height
    if vertical_max_render_height <= 0:
        return False

    horizontal_block_height = horizontal_layout.get("block_height") or 0.0
    vertical_block_height = vertical_layout.get("block_height") or 0.0
    if horizontal_block_height <= 0 or vertical_block_height <= 0:
        return False

    horizontal_fill = horizontal_block_height / max_render_height
    vertical_fill = vertical_block_height / vertical_max_render_height
    horizontal_font_size = float(horizontal_layout.get("font_size") or 0.0)
    vertical_font_size = float(vertical_layout.get("font_size") or 0.0)
    min_font_size_ratio = max(0.0, float(min_font_size_ratio or 1.0))
    if not ignore_min_font_size_gain:
        min_font_size_ratio = max(min_font_size_ratio, AUTO_VERTICAL_MIN_FONT_SIZE_GAIN)
    return (
        vertical_font_size >= horizontal_font_size * min_font_size_ratio
        and horizontal_fill <= max_horizontal_fill
        and vertical_fill >= horizontal_fill + min_fill_gain
    )


def _layout_fill_ratios(
    layout_data: dict,
    max_render_width: float,
    max_render_height: float,
) -> tuple[float, float]:
    width_ratio = 0.0
    height_ratio = 0.0
    if max_render_width > 0:
        width_ratio = float(layout_data.get("max_line_width") or 0.0) / max_render_width
    if max_render_height > 0:
        height_ratio = float(layout_data.get("block_height") or 0.0) / max_render_height
    return width_ratio, height_ratio


WRAPPED_SYMBOL_ORPHAN_MAX_CHARS = 1
WRAPPED_SYMBOL_RETRY_FACTORS = (0.97, 0.94, 0.90, 0.85)
WRAPPED_SYMBOL_SYMMETRIC_QUOTES = {'"', "'"}
WRAPPED_SYMBOL_OPEN_TO_CLOSE = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
    "\u2018": "\u2019",
    "\u201c": "\u201d",
    "\u3008": "\u3009",
    "\u300a": "\u300b",
    "\u300c": "\u300d",
    "\u300e": "\u300f",
    "\u3010": "\u3011",
    "\u3014": "\u3015",
    "\u3016": "\u3017",
    "\u3018": "\u3019",
    "\u301a": "\u301b",
    "\uff08": "\uff09",
    "\uff3b": "\uff3d",
    "\uff5b": "\uff5d",
    "\uff1c": "\uff1e",
}
WRAPPED_SYMBOL_OPENERS = (
    set(WRAPPED_SYMBOL_OPEN_TO_CLOSE) | WRAPPED_SYMBOL_SYMMETRIC_QUOTES
)
WRAPPED_SYMBOL_CLOSERS = (
    set(WRAPPED_SYMBOL_OPEN_TO_CLOSE.values()) | WRAPPED_SYMBOL_SYMMETRIC_QUOTES
)


def _plain_layout_line_text(line_data: dict) -> str:
    line_text = str(line_data.get("text_with_markers") or "")
    plain_text = "".join(segment for segment, _ in parse_styled_segments(line_text))
    return strip_no_space_before_marker(plain_text).replace(
        NO_SPACE_BEFORE_MARKER, ""
    )


def _unclosed_wrapped_symbol_closers(text: str) -> list[str]:
    stack: list[str] = []
    for ch in text:
        if ch in WRAPPED_SYMBOL_SYMMETRIC_QUOTES:
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                stack.append(ch)
            continue

        closer = WRAPPED_SYMBOL_OPEN_TO_CLOSE.get(ch)
        if closer:
            stack.append(closer)
            continue

        if ch in WRAPPED_SYMBOL_CLOSERS:
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx] == ch:
                    del stack[idx:]
                    break
    return stack


def _line_starts_with_wrapped_symbol_tail(text: str) -> bool:
    visible_chars = 0
    for ch in text.lstrip():
        if ch.isspace():
            continue
        if ch in WRAPPED_SYMBOL_CLOSERS:
            return visible_chars <= WRAPPED_SYMBOL_ORPHAN_MAX_CHARS
        if ch in WRAPPED_SYMBOL_OPENERS:
            return False
        visible_chars += 1
        if visible_chars > WRAPPED_SYMBOL_ORPHAN_MAX_CHARS:
            return False
    return False


def _line_ends_with_wrapped_symbol_head(text: str) -> bool:
    visible_chars = 0
    for ch in reversed(text.rstrip()):
        if ch.isspace():
            continue
        if ch in WRAPPED_SYMBOL_SYMMETRIC_QUOTES:
            return 0 < visible_chars <= WRAPPED_SYMBOL_ORPHAN_MAX_CHARS
        if ch in WRAPPED_SYMBOL_OPEN_TO_CLOSE:
            return visible_chars <= WRAPPED_SYMBOL_ORPHAN_MAX_CHARS
        if ch in WRAPPED_SYMBOL_CLOSERS:
            return False
        visible_chars += 1
        if visible_chars > WRAPPED_SYMBOL_ORPHAN_MAX_CHARS:
            return False
    return False


def _layout_has_wrapped_symbol_orphan(layout_data: dict) -> bool:
    if layout_data.get("orientation") == "vertical":
        return False

    lines = layout_data.get("lines") or []
    if len(lines) < 2:
        return False

    plain_lines = [_plain_layout_line_text(line) for line in lines]
    text_before_boundary = ""
    for idx in range(len(plain_lines) - 1):
        current_line = plain_lines[idx]
        next_line = plain_lines[idx + 1]
        text_before_boundary += current_line

        if _line_ends_with_wrapped_symbol_head(current_line):
            return True

        if (
            _line_starts_with_wrapped_symbol_tail(next_line)
            and _unclosed_wrapped_symbol_closers(text_before_boundary)
        ):
            return True

    return False


def _wrapped_symbol_retry_max_font_sizes(
    layout_data: dict, config: RenderingConfig
) -> list[int]:
    font_size = int(layout_data.get("font_size") or 0)
    min_font_size = int(getattr(config, "min_font_size", 1) or 1)
    if font_size <= min_font_size:
        return []

    candidates = [font_size - 1]
    for factor in WRAPPED_SYMBOL_RETRY_FACTORS:
        candidates.append(int(round(font_size * factor)))

    try:
        tight_fit_factor = float(
            getattr(config, "tight_fit_shrink_factor", 0.85) or 0.85
        )
    except (TypeError, ValueError):
        tight_fit_factor = 0.85
    candidates.append(int(round(font_size * min(0.98, max(0.50, tight_fit_factor)))))

    retry_sizes: list[int] = []
    for candidate in candidates:
        retry_size = min(font_size - 1, max(min_font_size, int(candidate)))
        if retry_size >= min_font_size and retry_size not in retry_sizes:
            retry_sizes.append(retry_size)
    return retry_sizes


def _should_retry_tight_fit_shrink(
    layout_data: dict,
    max_render_width: float,
    max_render_height: float,
    config: RenderingConfig,
) -> bool:
    if not getattr(config, "tight_fit_shrink_enabled", True):
        return False
    if getattr(config, "tight_fit_shrink_exempt", False):
        return False

    font_size = int(layout_data.get("font_size") or 0)
    min_font_size = int(getattr(config, "tight_fit_shrink_min_font_size", 18) or 18)
    if font_size <= min_font_size:
        return False

    width_ratio, height_ratio = _layout_fill_ratios(
        layout_data, max_render_width, max_render_height
    )
    width_threshold = float(getattr(config, "tight_fit_shrink_width_ratio", 0.88) or 0.88)
    height_threshold = float(
        getattr(config, "tight_fit_shrink_height_ratio", 0.86) or 0.86
    )
    return width_ratio >= width_threshold or height_ratio >= height_threshold


def _tight_fit_retry_max_font_size(layout_data: dict, config: RenderingConfig) -> int:
    font_size = int(layout_data.get("font_size") or 0)
    factor = float(getattr(config, "tight_fit_shrink_factor", 0.85) or 0.85)
    factor = min(0.98, max(0.50, factor))
    retry_size = int(round(font_size * factor))
    retry_size = max(
        int(config.min_font_size),
        _tight_fit_shrink_min_font_size(config),
        retry_size,
    )
    return min(font_size - 1, retry_size)


def _tight_fit_shrink_min_font_size(config: RenderingConfig) -> int:
    try:
        min_font_size = int(
            getattr(config, "tight_fit_shrink_min_font_size", 18) or 18
        )
    except (TypeError, ValueError):
        min_font_size = 18
    return max(1, min_font_size)


def _tight_fit_shrink_max_retries(config: RenderingConfig) -> int:
    try:
        retries = int(getattr(config, "tight_fit_shrink_max_retries", 1) or 1)
    except (TypeError, ValueError):
        retries = 1
    return max(1, min(4, retries))


def render_text_skia(
    pil_image: Image.Image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font_dir: str,
    cleaned_mask: Optional[np.ndarray] = None,
    bubble_color_bgr: Optional[Tuple[int, int, int]] = (255, 255, 255),
    config: Optional[RenderingConfig] = None,
    raise_on_safe_error: bool = False,
    verbose: bool = False,
    bubble_id: Optional[str] = None,
    rotation_deg: float = 0.0,
    vertical_stack: bool = False,
    text_color_rgb: Optional[Tuple[int, int, int]] = None,
    text_background_color: Optional[Tuple[int, int, int]] = None,
    layout_only: bool = False,
    auto_vertical_ignore_word_count: bool = False,
    horizontal_compare_bbox: Optional[Tuple[int, int, int, int]] = None,
    vertical_compare_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Image.Image:
    """
    Fits and renders text within a bounding box using Skia and HarfBuzz.

    This is the high-level orchestrator that coordinates:
    1. Font loading (font_manager)
    2. Safe area calculation (image_utils)
    3. Layout optimization (layout_engine)
    4. Text rendering (drawing_engine)

    Uses the 5-step Distance Transform Insetting Method for safe area calculation:
    1. Establish Safe Zone: Create safe_area_mask using cv2.distanceTransform()
    2. Find Unbiased Anchor: Calculate centroid of the safe_area_mask
    3. Measure Available Space: Ray cast from centroid to find distances to edges
    4. Calculate Symmetrical Dimensions: Use min distances for width/height
    5. Construct Final Box: Create centered rectangle within safe zone

    This ensures text is perfectly centered and never touches bubble boundaries.

    Args:
        pil_image: PIL Image object to draw onto.
        text: Text to render.
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        font_dir: Directory containing font files.
        cleaned_mask: Binary mask of the cleaned bubble (0/255). Used for safe area calculation.
        bubble_color_bgr: Background color of the bubble (BGR tuple). Used to determine text color.
        config: RenderingConfig object containing all rendering parameters. If None, uses defaults.
        verbose: Whether to print detailed logs.

    Returns:
        Modified PIL Image object with rendered text.

    Raises:
        RenderingError: If rendering fails due to invalid inputs, font issues, or layout problems
        FontError: If font loading fails
    """
    x1, y1, x2, y2 = bbox
    bubble_width = x2 - x1
    bubble_height = y2 - y1

    if bubble_width <= 0 or bubble_height <= 0:
        log_message(f"Invalid bbox dimensions: {bbox}", always_print=True)
        raise RenderingError(f"Invalid bounding box dimensions: {bbox}")

    # Keep dash-like glyphs intact so vertical layout can rotate them correctly.
    # Preserve explicit line breaks from translators. Some providers return real
    # newlines, while others may leave literal "\\n" in the parsed text.
    normalized_text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\\n", "\n")
    )
    normalized_text = attach_korean_postposition_particles(normalized_text)

    clean_text = "\n".join(
        line.strip() for line in normalized_text.split("\n")
    ).strip()
    if not clean_text:
        return pil_image

    layout_text = clean_text
    short_text_ignores_auto_vertical_font_gain = bool(
        getattr(config, "short_text_ignores_auto_vertical_font_gain", False)
        and 0
        < _count_auto_vertical_limit_chars(
            _plain_text_for_layout_policy(clean_text)
        )
        <= 3
    )

    # Initialize config with defaults if not provided
    if config is None:
        config = RenderingConfig()

    padding_x, padding_y = _render_padding_pair(config)
    layout_box_top_left = None
    safe_area_result = None
    safe_area_fallback_logged = False
    safe_area_moved_to_pole = False
    if cleaned_mask is not None:
        try:
            safe_area_data = calculate_centroid_expansion_box(
                cleaned_mask,
                padding_pixels=config.padding_pixels,
                padding_horizontal_pixels=padding_x,
                padding_vertical_pixels=padding_y,
                verbose=verbose,
                return_constricted=True,
            )
            safe_area_result = (safe_area_data[0], safe_area_data[1])
            safe_area_moved_to_pole = bool(safe_area_data[2])
        except ImageProcessingError:
            # Safe area calculation failed, will use fallback below
            safe_area_result = None
            if raise_on_safe_error:
                raise
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )
            safe_area_fallback_logged = True

    bbox_fallback_scale = None
    constricted_tall_short_bbox_fallback_applied = False
    if (
        safe_area_result is not None
        and _should_use_constricted_tall_short_bbox_fallback(
            safe_area_moved_to_pole,
            float(bubble_width),
            float(bubble_height),
            layout_text,
        )
    ):
        bbox_fallback_scale = CONSTRICTED_TALL_SHORT_BBOX_FALLBACK_SCALE
        safe_box, _safe_center = safe_area_result
        _safe_x, _safe_y, safe_w, safe_h = safe_box
        log_message(
            "Constricted tall short text; falling back to original bbox "
            f"{bbox_fallback_scale:.2f}x layout "
            f"(bbox={bubble_width}x{bubble_height}, safe={safe_w}x{safe_h}, "
            f"chars={_count_small_text_padding_retry_chars(layout_text)})",
            verbose=verbose,
        )
        safe_area_result = None
        safe_area_fallback_logged = True
        constricted_tall_short_bbox_fallback_applied = True

    collapsed_small_bbox_fallback_applied = False
    if safe_area_result is not None and cleaned_mask is not None:
        safe_box, _safe_center = safe_area_result
        _safe_x, _safe_y, safe_w, safe_h = safe_box
        if _safe_area_collapsed_for_small_bbox(
            float(bubble_width),
            float(bubble_height),
            float(safe_w),
            float(safe_h),
            config,
        ):
            zero_safe_area_result = None
            try:
                zero_safe_area_result = calculate_centroid_expansion_box(
                    cleaned_mask,
                    padding_pixels=0.0,
                    padding_horizontal_pixels=0.0,
                    padding_vertical_pixels=0.0,
                    verbose=verbose,
                )
            except ImageProcessingError:
                zero_safe_area_result = None

            zero_still_collapsed = True
            zero_w = zero_h = 0
            if zero_safe_area_result is not None:
                zero_box, _zero_center = zero_safe_area_result
                _zero_x, _zero_y, zero_w, zero_h = zero_box
                zero_still_collapsed = _safe_area_collapsed_for_small_bbox(
                    float(bubble_width),
                    float(bubble_height),
                    float(zero_w),
                    float(zero_h),
                    config,
                )

            if zero_still_collapsed:
                log_message(
                    "Safe area collapsed for small bbox; "
                    "falling back to original bbox layout "
                    f"(bbox={bubble_width}x{bubble_height}, "
                    f"safe={safe_w}x{safe_h}, zero={zero_w}x{zero_h})",
                    verbose=verbose,
                )
                safe_area_result = None
                safe_area_fallback_logged = True
                collapsed_small_bbox_fallback_applied = True

    if safe_area_result is not None:
        guaranteed_box, _ = safe_area_result
        box_x, box_y, box_w, box_h = guaranteed_box
        center_x_override = getattr(config, "safe_area_center_x_override", None)
        if center_x_override is not None:
            try:
                override_x = float(center_x_override)
            except (TypeError, ValueError):
                override_x = None
            if override_x is not None:
                shifted_box_x = int(round(override_x - box_w / 2.0))
                image_w, _image_h = pil_image.size
                shifted_box_x = max(0, min(shifted_box_x, image_w - box_w))
                if shifted_box_x != box_x:
                    log_message(
                        "Applying safe area center-x override "
                        f"({box_x + box_w / 2.0:.1f} -> {shifted_box_x + box_w / 2.0:.1f})",
                        verbose=verbose,
                    )
                    box_x = shifted_box_x
        layout_box_top_left = (box_x, box_y)
        max_render_width = float(box_w)
        max_render_height = float(box_h)
        target_center_x = box_x + box_w / 2.0
        target_center_y = box_y + box_h / 2.0
        log_message("Using centroid-based safe area calculation", verbose=verbose)
    else:
        # Fallback to padded bbox
        if not safe_area_fallback_logged:
            log_message(
                "Safe area calculation failed, falling back to padded bbox method",
                verbose=verbose,
            )
        fallback_padding_ratio = FALLBACK_PADDING_RATIO
        if bbox_fallback_scale is not None:
            fallback_padding_ratio = (1.0 - bbox_fallback_scale) / 2.0
        elif (
            vertical_stack
            and cleaned_mask is None
            and bubble_height >= bubble_width * 3
        ):
            fallback_padding_ratio = 0.02
        max_render_width = bubble_width * (1 - 2 * fallback_padding_ratio)
        max_render_height = bubble_height * (1 - 2 * fallback_padding_ratio)

        if max_render_width <= 0 or max_render_height <= 0:
            max_render_width = max(1.0, float(bubble_width))
            max_render_height = max(1.0, float(bubble_height))

        target_center_x = x1 + bubble_width / 2.0
        target_center_y = y1 + bubble_height / 2.0

    def _fallback_geometry_for_bbox(
        box: Tuple[int, int, int, int],
        use_vertical_stack: bool,
    ) -> Optional[dict]:
        try:
            bx1, by1, bx2, by2 = [int(round(float(v))) for v in box]
        except (TypeError, ValueError):
            return None
        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= 0 or bh <= 0:
            return None

        fallback_padding_ratio = FALLBACK_PADDING_RATIO
        if bbox_fallback_scale is not None:
            fallback_padding_ratio = (1.0 - bbox_fallback_scale) / 2.0
        elif use_vertical_stack and cleaned_mask is None and bh >= bw * 3:
            fallback_padding_ratio = 0.02

        active_max_width = bw * (1 - 2 * fallback_padding_ratio)
        active_max_height = bh * (1 - 2 * fallback_padding_ratio)
        if active_max_width <= 0 or active_max_height <= 0:
            active_max_width = max(1.0, float(bw))
            active_max_height = max(1.0, float(bh))

        return {
            "bbox": (bx1, by1, bx2, by2),
            "max_render_width": active_max_width,
            "max_render_height": active_max_height,
            "target_center_x": bx1 + bw / 2.0,
            "target_center_y": by1 + bh / 2.0,
            "layout_box_top_left": None,
        }

    horizontal_compare_geometry = (
        _fallback_geometry_for_bbox(horizontal_compare_bbox, False)
        if horizontal_compare_bbox is not None
        else None
    )
    vertical_compare_geometry = (
        _fallback_geometry_for_bbox(vertical_compare_bbox, True)
        if vertical_compare_bbox is not None
        else None
    )

    layout_collision_mask = (
        None
        if collapsed_small_bbox_fallback_applied
        or constricted_tall_short_bbox_fallback_applied
        else cleaned_mask
    )
    if getattr(config, "skip_layout_collision_check", False):
        layout_collision_mask = None
        log_message(
            "Skipping layout collision check for pre-validated conjoined safe-area override",
            verbose=verbose,
        )

    def _apply_render_geometry(geometry: Optional[dict]) -> None:
        nonlocal x1, y1, x2, y2
        nonlocal bubble_width, bubble_height
        nonlocal max_render_width, max_render_height
        nonlocal target_center_x, target_center_y
        nonlocal layout_box_top_left, layout_collision_mask

        if geometry is None:
            return
        x1, y1, x2, y2 = geometry["bbox"]
        bubble_width = x2 - x1
        bubble_height = y2 - y1
        max_render_width = geometry["max_render_width"]
        max_render_height = geometry["max_render_height"]
        target_center_x = geometry["target_center_x"]
        target_center_y = geometry["target_center_y"]
        layout_box_top_left = geometry["layout_box_top_left"]
        layout_collision_mask = None

    def _geometry_max_height(geometry: Optional[dict], default: float) -> float:
        if geometry is None:
            return default
        try:
            return float(geometry["max_render_height"])
        except (TypeError, ValueError):
            return default

    def _geometry_max_width(geometry: Optional[dict], default: float) -> float:
        if geometry is None:
            return default
        try:
            return float(geometry["max_render_width"])
        except (TypeError, ValueError):
            return default

    def _zero_padding_geometry_for_current_box() -> Optional[dict]:
        if cleaned_mask is not None:
            try:
                zero_safe_area_result = calculate_centroid_expansion_box(
                    cleaned_mask,
                    padding_pixels=0.0,
                    padding_horizontal_pixels=0.0,
                    padding_vertical_pixels=0.0,
                    verbose=verbose,
                )
            except ImageProcessingError:
                return None

            zero_box, _ = zero_safe_area_result
            box_x, box_y, box_w, box_h = zero_box
            center_x_override = getattr(config, "safe_area_center_x_override", None)
            if center_x_override is not None:
                try:
                    override_x = float(center_x_override)
                except (TypeError, ValueError):
                    override_x = None
                if override_x is not None:
                    shifted_box_x = int(round(override_x - box_w / 2.0))
                    image_w, _image_h = pil_image.size
                    box_x = max(0, min(shifted_box_x, image_w - box_w))

            return {
                "bbox": (x1, y1, x2, y2),
                "max_render_width": float(max(1, box_w)),
                "max_render_height": float(max(1, box_h)),
                "target_center_x": box_x + box_w / 2.0,
                "target_center_y": box_y + box_h / 2.0,
                "layout_box_top_left": (box_x, box_y),
            }

        if bubble_width <= 0 or bubble_height <= 0:
            return None
        return {
            "bbox": (x1, y1, x2, y2),
            "max_render_width": float(max(1, bubble_width)),
            "max_render_height": float(max(1, bubble_height)),
            "target_center_x": x1 + bubble_width / 2.0,
            "target_center_y": y1 + bubble_height / 2.0,
            "layout_box_top_left": None,
        }

    try:
        font_variants = dict(find_font_variants(font_dir, verbose=verbose))
        _apply_emphasis_font_variants(
            font_variants,
            getattr(config, "emphasis_font_dir", None),
            verbose=verbose,
        )
        regular_font_path = font_variants.get("regular")
    except FontError as e:
        raise RenderingError(f"Font loading failed: {e}") from e

    # Build cmap information for the selected family, then add one automatic
    # fallback font if the family is missing characters such as 【】.  Unlike the
    # old sanitizer, this keeps supported special characters alive and renders
    # only those characters with the fallback font.
    font_cmaps = {}
    for style_key, style_path in font_variants.items():
        if style_path:
            font_cmaps[style_key] = get_font_cmap(str(style_path))

    missing_codepoints = _style_sensitive_missing_codepoints(
        layout_text, font_cmaps
    )
    fallback_font_path = None
    if missing_codepoints:
        fallback_font_path = find_fallback_font_for_codepoints(
            missing_codepoints,
            primary_font_path=str(regular_font_path),
            primary_font_dir=font_dir,
            verbose=verbose,
        )
        if fallback_font_path:
            font_cmaps[FALLBACK_FONT_KEY] = get_font_cmap(str(fallback_font_path))

    layout_text = sanitize_text_with_font_fallback(
        layout_text, font_cmaps, verbose=verbose
    )
    if not layout_text.strip():
        log_message(
            "All text characters unsupported by selected font and fallback fonts, skipping render",
            always_print=True,
        )
        return pil_image
    vertical_layout_text = _collapse_llm_line_breaks_for_vertical(layout_text)

    try:
        _, regular_typeface, regular_hb_face = load_font_resources(
            str(regular_font_path)
        )
    except FontError as e:
        raise RenderingError(f"Font resource loading failed: {e}") from e

    available_features = get_font_features(str(regular_font_path))
    features_to_enable = {
        "kern": "kern" in available_features["GPOS"],
        "liga": config.use_ligatures and "liga" in available_features["GSUB"],
        "calt": "calt" in available_features["GSUB"],
    }
    log_message(
        f"Font features: {[k for k, v in features_to_enable.items() if v]}",
        verbose=verbose,
    )

    # Pre-load all required font variants for layout engine
    preload_hb_faces = {"regular": regular_hb_face}
    for style_key in ["italic", "bold", "bold_italic"]:
        style_path = font_variants.get(style_key)
        if style_path:
            _, _typeface, _hb_face = load_font_resources(str(style_path))
            if _hb_face:
                preload_hb_faces[style_key] = _hb_face

    if fallback_font_path:
        _, _fallback_typeface, _fallback_hb_face = load_font_resources(
            str(fallback_font_path)
        )
        if _fallback_hb_face:
            preload_hb_faces[FALLBACK_FONT_KEY] = _fallback_hb_face

    layout_max_font_size = _effective_max_font_size(layout_text, config)
    layout_line_spacing_mult = _effective_line_spacing_mult(layout_text, config)
    vertical_line_spacing_mult = _effective_vertical_line_spacing_mult(
        vertical_layout_text, config, layout_line_spacing_mult
    )
    layout_vertical_max_font_size = _effective_vertical_max_font_size(
        _effective_max_font_size(vertical_layout_text, config), config
    )

    def _find_layout(
        use_vertical_stack: bool,
        max_font_size_override: Optional[int] = None,
        text_override: Optional[str] = None,
        column_gap_mult_override: Optional[float] = None,
        geometry_override: Optional[dict] = None,
        use_compare_geometry: bool = True,
    ) -> dict:
        text_for_layout = (
            text_override
            if text_override is not None
            else (vertical_layout_text if use_vertical_stack else layout_text)
        )
        line_spacing_for_layout = (
            vertical_line_spacing_mult
            if use_vertical_stack
            else layout_line_spacing_mult
        )
        effective_max_font_size = (
            layout_vertical_max_font_size if use_vertical_stack else layout_max_font_size
        )
        if max_font_size_override is not None:
            effective_max_font_size = min(effective_max_font_size, max_font_size_override)
        column_gap_mult = (
            column_gap_mult_override
            if column_gap_mult_override is not None
            else (
                getattr(config, "force_vertical_column_gap_mult", 1.0)
                if use_vertical_stack
                else 1.0
            )
        )
        active_max_render_width = max_render_width
        active_max_render_height = max_render_height
        active_layout_collision_mask = layout_collision_mask
        active_layout_box_top_left = layout_box_top_left
        if geometry_override is None and use_compare_geometry:
            geometry_override = (
                vertical_compare_geometry
                if use_vertical_stack
                else horizontal_compare_geometry
            )
        if geometry_override is not None:
            active_max_render_width = geometry_override["max_render_width"]
            active_max_render_height = geometry_override["max_render_height"]
            active_layout_collision_mask = None
            active_layout_box_top_left = geometry_override["layout_box_top_left"]

        return find_optimal_layout(
            text_for_layout,
            active_max_render_width,
            active_max_render_height,
            regular_hb_face,
            regular_typeface,
            preload_hb_faces,
            features_to_enable,
            config.min_font_size,
            effective_max_font_size,
            line_spacing_for_layout,
            False if use_vertical_stack else config.hyphenate_before_scaling,
            config.hyphen_penalty,
            config.hyphenation_min_word_length,
            config.badness_exponent,
            verbose,
            bubble_id,
            active_layout_collision_mask,
            active_layout_box_top_left,
            config.detach_trailing_punctuation,
            use_vertical_stack,
            font_cmaps,
            column_gap_mult,
            layout_scale=getattr(config, "layout_scale", 1.0),
            wrapped_symbol_relax_font_size_threshold=getattr(
                config,
                "wrapped_symbol_relax_font_size_threshold",
                None,
            ),
        )

    def _finalize_orientation_candidate(
        layout_data: dict,
        use_vertical_stack: bool,
        geometry_override: Optional[dict],
    ) -> dict:
        candidate_width = _geometry_max_width(
            geometry_override, max_render_width
        )
        candidate_height = _geometry_max_height(
            geometry_override, max_render_height
        )
        for shrink_attempt in range(_tight_fit_shrink_max_retries(config)):
            if not _should_retry_tight_fit_shrink(
                layout_data, candidate_width, candidate_height, config
            ):
                break
            retry_max_font_size = _tight_fit_retry_max_font_size(
                layout_data, config
            )
            if retry_max_font_size < int(config.min_font_size):
                break
            original_font_size = layout_data.get("font_size")
            original_width_ratio, original_height_ratio = _layout_fill_ratios(
                layout_data, candidate_width, candidate_height
            )
            try:
                retry_layout_data = _find_layout(
                    use_vertical_stack,
                    max_font_size_override=retry_max_font_size,
                    geometry_override=geometry_override,
                )
                if (
                    int(retry_layout_data.get("font_size") or 0)
                    < _tight_fit_shrink_min_font_size(config)
                ):
                    break
                layout_data = retry_layout_data
                log_message(
                    "Finalized orientation candidate tight fit "
                    f"attempt {shrink_attempt + 1} "
                    f"from size {original_font_size} "
                    f"to {layout_data.get('font_size')} "
                    f"(fill {original_width_ratio:.2f}w/"
                    f"{original_height_ratio:.2f}h)",
                    verbose=verbose,
                )
            except RenderingError:
                break

        if _layout_has_wrapped_symbol_orphan(layout_data):
            original_font_size = layout_data.get("font_size")
            for retry_max_font_size in _wrapped_symbol_retry_max_font_sizes(
                layout_data, config
            ):
                try:
                    retry_layout_data = _find_layout(
                        use_vertical_stack,
                        max_font_size_override=retry_max_font_size,
                        geometry_override=geometry_override,
                    )
                except RenderingError:
                    continue
                if _layout_has_wrapped_symbol_orphan(retry_layout_data):
                    continue
                layout_data = retry_layout_data
                log_message(
                    "Finalized orientation candidate wrapped punctuation "
                    f"from size {original_font_size} "
                    f"to {layout_data.get('font_size')}",
                    verbose=verbose,
                )
                break

        return layout_data

    def _log_orientation_comparison(
        horizontal_layout: dict,
        vertical_layout: dict,
        horizontal_geometry: Optional[dict],
        vertical_geometry: Optional[dict],
    ) -> None:
        horizontal_width = _geometry_max_width(
            horizontal_geometry, max_render_width
        )
        horizontal_height = _geometry_max_height(
            horizontal_geometry, max_render_height
        )
        vertical_width = _geometry_max_width(
            vertical_geometry, max_render_width
        )
        vertical_height = _geometry_max_height(
            vertical_geometry, max_render_height
        )
        horizontal_fill = _layout_fill_ratios(
            horizontal_layout, horizontal_width, horizontal_height
        )
        vertical_fill = _layout_fill_ratios(
            vertical_layout, vertical_width, vertical_height
        )
        log_message(
            "Comparing finalized orientations: "
            f"horizontal size={horizontal_layout.get('font_size')} "
            f"fill={horizontal_fill[0]:.2f}w/{horizontal_fill[1]:.2f}h, "
            f"vertical size={vertical_layout.get('font_size')} "
            f"fill={vertical_fill[0]:.2f}w/{vertical_fill[1]:.2f}h",
            verbose=verbose,
        )

    forced_vertical_retry_columns = 1
    forced_vertical_retry_font_size = 12
    forced_vertical_split_text = None
    if vertical_stack and getattr(config, "force_vertical_fallback_only", False):
        try:
            forced_vertical_retry_columns = max(
                1, int(getattr(config, "force_vertical_retry_columns", 1) or 1)
            )
        except (TypeError, ValueError):
            forced_vertical_retry_columns = 1
        try:
            forced_vertical_retry_font_size = max(
                1, int(getattr(config, "force_vertical_retry_font_size", 12) or 12)
            )
        except (TypeError, ValueError):
            forced_vertical_retry_font_size = 12
        if forced_vertical_retry_columns > 1:
            forced_vertical_split_text = _split_forced_vertical_columns(
                vertical_layout_text, forced_vertical_retry_columns
            )
            if forced_vertical_split_text == vertical_layout_text:
                forced_vertical_split_text = None

    def _try_forced_vertical_split(reason: str) -> Optional[dict]:
        nonlocal vertical_layout_text
        if not forced_vertical_split_text:
            return None
        try:
            split_layout = _find_layout(
                True,
                text_override=forced_vertical_split_text,
                column_gap_mult_override=getattr(
                    config, "force_vertical_column_gap_mult", 1.0
                ),
            )
        except RenderingError:
            return None
        vertical_layout_text = forced_vertical_split_text
        log_message(
            "Switching forced vertical layout to "
            f"{forced_vertical_retry_columns} columns because {reason}",
            verbose=verbose,
        )
        return split_layout

    def _try_zero_padding_layout_after_failure(
        use_vertical_stack: bool,
        reason: str,
    ) -> Optional[dict]:
        nonlocal zero_padding_retry_applied
        if not _should_retry_zero_padding_after_layout_failure(layout_text, config):
            return None
        zero_padding_geometry = _zero_padding_geometry_for_current_box()
        if zero_padding_geometry is None:
            return None
        try:
            retry_layout = _find_layout(
                use_vertical_stack,
                geometry_override=zero_padding_geometry,
                use_compare_geometry=False,
            )
        except RenderingError:
            return None
        _apply_render_geometry(zero_padding_geometry)
        zero_padding_retry_applied = True
        log_message(
            "Retried failed small text with zero padding "
            f"after {reason}; size {retry_layout.get('font_size')}",
            verbose=verbose,
        )
        return retry_layout

    orientation_candidates_finalized = False
    orientation_comparison_completed = False
    zero_padding_retry_applied = False
    layout_data = None

    try:
        layout_data = _find_layout(vertical_stack)
    except RenderingError as e:
        zero_padding_layout = _try_zero_padding_layout_after_failure(
            vertical_stack,
            "initial layout failure",
        )
        if zero_padding_layout is not None:
            layout_data = zero_padding_layout
        elif vertical_stack and (
            getattr(config, "compare_forced_vertical_layout", False)
            or getattr(config, "force_vertical_fallback_only", False)
        ):
            try:
                split_layout_data = None
                if getattr(config, "force_vertical_fallback_only", False):
                    split_layout_data = _try_forced_vertical_split(
                        "single-column layout failed"
                    )
                if split_layout_data is not None:
                    layout_data = split_layout_data
                    vertical_stack = True
                else:
                    layout_data = _find_layout(
                        False,
                        geometry_override=horizontal_compare_geometry,
                    )
                    vertical_stack = False
                    if horizontal_compare_geometry is not None:
                        x1, y1, x2, y2 = horizontal_compare_geometry["bbox"]
                        bubble_width = x2 - x1
                        bubble_height = y2 - y1
                        max_render_width = horizontal_compare_geometry[
                            "max_render_width"
                        ]
                        max_render_height = horizontal_compare_geometry[
                            "max_render_height"
                        ]
                        target_center_x = horizontal_compare_geometry[
                            "target_center_x"
                        ]
                        target_center_y = horizontal_compare_geometry[
                            "target_center_y"
                        ]
                        layout_box_top_left = horizontal_compare_geometry[
                            "layout_box_top_left"
                        ]
                        layout_collision_mask = None
                    log_message(
                        "Using horizontal layout because forced vertical layout failed",
                        verbose=verbose,
                    )
            except RenderingError:
                zero_padding_layout = _try_zero_padding_layout_after_failure(
                    False,
                    "forced vertical fallback failure",
                )
                if zero_padding_layout is not None:
                    layout_data = zero_padding_layout
                    vertical_stack = False
                else:
                    raise RenderingError(f"Layout optimization failed: {e}") from e
        else:
            force_compare_vertical_text = bool(
                getattr(config, "force_compare_vertical_text", False)
            )
            if not (
                not vertical_stack
                and (
                    force_compare_vertical_text
                    or (
                        config.auto_vertical_text
                        and _should_try_auto_vertical_text(
                            vertical_layout_text,
                            (
                                vertical_compare_geometry["max_render_width"]
                                if vertical_compare_geometry is not None
                                else max_render_width
                            ),
                            (
                                vertical_compare_geometry["max_render_height"]
                                if vertical_compare_geometry is not None
                                else max_render_height
                            ),
                            config,
                            ignore_word_count=auto_vertical_ignore_word_count,
                        )
                    )
                )
            ):
                zero_padding_layout = _try_zero_padding_layout_after_failure(
                    vertical_stack,
                    "layout failure",
                )
                if zero_padding_layout is not None:
                    layout_data = zero_padding_layout
                else:
                    raise RenderingError(f"Layout optimization failed: {e}") from e

            if layout_data is None:
                try:
                    layout_data = _find_layout(True)
                    vertical_stack = True
                    log_message(
                        "Using auto vertical text layout after horizontal layout failed",
                        verbose=verbose,
                    )
                except RenderingError:
                    zero_padding_layout = _try_zero_padding_layout_after_failure(
                        True,
                        "auto vertical layout failure",
                    )
                    if zero_padding_layout is not None:
                        layout_data = zero_padding_layout
                        vertical_stack = True
                    else:
                        raise RenderingError(f"Layout optimization failed: {e}") from e
    else:
        force_compare_vertical_text = bool(
            getattr(config, "force_compare_vertical_text", False)
        )
        should_compare_auto_vertical = (
            config.auto_vertical_text
            and _should_try_auto_vertical_text(
                vertical_layout_text,
                (
                    vertical_compare_geometry["max_render_width"]
                    if vertical_compare_geometry is not None
                    else max_render_width
                ),
                (
                    vertical_compare_geometry["max_render_height"]
                    if vertical_compare_geometry is not None
                    else max_render_height
                ),
                config,
                ignore_word_count=auto_vertical_ignore_word_count,
            )
        )
        tall_bubble_vertical_policy = _should_use_tall_bubble_vertical_policy(
            vertical_layout_text,
            bubble_width,
            bubble_height,
            (
                vertical_compare_geometry["max_render_width"]
                if vertical_compare_geometry is not None
                else max_render_width
            ),
            (
                vertical_compare_geometry["max_render_height"]
                if vertical_compare_geometry is not None
                else max_render_height
            ),
            config,
            ignore_word_count=auto_vertical_ignore_word_count,
        )
        if (
            not vertical_stack
            and (force_compare_vertical_text or should_compare_auto_vertical)
        ):
            try:
                layout_data = _finalize_orientation_candidate(
                    layout_data,
                    False,
                    horizontal_compare_geometry,
                )
                vertical_layout_data = _find_layout(True)
                vertical_layout_data = _finalize_orientation_candidate(
                    vertical_layout_data,
                    True,
                    vertical_compare_geometry,
                )
                orientation_candidates_finalized = True
                orientation_comparison_completed = True
                _log_orientation_comparison(
                    layout_data,
                    vertical_layout_data,
                    horizontal_compare_geometry,
                    vertical_compare_geometry,
                )
                max_horizontal_fill = AUTO_VERTICAL_MAX_HORIZONTAL_FILL
                min_fill_gain = AUTO_VERTICAL_MIN_FILL_GAIN
                min_font_size_ratio = 1.0
                ignore_min_font_size_gain = (
                    short_text_ignores_auto_vertical_font_gain
                    or bool(
                        getattr(config, "ignore_auto_vertical_font_gain", False)
                    )
                )
                if force_compare_vertical_text:
                    try:
                        max_horizontal_fill = float(
                            getattr(
                                config,
                                "forced_vertical_max_horizontal_fill",
                                AUTO_VERTICAL_MAX_HORIZONTAL_FILL,
                            )
                        )
                    except (TypeError, ValueError):
                        max_horizontal_fill = AUTO_VERTICAL_MAX_HORIZONTAL_FILL
                    try:
                        min_fill_gain = float(
                            getattr(
                                config,
                                "forced_vertical_min_fill_gain",
                                AUTO_VERTICAL_MIN_FILL_GAIN,
                            )
                        )
                    except (TypeError, ValueError):
                        min_fill_gain = AUTO_VERTICAL_MIN_FILL_GAIN
                    try:
                        min_font_size_ratio = float(
                            getattr(config, "forced_vertical_min_font_size_ratio", 1.0)
                        )
                    except (TypeError, ValueError):
                        min_font_size_ratio = 1.0
                if tall_bubble_vertical_policy:
                    max_horizontal_fill = (
                        TALL_BUBBLE_VERTICAL_MAX_HORIZONTAL_FILL
                    )
                    min_fill_gain = TALL_BUBBLE_VERTICAL_MIN_FILL_GAIN
                    min_font_size_ratio = (
                        TALL_BUBBLE_VERTICAL_MIN_FONT_SIZE_RATIO
                    )
                    ignore_min_font_size_gain = True
                    log_message(
                        "Using tall-bubble vertical comparison policy "
                        f"(aspect={bubble_height / bubble_width:.2f}, "
                        f"max_horizontal_fill={max_horizontal_fill:.2f}, "
                        f"min_fill_gain={min_fill_gain:.2f}, "
                        f"min_font_ratio={min_font_size_ratio:.2f})",
                        verbose=verbose,
                    )
                if _auto_vertical_layout_is_better(
                    layout_data,
                    vertical_layout_data,
                    _geometry_max_height(horizontal_compare_geometry, max_render_height),
                    max_horizontal_fill,
                    min_fill_gain,
                    min_font_size_ratio,
                    vertical_max_render_height=_geometry_max_height(
                        vertical_compare_geometry, max_render_height
                    ),
                    ignore_min_font_size_gain=ignore_min_font_size_gain,
                ):
                    layout_data = vertical_layout_data
                    vertical_stack = True
                    log_message("Using auto vertical text layout", verbose=verbose)
                elif force_compare_vertical_text:
                    log_message(
                        "Keeping horizontal layout after vertical comparison",
                        verbose=verbose,
                    )
            except RenderingError:
                if force_compare_vertical_text:
                    log_message(
                        "Vertical comparison failed; keeping horizontal layout",
                        verbose=verbose,
                    )
                pass

    if (
        vertical_stack
        and getattr(config, "force_vertical_fallback_only", False)
        and forced_vertical_split_text
        and (layout_data.get("font_size") or 0) <= forced_vertical_retry_font_size
        and "\n" not in vertical_layout_text
    ):
        split_layout_data = _try_forced_vertical_split(
            f"font size {layout_data.get('font_size')} <= {forced_vertical_retry_font_size}"
        )
        if split_layout_data is not None:
            layout_data = split_layout_data

    compare_forced_vertical_layout = bool(
        vertical_stack
        and not orientation_comparison_completed
        and not getattr(config, "force_vertical_fallback_only", False)
        and (
            (
                config.auto_vertical_text
                and not (rotation_deg and abs(rotation_deg) > 0.01)
            )
            or getattr(config, "compare_forced_vertical_layout", False)
        )
    )
    if compare_forced_vertical_layout:
        try:
            horizontal_layout_data = _find_layout(
                False,
                geometry_override=horizontal_compare_geometry,
            )
            horizontal_layout_data = _finalize_orientation_candidate(
                horizontal_layout_data,
                False,
                horizontal_compare_geometry,
            )
            layout_data = _finalize_orientation_candidate(
                layout_data,
                True,
                vertical_compare_geometry,
            )
            orientation_candidates_finalized = True
            orientation_comparison_completed = True
            _log_orientation_comparison(
                horizontal_layout_data,
                layout_data,
                horizontal_compare_geometry,
                vertical_compare_geometry,
            )
            try:
                forced_vertical_max_horizontal_fill = float(
                    getattr(
                        config,
                        "forced_vertical_max_horizontal_fill",
                        AUTO_VERTICAL_MAX_HORIZONTAL_FILL,
                    )
                )
            except (TypeError, ValueError):
                forced_vertical_max_horizontal_fill = AUTO_VERTICAL_MAX_HORIZONTAL_FILL
            try:
                forced_vertical_min_fill_gain = float(
                    getattr(
                        config,
                        "forced_vertical_min_fill_gain",
                        AUTO_VERTICAL_MIN_FILL_GAIN,
                    )
                )
            except (TypeError, ValueError):
                forced_vertical_min_fill_gain = AUTO_VERTICAL_MIN_FILL_GAIN
            try:
                forced_vertical_min_font_size_ratio = float(
                    getattr(config, "forced_vertical_min_font_size_ratio", 1.0)
                )
            except (TypeError, ValueError):
                forced_vertical_min_font_size_ratio = 1.0
            vertical_better = _auto_vertical_layout_is_better(
                horizontal_layout_data,
                layout_data,
                _geometry_max_height(horizontal_compare_geometry, max_render_height),
                forced_vertical_max_horizontal_fill,
                forced_vertical_min_fill_gain,
                forced_vertical_min_font_size_ratio,
                vertical_max_render_height=_geometry_max_height(
                    vertical_compare_geometry, max_render_height
                ),
                ignore_min_font_size_gain=(
                    short_text_ignores_auto_vertical_font_gain
                    or bool(
                        getattr(config, "ignore_auto_vertical_font_gain", False)
                    )
                ),
            )
            if not vertical_better:
                layout_data = horizontal_layout_data
                vertical_stack = False
                if horizontal_compare_geometry is not None:
                    x1, y1, x2, y2 = horizontal_compare_geometry["bbox"]
                    bubble_width = x2 - x1
                    bubble_height = y2 - y1
                    max_render_width = horizontal_compare_geometry[
                        "max_render_width"
                    ]
                    max_render_height = horizontal_compare_geometry[
                        "max_render_height"
                    ]
                    target_center_x = horizontal_compare_geometry["target_center_x"]
                    target_center_y = horizontal_compare_geometry["target_center_y"]
                    layout_box_top_left = horizontal_compare_geometry[
                        "layout_box_top_left"
                    ]
                    layout_collision_mask = None
                if not (rotation_deg and abs(rotation_deg) > 0.01):
                    rotation_deg = 0.0
                log_message(
                    "Using horizontal layout because forced vertical layout "
                    "did not pass auto-vertical quality checks",
                    verbose=verbose,
                )
        except RenderingError:
            pass

    selected_geometry = (
        vertical_compare_geometry
        if layout_data.get("orientation") == "vertical"
        else horizontal_compare_geometry
    )
    _apply_render_geometry(selected_geometry)

    if not orientation_candidates_finalized:
        retry_count = _tight_fit_shrink_max_retries(config)
    else:
        retry_count = 0
    for shrink_attempt in range(retry_count):
        if not _should_retry_tight_fit_shrink(
            layout_data, max_render_width, max_render_height, config
        ):
            break
        retry_max_font_size = _tight_fit_retry_max_font_size(layout_data, config)
        if retry_max_font_size < int(config.min_font_size):
            break
        try:
            retry_vertical_stack = layout_data.get("orientation") == "vertical"
            original_font_size = layout_data.get("font_size")
            original_width_ratio, original_height_ratio = _layout_fill_ratios(
                layout_data, max_render_width, max_render_height
            )
            retry_layout_data = _find_layout(
                retry_vertical_stack,
                max_font_size_override=retry_max_font_size,
            )
            if (
                int(retry_layout_data.get("font_size") or 0)
                < _tight_fit_shrink_min_font_size(config)
            ):
                break
            layout_data = retry_layout_data
            if retry_vertical_stack:
                vertical_stack = True
            log_message(
                "Retried tight text fit "
                f"attempt {shrink_attempt + 1} "
                f"from size {original_font_size} to {layout_data.get('font_size')} "
                f"(fill {original_width_ratio:.2f}w/{original_height_ratio:.2f}h)",
                verbose=verbose,
            )
        except RenderingError:
            break

    if (
        not orientation_candidates_finalized
        and _layout_has_wrapped_symbol_orphan(layout_data)
    ):
        retry_vertical_stack = layout_data.get("orientation") == "vertical"
        original_font_size = layout_data.get("font_size")
        for retry_max_font_size in _wrapped_symbol_retry_max_font_sizes(
            layout_data, config
        ):
            try:
                retry_layout_data = _find_layout(
                    retry_vertical_stack,
                    max_font_size_override=retry_max_font_size,
                )
            except RenderingError:
                continue

            if not _layout_has_wrapped_symbol_orphan(retry_layout_data):
                layout_data = retry_layout_data
                if retry_vertical_stack:
                    vertical_stack = True
                log_message(
                    "Retried wrapped punctuation fit "
                    f"from size {original_font_size} "
                    f"to {layout_data.get('font_size')}",
                    verbose=verbose,
                )
                break

    if (
        not zero_padding_retry_applied
        and _should_retry_zero_padding_for_small_text(layout_data, layout_text, config)
    ):
        zero_padding_geometry = _zero_padding_geometry_for_current_box()
        if zero_padding_geometry is not None:
            retry_vertical_stack = layout_data.get("orientation") == "vertical"
            original_font_size = int(layout_data.get("font_size") or 0)
            try:
                zero_padding_layout = _find_layout(
                    retry_vertical_stack,
                    geometry_override=zero_padding_geometry,
                    use_compare_geometry=False,
                )
            except RenderingError:
                zero_padding_layout = None
            if zero_padding_layout is not None:
                retry_font_size = int(zero_padding_layout.get("font_size") or 0)
                if retry_font_size >= original_font_size:
                    layout_data = zero_padding_layout
                    vertical_stack = retry_vertical_stack
                    _apply_render_geometry(zero_padding_geometry)
                    log_message(
                        "Retried small text with zero padding "
                        f"from size {original_font_size} to {retry_font_size}",
                        verbose=verbose,
                    )

    text_alignment = str(getattr(config, "text_alignment", "center") or "center").lower()
    if text_alignment not in {"left", "center", "right"}:
        text_alignment = "center"
    if layout_data.get("orientation") != "vertical":
        layout_data["text_alignment"] = text_alignment
        layout_data["alignment_width"] = float(
            max_render_width or layout_data.get("max_line_width") or 0.0
        )

    try:
        target_center_x += float(layout_data.get("position_offset_x") or 0.0)
        target_center_y += float(layout_data.get("position_offset_y") or 0.0)
    except (TypeError, ValueError):
        pass

    if layout_only:
        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
        result = Image.new("RGBA", (1, 1))
        result.info["font_size"] = layout_data["font_size"]
        result.info["orientation"] = layout_data.get("orientation", "horizontal")
        result.info["render_bbox"] = (x1, y1, x2, y2)
        result.info["position_offset"] = (
            layout_data.get("position_offset_x", 0.0),
            layout_data.get("position_offset_y", 0.0),
        )
        return result

    required_styles = {"regular"} | {
        style for _, style in parse_styled_segments(clean_text)
    }
    log_message(f"Required styles: {sorted(required_styles)}", verbose=verbose)

    loaded_typefaces = {"regular": regular_typeface}
    loaded_hb_faces = {"regular": regular_hb_face}

    for style in ["italic", "bold", "bold_italic"]:
        if style in required_styles:
            font_path = font_variants.get(style)
            if font_path:
                log_message(f"Loading {style}: {font_path.name}", verbose=verbose)
                _, typeface, hb_face = load_font_resources(str(font_path))
                if typeface and hb_face:
                    loaded_typefaces[style] = typeface
                    loaded_hb_faces[style] = hb_face
                else:
                    log_message(
                        f"Failed to load {style} variant, using regular",
                        verbose=verbose,
                    )
            else:
                log_message(
                    f"Style '{style}' not found, using regular",
                    verbose=verbose,
                )

    if fallback_font_path:
        _, fallback_typeface, fallback_hb_face = load_font_resources(
            str(fallback_font_path)
        )
        if fallback_typeface and fallback_hb_face:
            loaded_typefaces[FALLBACK_FONT_KEY] = fallback_typeface
            loaded_hb_faces[FALLBACK_FONT_KEY] = fallback_hb_face

    # Determine text color contrast based on sampled background brightness
    text_color = skia.ColorBLACK
    if text_color_rgb is not None:
        text_color = skia.Color(text_color_rgb[0], text_color_rgb[1], text_color_rgb[2])
    elif bubble_color_bgr is not None:
        try:
            # bubble_color_bgr may be a grayscale proxy; treat as BGR
            bg_brightness = (
                bubble_color_bgr[0] + bubble_color_bgr[1] + bubble_color_bgr[2]
            ) / 3.0
            # If background is dark, use white text; if light, use black
            text_color = (
                skia.ColorWHITE
                if bg_brightness < GRAYSCALE_MIDPOINT
                else skia.ColorBLACK
            )
        except Exception:
            text_color = skia.ColorBLACK

    skia_bg_color = None
    if text_background_color is not None:
        skia_bg_color = skia.Color(
            text_background_color[0],
            text_background_color[1],
            text_background_color[2],
        )

    skia_outline_color = None
    outline_color_rgb = getattr(config, "outline_color_rgb", None)
    if outline_color_rgb is not None:
        try:
            skia_outline_color = skia.Color(
                int(outline_color_rgb[0]),
                int(outline_color_rgb[1]),
                int(outline_color_rgb[2]),
            )
        except Exception:
            skia_outline_color = None

    # Apply supersampling if enabled
    if config.supersampling_factor > 1:
        log_message(
            f"Using supersampling factor {config.supersampling_factor}", verbose=verbose
        )

        # Crop the bbox region from the original image
        # Ensure bbox coordinates are within image bounds
        img_width, img_height = pil_image.size
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(img_width, x2)
        crop_y2 = min(img_height, y2)

        cropped_region = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_width = crop_x2 - crop_x1
        crop_height = crop_y2 - crop_y1

        # Upscale the cropped region
        factor = config.supersampling_factor
        scaled_width = int(crop_width * factor)
        scaled_height = int(crop_height * factor)
        upscaled_region = cropped_region.resize(
            (scaled_width, scaled_height), Image.Resampling.LANCZOS
        )

        # Scale coordinates relative to bbox origin
        scaled_target_center_x = (target_center_x - crop_x1) * factor
        scaled_target_center_y = (target_center_y - crop_y1) * factor

        # Scale font size in layout_data for rendering
        scaled_layout_data = layout_data.copy()
        scaled_layout_data["lines"] = [
            line_data.copy() for line_data in layout_data["lines"]
        ]
        scaled_layout_data["font_size"] = layout_data["font_size"] * factor
        scaled_layout_data["line_height"] = layout_data["line_height"] * factor
        scaled_layout_data["max_line_width"] = layout_data["max_line_width"] * factor
        if layout_data.get("alignment_width") is not None:
            scaled_layout_data["alignment_width"] = (
                layout_data["alignment_width"] * factor
            )
        if layout_data.get("block_height") is not None:
            scaled_layout_data["block_height"] = layout_data["block_height"] * factor
        if layout_data.get("block_width") is not None:
            scaled_layout_data["block_width"] = layout_data["block_width"] * factor

        # Scale per-line and per-glyph measurements
        for line_data in scaled_layout_data["lines"]:
            for key in (
                "width",
                "height",
                "advance_height",
                "baseline_offset_y",
                "left",
                "right",
                "top",
                "bottom",
                "origin_y",
                "origin_x",
                "advance_width",
                "center_offset",
                "alignment_center_offset",
                "original_left",
                "original_right",
                "original_top",
                "original_bottom",
                "ellipsis_dot_radius",
                "ellipsis_dot_step",
                "centered_dot_radius",
                "dash_thickness",
                "dash_length",
                "quote_draw_width",
                "quote_draw_height",
                "quote_stroke_width",
                "quote_x_offset",
                "quote_y_offset",
            ):
                if key in line_data:
                    line_data[key] = line_data[key] * factor

        # Scale metrics - create a simple object with scaled attributes
        original_metrics = layout_data["metrics"]

        class ScaledMetrics:
            def __init__(self, original, scale_factor):
                self.fAscent = original.fAscent * scale_factor
                self.fDescent = original.fDescent * scale_factor
                # Preserve other attributes if they exist
                if hasattr(original, "fLeading"):
                    self.fLeading = original.fLeading * scale_factor
                if hasattr(original, "fXMin"):
                    self.fXMin = original.fXMin * scale_factor
                if hasattr(original, "fXMax"):
                    self.fXMax = original.fXMax * scale_factor
                if hasattr(original, "fYMin"):
                    self.fYMin = original.fYMin * scale_factor
                if hasattr(original, "fYMax"):
                    self.fYMax = original.fYMax * scale_factor

        scaled_metrics = ScaledMetrics(original_metrics, factor)
        scaled_layout_data["metrics"] = scaled_metrics

        # Create Skia surface from upscaled region
        try:
            scaled_surface = pil_to_skia_surface(upscaled_region)
        except RenderingError as e:
            raise RenderingError(f"Scaled surface preparation failed: {e}") from e

        # Render text at high resolution
        success = draw_layout(
            scaled_surface,
            scaled_layout_data,
            (
                0.0
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else scaled_target_center_x
            ),
            (
                0.0
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else scaled_target_center_y
            ),
            loaded_typefaces,
            loaded_hb_faces,
            regular_typeface,
            regular_hb_face,
            features_to_enable,
            text_color,
            config.use_subpixel_rendering,
            config.font_hinting,
            config.outline_width * factor,  # Scale outline width too
            skia_outline_color,
            verbose,
            pre_translate_x=(
                float(scaled_target_center_x)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_translate_y=(
                float(scaled_target_center_y)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_rotate_deg=(
                float(rotation_deg)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            text_background_color=skia_bg_color,
            font_cmaps=font_cmaps,
        )

        if not success:
            log_message("Drawing failed", always_print=True)
            raise RenderingError("Text drawing failed")

        # Convert back to PIL and downscale
        try:
            scaled_pil_result = skia_surface_to_pil(scaled_surface)
        except RenderingError as e:
            raise RenderingError(f"Scaled conversion failed: {e}") from e

        # Downscale using LANCZOS for high quality
        downscaled_result = scaled_pil_result.resize(
            (crop_width, crop_height), Image.Resampling.LANCZOS
        )

        # Paste the result back onto the original image
        final_pil_image = pil_image.copy()
        final_pil_image.paste(downscaled_result, (crop_x1, crop_y1))

        log_message(
            f"Rendered at size {layout_data['font_size']} with {factor}x supersampling",
            verbose=verbose,
        )
        final_pil_image.info["font_size"] = layout_data["font_size"]
        final_pil_image.info["orientation"] = layout_data.get("orientation", "horizontal")
        final_pil_image.info["render_bbox"] = (x1, y1, x2, y2)
        return final_pil_image
    else:
        # Normal rendering path (no supersampling)
        try:
            surface = pil_to_skia_surface(pil_image)
        except RenderingError as e:
            raise RenderingError(f"Surface preparation failed: {e}") from e

        # Delegate rotation/translate to drawing_engine so Skia state is consistent
        success = draw_layout(
            surface,
            layout_data,
            0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_x,
            0.0 if (rotation_deg and abs(rotation_deg) > 0.01) else target_center_y,
            loaded_typefaces,
            loaded_hb_faces,
            regular_typeface,
            regular_hb_face,
            features_to_enable,
            text_color,
            config.use_subpixel_rendering,
            config.font_hinting,
            config.outline_width,
            skia_outline_color,
            verbose,
            pre_translate_x=(
                float(target_center_x)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_translate_y=(
                float(target_center_y)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            pre_rotate_deg=(
                float(rotation_deg)
                if (rotation_deg and abs(rotation_deg) > 0.01)
                else 0.0
            ),
            text_background_color=skia_bg_color,
            font_cmaps=font_cmaps,
        )

        if not success:
            log_message("Drawing failed", always_print=True)
            raise RenderingError("Text drawing failed")

        try:
            final_pil_image = skia_surface_to_pil(surface)
        except RenderingError as e:
            raise RenderingError(f"Final conversion failed: {e}") from e

        log_message(f"Rendered at size {layout_data['font_size']}", verbose=verbose)
        final_pil_image.info["font_size"] = layout_data["font_size"]
        final_pil_image.info["orientation"] = layout_data.get("orientation", "horizontal")
        final_pil_image.info["render_bbox"] = (x1, y1, x2, y2)
        return final_pil_image
