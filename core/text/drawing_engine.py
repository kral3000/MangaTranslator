import os
import threading
from typing import Dict, Optional, Tuple

import skia
import uharfbuzz as hb
from PIL import Image

from core.text.font_manager import LRUCache, load_font_data
from core.text.layout_engine import shape_line
from core.text.font_fallback import FALLBACK_FONT_KEY, split_text_by_font_support
from core.text.text_processing import is_rtl_script, parse_styled_segments
from utils.exceptions import FontError, RenderingError
from utils.logging import log_message

_typeface_cache = LRUCache(max_size=50)
_hb_face_cache = LRUCache(max_size=50)
_font_cache_lock = threading.RLock()

DASH_LINE_CHARS = set("-‐‑‒–—―−─━ーｰㅡ")
HORIZONTAL_DOT_ELLIPSIS_CHARS = {".", "\uff0e"}
HORIZONTAL_ELLIPSIS_CHARS = {"\u2025", "\u2026", "\u22ef", "\ufe19", "\u22ee"}
HB_26_6_SCALE_FACTOR = 64.0
VERTICAL_DRAW_QUOTE_OPEN_GLYPHS = {
    "'": "\u300c",
    '"': "\u300e",
    "\u2018": "\u300c",
    "\u201c": "\u300e",
}
VERTICAL_DRAW_QUOTE_CLOSE_GLYPHS = {
    "'": "\u300d",
    '"': "\u300f",
    "\u2019": "\u300d",
    "\u201d": "\u300f",
}
VERTICAL_DRAW_SYMMETRIC_QUOTES = {"'", '"'}
VERTICAL_DRAW_QUOTE_DISPLAY_INFO = {
    "\u300c": ("open", "single"),
    "\u300d": ("close", "single"),
    "\u300e": ("open", "double"),
    "\u300f": ("close", "double"),
}


def _vertical_draw_quote_text(text: str, quote_stack) -> str:
    """Final guard: render straight quotes as CJK corner quotes in vertical text."""
    if len(text) != 1:
        return text
    if text not in VERTICAL_DRAW_QUOTE_OPEN_GLYPHS and text not in VERTICAL_DRAW_QUOTE_CLOSE_GLYPHS:
        return text
    if text not in VERTICAL_DRAW_SYMMETRIC_QUOTES:
        return VERTICAL_DRAW_QUOTE_CLOSE_GLYPHS.get(
            text, VERTICAL_DRAW_QUOTE_OPEN_GLYPHS.get(text, text)
        )
    if quote_stack and quote_stack[-1] == text:
        quote_stack.pop()
        return VERTICAL_DRAW_QUOTE_CLOSE_GLYPHS[text]
    quote_stack.append(text)
    return VERTICAL_DRAW_QUOTE_OPEN_GLYPHS[text]


def _draw_vertical_corner_quote(
    canvas,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    role: str,
    kind: str,
    paint,
    outline_paint,
    stroke_width: float,
    outline_width: float,
) -> None:
    def make_quote_paint(base_paint, width_value: float):
        quote_paint = skia.Paint(
            AntiAlias=True,
            Color=base_paint.getColor(),
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=width_value,
            StrokeJoin=skia.Paint.kRound_Join,
        )
        return quote_paint

    def draw_marks(mark_paint):
        left = float(center_x - width / 2.0)
        right = float(center_x + width / 2.0)
        top = float(center_y - height / 2.0)
        bottom = float(center_y + height / 2.0)

        if role == "close":
            canvas.drawLine(left, bottom, right, bottom, mark_paint)
            canvas.drawLine(right, top, right, bottom, mark_paint)
            if kind == "double":
                inset = max(stroke_width * 2.0, min(width, height) * 0.22)
                canvas.drawLine(left, bottom - inset, right - inset, bottom - inset, mark_paint)
                canvas.drawLine(right - inset, top + inset, right - inset, bottom - inset, mark_paint)
        else:
            canvas.drawLine(left, top, right, top, mark_paint)
            canvas.drawLine(left, top, left, bottom, mark_paint)
            if kind == "double":
                inset = max(stroke_width * 2.0, min(width, height) * 0.22)
                canvas.drawLine(left + inset, top + inset, right, top + inset, mark_paint)
                canvas.drawLine(left + inset, top + inset, left + inset, bottom, mark_paint)

    if outline_paint:
        draw_marks(make_quote_paint(outline_paint, stroke_width + max(1.0, outline_width * 2.0)))
    draw_marks(make_quote_paint(paint, stroke_width))


def _draw_attached_vertical_quotes(
    canvas,
    line_data: Dict,
    unit_center_x: float,
    baseline_x: float,
    baseline_y: float,
    final_font_size: float,
    paint,
    outline_paint,
    outline_width: float,
    verbose: bool = False,
) -> None:
    attached_quotes = line_data.get("attached_vertical_quotes") or []
    if not attached_quotes:
        return

    left = float(line_data.get("left", 0.0))
    right = float(line_data.get("right", line_data.get("width", 0.0)))
    top = float(line_data.get("top", -line_data.get("height", 0.0)))
    bottom = float(line_data.get("bottom", 0.0))

    if line_data.get("rotate_in_vertical"):
        target_left = float(unit_center_x + left)
        target_right = float(unit_center_x + right)
        target_top = float(baseline_y + top)
        target_bottom = float(baseline_y + bottom)
    else:
        target_left = float(baseline_x + left)
        target_right = float(baseline_x + right)
        target_top = float(baseline_y + top)
        target_bottom = float(baseline_y + bottom)

    margin = max(1.2, float(final_font_size) * 0.05)
    vertical_nudge = float(final_font_size) * 0.06

    for quote in attached_quotes:
        role = str(quote.get("quote_role", "open"))
        kind = str(quote.get("quote_kind", "single"))
        quote_width = max(
            float(quote.get("quote_width", 0.0)),
            float(final_font_size) * 0.34,
        )
        quote_height = max(
            float(quote.get("quote_height", 0.0)),
            float(final_font_size) * 0.46,
        )
        quote_stroke_width = max(
            float(quote.get("quote_stroke_width", 0.0)),
            float(final_font_size) * 0.068,
        )

        if role == "close":
            right_edge = target_right + margin
            bottom_edge = target_bottom + vertical_nudge
            center_x = right_edge - quote_width / 2.0
            center_y = bottom_edge - quote_height / 2.0
        else:
            left_edge = target_left - margin
            top_edge = target_top - vertical_nudge
            center_x = left_edge + quote_width / 2.0
            center_y = top_edge + quote_height / 2.0

        _draw_vertical_corner_quote(
            canvas,
            float(center_x),
            float(center_y),
            quote_width,
            quote_height,
            role,
            kind,
            paint,
            outline_paint,
            quote_stroke_width,
            outline_width,
        )
        log_message(
            f"Rendered attached vertical quote '{quote.get('quote_text', '')}'",
            verbose=verbose,
        )


def _is_dash_line_text(text: str) -> bool:
    return len(text) >= 2 and all(ch in DASH_LINE_CHARS for ch in text)


def _is_dot_ellipsis_text(text: str) -> bool:
    return len(text) >= 2 and all(ch in HORIZONTAL_DOT_ELLIPSIS_CHARS for ch in text)


def _is_horizontal_ellipsis_text(text: str) -> bool:
    return _is_dot_ellipsis_text(text) or (
        bool(text) and all(ch in HORIZONTAL_ELLIPSIS_CHARS for ch in text)
    )


def _ellipsis_dot_count(text: str) -> int:
    count = 0
    for ch in text:
        if ch in HORIZONTAL_DOT_ELLIPSIS_CHARS:
            count += 1
        elif ch == "\u2025":
            count += 2
        elif ch in HORIZONTAL_ELLIPSIS_CHARS:
            count += 3
    return max(2, count)


def _iter_special_text_parts(text: str):
    normal = []

    def flush_normal():
        if normal:
            value = "".join(normal)
            normal.clear()
            return value
        return None

    i = 0
    while i < len(text):
        ch = text[i]
        if ch in DASH_LINE_CHARS:
            j = i + 1
            while j < len(text) and text[j] in DASH_LINE_CHARS:
                j += 1
            run = text[i:j]
            if _is_dash_line_text(run):
                value = flush_normal()
                if value:
                    yield value, "text"
                yield run, "dash"
            else:
                normal.append(run)
            i = j
            continue
        if ch in HORIZONTAL_DOT_ELLIPSIS_CHARS:
            j = i + 1
            while j < len(text) and text[j] in HORIZONTAL_DOT_ELLIPSIS_CHARS:
                j += 1
            run = text[i:j]
            if _is_dot_ellipsis_text(run):
                value = flush_normal()
                if value:
                    yield value, "text"
                yield run, "ellipsis"
            else:
                normal.append(run)
            i = j
            continue
        if ch in HORIZONTAL_ELLIPSIS_CHARS:
            j = i + 1
            while j < len(text) and text[j] in HORIZONTAL_ELLIPSIS_CHARS:
                j += 1
            value = flush_normal()
            if value:
                yield value, "text"
            yield text[i:j], "ellipsis"
            i = j
            continue
        normal.append(ch)
        i += 1

    value = flush_normal()
    if value:
        yield value, "text"


def _measure_shaped_bounds(infos, positions, hb_font: hb.Font, font_size: float):
    cursor_x = 0
    left = float("inf")
    right = float("-inf")
    top = float("inf")
    bottom = float("-inf")

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
            glyph_y1 = origin_y - font_size * HB_26_6_SCALE_FACTOR
            glyph_y2 = origin_y
        left = min(left, glyph_x1, glyph_x2)
        right = max(right, glyph_x1, glyph_x2)
        top = min(top, glyph_y1, glyph_y2)
        bottom = max(bottom, glyph_y1, glyph_y2)
        cursor_x += pos.x_advance

    width = max(0.0, cursor_x / HB_26_6_SCALE_FACTOR)
    if left == float("inf") or right == float("-inf"):
        return width, -font_size * 0.2, 0.0
    return (
        width,
        top / HB_26_6_SCALE_FACTOR,
        bottom / HB_26_6_SCALE_FACTOR,
    )


def _draw_filled_rect(canvas, center_x, center_y, width, height, paint_obj):
    rect = skia.Rect.MakeXYWH(
        float(center_x - width / 2.0),
        float(center_y - height / 2.0),
        float(width),
        float(height),
    )
    canvas.drawRect(rect, paint_obj)


def _draw_horizontal_dash(canvas, start_x, baseline_y, width, top, bottom, font_size, paint_obj, outline_paint):
    thickness = max(1.1, float(font_size) * 0.07)
    line_width = max(thickness, float(width))
    center_x = float(start_x) + line_width / 2.0
    center_y = float(baseline_y) + (float(top) + float(bottom)) / 2.0
    if outline_paint:
        _draw_filled_rect(canvas, center_x, center_y, line_width, thickness + 1.0, outline_paint)
    _draw_filled_rect(canvas, center_x, center_y, line_width, thickness, paint_obj)


def _draw_horizontal_ellipsis(canvas, start_x, baseline_y, width, top, bottom, text, font_size, paint_obj, outline_paint):
    dot_count = _ellipsis_dot_count(text)
    radius = max(0.95, float(font_size) * 0.065)
    center_y = float(baseline_y) + (float(top) + float(bottom)) / 2.0
    for idx in range(dot_count):
        dot_x = float(start_x) + float(width) * (idx + 0.5) / dot_count
        if outline_paint:
            canvas.drawCircle(float(dot_x), float(center_y), float(radius + 0.5), outline_paint)
        canvas.drawCircle(float(dot_x), float(center_y), float(radius), paint_obj)


def _draw_vertical_ellipsis(
    canvas,
    center_x,
    center_y,
    dot_count,
    dot_step,
    radius,
    paint_obj,
    outline_paint,
):
    first_offset = -float(dot_step) * (int(dot_count) - 1) / 2.0
    for idx in range(int(dot_count)):
        dot_y = float(center_y) + first_offset + float(dot_step) * idx
        if outline_paint:
            canvas.drawCircle(
                float(center_x), float(dot_y), float(radius + 0.5), outline_paint
            )
        canvas.drawCircle(float(center_x), float(dot_y), float(radius), paint_obj)


def _horizontal_line_start_x(
    target_center_x: float,
    alignment_width: float,
    line_width: float,
    text_alignment: str,
) -> float:
    left = float(target_center_x) - float(alignment_width) / 2.0
    if text_alignment == "left":
        return left
    if text_alignment == "right":
        return left + float(alignment_width) - float(line_width)
    return float(target_center_x) - float(line_width) / 2.0


def load_font_resources(
    font_path: str,
) -> Tuple[bytes, skia.Typeface, hb.Face]:
    """
    Loads font data, Skia Typeface, and HarfBuzz Face, using LRU caching.

    Args:
        font_path: Path to the font file

    Returns:
        Tuple of (font_data, skia_typeface, harfbuzz_face)

    Raises:
        FontError: If font data cannot be loaded or Skia/HarfBuzz resources fail to load
    """
    try:
        font_data = load_font_data(font_path)
    except FontError:
        raise

    with _font_cache_lock:
        typeface = _typeface_cache.get(font_path)
        if typeface is None:
            skia_data = skia.Data.MakeWithoutCopy(font_data)
            typeface = skia.Typeface.MakeFromData(skia_data)
            if typeface is None:
                log_message(
                    f"Skia typeface load failed: {os.path.basename(font_path)}",
                    always_print=True,
                )
                raise FontError(
                    f"Failed to create Skia typeface from font: {font_path}"
                )
            _typeface_cache.put(font_path, typeface)

        hb_face = _hb_face_cache.get(font_path)
        if hb_face is None:
            try:
                hb_face = hb.Face(font_data)
                _hb_face_cache.put(font_path, hb_face)
            except Exception as e:
                log_message(
                    f"HarfBuzz face load failed: {os.path.basename(font_path)}: {e}",
                    always_print=True,
                )
                # Clean up Skia cache if HarfBuzz fails to avoid inconsistent state
                if font_path in _typeface_cache:
                    del _typeface_cache[font_path]
                raise FontError(
                    f"Failed to create HarfBuzz face from font: {font_path}"
                ) from e

    return font_data, typeface, hb_face


def pil_to_skia_surface(pil_image: Image.Image) -> skia.Surface:
    """Converts a PIL image to a Skia Surface.

    Raises:
        RenderingError: If conversion fails
    """
    try:
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")
        skia_image = skia.Image.frombytes(
            pil_image.tobytes(), pil_image.size, skia.kRGBA_8888_ColorType
        )
        if skia_image is None:
            log_message("PIL to Skia conversion failed", always_print=True)
            raise RenderingError("Failed to create Skia image from PIL")
        surface = skia.Surface(pil_image.width, pil_image.height)
        with surface as canvas:
            canvas.drawImage(skia_image, 0, 0)
        return surface
    except Exception as e:
        log_message(f"PIL to Skia conversion error: {e}", always_print=True)
        raise RenderingError("PIL to Skia conversion failed") from e


def skia_surface_to_pil(surface: skia.Surface) -> Image.Image:
    """Converts a Skia Surface back to a PIL image.

    Raises:
        RenderingError: If conversion fails
    """
    try:
        skia_image: Optional[skia.Image] = surface.makeImageSnapshot()
        if skia_image is None:
            log_message("Skia surface snapshot failed", always_print=True)
            raise RenderingError("Failed to create Skia image snapshot")

        skia_image = skia_image.convert(
            alphaType=skia.kUnpremul_AlphaType, colorType=skia.kRGBA_8888_ColorType
        )
        pil_image = Image.fromarray(skia_image)
        return pil_image
    except Exception as e:
        log_message(f"Skia to PIL conversion error: {e}", always_print=True)
        raise RenderingError("Skia to PIL conversion failed") from e


def draw_layout(
    surface: skia.Surface,
    layout_data: Dict,
    target_center_x: float,
    target_center_y: float,
    loaded_typefaces: Dict[str, Optional[skia.Typeface]],
    loaded_hb_faces: Dict[str, Optional[hb.Face]],
    regular_typeface: skia.Typeface,
    regular_hb_face: hb.Face,
    features_to_enable: Dict[str, bool],
    text_color: int,
    use_subpixel_rendering: bool,
    font_hinting: str,
    outline_width: float = 0.0,
    outline_color: Optional[int] = None,
    verbose: bool = False,
    pre_translate_x: float = 0.0,
    pre_translate_y: float = 0.0,
    pre_rotate_deg: float = 0.0,
    text_background_color: Optional[int] = None,
    font_cmaps: Optional[Dict[str, set]] = None,
) -> bool:
    """
    Draws the text layout onto a Skia surface.

    This is a "dumb" renderer - all layout decisions have been made.
    It just executes the drawing plan.

    Args:
        surface: Skia surface to draw on
        layout_data: Layout data from layout_engine (font_size, lines, metrics, etc.)
        target_center_x: X coordinate of the text block center
        target_center_y: Y coordinate of the text block center
        loaded_typefaces: Dictionary of Skia typefaces for each style
        loaded_hb_faces: Dictionary of HarfBuzz faces for each style
        regular_typeface: Regular Skia typeface (fallback)
        regular_hb_face: Regular HarfBuzz face (fallback)
        features_to_enable: HarfBuzz features to enable
        text_color: Skia color for the text
        use_subpixel_rendering: Whether to use subpixel rendering
        font_hinting: Font hinting level ("none", "slight", "normal", "full")
        outline_width: Width of text outline (0 = no outline)
        verbose: Whether to print detailed logs

    Returns:
        True if successful, False otherwise
    """
    try:
        final_font_size = layout_data["font_size"]
        final_lines_data = layout_data["lines"]
        final_metrics = layout_data["metrics"]
        final_max_line_width = layout_data["max_line_width"]
        final_line_height = layout_data["line_height"]

        for line_data in final_lines_data:
            line_data["segments"] = parse_styled_segments(
                line_data["text_with_markers"]
            )

        hinting_map = {
            "none": skia.FontHinting.kNone,
            "slight": skia.FontHinting.kSlight,
            "normal": skia.FontHinting.kNormal,
            "full": skia.FontHinting.kFull,
        }
        skia_hinting = hinting_map.get(font_hinting.lower(), skia.FontHinting.kNone)

        paint = skia.Paint(AntiAlias=True, Color=text_color)

        outline_paint = None
        if outline_width > 0:
            r = skia.ColorGetR(text_color)
            g = skia.ColorGetG(text_color)
            b = skia.ColorGetB(text_color)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            if outline_color is None:
                outline_color = skia.ColorBLACK if lum >= 80 else skia.ColorWHITE

            outline_paint = skia.Paint(
                AntiAlias=True,
                Color=outline_color,
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=outline_width,
                StrokeJoin=skia.Paint.kRound_Join,
            )

        text_alignment = str(layout_data.get("text_alignment") or "center").lower()
        alignment_width = max(
            float(final_max_line_width),
            float(layout_data.get("alignment_width") or final_max_line_width),
        )

        num_lines = len(final_lines_data)
        if num_lines > 0:
            total_visual_height = (
                (num_lines - 1) * final_line_height
                - final_metrics.fAscent
                + final_metrics.fDescent
            )
            block_top_y = target_center_y - (total_visual_height / 2.0)
            first_baseline_y = block_top_y - final_metrics.fAscent
        else:
            first_baseline_y = (
                target_center_y - (final_metrics.fAscent + final_metrics.fDescent) / 2.0
            )

        log_message(
            f"Rendering at size {final_font_size}, center: ({target_center_x:.0f}, {target_center_y:.0f})",
            verbose=verbose,
        )

        def _resolve_style_resources(
            resource_key: str,
            requested_style: Optional[str] = None,
        ):
            requested_style = requested_style or resource_key
            wants_bold = requested_style in {"bold", "bold_italic"}
            wants_italic = requested_style in {"italic", "bold_italic"}

            if resource_key == FALLBACK_FONT_KEY:
                typeface = loaded_typefaces.get(FALLBACK_FONT_KEY)
                hb_face = loaded_hb_faces.get(FALLBACK_FONT_KEY)
                if typeface and hb_face:
                    return typeface, hb_face, wants_bold, wants_italic

            typeface = loaded_typefaces.get(resource_key)
            hb_face = loaded_hb_faces.get(resource_key)
            if typeface and hb_face:
                actual_bold = resource_key in {"bold", "bold_italic"}
                actual_italic = resource_key in {"italic", "bold_italic"}
                return (
                    typeface,
                    hb_face,
                    wants_bold and not actual_bold,
                    wants_italic and not actual_italic,
                )

            if requested_style == "bold_italic":
                typeface = loaded_typefaces.get("bold_italic")
                hb_face = loaded_hb_faces.get("bold_italic")
                if typeface and hb_face:
                    return typeface, hb_face, False, False
                typeface = loaded_typefaces.get("bold")
                hb_face = loaded_hb_faces.get("bold")
                if typeface and hb_face:
                    return typeface, hb_face, False, True
                typeface = loaded_typefaces.get("italic")
                hb_face = loaded_hb_faces.get("italic")
                if typeface and hb_face:
                    return typeface, hb_face, True, False
            elif requested_style == "bold":
                typeface = loaded_typefaces.get("bold")
                hb_face = loaded_hb_faces.get("bold")
                if typeface and hb_face:
                    return typeface, hb_face, False, False
            elif requested_style == "italic":
                typeface = loaded_typefaces.get("italic")
                hb_face = loaded_hb_faces.get("italic")
                if typeface and hb_face:
                    return typeface, hb_face, False, False
            return regular_typeface, regular_hb_face, wants_bold, wants_italic

        def _apply_synthetic_style(
            font: skia.Font,
            synthetic_bold: bool,
            synthetic_italic: bool,
        ) -> None:
            if synthetic_bold and hasattr(font, "setEmbolden"):
                font.setEmbolden(True)
            if synthetic_italic and hasattr(font, "setSkewX"):
                font.setSkewX(-0.18)

        def _synthetic_bold_paint(
            synthetic_bold: bool,
            font_size: float,
        ) -> Optional[skia.Paint]:
            if not synthetic_bold:
                return None
            return skia.Paint(
                AntiAlias=True,
                Color=text_color,
                Style=skia.Paint.kStroke_Style,
                StrokeWidth=max(0.9, float(font_size) * 0.035),
                StrokeJoin=skia.Paint.kRound_Join,
            )

        with surface as canvas:
            # Apply optional pre-transform (used for rotated OSB rendering)
            need_transform = (
                abs(pre_translate_x) > 1e-3
                or abs(pre_translate_y) > 1e-3
                or abs(pre_rotate_deg) > 1e-3
            )
            if need_transform:
                canvas.save()
                if abs(pre_translate_x) > 1e-3 or abs(pre_translate_y) > 1e-3:
                    canvas.translate(float(pre_translate_x), float(pre_translate_y))
                if abs(pre_rotate_deg) > 1e-3:
                    canvas.rotate(float(pre_rotate_deg))

            bg_paint = None
            if text_background_color is not None:
                bg_paint = skia.Paint(AntiAlias=False, Color=text_background_color)

            if layout_data.get("orientation") == "vertical":
                block_height = layout_data.get("block_height") or 0.0
                block_top_y = target_center_y - block_height / 2.0
                vertical_quote_stack = []

                for i, line_data in enumerate(final_lines_data):
                    segment_text = _vertical_draw_quote_text(
                        str(line_data.get("text_with_markers", "")),
                        vertical_quote_stack,
                    )
                    # Vertical layouts may contain multiple columns when the
                    # translated text has explicit newlines. origin_x is the
                    # per-column offset from the text block center.
                    unit_center_x = target_center_x + line_data.get("origin_x", 0.0)
                    if line_data.get("draw_vertical_dash"):
                        left = line_data.get("left", 0.0)
                        right = line_data.get("right", line_data.get("width", 0.0))
                        top = line_data.get("top", -line_data.get("height", 0.0))
                        bottom = line_data.get("bottom", 0.0)
                        center_y = block_top_y + line_data.get("origin_y", 0.0)
                        thickness = float(
                            line_data.get(
                                "dash_thickness",
                                max(1.2, final_font_size * 0.075),
                            )
                        )
                        dash_length = float(
                            line_data.get(
                                "dash_length",
                                max(line_data.get("height", 0.0), final_font_size),
                            )
                        )

                        if bg_paint is not None:
                            pad_x = final_font_size * 0.1
                            pad_y = final_font_size * 0.05
                            rect = skia.Rect.MakeXYWH(
                                unit_center_x + left - pad_x,
                                center_y + top - pad_y,
                                (right - left) + 2 * pad_x,
                                (bottom - top) + 2 * pad_y,
                            )
                            canvas.drawRect(rect, bg_paint)

                        if outline_paint:
                            _draw_filled_rect(
                                canvas,
                                unit_center_x,
                                center_y,
                                thickness + max(1.0, outline_width),
                                dash_length + max(1.0, outline_width),
                                outline_paint,
                            )
                        _draw_filled_rect(
                            canvas,
                            unit_center_x,
                            center_y,
                            thickness,
                            dash_length,
                            paint,
                        )
                        log_message(
                            f"Rendered vertical dash '{segment_text}'",
                            verbose=verbose,
                        )
                        continue
                    if line_data.get("draw_vertical_ellipsis"):
                        left = line_data.get("left", 0.0)
                        right = line_data.get("right", line_data.get("width", 0.0))
                        top = line_data.get("top", -line_data.get("height", 0.0))
                        bottom = line_data.get("bottom", 0.0)
                        center_y = block_top_y + line_data.get("origin_y", 0.0)
                        radius = float(
                            line_data.get(
                                "ellipsis_dot_radius",
                                max(1.0, final_font_size * 0.07),
                            )
                        )
                        dot_step = float(
                            line_data.get(
                                "ellipsis_dot_step",
                                radius * 2.0 + max(1.0, final_font_size * 0.16),
                            )
                        )
                        dot_count = max(
                            2, int(line_data.get("ellipsis_dot_count", 3))
                        )

                        if bg_paint is not None:
                            pad_x = final_font_size * 0.1
                            pad_y = final_font_size * 0.05
                            rect = skia.Rect.MakeXYWH(
                                unit_center_x + left - pad_x,
                                center_y + top - pad_y,
                                (right - left) + 2 * pad_x,
                                (bottom - top) + 2 * pad_y,
                            )
                            canvas.drawRect(rect, bg_paint)

                        _draw_vertical_ellipsis(
                            canvas,
                            unit_center_x,
                            center_y,
                            dot_count,
                            dot_step,
                            radius,
                            paint,
                            outline_paint,
                        )
                        log_message(
                            "Rendered vertical ellipsis",
                            verbose=verbose,
                        )
                        continue
                    if line_data.get("draw_vertical_centered_dot"):
                        left = line_data.get("left", 0.0)
                        right = line_data.get("right", line_data.get("width", 0.0))
                        top = line_data.get("top", -line_data.get("height", 0.0))
                        bottom = line_data.get("bottom", 0.0)
                        center_y = block_top_y + line_data.get("origin_y", 0.0)
                        radius = float(
                            line_data.get(
                                "centered_dot_radius",
                                max(0.85, final_font_size * 0.07),
                            )
                        )

                        if bg_paint is not None:
                            pad_x = final_font_size * 0.1
                            pad_y = final_font_size * 0.05
                            rect = skia.Rect.MakeXYWH(
                                unit_center_x + left - pad_x,
                                center_y + top - pad_y,
                                (right - left) + 2 * pad_x,
                                (bottom - top) + 2 * pad_y,
                            )
                            canvas.drawRect(rect, bg_paint)

                        if outline_paint:
                            canvas.drawCircle(
                                float(unit_center_x),
                                float(center_y),
                                float(radius + max(1.0, outline_width)),
                                outline_paint,
                            )
                        canvas.drawCircle(
                            float(unit_center_x),
                            float(center_y),
                            float(radius),
                            paint,
                        )
                        log_message(
                            "Rendered centered vertical '.'",
                            verbose=verbose,
                        )
                        continue
                    quote_info = VERTICAL_DRAW_QUOTE_DISPLAY_INFO.get(segment_text)
                    if line_data.get("draw_vertical_quote") or quote_info is not None:
                        role = line_data.get("quote_role")
                        kind = line_data.get("quote_kind")
                        if quote_info is not None:
                            role = role or quote_info[0]
                            kind = kind or quote_info[1]
                        role = role or "open"
                        kind = kind or "single"
                        center_y = block_top_y + line_data.get("origin_y", 0.0)
                        quote_width = float(
                            line_data.get(
                                "quote_draw_width",
                                line_data.get(
                                    "width",
                                    max(4.0, final_font_size * 0.36),
                                ),
                            )
                        )
                        quote_height = float(
                            line_data.get(
                                "quote_draw_height",
                                line_data.get(
                                    "height",
                                    max(4.0, final_font_size * 0.42),
                                ),
                            )
                        )
                        quote_center_x = float(
                            unit_center_x + line_data.get("quote_x_offset", 0.0)
                        )
                        quote_center_y = float(
                            center_y + line_data.get("quote_y_offset", 0.0)
                        )
                        quote_cell_width = float(
                            line_data.get(
                                "width",
                                max(4.0, final_font_size * 0.36),
                            )
                        )
                        quote_cell_height = float(
                            line_data.get(
                                "height",
                                max(4.0, final_font_size * 0.42),
                            )
                        )
                        quote_stroke_width = float(
                            line_data.get(
                                "quote_stroke_width",
                                max(1.6, final_font_size * 0.085),
                            )
                        )

                        if bg_paint is not None:
                            pad = final_font_size * 0.08
                            rect = skia.Rect.MakeXYWH(
                                unit_center_x - quote_cell_width / 2.0 - pad,
                                center_y - quote_cell_height / 2.0 - pad,
                                quote_cell_width + 2 * pad,
                                quote_cell_height + 2 * pad,
                            )
                            canvas.drawRect(rect, bg_paint)

                        _draw_vertical_corner_quote(
                            canvas,
                            quote_center_x,
                            quote_center_y,
                            quote_width,
                            quote_height,
                            str(role),
                            str(kind),
                            paint,
                            outline_paint,
                            quote_stroke_width,
                            outline_width,
                        )
                        log_message(
                            f"Rendered vertical quote '{segment_text}'",
                            verbose=verbose,
                        )
                        continue

                    style_name = line_data.get("style", "regular")
                    font_key = line_data.get("font_key", style_name)
                    (
                        typeface_to_use,
                        hb_face_to_use,
                        synthetic_bold,
                        synthetic_italic,
                    ) = _resolve_style_resources(
                        font_key,
                        style_name,
                    )

                    skia_font_segment = skia.Font(typeface_to_use, final_font_size)
                    skia_font_segment.setSubpixel(use_subpixel_rendering)
                    skia_font_segment.setHinting(skia_hinting)
                    _apply_synthetic_style(
                        skia_font_segment,
                        synthetic_bold,
                        synthetic_italic,
                    )
                    synthetic_bold_paint = _synthetic_bold_paint(
                        synthetic_bold,
                        final_font_size,
                    )

                    hb_font_segment = hb.Font(hb_face_to_use)
                    hb_font_segment.ptem = float(final_font_size)
                    hb_scale = int(final_font_size * 64)
                    hb_font_segment.scale = (hb_scale, hb_scale)

                    try:
                        infos, positions, _ = shape_line(
                            segment_text, hb_font_segment, features_to_enable
                        )
                        if not infos:
                            continue
                    except Exception as e:
                        log_message(
                            f"Shaping failed for '{segment_text}': {e}",
                            always_print=True,
                        )
                        continue

                    left = line_data.get("left", 0.0)
                    right = line_data.get("right", line_data.get("width", 0.0))
                    top = line_data.get("top", -line_data.get("height", 0.0))
                    bottom = line_data.get("bottom", 0.0)
                    baseline_y = block_top_y + line_data.get("origin_y", 0.0)
                    rotate_in_vertical = bool(line_data.get("rotate_in_vertical"))
                    rotation_center_x = unit_center_x
                    rotation_center_y = baseline_y

                    if rotate_in_vertical:
                        original_left = line_data.get("original_left", left)
                        original_right = line_data.get("original_right", right)
                        original_top = line_data.get("original_top", top)
                        original_bottom = line_data.get("original_bottom", bottom)
                        baseline_x = rotation_center_x - (
                            original_left + original_right
                        ) / 2.0
                        baseline_y = rotation_center_y - (
                            original_top + original_bottom
                        ) / 2.0
                    else:
                        alignment_center_offset = line_data.get(
                            "alignment_center_offset"
                        )
                        if alignment_center_offset is not None:
                            baseline_x = unit_center_x - float(
                                alignment_center_offset
                            )
                        else:
                            baseline_x = unit_center_x - (left + right) / 2.0

                    if bg_paint is not None:
                        pad_x = final_font_size * 0.1
                        pad_y = final_font_size * 0.05
                        display_origin_x = (
                            rotation_center_x if rotate_in_vertical else baseline_x
                        )
                        display_origin_y = (
                            rotation_center_y if rotate_in_vertical else baseline_y
                        )
                        rect = skia.Rect.MakeXYWH(
                            display_origin_x + left - pad_x,
                            display_origin_y + top - pad_y,
                            (right - left) + 2 * pad_x,
                            (bottom - top) + 2 * pad_y,
                        )
                        canvas.drawRect(rect, bg_paint)

                    builder = skia.TextBlobBuilder()
                    glyph_ids = [info.codepoint for info in infos]
                    skia_point_positions = []
                    segment_cursor_x = 0.0

                    for _, pos in zip(infos, positions):
                        glyph_x = (
                            baseline_x
                            + segment_cursor_x
                            + (pos.x_offset / HB_26_6_SCALE_FACTOR)
                        )
                        glyph_y = baseline_y - (pos.y_offset / HB_26_6_SCALE_FACTOR)
                        skia_point_positions.append(skia.Point(glyph_x, glyph_y))
                        segment_cursor_x += pos.x_advance / HB_26_6_SCALE_FACTOR

                    try:
                        _ = builder.allocRunPos(
                            skia_font_segment, glyph_ids, skia_point_positions
                        )
                        text_blob = builder.make()
                        if text_blob:
                            if rotate_in_vertical:
                                canvas.save()
                                try:
                                    canvas.translate(
                                        float(rotation_center_x),
                                        float(rotation_center_y),
                                    )
                                    canvas.rotate(
                                        float(line_data.get("rotation_degrees", 90.0))
                                    )
                                    canvas.translate(
                                        float(-rotation_center_x),
                                        float(-rotation_center_y),
                                    )
                                    if outline_paint:
                                        canvas.drawTextBlob(
                                            text_blob, 0, 0, outline_paint
                                        )
                                    if synthetic_bold_paint:
                                        canvas.drawTextBlob(
                                            text_blob, 0, 0, synthetic_bold_paint
                                        )
                                    canvas.drawTextBlob(text_blob, 0, 0, paint)
                                finally:
                                    canvas.restore()
                            else:
                                if outline_paint:
                                    canvas.drawTextBlob(text_blob, 0, 0, outline_paint)
                                if synthetic_bold_paint:
                                    canvas.drawTextBlob(
                                        text_blob, 0, 0, synthetic_bold_paint
                                    )
                                canvas.drawTextBlob(text_blob, 0, 0, paint)
                            _draw_attached_vertical_quotes(
                                canvas,
                                line_data,
                                unit_center_x,
                                baseline_x,
                                baseline_y,
                                float(final_font_size),
                                paint,
                                outline_paint,
                                outline_width,
                                verbose,
                            )
                            log_message(
                                f"Rendered vertical '{segment_text}' ({style_name})",
                                verbose=verbose,
                            )
                    except Exception as e:
                        log_message(
                            f"Skia rendering error for '{segment_text}': {e}",
                            always_print=True,
                        )

            else:
                current_baseline_y = first_baseline_y
                for i, line_data in enumerate(final_lines_data):
                    line_width_measured = line_data["width"]

                    line_start_x = _horizontal_line_start_x(
                        target_center_x,
                        alignment_width,
                        line_width_measured,
                        text_alignment,
                    )
                    cursor_x = line_start_x

                    if bg_paint is not None:
                        pad_x = final_font_size * 0.1
                        pad_y = final_font_size * 0.05
                        rect = skia.Rect.MakeXYWH(
                            line_start_x - pad_x,
                            current_baseline_y + final_metrics.fAscent - pad_y,
                            line_width_measured + 2 * pad_x,
                            -final_metrics.fAscent + final_metrics.fDescent + 2 * pad_y,
                        )
                        canvas.drawRect(rect, bg_paint)

                    segments = line_data.get("segments", [])
                    log_message(f"Line {i}: {len(segments)} segments", verbose=verbose)

                    is_line_rtl = is_rtl_script(line_data.get("text_with_markers", ""))
                    if is_line_rtl:
                        cursor_x = line_start_x + line_width_measured

                    for segment_text, style_name in segments:
                        for run_text, font_key in split_text_by_font_support(
                            segment_text, style_name, font_cmaps
                        ):
                            (
                                typeface_to_use,
                                hb_face_to_use,
                                synthetic_bold,
                                synthetic_italic,
                            ) = _resolve_style_resources(
                                font_key,
                                style_name,
                            )
                            if not typeface_to_use or not hb_face_to_use:
                                log_message(
                                    f"ERROR: No font resources for style '{font_key}' - skipping segment",
                                    always_print=True,
                                )
                                continue

                            skia_font_segment = skia.Font(typeface_to_use, final_font_size)
                            skia_font_segment.setSubpixel(use_subpixel_rendering)
                            skia_font_segment.setHinting(skia_hinting)
                            _apply_synthetic_style(
                                skia_font_segment,
                                synthetic_bold,
                                synthetic_italic,
                            )
                            synthetic_bold_paint = _synthetic_bold_paint(
                                synthetic_bold,
                                final_font_size,
                            )

                            hb_font_segment = hb.Font(hb_face_to_use)
                            hb_font_segment.ptem = float(final_font_size)
                            hb_scale = int(final_font_size * 64)
                            hb_font_segment.scale = (hb_scale, hb_scale)

                            try:
                                infos, positions, _ = shape_line(
                                    run_text, hb_font_segment, features_to_enable
                                )
                                if not infos:
                                    log_message(
                                        f"No glyphs for segment '{run_text}'",
                                        verbose=verbose,
                                    )
                                    continue
                            except Exception as e:
                                log_message(
                                    f"Shaping failed for '{run_text}': {e}",
                                    always_print=True,
                                )
                                continue

                            segment_width_calculated = (
                                sum(pos.x_advance for pos in positions)
                                / HB_26_6_SCALE_FACTOR
                            )

                            if is_line_rtl:
                                cursor_x -= segment_width_calculated
                                segment_start_x = cursor_x
                            else:
                                segment_start_x = cursor_x

                            try:
                                part_cursor_x = segment_start_x
                                for part_text, special_kind in _iter_special_text_parts(
                                    run_text
                                ):
                                    part_infos, part_positions, _ = shape_line(
                                        part_text,
                                        hb_font_segment,
                                        features_to_enable,
                                    )
                                    if not part_infos:
                                        continue
                                    part_width, part_top, part_bottom = (
                                        _measure_shaped_bounds(
                                            part_infos,
                                            part_positions,
                                            hb_font_segment,
                                            final_font_size,
                                        )
                                    )
                                    if special_kind == "dash":
                                        _draw_horizontal_dash(
                                            canvas,
                                            part_cursor_x,
                                            current_baseline_y,
                                            part_width,
                                            part_top,
                                            part_bottom,
                                            final_font_size,
                                            paint,
                                            outline_paint,
                                        )
                                    elif special_kind == "ellipsis":
                                        _draw_horizontal_ellipsis(
                                            canvas,
                                            part_cursor_x,
                                            current_baseline_y,
                                            part_width,
                                            part_top,
                                            part_bottom,
                                            part_text,
                                            final_font_size,
                                            paint,
                                            outline_paint,
                                        )
                                    else:
                                        builder = skia.TextBlobBuilder()
                                        glyph_ids = [
                                            info.codepoint for info in part_infos
                                        ]
                                        skia_point_positions = []
                                        part_segment_cursor_x = 0.0
                                        for _, pos in zip(part_infos, part_positions):
                                            glyph_x = (
                                                part_cursor_x
                                                + part_segment_cursor_x
                                                + (
                                                    pos.x_offset
                                                    / HB_26_6_SCALE_FACTOR
                                                )
                                            )
                                            glyph_y = current_baseline_y - (
                                                pos.y_offset / HB_26_6_SCALE_FACTOR
                                            )
                                            skia_point_positions.append(
                                                skia.Point(glyph_x, glyph_y)
                                            )
                                            part_segment_cursor_x += (
                                                pos.x_advance
                                                / HB_26_6_SCALE_FACTOR
                                            )

                                        _ = builder.allocRunPos(
                                            skia_font_segment,
                                            glyph_ids,
                                            skia_point_positions,
                                        )
                                        text_blob = builder.make()
                                        if text_blob:
                                            if outline_paint:
                                                canvas.drawTextBlob(
                                                    text_blob, 0, 0, outline_paint
                                                )
                                            if synthetic_bold_paint:
                                                canvas.drawTextBlob(
                                                    text_blob,
                                                    0,
                                                    0,
                                                    synthetic_bold_paint,
                                                )
                                            canvas.drawTextBlob(text_blob, 0, 0, paint)
                                        else:
                                            log_message(
                                                f"TextBlob build failed for '{part_text}'",
                                                verbose=verbose,
                                            )
                                    part_cursor_x += part_width
                                if not is_line_rtl:
                                    cursor_x += segment_width_calculated
                                log_message(
                                    f"Rendered '{run_text}' ({font_key}) width={segment_width_calculated:.0f}",
                                    verbose=verbose,
                                )
                            except Exception as e:
                                log_message(
                                    f"Skia rendering error for '{run_text}': {e}",
                                    always_print=True,
                                )
                                if not is_line_rtl:
                                    cursor_x += segment_width_calculated

                    current_baseline_y += final_line_height

            if need_transform:
                canvas.restore()

        return True

    except Exception as e:
        log_message(f"Drawing failed: {e}", always_print=True)
        return False
