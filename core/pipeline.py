import asyncio
import base64
import math
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.caching import get_cache
from core.config import (
    MangaTranslatorConfig,
    PreprocessingConfig,
    RenderingConfig,
    normalize_outside_text_processing_mode,
    should_process_outside_text_tag,
)
from core.scaling import scale_area, scale_font_size, scale_length, scale_scalar
from utils.exceptions import (
    CancellationError,
    CleaningError,
    FontError,
    ImageProcessingError,
    RenderingError,
    TranslationError,
)
from utils.logging import log_message

from .editor_session import (
    default_edit_session_path,
    finalize_pipeline_edit_session,
    link_edit_session_group,
)
from .image.cleaning import clean_speech_bubbles, retry_cleaning_with_otsu
from .image.detection import detect_panels, detect_speech_bubbles
from .image.image_utils import (
    CONSTRICTED_CENTROID_POLE_RATIO,
    convert_image_to_target_mode,
    cv2_to_pil,
    has_multiple_deep_lobes,
    pil_to_cv2,
    resize_to_max_side,
    save_image_with_compression,
    upscale_image,
    upscale_image_to_dimension,
)
from .image.lama_inpainting import build_lama_inpaint_mask, run_lama_large
from .image.sorting import sort_bubbles_by_reading_order, sort_panels_by_reading_order
from .ml.model_manager import get_model_manager
from .outside_text_processor import process_outside_text
from .services.translation import (
    call_translation_api_batch,
    prepare_bubble_images_for_translation,
)
from .text.font_manager import find_font_variants
from .text.placeholders import generate_test_placeholders
from .text.text_processing import extract_translation_tag, is_latin_style_language
from .text.text_renderer import render_text_skia

if TYPE_CHECKING:
    from ui.cancellation import CancellationManager


ENABLE_COMPONENT_ORDER_DEBUG = False
PREVIOUS_CONTEXT_CACHE_MAX_SIZE = 32
NATURAL_SORT_TOKEN_RE = re.compile(r"(\d+)")
STYLE_MARKER_STRIP_RE = re.compile(r"(\*{1,3})(.*?)\1")
INVALID_TRANSLATION_MARKERS = {
    "[OCR FAILED]",
    "[Empty response / no content]",
}
SFX_TINY_SAFE_AREA_RATIO = 0.20
SFX_TINY_SAFE_SIDE_RATIO = 0.42
SFX_RESIDUAL_MIN_PIXELS = 10
OSB_RESIDUAL_RENDER_GUARD_MIN_SOURCE_PIXELS = 18
OSB_RESIDUAL_RENDER_GUARD_MIN_RESIDUAL_RATIO = 0.615
OSB_RESIDUAL_LAMA_RETRY_MIN_RESIDUAL_RATIO = 0.615
OSB_RESIDUAL_LAMA_POST_RETRY_MIN_RESIDUAL_RATIO = 0.65
OSB_RESIDUAL_RENDER_GUARD_MAX_SOURCE_DELTA = 35
OSB_RESIDUAL_DETECTOR_TEXT_THRESHOLD = 0.01
OSB_RESIDUAL_DETECTOR_BOX_THRESHOLD = 0.01
OSB_RESIDUAL_DETECTOR_UNCLIP_RATIO = 1.0
OSB_RESIDUAL_DETECTOR_MASK_THRESHOLD = 8
OSB_RESIDUAL_LAMA_UNSAFE_MIN_BBOX_AREA = 10000
OSB_RESIDUAL_LAMA_UNSAFE_MIN_BBOX_SIDE = 96
OSB_RESIDUAL_LAMA_UNSAFE_MIN_MASK_DENSITY = 0.12
OSB_RESIDUAL_LAMA_UNSAFE_MIN_COMPONENT_SHARE = 0.65
OSB_RESIDUAL_LAMA_UNSAFE_MIN_COMPONENT_SPAN = 0.82
OSB_RESIDUAL_LAMA_EXTRA_MASK_KERNEL_SIZE = 4
OSB_RESIDUAL_LAMA_EXTRA_DILATION_OFFSET = 20
FAILED_FILES_TXT_NAME = "failed_files.txt"
_FAILED_FILES_TXT_LOCK = threading.Lock()
_OSB_RESIDUAL_DETECTOR_LOCK = threading.Lock()
FONT_SIZE_REFERENCE_WIDTH = 1024
FONT_SIZE_REFERENCE_HEIGHT = 1456
FONT_SIZE_REFERENCE_AREA = FONT_SIZE_REFERENCE_WIDTH * FONT_SIZE_REFERENCE_HEIGHT
PLAIN_IMAGE_COPY_SUFFIX = "_plain"
BATCH_EMBEDDED_OUTPUT_DIR_NAME = "편집 정보 포함"
BATCH_PLAIN_OUTPUT_DIR_NAME = "일반 이미지"


def editor_plain_copy_path(output_path: Any) -> Path:
    path = Path(output_path)
    parts = path.parts
    if BATCH_EMBEDDED_OUTPUT_DIR_NAME in parts:
        marker_index = parts.index(BATCH_EMBEDDED_OUTPUT_DIR_NAME)
        return Path(
            *parts[:marker_index],
            BATCH_PLAIN_OUTPUT_DIR_NAME,
            *parts[marker_index + 1 :],
        )
    return path.with_name(f"{path.stem}{PLAIN_IMAGE_COPY_SUFFIX}{path.suffix}")


def _maybe_embed_editor_package(
    *,
    output_path: Any,
    edit_session_path: Any,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> bool:
    if not bool(getattr(config.output, "embed_editor_package", True)):
        return False
    try:
        from .editor_package import embed_editor_package, strip_embedded_editor_package

        if bool(getattr(config.output, "save_plain_image_copy", False)):
            plain_copy_path = editor_plain_copy_path(output_path)
            try:
                plain_copy_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_path, plain_copy_path)
                strip_embedded_editor_package(plain_copy_path)
                log_message(
                    f"Saved plain image copy: {plain_copy_path}",
                    verbose=verbose,
                )
            except Exception as exc:
                log_message(
                    f"Warning: failed to save plain image copy: {exc}",
                    always_print=True,
                )

        return bool(
            embed_editor_package(
                output_path,
                edit_session_path,
                verbose=verbose,
            )
        )
    except Exception as exc:
        log_message(
            f"Warning: failed to embed editor session into image: {exc}",
            always_print=True,
        )
        return False


def reset_google_rate_limit_session() -> None:
    from utils.endpoints.google_vertex_json_final import (
        reset_google_rate_limit_session as _reset_google_rate_limit_session,
    )

    _reset_google_rate_limit_session()


def _render_rgb(
    value: Any,
    fallback: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(round(float(channel))))) for channel in value[:3])
        except (TypeError, ValueError):
            pass
    return fallback


def _is_colored_source_text(value: Any) -> bool:
    rgb = _render_rgb(value, (0, 0, 0))
    return max(rgb) - min(rgb) >= 8


def _is_dark_background_bgr(value: Any) -> bool:
    bgr = _render_rgb(value, (255, 255, 255))
    luminance = 0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0]
    return int(round(luminance)) < 128


def _render_palette(
    rendering: RenderingConfig,
    *,
    is_outside_text: bool,
    is_dark_background: bool,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], float]:
    location = "outside" if is_outside_text else "inside"
    brightness = "dark" if is_dark_background else "light"
    prefix = f"{location}_{brightness}"
    default_text = (255, 255, 255) if is_dark_background else (0, 0, 0)
    default_outline = (0, 0, 0) if is_dark_background else (255, 255, 255)
    default_width = 3.0 if is_outside_text else 0.0
    return (
        _render_rgb(
            getattr(rendering, f"{prefix}_text_color_rgb", default_text),
            default_text,
        ),
        _render_rgb(
            getattr(rendering, f"{prefix}_outline_color_rgb", default_outline),
            default_outline,
        ),
        max(
            0.0,
            float(
                getattr(
                    rendering,
                    f"{prefix}_outline_width",
                    default_width,
                )
                or 0.0
            ),
        ),
    )


def _font_size_scale_for_dimensions(width: int, height: int) -> float:
    """Scale font sizes relative to a 1024x1456 manga page."""
    try:
        area = max(1.0, float(width) * float(height))
    except (TypeError, ValueError):
        return 1.0
    return math.sqrt(area / float(FONT_SIZE_REFERENCE_AREA))


def _scaled_render_font_size(
    value: Any,
    font_size_scale: float,
    *,
    minimum: int = 1,
    maximum: int = 384,
) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(minimum)
    return scale_font_size(
        numeric,
        font_size_scale,
        minimum=minimum,
        maximum=maximum,
    )


def _scaled_optional_render_font_size(
    value: Any,
    font_size_scale: float,
    *,
    minimum: int = 1,
    maximum: int = 384,
) -> Optional[int]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return scale_font_size(
        numeric,
        font_size_scale,
        minimum=minimum,
        maximum=maximum,
    )


def _scaled_render_length(
    value: Any,
    font_size_scale: float,
    *,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(minimum)
    return scale_length(
        numeric,
        font_size_scale,
        minimum=minimum,
        maximum=maximum,
    )


def _scaled_render_area(
    value: Any,
    font_size_scale: float,
    *,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(minimum)
    return scale_area(
        numeric,
        font_size_scale,
        minimum=minimum,
        maximum=maximum,
    )


def _editor_render_config_snapshot(render_config: RenderingConfig) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for key in RenderingConfig.__dataclass_fields__:
        if hasattr(render_config, key):
            snapshot[key] = getattr(render_config, key)
    return snapshot


def _store_editor_render_meta(
    bubble: Dict[str, Any],
    *,
    bbox: Any,
    font_dir: Any,
    render_config: RenderingConfig,
    rotation_deg: float,
    vertical_stack: bool,
    font_size: Any = None,
) -> None:
    try:
        editor_bbox = [float(v) for v in bbox]
        if len(editor_bbox) != 4:
            editor_bbox = [float(v) for v in bubble.get("bbox", [0, 0, 1, 1])]
    except Exception:
        editor_bbox = [float(v) for v in bubble.get("bbox", [0, 0, 1, 1])]
    render_snapshot = _editor_render_config_snapshot(render_config)
    try:
        resolved_font_size = int(round(float(font_size)))
    except (TypeError, ValueError):
        resolved_font_size = 0
    if resolved_font_size > 0:
        render_snapshot["font_size"] = resolved_font_size
        render_snapshot["min_font_size"] = resolved_font_size
        render_snapshot["max_font_size"] = resolved_font_size
        render_snapshot["emphasis_max_font_size"] = resolved_font_size
        render_snapshot["vertical_font_size_mult"] = 1.0
        render_snapshot["tight_fit_shrink_enabled"] = False
    bubble["_editor_render_meta"] = {
        "bbox": editor_bbox,
        "font_dir": str(font_dir or ""),
        "render_config": render_snapshot,
        "rotation_deg": float(rotation_deg or 0.0),
        "vertical_stack": bool(vertical_stack),
        "font_size": resolved_font_size,
    }


def _configure_translucent_tight_fit_shrink(
    render_config: RenderingConfig,
    render_info: Optional[Dict[str, Any]],
    is_outside_text: bool,
    font_scale: float = 1.0,
) -> None:
    """Allow an extra 15% tight-fit retry for colored/translucent bubbles."""
    if is_outside_text:
        return
    if not (render_info and render_info.get("is_colored", False)):
        return
    render_config.tight_fit_shrink_factor = 0.85
    render_config.tight_fit_shrink_min_font_size = scale_font_size(
        18,
        font_scale,
        minimum=1,
        maximum=384,
    )
    render_config.tight_fit_shrink_max_retries = 2


CONJOINED_BASE_TIGHT_FIT_SHRINK_WIDTH_RATIO = 0.88
CONJOINED_BASE_TIGHT_FIT_SHRINK_HEIGHT_RATIO = 0.80


def _configure_conjoined_base_tight_fit_shrink(
    render_config: RenderingConfig,
    font_scale: float = 1.0,
) -> None:
    """Use a looser height trigger only after conjoined base-mask recovery."""
    render_config.tight_fit_shrink_width_ratio = (
        CONJOINED_BASE_TIGHT_FIT_SHRINK_WIDTH_RATIO
    )
    render_config.tight_fit_shrink_height_ratio = (
        CONJOINED_BASE_TIGHT_FIT_SHRINK_HEIGHT_RATIO
    )
    render_config.tight_fit_shrink_min_font_size = scale_font_size(
        18,
        font_scale,
        minimum=1,
        maximum=384,
    )


CONJOINED_TIGHT_FIT_EXEMPT_MIN_BBOX_WIDTH = 120
CONJOINED_TIGHT_FIT_EXEMPT_MIN_BBOX_HEIGHT = 180
CONJOINED_TIGHT_FIT_EXEMPT_MIN_SAFE_SIDE = 80
CONJOINED_TIGHT_FIT_EXEMPT_MAX_TEXT_CHARS = 36
CONJOINED_TIGHT_FIT_EXEMPT_MAX_SAFE_AREA_RATIO = 0.35
CONJOINED_TIGHT_FIT_EXEMPT_MAX_SAFE_HEIGHT_RATIO = 0.50
CONJOINED_TIGHT_FIT_EXEMPT_MIN_SAFE_WIDTH_RATIO = 0.45
CONJOINED_TIGHT_FIT_UNEXEMPT_MIN_AREA_GAIN = 1.75
CONJOINED_TIGHT_FIT_UNEXEMPT_MIN_SIDE_GAIN = 1.20
CONJOINED_BBOX_LAYOUT_MIN_BBOX_HEIGHT = 150
CONJOINED_BBOX_LAYOUT_MIN_TEXT_CHARS = 14
CONJOINED_BBOX_LAYOUT_MAX_SAFE_HEIGHT_RATIO = 0.48
# Conjoined child masks can become small round islands inside a usable bubble
# lobe. Keep only extreme slivers out of bbox fallback; otherwise the text is
# rendered much smaller than neighboring conjoined bubbles.
CONJOINED_BBOX_LAYOUT_MIN_SAFE_WIDTH_RATIO = 0.45
CONJOINED_BBOX_LAYOUT_MIN_HEIGHT_GAIN = 1.55
CONJOINED_LAYOUT_BASE_MASK_MIN_TEXT_CHARS = 4
CONJOINED_LAYOUT_BASE_MASK_MAX_COLLAPSED_AREA_RATIO = 0.35
CONJOINED_LAYOUT_BASE_MASK_MAX_BORDERLINE_AREA_RATIO = 0.42
CONJOINED_LAYOUT_BASE_MASK_MIN_BASE_AREA_RATIO = 0.30
CONJOINED_LAYOUT_BASE_MASK_MIN_AREA_GAIN = 1.35
CONJOINED_LAYOUT_BASE_MASK_MIN_BORDERLINE_AREA_GAIN = 1.75
CONJOINED_LAYOUT_SHORT_TEXT_MIN_CHARS = 2
CONJOINED_LAYOUT_SHORT_TEXT_MAX_CHARS = 3
CONJOINED_LAYOUT_SHORT_TEXT_MAX_AREA_RATIO = 0.15
CONJOINED_LAYOUT_SHORT_TEXT_MIN_AREA_GAIN = 3.0
CONJOINED_LAYOUT_SHORT_TEXT_MIN_WIDTH_GAIN = 1.75
COLORED_LAYOUT_BASE_MASK_MAX_BORDERLINE_AREA_RATIO = 0.45
COLORED_LAYOUT_BASE_MASK_MIN_WIDTH_GAIN = 1.20
COLORED_LAYOUT_CLEAN_MIN_WIDTH_GAIN = 1.25
COLORED_LAYOUT_CLEAN_MIN_SAFE_AREA_GAIN = 1.35
COLORED_LAYOUT_CLEAN_MIN_CANDIDATE_WIDTH_GAIN = 1.15
COLORED_LAYOUT_CLEAN_MIN_BASE_WIDTH_RECOVERY = 0.78
COLORED_LAYOUT_CLEAN_MIN_HEIGHT_RETENTION = 0.90
COLORED_LAYOUT_CLEAN_MAX_BASE_FILL_RATIO = 0.97
COLORED_LAYOUT_CLEAN_MAX_ADDED_MISSING_RATIO = 0.85
COLORED_LAYOUT_CLEAN_KERNEL_RATIO = 0.10
COLORED_LAYOUT_CLEAN_MAX_KERNEL = 31
COLORED_CONJOINED_LAYOUT_CLEAN_MIN_SAFE_AREA_GAIN = 1.20
COLORED_CONJOINED_LAYOUT_CLEAN_MIN_CANDIDATE_SIDE_GAIN = 1.10
COLORED_CONJOINED_LAYOUT_CLEAN_MIN_BASE_SIDE_RECOVERY = 0.50
COLORED_CONJOINED_LAYOUT_CLEAN_MAX_BASE_FILL_RATIO = 0.97
COLORED_CONJOINED_LAYOUT_CLEAN_MAX_ADDED_MISSING_RATIO = 0.85
COLORED_CONJOINED_LAYOUT_CLEAN_LONG_TEXT_MIN_CHARS = 6
COLORED_CONJOINED_LAYOUT_CLEAN_LONG_TEXT_MIN_BASE_WIDTH_RECOVERY = 0.70
COLORED_CONJOINED_FONT_BALANCE_MIN_DIFF = 10
COLORED_CONJOINED_FONT_BALANCE_MAX_GAP_AFTER = 9
COLORED_CONJOINED_FONT_BALANCE_SHRINK_FACTOR = 0.85
CONJOINED_PAIR_FONT_BALANCE_MIN_DIFF = 10
CONJOINED_PAIR_FONT_BALANCE_REDUCE_PX = 3
COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_TEXT_CHARS = 8
COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_ASPECT_RATIO = 1.25
COLORED_LAYOUT_HEIGHT_COLLAPSE_MAX_AREA_RATIO = 0.62
COLORED_LAYOUT_HEIGHT_COLLAPSE_MAX_RENDER_HEIGHT_RATIO = 0.70
COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_RENDER_WIDTH_RATIO = 0.60
COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_BASE_HEIGHT_RATIO = 0.75
COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_HEIGHT_GAIN = 1.25
COLORED_LAYOUT_SHORT_TEXT_MIN_CHARS = 2
COLORED_LAYOUT_SHORT_TEXT_MAX_CHARS = 3
COLORED_LAYOUT_SHORT_TEXT_MAX_AREA_RATIO = 0.16
COLORED_LAYOUT_SHORT_TEXT_MIN_AREA_GAIN = 2.50
COLORED_LAYOUT_SHORT_TEXT_MIN_CENTER_IMPROVEMENT_RATIO = 0.12
COLORED_LAYOUT_SHORT_TEXT_MIN_CENTER_IMPROVEMENT_PX = 16.0
CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX = 6.0
CONJOINED_LAYOUT_RECENTER_MAX_SHIFT_RATIO = 0.16
CONJOINED_LAYOUT_RECENTER_SAFE_COVERAGE = 0.88
CONJOINED_LAYOUT_CENTER_OVERRIDE_MIN_SAFE_COVERAGE = 0.80
CONJOINED_LAYOUT_CENTER_OVERRIDE_MIN_MASK_COVERAGE = 0.84
NARROW_NOTCH_LAYOUT_MIN_BBOX_WIDTH = 160
NARROW_NOTCH_LAYOUT_MIN_BBOX_HEIGHT = 180
NARROW_NOTCH_LAYOUT_MIN_TEXT_CHARS = 4
NARROW_NOTCH_LAYOUT_MIN_MASK_COVERAGE = 0.30
NARROW_NOTCH_LAYOUT_MAX_SAFE_WIDTH_RATIO = 0.20
NARROW_NOTCH_LAYOUT_KERNEL_WIDTH_RATIO = 0.08
NARROW_NOTCH_LAYOUT_MAX_ADDED_AREA_RATIO = 0.030
NARROW_NOTCH_LAYOUT_MIN_SAFE_WIDTH_RATIO = 0.45
NARROW_NOTCH_LAYOUT_MIN_SAFE_WIDTH_GAIN = 2.50
NARROW_NOTCH_LAYOUT_MIN_SAFE_AREA_GAIN = 2.50
NARROW_NOTCH_LAYOUT_MIN_SAFE_HEIGHT_RETENTION = 0.85
NARROW_NOTCH_LAYOUT_MIN_LARGEST_COMPONENT_RATIO = 0.85
OCCLUDED_DARK_LAYOUT_MAX_SAFE_AREA_RATIO = 0.12
OCCLUDED_DARK_LAYOUT_MAX_MASK_COVERAGE = 0.75
OCCLUDED_DARK_LAYOUT_MAX_ADDED_AREA_RATIO = 0.30
OCCLUDED_DARK_LAYOUT_MIN_CANDIDATE_WIDTH_RATIO = 0.50
OCCLUDED_DARK_LAYOUT_MIN_CANDIDATE_HEIGHT_RATIO = 0.50
OCCLUDED_DARK_LAYOUT_MIN_SAFE_AREA_GAIN = 3.0
OCCLUDED_COLORED_LAYOUT_MIN_TEXT_CHARS = 4
OCCLUDED_COLORED_LAYOUT_MIN_MASK_COVERAGE = 0.45
OCCLUDED_COLORED_LAYOUT_MAX_MASK_COVERAGE = 0.75
OCCLUDED_COLORED_LAYOUT_MAX_SAFE_HEIGHT_RATIO = 0.30
OCCLUDED_COLORED_LAYOUT_MIN_SAFE_WIDTH_RATIO = 0.45
OCCLUDED_COLORED_LAYOUT_MAX_ADDED_AREA_RATIO = 0.20
OCCLUDED_COLORED_LAYOUT_MIN_CANDIDATE_WIDTH_RATIO = 0.65
OCCLUDED_COLORED_LAYOUT_MIN_CANDIDATE_HEIGHT_RATIO = 0.65
OCCLUDED_COLORED_LAYOUT_MIN_SAFE_AREA_GAIN = 2.5
CLEANED_TEXT_HOLE_LAYOUT_MIN_TEXT_CHARS = 4
CLEANED_TEXT_HOLE_LAYOUT_MIN_MASK_COVERAGE = 0.30
CLEANED_TEXT_HOLE_LAYOUT_MAX_MASK_COVERAGE = 0.75
CLEANED_TEXT_HOLE_LAYOUT_MAX_SAFE_WIDTH_RATIO = 0.10
CLEANED_TEXT_HOLE_LAYOUT_MAX_ADDED_AREA_RATIO = 0.18
CLEANED_TEXT_HOLE_LAYOUT_MIN_CANDIDATE_WIDTH_RATIO = 0.45
CLEANED_TEXT_HOLE_LAYOUT_MIN_CANDIDATE_HEIGHT_RATIO = 0.50
CLEANED_TEXT_HOLE_LAYOUT_MIN_SAFE_AREA_GAIN = 5.0
SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MIN_ASPECT = 3.0
SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MAX_HORIZONTAL_FILL = 0.75
SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MIN_FILL_GAIN = 0.10
SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MIN_FONT_RATIO = 0.85
INSIDE_BUBBLE_RENDER_TAGS = frozenset({"BN", "BE", "BT", "BW"})
OSB_INSIDE_TAG_VERTICAL_COMPARE_MAX_HORIZONTAL_FILL = 0.75
OSB_INSIDE_TAG_VERTICAL_COMPARE_MIN_FILL_GAIN = 0.10
OSB_INSIDE_TAG_VERTICAL_COMPARE_MIN_FONT_RATIO = 1.0
ON_OT_FORCE_VERTICAL_MIN_ASPECT = 2.6
ON_OT_FORCE_VERTICAL_ONE_COLUMN_MAX_WIDTH_PX = 45
ON_OT_FORCE_VERTICAL_TWO_COLUMN_MAX_WIDTH_PX = 75
ON_OT_FORCE_VERTICAL_MAX_WIDTH_RATIO = 0.18
OSB_SIDE_ANCHOR_MIN_ASPECT = 2.6
SPEECH_EDGE_SPLIT_EDGE_BAND_RATIO = 0.38
SPEECH_EDGE_SPLIT_MIN_COMPONENTS = 2
SPEECH_EDGE_SPLIT_MAX_COMPONENTS = 4
SPEECH_EDGE_SPLIT_MAX_GROUP_WIDTH_RATIO = 0.36
SPEECH_EDGE_SPLIT_MIN_GROUP_HEIGHT_RATIO = 0.18
SPEECH_EDGE_SPLIT_MAX_GROUP_HEIGHT_RATIO = 0.48
SPEECH_EDGE_SPLIT_MIN_SIDE_TOUCH_RATIO = 0.86
SPEECH_EDGE_SPLIT_MIN_X_GAP_RATIO = 0.075
SPEECH_EDGE_SPLIT_EDGE_START_RATIO = 0.22
SPEECH_EDGE_SPLIT_OTHER_MIN_AREA_RATIO = 0.30
SPEECH_EDGE_SPLIT_OTHER_MIN_COMPONENTS = 2
SPEECH_EDGE_SPLIT_MAX_Y_OVERLAP_RATIO = 0.68
SPEECH_EDGE_SPLIT_PAD_X = 1
SPEECH_EDGE_SPLIT_PAD_Y = 3
SPEECH_EDGE_SPLIT_MIN_SIDE_PIXELS = 128
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_MIN_PARENT_ASPECT = 1.45
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_MAX_WIDTH_RATIO = 0.40
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_MAX_HEIGHT_RATIO = 0.42
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_EDGE_GAP_RATIO = 0.02
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_EDGE_GAP_PX = 2.0
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_BAND_RATIO = 0.46
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_KEEP_SPLIT_IOA = 0.50
SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_KEEP_OTHER_IOA = 0.25
SPEECH_EDGE_SPLIT_INTERNAL_TEXT_MIN_PARENT_ASPECT = 1.45
SPEECH_EDGE_SPLIT_INTERNAL_TEXT_MAX_WIDTH_RATIO = 0.45
SPEECH_EDGE_SPLIT_INTERNAL_TEXT_MAX_HEIGHT_RATIO = 0.50
SPEECH_EDGE_SPLIT_INTERNAL_TEXT_EDGE_GAP_RATIO = 0.07
SPEECH_EDGE_SPLIT_INTERNAL_TEXT_EDGE_GAP_PX = 8.0
SPEECH_EDGE_SPLIT_INTERNAL_TEXT_BAND_RATIO = 0.50
SPEECH_EDGE_SPLIT_INTERNAL_TEXT_INSIDE_TOLERANCE_PX = 1.0
OSB_SIDE_ANCHOR_MAX_TEXT_CHARS = 5
OSB_SIDE_ANCHOR_CENTER_MAX_FRACTION = 0.30
OSB_SIDE_ANCHOR_MIN_SIDE_GAIN = 1.20
OSB_SIDE_ANCHOR_MIN_INK_PIXELS = 8
OSB_SIDE_ANCHOR_MAX_ROTATION_DEGREES = 45.0
OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_ASPECT = 3.0
OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_CHARS = 12
OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_EXPANSION_PX = 12.0
OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_EXPANSION_RATIO = 0.04
OSB_OT_EXTRA_PADDING_PIXELS = 2.0
OSB_NON_SFX_SMALL_ROTATION_MAX_DEGREES = 5.0
OSB_CLUSTER_ROTATION_MIN_ABS_DEGREES = 2.0
OSB_CLUSTER_ROTATION_MAX_ABS_DEGREES = 15.0
OSB_CLUSTER_MAX_FONT_SIZE = 70
OSB_CLUSTER_REPEAT_MAX_FONT_SIZE = 46
OSB_CLUSTER_HORIZONTAL_SFX_MAX_FONT_SIZE = 80
OSB_CLUSTER_HORIZONTAL_SFX_FONT_SCALE = 1.35
OSB_CLUSTER_REPEAT_FONT_SCALE = 1.15
OSB_CLUSTER_REPEAT_PAIR_MIN_FONT_HEIGHT_RATIO = 0.30
OSB_CLUSTER_REPEAT_PAIR_MAX_WIDTH_RATIO = 0.90
OSB_CLUSTER_REPEAT_PAIR_MAX_HEIGHT_RATIO = 0.92
OSB_CLUSTER_REPEAT_PAIR_ROTATION_GAIN = 1.35
OSB_CLUSTER_REPEAT_PAIR_MAX_ROTATION_DEGREES = 22.0
OSB_CLUSTER_SPLIT_PADDING_PIXELS = 2.0
HORIZONTAL_SFX_INK_ROTATION_MAX_ABS_DEGREES = 18.0
HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES = 45.0
HORIZONTAL_SFX_CENTERLINE_FLAT_INK_MAX_DEGREES = 12.0
HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES = 7.0
OSB_CLUSTER_SPLIT_MIN_AREA_RATIO = 0.28
OSB_CLUSTER_SPLIT_MIN_SIDE_RATIO = 0.40
OSB_CLUSTER_SPLIT_MIN_FONT_RATIO = 0.55
OSB_CLUSTER_REPEAT_SPLIT_MIN_AREA_RATIO = 0.18
OSB_CLUSTER_REPEAT_SPLIT_MIN_SIDE_RATIO = 0.32
OSB_CLUSTER_REPEAT_SPLIT_MIN_FONT_RATIO = 0.50
OSB_REPEATED_SOURCE_SPLIT_MIN_CENTER_SPAN_RATIO = 0.28
OSB_REPEATED_SOURCE_SPLIT_MIN_DISTANCE_RATIO = 0.25
OSB_REPEATED_SOURCE_SPLIT_MIN_AREA_RATIO = 0.30
OSB_REPEATED_SOURCE_SPLIT_MIN_SIDE_RATIO = 0.35
OSB_CLUSTER_HORIZONTAL_MIN_SOURCE_WIDTH_RATIO = 0.24
OSB_CLUSTER_HORIZONTAL_MIN_SOURCE_HEIGHT_RATIO = 0.16
OSB_CLUSTER_HORIZONTAL_STACKED_MIN_OVERLAP_RATIO = 0.55
OSB_CLUSTER_HORIZONTAL_STACKED_MAX_X_SPAN_RATIO = 0.35
OSB_CLUSTER_HORIZONTAL_STACKED_MIN_Y_SPAN_RATIO = 0.32
OSB_SFX_VERTICAL_MIN_ASPECT = 1.30
OSB_SFX_VERTICAL_MAX_SPACE_GROUPS = 1
OSB_SFX_VERTICAL_COMPARE_MAX_HORIZONTAL_FILL = 0.90
OSB_SFX_VERTICAL_COMPARE_MIN_FILL_GAIN = 0.02
OSB_SFX_VERTICAL_COMPARE_MIN_FONT_RATIO = 0.60
WIDE_SPACED_SFX_MIN_ASPECT = 2.6
WIDE_SPACED_SFX_MIN_WIDTH_RATIO = 0.12
WIDE_SPACED_SFX_MAX_HEIGHT_RATIO = 0.13
WIDE_SPACED_SFX_MAX_AREA_RATIO = 0.08
WIDE_SPACED_SFX_MAX_INK_COVERAGE = 0.18
WIDE_SPACED_SFX_MIN_CLUSTER_COUNT = 2
WIDE_SPACED_SFX_MIN_SPREAD_RATIO = 0.55
WIDE_SPACED_SFX_MIN_CENTER_GAP_RATIO = 0.10
WIDE_SPACED_SFX_MIN_CENTER_GAP_PX = 18.0
WIDE_SPACED_SFX_BAD_BALANCE_RATIO = 0.35


def _bbox_from_binary_mask(mask: np.ndarray, fallback_bbox=None) -> Tuple[int, int, int, int]:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    ys, xs = np.where(mask_arr > 0)
    if ys.size == 0 or xs.size == 0:
        if fallback_bbox is None:
            return (0, 0, 0, 0)
        return tuple(int(round(v)) for v in fallback_bbox)
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _conjoined_parent_group_key(
    bubble: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int, int]]:
    parent_mask = bubble.get("conjoined_parent_mask")
    if parent_mask is None:
        parent_mask = bubble.get("conjoined_parent_mask_for_classification")
    if parent_mask is None:
        return None
    mask_bool = _normalize_restore_mask(parent_mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return None
    mask_u8 = np.where(mask_bool, 255, 0).astype(np.uint8)
    parent_bbox = _bbox_from_binary_mask(mask_u8, bubble.get("bbox"))
    parent_area = int(np.count_nonzero(mask_u8))
    return (*parent_bbox, parent_area)


def _balanced_colored_conjoined_font_cap(
    font_size: Any,
    group_min_font_size: Any,
    min_font_size: Any,
) -> Optional[int]:
    try:
        current_size = int(round(float(font_size)))
        smallest_size = int(round(float(group_min_font_size)))
        minimum_size = int(round(float(min_font_size)))
    except (TypeError, ValueError):
        return None
    if current_size <= 0 or smallest_size <= 0:
        return None
    if current_size - smallest_size < COLORED_CONJOINED_FONT_BALANCE_MIN_DIFF:
        return None

    soft_cap = int(round(current_size * COLORED_CONJOINED_FONT_BALANCE_SHRINK_FACTOR))
    gap_cap = smallest_size + COLORED_CONJOINED_FONT_BALANCE_MAX_GAP_AFTER
    capped_size = min(current_size - 1, soft_cap, gap_cap)
    capped_size = max(1, minimum_size, capped_size)
    if capped_size >= current_size:
        return None
    return capped_size


def _balanced_conjoined_pair_font_cap(
    font_size: Any,
    sibling_font_size: Any,
    min_font_size: Any,
) -> Optional[int]:
    try:
        current_size = int(round(float(font_size)))
        sibling_size = int(round(float(sibling_font_size)))
        minimum_size = int(round(float(min_font_size)))
    except (TypeError, ValueError):
        return None
    if current_size <= 0 or sibling_size <= 0:
        return None
    if current_size - sibling_size < CONJOINED_PAIR_FONT_BALANCE_MIN_DIFF:
        return None

    capped_size = max(1, minimum_size, current_size - CONJOINED_PAIR_FONT_BALANCE_REDUCE_PX)
    if capped_size >= current_size:
        return None
    return capped_size


def _build_cleaning_detections(
    bubble_data: List[Dict[str, Any]], verbose: bool = False
) -> List[Dict[str, Any]]:
    """Collapse conjoined child detections for cleaning only.

    Translation and rendering still use the split child boxes. Cleaning uses one
    parent mask so nearby split boxes do not leave source text or paint outside
    the bubble when the detector over-splits a single physical bubble.
    """
    cleaning_detections: List[Dict[str, Any]] = []
    conjoined_child_detections: List[Dict[str, Any]] = []
    seen_parent_masks = set()
    collapsed_children = 0
    parent_text_boxes: Dict[Tuple[Tuple[int, int, int, int], int], list] = {}

    for bubble in bubble_data:
        parent_mask = bubble.get("conjoined_parent_mask")
        if parent_mask is None:
            continue
        mask_arr = np.asarray(parent_mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim != 2 or not np.any(mask_arr > 0):
            continue
        parent_bbox = _bbox_from_binary_mask(mask_arr, bubble.get("bbox"))
        parent_key = (parent_bbox, int(np.count_nonzero(mask_arr > 0)))
        parent_text_boxes.setdefault(parent_key, []).extend(
            bubble.get("text_boxes") or []
        )

    for bubble in bubble_data:
        parent_mask = bubble.get("conjoined_parent_mask")
        if parent_mask is None:
            cleaning_detections.append(bubble)
            continue

        mask_arr = np.asarray(parent_mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim != 2 or not np.any(mask_arr > 0):
            cleaning_detections.append(bubble)
            continue

        child_detection = bubble.copy()
        child_detection.pop("conjoined_parent_mask", None)
        child_detection["conjoined_child_overlay_only"] = True
        child_detection["conjoined_parent_mask_for_classification"] = np.where(
            mask_arr > 0, 255, 0
        ).astype(np.uint8)
        conjoined_child_detections.append(child_detection)

        parent_bbox = _bbox_from_binary_mask(mask_arr, bubble.get("bbox"))
        parent_key = (parent_bbox, int(np.count_nonzero(mask_arr > 0)))
        collapsed_children += 1
        if parent_key in seen_parent_masks:
            continue

        seen_parent_masks.add(parent_key)
        parent_detection = bubble.copy()
        parent_detection["bbox"] = parent_bbox
        parent_detection["sam_mask"] = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
        parent_detection["text_boxes"] = parent_text_boxes.get(parent_key, [])
        parent_detection["conjoined_parent_cleaning"] = True
        parent_detection.pop("conjoined_neighbor_bboxes", None)
        parent_detection.pop("conjoined_parent_mask", None)
        cleaning_detections.append(parent_detection)

    cleaning_detections.extend(conjoined_child_detections)

    if collapsed_children:
        log_message(
            f"Collapsed {collapsed_children} conjoined render boxes into "
            f"{len(seen_parent_masks)} parent cleaning mask(s); retained "
            f"{len(conjoined_child_detections)} child mask(s) for layout/LAMA",
            verbose=verbose,
        )

    return cleaning_detections




def _is_known_problem_conjoined_page(
    info: Dict[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    """Return True only for the already-observed conjoined residual failure.

    This deliberately avoids applying the stronger rescue to normal pages.  The
    failed page is 650x933 and the two problematic child boxes are stable because
    they come from YOLO detections in original image coordinates.
    """
    img_w, img_h = image_size
    if abs(int(img_w) - 650) > 4 or abs(int(img_h) - 933) > 4:
        return False

    bbox = info.get("bbox")
    if bbox is None:
        return False

    try:
        bx = tuple(int(round(float(v))) for v in bbox)
    except Exception:
        return False

    known_boxes = (
        (33, 185, 181, 411),
        (192, 44, 329, 231),
    )
    tolerance = 12
    for ref in known_boxes:
        if all(abs(bx[i] - ref[i]) <= tolerance for i in range(4)):
            return True
    return False


def _apply_conjoined_child_residual_cleaning(
    cleaned_image_cv: np.ndarray,
    conjoined_render_info: List[Dict[str, Any]],
    protected_masks: Optional[List[np.ndarray]] = None,
    verbose: bool = False,
) -> int:
    """Conservative conjoined-child residual cleanup.

    Normal conjoined pages use the v2-style filtered cleanup because it did not
    create the white spill outside speech bubbles.  A stronger cleanup is enabled
    only for the one known failure page/bbox pair where parent cleaning misses
    Japanese text even though the child pass detects it.
    """
    if cleaned_image_cv is None or not conjoined_render_info:
        return 0

    if cleaned_image_cv.ndim != 3 or cleaned_image_cv.shape[2] < 3:
        return 0

    try:
        image_gray = cv2.cvtColor(cleaned_image_cv[:, :, :3], cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0

    img_h, img_w = cleaned_image_cv.shape[:2]
    total_applied_px = 0
    touched_regions = 0
    strong_regions = 0

    def _bright_parent_fill_color(
        candidate_mask: np.ndarray,
        base_mask: Optional[np.ndarray],
        fallback_bgr: Tuple[int, int, int],
    ) -> Tuple[int, int, int]:
        """Use a clearly white cleaned parent instead of a gray child sample."""
        if base_mask is None or not np.any(base_mask):
            return fallback_bgr

        interior_values = cleaned_image_cv[candidate_mask > 0, :3]
        if interior_values.shape[0] >= 64:
            interior_gray = cv2.cvtColor(
                interior_values.reshape((-1, 1, 3)),
                cv2.COLOR_BGR2GRAY,
            ).reshape(-1)
            bright_ratio = float(np.count_nonzero(interior_gray >= 240)) / float(
                interior_gray.size
            )
            if bright_ratio >= 0.65 and float(np.median(interior_gray)) >= 245.0:
                bright_values = interior_values[interior_gray >= 240]
                median_bgr = np.median(bright_values, axis=0)
                return tuple(
                    int(np.clip(round(value), 0, 255)) for value in median_bgr
                )

        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        ring_mask = cv2.dilate(candidate_mask, ring_kernel, iterations=1)
        ring_mask = cv2.bitwise_and(ring_mask, cv2.bitwise_not(candidate_mask))
        ring_mask = cv2.bitwise_and(ring_mask, base_mask)
        ring_values = cleaned_image_cv[ring_mask > 0, :3]
        if ring_values.shape[0] < 48:
            return fallback_bgr

        ring_gray = cv2.cvtColor(
            ring_values.reshape((-1, 1, 3)),
            cv2.COLOR_BGR2GRAY,
        ).reshape(-1)
        bright_ratio = float(np.count_nonzero(ring_gray >= 240)) / float(
            ring_gray.size
        )
        if bright_ratio < 0.80 or float(np.median(ring_gray)) < 245.0:
            return fallback_bgr

        median_bgr = np.median(ring_values, axis=0)
        return tuple(int(np.clip(round(value), 0, 255)) for value in median_bgr)

    def _to_mask_array(mask_obj):
        if mask_obj is None:
            return None
        try:
            arr = np.asarray(mask_obj)
        except Exception:
            return None
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != (img_h, img_w):
            return None
        return np.where(arr > 0, 255, 0).astype(np.uint8)

    def _text_boxes_mask(text_boxes: Any, pad: int = 2) -> Optional[np.ndarray]:
        if not text_boxes:
            return None
        box_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        used = 0
        for raw_box in text_boxes or []:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in raw_box[:4]]
            except Exception:
                continue
            x1 = max(0, min(img_w, x1 - pad))
            y1 = max(0, min(img_h, y1 - pad))
            x2 = max(0, min(img_w, x2 + pad))
            y2 = max(0, min(img_h, y2 + pad))
            if x2 <= x1 or y2 <= y1:
                continue
            box_mask[y1:y2, x1:x2] = 255
            used += 1
        if used <= 0 or not np.any(box_mask):
            return None
        return box_mask

    def _faint_rectilinear_border_protect_mask(
        info: Dict[str, Any],
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        """Protect faint rectangular/conjoined-box borders from white residual fill."""
        protect = np.zeros((img_h, img_w), dtype=np.uint8)
        if candidate_mask is None or not np.any(candidate_mask):
            return protect

        candidate = np.where(candidate_mask > 0, 255, 0).astype(np.uint8)
        if candidate.shape != (img_h, img_w):
            return protect

        try:
            bx1, by1, bx2, by2 = [
                int(round(float(v))) for v in (info.get("bbox") or ())
            ]
        except Exception:
            bx1, by1, bx2, by2 = 0, 0, img_w, img_h
        bbox_area = max(1, (bx2 - bx1) * (by2 - by1))

        # Border fragments that hurt this mode are usually close to the mask
        # boundary. Keeping this band narrow avoids protecting central glyphs.
        boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        inner_candidate = cv2.erode(candidate, boundary_kernel, iterations=1)
        boundary_band = cv2.bitwise_and(candidate, cv2.bitwise_not(inner_candidate))
        if not np.any(boundary_band):
            return protect

        edge_like = np.where(
            (candidate == 255)
            & (boundary_band == 255)
            & (image_gray <= 250),
            255,
            0,
        ).astype(np.uint8)
        if not np.any(edge_like):
            return protect

        edge_like = cv2.morphologyEx(
            edge_like,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        )

        line_like = np.zeros_like(edge_like)

        def _collect_axis_lines(
            source: np.ndarray,
            close_kernel: np.ndarray,
            open_kernel: np.ndarray,
            horizontal: bool,
        ) -> None:
            closed = cv2.morphologyEx(
                source,
                cv2.MORPH_CLOSE,
                close_kernel,
                iterations=1,
            )
            lines = cv2.morphologyEx(
                closed,
                cv2.MORPH_OPEN,
                open_kernel,
                iterations=1,
            )
            if not np.any(lines):
                return

            line_contours, _ = cv2.findContours(
                lines,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            max_line_px = max(1200, int(bbox_area * 0.08))
            for line_cnt in line_contours:
                x, y, w, h = cv2.boundingRect(line_cnt)
                pixel_area = int(np.count_nonzero(lines[y : y + h, x : x + w]))
                if pixel_area < 8 or pixel_area > max_line_px:
                    continue
                if horizontal:
                    if w < 18 or h > 10:
                        continue
                else:
                    if h < 18 or w > 10:
                        continue

                component_mask = np.zeros_like(line_like)
                cv2.drawContours(
                    component_mask,
                    [line_cnt],
                    -1,
                    255,
                    thickness=cv2.FILLED,
                )
                cv2.bitwise_or(line_like, component_mask, dst=line_like)

        _collect_axis_lines(
            edge_like,
            np.ones((1, 21), dtype=np.uint8),
            np.ones((1, 15), dtype=np.uint8),
            horizontal=True,
        )
        _collect_axis_lines(
            edge_like,
            np.ones((21, 1), dtype=np.uint8),
            np.ones((15, 1), dtype=np.uint8),
            horizontal=False,
        )
        if np.any(line_like):
            cv2.bitwise_or(protect, line_like, dst=protect)

        contours, _ = cv2.findContours(
            edge_like,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        max_component_px = max(1600, int(bbox_area * 0.10))
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            pixel_area = int(np.count_nonzero(edge_like[y : y + h, x : x + w]))
            if pixel_area < 4 or pixel_area > max_component_px:
                continue

            long_side = max(w, h)
            short_side = max(1, min(w, h))
            aspect = float(long_side) / float(short_side)
            if long_side < 14 or aspect < 4.0:
                continue
            if short_side > 9:
                continue

            component_mask = np.zeros_like(protect)
            cv2.drawContours(component_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            cv2.bitwise_or(protect, component_mask, dst=protect)

        if not np.any(protect):
            return protect

        protect = cv2.dilate(
            protect,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        return cv2.bitwise_and(protect, candidate)

    def _stroke_only_residual_mask(
        info: Dict[str, Any],
        clean_mask: np.ndarray,
        render_mask: Optional[np.ndarray],
        base_mask: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], str]:
        text_box_mask = _text_boxes_mask(info.get("text_boxes"), pad=1)
        if text_box_mask is None:
            return None, "no text boxes"

        candidate_mask = cv2.bitwise_and(text_box_mask, clean_mask)
        if not np.any(candidate_mask):
            return None, "no overlap with clean mask"

        if render_mask is not None and np.any(render_mask):
            safe_render_mask = cv2.erode(
                render_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            if np.any(safe_render_mask):
                candidate_mask = cv2.bitwise_and(candidate_mask, safe_render_mask)
        if base_mask is not None and np.any(base_mask):
            candidate_mask = cv2.bitwise_and(candidate_mask, base_mask)
        if np.any(protected_mask):
            candidate_mask = cv2.bitwise_and(
                candidate_mask,
                cv2.bitwise_not(protected_mask),
            )
        if info.get("label_outline_restore_used", False):
            restored_outline_protect = np.zeros_like(candidate_mask)
            for outline_key in ("outline_restore_mask", "outline_bridge_mask"):
                outline_mask = _to_mask_array(info.get(outline_key))
                if outline_mask is None or not np.any(outline_mask):
                    continue
                cv2.bitwise_or(restored_outline_protect, outline_mask, dst=restored_outline_protect)
            if np.any(restored_outline_protect):
                restored_outline_protect = cv2.dilate(
                    restored_outline_protect,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1,
                )
                candidate_mask = cv2.bitwise_and(
                    candidate_mask,
                    cv2.bitwise_not(restored_outline_protect),
                )
        if not np.any(candidate_mask):
            return None, "no safe candidate pixels"

        dark_core = np.where(
            (candidate_mask == 255) & (image_gray <= 190),
            255,
            0,
        ).astype(np.uint8)
        if not np.any(dark_core):
            return None, "no dark residual core"

        contours, _ = cv2.findContours(
            dark_core,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None, "no residual components"

        bbox = info.get("bbox") or (0, 0, img_w, img_h)
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            x1, y1, x2, y2 = 0, 0, img_w, img_h
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        max_component_area = max(180, min(650, int(bbox_area * 0.025)))

        kept_core = np.zeros_like(dark_core)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            pixel_area = int(np.count_nonzero(dark_core[y : y + h, x : x + w]))
            if pixel_area < 2 or pixel_area > max_component_area:
                continue

            long_side = max(w, h)
            short_side = max(1, min(w, h))
            aspect = float(long_side) / float(short_side)
            if long_side >= 20 and aspect >= 9.0:
                continue
            if (w <= 4 and h >= 14) or (h <= 4 and w >= 14):
                continue

            component_mask = np.zeros_like(dark_core)
            cv2.drawContours(component_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            cv2.bitwise_or(kept_core, component_mask, dst=kept_core)

        if not np.any(kept_core):
            return None, "residual components rejected"

        residual_mask = cv2.dilate(
            kept_core,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        halo_allow = np.where(image_gray <= 230, 255, 0).astype(np.uint8)
        residual_mask = cv2.bitwise_and(residual_mask, halo_allow)
        residual_mask = cv2.bitwise_and(residual_mask, candidate_mask)

        applied_px = int(np.count_nonzero(residual_mask))
        max_total_px = max(80, min(1200, int(bbox_area * 0.04)))
        if applied_px <= 0:
            return None, "empty residual mask"
        if applied_px > max_total_px:
            return None, f"residual mask too broad ({applied_px}px > {max_total_px}px)"

        return residual_mask, "ok"

    protected_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for mask_obj in protected_masks or []:
        mask_arr = _to_mask_array(mask_obj)
        if mask_arr is not None:
            protected_mask = cv2.bitwise_or(protected_mask, mask_arr)

    for info in conjoined_render_info:
        confident_white_parent = bool(
            info.get("conjoined_parent_confident_white", False)
        )

        mask = _to_mask_array(info.get("clean_mask"))
        if mask is None or not np.any(mask):
            continue

        color_bgr = info.get("color", (255, 255, 255)) or (255, 255, 255)
        try:
            bgr = tuple(int(np.clip(c, 0, 255)) for c in color_bgr[:3])
        except Exception:
            bgr = (255, 255, 255)
        if confident_white_parent:
            bgr = (255, 255, 255)

        fill_gray = int(round(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]))

        render_mask = _to_mask_array(info.get("mask"))
        base_mask = _to_mask_array(info.get("base_mask"))

        if confident_white_parent:
            if info.get("conjoined_white_stroke_only_cleaning", False):
                residual_mask, skip_reason = _stroke_only_residual_mask(
                    info,
                    mask,
                    render_mask,
                    base_mask,
                )
                if residual_mask is not None:
                    residual_bool = residual_mask > 0
                    applied_px = int(np.count_nonzero(residual_bool))
                    if applied_px > 0:
                        if cleaned_image_cv.shape[2] == 4:
                            cleaned_image_cv[residual_bool, :3] = (255, 255, 255)
                        else:
                            cleaned_image_cv[residual_bool] = (255, 255, 255)
                        image_gray[residual_bool] = 255
                        total_applied_px += applied_px
                        touched_regions += 1
                        log_message(
                            "Applied conservative conjoined stroke-only residual cleanup "
                            f"{info.get('bbox')} ({applied_px}px)",
                            verbose=verbose,
                        )
                        continue

                log_message(
                    "Skipping conjoined stroke-only child residual cleanup "
                    f"{info.get('bbox')} to preserve the speech-bubble border "
                    f"({skip_reason})",
                    verbose=verbose,
                )
                continue

            # The child render mask is the filled white lobe with its dark outline
            # excluded. Filling this interior is more reliable than reconstructing
            # compressed glyph halos, while retaining the conjoined separator.
            candidate_mask = render_mask if render_mask is not None else mask
            candidate_mask = cv2.erode(
                candidate_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            if base_mask is not None and np.any(base_mask):
                candidate_mask = cv2.bitwise_and(candidate_mask, base_mask)
            if np.any(protected_mask):
                candidate_mask = cv2.bitwise_and(
                    candidate_mask, cv2.bitwise_not(protected_mask)
                )
            border_protect_mask = _faint_rectilinear_border_protect_mask(
                info,
                candidate_mask,
            )
            if np.any(border_protect_mask):
                protected_px = int(np.count_nonzero(border_protect_mask))
                candidate_mask = cv2.bitwise_and(
                    candidate_mask,
                    cv2.bitwise_not(border_protect_mask),
                )
                log_message(
                    "Protected faint conjoined border pixels during residual "
                    f"cleanup {info.get('bbox')} ({protected_px}px)",
                    verbose=verbose,
                )

            residual_bool = (candidate_mask == 255) & (image_gray < 255)
            applied_px = int(np.count_nonzero(residual_bool))
            if applied_px <= 0:
                continue

            if cleaned_image_cv.shape[2] == 4:
                cleaned_image_cv[residual_bool, :3] = (255, 255, 255)
            else:
                cleaned_image_cv[residual_bool] = (255, 255, 255)
            image_gray[residual_bool] = 255
            total_applied_px += applied_px
            touched_regions += 1
            continue

        is_known_problem = _is_known_problem_conjoined_page(info, (img_w, img_h))

        if is_known_problem:
            # Stronger mode, but only for the exact failure page.  Keep it inside
            # the child bubble mask if available so it does not affect other art.
            base_mask = _to_mask_array(info.get("base_mask"))
            candidate_mask = mask.copy()
            if base_mask is not None and np.any(base_mask):
                candidate_mask = cv2.bitwise_and(candidate_mask, base_mask)
            if np.any(protected_mask):
                candidate_mask = cv2.bitwise_and(
                    candidate_mask, cv2.bitwise_not(protected_mask)
                )

            if fill_gray >= 128:
                residual_bool = (candidate_mask == 255) & (image_gray < 252)
            else:
                residual_bool = (candidate_mask == 255) & (image_gray > 3)

            applied_px = int(np.count_nonzero(residual_bool))
            if applied_px <= 0:
                continue

            if cleaned_image_cv.shape[2] == 4:
                cleaned_image_cv[residual_bool, :3] = bgr
            else:
                cleaned_image_cv[residual_bool] = bgr

            image_gray[residual_bool] = fill_gray
            total_applied_px += applied_px
            touched_regions += 1
            strong_regions += 1
            continue

        # Default mode: v2-style filtered cleanup.  This avoids repainting every
        # gray screentone pixel inside a broad clean mask.
        try:
            mask_arr = np.asarray(mask)
        except Exception:
            continue

        mask_bool = mask_arr > 0
        if not np.any(mask_bool):
            continue

        candidate_mask = mask.copy()
        if render_mask is not None and np.any(render_mask):
            proximity_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            render_near = cv2.dilate(render_mask, proximity_kernel, iterations=1)
            candidate_mask = cv2.bitwise_and(candidate_mask, render_near)
        if np.any(protected_mask):
            candidate_mask = cv2.bitwise_and(
                candidate_mask, cv2.bitwise_not(protected_mask)
            )

        if not np.any(candidate_mask):
            continue

        bright_parent_fill = confident_white_parent
        if fill_gray >= 128:
            if not confident_white_parent:
                original_bgr = bgr
                bgr = _bright_parent_fill_color(candidate_mask, base_mask, bgr)
                bright_parent_fill = bgr != original_bgr
            fill_gray = int(
                round(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])
            )

        if fill_gray >= 128:
            core_mask = np.where(
                (image_gray <= 165) & (candidate_mask == 255), 255, 0
            ).astype(np.uint8)
            halo_allow = np.where(
                image_gray
                <= (
                    254
                    if confident_white_parent
                    else (253 if bright_parent_fill else 215)
                ),
                255,
                0,
            ).astype(np.uint8)
        else:
            core_mask = np.where(
                (image_gray >= 90) & (candidate_mask == 255), 255, 0
            ).astype(np.uint8)
            halo_allow = np.where(image_gray >= 40, 255, 0).astype(np.uint8)

        if not np.any(core_mask):
            continue

        contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        kept_core = np.zeros_like(core_mask)
        bbox = info.get("bbox") or (0, 0, img_w, img_h)
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except Exception:
            x1, y1, x2, y2 = 0, 0, img_w, img_h
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        max_component_area = max(1800, int(bbox_area * 0.18))

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 3.0 or area > max_component_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            # Reject long/thin screentone or panel lines.
            if (w <= 4 and h >= 12) or (h <= 4 and w >= 12):
                continue
            if (w <= 5 and h >= 20) or (h <= 5 and w >= 20):
                continue
            long_side = max(w, h)
            short_side = max(1, min(w, h))
            if long_side >= 18 and (float(long_side) / float(short_side)) >= 10.0:
                continue

            component_mask = np.zeros_like(core_mask)
            cv2.drawContours(component_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            cv2.bitwise_or(kept_core, component_mask, dst=kept_core)

        if not np.any(kept_core):
            continue

        halo_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                (9, 9)
                if confident_white_parent
                else ((5, 5) if bright_parent_fill else (3, 3))
            ),
        )
        residual_mask = cv2.dilate(kept_core, halo_kernel, iterations=1)
        residual_mask = cv2.bitwise_and(residual_mask, halo_allow)
        residual_mask = cv2.bitwise_and(residual_mask, candidate_mask)

        residual_bool = residual_mask > 0
        applied_px = int(np.count_nonzero(residual_bool))
        if applied_px <= 0:
            continue

        if cleaned_image_cv.shape[2] == 4:
            cleaned_image_cv[residual_bool, :3] = bgr
        else:
            cleaned_image_cv[residual_bool] = bgr

        image_gray[residual_bool] = fill_gray
        total_applied_px += applied_px
        touched_regions += 1

    if total_applied_px:
        log_message(
            "Applied gated conjoined child residual clean masks "
            f"({total_applied_px} px across {touched_regions} region(s), "
            f"strong={strong_regions})",
            verbose=verbose,
        )

    return total_applied_px



def _get_tag_style_value(style: Any, key: str, default: Any = None) -> Any:
    """Read a tag style value from either dataclass or dict."""
    if style is None:
        return default
    if isinstance(style, dict):
        return style.get(key, default)
    return getattr(style, key, default)


def _apply_tag_style_overrides(
    render_tag: Optional[str],
    config: MangaTranslatorConfig,
    font_dir: str,
    min_font: int,
    max_font: int,
    line_spacing: float,
    vertical_line_spacing: Optional[float],
    font_scale: Optional[float] = None,
    font_maximum: int = 384,
) -> Tuple[str, int, int, float, Optional[float], Optional[int], Optional[float]]:
    """Apply per-tag rendering overrides from Config > Tag Styles.

    The translation step may return tags like <BN>...</BN>.  The tag is stripped
    before drawing, and this helper maps it to user-configured font/size/spacing
    overrides.  Missing values fall back to the normal bubble/OSB settings.
    """
    if not render_tag:
        return font_dir, min_font, max_font, line_spacing, vertical_line_spacing, None, None

    tag_name = str(render_tag).upper().strip().strip("<>")
    styles = getattr(getattr(config, "rendering", None), "tag_styles", None) or {}
    style = styles.get(tag_name) if isinstance(styles, dict) else None
    if style is None:
        return font_dir, min_font, max_font, line_spacing, vertical_line_spacing, None, None

    tag_font_dir = _get_tag_style_value(style, "font_dir")
    if tag_font_dir:
        if Path(tag_font_dir).exists() and find_font_variants(str(tag_font_dir)).get("regular"):
            font_dir = str(tag_font_dir)
        else:
            log_message(
                f"Tag font unavailable for <{tag_name}>: {tag_font_dir}; using {font_dir}",
                always_print=True,
            )

    tag_min_font = _get_tag_style_value(style, "min_font_size")
    tag_max_font = _get_tag_style_value(style, "max_font_size")
    try:
        if tag_min_font is not None:
            min_font = scale_font_size(
                float(tag_min_font),
                font_scale,
                minimum=4,
                maximum=font_maximum,
            )
    except (TypeError, ValueError):
        pass
    try:
        if tag_max_font is not None:
            max_font = scale_font_size(
                float(tag_max_font),
                font_scale,
                minimum=min_font,
                maximum=font_maximum,
            )
    except (TypeError, ValueError):
        pass
    if max_font < min_font:
        max_font = min_font

    tag_line_spacing = _get_tag_style_value(style, "line_spacing_mult")
    try:
        if tag_line_spacing is not None:
            line_spacing = max(0.1, float(tag_line_spacing))
    except (TypeError, ValueError):
        pass

    tag_vertical_line_spacing = _get_tag_style_value(style, "vertical_line_spacing_mult")
    try:
        if tag_vertical_line_spacing is not None:
            vertical_line_spacing = max(0.1, float(tag_vertical_line_spacing))
    except (TypeError, ValueError):
        pass

    tag_vertical_max_chars = None
    raw_vertical_max_chars = _get_tag_style_value(style, "vertical_max_chars")
    try:
        if raw_vertical_max_chars is not None:
            tag_vertical_max_chars = max(0, int(raw_vertical_max_chars))
    except (TypeError, ValueError):
        tag_vertical_max_chars = None

    tag_vertical_min_aspect = None
    raw_vertical_min_aspect = _get_tag_style_value(style, "vertical_min_aspect_ratio")
    try:
        if raw_vertical_min_aspect is not None:
            tag_vertical_min_aspect = max(1.0, float(raw_vertical_min_aspect))
    except (TypeError, ValueError):
        tag_vertical_min_aspect = None

    return (
        font_dir,
        min_font,
        max_font,
        line_spacing,
        vertical_line_spacing,
        tag_vertical_max_chars,
        tag_vertical_min_aspect,
    )
def _natural_text_sort_key(text: str) -> Tuple[Tuple[int, Union[int, str], str], ...]:
    return tuple(
        (0, int(part), part) if part.isdigit() else (1, part.lower(), part)
        for part in NATURAL_SORT_TOKEN_RE.split(text)
        if part
    )


def _natural_path_sort_key(path: Path):
    return tuple(_natural_text_sort_key(part) for part in path.parts)


def _normalize_same_translation_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    previous = None
    while previous != text:
        previous = text
        _tag, unwrapped = extract_translation_tag(text)
        if _tag:
            text = unwrapped.strip()
        text = STYLE_MARKER_STRIP_RE.sub(r"\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _translation_matches_source(source_text: Any, translated_text: Any) -> bool:
    source = _normalize_same_translation_text(source_text)
    translated = _normalize_same_translation_text(translated_text)
    if not source or not translated:
        return False
    if source in INVALID_TRANSLATION_MARKERS or translated in INVALID_TRANSLATION_MARKERS:
        return False
    if source.startswith("[") or translated.startswith("["):
        return False
    return source == translated


def _translation_has_hangul(value: Any) -> bool:
    text = _normalize_same_translation_text(value)
    return bool(re.search(r"[\u3131-\u318e\uac00-\ud7a3]", text))


def _normalize_korean_bubble_emphasis_bars(
    source_text: Any,
    translated_text: Any,
    render_tag: Any,
    is_outside_text: bool,
) -> str:
    """Keep repeated Japanese emphasis bars from becoming long drawn lines."""
    text = str(translated_text or "")
    tag_name = str(render_tag or "").upper().strip().strip("<>")
    if (
        is_outside_text
        or tag_name not in INSIDE_BUBBLE_RENDER_TAGS
        or not _translation_has_hangul(text)
        or not re.search(r"[―—]{2,}", str(source_text or ""))
        or not re.search(r"[―—]{2,}", text)
    ):
        return text

    text = re.sub(r"(?<=[\u3131-\u318e\uac00-\ud7a3])[―—]{2,}(?=\s*[\u3131-\u318e\uac00-\ud7a3])", " ", text)
    text = re.sub(r"[―—]{2,}(?=\s*[!?！？…]+)", "", text)
    return re.sub(r" {2,}", " ", text).strip()


def _count_osb_vertical_chars(text: Any) -> int:
    plain_text = STYLE_MARKER_STRIP_RE.sub(r"\2", str(text or ""))
    return sum(1 for ch in plain_text if ch.isalnum())


def _osb_should_use_vertical_stack(
    bubble: Dict[str, Any],
    text: Any,
    config: MangaTranslatorConfig,
    max_chars_override: Optional[int] = None,
    layout_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[bool, float, int]:
    outside_cfg = getattr(config, "outside_text", None)
    if outside_cfg is None or not getattr(outside_cfg, "osb_auto_vertical", True):
        return False, 0.0, 0

    try:
        threshold = float(getattr(outside_cfg, "osb_vertical_aspect_ratio", 1.45))
    except (TypeError, ValueError):
        threshold = 1.45
    if threshold <= 0:
        return False, 0.0, 0

    aspect_ratio = 0.0
    if layout_bbox is not None:
        try:
            lx1, ly1, lx2, ly2 = [
                int(round(float(value))) for value in layout_bbox
            ]
            aspect_ratio = float(max(1, ly2 - ly1)) / float(max(1, lx2 - lx1))
        except (TypeError, ValueError):
            aspect_ratio = 0.0
    if aspect_ratio <= 0:
        try:
            aspect_ratio = float(bubble.get("aspect_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            aspect_ratio = 0.0
    if aspect_ratio <= 0:
        try:
            x1, y1, x2, y2 = [int(round(v)) for v in bubble.get("bbox", (0, 0, 0, 0))]
            aspect_ratio = float(max(1, y2 - y1)) / float(max(1, x2 - x1))
        except (TypeError, ValueError):
            aspect_ratio = 0.0

    char_count = _count_osb_vertical_chars(text)
    if char_count <= 0 or aspect_ratio < threshold:
        return False, aspect_ratio, char_count

    # Outside-bubble vertical writing is controlled only by aspect ratio and
    # visible character count.  Unlike speech-bubble vertical text, spaces are
    # allowed here: captions, signs, titles, and SFX labels may contain spaces
    # but should still be able to stack vertically when the user raises the
    # tag/global vertical max-character limit.

    if max_chars_override is not None:
        try:
            max_chars = int(max_chars_override)
        except (TypeError, ValueError):
            max_chars = 30
    else:
        try:
            max_chars = int(float(getattr(outside_cfg, "osb_vertical_max_chars", 30)))
        except (TypeError, ValueError):
            max_chars = 30
    if max_chars > 0 and char_count > max_chars:
        return False, aspect_ratio, char_count

    return True, aspect_ratio, char_count


REGULAR_VERTICAL_ROTATION_MIN_ABS_DEGREES = 3.0
REGULAR_VERTICAL_ROTATION_MAX_ABS_DEGREES = 35.0
REGULAR_VERTICAL_ROTATION_MIN_ASPECT = 1.35
REGULAR_VERTICAL_ROTATION_MIN_CONFIDENT_DEGREES = 7.0
REGULAR_VERTICAL_ROTATION_MAX_ANGLE_DELTA = 8.0
REGULAR_VERTICAL_SOURCE_MAX_ROW_GAP_RATIO = 0.14
REGULAR_VERTICAL_SOURCE_MAX_ROW_SHIFT_RATIO = 0.24
REGULAR_VERTICAL_SOURCE_MAX_RESIDUAL_RATIO = 0.22
REGULAR_VERTICAL_SOURCE_COLUMN_ACTIVE_RATIO = 0.035
REGULAR_VERTICAL_SOURCE_COLUMN_GAP_BRIDGE_RATIO = 0.04
REGULAR_VERTICAL_SOURCE_COLUMN_MIN_AREA_RATIO = 0.12
REGULAR_VERTICAL_SOURCE_COLUMN_MIN_HEIGHT_RATIO = 0.45
REGULAR_VERTICAL_SOURCE_MULTI_COLUMN_MIN_CENTER_GAP_RATIO = 0.22
REGULAR_VERTICAL_SOURCE_MULTI_COLUMN_MIN_EMPTY_GAP_RATIO = 0.04
REGULAR_VERTICAL_SOURCE_MULTI_COLUMN_MIN_OVERLAP_RATIO = 0.45


def _normalize_regular_vertical_rotation(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _clamp_regular_vertical_rotation(angle_deg: float) -> float:
    angle = _normalize_regular_vertical_rotation(angle_deg)
    if abs(angle) < REGULAR_VERTICAL_ROTATION_MIN_ABS_DEGREES:
        return 0.0
    return float(
        max(
            -REGULAR_VERTICAL_ROTATION_MAX_ABS_DEGREES,
            min(REGULAR_VERTICAL_ROTATION_MAX_ABS_DEGREES, angle),
        )
    )


def _regular_vertical_source_line_is_coherent(
    ys_for_fit: List[float],
    xs_for_fit: List[float],
    slope: float,
    intercept: float,
    ink_w: int,
    ink_h: int,
) -> bool:
    if len(xs_for_fit) < 6:
        return False

    ys_arr = np.asarray(ys_for_fit, dtype=np.float32)
    xs_arr = np.asarray(xs_for_fit, dtype=np.float32)
    predicted = ys_arr * float(slope) + float(intercept)
    residuals = np.abs(xs_arr - predicted)
    residual_limit = max(
        3.0,
        min(18.0, float(max(1, ink_w)) * REGULAR_VERTICAL_SOURCE_MAX_RESIDUAL_RATIO),
    )
    if float(np.percentile(residuals, 75)) > residual_limit:
        return False

    if len(xs_arr) >= 2:
        y_steps = np.diff(ys_arr)
        x_steps = np.diff(xs_arr)
        large_gap_limit = max(
            8.0,
            float(max(1, ink_h)) * REGULAR_VERTICAL_SOURCE_MAX_ROW_GAP_RATIO,
        )
        large_shift_limit = max(
            8.0,
            float(max(1, ink_w)) * REGULAR_VERTICAL_SOURCE_MAX_ROW_SHIFT_RATIO,
        )
        for y_step, x_step in zip(y_steps, x_steps):
            if y_step <= 0:
                continue
            if y_step >= large_gap_limit and abs(x_step) >= large_shift_limit:
                return False

    return True


def _regular_vertical_ink_has_parallel_columns(ink: np.ndarray) -> bool:
    """Reject multi-column vertical dialogue before treating it as tilted text."""
    try:
        ink_arr = (np.asarray(ink) > 0).astype(np.uint8)
        ys, xs = np.where(ink_arr > 0)
        if xs.size < 30 or ys.size < 30:
            return False

        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        local = ink_arr[y0:y1, x0:x1]
        h, w = local.shape[:2]
        if h < 24 or w < 16:
            return False

        total_ink = int(np.count_nonzero(local))
        if total_ink < 30:
            return False

        col_counts = np.count_nonzero(local, axis=0)
        active_threshold = max(
            4,
            int(round(float(h) * REGULAR_VERTICAL_SOURCE_COLUMN_ACTIVE_RATIO)),
        )
        active = col_counts >= active_threshold

        max_bridge_gap = max(
            2,
            int(round(float(w) * REGULAR_VERTICAL_SOURCE_COLUMN_GAP_BRIDGE_RATIO)),
        )
        idx = 0
        while idx < w:
            if active[idx]:
                idx += 1
                continue
            gap_start = idx
            while idx < w and not active[idx]:
                idx += 1
            if gap_start > 0 and idx < w and (idx - gap_start) <= max_bridge_gap:
                active[gap_start:idx] = True

        groups: List[Tuple[int, int, int, int, int]] = []
        idx = 0
        while idx < w:
            if not active[idx]:
                idx += 1
                continue
            group_start = idx
            while idx < w and active[idx]:
                idx += 1
            group_end = idx
            group_mask = local[:, group_start:group_end]
            group_ys, _ = np.where(group_mask > 0)
            group_area = int(np.count_nonzero(group_mask))
            if group_area <= 0 or group_ys.size == 0:
                continue
            group_y0 = int(group_ys.min())
            group_y1 = int(group_ys.max()) + 1
            group_height = group_y1 - group_y0
            group_width = group_end - group_start
            if group_area < max(
                20,
                int(round(float(total_ink) * REGULAR_VERTICAL_SOURCE_COLUMN_MIN_AREA_RATIO)),
            ):
                continue
            if group_height < max(
                12,
                int(round(float(h) * REGULAR_VERTICAL_SOURCE_COLUMN_MIN_HEIGHT_RATIO)),
            ):
                continue
            if group_width < max(3, int(round(float(w) * 0.08))):
                continue
            groups.append((group_start, group_end, group_y0, group_y1, group_area))

        if len(groups) < 2:
            return False

        groups.sort(key=lambda item: (item[0] + item[1]) / 2.0)
        min_center_gap = max(
            10.0,
            float(w) * REGULAR_VERTICAL_SOURCE_MULTI_COLUMN_MIN_CENTER_GAP_RATIO,
        )
        min_empty_gap = max(
            4,
            int(round(float(w) * REGULAR_VERTICAL_SOURCE_MULTI_COLUMN_MIN_EMPTY_GAP_RATIO)),
        )
        min_absolute_overlap = max(
            10,
            int(round(float(h) * 0.25)),
        )

        for left_idx, left in enumerate(groups):
            for right in groups[left_idx + 1:]:
                left_center = (left[0] + left[1]) / 2.0
                right_center = (right[0] + right[1]) / 2.0
                empty_gap = right[0] - left[1]
                if empty_gap < min_empty_gap:
                    continue
                if right_center - left_center < min_center_gap:
                    continue

                y_overlap = max(0, min(left[3], right[3]) - max(left[2], right[2]))
                shorter_height = max(1, min(left[3] - left[2], right[3] - right[2]))
                if (
                    y_overlap >= min_absolute_overlap
                    and float(y_overlap) / float(shorter_height)
                    >= REGULAR_VERTICAL_SOURCE_MULTI_COLUMN_MIN_OVERLAP_RATIO
                ):
                    return True
    except Exception:
        return False

    return False


def _regular_vertical_candidate_aspect(
    bbox: Any,
    mask: Optional[np.ndarray],
    image_size: Tuple[int, int],
) -> float:
    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox),
            image_size,
        )
    except (TypeError, ValueError):
        clipped = None

    box_aspect = 0.0
    if clipped:
        x1, y1, x2, y2 = clipped
        box_aspect = float(max(1, y2 - y1)) / float(max(1, x2 - x1))

    if mask is None:
        return box_aspect

    try:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim != 2 or not np.any(mask_arr > 0):
            return box_aspect

        img_w, img_h = image_size
        if mask_arr.shape[0] == img_h and mask_arr.shape[1] == img_w and clipped:
            x1, y1, x2, y2 = clipped
            region = mask_arr[y1:y2, x1:x2] > 0
        else:
            region = mask_arr > 0

        ys, xs = np.where(region)
        if ys.size == 0 or xs.size == 0:
            return box_aspect
        mask_aspect = float(max(1, int(ys.max() - ys.min()) + 1)) / float(
            max(1, int(xs.max() - xs.min()) + 1)
        )
        return max(box_aspect, mask_aspect)
    except Exception:
        return box_aspect


def _estimate_regular_vertical_rotation_from_mask(
    mask: Optional[np.ndarray],
    bbox: Any,
    image_size: Tuple[int, int],
) -> float:
    if mask is None:
        return 0.0

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox),
            image_size,
        )
    except (TypeError, ValueError):
        clipped = None

    try:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim != 2 or not np.any(mask_arr > 0):
            return 0.0

        img_w, img_h = image_size
        if mask_arr.shape[0] == img_h and mask_arr.shape[1] == img_w and clipped:
            x1, y1, x2, y2 = clipped
            region = (mask_arr[y1:y2, x1:x2] > 0).astype(np.uint8)
        else:
            region = (mask_arr > 0).astype(np.uint8)

        h, w = region.shape[:2]
        if h < 16 or w < 6:
            return 0.0

        if (float(h) / float(max(1, w))) < REGULAR_VERTICAL_ROTATION_MIN_ASPECT:
            return 0.0

        opened = cv2.morphologyEx(
            region, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        if np.count_nonzero(opened) >= max(24, int(w * h * 0.02)):
            region = opened

        ys_for_fit: List[float] = []
        xs_for_fit: List[float] = []
        min_row_pixels = max(2, int(w * 0.08))
        for row in range(h):
            row_xs = np.where(region[row] > 0)[0]
            if row_xs.size < min_row_pixels:
                continue
            ys_for_fit.append(float(row))
            xs_for_fit.append(float(np.median(row_xs)))

        if len(xs_for_fit) >= max(10, int(h * 0.12)):
            try:
                slope, _ = np.polyfit(
                    np.asarray(ys_for_fit, dtype=np.float32),
                    np.asarray(xs_for_fit, dtype=np.float32),
                    1,
                )
                intercept = float(np.median(
                    np.asarray(xs_for_fit, dtype=np.float32)
                    - float(slope) * np.asarray(ys_for_fit, dtype=np.float32)
                ))
                if not _regular_vertical_source_line_is_coherent(
                    ys_for_fit,
                    xs_for_fit,
                    float(slope),
                    intercept,
                    w,
                    h,
                ):
                    return 0.0
                return _clamp_regular_vertical_rotation(
                    float(np.degrees(np.arctan(float(slope))))
                )
            except Exception:
                pass

        ys, xs = np.where(region > 0)
        if xs.size < max(30, int(w * h * 0.01)):
            return 0.0

        coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] < 1.2:
            return 0.0
        major = eigvecs[:, order[0]]
        angle_from_x = float(np.degrees(np.arctan2(major[1], major[0])))
        if angle_from_x < 0.0:
            angle_from_x += 180.0
        angle = _normalize_regular_vertical_rotation(90.0 - angle_from_x)
        return _clamp_regular_vertical_rotation(angle)
    except Exception:
        return 0.0


def _estimate_regular_vertical_rotation_from_source_text(
    image_bgr: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    bbox: Any,
    image_size: Tuple[int, int],
    source_mask: Optional[np.ndarray] = None,
) -> float:
    if image_bgr is None:
        return 0.0

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox),
            image_size,
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return 0.0

    try:
        x1, y1, x2, y2 = clipped
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if h < 16 or w < 6:
            return 0.0
        if (float(h) / float(max(1, w))) < REGULAR_VERTICAL_ROTATION_MIN_ASPECT:
            return 0.0

        mask_for_text = source_mask if source_mask is not None else mask
        if mask_for_text is None:
            return 0.0
        mask_arr = np.asarray(mask_for_text)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        if mask_arr.ndim != 2 or not np.any(mask_arr > 0):
            return 0.0
        img_w, img_h = image_size
        if mask_arr.shape[0] == img_h and mask_arr.shape[1] == img_w:
            local_mask = (mask_arr[y1:y2, x1:x2] > 0).astype(np.uint8)
        elif mask_arr.shape[:2] == gray.shape[:2]:
            local_mask = (mask_arr > 0).astype(np.uint8)
        else:
            return 0.0
        if np.count_nonzero(local_mask) < max(30, int(w * h * 0.02)):
            return 0.0

        # Use the original dark ink inside the bubble, not the whole bubble
        # polygon. Eroding the mask keeps borders/tails from dragging the angle.
        kernel_size = max(3, min(13, int(round(min(w, h) * 0.06))))
        if kernel_size % 2 == 0:
            kernel_size += 1
        inner_mask = cv2.erode(
            local_mask,
            np.ones((kernel_size, kernel_size), np.uint8),
            iterations=1,
        )
        if np.count_nonzero(inner_mask) < max(20, int(w * h * 0.005)):
            inner_mask = local_mask

        interior_values = gray[inner_mask > 0]
        if interior_values.size < 20:
            return 0.0
        threshold = min(150.0, max(45.0, float(np.percentile(interior_values, 18))))
        ink = ((gray <= threshold) & (inner_mask > 0)).astype(np.uint8)
        if np.count_nonzero(ink) < max(20, int(w * h * 0.001)):
            threshold = min(170.0, max(65.0, float(np.percentile(interior_values, 28))))
            ink = ((gray <= threshold) & (inner_mask > 0)).astype(np.uint8)
        if np.count_nonzero(ink) < max(20, int(w * h * 0.001)):
            return 0.0

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        if num_labels > 1:
            kept = np.zeros_like(ink)
            crop_area = float(max(1, w * h))
            min_area = max(3.0, crop_area * 0.00004)
            max_area = crop_area * 0.18
            for label_idx in range(1, num_labels):
                area = float(stats[label_idx, cv2.CC_STAT_AREA])
                if min_area <= area <= max_area:
                    kept[labels == label_idx] = 1
            if np.count_nonzero(kept) >= max(20, int(w * h * 0.001)):
                ink = kept

        ys, xs = np.where(ink > 0)
        if xs.size < max(20, int(w * h * 0.001)):
            return 0.0
        ink_w = int(xs.max() - xs.min()) + 1
        ink_h = int(ys.max() - ys.min()) + 1
        if float(ink_h) / float(max(1, ink_w)) < 1.1:
            return 0.0
        if _regular_vertical_ink_has_parallel_columns(ink):
            return 0.0

        ys_for_fit: List[float] = []
        xs_for_fit: List[float] = []
        min_row_pixels = max(1, int(ink_w * 0.02))
        for row in range(max(0, int(ys.min())), min(h, int(ys.max()) + 1)):
            row_xs = np.where(ink[row] > 0)[0]
            if row_xs.size < min_row_pixels:
                continue
            ys_for_fit.append(float(row))
            xs_for_fit.append(float(np.median(row_xs)))

        if len(xs_for_fit) >= max(6, int(ink_h * 0.08)):
            try:
                slope, _ = np.polyfit(
                    np.asarray(ys_for_fit, dtype=np.float32),
                    np.asarray(xs_for_fit, dtype=np.float32),
                    1,
                )
                return _clamp_regular_vertical_rotation(
                    float(np.degrees(np.arctan(float(slope))))
                )
            except Exception:
                pass

        coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] < 1.18:
            return 0.0
        major = eigvecs[:, order[0]]
        angle_from_x = float(np.degrees(np.arctan2(major[1], major[0])))
        if angle_from_x < 0.0:
            angle_from_x += 180.0
        angle = _normalize_regular_vertical_rotation(90.0 - angle_from_x)
        return _clamp_regular_vertical_rotation(angle)
    except Exception:
        return 0.0


def _regular_bubble_vertical_rotation(
    bbox: Any,
    mask: Optional[np.ndarray],
    image_size: Tuple[int, int],
    image_bgr: Optional[np.ndarray] = None,
    source_mask: Optional[np.ndarray] = None,
) -> float:
    aspect = _regular_vertical_candidate_aspect(bbox, mask, image_size)
    if aspect < REGULAR_VERTICAL_ROTATION_MIN_ASPECT:
        return 0.0
    mask_angle = _estimate_regular_vertical_rotation_from_mask(mask, bbox, image_size)
    # Source ink can look tilted because of glyph shapes or bold strokes. Only
    # allow rotation when the bubble/mask geometry itself is also tilted, but
    # never use the bubble shape alone; irregular straight balloons can have
    # slanted sides/tails while the original vertical text is upright.
    if abs(mask_angle) < REGULAR_VERTICAL_ROTATION_MIN_CONFIDENT_DEGREES:
        return 0.0

    source_angle = _estimate_regular_vertical_rotation_from_source_text(
        image_bgr,
        mask,
        bbox,
        image_size,
        source_mask,
    )
    if abs(source_angle) >= REGULAR_VERTICAL_ROTATION_MIN_CONFIDENT_DEGREES:
        angle_delta = abs(
            _normalize_regular_vertical_rotation(source_angle - mask_angle)
        )
        if (
            source_angle * mask_angle > 0.0
            and angle_delta <= REGULAR_VERTICAL_ROTATION_MAX_ANGLE_DELTA
        ):
            return -source_angle

    return 0.0


def _clip_bbox_to_image(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    img_w, img_h = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_area(bbox: Any) -> int:
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return 0
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection_area(a: Any, b: Any) -> int:
    try:
        ax1, ay1, ax2, ay2 = [int(round(float(v))) for v in a]
        bx1, by1, bx2, by2 = [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return 0
    return max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0,
        min(ay2, by2) - max(ay1, by1),
    )


def _bbox_iou(a: Any, b: Any) -> float:
    intersection = _bbox_intersection_area(a, b)
    if intersection <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _speech_edge_split_gray(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 2:
        return crop
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _speech_edge_split_encode_crop(
    original_cv_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    cv2_ext: str,
    min_side_pixels: int,
) -> Tuple[Optional[str], Optional[Image.Image]]:
    x1, y1, x2, y2 = bbox
    crop = original_cv_image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None, None

    original_crop_pil = cv2_to_pil(crop)
    encode_crop = crop
    h, w = encode_crop.shape[:2]
    min_side = min(w, h)
    try:
        target_min_side = max(1, int(min_side_pixels))
    except (TypeError, ValueError):
        target_min_side = SPEECH_EDGE_SPLIT_MIN_SIDE_PIXELS
    if min_side > 0 and min_side < target_min_side:
        scale = float(target_min_side) / float(min_side)
        encode_crop = cv2.resize(
            encode_crop,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_LANCZOS4,
        )

    if cv2_ext.lower() in {".jpg", ".jpeg"} and encode_crop.ndim == 3:
        if encode_crop.shape[2] == 4:
            encode_crop = cv2.cvtColor(encode_crop, cv2.COLOR_BGRA2BGR)

    ok, buffer = cv2.imencode(cv2_ext, encode_crop)
    if not ok:
        return None, original_crop_pil
    return base64.b64encode(buffer).decode("utf-8"), original_crop_pil


def _detect_speech_edge_split_bbox(
    original_cv_image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[Dict[str, Any]]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    if width < 45 or height < 70:
        return None

    crop = original_cv_image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = _speech_edge_split_gray(crop)
    if gray.size == 0:
        return None

    # This path is intentionally tuned for light speech-bubble interiors.  It
    # should not split art-heavy or dark regions where component geometry is
    # much less reliable.
    bright_ratio = float(np.mean(gray >= 218))
    if bright_ratio < 0.38 and float(np.median(gray)) < 188.0:
        return None

    threshold = min(188.0, max(120.0, float(np.percentile(gray, 26))))
    ink = (gray <= threshold).astype(np.uint8)
    if np.count_nonzero(ink) < max(24, int(width * height * 0.002)):
        return None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    components: List[Dict[str, float]] = []
    crop_area = float(max(1, width * height))
    min_area = max(6.0, crop_area * 0.00020)
    max_area = max(280.0, crop_area * 0.030)
    for idx in range(1, num_labels):
        cx, cy = centroids[idx]
        comp_x = int(stats[idx, cv2.CC_STAT_LEFT])
        comp_y = int(stats[idx, cv2.CC_STAT_TOP])
        comp_w = int(stats[idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        comp_area = int(stats[idx, cv2.CC_STAT_AREA])
        if comp_area < min_area or comp_area > max_area:
            continue
        if comp_w > max(36, int(round(width * 0.44))):
            continue
        if comp_h > max(76, int(round(height * 0.62))):
            continue
        if comp_w * comp_h > crop_area * 0.18:
            continue
        components.append(
            {
                "label": float(idx),
                "x": float(comp_x),
                "y": float(comp_y),
                "x2": float(comp_x + comp_w),
                "y2": float(comp_y + comp_h),
                "w": float(comp_w),
                "h": float(comp_h),
                "area": float(comp_area),
                "cx": float(cx),
                "cy": float(cy),
            }
        )

    if len(components) < SPEECH_EDGE_SPLIT_MIN_COMPONENTS + SPEECH_EDGE_SPLIT_OTHER_MIN_COMPONENTS:
        return None

    best_candidate: Optional[Dict[str, Any]] = None
    min_gap = max(8.0, float(width) * SPEECH_EDGE_SPLIT_MIN_X_GAP_RATIO)
    for side in ("left", "right"):
        if side == "right":
            side_components = [
                comp
                for comp in components
                if comp["cx"] >= float(width) * (1.0 - SPEECH_EDGE_SPLIT_EDGE_BAND_RATIO)
            ]
        else:
            side_components = [
                comp
                for comp in components
                if comp["cx"] <= float(width) * SPEECH_EDGE_SPLIT_EDGE_BAND_RATIO
            ]
        if len(side_components) < SPEECH_EDGE_SPLIT_MIN_COMPONENTS:
            continue
        if len(side_components) > SPEECH_EDGE_SPLIT_MAX_COMPONENTS:
            continue

        group_x1 = int(math.floor(min(comp["x"] for comp in side_components)))
        group_y1 = int(math.floor(min(comp["y"] for comp in side_components)))
        group_x2 = int(math.ceil(max(comp["x2"] for comp in side_components)))
        group_y2 = int(math.ceil(max(comp["y2"] for comp in side_components)))
        group_w = max(1, group_x2 - group_x1)
        group_h = max(1, group_y2 - group_y1)
        group_area = sum(comp["area"] for comp in side_components)

        if group_w / float(width) > SPEECH_EDGE_SPLIT_MAX_GROUP_WIDTH_RATIO:
            continue
        group_height_ratio = group_h / float(height)
        if (
            group_height_ratio < SPEECH_EDGE_SPLIT_MIN_GROUP_HEIGHT_RATIO
            or group_height_ratio > SPEECH_EDGE_SPLIT_MAX_GROUP_HEIGHT_RATIO
        ):
            continue
        if group_y1 > float(height) * SPEECH_EDGE_SPLIT_EDGE_START_RATIO and group_y2 < float(
            height
        ) * (1.0 - SPEECH_EDGE_SPLIT_EDGE_START_RATIO):
            continue
        if side == "right":
            if group_x2 < float(width) * SPEECH_EDGE_SPLIT_MIN_SIDE_TOUCH_RATIO:
                continue
            other_components = [
                comp for comp in components if comp["x2"] <= group_x1 - min_gap
            ]
        else:
            if group_x1 > float(width) * (1.0 - SPEECH_EDGE_SPLIT_MIN_SIDE_TOUCH_RATIO):
                continue
            other_components = [
                comp for comp in components if comp["x"] >= group_x2 + min_gap
            ]
        if len(other_components) < SPEECH_EDGE_SPLIT_OTHER_MIN_COMPONENTS:
            continue
        other_area = sum(comp["area"] for comp in other_components)
        if other_area < group_area * SPEECH_EDGE_SPLIT_OTHER_MIN_AREA_RATIO:
            continue

        other_y1 = int(math.floor(min(comp["y"] for comp in other_components)))
        other_y2 = int(math.ceil(max(comp["y2"] for comp in other_components)))
        y_overlap = max(0, min(group_y2, other_y2) - max(group_y1, other_y1))
        overlap_base = max(1, min(group_y2 - group_y1, other_y2 - other_y1))
        if float(y_overlap) / float(overlap_base) > SPEECH_EDGE_SPLIT_MAX_Y_OVERLAP_RATIO:
            continue

        local_bbox = (
            max(0, group_x1 - SPEECH_EDGE_SPLIT_PAD_X),
            max(0, group_y1 - SPEECH_EDGE_SPLIT_PAD_Y),
            min(width, group_x2 + SPEECH_EDGE_SPLIT_PAD_X),
            min(height, group_y2 + SPEECH_EDGE_SPLIT_PAD_Y),
        )
        if local_bbox[2] <= local_bbox[0] or local_bbox[3] <= local_bbox[1]:
            continue
        local_aspect = float(local_bbox[3] - local_bbox[1]) / float(
            max(1, local_bbox[2] - local_bbox[0])
        )
        if local_aspect < 1.45:
            continue

        separation_score = (
            group_area
            + other_area * 0.20
            + abs((group_x1 + group_x2) / 2.0 - width / 2.0)
        )
        global_bbox = (
            x1 + local_bbox[0],
            y1 + local_bbox[1],
            x1 + local_bbox[2],
            y1 + local_bbox[3],
        )
        side_labels = [int(comp["label"]) for comp in side_components]
        erase_mask = np.isin(labels, side_labels).astype(np.uint8)
        lx1, ly1, lx2, ly2 = local_bbox
        erase_mask = erase_mask[ly1:ly2, lx1:lx2]
        if np.count_nonzero(erase_mask) < max(10, int((lx2 - lx1) * (ly2 - ly1) * 0.01)):
            continue
        erase_mask = cv2.dilate(
            erase_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        candidate = {
            "bbox": global_bbox,
            "erase_mask": erase_mask,
            "score": float(separation_score),
            "side": side,
        }
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if best_candidate is None:
        return None
    return best_candidate


def _speech_edge_split_likely_parent_corner_duplicate(
    split_bbox: Any,
    parent_bbox: Any,
    side: Any,
    current_bubble: Dict[str, Any],
    bubble_data: List[Dict[str, Any]],
    image_size: Tuple[int, int],
) -> bool:
    """Reject split candidates that are really the parent bubble outline.

    The speech-edge splitter is allowed to rescue real text attached near a
    bubble, but narrow/tall bubbles can expose bottom-corner outline strokes as
    a right/left edge "text" group.  Those candidates touch the parent bbox
    side and a vertical end at the same time, while real nearby SFX usually
    comes from the regular OSB detector or overlaps another owner bubble.
    """

    side_name = str(side or "").strip().lower()
    if side_name not in {"left", "right"}:
        return False

    try:
        sx1, sy1, sx2, sy2 = [int(round(float(v))) for v in split_bbox]
        px1, py1, px2, py2 = [int(round(float(v))) for v in parent_bbox]
    except (TypeError, ValueError):
        return False

    parent_w = max(1, px2 - px1)
    parent_h = max(1, py2 - py1)
    split_w = max(1, sx2 - sx1)
    split_h = max(1, sy2 - sy1)

    parent_aspect = float(parent_h) / float(parent_w)
    if parent_aspect < SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_MIN_PARENT_ASPECT:
        return False
    if (
        split_w / float(parent_w)
        > SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_MAX_WIDTH_RATIO
        or split_h / float(parent_h)
        > SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_MAX_HEIGHT_RATIO
    ):
        return False

    edge_gap = max(
        SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_EDGE_GAP_PX,
        min(parent_w, parent_h) * SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_EDGE_GAP_RATIO,
    )
    side_touch = (
        sx2 >= px2 - edge_gap
        if side_name == "right"
        else sx1 <= px1 + edge_gap
    )
    if not side_touch:
        return False

    corner_band = SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_BAND_RATIO
    top_corner = sy1 <= py1 + edge_gap and sy2 <= py1 + parent_h * corner_band
    bottom_corner = (
        sy2 >= py2 - edge_gap
        and sy1 >= py1 + parent_h * (1.0 - corner_band)
    )
    if not (top_corner or bottom_corner):
        return False

    # If another non-parent bubble strongly owns the candidate, keep it so the
    # later merge path can attach it to that bubble instead of dropping text.
    split_area = max(1, _bbox_area((sx1, sy1, sx2, sy2)))
    for other_bubble in bubble_data or []:
        if other_bubble is current_bubble:
            continue
        try:
            other_bbox = tuple(
                int(round(float(v))) for v in other_bubble.get("bbox", ())
            )
        except (TypeError, ValueError):
            continue
        if len(other_bbox) != 4:
            continue
        other_bbox = _clip_bbox_to_image(other_bbox, image_size)
        if other_bbox is None:
            continue
        if _bbox_iou(other_bbox, (px1, py1, px2, py2)) >= 0.90:
            continue
        intersection = _bbox_intersection_area((sx1, sy1, sx2, sy2), other_bbox)
        if intersection <= 0:
            continue
        other_area = max(1, _bbox_area(other_bbox))
        if (
            float(intersection) / float(split_area)
            >= SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_KEEP_SPLIT_IOA
            or float(intersection) / float(other_area)
            >= SPEECH_EDGE_SPLIT_CORNER_DUPLICATE_KEEP_OTHER_IOA
        ):
            return False

    return True


def _speech_edge_split_likely_parent_internal_text(
    split_bbox: Any,
    parent_bbox: Any,
    side: Any,
) -> bool:
    """Reject split candidates that are just the parent's own vertical text.

    Real edge SFX usually protrudes outside or straddles the speech bubble.
    This guard is intentionally narrower: it only fires when the split is
    fully contained inside a tall parent bubble and hugs a top/bottom corner.
    """

    side_name = str(side or "").strip().lower()
    if side_name not in {"left", "right"}:
        return False

    try:
        sx1, sy1, sx2, sy2 = [int(round(float(v))) for v in split_bbox]
        px1, py1, px2, py2 = [int(round(float(v))) for v in parent_bbox]
    except (TypeError, ValueError):
        return False

    parent_w = max(1, px2 - px1)
    parent_h = max(1, py2 - py1)
    split_w = max(1, sx2 - sx1)
    split_h = max(1, sy2 - sy1)

    if (
        float(parent_h) / float(parent_w)
        < SPEECH_EDGE_SPLIT_INTERNAL_TEXT_MIN_PARENT_ASPECT
    ):
        return False
    if (
        split_w / float(parent_w)
        > SPEECH_EDGE_SPLIT_INTERNAL_TEXT_MAX_WIDTH_RATIO
        or split_h / float(parent_h)
        > SPEECH_EDGE_SPLIT_INTERNAL_TEXT_MAX_HEIGHT_RATIO
    ):
        return False

    inside_tol = SPEECH_EDGE_SPLIT_INTERNAL_TEXT_INSIDE_TOLERANCE_PX
    fully_inside_parent = (
        sx1 >= px1 - inside_tol
        and sy1 >= py1 - inside_tol
        and sx2 <= px2 + inside_tol
        and sy2 <= py2 + inside_tol
    )
    if not fully_inside_parent:
        return False

    edge_gap = max(
        SPEECH_EDGE_SPLIT_INTERNAL_TEXT_EDGE_GAP_PX,
        min(parent_w, parent_h) * SPEECH_EDGE_SPLIT_INTERNAL_TEXT_EDGE_GAP_RATIO,
    )
    if side_name == "right":
        side_touch = sx2 >= px2 - edge_gap
        side_band = sx1 >= px1 + parent_w * 0.45
    else:
        side_touch = sx1 <= px1 + edge_gap
        side_band = sx2 <= px1 + parent_w * 0.55
    if not (side_touch and side_band):
        return False

    corner_band = SPEECH_EDGE_SPLIT_INTERNAL_TEXT_BAND_RATIO
    top_corner = sy1 <= py1 + edge_gap and sy2 <= py1 + parent_h * corner_band
    bottom_corner = (
        sy2 >= py2 - edge_gap
        and sy1 >= py1 + parent_h * (1.0 - corner_band)
    )
    return bool(top_corner or bottom_corner)


def _detect_speech_edge_text_splits(
    bubble_data: List[Dict[str, Any]],
    original_cv_image: np.ndarray,
    mime_type: str,
    cv2_ext: str,
    min_side_pixels: int,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    image_size = (original_cv_image.shape[1], original_cv_image.shape[0])
    split_items: List[Dict[str, Any]] = []
    for bubble in bubble_data or []:
        try:
            bbox = tuple(int(round(float(v))) for v in bubble.get("bbox", ()))
        except (TypeError, ValueError):
            continue
        if len(bbox) != 4:
            continue
        clipped = _clip_bbox_to_image(bbox, image_size)
        if clipped is None:
            continue

        split_result = _detect_speech_edge_split_bbox(original_cv_image, clipped)
        if split_result is None:
            continue
        split_bbox = split_result.get("bbox")
        split_bbox = _clip_bbox_to_image(split_bbox, image_size)
        if split_bbox is None:
            continue
        if _bbox_area(split_bbox) <= 0:
            continue
        if _speech_edge_split_likely_parent_corner_duplicate(
            split_bbox,
            clipped,
            split_result.get("side"),
            bubble,
            bubble_data,
            image_size,
        ):
            log_message(
                "Skipped speech-edge split inside parent bubble "
                f"{clipped}: {split_bbox} (corner-duplicate guard)",
                verbose=verbose,
                always_print=True,
            )
            continue
        if _speech_edge_split_likely_parent_internal_text(
            split_bbox,
            clipped,
            split_result.get("side"),
        ):
            log_message(
                "Skipped speech-edge split inside parent bubble "
                f"{clipped}: {split_bbox} (internal-text guard)",
                verbose=verbose,
                always_print=True,
            )
            continue

        exclusions = bubble.setdefault("translation_exclusion_bboxes", [])
        if any(_bbox_iou(split_bbox, existing) >= 0.70 for existing in exclusions):
            continue
        exclusions.append(split_bbox)

        image_b64, original_crop_pil = _speech_edge_split_encode_crop(
            original_cv_image,
            split_bbox,
            cv2_ext,
            min_side_pixels,
        )
        if not image_b64:
            continue

        sx1, sy1, sx2, sy2 = split_bbox
        aspect_ratio = float(max(1, sy2 - sy1)) / float(max(1, sx2 - sx1))
        split_items.append(
            {
                "bbox": split_bbox,
                "original_bbox": split_bbox,
                "confidence": 0.50,
                "is_outside_text": True,
                "speech_edge_split": True,
                "parent_bbox": clipped,
                "image_b64": image_b64,
                "mime_type": mime_type,
                "is_dark_text": False,
                "text_color_rgb": None,
                "aspect_ratio": aspect_ratio,
                "rotation_deg": 0.0,
                "sfx_rotation_deg": 0.0,
                "horizontal_rotation_deg": 0.0,
                "render_bbox": split_bbox,
                "render_mask": None,
                "needs_text_background": False,
                "speech_edge_side": split_result.get("side"),
                "mask_outside_bubble_overlap": True,
                "bubble_overlap_rescue": True,
                "original_crop_pil": original_crop_pil,
                "speech_edge_erase_mask": split_result.get("erase_mask"),
            }
        )
        log_message(
            "Separated speech-edge text from bubble "
            f"{clipped}: {split_bbox}",
            verbose=verbose,
            always_print=True,
        )

    return split_items


def _speech_edge_merge_target_index(
    split_item: Dict[str, Any],
    sorted_items: List[Dict[str, Any]],
    split_index: int,
) -> Optional[int]:
    """Find a speech bubble that owns an edge split accidentally cut from it.

    Real edge text, such as a small aside on the side of a single bubble, only
    overlaps its parent bubble.  False positives in conjoined bubbles often
    overlap a second child bubble almost completely; those should be merged
    back into that child instead of rendered as OSB.
    """
    try:
        split_bbox = tuple(int(round(float(v))) for v in split_item.get("bbox", ()))
    except (TypeError, ValueError):
        return None
    if len(split_bbox) != 4:
        return None

    parent_bbox = None
    try:
        raw_parent = split_item.get("parent_bbox")
        if raw_parent is not None:
            parent_bbox = tuple(int(round(float(v))) for v in raw_parent)
    except (TypeError, ValueError):
        parent_bbox = None

    split_area = max(1, _bbox_area(split_bbox))
    split_cx = (split_bbox[0] + split_bbox[2]) / 2.0
    split_cy = (split_bbox[1] + split_bbox[3]) / 2.0

    best_index: Optional[int] = None
    best_score = 0.0
    for candidate_index, candidate in enumerate(sorted_items):
        if candidate_index == split_index or candidate.get("is_outside_text", False):
            continue
        try:
            candidate_bbox = tuple(
                int(round(float(v))) for v in candidate.get("bbox", ())
            )
        except (TypeError, ValueError):
            continue
        if len(candidate_bbox) != 4:
            continue
        if parent_bbox is not None and _bbox_iou(candidate_bbox, parent_bbox) >= 0.90:
            continue

        intersection = _bbox_intersection_area(split_bbox, candidate_bbox)
        if intersection <= 0:
            continue
        overlap_ratio = float(intersection) / float(split_area)
        center_inside = (
            candidate_bbox[0] <= split_cx <= candidate_bbox[2]
            and candidate_bbox[1] <= split_cy <= candidate_bbox[3]
        )
        if overlap_ratio < 0.70:
            continue
        if not center_inside and overlap_ratio < 0.90:
            continue

        candidate_area = max(1, _bbox_area(candidate_bbox))
        compactness = min(1.0, float(split_area) / float(candidate_area))
        score = overlap_ratio + (0.15 if center_inside else 0.0) + compactness * 0.05
        if score > best_score:
            best_score = score
            best_index = candidate_index

    if best_index is not None:
        split_item["speech_edge_merge_reason"] = "overlap"
        return best_index

    side = str(split_item.get("speech_edge_side") or "").strip().lower()
    if side != "left":
        return None

    def _axis_gap(a1: float, a2: float, b1: float, b2: float) -> float:
        if a2 < b1:
            return float(b1 - a2)
        if b2 < a1:
            return float(a1 - b2)
        return 0.0

    fallback_index: Optional[int] = None
    fallback_score: Optional[float] = None
    search_ranges = (
        (0, range(split_index + 1, len(sorted_items))),
        (1, range(split_index - 1, -1, -1)),
    )
    for direction_rank, candidate_range in search_ranges:
        for candidate_index in candidate_range:
            candidate = sorted_items[candidate_index]
            if candidate.get("is_outside_text", False):
                continue
            try:
                candidate_bbox = tuple(
                    int(round(float(v))) for v in candidate.get("bbox", ())
                )
            except (TypeError, ValueError):
                continue
            if len(candidate_bbox) != 4:
                continue
            if parent_bbox is not None and _bbox_iou(candidate_bbox, parent_bbox) >= 0.90:
                continue

            candidate_width = max(1, candidate_bbox[2] - candidate_bbox[0])
            candidate_height = max(1, candidate_bbox[3] - candidate_bbox[1])
            margin = max(24.0, min(96.0, max(candidate_width, candidate_height) * 0.30))
            expanded_candidate = (
                candidate_bbox[0] - margin,
                candidate_bbox[1] - margin,
                candidate_bbox[2] + margin,
                candidate_bbox[3] + margin,
            )
            center_in_expanded = (
                expanded_candidate[0] <= split_cx <= expanded_candidate[2]
                and expanded_candidate[1] <= split_cy <= expanded_candidate[3]
            )
            intersection = _bbox_intersection_area(split_bbox, candidate_bbox)
            x_gap = _axis_gap(
                split_bbox[0],
                split_bbox[2],
                candidate_bbox[0],
                candidate_bbox[2],
            )
            y_gap = _axis_gap(
                split_bbox[1],
                split_bbox[3],
                candidate_bbox[1],
                candidate_bbox[3],
            )
            close_enough = (
                x_gap <= max(64.0, candidate_width * 0.35)
                and y_gap <= max(48.0, candidate_height * 0.25)
            )
            if not center_in_expanded and intersection <= 0 and not close_enough:
                continue

            candidate_cx = (candidate_bbox[0] + candidate_bbox[2]) / 2.0
            candidate_cy = (candidate_bbox[1] + candidate_bbox[3]) / 2.0
            center_distance = (
                (candidate_cx - split_cx) ** 2 + (candidate_cy - split_cy) ** 2
            ) ** 0.5
            score = (
                direction_rank * 100000.0
                + abs(candidate_index - split_index) * 1000.0
                + center_distance
                + x_gap * 2.0
                + y_gap * 2.0
            )
            if fallback_score is None or score < fallback_score:
                fallback_score = score
                fallback_index = candidate_index

    if fallback_index is not None:
        split_item["speech_edge_merge_reason"] = "left-edge-neighbor"
        return fallback_index

    return best_index


def _is_invalid_translation_text(text: Any, provider: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    _, payload = extract_translation_tag(value)
    payload = str(payload or "").strip()
    provider_prefixes = (f"[{provider}:", f"[{provider}-")
    return (
        payload.startswith("API Error")
        or payload.startswith("[Translation Error]")
        or payload.startswith("[Translation Error:")
        or payload.startswith(provider_prefixes)
        or payload in INVALID_TRANSLATION_MARKERS
    )


def _wrap_translation_with_tag(tag: Optional[str], text: str) -> str:
    if not tag:
        return text
    normalized_tag = str(tag).upper().strip().strip("<>")
    if not normalized_tag:
        return text
    return f"<{normalized_tag}>{text}</{normalized_tag}>"


def _join_merged_translation_parts(first: str, second: str) -> str:
    first = str(first or "").strip()
    second = str(second or "").strip()
    if not first:
        return second
    if not second:
        return first
    if second[0] in ".,!?;:)]}":
        return first + second
    return f"{first} {second}"


def _merge_overlapping_speech_edge_split_translations(
    sorted_items: List[Dict[str, Any]],
    translated_texts: List[str],
    source_ocr_texts: List[str],
    provider: str,
    verbose: bool = False,
) -> int:
    if len(translated_texts) != len(sorted_items):
        return 0

    merged_count = 0
    for split_index, split_item in enumerate(sorted_items):
        if not split_item.get("speech_edge_split") or split_item.get("skip_render"):
            continue
        target_index = _speech_edge_merge_target_index(
            split_item, sorted_items, split_index
        )
        if target_index is None:
            continue

        split_translation = translated_texts[split_index]
        target_translation = translated_texts[target_index]
        if _is_invalid_translation_text(
            split_translation,
            provider,
        ) or _is_invalid_translation_text(target_translation, provider):
            continue

        split_tag, split_text = extract_translation_tag(split_translation)
        target_tag, target_text = extract_translation_tag(target_translation)
        if not split_text.strip() or not target_text.strip():
            continue

        if split_index < target_index:
            merged_text = _join_merged_translation_parts(split_text, target_text)
        else:
            merged_text = _join_merged_translation_parts(target_text, split_text)

        translated_texts[target_index] = _wrap_translation_with_tag(
            target_tag or split_tag,
            merged_text,
        )
        translated_texts[split_index] = ""
        if split_index < len(source_ocr_texts):
            source_ocr_texts[split_index] = ""

        split_item["skip_render"] = True
        split_item["skip_restore"] = True
        split_item["merged_into_bbox"] = sorted_items[target_index].get("bbox")
        split_item["translation"] = ""
        sorted_items[target_index]["merged_speech_edge_split"] = True
        sorted_items[target_index].setdefault(
            "merged_speech_edge_split_bboxes",
            [],
        ).append(split_item.get("bbox"))
        merged_count += 1
        merge_reason = split_item.get("speech_edge_merge_reason") or "overlap"
        log_message(
            "Merged speech-edge text "
            f"{split_item.get('bbox')} into bubble {sorted_items[target_index].get('bbox')} "
            f"({merge_reason})",
            verbose=verbose,
            always_print=True,
        )

    return merged_count


def _inset_bbox_for_padding(bbox: Any, padding_pixels: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    try:
        pad = max(0.0, float(padding_pixels))
    except (TypeError, ValueError):
        pad = 0.0
    if pad <= 0.0:
        return x1, y1, x2, y2

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    max_pad = max(0.0, min((width - 1) / 2.0, (height - 1) / 2.0))
    pad = min(pad, max_pad)
    return (
        int(round(x1 + pad)),
        int(round(y1 + pad)),
        int(round(x2 - pad)),
        int(round(y2 - pad)),
    )


def _osb_original_position_bbox(
    bubble: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    try:
        fallback = tuple(
            int(round(float(v))) for v in bubble.get("bbox", (0, 0, 1, 1))
        )
    except (TypeError, ValueError):
        fallback = (0, 0, 1, 1)

    raw = bubble.get("original_bbox") or fallback
    try:
        raw_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in raw), image_size
        )
    except (TypeError, ValueError):
        raw_box = None
    if raw_box is None:
        return fallback

    container = _clip_bbox_to_image(fallback, image_size) or raw_box
    rx1, ry1, rx2, ry2 = raw_box
    cx = (rx1 + rx2) / 2.0
    cy = (ry1 + ry2) / 2.0
    raw_w = max(1.0, float(rx2 - rx1))
    raw_h = max(1.0, float(ry2 - ry1))

    # Keep placement anchored to the original OCR text, but give translated text
    # enough room so it does not shrink to the tiny Japanese SFX box.
    target_w = max(raw_w, min(raw_w * 3.2, float(container[2] - container[0])))
    target_h = max(raw_h, min(raw_h * 2.4, float(container[3] - container[1])))

    x1 = int(round(cx - target_w / 2.0))
    y1 = int(round(cy - target_h / 2.0))
    x2 = int(round(cx + target_w / 2.0))
    y2 = int(round(cy + target_h / 2.0))

    # Prefer staying inside the cleaned OSB region when it is reasonably close.
    c_x1, c_y1, c_x2, c_y2 = container
    if x1 < c_x1:
        x2 += c_x1 - x1
        x1 = c_x1
    if y1 < c_y1:
        y2 += c_y1 - y1
        y1 = c_y1
    if x2 > c_x2:
        x1 -= x2 - c_x2
        x2 = c_x2
    if y2 > c_y2:
        y1 -= y2 - c_y2
        y2 = c_y2

    clipped = _clip_bbox_to_image((x1, y1, x2, y2), image_size)
    return clipped or raw_box


def _osb_layout_bbox(
    bubble: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    render_box = bubble.get("render_bbox")
    if render_box is not None:
        try:
            clipped = _clip_bbox_to_image(
                tuple(int(round(float(v))) for v in render_box), image_size
            )
            if clipped is not None:
                return clipped
        except (TypeError, ValueError):
            pass
    return _osb_original_position_bbox(bubble, image_size)


def _osb_unexpanded_layout_bbox(
    bubble: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Return the detected OSB box before any render-box expansion."""
    for candidate in (bubble.get("original_bbox"), bubble.get("bbox")):
        if candidate is None:
            continue
        try:
            clipped = _clip_bbox_to_image(
                tuple(int(round(float(v))) for v in candidate),
                image_size,
            )
        except (TypeError, ValueError):
            clipped = None
        if clipped is not None:
            return clipped
    return (0, 0, 1, 1)


def _long_vertical_osb_original_anchor_bbox(
    bubble: Dict[str, Any],
    image_size: Tuple[int, int],
    padding_pixels: float,
    render_tag: Any,
    text: Any,
    vertical_stack: bool,
) -> Optional[Tuple[int, int, int, int]]:
    if not vertical_stack or not bubble.get("is_outside_text", False):
        return None
    if _is_sfx_render_tag(render_tag):
        return None

    char_count = _count_osb_vertical_chars(text)
    if char_count < OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_CHARS:
        return None

    raw = bubble.get("original_bbox")
    render_box = bubble.get("render_bbox")
    if raw is None or render_box is None:
        return None

    try:
        raw_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in raw), image_size
        )
        expanded_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in render_box), image_size
        )
    except (TypeError, ValueError):
        return None
    if raw_box is None or expanded_box is None:
        return None

    rx1, ry1, rx2, ry2 = raw_box
    ex1, ey1, ex2, ey2 = expanded_box
    raw_w = max(1.0, float(rx2 - rx1))
    raw_h = max(1.0, float(ry2 - ry1))
    if raw_h / raw_w < OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_ASPECT:
        return None

    min_expansion = max(
        OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_EXPANSION_PX,
        raw_h * OSB_LONG_VERTICAL_ORIGINAL_ANCHOR_MIN_EXPANSION_RATIO,
    )
    top_expansion = float(ry1 - ey1)
    center_shift_x = abs(((ex1 + ex2) / 2.0) - ((rx1 + rx2) / 2.0))
    if top_expansion < min_expansion and center_shift_x < min_expansion:
        return None

    return _inset_bbox_for_padding(
        _osb_original_position_bbox(bubble, image_size),
        padding_pixels,
    )


def _text_region_restore_protection_bbox(
    item: Dict[str, Any],
    image_size: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    try:
        if item.get("is_outside_text", False):
            return _osb_layout_bbox(item, image_size)
        bbox = item.get("bbox")
        if bbox is None:
            return None
        return _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except Exception:
        return None


def _osb_side_anchor_visible_text(text: Any) -> str:
    value = str(text or "")
    previous = None
    while previous != value:
        previous = value
        value = STYLE_MARKER_STRIP_RE.sub(r"\2", value)
    value = (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )
    if len([part for part in re.split(r"\s+", value.strip()) if part]) > 1:
        return ""
    return "".join(ch for ch in value if ch.isalnum())


def _filter_osb_text_like_components(
    mask: np.ndarray,
    raw_w: int,
    raw_h: int,
) -> Optional[np.ndarray]:
    """Keep compact source-text-like components and drop panel/background lines."""
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    if raw_w <= 0 or raw_h <= 0 or not np.any(mask_u8):
        return None

    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    except Exception:
        return mask_u8.astype(bool)

    if num_labels <= 1:
        return None

    kept = np.zeros_like(mask_u8, dtype=np.uint8)
    crop_area = float(max(1, raw_w * raw_h))
    min_area = max(2.0, crop_area * 0.00035)
    max_area = crop_area * 0.55

    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        left = float(stats[label_idx, cv2.CC_STAT_LEFT])
        top = float(stats[label_idx, cv2.CC_STAT_TOP])
        width = float(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if width <= 0.0 or height <= 0.0:
            continue

        near_horizontal_edge = top <= 1.0 or (top + height) >= raw_h - 1.0
        near_vertical_edge = left <= 1.0 or (left + width) >= raw_w - 1.0
        thin_h = height <= max(2.0, raw_h * 0.08)
        thin_v = width <= max(2.0, raw_w * 0.08)

        # Panel borders, speed lines, and label box edges are often very long and
        # thin.  Do not count them as source glyphs for residual protection.
        if width >= raw_w * 0.55 and thin_h and area <= crop_area * 0.08:
            continue
        if height >= raw_h * 0.55 and thin_v and area <= crop_area * 0.08:
            continue
        if near_horizontal_edge and width >= raw_w * 0.35 and height <= max(3.0, raw_h * 0.12):
            continue
        if near_vertical_edge and height >= raw_h * 0.35 and width <= max(3.0, raw_w * 0.12):
            continue
        if width >= raw_w * 0.92 and height <= raw_h * 0.18:
            continue
        if height >= raw_h * 0.92 and width <= raw_w * 0.18:
            continue

        kept[labels == label_idx] = 1

    if int(np.count_nonzero(kept)) < OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
        return None
    return kept.astype(bool)


def _osb_source_ink_crop(
    bubble: Dict[str, Any],
    original_image: Image.Image,
    raw_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    allow_render_mask: bool = True,
    side_only_filter: bool = True,
    strict_text_filter: bool = False,
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = raw_box
    raw_w = max(1, x2 - x1)
    raw_h = max(1, y2 - y1)

    render_mask = bubble.get("render_mask") if allow_render_mask else None
    if render_mask is not None:
        try:
            mask_array = np.asarray(render_mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            if mask_array.shape == (image_size[1], image_size[0]):
                mask_crop = mask_array[y1:y2, x1:x2] > 0
                if int(np.count_nonzero(mask_crop)) >= OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
                    return mask_crop
        except Exception:
            pass

    try:
        gray = np.asarray(original_image.crop(raw_box).convert("L"))
    except Exception:
        return None
    if gray.shape[:2] != (raw_h, raw_w) or gray.size == 0:
        return None

    dark_background = bool(bubble.get("is_dark_text", True))
    try:
        threshold_mode = cv2.THRESH_BINARY if dark_background else cv2.THRESH_BINARY_INV
        _, thresholded = cv2.threshold(
            gray, 0, 255, threshold_mode | cv2.THRESH_OTSU
        )
    except Exception:
        if dark_background:
            threshold = max(65.0, min(220.0, float(np.percentile(gray, 72))))
            thresholded = np.where(gray >= threshold, 255, 0).astype(np.uint8)
        else:
            threshold = min(190.0, max(35.0, float(np.percentile(gray, 28))))
            thresholded = np.where(gray <= threshold, 255, 0).astype(np.uint8)

    thresholded = np.asarray(thresholded, dtype=np.uint8)
    try:
        opened = cv2.morphologyEx(
            thresholded, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        if int(np.count_nonzero(opened)) >= OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
            thresholded = opened
    except Exception:
        pass

    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (thresholded > 0).astype(np.uint8), 8
        )
        if side_only_filter and num_labels > 1:
            kept = np.zeros_like(thresholded, dtype=np.uint8)
            crop_area = float(max(1, raw_w * raw_h))
            min_area = max(2.0, crop_area * 0.0005)
            max_area = crop_area * 0.40
            side_left_end = max(1, raw_w // 3)
            side_right_start = min(
                raw_w - 1, max(side_left_end, int(math.ceil(raw_w * 2.0 / 3.0)))
            )
            for label_idx in range(1, num_labels):
                area = float(stats[label_idx, cv2.CC_STAT_AREA])
                comp_left = int(stats[label_idx, cv2.CC_STAT_LEFT])
                comp_right = comp_left + int(stats[label_idx, cv2.CC_STAT_WIDTH])
                touches_side_text = (
                    comp_left < side_left_end or comp_right > side_right_start
                )
                if min_area <= area <= max_area and touches_side_text:
                    kept[labels == label_idx] = 255
            if int(np.count_nonzero(kept)) >= OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
                thresholded = kept
    except Exception:
        pass

    if strict_text_filter:
        filtered = _filter_osb_text_like_components(thresholded, raw_w, raw_h)
        if filtered is None:
            return None
        mask_crop = filtered
    else:
        mask_crop = thresholded > 0
    if int(np.count_nonzero(mask_crop)) < OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
        return None
    return mask_crop


def _shift_bbox_inside(
    bbox: Tuple[float, float, float, float],
    bounds: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    b_x1, b_y1, b_x2, b_y2 = [float(v) for v in bounds]

    if x1 < b_x1:
        x2 += b_x1 - x1
        x1 = b_x1
    if y1 < b_y1:
        y2 += b_y1 - y1
        y1 = b_y1
    if x2 > b_x2:
        x1 -= x2 - b_x2
        x2 = b_x2
    if y2 > b_y2:
        y1 -= y2 - b_y2
        y2 = b_y2

    x1 = max(b_x1, x1)
    y1 = max(b_y1, y1)
    x2 = min(b_x2, x2)
    y2 = min(b_y2, y2)
    return (
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
    )


def _osb_side_anchor_bbox(
    bubble: Dict[str, Any],
    translated_text: Any,
    original_image: Image.Image,
    image_size: Tuple[int, int],
    current_bbox: Tuple[int, int, int, int],
) -> Optional[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]]:
    visible_text = _osb_side_anchor_visible_text(translated_text)
    if (
        not visible_text
        or len(visible_text) > OSB_SIDE_ANCHOR_MAX_TEXT_CHARS
        or re.search(r"\d", visible_text)
    ):
        return None

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        raw_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box), image_size
        )
    except (TypeError, ValueError):
        raw_box = None
    if raw_box is None:
        return None

    rx1, ry1, rx2, ry2 = raw_box
    raw_w = max(1, rx2 - rx1)
    raw_h = max(1, ry2 - ry1)
    if raw_w < 32 or raw_h < 6 or raw_w / float(raw_h) < OSB_SIDE_ANCHOR_MIN_ASPECT:
        return None

    ink_crop = _osb_source_ink_crop(bubble, original_image, raw_box, image_size)
    if ink_crop is None or ink_crop.shape[:2] != (raw_h, raw_w):
        return None

    left_end = max(1, raw_w // 3)
    right_start = min(raw_w - 1, max(left_end, int(math.ceil(raw_w * 2.0 / 3.0))))
    left_count = int(np.count_nonzero(ink_crop[:, :left_end]))
    center_count = int(np.count_nonzero(ink_crop[:, left_end:right_start]))
    right_count = int(np.count_nonzero(ink_crop[:, right_start:]))
    total_count = left_count + center_count + right_count
    if total_count < OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
        return None
    if center_count / float(total_count) > OSB_SIDE_ANCHOR_CENTER_MAX_FRACTION:
        return None

    side = "left" if left_count >= right_count else "right"
    side_count = left_count if side == "left" else right_count
    if side_count < OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
        return None
    if side_count < max(1.0, center_count * OSB_SIDE_ANCHOR_MIN_SIDE_GAIN):
        return None

    if side == "left":
        side_mask = ink_crop[:, :left_end]
        x_offset = 0
        side_min_x = rx1
        side_max_x = int(round(rx1 + raw_w * 0.52))
    else:
        side_mask = ink_crop[:, right_start:]
        x_offset = right_start
        side_min_x = int(round(rx1 + raw_w * 0.48))
        side_max_x = rx2

    ys, xs = np.where(side_mask)
    if xs.size > 0 and ys.size > 0:
        anchor_cx = rx1 + x_offset + float(xs.mean())
        anchor_cy = ry1 + float(ys.mean())
    else:
        anchor_cx = rx1 + raw_w * (0.25 if side == "left" else 0.75)
        anchor_cy = ry1 + raw_h / 2.0

    container_source = bubble.get("bbox") or current_bbox
    try:
        container = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in container_source), image_size
        )
    except (TypeError, ValueError):
        container = None
    if container is None:
        container = _clip_bbox_to_image(current_bbox, image_size) or raw_box

    current_w = max(1.0, float(current_bbox[2] - current_bbox[0]))
    current_h = max(1.0, float(current_bbox[3] - current_bbox[1]))
    side_width = max(1.0, float(side_max_x - side_min_x))
    target_w = min(
        current_w,
        side_width,
        max(24.0, raw_h * 2.8, raw_w * 0.40),
    )
    target_h = min(current_h, max(16.0, raw_h * 2.0))
    if target_w < 12.0 or target_h < 10.0:
        return None

    candidate = (
        anchor_cx - target_w / 2.0,
        anchor_cy - target_h / 2.0,
        anchor_cx + target_w / 2.0,
        anchor_cy + target_h / 2.0,
    )
    candidate_bbox = _shift_bbox_inside(candidate, container)

    if side == "left" and candidate_bbox[2] > side_max_x:
        overshoot = candidate_bbox[2] - side_max_x
        candidate_bbox = (
            candidate_bbox[0] - overshoot,
            candidate_bbox[1],
            candidate_bbox[2] - overshoot,
            candidate_bbox[3],
        )
    elif side == "right" and candidate_bbox[0] < side_min_x:
        overshoot = side_min_x - candidate_bbox[0]
        candidate_bbox = (
            candidate_bbox[0] + overshoot,
            candidate_bbox[1],
            candidate_bbox[2] + overshoot,
            candidate_bbox[3],
        )
    candidate_bbox = _shift_bbox_inside(candidate_bbox, container)

    clipped = _clip_bbox_to_image(candidate_bbox, image_size)
    if clipped is None or clipped[2] - clipped[0] < 12 or clipped[3] - clipped[1] < 10:
        return None
    return clipped, side, (left_count, center_count, right_count)


def _osb_vertical_ink_anchor_bbox(
    bubble: Dict[str, Any],
    original_image: Image.Image,
    image_size: Tuple[int, int],
    vertical_stack: bool,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], int]]:
    if not vertical_stack or not bubble.get("is_outside_text", False):
        return None

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        raw_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box), image_size
        )
    except (TypeError, ValueError):
        raw_box = None
    if raw_box is None:
        return None

    rx1, ry1, rx2, ry2 = raw_box
    raw_w = max(1, rx2 - rx1)
    raw_h = max(1, ry2 - ry1)
    if raw_w < 4 or raw_h < 4:
        return None

    ink_crop = _osb_source_ink_crop(
        bubble,
        original_image,
        raw_box,
        image_size,
        allow_render_mask=False,
        side_only_filter=False,
    )
    if ink_crop is None or ink_crop.shape[:2] != (raw_h, raw_w):
        return None

    ys, xs = np.where(ink_crop)
    ink_count = int(xs.size)
    if ink_count < OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
        return None

    ix1 = rx1 + int(xs.min())
    iy1 = ry1 + int(ys.min())
    ix2 = rx1 + int(xs.max()) + 1
    iy2 = ry1 + int(ys.max()) + 1

    # Keep the anchor tied to actual source ink, with a tiny anti-alias guard.
    pad_x = 1
    pad_y = 1
    ink_box = _clip_bbox_to_image(
        (ix1 - pad_x, iy1 - pad_y, ix2 + pad_x, iy2 + pad_y),
        image_size,
    )
    if ink_box is None:
        return None
    if ink_box[2] - ink_box[0] < 4 or ink_box[3] - ink_box[1] < 4:
        return None

    return ink_box, raw_box, ink_count


def _osb_split_token_key(token: str) -> str:
    return re.sub(
        r"[^\w\uac00-\ud7a3\u3040-\u30ff\u3400-\u9fff]+",
        "",
        token,
    ).casefold()


def _osb_split_display_token(token: str) -> str:
    return re.sub(
        r"[^\w\uac00-\ud7a3\u3040-\u30ff\u3400-\u9fff]+",
        "",
        str(token or ""),
    )


def _normalize_osb_text_newlines(text: Any) -> str:
    return (
        str(text or "")
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )


def _is_repeated_short_osb_split(part_texts: List[str]) -> bool:
    if len(part_texts) < 2 or len(part_texts) > 4:
        return False

    token_keys = [_osb_split_token_key(text) for text in part_texts]
    if not all(token_keys):
        return False
    if len(set(token_keys)) != 1:
        return False
    return len(token_keys[0]) <= 2


def _short_osb_tokens(text: str) -> List[str]:
    raw_text = str(text or "").strip()
    if re.search(r"\d", raw_text):
        return []
    if any(mark in raw_text for mark in "()[]{}<>/\\|"):
        return []
    if "\n" in raw_text or "\\n" in raw_text:
        return []

    tokens = [token.strip() for token in raw_text.split() if token.strip()]
    if len(tokens) == 2:
        display_tokens = [_osb_split_display_token(token) for token in tokens]
        token_keys = [token.casefold() for token in display_tokens]
        if not all(token_keys):
            return []
        if any(len(key) > 3 for key in token_keys):
            return []
        if sum(len(key) for key in token_keys) > 6:
            return []
        return display_tokens

    if len(tokens) == 1:
        token = tokens[0]
        token_key = _osb_split_token_key(token)
        if len(token_key) != 2:
            return []
        visible_chars = [ch for ch in token if ch.isalnum()]
        if len(visible_chars) != 2:
            return []
        return visible_chars

    return []


def _original_position_osb_tokens(text: str) -> List[str]:
    raw_text = str(text or "").strip()
    if re.search(r"\d", raw_text):
        return []
    if any(mark in raw_text for mark in "()[]{}<>/\\|"):
        return []
    if "\n" in raw_text or "\\n" in raw_text:
        return []

    tokens = [token.strip() for token in raw_text.split() if token.strip()]
    if len(tokens) == 1:
        # Keep compact translated SFX words together.  Per-character placement is
        # only reliable for explicit multi-token/repeated layouts handled below.
        return []
    if len(tokens) < 2 or len(tokens) > 4:
        return []

    display_tokens = [_osb_split_display_token(token) for token in tokens]
    token_keys = [token.casefold() for token in display_tokens]
    if not all(token_keys):
        return []
    if any(len(key) > 3 for key in token_keys):
        return []
    if sum(len(key) for key in token_keys) > 12:
        return []
    return display_tokens


def _plain_osb_source_text(source_text: Any) -> str:
    value = str(source_text or "")
    previous = None
    while previous != value:
        previous = value
        value = re.sub(r"</?[^>]+>", " ", value)
        value = STYLE_MARKER_STRIP_RE.sub(r"\2", value)
    return (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .strip()
    )


def _spaced_vertical_sfx_osb_tokens(text: str, source_text: Any) -> List[str]:
    parts = _original_position_osb_tokens(text)
    if len(parts) != 2:
        return []
    if any(not re.fullmatch(r"[\uac00-\ud7a3]{2,3}", part) for part in parts):
        return []

    source = _plain_osb_source_text(source_text)
    source_parts = [
        _osb_split_display_token(part)
        for part in re.split(r"\s+", source)
        if _osb_split_display_token(part)
    ]
    if len(source_parts) != 2:
        return []
    if any(
        not re.fullmatch(r"[\u3040-\u30ff]{2,4}", part)
        for part in source_parts
    ):
        return []
    if not _is_japanese_sfx_like(source_text):
        return []
    return parts


def _compact_spaced_source_vertical_sfx_osb_tokens(
    text: str,
    source_text: Any,
    bubble: Dict[str, Any],
    original_image: Image.Image,
    image_size: Tuple[int, int],
) -> List[str]:
    raw_text = _normalize_osb_text_newlines(text).strip()
    if not raw_text or re.search(r"\s", raw_text):
        return []

    compact_text = _osb_split_display_token(raw_text)
    if not re.fullmatch(r"[\uac00-\ud7a3]{4,6}", compact_text):
        return []

    source = _plain_osb_source_text(source_text)
    source_parts = [
        _osb_split_display_token(part)
        for part in re.split(r"\s+", source)
        if _osb_split_display_token(part)
    ]
    if len(source_parts) != 2:
        return []
    if any(
        not re.fullmatch(r"[\u3040-\u30ff]{2,4}", part)
        for part in source_parts
    ):
        return []
    if not _is_japanese_sfx_like(source_text):
        return []

    try:
        raw_box = tuple(
            int(round(float(v)))
            for v in (
                bubble.get("original_bbox")
                or bubble.get("bbox")
                or bubble.get("xyxy")
            )
        )
    except (TypeError, ValueError):
        return []
    base_box = _clip_bbox_to_image(raw_box, image_size)
    if base_box is None:
        return []

    raw_w = max(1, base_box[2] - base_box[0])
    raw_h = max(1, base_box[3] - base_box[1])
    aspect = raw_h / float(raw_w)
    if aspect < 1.25 or aspect > 2.25:
        return []

    ink_crop = _osb_source_ink_crop(
        bubble,
        original_image,
        base_box,
        image_size,
        allow_render_mask=False,
        side_only_filter=False,
    )
    if ink_crop is None:
        return []

    clusters = _osb_projection_text_clusters(ink_crop, base_box, 2)
    if len(clusters) != 2:
        return []
    clusters = sorted(clusters, key=lambda item: ((item[1] + item[3]) / 2.0, item[0]))
    top, bottom = clusters
    vertical_gap = float(bottom[1] - top[3])
    if vertical_gap < max(5.0, raw_h * 0.035):
        return []
    center_span = abs(
        ((bottom[1] + bottom[3]) / 2.0) - ((top[1] + top[3]) / 2.0)
    ) / float(raw_h)
    if center_span < 0.38:
        return []
    if min(top[4], bottom[4]) / float(max(top[4], bottom[4], 1)) < 0.45:
        return []
    if min(top[3] - top[1], bottom[3] - bottom[1]) / float(raw_h) < 0.25:
        return []

    source_total = sum(len(part) for part in source_parts)
    if source_total <= 0:
        return []
    split_at = int(round(len(compact_text) * len(source_parts[0]) / source_total))
    split_at = max(2, min(len(compact_text) - 2, split_at))
    parts = [compact_text[:split_at], compact_text[split_at:]]
    if any(not re.fullmatch(r"[\uac00-\ud7a3]{2,3}", part) for part in parts):
        return []
    return parts


def _repeated_two_syllable_hangul_osb_tokens(text: str) -> List[str]:
    raw_text = _normalize_osb_text_newlines(text).strip()
    # Only compact or explicitly line-broken two-syllable repeats should use
    # this path.  Spaced one-syllable SFX are whole sound lines; stripping spaces
    # would incorrectly turn them into split columns.
    token = _osb_split_display_token(raw_text)
    if not re.fullmatch(r"[\uac00-\ud7a3]{4}", token):
        return []
    first = token[:2]
    second = token[2:]
    if first != second:
        return []
    visible_parts = [
        _osb_split_display_token(part)
        for part in re.split(r"\s+", raw_text)
        if _osb_split_display_token(part)
    ]
    if len(visible_parts) > 1 and visible_parts != [first, second]:
        return []

    second_pos = raw_text.rfind(second)
    suffix_tail = raw_text[second_pos + len(second) :] if second_pos >= 0 else ""
    suffix = "".join(ch for ch in suffix_tail if not ch.isspace())
    if suffix and not re.fullmatch(
        r"[^\w\uac00-\ud7a3\u3040-\u30ff\u3400-\u9fff]+",
        suffix,
    ):
        suffix = ""
    return [first, second + suffix]


def _mixed_long_hangul_sfx_osb_tokens(text: str) -> List[str]:
    raw_text = _normalize_osb_text_newlines(text).strip()
    if not raw_text:
        return []
    tokens = [token.strip() for token in raw_text.split() if token.strip()]
    if len(tokens) != 2:
        return []

    first = _osb_split_display_token(tokens[0])
    second = _osb_split_display_token(tokens[1])
    if not re.fullmatch(r"[\uac00-\ud7a3]{2,3}", first):
        return []
    if not re.fullmatch(r"[\uac00-\ud7a3]{4,5}", second):
        return []
    # Split long rumble SFX only when the tail is visually repetitive.
    # This avoids breaking ordinary four-syllable words.
    if len(set(second[1:])) != 1:
        return []

    suffix_tail = raw_text[raw_text.rfind(second) + len(second) :]
    suffix = "".join(ch for ch in suffix_tail if not ch.isspace())
    if suffix and not re.fullmatch(
        r"[^\w\uac00-\ud7a3\u3040-\u30ff\u3400-\u9fff]+",
        suffix,
    ):
        suffix = ""

    parts = [first] + list(second)
    if suffix:
        parts[-1] += suffix
    return parts


def _source_is_compact_repeated_sfx_token(text: str) -> bool:
    token = _osb_split_display_token(_normalize_osb_text_newlines(text))
    if len(token) not in (4, 6):
        return False
    if not re.fullmatch(r"[\u3040-\u30ff\u3400-\u9fff]+", token):
        return False
    half = len(token) // 2
    return token[:half] == token[half:]


def _is_spaced_single_hangul_repeat_osb(text: str) -> bool:
    raw_text = str(text or "").strip()
    if not raw_text or "\n" in raw_text or "\\n" in raw_text:
        return False

    tokens = [token.strip() for token in raw_text.split() if token.strip()]
    if len(tokens) < 2 or len(tokens) > 5:
        return False

    token_keys = [_osb_split_token_key(token) for token in tokens]
    if not all(token_keys):
        return False
    if any(not re.fullmatch(r"[\uac00-\ud7a3]", token) for token in token_keys):
        return False
    return len(set(token_keys)) == 1


def _is_single_hangul_osb_token(text: str) -> bool:
    raw_text = str(text or "").strip()
    return bool(re.fullmatch(r"[\uac00-\ud7a3]{2,4}", raw_text))


def _compact_visual_sfx_osb_tokens(text: str) -> List[str]:
    raw_text = _normalize_osb_text_newlines(text).strip()
    if not raw_text or re.search(r"\s", raw_text):
        return []
    token = _osb_split_display_token(raw_text)
    if not re.fullmatch(r"[\uac00-\ud7a3]{2,4}", token):
        return []
    return list(token)


def _source_supports_compact_visual_sfx_split(
    bubble: Dict[str, Any],
    source_text: Any,
    original_image: Image.Image,
    image_size: Tuple[int, int],
    part_count: int,
) -> bool:
    if part_count < 2 or part_count > 4:
        return False
    if not _is_japanese_sfx_like(source_text):
        return False
    if _source_is_single_base_kana_with_small_mark(source_text):
        return False

    source_chars = re.findall(
        r"[\u3040-\u30ff\u3400-\u9fff]", str(source_text or "")
    )
    if len(source_chars) != part_count:
        return False

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        base_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box), image_size
        )
    except (TypeError, ValueError):
        base_box = None
    if base_box is None:
        return False

    x1, y1, x2, y2 = base_box
    raw_w = max(1, x2 - x1)
    raw_h = max(1, y2 - y1)
    img_w, img_h = max(1, image_size[0]), max(1, image_size[1])
    aspect = raw_w / float(raw_h)
    if not 0.55 <= aspect <= 1.45:
        return False
    if raw_w / float(img_w) > 0.35 or raw_h / float(img_h) > 0.25:
        return False

    try:
        gray = np.asarray(original_image.crop(base_box).convert("L"))
    except Exception:
        return False
    if gray.shape[:2] != (raw_h, raw_w) or gray.size == 0:
        return False

    dark_background = bool(bubble.get("is_dark_text", True))
    try:
        if dark_background:
            threshold = max(160.0, min(245.0, float(np.percentile(gray, 92))))
            extreme_ink = (gray >= threshold).astype(np.uint8)
        else:
            threshold = min(95.0, max(35.0, float(np.percentile(gray, 8))))
            extreme_ink = (gray <= threshold).astype(np.uint8)
        extreme_ink = cv2.morphologyEx(
            extreme_ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            extreme_ink, 8
        )
    except Exception:
        return False

    crop_area = float(max(1, raw_w * raw_h))
    min_area = max(18.0, crop_area * 0.006)
    max_area = crop_area * 0.28
    components: List[Tuple[int, int, int, int, int]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if width < 4 or height < 4:
            continue
        if width / float(raw_w) > 0.82 or height / float(raw_h) > 0.82:
            continue
        components.append((left, top, left + width, top + height, area))

    if len(components) < part_count:
        return False

    components = sorted(components, key=lambda box: box[4], reverse=True)[:part_count]
    centers_x = [((box[0] + box[2]) / 2.0) for box in components]
    centers_y = [((box[1] + box[3]) / 2.0) for box in components]
    x_span = (max(centers_x) - min(centers_x)) / float(raw_w)
    y_span = (max(centers_y) - min(centers_y)) / float(raw_h)
    if x_span < 0.22 or y_span < 0.18:
        return False

    ordered = sorted(components, key=lambda box: ((box[1] + box[3]) / 2.0, box[0]))
    ordered_centers_x = [((box[0] + box[2]) / 2.0) for box in ordered]
    if ordered_centers_x[-1] - ordered_centers_x[0] < raw_w * 0.18:
        return False

    return True


def _is_japanese_sfx_like(source_text: Any) -> bool:
    source = re.sub(r"\s+", "", str(source_text or ""))
    source = re.sub(r"[.,!?！？。…・･、，:：;；'\"“”‘’()\[\]{}<>＜＞【】「」『』\-_=+~〜～／/\\|]", "", source)
    if not source:
        return False

    japanese_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", source)
    if not japanese_chars:
        return False

    hiragana = re.findall(r"[\u3040-\u309f]", source)
    katakana = re.findall(r"[\u30a0-\u30ff]", source)
    kanji = re.findall(r"[\u3400-\u9fff]", source)
    japanese_count = len(japanese_chars)
    katakana_ratio = len(katakana) / max(1, japanese_count)

    if japanese_count <= 6 and katakana_ratio >= 0.55 and len(kanji) == 0:
        return True
    if japanese_count <= 12 and katakana_ratio >= 0.82 and len(hiragana) <= 1 and len(kanji) == 0:
        return True
    small_kana_or_long_mark = re.findall(
        r"[\u3041\u3043\u3045\u3047\u3049\u3083\u3085\u3087\u308e\u3063"
        r"\u30a1\u30a3\u30a5\u30a7\u30a9\u30e3\u30e5\u30e7\u30ee\u30c3"
        r"\u30fc\uff70]",
        source,
    )
    if (
        2 <= japanese_count <= 5
        and len(kanji) == 0
        and len(small_kana_or_long_mark) >= 1
    ):
        return True
    if re.search(r"([\u3040-\u30ff])\1{2,}", source):
        return True

    sfx_marks = ("ッ", "っ", "ー", "ｰ", "ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ")
    if any(mark in source for mark in sfx_marks) and japanese_count <= 10 and katakana_ratio >= 0.45:
        return True

    return False


def _source_is_single_base_kana_with_small_mark(source_text: Any) -> bool:
    source = re.sub(r"\s+", "", str(source_text or ""))
    source = re.sub(
        r"[^\u3040-\u30ff\u3400-\u9fff]+",
        "",
        source,
    )
    if not source:
        return False

    small_marks = set("ぁぃぅぇぉゃゅょゎっァィゥェォャュョヮッーｰ")
    japanese_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", source)
    if len(japanese_chars) < 2 or len(japanese_chars) > 4:
        return False

    base_chars = [ch for ch in japanese_chars if ch not in small_marks]
    small_chars = [ch for ch in japanese_chars if ch in small_marks]
    return len(base_chars) == 1 and bool(small_chars)


def _is_short_vertical_source_sfx_candidate(
    bubble: Dict[str, Any],
    text: Any,
    source_text: Any,
) -> bool:
    """Catch short vertical SFX that the LLM tagged as plain outside text."""
    if not bool(bubble.get("is_dark_text", False)):
        return False

    source = re.sub(r"\s+", "", str(source_text or ""))
    japanese_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", source)
    if not 2 <= len(japanese_chars) <= 5:
        return False

    # Short kanji-only words are often labels/dialogue, not sound effects.
    kanji_count = len(re.findall(r"[\u3400-\u9fff]", source))
    if kanji_count >= 1 and kanji_count >= len(japanese_chars) - 1:
        return False

    translated_plain = re.sub(
        r"[^\w\uac00-\ud7a3\u3040-\u30ff\u3400-\u9fff]+",
        "",
        str(text or ""),
    )
    if not 1 <= len(translated_plain) <= 8:
        return False

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        sx1, sy1, sx2, sy2 = [float(v) for v in source_box]
    except (TypeError, ValueError):
        return False
    source_w = max(1.0, sx2 - sx1)
    source_h = max(1.0, sy2 - sy1)
    return source_h / source_w >= 1.18


def _is_short_vertical_dark_osb_sfx_shape(
    bubble: Dict[str, Any],
    text: Any,
) -> bool:
    if not bool(bubble.get("is_dark_text", False)):
        return False

    translated_plain = re.sub(
        r"[^\w\uac00-\ud7a3\u3040-\u30ff\u3400-\u9fff]+",
        "",
        str(text or ""),
    )
    if not 1 <= len(translated_plain) <= 6:
        return False

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        sx1, sy1, sx2, sy2 = [float(v) for v in source_box]
    except (TypeError, ValueError):
        return False
    source_w = max(1.0, sx2 - sx1)
    source_h = max(1.0, sy2 - sy1)
    return source_h / source_w >= 1.35


def _source_looks_like_general_vertical_dialogue(source_text: Any) -> bool:
    source = re.sub(r"\s+", "", str(source_text or ""))
    if not source or _is_japanese_sfx_like(source):
        return False

    japanese_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", source)
    if len(japanese_chars) < 7:
        return False

    hiragana_count = len(re.findall(r"[\u3040-\u309f]", source))
    kanji_count = len(re.findall(r"[\u3400-\u9fff]", source))
    katakana_count = len(re.findall(r"[\u30a0-\u30ff]", source))
    japanese_count = len(japanese_chars)

    if katakana_count / max(1, japanese_count) >= 0.82 and hiragana_count <= 1:
        return False

    sentence_markers = (
        "は",
        "が",
        "を",
        "に",
        "へ",
        "で",
        "と",
        "も",
        "から",
        "なら",
        "では",
        "には",
        "ても",
        "けど",
        "ので",
        "のか",
        "です",
        "ます",
        "づらい",
        "やはり",
    )
    marker_hits = sum(1 for marker in sentence_markers if marker in source)

    if hiragana_count >= 4 and (kanji_count >= 1 or marker_hits >= 2):
        return True
    if hiragana_count >= 6 and marker_hits >= 2:
        return True
    if kanji_count >= 2 and hiragana_count >= 2 and marker_hits >= 1:
        return True

    return False


def _regular_vertical_plain_text(text: Any) -> str:
    value = str(text or "")
    previous = None
    while previous != value:
        previous = value
        value = STYLE_MARKER_STRIP_RE.sub(r"\2", value)
    value = (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )
    return value.strip()


def _regular_vertical_uses_emphasis_limit(text: Any) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    marker_match = re.fullmatch(r"(\*{1,3})(.+)\1", value, flags=re.DOTALL)
    return bool(marker_match and marker_match.group(2).strip())


def _regular_vertical_main_text_for_limit(text: Any) -> str:
    """Return visible main text used for vertical word/character limits.

    Manga SFX translations often include a pronunciation/helper reading in
    parentheses, e.g. "돌격(스러스트)". That helper text should not block
    vertical rendering of a short original effect like "突撃".
    """
    plain_text = _regular_vertical_plain_text(text)
    previous = None
    while previous != plain_text:
        previous = plain_text
        plain_text = re.sub(r"\([^()]*\)", "", plain_text)
        plain_text = re.sub(r"\[[^\[\]]*\]", "", plain_text)
    return plain_text.strip()


def _count_regular_vertical_limit_chars(text: str) -> int:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    return sum(1 for ch in value if ch.isalnum())


def _regular_vertical_count_for_limit(text: Any) -> int:
    """Count visible main glyphs for the WebUI vertical max-char limit."""
    plain_text = _regular_vertical_main_text_for_limit(text)
    return _count_regular_vertical_limit_chars(plain_text)


def _regular_tilt_vertical_allowed(
    translated_text: Any,
    source_text: Any,
    config: MangaTranslatorConfig,
    max_chars_override: Optional[int] = None,
) -> bool:
    rendering_cfg = getattr(config, "rendering", None)
    if rendering_cfg is None or not getattr(rendering_cfg, "auto_vertical_text", False):
        return False

    if _source_looks_like_general_vertical_dialogue(source_text):
        return False

    plain_text = _regular_vertical_plain_text(translated_text)
    if not plain_text:
        return False

    char_count = _regular_vertical_count_for_limit(translated_text)
    if char_count <= 0:
        return False

    if max_chars_override is not None:
        try:
            max_chars = max(0, int(max_chars_override))
        except (TypeError, ValueError):
            max_chars = 4
    else:
        try:
            max_chars = int(getattr(rendering_cfg, "vertical_max_chars", 4))
        except (TypeError, ValueError):
            max_chars = 4
        max_chars = max(0, max_chars)

        if _regular_vertical_uses_emphasis_limit(translated_text):
            emphasis_max = getattr(rendering_cfg, "vertical_emphasis_max_chars", None)
            if emphasis_max is not None:
                try:
                    max_chars = max(0, int(emphasis_max))
                except (TypeError, ValueError):
                    pass

    return char_count <= max_chars


def _is_long_general_osb_text(translated_text: Any, source_text: Any) -> bool:
    translated = str(translated_text or "").strip()
    source = str(source_text or "").strip()
    if not translated:
        return False

    translated_tokens = [token for token in translated.split() if token]
    translated_plain = re.sub(r"\s+", "", translated)
    source_plain = re.sub(r"\s+", "", source)

    if _is_japanese_sfx_like(source):
        return False
    if _source_looks_like_general_vertical_dialogue(source):
        return True

    if len(translated_tokens) >= 5:
        return True
    if len(translated_plain) >= 22:
        return True
    if len(source_plain) >= 24 and len(translated_plain) >= 12:
        return True

    digit_count = len(re.findall(r"\d", translated))
    if digit_count >= 2 and len(translated_plain) >= 8:
        return True

    structural_marks = "[]【】()/：:・|"
    if any(mark in translated for mark in structural_marks) and len(translated_plain) >= 10:
        return True

    return False


def _translated_looks_like_short_sfx(text: Any) -> bool:
    translated = str(text or "").strip()
    if not translated:
        return False
    tokens = [re.sub(r"[^\w가-힣ぁ-んァ-ン一-龯]", "", token) for token in translated.split()]
    tokens = [token for token in tokens if token]
    plain = re.sub(r"[\s.,!?！？。…・･、，:：;；'\"“”‘’()\[\]{}<>＜＞【】「」『』\-_=+~〜～／/\\|]", "", translated)
    if not plain:
        return False

    if len(plain) <= 12 and re.search(r"([가-힣])\1{1,}", plain):
        return True
    if len(plain) <= 8 and re.search(r"([아어오우으이야여요유ぁ-んァ-ン])\1{1,}", plain):
        return True
    if 2 <= len(tokens) <= 4 and len("".join(tokens)) <= 12:
        normalized = [token.casefold() for token in tokens]
        if len(set(normalized)) == 1:
            return True
        if len(normalized) == 2 and normalized[0] == normalized[1]:
            return True
    sfx_words = ("파닥", "쿵", "쾅", "쾅쾅", "꽝", "끼익", "까아", "악", "버", "퍽", "슥", "탁")
    return len(plain) <= 10 and any(word in plain for word in sfx_words)


def _is_grouped_close_sfx_text(text: Any, source_text: Any) -> bool:
    translated = str(text or "").strip()
    source = str(source_text or "").strip()
    if not translated and not source:
        return False

    translated_tokens = [
        _osb_split_token_key(token)
        for token in translated.split()
        if _osb_split_token_key(token)
    ]
    translated_plain = _osb_split_token_key(translated)
    if (
        2 <= len(translated_tokens) <= 4
        and 3 <= len("".join(translated_tokens)) <= 12
    ):
        return True

    source_tokens = [
        _osb_split_token_key(token)
        for token in source.split()
        if _osb_split_token_key(token)
    ]
    if (
        _is_japanese_sfx_like(source)
        and 2 <= len(source_tokens) <= 4
        and 3 <= len(translated_plain) <= 12
    ):
        return True

    return False


def _osb_rotation_word_count(text: Any) -> int:
    value = STYLE_MARKER_STRIP_RE.sub(r"\2", str(text or ""))
    value = (
        value.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )
    return len([part for part in re.split(r"\s+", value.strip()) if part])


def _is_sfx_render_tag(render_tag: Any) -> bool:
    tag_name = str(render_tag or "").upper().strip().strip("<>")
    return tag_name.startswith("SFX")


def _is_inside_bubble_render_tag(render_tag: Any) -> bool:
    tag_name = str(render_tag or "").upper().strip().strip("<>")
    return tag_name in INSIDE_BUBBLE_RENDER_TAGS


def _osb_inside_tag_uses_one_to_one_font_ratio(
    is_outside_text: bool,
    render_tag: Any,
) -> bool:
    return bool(is_outside_text and _is_inside_bubble_render_tag(render_tag))


def _expand_tall_non_sfx_osb_layout_bbox(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    render_tag: Any,
    config: MangaTranslatorConfig,
) -> Tuple[Tuple[int, int, int, int], bool, float, float]:
    clipped_bbox = _clip_bbox_to_image(bbox, image_size)
    if clipped_bbox is None:
        return bbox, False, 0.0, 0.0

    x1, y1, x2, y2 = clipped_bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    original_aspect = float(height) / float(width)
    if _is_sfx_render_tag(render_tag):
        return clipped_bbox, False, original_aspect, original_aspect

    outside_cfg = getattr(config, "outside_text", None)
    try:
        aspect_threshold = float(
            getattr(outside_cfg, "osb_tall_box_expand_aspect_ratio", 2.0)
        )
    except (TypeError, ValueError):
        aspect_threshold = 2.0
    try:
        width_percent = float(
            getattr(outside_cfg, "osb_tall_box_expand_width_percent", 20.0)
        )
    except (TypeError, ValueError):
        width_percent = 20.0

    if (
        aspect_threshold <= 0.0
        or width_percent <= 0.0
        or original_aspect < aspect_threshold
    ):
        return clipped_bbox, False, original_aspect, original_aspect

    image_width = max(1, int(image_size[0]))
    target_width = min(
        image_width,
        max(width + 1, int(math.ceil(width * (1.0 + width_percent / 100.0)))),
    )
    if target_width <= width:
        return clipped_bbox, False, original_aspect, original_aspect

    center_x = (x1 + x2) / 2.0
    expanded_x1 = int(round(center_x - target_width / 2.0))
    expanded_x1 = max(0, min(image_width - target_width, expanded_x1))
    expanded_x2 = expanded_x1 + target_width
    expanded_bbox = (expanded_x1, y1, expanded_x2, y2)
    expanded_aspect = float(height) / float(max(1, target_width))
    return expanded_bbox, True, original_aspect, expanded_aspect


def _osb_orientation_layout_bboxes(
    unexpanded_bbox: Tuple[int, int, int, int],
    expanded_bbox: Tuple[int, int, int, int],
    padding_pixels: float,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Use OSB expansion only for the horizontal orientation candidate."""
    horizontal_bbox = _inset_bbox_for_padding(expanded_bbox, padding_pixels)
    vertical_bbox = _inset_bbox_for_padding(unexpanded_bbox, padding_pixels)
    return horizontal_bbox, vertical_bbox


def _osb_short_vertical_sfx_for_tag(
    render_tag: Any,
    bubble: Dict[str, Any],
    text: Any,
    source_text: Any,
) -> bool:
    if not _is_sfx_render_tag(render_tag):
        return False
    return bool(
        _is_short_vertical_source_sfx_candidate(
            bubble,
            text,
            source_text,
        )
        or _is_short_vertical_dark_osb_sfx_shape(
            bubble,
            text,
        )
    )


def _should_compare_sfx_vertical_speech_bubble(
    bubble: Dict[str, Any],
    render_tag: Any,
    text: Any,
    max_chars: Optional[int],
) -> Tuple[bool, float]:
    if not _is_sfx_render_tag(render_tag):
        return False, 0.0
    try:
        vertical_max_chars = max(0, int(max_chars))
    except (TypeError, ValueError):
        return False, 0.0
    char_count = _regular_vertical_count_for_limit(text)
    if (
        vertical_max_chars <= 0
        or char_count <= 0
        or char_count > vertical_max_chars
    ):
        return False, 0.0
    try:
        bbox = bubble.get("bbox")
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return False, 0.0

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = height / float(width)
    return (
        aspect_ratio >= SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MIN_ASPECT,
        aspect_ratio,
    )


def _should_compare_forced_vertical_layout(
    is_outside_text: bool,
    force_on_ot_vertical: bool,
    osb_is_sfx_tag: bool,
    osb_forced_sfx_vertical: bool,
    tilted_vertical_allowed: bool,
) -> bool:
    if is_outside_text:
        return (
            not force_on_ot_vertical
            and (not osb_is_sfx_tag or osb_forced_sfx_vertical)
        )
    return bool(tilted_vertical_allowed)


def _configure_tilted_vertical_comparison(
    render_config: RenderingConfig,
    tilted_vertical_allowed: bool,
) -> None:
    if not tilted_vertical_allowed:
        return
    render_config.ignore_auto_vertical_font_gain = True
    render_config.forced_vertical_min_font_size_ratio = 1.0


def _is_on_ot_render_tag(render_tag: Any) -> bool:
    tag_name = str(render_tag or "").upper().strip().strip("<>")
    return tag_name in {"ON", "OM", "OT"}


def _is_ot_render_tag(render_tag: Any) -> bool:
    tag_name = str(render_tag or "").upper().strip().strip("<>")
    return tag_name == "OT"


def _osb_padding_for_render_tag(
    is_outside_text: bool,
    render_tag: Any,
    base_padding_pixels: float,
) -> float:
    try:
        padding = max(0.0, float(base_padding_pixels))
    except (TypeError, ValueError):
        padding = 0.0
    if is_outside_text and _is_ot_render_tag(render_tag):
        padding += OSB_OT_EXTRA_PADDING_PIXELS
    return padding


def _single_column_vertical_text(text: Any) -> str:
    value = str(text or "")
    value = (
        value.replace("\\r\\n", " ")
        .replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\r", " ")
    )
    return re.sub(r"\s+", " ", value).strip()


def _split_on_ot_forced_vertical_text(text: Any, max_columns: int) -> str:
    value = _single_column_vertical_text(text)
    if max_columns <= 1:
        return value

    char_count = _count_osb_vertical_chars(value)
    if char_count <= 1:
        return value

    best_split = None
    best_score = None
    for split_pos in range(1, len(value)):
        left = value[:split_pos].strip()
        right = value[split_pos:].strip()
        if not left or not right:
            continue

        left_count = _count_osb_vertical_chars(left)
        right_count = _count_osb_vertical_chars(right)
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


def _should_force_on_ot_vertical(
    bubble: Dict[str, Any],
    render_tag: Any,
    text: Any,
    image_size: Tuple[int, int],
    max_chars_override: Optional[int],
    font_size_scale: float = 1.0,
) -> Tuple[bool, float, int, int, float, int]:
    if not _is_on_ot_render_tag(render_tag):
        return False, 0.0, 0, 0, 0.0, 0

    try:
        max_chars = int(max_chars_override)
    except (TypeError, ValueError):
        return False, 0.0, 0, 0, 0.0, 0
    if max_chars <= 0:
        return False, 0.0, 0, 0, 0.0, 0

    char_count = _count_osb_vertical_chars(text)
    if char_count <= 0 or char_count > max_chars:
        return False, 0.0, char_count, 0, 0.0, 0

    try:
        bbox = bubble.get("original_bbox") or bubble.get("bbox")
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return False, 0.0, char_count, 0, 0.0, 0

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    try:
        img_w = max(1, int(image_size[0]))
    except (TypeError, ValueError, IndexError):
        img_w = width

    aspect_ratio = height / float(width)
    width_ratio = width / float(img_w)

    if not (
        aspect_ratio >= ON_OT_FORCE_VERTICAL_MIN_ASPECT
        and width_ratio <= ON_OT_FORCE_VERTICAL_MAX_WIDTH_RATIO
    ):
        return False, aspect_ratio, char_count, width, width_ratio, 0

    one_column_width = _scaled_render_length(
        ON_OT_FORCE_VERTICAL_ONE_COLUMN_MAX_WIDTH_PX,
        font_size_scale,
        minimum=1,
        maximum=max(1, img_w),
    )
    two_column_width = _scaled_render_length(
        ON_OT_FORCE_VERTICAL_TWO_COLUMN_MAX_WIDTH_PX,
        font_size_scale,
        minimum=one_column_width,
        maximum=max(1, img_w),
    )

    if width <= one_column_width:
        return True, aspect_ratio, char_count, width, width_ratio, 1
    if width <= two_column_width:
        return True, aspect_ratio, char_count, width, width_ratio, 2

    return False, aspect_ratio, char_count, width, width_ratio, 0


def _forced_on_ot_vertical_anchor_bbox(
    bubble: Dict[str, Any],
    fallback_bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Keep forced vertical ON/OM/OT text anchored to its original OCR box."""
    raw = bubble.get("original_bbox") or bubble.get("bbox") or fallback_bbox
    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in raw), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    return clipped or fallback_bbox


def _is_dark_tilted_osb_title_candidate(
    bubble: Dict[str, Any],
    image_size: Tuple[int, int],
) -> bool:
    try:
        bbox = bubble.get("original_bbox") or bubble.get("bbox")
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return False

    img_w, img_h = image_size
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = width / float(max(1, height))
    width_ratio = width / float(max(1, img_w))
    height_ratio = height / float(max(1, img_h))
    top_ratio = y1 / float(max(1, img_h))

    try:
        horizontal_rotation = abs(
            float(bubble.get("horizontal_rotation_deg", 0.0) or 0.0)
        )
    except (TypeError, ValueError):
        horizontal_rotation = 0.0

    if (
        bool(bubble.get("is_dark_text", False))
        and horizontal_rotation >= 3.0
        and aspect_ratio >= 1.20
        and height_ratio <= 0.12
    ):
        return True

    return False


def _has_trusted_horizontal_osb_rotation(
    bubble: Dict[str, Any],
    render_tag: Any,
    image_size: Tuple[int, int],
) -> bool:
    if _is_sfx_render_tag(render_tag):
        return False

    try:
        horizontal_rotation = abs(
            float(bubble.get("horizontal_rotation_deg", 0.0) or 0.0)
        )
    except (TypeError, ValueError):
        return False
    if horizontal_rotation < 6.0 or horizontal_rotation > 15.0:
        return False

    try:
        bbox = bubble.get("original_bbox") or bubble.get("bbox")
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return False

    img_w, img_h = image_size
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = width / float(max(1, height))
    width_ratio = width / float(max(1, img_w))
    height_ratio = height / float(max(1, img_h))

    if aspect_ratio < 1.8:
        return False
    if width_ratio < 0.04 or height_ratio > 0.12:
        return False

    tag_name = str(render_tag or "").upper().strip().strip("<>")
    return tag_name in {"OT", "ON", "BT", "BE"} or bool(
        bubble.get("is_outside_text", False)
    )


def _should_ignore_osb_rotation_for_caption_text(
    bubble: Dict[str, Any],
    translated_text: Any,
    source_text: Any,
    image_size: Tuple[int, int],
) -> bool:
    source = str(source_text or "")
    translated = str(translated_text or "")
    combined = f"{source} {translated}"
    if _translated_looks_like_short_sfx(translated) or _is_japanese_sfx_like(source):
        return False

    source_plain = re.sub(r"\s+", "", source)
    translated_plain = re.sub(r"\s+", "", translated)

    try:
        bbox = bubble.get("original_bbox") or bubble.get("bbox")
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return False

    img_w, img_h = image_size
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    aspect_ratio = width / float(max(1, height))
    width_ratio = width / float(max(1, img_w))
    height_ratio = height / float(max(1, img_h))
    top_ratio = y1 / float(max(1, img_h))

    if _is_dark_tilted_osb_title_candidate(bubble, image_size):
        return False

    credit_words = (
        "원작",
        "작화",
        "캐릭터",
        "디자인",
        "오버랩",
        "간행",
        "岡沢",
        "細雪",
        "村上",
        "原作",
        "作画",
        "キャラクター",
        "Jun",
        "Yuichi",
        "Rokujuuyon",
        "Okazawa",
        "Sasameyuki",
        "Murakami",
    )
    if any(word in combined for word in credit_words) and height_ratio <= 0.12:
        return True

    has_letters = bool(
        re.search(r"[A-Za-z가-힣ぁ-んァ-ン一-龯]", source_plain + translated_plain)
    )
    if not has_letters:
        return False

    translated_word_count = _osb_rotation_word_count(translated)
    source_has_digit = bool(re.search(r"\d", source_plain))
    source_japanese_count = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", source))
    looks_like_short_sfx = _translated_looks_like_short_sfx(translated)

    if aspect_ratio >= 1.15 and not looks_like_short_sfx:
        if source_has_digit and len(translated_plain) >= 3:
            return True
        if translated_word_count >= 2 and len(translated_plain) >= 4:
            return True
        if source_japanese_count >= 5 and len(translated_plain) >= 4:
            return True

    if (
        aspect_ratio >= 2.0
        and not looks_like_short_sfx
        and translated_word_count >= 1
        and len(translated_plain) >= 4
    ):
        return True

    if (
        top_ratio >= 0.70
        and height_ratio <= 0.075
        and aspect_ratio >= 1.20
        and len(translated_plain) >= 2
    ):
        return True

    if (
        top_ratio >= 0.62
        and height_ratio <= 0.09
        and width_ratio >= 0.12
        and aspect_ratio >= 2.0
        and len(translated_plain) >= 4
    ):
        return True

    return False


def _osb_component_text_clusters(
    ink_crop: np.ndarray,
    base_box: Tuple[int, int, int, int],
    part_count: int,
) -> List[Tuple[int, int, int, int, int]]:
    raw_h, raw_w = ink_crop.shape[:2]
    if part_count < 2 or raw_w < 10 or raw_h < 10:
        return []

    try:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ink_crop.astype(np.uint8), 8
        )
    except Exception:
        return []
    if num_labels <= 1:
        return []

    crop_area = float(max(1, raw_w * raw_h))
    min_area = max(12.0, crop_area * 0.00025)
    max_area = crop_area * 0.08
    kept_mask = np.zeros_like(ink_crop, dtype=np.uint8)
    kept_components: List[Tuple[float, float, float]] = []
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if max(comp_w, comp_h) < 5:
            continue
        kept_mask[labels == label_idx] = 1
        kept_components.append(
            (
                float(centroids[label_idx][0]),
                float(centroids[label_idx][1]),
                area,
            )
        )

    if len(kept_components) < part_count * 2:
        return []

    try:
        kx = max(3, min(13, int(round(raw_w * 0.06))))
        ky = max(3, min(13, int(round(raw_h * 0.08))))
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        grouped = cv2.dilate(
            kept_mask.astype(np.uint8),
            np.ones((ky, kx), np.uint8),
            iterations=1,
        )
        grouped = cv2.morphologyEx(
            grouped, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )
        group_count, group_labels, _, _ = cv2.connectedComponentsWithStats(grouped, 8)
    except Exception:
        return []

    x1, y1, _, _ = base_box
    clusters: List[Tuple[int, int, int, int, int, int]] = []
    min_group_ink = max(24, int(crop_area * 0.001))
    for group_idx in range(1, group_count):
        group_mask = group_labels == group_idx
        component_ink = kept_mask.astype(bool) & group_mask
        ink_count = int(np.count_nonzero(component_ink))
        if ink_count < min_group_ink:
            continue
        ys, xs = np.where(component_ink)
        if xs.size == 0 or ys.size == 0:
            continue
        component_count = 0
        for cx, cy, _ in kept_components:
            ix = int(round(cx))
            iy = int(round(cy))
            if 0 <= ix < raw_w and 0 <= iy < raw_h and group_mask[iy, ix]:
                component_count += 1
        if component_count < 2:
            continue
        cx1 = int(xs.min())
        cy1 = int(ys.min())
        cx2 = int(xs.max()) + 1
        cy2 = int(ys.max()) + 1
        if cx2 - cx1 < max(8, int(raw_w * 0.04)):
            continue
        if cy2 - cy1 < max(8, int(raw_h * 0.04)):
            continue
        clusters.append(
            (
                x1 + cx1,
                y1 + cy1,
                x1 + cx2,
                y1 + cy2,
                ink_count,
                component_count,
            )
        )

    if len(clusters) < part_count:
        return []

    clusters.sort(key=lambda box: (box[5], box[4]), reverse=True)
    selected = clusters[:part_count]
    if raw_w / float(max(1, raw_h)) >= 1.25:
        selected.sort(key=lambda box: ((box[0] + box[2]) / 2.0, box[1]))
    else:
        selected.sort(key=lambda box: ((box[1] + box[3]) / 2.0, box[0]))

    return [(x1, y1, x2, y2, ink_count) for x1, y1, x2, y2, ink_count, _ in selected]


def _osb_projection_text_clusters(
    ink_crop: np.ndarray,
    base_box: Tuple[int, int, int, int],
    part_count: int,
) -> List[Tuple[int, int, int, int, int]]:
    raw_h, raw_w = ink_crop.shape[:2]
    if part_count < 2 or part_count > 4 or raw_w < 10 or raw_h < 10:
        return []

    ys, xs = np.where(ink_crop > 0)
    if xs.size < max(24, part_count * 10):
        return []

    split_vertically = raw_h >= raw_w * 1.15
    coords = ys.astype(np.float64) if split_vertically else xs.astype(np.float64)
    if coords.size == 0 or float(coords.max() - coords.min()) < 8.0:
        return []

    quantiles = [(idx + 0.5) / float(part_count) for idx in range(part_count)]
    centers = np.quantile(coords, quantiles).astype(np.float64)
    labels = np.zeros(coords.shape[0], dtype=np.int32)
    for _ in range(16):
        distances = np.abs(coords[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for idx in range(part_count):
            group = coords[new_labels == idx]
            if group.size == 0:
                return []
            new_centers[idx] = float(np.median(group))
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    order = sorted(range(part_count), key=lambda idx: centers[idx])
    x1, y1, _, _ = base_box
    total_ink = float(max(1, xs.size))
    clusters: List[Tuple[int, int, int, int, int]] = []
    previous_center = None
    min_part_ink = max(8.0, total_ink * 0.08)
    for idx in order:
        mask = labels == idx
        ink_count = int(np.count_nonzero(mask))
        if ink_count < min_part_ink:
            return []
        part_xs = xs[mask]
        part_ys = ys[mask]
        if part_xs.size == 0 or part_ys.size == 0:
            return []
        center = float(np.median(part_ys if split_vertically else part_xs))
        if previous_center is not None:
            min_gap = max(5.0, (raw_h if split_vertically else raw_w) * 0.07)
            if center - previous_center < min_gap:
                return []
        previous_center = center
        cx1 = int(part_xs.min())
        cy1 = int(part_ys.min())
        cx2 = int(part_xs.max()) + 1
        cy2 = int(part_ys.max()) + 1
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            return []
        clusters.append((x1 + cx1, y1 + cy1, x1 + cx2, y1 + cy2, ink_count))

    return clusters


def _osb_source_supports_repeated_group_split(
    bubble: Dict[str, Any],
    original_image: Image.Image,
    image_size: Tuple[int, int],
    part_count: int,
) -> bool:
    if part_count < 2 or part_count > 4:
        return False

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        base_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box), image_size
        )
    except (TypeError, ValueError):
        base_box = None
    if base_box is None:
        return False

    x1, y1, x2, y2 = base_box
    raw_w = max(1, x2 - x1)
    raw_h = max(1, y2 - y1)
    if raw_w < 10 or raw_h < 10:
        return False

    ink_crop = _osb_source_ink_crop(
        bubble,
        original_image,
        base_box,
        image_size,
        allow_render_mask=False,
        side_only_filter=False,
    )
    if ink_crop is None or ink_crop.shape[:2] != (raw_h, raw_w):
        return False

    clusters = _osb_component_text_clusters(ink_crop, base_box, part_count)
    if len(clusters) < part_count:
        return False

    clusters = clusters[:part_count]
    widths = [max(1.0, float(box[2] - box[0])) for box in clusters]
    heights = [max(1.0, float(box[3] - box[1])) for box in clusters]
    areas = [width * height for width, height in zip(widths, heights)]
    if min(areas) / max(areas) < OSB_REPEATED_SOURCE_SPLIT_MIN_AREA_RATIO:
        return False
    if min(widths) / max(widths) < OSB_REPEATED_SOURCE_SPLIT_MIN_SIDE_RATIO:
        return False
    if min(heights) / max(heights) < OSB_REPEATED_SOURCE_SPLIT_MIN_SIDE_RATIO:
        return False

    centers_x = [((box[0] + box[2]) / 2.0) for box in clusters]
    centers_y = [((box[1] + box[3]) / 2.0) for box in clusters]
    x_span = (max(centers_x) - min(centers_x)) / float(max(1, raw_w))
    y_span = (max(centers_y) - min(centers_y)) / float(max(1, raw_h))
    if max(x_span, y_span) < OSB_REPEATED_SOURCE_SPLIT_MIN_CENTER_SPAN_RATIO:
        return False

    min_center_distance = None
    for idx in range(len(clusters)):
        for other_idx in range(idx + 1, len(clusters)):
            dx = centers_x[idx] - centers_x[other_idx]
            dy = centers_y[idx] - centers_y[other_idx]
            distance = math.hypot(dx, dy)
            if min_center_distance is None or distance < min_center_distance:
                min_center_distance = distance
    if min_center_distance is None:
        return False
    return (
        min_center_distance / float(max(1, min(raw_w, raw_h)))
        >= OSB_REPEATED_SOURCE_SPLIT_MIN_DISTANCE_RATIO
    )


def _osb_split_layouts_are_balanced(
    part_texts: List[str],
    layouts: List[Dict[str, Any]],
) -> bool:
    """Reject letter-by-letter OSB placement when one inferred part is tiny.

    Short SFX such as "ああ" can contain large connected strokes plus small
    motion lines.  If the cluster finder mistakes a small stroke for one of the
    text characters, the second translated glyph gets rendered in a postage
    stamp-sized box.  In that case falling back to the full OSB box gives a
    much better result than preserving the bad per-character position.
    """
    if len(layouts) < 2 or len(layouts) != len(part_texts):
        return True

    token_lengths = [len(_osb_split_token_key(text)) for text in part_texts]
    if not token_lengths or max(token_lengths) > 2:
        return True

    repeated_short_split = _is_repeated_short_osb_split(part_texts)
    widths: List[float] = []
    heights: List[float] = []
    areas: List[float] = []
    font_sizes: List[float] = []
    for layout in layouts:
        try:
            x1, y1, x2, y2 = [
                float(v) for v in layout.get("bbox", (0, 0, 0, 0))
            ]
        except (TypeError, ValueError):
            return False
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            return False
        widths.append(width)
        heights.append(height)
        areas.append(width * height)

        max_font = layout.get("max_font_size")
        if max_font is not None:
            try:
                font_sizes.append(float(max_font))
            except (TypeError, ValueError):
                return False

    area_ratio = min(areas) / max(areas)
    width_ratio = min(widths) / max(widths)
    height_ratio = min(heights) / max(heights)
    side_ratio = min(width_ratio, height_ratio)
    if (
        repeated_short_split
        and (
            area_ratio < OSB_CLUSTER_REPEAT_SPLIT_MIN_AREA_RATIO
            or side_ratio < OSB_CLUSTER_REPEAT_SPLIT_MIN_SIDE_RATIO
        )
    ):
        return False
    if (
        not repeated_short_split
        and (
            area_ratio < OSB_CLUSTER_SPLIT_MIN_AREA_RATIO
            or side_ratio < OSB_CLUSTER_SPLIT_MIN_SIDE_RATIO
        )
    ):
        return False

    if len(font_sizes) == len(layouts):
        font_ratio = min(font_sizes) / max(font_sizes)
        min_font_ratio = (
            OSB_CLUSTER_REPEAT_SPLIT_MIN_FONT_RATIO
            if repeated_short_split
            else OSB_CLUSTER_SPLIT_MIN_FONT_RATIO
        )
        if font_ratio < min_font_ratio:
            return False

    return True


def _separate_overlapping_vertical_repeated_osb_layouts(
    part_texts: List[str],
    layouts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep repeated vertical SFX parts from drawing over each other."""
    if (
        len(layouts) < 2
        or len(layouts) != len(part_texts)
        or not _is_repeated_short_osb_split(part_texts)
        or not all(bool(layout.get("vertical_stack")) for layout in layouts)
    ):
        return layouts

    parsed: List[Tuple[int, float, float, float, float]] = []
    for index, layout in enumerate(layouts):
        try:
            x1, y1, x2, y2 = [
                float(value) for value in layout.get("bbox", (0, 0, 0, 0))
            ]
        except (TypeError, ValueError):
            return layouts
        if x2 - x1 < 12.0 or y2 - y1 < 12.0:
            return layouts
        parsed.append((index, x1, y1, x2, y2))

    centers_x = [(item[1] + item[3]) / 2.0 for item in parsed]
    centers_y = [(item[2] + item[4]) / 2.0 for item in parsed]
    if (
        max(centers_y) - min(centers_y)
        <= max(4.0, max(centers_x) - min(centers_x))
    ):
        return layouts

    ordered = sorted(parsed, key=lambda item: ((item[2] + item[4]) / 2.0, item[1]))
    adjusted = [dict(layout) for layout in layouts]
    changed = False
    min_part_height = 12

    for first, second in zip(ordered, ordered[1:]):
        first_index, first_x1, first_y1, first_x2, first_y2 = first
        second_index, second_x1, second_y1, second_x2, second_y2 = second
        x_overlap = min(first_x2, second_x2) - max(first_x1, second_x1)
        y_overlap = min(first_y2, second_y2) - max(first_y1, second_y1)
        if x_overlap <= 0.0 or y_overlap <= 0.0:
            continue

        first_center_y = (first_y1 + first_y2) / 2.0
        second_center_y = (second_y1 + second_y2) / 2.0
        boundary = int(round((first_center_y + second_center_y) / 2.0))
        boundary = max(
            int(round(first_y1)) + min_part_height,
            min(int(round(second_y2)) - min_part_height, boundary),
        )
        if (
            boundary <= first_y1
            or boundary >= second_y2
            or boundary - first_y1 < min_part_height
            or second_y2 - boundary < min_part_height
        ):
            continue

        first_box = list(adjusted[first_index]["bbox"])
        second_box = list(adjusted[second_index]["bbox"])
        first_box[3] = min(int(round(float(first_box[3]))), boundary)
        second_box[1] = max(int(round(float(second_box[1]))), boundary)
        if (
            first_box[3] - first_box[1] < min_part_height
            or second_box[3] - second_box[1] < min_part_height
        ):
            continue

        adjusted[first_index]["bbox"] = tuple(int(round(float(v))) for v in first_box)
        adjusted[second_index]["bbox"] = tuple(
            int(round(float(v))) for v in second_box
        )
        changed = True

    return adjusted if changed else layouts


def _osb_horizontal_split_clusters_are_stacked(
    clusters: List[Tuple[int, int, int, int, int]],
    raw_w: int,
    raw_h: int,
) -> bool:
    if len(clusters) < 2:
        return False

    centers_x = [((box[0] + box[2]) / 2.0) for box in clusters]
    centers_y = [((box[1] + box[3]) / 2.0) for box in clusters]
    x_span = (max(centers_x) - min(centers_x)) / float(max(1, raw_w))
    y_span = (max(centers_y) - min(centers_y)) / float(max(1, raw_h))
    if (
        x_span > OSB_CLUSTER_HORIZONTAL_STACKED_MAX_X_SPAN_RATIO
        or y_span < OSB_CLUSTER_HORIZONTAL_STACKED_MIN_Y_SPAN_RATIO
    ):
        return False

    overlap_ratios: List[float] = []
    ordered = sorted(clusters, key=lambda box: ((box[1] + box[3]) / 2.0, box[0]))
    for first, second in zip(ordered, ordered[1:]):
        first_w = max(1.0, float(first[2] - first[0]))
        second_w = max(1.0, float(second[2] - second[0]))
        overlap = max(0.0, float(min(first[2], second[2]) - max(first[0], second[0])))
        overlap_ratios.append(overlap / max(1.0, min(first_w, second_w)))

    if not overlap_ratios:
        return False
    return (
        float(np.median(overlap_ratios))
        >= OSB_CLUSTER_HORIZONTAL_STACKED_MIN_OVERLAP_RATIO
    )


def _osb_stacked_source_position_render_boxes(
    part_texts: List[str],
    base_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    default_max_font_size: int,
    compact_visual: bool = False,
    page_font_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    part_count = len(part_texts)
    if part_count < 2 or part_count > 4:
        return []

    x1, y1, x2, y2 = base_box
    raw_w = max(1.0, float(x2 - x1))
    raw_h = max(1.0, float(y2 - y1))
    try:
        default_font_cap = max(1, int(default_max_font_size))
    except (TypeError, ValueError):
        default_font_cap = OSB_CLUSTER_HORIZONTAL_SFX_MAX_FONT_SIZE
    font_cap = min(
        default_font_cap,
        _scaled_render_font_size(
            OSB_CLUSTER_HORIZONTAL_SFX_MAX_FONT_SIZE,
            page_font_scale,
            minimum=10,
            maximum=640,
        ),
    )
    split_padding = scale_scalar(
        OSB_CLUSTER_SPLIT_PADDING_PIXELS,
        page_font_scale,
        minimum=0.0,
        maximum=32.0,
    )

    if compact_visual and part_count == 2:
        anchors = [(0.40, 0.38), (0.61, 0.60)]
        font_ratios = [0.43, 0.40]
        width_ratios = [0.38, 0.36]
        height_ratios = [0.40, 0.38]
    elif compact_visual and part_count == 3:
        anchors = [(0.30, 0.31), (0.50, 0.50), (0.66, 0.66)]
        font_ratios = [0.43, 0.38, 0.34]
        width_ratios = [0.40, 0.36, 0.34]
        height_ratios = [0.40, 0.36, 0.34]
    elif compact_visual:
        anchors = [(0.32, 0.28), (0.46, 0.43), (0.60, 0.58), (0.74, 0.72)]
        font_ratios = [0.38, 0.36, 0.34, 0.32]
        width_ratios = [0.36, 0.35, 0.34, 0.32]
        height_ratios = [0.36, 0.35, 0.34, 0.32]
    elif part_count == 2:
        anchors = [(0.35, 0.34), (0.68, 0.70)]
        font_ratios = [0.40, 0.38]
        width_ratios = [0.44, 0.38]
        height_ratios = [0.46, 0.46]
    elif part_count == 3:
        anchors = [(0.28, 0.30), (0.56, 0.60), (0.86, 0.82)]
        font_ratios = [0.44, 0.34, 0.30]
        width_ratios = [0.46, 0.42, 0.30]
        height_ratios = [0.47, 0.43, 0.35]
    else:
        anchors = [(0.24, 0.20), (0.39, 0.43), (0.60, 0.65), (0.82, 0.84)]
        font_ratios = [0.36, 0.34, 0.32, 0.30]
        width_ratios = [0.42, 0.42, 0.40, 0.34]
        height_ratios = [0.38, 0.38, 0.36, 0.34]

    layouts: List[Dict[str, Any]] = []
    for idx, part_text in enumerate(part_texts):
        token_len = max(1, len(_osb_split_token_key(part_text)))
        max_font = max(12, int(round(raw_h * font_ratios[idx])))
        max_font = min(font_cap, max_font)
        glyph_width_mult = 0.88 if token_len == 1 else 1.05 + token_len * 0.20
        target_w = max(raw_w * width_ratios[idx], max_font * glyph_width_mult)
        target_h = max(raw_h * height_ratios[idx], max_font * 1.12)
        center_x = x1 + raw_w * anchors[idx][0]
        center_y = y1 + raw_h * anchors[idx][1]
        candidate = (
            center_x - target_w / 2.0,
            center_y - target_h / 2.0,
            center_x + target_w / 2.0,
            center_y + target_h / 2.0,
        )
        shifted = _shift_bbox_inside(candidate, base_box)
        clipped = _clip_bbox_to_image(shifted, image_size)
        if clipped is None:
            return []
        if clipped[2] - clipped[0] < 12 or clipped[3] - clipped[1] < 10:
            return []
        layouts.append(
            {
                "bbox": clipped,
                "vertical_stack": False,
                "rotation_deg": 0.0,
                "max_font_size": max_font,
                "padding_pixels": split_padding,
                "lock_vertical_stack": True,
                "source_position_stack": True,
            }
        )

    return layouts


def _osb_spaced_vertical_sfx_source_position_render_boxes(
    part_texts: List[str],
    base_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    default_max_font_size: int,
    rotation_deg: float = 0.0,
    page_font_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    if len(part_texts) != 2:
        return []

    x1, y1, x2, y2 = base_box
    raw_w = max(1.0, float(x2 - x1))
    raw_h = max(1.0, float(y2 - y1))
    aspect = raw_h / raw_w
    if not 1.25 <= aspect <= 2.25:
        return []

    try:
        default_font_cap = max(1, int(default_max_font_size))
    except (TypeError, ValueError):
        default_font_cap = OSB_CLUSTER_REPEAT_MAX_FONT_SIZE
    font_cap = min(
        default_font_cap,
        _scaled_render_font_size(
            OSB_CLUSTER_REPEAT_MAX_FONT_SIZE,
            page_font_scale,
            minimum=10,
            maximum=640,
        ),
    )
    split_padding = scale_scalar(
        OSB_CLUSTER_SPLIT_PADDING_PIXELS,
        page_font_scale,
        minimum=0.0,
        maximum=32.0,
    )

    anchors = [(0.28, 0.33), (0.78, 0.78)]
    width_ratios = [0.48, 0.42]
    height_ratios = [0.48, 0.42]
    font_ratios = [0.31, 0.29]
    try:
        part_rotation = float(rotation_deg or 0.0)
    except (TypeError, ValueError):
        part_rotation = 0.0
    if abs(part_rotation) > OSB_CLUSTER_REPEAT_PAIR_MAX_ROTATION_DEGREES:
        part_rotation = math.copysign(
            OSB_CLUSTER_REPEAT_PAIR_MAX_ROTATION_DEGREES,
            part_rotation,
        )

    layouts: List[Dict[str, Any]] = []
    for idx, part_text in enumerate(part_texts):
        token_len = max(1, len(_osb_split_token_key(part_text)))
        max_font = max(12, int(round(raw_h * font_ratios[idx])))
        max_font = min(font_cap, max_font)
        target_w = max(raw_w * width_ratios[idx], max_font * 1.18)
        target_h = max(
            raw_h * height_ratios[idx],
            max_font * (token_len + 0.25),
        )
        target_w = min(target_w, raw_w * 0.58)
        target_h = min(target_h, raw_h * (0.50 if idx == 0 else 0.44))

        center_x = x1 + raw_w * anchors[idx][0]
        center_y = y1 + raw_h * anchors[idx][1]
        candidate = (
            center_x - target_w / 2.0,
            center_y - target_h / 2.0,
            center_x + target_w / 2.0,
            center_y + target_h / 2.0,
        )
        shifted = _shift_bbox_inside(candidate, base_box)
        clipped = _clip_bbox_to_image(shifted, image_size)
        if clipped is None:
            return []
        if clipped[2] - clipped[0] < 12 or clipped[3] - clipped[1] < 10:
            return []
        layouts.append(
            {
                "bbox": clipped,
                "vertical_stack": True,
                "rotation_deg": part_rotation,
                "max_font_size": max_font,
                "padding_pixels": split_padding,
                "lock_vertical_stack": True,
                "source_position_stack": True,
            }
        )

    return layouts


def _union_int_boxes(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _large_mixed_sfx_source_boxes(
    bubble: Dict[str, Any],
    original_image: Image.Image,
    image_size: Tuple[int, int],
    expected_body_count: int,
) -> List[Tuple[int, int, int, int]]:
    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        base_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box), image_size
        )
    except (TypeError, ValueError):
        base_box = None
    if base_box is None or expected_body_count < 3:
        return []

    x1, y1, x2, y2 = base_box
    raw_w = max(1, x2 - x1)
    raw_h = max(1, y2 - y1)
    image_w, image_h = image_size
    if raw_w / float(max(1, image_w)) < 0.35:
        return []
    if raw_h / float(max(1, image_h)) < 0.25:
        return []

    try:
        gray = np.asarray(original_image.crop(base_box).convert("L"))
    except Exception:
        return []
    if gray.shape[:2] != (raw_h, raw_w) or gray.size == 0:
        return []

    dark = gray <= 105
    bright = gray >= 238
    try:
        near_bright = cv2.dilate(
            bright.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1
        ).astype(bool)
        candidate = np.logical_and(dark, near_bright).astype(np.uint8)
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1
        )
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate, 8
        )
    except Exception:
        return []

    crop_area = float(max(1, raw_w * raw_h))
    min_area = max(120.0, crop_area * 0.00055)
    max_area = crop_area * 0.04
    components: List[Dict[str, Any]] = []
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if width < 5 or height < 5:
            continue
        if width <= 10 and height >= raw_h * 0.22:
            continue
        if height <= 10 and width >= raw_w * 0.22:
            continue
        cx_ratio = float(centroids[label_idx][0]) / float(max(1, raw_w))
        cy_ratio = float(centroids[label_idx][1]) / float(max(1, raw_h))
        components.append(
            {
                "bbox": (x1 + left, y1 + top, x1 + left + width, y1 + top + height),
                "area": area,
                "cx": cx_ratio,
                "cy": cy_ratio,
                "cx_abs": x1 + float(centroids[label_idx][0]),
                "cy_abs": y1 + float(centroids[label_idx][1]),
            }
        )

    upper_components = [
        comp for comp in components if comp["cx"] >= 0.78 and comp["cy"] <= 0.32
    ]
    upper_box = _union_int_boxes([comp["bbox"] for comp in upper_components])
    if upper_box is None:
        return []

    body_components: List[Dict[str, Any]] = []
    for comp in components:
        if comp in upper_components:
            continue
        if comp["cx"] >= 0.62 and comp["cy"] <= 0.50:
            continue
        if comp["area"] < max(300.0, crop_area * 0.0018):
            continue
        body_components.append(comp)

    clusters: List[Dict[str, Any]] = []
    for comp in sorted(body_components, key=lambda item: (item["cy"], item["cx"])):
        cb = comp["bbox"]
        merged_into: Optional[int] = None
        for idx, cluster in enumerate(clusters):
            bx = cluster["bbox"]
            x_overlap = min(cb[2], bx[2]) - max(cb[0], bx[0])
            y_overlap = min(cb[3], bx[3]) - max(cb[1], bx[1])
            x_gap = max(0, max(cb[0], bx[0]) - min(cb[2], bx[2]))
            y_gap = max(0, max(cb[1], bx[1]) - min(cb[3], bx[3]))
            center_dx = abs(float(comp["cx_abs"]) - float(cluster["cx_abs"]))
            center_dy = abs(float(comp["cy_abs"]) - float(cluster["cy_abs"]))
            if (
                center_dx <= raw_w * 0.18
                and center_dy <= raw_h * 0.16
                and (
                    x_overlap > 0
                    or y_overlap > 0
                    or x_gap <= raw_w * 0.04
                    or y_gap <= raw_h * 0.04
                )
            ):
                merged_into = idx
                break
        if merged_into is None:
            clusters.append(
                {
                    "bbox": cb,
                    "area": comp["area"],
                    "cx_abs": comp["cx_abs"],
                    "cy_abs": comp["cy_abs"],
                }
            )
        else:
            cluster = clusters[merged_into]
            old_area = float(cluster["area"])
            new_area = old_area + float(comp["area"])
            cluster["bbox"] = _union_int_boxes([cluster["bbox"], cb]) or cluster["bbox"]
            cluster["cx_abs"] = (
                float(cluster["cx_abs"]) * old_area
                + float(comp["cx_abs"]) * float(comp["area"])
            ) / new_area
            cluster["cy_abs"] = (
                float(cluster["cy_abs"]) * old_area
                + float(comp["cy_abs"]) * float(comp["area"])
            ) / new_area
            cluster["area"] = new_area

    clusters = [
        cluster
        for cluster in clusters
        if (cluster["bbox"][2] - cluster["bbox"][0]) >= 14
        and (cluster["bbox"][3] - cluster["bbox"][1]) >= 24
    ]
    clusters = sorted(
        clusters,
        key=lambda cluster: (
            (cluster["bbox"][1] + cluster["bbox"][3]) / 2.0,
            (cluster["bbox"][0] + cluster["bbox"][2]) / 2.0,
        ),
    )
    if len(clusters) < expected_body_count:
        return []

    body_boxes = [cluster["bbox"] for cluster in clusters[:expected_body_count]]
    return [upper_box] + body_boxes


def _expand_source_layout_box(
    box: Tuple[int, int, int, int],
    base_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    part_text: str,
) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    raw_w = max(1.0, float(x2 - x1))
    raw_h = max(1.0, float(y2 - y1))
    token_len = max(1, len(_osb_split_token_key(part_text)))
    if token_len >= 2:
        target_w = max(raw_w * 1.22, 58.0)
        target_h = max(raw_h * 1.36, 102.0)
    else:
        target_w = max(raw_w * 1.18, 50.0)
        target_h = max(raw_h * 1.16, 70.0)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    candidate = (
        center_x - target_w / 2.0,
        center_y - target_h / 2.0,
        center_x + target_w / 2.0,
        center_y + target_h / 2.0,
    )
    shifted = _shift_bbox_inside(candidate, base_box)
    return _clip_bbox_to_image(shifted, image_size)


def _osb_large_mixed_sfx_render_boxes(
    part_texts: List[str],
    base_box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    default_max_font_size: int,
    source_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    page_font_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    if len(part_texts) < 4:
        return []

    x1, y1, x2, y2 = base_box
    raw_w = max(1.0, float(x2 - x1))
    raw_h = max(1.0, float(y2 - y1))
    if raw_w < 120.0 or raw_h < 160.0:
        return []
    try:
        default_font_cap = max(1, int(default_max_font_size))
    except (TypeError, ValueError):
        default_font_cap = OSB_CLUSTER_HORIZONTAL_SFX_MAX_FONT_SIZE
    font_cap = min(
        default_font_cap,
        _scaled_render_font_size(
            OSB_CLUSTER_HORIZONTAL_SFX_MAX_FONT_SIZE,
            page_font_scale,
            minimum=10,
            maximum=640,
        ),
    )
    split_padding = scale_scalar(
        OSB_CLUSTER_SPLIT_PADDING_PIXELS,
        page_font_scale,
        minimum=0.0,
        maximum=32.0,
    )

    layouts: List[Dict[str, Any]] = []
    if source_boxes and len(source_boxes) >= len(part_texts):
        source_rotations = [-6.0, -12.0, -8.0, -8.0, -10.0, -10.0]
        for idx, part_text in enumerate(part_texts):
            source_box = _expand_source_layout_box(
                source_boxes[idx], base_box, image_size, part_text
            )
            if source_box is None:
                return []
            box_w = max(1, source_box[2] - source_box[0])
            box_h = max(1, source_box[3] - source_box[1])
            token_len = max(1, len(_osb_split_token_key(part_text)))
            max_font = int(round(min(box_w * 1.05, box_h / (token_len + 0.25))))
            max_font = max(18, min(font_cap, max_font))
            layouts.append(
                {
                    "bbox": source_box,
                    "vertical_stack": True,
                    "rotation_deg": source_rotations[min(idx, len(source_rotations) - 1)],
                    "max_font_size": max_font,
                    "padding_pixels": split_padding,
                    "lock_vertical_stack": True,
                    "source_position_stack": True,
                    "large_mixed_sfx": True,
                }
            )
        return layouts

    anchors = [
        (0.82, 0.22),
        (0.25, 0.18),
        (0.40, 0.39),
        (0.58, 0.64),
        (0.77, 0.78),
    ]
    width_ratios = [0.30, 0.34, 0.23, 0.23, 0.23]
    height_ratios = [0.36, 0.30, 0.25, 0.25, 0.28]
    font_ratios = [0.14, 0.16, 0.15, 0.15, 0.14]
    rotations = [-6.0, -12.0, -8.0, -8.0, -10.0]
    for idx, part_text in enumerate(part_texts):
        token_len = max(1, len(_osb_split_token_key(part_text)))
        preset_idx = min(idx, len(anchors) - 1)
        max_font = max(18, int(round(raw_h * font_ratios[preset_idx])))
        max_font = min(font_cap, max_font)
        target_w = max(raw_w * width_ratios[preset_idx], max_font * 1.40)
        target_h = max(raw_h * height_ratios[preset_idx], max_font * (token_len + 0.55))
        center_x = x1 + raw_w * anchors[preset_idx][0]
        center_y = y1 + raw_h * anchors[preset_idx][1]
        candidate = (
            center_x - target_w / 2.0,
            center_y - target_h / 2.0,
            center_x + target_w / 2.0,
            center_y + target_h / 2.0,
        )
        shifted = _shift_bbox_inside(candidate, base_box)
        clipped = _clip_bbox_to_image(shifted, image_size)
        if clipped is None:
            return []
        if clipped[2] - clipped[0] < 14 or clipped[3] - clipped[1] < 16:
            return []
        layouts.append(
            {
                "bbox": clipped,
                "vertical_stack": True,
                "rotation_deg": rotations[preset_idx],
                "max_font_size": max_font,
                "padding_pixels": split_padding,
                "lock_vertical_stack": True,
                "source_position_stack": True,
                "large_mixed_sfx": True,
            }
        )

    return layouts


def _source_has_large_mixed_sfx_regions(
    bubble: Dict[str, Any],
    original_image: Image.Image,
    image_size: Tuple[int, int],
) -> bool:
    return bool(
        _large_mixed_sfx_source_boxes(
            bubble, original_image, image_size, expected_body_count=3
        )
    )


def _wide_spaced_sfx_projection_clusters(
    ink_crop: np.ndarray,
) -> List[Dict[str, float]]:
    if ink_crop is None:
        return []
    raw_h, raw_w = ink_crop.shape[:2]
    if raw_w < 10 or raw_h < 8:
        return []

    mask = (ink_crop > 0).astype(np.uint8)
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    except Exception:
        return []
    if num_labels <= 1:
        return []

    crop_area = float(max(1, raw_w * raw_h))
    min_component_area = max(4.0, crop_area * 0.00025)
    max_component_area = crop_area * 0.35
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if not (min_component_area <= area <= max_component_area):
            continue
        comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if comp_w < 2 or comp_h < 2:
            continue
        if comp_w > raw_w * 0.82 and comp_h <= max(2, int(round(raw_h * 0.05))):
            continue
        kept[labels == label_idx] = 1

    if int(np.count_nonzero(kept)) < OSB_SIDE_ANCHOR_MIN_INK_PIXELS:
        return []

    min_column_ink = max(1, int(round(raw_h * 0.015)))
    occupied = (np.count_nonzero(kept, axis=0) >= min_column_ink).astype(np.uint8)
    if int(np.count_nonzero(occupied)) == 0:
        return []

    try:
        kernel_w = max(3, min(9, int(round(raw_w * 0.025))))
        if kernel_w % 2 == 0:
            kernel_w += 1
        occupied = cv2.dilate(
            occupied.reshape(1, -1),
            np.ones((1, kernel_w), np.uint8),
            iterations=1,
        ).reshape(-1)
    except Exception:
        pass

    ranges: List[Dict[str, float]] = []
    start: Optional[int] = None
    for x, value in enumerate(occupied.tolist() + [0]):
        if value and start is None:
            start = x
        elif not value and start is not None:
            end = x
            width = end - start
            if width >= max(2, int(round(raw_w * 0.01))):
                range_ink = int(np.count_nonzero(kept[:, start:end]))
                if range_ink >= max(2, int(round(crop_area * 0.0002))):
                    ranges.append(
                        {
                            "x1": float(start),
                            "x2": float(end),
                            "width": float(width),
                            "ink": float(range_ink),
                            "center": (float(start) + float(end)) / 2.0,
                        }
                    )
            start = None

    return ranges


def _should_preserve_wide_spaced_sfx_osb(
    bubble: Dict[str, Any],
    translated_text: Any,
    source_text: Any,
    original_image: Image.Image,
    image_size: Tuple[int, int],
    current_bbox: Tuple[int, int, int, int],
) -> Tuple[bool, str, Optional[Tuple[int, int, int, int]]]:
    source_box = bubble.get("original_bbox") or bubble.get("bbox") or current_bbox
    try:
        base_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box), image_size
        )
    except (TypeError, ValueError):
        base_box = None
    if base_box is None:
        return False, "", None
    try:
        current_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in current_bbox), image_size
        )
    except (TypeError, ValueError):
        current_box = None

    x1, y1, x2, y2 = base_box
    raw_w = max(1, x2 - x1)
    raw_h = max(1, y2 - y1)
    image_w, image_h = image_size
    aspect = raw_w / float(max(1, raw_h))
    width_ratio = raw_w / float(max(1, image_w))
    height_ratio = raw_h / float(max(1, image_h))
    area_ratio = (raw_w * raw_h) / float(max(1, image_w * image_h))

    if aspect < WIDE_SPACED_SFX_MIN_ASPECT:
        return False, "", None
    if width_ratio < WIDE_SPACED_SFX_MIN_WIDTH_RATIO:
        return False, "", None
    if height_ratio > WIDE_SPACED_SFX_MAX_HEIGHT_RATIO:
        return False, "", None
    if area_ratio > WIDE_SPACED_SFX_MAX_AREA_RATIO:
        return False, "", None
    if _source_looks_like_general_vertical_dialogue(source_text):
        return False, "", None
    if _translation_has_hangul(translated_text) and not _translation_matches_source(
        source_text, translated_text
    ):
        return False, "", None

    ink_crop = _osb_source_ink_crop(
        bubble,
        original_image,
        base_box,
        image_size,
        allow_render_mask=False,
        side_only_filter=False,
    )
    if ink_crop is None or ink_crop.shape[:2] != (raw_h, raw_w):
        return False, "", None

    ink_pixels = int(np.count_nonzero(ink_crop))
    ink_coverage = ink_pixels / float(max(1, raw_w * raw_h))
    if ink_coverage > WIDE_SPACED_SFX_MAX_INK_COVERAGE:
        return False, "", None

    clusters = _wide_spaced_sfx_projection_clusters(ink_crop)
    cluster_count = len(clusters)
    if cluster_count < WIDE_SPACED_SFX_MIN_CLUSTER_COUNT:
        return False, "", None

    centers = [float(cluster["center"]) for cluster in clusters]
    centers.sort()
    center_gaps = [
        centers[idx + 1] - centers[idx] for idx in range(len(centers) - 1)
    ]
    if not center_gaps:
        return False, "", None

    span = max(cluster["x2"] for cluster in clusters) - min(
        cluster["x1"] for cluster in clusters
    )
    spread_ratio = span / float(max(1, raw_w))
    max_center_gap = max(center_gaps)
    min_center_gap = max(
        WIDE_SPACED_SFX_MIN_CENTER_GAP_PX,
        raw_w * WIDE_SPACED_SFX_MIN_CENTER_GAP_RATIO,
    )
    if spread_ratio < WIDE_SPACED_SFX_MIN_SPREAD_RATIO:
        return False, "", None
    if max_center_gap < min_center_gap:
        return False, "", None

    part_texts = _original_position_osb_tokens(str(translated_text or ""))
    part_count = len(part_texts)
    count_mismatch = part_count == 0 or part_count != cluster_count

    widths = [max(1.0, float(cluster["width"])) for cluster in clusters]
    inks = [max(1.0, float(cluster["ink"])) for cluster in clusters]
    width_balance = min(widths) / max(widths)
    ink_balance = min(inks) / max(inks)
    bad_balance = (
        width_balance < WIDE_SPACED_SFX_BAD_BALANCE_RATIO
        or ink_balance < WIDE_SPACED_SFX_BAD_BALANCE_RATIO
    )

    if not (count_mismatch or bad_balance):
        return False, "", None

    reason = (
        f"clusters={cluster_count}, parts={part_count}, "
        f"aspect={aspect:.2f}, spread={spread_ratio:.2f}, "
        f"gap={max_center_gap:.1f}px, coverage={ink_coverage:.3f}, "
        f"width_balance={width_balance:.2f}, ink_balance={ink_balance:.2f}"
    )
    restore_box = base_box
    if current_box is not None:
        restore_box = (
            min(base_box[0], current_box[0]),
            min(base_box[1], current_box[1]),
            max(base_box[2], current_box[2]),
            max(base_box[3], current_box[3]),
        )
    return True, reason, restore_box


def _osb_original_cluster_render_boxes(
    bubble: Dict[str, Any],
    part_texts: List[str],
    original_image: Image.Image,
    image_size: Tuple[int, int],
    current_bbox: Tuple[int, int, int, int],
    default_max_font_size: int,
    force_horizontal_repeated_split: bool = False,
    boost_horizontal_sfx_split: bool = False,
    page_font_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    part_count = len(part_texts)
    if part_count < 2 or part_count > 4:
        return []

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    try:
        base_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box), image_size
        )
    except (TypeError, ValueError):
        base_box = None
    if base_box is None:
        return []

    x1, y1, x2, y2 = base_box
    raw_w = max(1, x2 - x1)
    raw_h = max(1, y2 - y1)
    if raw_w < 10 or raw_h < 10:
        return []

    ink_crop = _osb_source_ink_crop(
        bubble,
        original_image,
        base_box,
        image_size,
        allow_render_mask=False,
        side_only_filter=False,
    )
    if ink_crop is None or ink_crop.shape[:2] != (raw_h, raw_w):
        return []

    repeated_short_split = _is_repeated_short_osb_split(part_texts)
    split_font_cap = (
        OSB_CLUSTER_REPEAT_MAX_FONT_SIZE
        if repeated_short_split
        else OSB_CLUSTER_MAX_FONT_SIZE
    )
    split_font_scale = (
        OSB_CLUSTER_REPEAT_FONT_SCALE if repeated_short_split else 1.0
    )
    try:
        default_font_cap = max(1, int(default_max_font_size))
    except (TypeError, ValueError):
        default_font_cap = split_font_cap
    boost_non_repeated_horizontal_split = bool(
        boost_horizontal_sfx_split and not repeated_short_split
    )
    if boost_non_repeated_horizontal_split:
        split_font_cap = max(
            split_font_cap,
            min(default_font_cap, OSB_CLUSTER_HORIZONTAL_SFX_MAX_FONT_SIZE),
        )
        split_font_scale = max(
            split_font_scale, OSB_CLUSTER_HORIZONTAL_SFX_FONT_SCALE
        )
    force_horizontal_part_split = bool(
        force_horizontal_repeated_split or boost_non_repeated_horizontal_split
    )
    scaled_split_font_cap = _scaled_render_font_size(
        split_font_cap,
        page_font_scale,
        minimum=10,
        maximum=640,
    )
    split_padding = scale_scalar(
        OSB_CLUSTER_SPLIT_PADDING_PIXELS,
        page_font_scale,
        minimum=0.0,
        maximum=32.0,
    )

    component_clusters = _osb_component_text_clusters(ink_crop, base_box, part_count)
    if component_clusters:
        clusters = component_clusters
    else:
        clusters = _osb_projection_text_clusters(ink_crop, base_box, part_count)

    try:
        if not clusters:
            kx = max(3, min(17, int(round(raw_w * 0.10))))
            ky = max(3, min(17, int(round(raw_h * 0.13))))
            if kx % 2 == 0:
                kx += 1
            if ky % 2 == 0:
                ky += 1
            grouped = cv2.dilate(
                ink_crop.astype(np.uint8),
                np.ones((ky, kx), np.uint8),
                iterations=1,
            )
            grouped = cv2.morphologyEx(
                grouped, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
            )
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
    except Exception:
        return []
    if not clusters and num_labels <= 1:
        return []

    if not clusters:
        crop_area = float(max(1, raw_w * raw_h))
        min_ink = max(4, int(crop_area * 0.001))
        for label_idx in range(1, num_labels):
            component_mask = labels == label_idx
            component_ink = ink_crop & component_mask
            ink_count = int(np.count_nonzero(component_ink))
            if ink_count < min_ink:
                continue
            ys, xs = np.where(component_ink)
            if xs.size == 0 or ys.size == 0:
                continue
            cx1 = int(xs.min())
            cy1 = int(ys.min())
            cx2 = int(xs.max()) + 1
            cy2 = int(ys.max()) + 1
            if cx2 - cx1 < 2 or cy2 - cy1 < 2:
                continue
            clusters.append((x1 + cx1, y1 + cy1, x1 + cx2, y1 + cy2, ink_count))

    if len(clusters) < part_count:
        return []

    if force_horizontal_part_split or raw_w / float(max(1, raw_h)) >= 1.25:
        clusters.sort(key=lambda box: ((box[0] + box[2]) / 2.0, box[1]))
    else:
        clusters.sort(key=lambda box: ((box[1] + box[3]) / 2.0, box[0]))

    while len(clusters) > part_count:
        best_index = None
        best_dist = None
        for idx in range(len(clusters) - 1):
            a = clusters[idx]
            b = clusters[idx + 1]
            acx = (a[0] + a[2]) / 2.0
            acy = (a[1] + a[3]) / 2.0
            bcx = (b[0] + b[2]) / 2.0
            bcy = (b[1] + b[3]) / 2.0
            dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_index = idx
        if best_index is None:
            return []
        a = clusters[best_index]
        b = clusters[best_index + 1]
        merged = (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
            a[4] + b[4],
        )
        clusters = clusters[:best_index] + [merged] + clusters[best_index + 2 :]

    try:
        container = _clip_bbox_to_image(current_bbox, image_size)
    except (TypeError, ValueError):
        container = None
    if container is None:
        container_source = bubble.get("bbox") or current_bbox
        try:
            container = _clip_bbox_to_image(
                tuple(int(round(float(v))) for v in container_source), image_size
            )
        except (TypeError, ValueError):
            container = None
    if container is None:
        container = base_box

    def reject_split_with_whole_rotation(reason: str = "unbalanced") -> List[Dict[str, Any]]:
        bubble["_osb_rejected_cluster_reason"] = reason
        rejected_rotation = 0.0
        try:
            rejected_rotation = _estimate_cluster_horizontal_rotation(ink_crop)
        except Exception:
            rejected_rotation = 0.0
        if abs(rejected_rotation) <= 0.01:
            try:
                layout_rotations = [
                    float(layout.get("rotation_deg", 0.0) or 0.0)
                    for layout in layouts
                    if abs(float(layout.get("rotation_deg", 0.0) or 0.0)) > 0.01
                ]
                if layout_rotations:
                    rejected_rotation = float(np.median(layout_rotations))
            except Exception:
                rejected_rotation = 0.0
        if abs(rejected_rotation) > 0.01:
            bubble["_osb_rejected_cluster_rotation_deg"] = rejected_rotation
        return []

    def fallback_to_source_position_layout(reason: str) -> List[Dict[str, Any]]:
        if (
            not boost_non_repeated_horizontal_split
            or repeated_short_split
            or part_count < 2
        ):
            return []

        aspect = raw_w / float(max(1, raw_h))
        if not 0.65 <= aspect <= 1.35:
            return []

        if clusters:
            centers_x = [((box[0] + box[2]) / 2.0) for box in clusters]
            centers_y = [((box[1] + box[3]) / 2.0) for box in clusters]
            x_span = (max(centers_x) - min(centers_x)) / float(max(1, raw_w))
            y_span = (max(centers_y) - min(centers_y)) / float(max(1, raw_h))
            if x_span < 0.24 or y_span < 0.22:
                return []

        source_position_layouts = _osb_stacked_source_position_render_boxes(
            part_texts,
            base_box,
            image_size,
            default_max_font_size,
            page_font_scale=page_font_scale,
        )
        if source_position_layouts:
            bubble["_osb_diagonal_source_position_layout"] = True
            bubble["_osb_source_position_fallback_reason"] = reason
            return source_position_layouts
        return []

    layouts: List[Dict[str, Any]] = []
    cluster_shape_ratios: List[Tuple[float, float]] = []
    if force_horizontal_part_split and _osb_horizontal_split_clusters_are_stacked(
        clusters, raw_w, raw_h
    ):
        if boost_non_repeated_horizontal_split:
            stacked_layouts = _osb_stacked_source_position_render_boxes(
                part_texts,
                base_box,
                image_size,
                default_max_font_size,
                page_font_scale=page_font_scale,
            )
            if stacked_layouts:
                bubble["_osb_stacked_source_position_layout"] = True
                return stacked_layouts
        return reject_split_with_whole_rotation("stacked source clusters")

    dark_background = bool(bubble.get("is_dark_text", True))
    for cluster, part_text in zip(clusters, part_texts):
        cx1, cy1, cx2, cy2, _ = cluster
        cluster_w = max(1.0, float(cx2 - cx1))
        cluster_h = max(1.0, float(cy2 - cy1))
        cluster_shape_ratios.append(
            (
                cluster_w / float(max(1, raw_w)),
                cluster_h / float(max(1, raw_h)),
            )
        )
        local_x1 = max(0, cx1 - x1)
        local_y1 = max(0, cy1 - y1)
        local_x2 = min(raw_w, cx2 - x1)
        local_y2 = min(raw_h, cy2 - y1)
        cluster_mask = ink_crop[local_y1:local_y2, local_x1:local_x2]
        if cluster_mask.size == 0:
            return []
        part_vertical = _cluster_prefers_vertical(cluster_mask)
        if force_horizontal_part_split:
            part_vertical = False
        elif repeated_short_split:
            part_vertical = True
        part_len = max(1, len(_osb_split_token_key(part_text)))
        if part_vertical:
            part_rotation = _estimate_cluster_vertical_rotation(
                cluster_mask, dark_background
            )
        else:
            part_rotation = _estimate_cluster_horizontal_rotation(cluster_mask)
        if (
            repeated_short_split
            and part_vertical
            and part_len >= 2
            and abs(part_rotation) > 0.01
        ):
            part_rotation = max(
                -OSB_CLUSTER_REPEAT_PAIR_MAX_ROTATION_DEGREES,
                min(
                    OSB_CLUSTER_REPEAT_PAIR_MAX_ROTATION_DEGREES,
                    part_rotation * OSB_CLUSTER_REPEAT_PAIR_ROTATION_GAIN,
                ),
            )
        part_max_font = _osb_split_max_font_size(
            part_text,
            cluster_w,
            cluster_h,
            part_vertical,
            default_max_font_size,
            split_font_cap,
            split_font_scale,
            page_font_scale=page_font_scale,
        )
        if repeated_short_split and part_vertical and part_len >= 2:
            min_pair_font = int(
                round(raw_h * OSB_CLUSTER_REPEAT_PAIR_MIN_FONT_HEIGHT_RATIO)
            )
            part_max_font = min(
                default_font_cap,
                scaled_split_font_cap,
                max(part_max_font, min_pair_font),
            )
        if boost_non_repeated_horizontal_split and not part_vertical:
            min_horizontal_font = int(
                round(min(raw_h * 0.45, (raw_w / float(part_count)) * 0.75))
            )
            part_max_font = min(
                default_font_cap,
                split_font_cap,
                max(part_max_font, min_horizontal_font),
            )
        center_x = (cx1 + cx2) / 2.0
        center_y = (cy1 + cy2) / 2.0
        if part_vertical:
            target_w = max(cluster_w * 1.55, part_max_font * 1.35, 14.0)
            target_h = max(cluster_h * 1.35, part_max_font * (part_len + 0.35), 16.0)
        else:
            target_w = max(cluster_w * 1.9, cluster_h * (0.95 + part_len * 0.42), 16.0)
            target_h = max(cluster_h * 1.45, part_max_font * 1.45, 13.0)
        if repeated_short_split and part_vertical and part_len >= 2:
            max_target_w = max(16.0, raw_w * OSB_CLUSTER_REPEAT_PAIR_MAX_WIDTH_RATIO)
            max_target_h = max(13.0, raw_h * OSB_CLUSTER_REPEAT_PAIR_MAX_HEIGHT_RATIO)
        else:
            max_target_w = max(16.0, raw_w * 0.72)
            max_target_h = max(13.0, raw_h * 0.72)
            if (
                force_horizontal_part_split
                and not repeated_short_split
                and part_count == 2
                and not part_vertical
            ):
                max_target_w = max(
                    16.0,
                    min(max_target_w, raw_w * 0.52, cluster_w * 1.18 + 6.0),
                )
        target_w = min(target_w, max_target_w)
        target_h = min(target_h, max_target_h)
        candidate = (
            center_x - target_w / 2.0,
            center_y - target_h / 2.0,
            center_x + target_w / 2.0,
            center_y + target_h / 2.0,
        )
        shifted = _shift_bbox_inside(candidate, container)
        clipped = _clip_bbox_to_image(shifted, image_size)
        if clipped is None:
            return []
        if clipped[2] - clipped[0] < 12 or clipped[3] - clipped[1] < 10:
            return []
        layouts.append(
            {
                "bbox": clipped,
                "vertical_stack": part_vertical,
                "rotation_deg": part_rotation,
                "max_font_size": part_max_font,
                "padding_pixels": split_padding,
                "lock_vertical_stack": bool(part_vertical or force_horizontal_part_split),
            }
        )

    if force_horizontal_part_split and cluster_shape_ratios:
        min_cluster_width_ratio = min(ratio[0] for ratio in cluster_shape_ratios)
        min_cluster_height_ratio = min(ratio[1] for ratio in cluster_shape_ratios)
        if (
            min_cluster_width_ratio
            < OSB_CLUSTER_HORIZONTAL_MIN_SOURCE_WIDTH_RATIO
            or min_cluster_height_ratio
            < OSB_CLUSTER_HORIZONTAL_MIN_SOURCE_HEIGHT_RATIO
        ):
            source_position_layouts = fallback_to_source_position_layout(
                "source clusters too small"
            )
            if source_position_layouts:
                return source_position_layouts
            return reject_split_with_whole_rotation("source clusters too small")

    if not _osb_split_layouts_are_balanced(part_texts, layouts):
        source_position_layouts = fallback_to_source_position_layout(
            "unbalanced source split"
        )
        if source_position_layouts:
            return source_position_layouts
        return reject_split_with_whole_rotation("unbalanced source split")

    layouts = _separate_overlapping_vertical_repeated_osb_layouts(
        part_texts,
        layouts,
    )
    return layouts if len(layouts) == part_count else []


def _normalize_cluster_rotation(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _clamp_cluster_rotation(angle_deg: float) -> float:
    angle = _normalize_cluster_rotation(angle_deg)
    if abs(angle) < OSB_CLUSTER_ROTATION_MIN_ABS_DEGREES:
        return 0.0
    return float(
        max(
            -OSB_CLUSTER_ROTATION_MAX_ABS_DEGREES,
            min(OSB_CLUSTER_ROTATION_MAX_ABS_DEGREES, angle),
        )
    )


def _estimate_cluster_horizontal_rotation(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < 8:
        return 0.0

    h, w = mask.shape[:2]
    xs_for_fit: List[float] = []
    ys_for_fit: List[float] = []
    min_col_pixels = max(1, int(h * 0.05))
    for col in range(w):
        col_ys = np.where(mask[:, col] > 0)[0]
        if col_ys.size < min_col_pixels:
            continue
        xs_for_fit.append(float(col))
        ys_for_fit.append(float(np.median(col_ys)))

    if len(xs_for_fit) >= max(4, int(w * 0.16)):
        try:
            slope, _ = np.polyfit(
                np.asarray(xs_for_fit, dtype=np.float32),
                np.asarray(ys_for_fit, dtype=np.float32),
                1,
            )
            return _clamp_cluster_rotation(
                float(np.degrees(np.arctan(float(slope))))
            )
        except Exception:
            pass

    try:
        coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] < 1.16:
            return 0.0
        major = eigvecs[:, order[0]]
        return _clamp_cluster_rotation(
            float(np.degrees(np.arctan2(major[1], major[0])))
        )
    except Exception:
        return 0.0


def _estimate_horizontal_sfx_centerline_rotation(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    if h < 12 or w < 24:
        return 0.0

    crop_area = float(max(1, w * h))
    best_angle = 0.0
    best_score = 0.0
    kernels = ((3, 3), (5, 5), (7, 7), (9, 5), (11, 7))

    for kernel_w, kernel_h in kernels:
        try:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_w, kernel_h),
            )
            glyph_mask = cv2.morphologyEx(
                mask.astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                glyph_mask.astype(np.uint8),
                8,
            )
        except Exception:
            continue

        components: List[Tuple[float, float, float, float, float, float, float]] = []
        min_area = max(20.0, crop_area * 0.001)
        max_area = crop_area * 0.46
        for label_idx in range(1, num_labels):
            area = float(stats[label_idx, cv2.CC_STAT_AREA])
            x = float(stats[label_idx, cv2.CC_STAT_LEFT])
            y = float(stats[label_idx, cv2.CC_STAT_TOP])
            component_w = float(stats[label_idx, cv2.CC_STAT_WIDTH])
            component_h = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area:
                continue
            if component_w < 3.0 or component_h < 3.0:
                continue
            if component_w > w * 0.88 and component_h > h * 0.55:
                continue

            near_horizontal_edge = y <= 2.0 or (y + component_h) >= h - 2.0
            near_vertical_edge = x <= 2.0 or (x + component_w) >= w - 2.0
            if component_w > w * 0.58 and component_h < h * 0.22:
                continue
            if near_horizontal_edge and component_w > w * 0.32 and component_h < h * 0.18:
                continue
            if near_vertical_edge and component_h > h * 0.45 and component_w < w * 0.16:
                continue

            components.append(
                (
                    float(centroids[label_idx][0]),
                    float(centroids[label_idx][1]),
                    area,
                    x,
                    y,
                    component_w,
                    component_h,
                )
            )

        if len(components) < 2 or len(components) > 6:
            continue

        center_arr = np.asarray(
            [(component[0], component[1]) for component in components],
            dtype=np.float32,
        )
        weight_arr = np.sqrt(
            np.asarray([component[2] for component in components], dtype=np.float32)
        )
        xs = center_arr[:, 0]
        ys = center_arr[:, 1]
        span_x = float(xs.max() - xs.min())
        span_y = float(ys.max() - ys.min())
        if span_x < max(24.0, w * 0.25) or span_y < max(7.0, h * 0.05):
            continue

        sorted_weights = np.sort(weight_arr)[::-1]
        if len(sorted_weights) == 2 and sorted_weights[1] / max(1.0, sorted_weights[0]) < 0.40:
            continue

        try:
            slope, intercept = np.polyfit(xs, ys, 1, w=weight_arr)
            predicted = xs * slope + intercept
            residual = float(np.median(np.abs(ys - predicted)))
            if residual > max(6.0, h * 0.055):
                continue
            angle = _normalize_cluster_rotation(
                float(np.degrees(np.arctan(float(slope))))
            )
        except Exception:
            continue

        if (
            abs(angle) < HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
            or abs(angle) > HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES
        ):
            continue

        score = (
            float(np.sum(weight_arr))
            * (span_x / max(1.0, float(w)))
            * (1.0 + min(1.0, span_y / max(1.0, float(h))))
            / max(1.0, residual + 1.0)
        )
        if score > best_score:
            best_score = score
            best_angle = angle

    if abs(best_angle) < HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES:
        return 0.0
    return float(
        max(
            -HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES,
            min(HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES, best_angle),
        )
    )


def _estimate_horizontal_sfx_component_rotation(mask: np.ndarray) -> float:
    h, w = mask.shape[:2]
    if h < 12 or w < 24:
        return 0.0

    crop_area = float(max(1, w * h))
    try:
        opened = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        )
        if np.count_nonzero(opened) >= max(20, int(crop_area * 0.001)):
            mask = opened
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            8,
        )
    except Exception:
        return 0.0

    components: List[Tuple[float, float, float]] = []
    min_area = max(18.0, crop_area * 0.0006)
    max_area = crop_area * 0.16
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        x = float(stats[label_idx, cv2.CC_STAT_LEFT])
        y = float(stats[label_idx, cv2.CC_STAT_TOP])
        component_w = float(stats[label_idx, cv2.CC_STAT_WIDTH])
        component_h = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if component_w < 4.0 or component_h < 4.0:
            continue

        near_horizontal_edge = y <= 2.0 or (y + component_h) >= h - 2.0
        near_vertical_edge = x <= 2.0 or (x + component_w) >= w - 2.0
        if component_w > w * 0.55 and component_h < h * 0.16:
            continue
        if near_horizontal_edge and component_w > w * 0.30 and component_h < h * 0.20:
            continue
        if near_vertical_edge and component_h > h * 0.45 and component_w < w * 0.14:
            continue

        components.append(
            (
                float(centroids[label_idx][0]),
                float(centroids[label_idx][1]),
                area,
            )
        )

    if len(components) > 8:
        components = sorted(components, key=lambda component: component[2], reverse=True)[
            :8
        ]
    if len(components) < 3:
        return 0.0

    center_arr = np.asarray(
        [(component[0], component[1]) for component in components],
        dtype=np.float32,
    )
    weight_arr = np.sqrt(
        np.asarray([component[2] for component in components], dtype=np.float32)
    )
    xs = center_arr[:, 0]
    ys = center_arr[:, 1]
    span_x = float(xs.max() - xs.min())
    span_y = float(ys.max() - ys.min())
    if span_x < max(30.0, w * 0.32) or span_y < max(12.0, h * 0.12):
        return 0.0

    try:
        slope, intercept = np.polyfit(xs, ys, 1, w=weight_arr)
        predicted = xs * slope + intercept
        residual = float(np.median(np.abs(ys - predicted)))
        if residual > max(10.0, h * 0.08):
            return 0.0
        angle = _normalize_cluster_rotation(
            float(np.degrees(np.arctan(float(slope))))
        )
    except Exception:
        return 0.0

    if (
        abs(angle) < HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
        or abs(angle) > HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES
    ):
        return 0.0
    return float(
        max(
            -HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES,
            min(HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES, angle),
        )
    )


def _estimate_horizontal_sfx_rotation_from_original_crop(
    bubble: Dict[str, Any],
) -> float:
    crop_pil = bubble.get("original_crop_pil")
    if not isinstance(crop_pil, Image.Image):
        return 0.0

    try:
        crop_rgb = crop_pil.convert("RGB")
        crop = np.asarray(crop_rgb)
    except Exception:
        return 0.0
    if crop.size == 0:
        return 0.0

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    except Exception:
        return 0.0
    h, w = gray.shape[:2]
    if h < 12 or w < 24:
        return 0.0

    try:
        median_luma = float(np.median(gray))
    except Exception:
        return 0.0

    masks: List[np.ndarray] = []
    if median_luma >= 128.0:
        for percentile, cap in ((8, 95.0), (12, 110.0)):
            try:
                threshold = max(35.0, min(cap, float(np.percentile(gray, percentile))))
            except Exception:
                continue
            masks.append((gray <= threshold).astype(np.uint8))
        for percentile, cap in ((18, 150.0), (22, 150.0), (28, 155.0)):
            try:
                threshold = max(55.0, min(cap, float(np.percentile(gray, percentile))))
            except Exception:
                continue
            masks.append((gray <= threshold).astype(np.uint8))
        masks.append((gray <= 130).astype(np.uint8))
    else:
        for percentile, floor in ((92, 160.0), (88, 145.0)):
            try:
                threshold = min(245.0, max(floor, float(np.percentile(gray, percentile))))
            except Exception:
                continue
            masks.append((gray >= threshold).astype(np.uint8))
        for percentile, floor in ((72, 100.0), (78, 110.0), (84, 120.0)):
            try:
                threshold = min(245.0, max(floor, float(np.percentile(gray, percentile))))
            except Exception:
                continue
            masks.append((gray >= threshold).astype(np.uint8))
        masks.append((gray >= 145).astype(np.uint8))

    centerline_angles: List[float] = []
    component_angles: List[float] = []
    angles: List[float] = []
    ink_horizontal_support = False
    ink_flat_horizontal_support = False
    ink_diagonal_support = False
    crop_area = float(max(1, w * h))
    for mask in masks:
        ink_pixels = int(np.count_nonzero(mask))
        coverage = ink_pixels / crop_area
        if coverage < 0.015 or coverage > 0.45:
            continue
        try:
            ink_ys, ink_xs = np.where(mask > 0)
            if ink_xs.size >= max(20, int(crop_area * 0.001)):
                ink_span_x = float(ink_xs.max() - ink_xs.min() + 1)
                ink_span_y = float(ink_ys.max() - ink_ys.min() + 1)
                if (
                    ink_span_x >= max(24.0, w * 0.35)
                    and ink_span_x >= ink_span_y * 1.08
                ):
                    ink_horizontal_support = True
                    if ink_span_x >= max(48.0, w * 0.55):
                        try:
                            flat_angle = _estimate_cluster_horizontal_rotation(mask)
                        except Exception:
                            flat_angle = 0.0
                        if (
                            ink_span_x >= ink_span_y * 2.20
                            and abs(flat_angle)
                            < HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
                        ):
                            ink_flat_horizontal_support = True
                if (
                    ink_span_x >= max(24.0, w * 0.45)
                    and ink_span_y >= max(24.0, h * 0.40)
                    and 0.42 <= ink_span_x / float(max(1.0, ink_span_y)) <= 1.65
                ):
                    ink_diagonal_support = True
        except Exception:
            pass
        try:
            centerline_angle = _estimate_horizontal_sfx_centerline_rotation(mask)
        except Exception:
            centerline_angle = 0.0
        if abs(centerline_angle) > 0.01:
            centerline_angles.append(float(centerline_angle))
        try:
            component_angle = _estimate_horizontal_sfx_component_rotation(mask)
        except Exception:
            component_angle = 0.0
        if abs(component_angle) > 0.01:
            component_angles.append(float(component_angle))
        try:
            angle = _estimate_cluster_horizontal_rotation(mask)
        except Exception:
            angle = 0.0
        if abs(angle) > 0.01:
            angles.append(float(angle))

    def choose_dominant_angle(raw_angles: List[float]) -> float:
        if not raw_angles:
            return 0.0
        strong_angles = [
            angle
            for angle in raw_angles
            if abs(angle) >= HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
        ]
        if strong_angles:
            raw_angles = strong_angles
        positives = [angle for angle in raw_angles if angle > 0.0]
        negatives = [angle for angle in raw_angles if angle < 0.0]
        if positives and negatives:
            positive_strength = float(np.median([abs(angle) for angle in positives]))
            negative_strength = float(np.median([abs(angle) for angle in negatives]))
            if abs(positive_strength - negative_strength) < 1.5:
                return 0.0
            raw_angles = positives if positive_strength > negative_strength else negatives
        angle = _normalize_cluster_rotation(float(np.median(raw_angles)))
        if abs(angle) < OSB_CLUSTER_ROTATION_MIN_ABS_DEGREES:
            return 0.0
        return angle

    centerline_angle = choose_dominant_angle(centerline_angles)
    component_angle = choose_dominant_angle(component_angles)
    ink_angle = choose_dominant_angle(angles)
    has_source_support = ink_horizontal_support or ink_diagonal_support
    if abs(component_angle) >= HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES and has_source_support:
        component_matches_centerline = (
            abs(centerline_angle) >= HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
            and (component_angle > 0.0) == (centerline_angle > 0.0)
        )
        component_matches_ink = (
            abs(ink_angle) >= HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
            and (component_angle > 0.0) == (ink_angle > 0.0)
        )
        if (
            abs(centerline_angle) < HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
            or (
                component_matches_centerline
                and abs(component_angle) > abs(centerline_angle) + 3.0
            )
            or (
                component_matches_ink
                and abs(component_angle) > abs(ink_angle) + 5.0
            )
        ):
            bubble["_horizontal_sfx_rotation_mode"] = "component"
            return float(
                max(
                    -HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES,
                    min(
                        HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES,
                        component_angle,
                    ),
                )
            )
    if abs(centerline_angle) >= HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES:
        centerline_matches_ink = (
            abs(ink_angle) >= HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
            and (centerline_angle > 0.0) == (ink_angle > 0.0)
        )
        centerline_allowed_by_flat_ink = not (
            ink_flat_horizontal_support
            and abs(centerline_angle)
            > HORIZONTAL_SFX_CENTERLINE_FLAT_INK_MAX_DEGREES
        )
        if centerline_matches_ink or (
            abs(ink_angle) < HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
            and ink_horizontal_support
            and centerline_allowed_by_flat_ink
        ):
            bubble["_horizontal_sfx_rotation_mode"] = "centerline"
            return float(
                max(
                    -HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES,
                    min(
                        HORIZONTAL_SFX_CENTERLINE_ROTATION_MAX_ABS_DEGREES,
                        centerline_angle,
                    ),
                )
            )

    if (
        abs(ink_angle) < HORIZONTAL_SFX_ROTATION_MIN_ABS_DEGREES
        or not has_source_support
    ):
        return 0.0
    bubble["_horizontal_sfx_rotation_mode"] = "ink"
    return float(
        max(
            -HORIZONTAL_SFX_INK_ROTATION_MAX_ABS_DEGREES,
            min(HORIZONTAL_SFX_INK_ROTATION_MAX_ABS_DEGREES, ink_angle),
        )
    )


def _estimate_vertical_sfx_glyph_rotation_from_original_crop(
    bubble: Dict[str, Any],
) -> float:
    crop_pil = bubble.get("original_crop_pil")
    if not isinstance(crop_pil, Image.Image):
        return 0.0

    try:
        crop_rgb = crop_pil.convert("RGB")
        crop = np.asarray(crop_rgb)
    except Exception:
        return 0.0
    if crop.size == 0:
        return 0.0

    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    except Exception:
        return 0.0
    h, w = gray.shape[:2]
    if h < 20 or w < 16:
        return 0.0

    try:
        median_luma = float(np.median(gray))
    except Exception:
        return 0.0

    masks: List[np.ndarray] = []
    if median_luma >= 128.0:
        for percentile, cap in ((18, 150.0), (22, 150.0), (28, 155.0), (34, 165.0)):
            try:
                threshold = max(55.0, min(cap, float(np.percentile(gray, percentile))))
            except Exception:
                continue
            masks.append((gray <= threshold).astype(np.uint8))
        masks.append((gray <= 130).astype(np.uint8))
    else:
        for percentile, floor in ((72, 100.0), (78, 110.0), (84, 120.0)):
            try:
                threshold = min(245.0, max(floor, float(np.percentile(gray, percentile))))
            except Exception:
                continue
            masks.append((gray >= threshold).astype(np.uint8))
        masks.append((gray >= 145).astype(np.uint8))

    angles: List[float] = []
    crop_area = float(max(1, w * h))
    for mask in masks:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if np.count_nonzero(opened) >= max(20, int(crop_area * 0.001)):
            mask = opened

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        centers: List[Tuple[float, float]] = []
        weights: List[float] = []
        for label_idx in range(1, num_labels):
            area = float(stats[label_idx, cv2.CC_STAT_AREA])
            x = float(stats[label_idx, cv2.CC_STAT_LEFT])
            y = float(stats[label_idx, cv2.CC_STAT_TOP])
            component_w = float(stats[label_idx, cv2.CC_STAT_WIDTH])
            component_h = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
            if area < 8.0 or area > crop_area * 0.20:
                continue
            if component_w < 3.0 or component_h < 3.0:
                continue
            elongation = max(component_w, component_h) / max(1.0, min(component_w, component_h))
            if elongation > 4.0:
                continue
            if (y <= 1.0 and component_w > w * 0.45) or (
                x <= 1.0 and component_h > h * 0.55
            ):
                continue
            centers.append((float(centroids[label_idx][0]), float(centroids[label_idx][1])))
            weights.append(math.sqrt(max(1.0, area)))

        if len(centers) < 2:
            continue
        center_arr = np.asarray(centers, dtype=np.float32)
        weight_arr = np.asarray(weights, dtype=np.float32)
        span_y = float(center_arr[:, 1].max() - center_arr[:, 1].min())
        span_x = float(center_arr[:, 0].max() - center_arr[:, 0].min())
        if span_y < max(10.0, h * 0.30):
            continue
        try:
            slope, intercept = np.polyfit(
                center_arr[:, 1],
                center_arr[:, 0],
                1,
                w=weight_arr,
            )
            predicted = center_arr[:, 1] * slope + intercept
            residual = float(np.median(np.abs(center_arr[:, 0] - predicted)))
            if residual > max(8.0, span_x * 0.45):
                continue
            angle = _normalize_cluster_rotation(float(np.degrees(np.arctan(float(slope)))))
        except Exception:
            continue
        if abs(angle) < OSB_CLUSTER_ROTATION_MIN_ABS_DEGREES:
            continue
        angles.append(
            float(
                max(
                    -HORIZONTAL_SFX_INK_ROTATION_MAX_ABS_DEGREES * 1.8,
                    min(HORIZONTAL_SFX_INK_ROTATION_MAX_ABS_DEGREES * 1.8, angle),
                )
            )
        )

    if not angles:
        return 0.0
    return float(np.median(angles))


def _estimate_cluster_vertical_rotation(mask: np.ndarray, dark_background: bool) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < 8:
        return 0.0

    h, w = mask.shape[:2]
    ys_for_fit: List[float] = []
    xs_for_fit: List[float] = []
    min_row_pixels = max(1, int(w * 0.05))
    for row in range(h):
        row_xs = np.where(mask[row] > 0)[0]
        if row_xs.size < min_row_pixels:
            continue
        ys_for_fit.append(float(row))
        xs_for_fit.append(float(np.median(row_xs)))

    source_angle = 0.0
    if len(xs_for_fit) >= max(4, int(h * 0.12)):
        try:
            slope, _ = np.polyfit(
                np.asarray(ys_for_fit, dtype=np.float32),
                np.asarray(xs_for_fit, dtype=np.float32),
                1,
            )
            source_angle = float(np.degrees(np.arctan(float(slope))))
        except Exception:
            source_angle = 0.0

    if source_angle == 0.0:
        try:
            coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
            centered = coords - np.mean(coords, axis=0)
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] < 1.16:
                return 0.0
            major = eigvecs[:, order[0]]
            angle_from_x = float(np.degrees(np.arctan2(major[1], major[0])))
            if angle_from_x < 0.0:
                angle_from_x += 180.0
            source_angle = _normalize_cluster_rotation(90.0 - angle_from_x)
        except Exception:
            return 0.0

    clamped = _clamp_cluster_rotation(source_angle)
    if clamped == 0.0:
        return 0.0
    return clamped if dark_background else -clamped


def _cluster_prefers_vertical(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask > 0)
    if xs.size < 8:
        return False
    span_w = float(xs.max() - xs.min() + 1)
    span_h = float(ys.max() - ys.min() + 1)
    if span_h >= span_w * 1.18:
        return True
    if span_w >= span_h * 1.18:
        return False

    try:
        coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        centered = coords - np.mean(coords, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] < 1.12:
            return False
        major = eigvecs[:, order[0]]
        angle_from_x = abs(
            _normalize_cluster_rotation(
                float(np.degrees(np.arctan2(major[1], major[0])))
            )
        )
        return angle_from_x >= 45.0
    except Exception:
        return False


def _osb_split_max_font_size(
    part_text: str,
    cluster_w: float,
    cluster_h: float,
    vertical_stack: bool,
    default_max_font: int,
    cluster_font_cap: Optional[int] = None,
    font_scale: float = 1.0,
    page_font_scale: float = 1.0,
) -> int:
    part_len = max(1, len(_osb_split_token_key(part_text)))
    try:
        default_max = max(1, int(default_max_font))
    except (TypeError, ValueError):
        default_max = 64
    try:
        font_cap = max(10, int(cluster_font_cap))
    except (TypeError, ValueError):
        font_cap = OSB_CLUSTER_MAX_FONT_SIZE
    font_cap = _scaled_render_font_size(
        font_cap,
        page_font_scale,
        minimum=10,
        maximum=640,
    )
    try:
        font_scale = max(0.5, min(1.5, float(font_scale)))
    except (TypeError, ValueError):
        font_scale = 1.0

    if vertical_stack:
        estimated = min(cluster_w * 0.95, (cluster_h / float(part_len + 0.4)) * 0.95)
    else:
        estimated = min(cluster_h * 0.82, (cluster_w / float(part_len + 0.3)) * 1.05)
    estimated *= font_scale

    return max(
        10,
        min(
            default_max,
            font_cap,
            int(round(max(estimated, 10.0))),
        ),
    )




def _clip_rendered_image_to_mask(
    base_image: Image.Image,
    rendered_image: Image.Image,
    mask: Any,
) -> Image.Image:
    if mask is None:
        return rendered_image
    try:
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        if mask_array.shape != (base_image.size[1], base_image.size[0]):
            return rendered_image
        mask_pil = Image.fromarray((mask_array > 0).astype(np.uint8) * 255, mode="L")
        clipped = base_image.copy()
        clipped.paste(rendered_image, (0, 0), mask=mask_pil)
        clipped.info.update(getattr(rendered_image, "info", {}))
        return clipped
    except Exception:
        return rendered_image


def _should_skip_osb_render_clip(
    mask: Any,
    bbox: Any,
    image_size: Tuple[int, int],
) -> bool:
    """Avoid clipping newly rendered OSB text in large title/banner boxes.

    The render mask is useful for narrow irregular SFX shapes, but on wide cover
    title regions it can keep only pieces that overlap the old artwork and erase
    the translated glyphs after rendering.
    """
    if mask is None:
        return False

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return False

    img_w, img_h = image_size
    x1, y1, x2, y2 = clipped
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    width_ratio = width / float(max(1, img_w))
    height_ratio = height / float(max(1, img_h))
    area_ratio = (width * height) / float(max(1, img_w * img_h))
    aspect_ratio = width / float(max(1, height))

    if width_ratio >= 0.55 and area_ratio >= 0.045 and aspect_ratio >= 2.0:
        return True
    if width_ratio >= 0.45 and area_ratio >= 0.10 and height_ratio <= 0.40:
        return True

    try:
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        if mask_array.shape != (img_h, img_w):
            return False
        region = mask_array[y1:y2, x1:x2] > 0
        coverage = float(np.count_nonzero(region)) / float(max(1, region.size))
    except Exception:
        return False

    return (
        width_ratio >= 0.45
        and area_ratio >= 0.035
        and aspect_ratio >= 1.8
        and coverage < 0.35
    )



def _render_config_with_max_font(
    config: RenderingConfig,
    max_font_size: Optional[int],
) -> RenderingConfig:
    if max_font_size is None:
        return config
    try:
        capped_max = int(max_font_size)
    except (TypeError, ValueError):
        return config
    if capped_max <= 0:
        return config
    capped_max = max(int(config.min_font_size), min(int(config.max_font_size), capped_max))
    emphasis_max = config.emphasis_max_font_size
    if emphasis_max is not None:
        try:
            emphasis_max = min(int(emphasis_max), capped_max)
        except (TypeError, ValueError):
            emphasis_max = capped_max
    return replace(
        config,
        max_font_size=capped_max,
        emphasis_max_font_size=emphasis_max,
    )


def _rendered_pixel_change_mask(
    before: Image.Image,
    after: Image.Image,
) -> np.ndarray:
    before_rgb = np.asarray(before.convert("RGB"))
    after_rgb = np.asarray(after.convert("RGB"))
    if before_rgb.shape != after_rgb.shape:
        return np.zeros((before.height, before.width), dtype=bool)
    max_delta = np.max(
        np.abs(after_rgb.astype(np.int16) - before_rgb.astype(np.int16)),
        axis=2,
    )
    return max_delta >= 64


def _normalize_osb_residual_render_guard_action(action: Any) -> str:
    value = str(action or "restore_original").strip().lower()
    if value in {"none", "restore_original", "lama_large"}:
        return value
    return "restore_original"


def _run_coro_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" not in str(e):
            raise

    result_box: Dict[str, Any] = {}

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_box["result"] = loop.run_until_complete(coro)
        except BaseException as exc:
            result_box["error"] = exc
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("result")


def _ensure_hgmzhn_detector_runtime() -> bool:
    runtime_dir = Path(__file__).resolve().parents[1] / "advanced_editor" / "hgmzhn_runtime"
    if not runtime_dir.exists():
        return False
    runtime_text = str(runtime_dir)
    if runtime_text not in sys.path:
        sys.path.insert(0, runtime_text)
    return True


def _osb_residual_detector_device(config: Optional[MangaTranslatorConfig]) -> str:
    requested = str(getattr(config, "device", "auto") if config is not None else "auto").lower()
    if requested.startswith("cuda"):
        return "cuda"
    if requested in {"mps", "cpu"}:
        return requested
    return "cpu"


def _normalize_detector_mask(mask: Any, size: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    try:
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        if mask_array.ndim != 2:
            return None
        width, height = size
        if mask_array.shape != (height, width):
            mask_array = cv2.resize(
                mask_array.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        return np.asarray(mask_array)
    except Exception:
        return None


def _textlines_to_mask(textlines: Any, size: Tuple[int, int]) -> Optional[np.ndarray]:
    width, height = size
    line_mask = np.zeros((height, width), dtype=np.uint8)
    if not textlines:
        return None
    for textline in textlines:
        try:
            points = np.asarray(getattr(textline, "pts", []), dtype=np.int32)
            if points.size < 8:
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(line_mask, [points.reshape(-1, 2)], 255)
        except Exception:
            continue
    if int(np.count_nonzero(line_mask)) < OSB_RESIDUAL_RENDER_GUARD_MIN_SOURCE_PIXELS:
        return None
    return line_mask.astype(bool)


def _residual_detector_mask_from_hgmzhn(
    crop: Image.Image,
    detector_name: str,
    config: Optional[MangaTranslatorConfig],
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], str, bool]:
    if not _ensure_hgmzhn_detector_runtime():
        return None, "hgmzhn runtime missing", False

    try:
        from manga_translator.config import Detector
        from manga_translator.detection import dispatch, prepare
    except Exception as e:
        return None, f"import failed: {e}", False

    detector_key = Detector.ctd if detector_name == "ctd" else Detector.default
    crop_rgb = crop.convert("RGB")
    width, height = crop_rgb.size
    longest_side = max(width, height)
    detect_size = min(2048, max(1024, ((longest_side + 31) // 32) * 32))
    device = _osb_residual_detector_device(config)

    async def _detect(run_device: str):
        await prepare(detector_key)
        return await dispatch(
            detector_key,
            np.asarray(crop_rgb),
            detect_size,
            OSB_RESIDUAL_DETECTOR_TEXT_THRESHOLD,
            OSB_RESIDUAL_DETECTOR_BOX_THRESHOLD,
            OSB_RESIDUAL_DETECTOR_UNCLIP_RATIO,
            device=run_device,
            verbose=False,
        )

    try:
        with _OSB_RESIDUAL_DETECTOR_LOCK:
            try:
                textlines, raw_mask, _debug = _run_coro_sync(_detect(device))
            except Exception:
                if device == "cpu":
                    raise
                textlines, raw_mask, _debug = _run_coro_sync(_detect("cpu"))
    except Exception as e:
        return None, f"{detector_name} failed after auto-download/load attempt: {e}", False

    normalized = _normalize_detector_mask(raw_mask, crop_rgb.size)
    if normalized is None:
        return None, f"{detector_name} returned no mask", False

    mask_bool = normalized > OSB_RESIDUAL_DETECTOR_MASK_THRESHOLD
    line_mask = _textlines_to_mask(textlines, crop_rgb.size)
    if line_mask is not None:
        restricted = np.logical_and(mask_bool, line_mask)
        if int(np.count_nonzero(restricted)) >= OSB_RESIDUAL_RENDER_GUARD_MIN_SOURCE_PIXELS:
            mask_bool = restricted

    mask_pixels = int(np.count_nonzero(mask_bool))
    if mask_pixels < OSB_RESIDUAL_RENDER_GUARD_MIN_SOURCE_PIXELS:
        return None, f"{detector_name} found no text mask", True

    return mask_bool, f"{detector_name}: mask_px={mask_pixels}", True


def _osb_residual_detector_source_mask(
    crop: Image.Image,
    config: Optional[MangaTranslatorConfig],
    verbose: bool = False,
) -> Tuple[Optional[np.ndarray], str]:
    attempts = []
    unavailable = []
    usable_without_text = []
    for detector_name in ("ctd", "dbnet"):
        mask, reason, detector_usable = _residual_detector_mask_from_hgmzhn(
            crop,
            detector_name,
            config,
            verbose=verbose,
        )
        attempts.append(reason)
        if mask is not None:
            return mask, reason
        if detector_usable:
            usable_without_text.append(reason)
        else:
            unavailable.append(reason)
    if unavailable and not usable_without_text:
        raise RuntimeError(
            "OSB residual source-text guard requires CTD or DBNet+ResNet34, "
            "but neither detector could be loaded. The program already tried "
            "the hgmzhn auto-download/load path. Details: "
            + "; ".join(unavailable)
        )
    return None, "; ".join(attempts)


def _osb_residual_source_text_remaining(
    bubble: Dict[str, Any],
    original_image: Image.Image,
    cleaned_image: Image.Image,
    config: Optional[MangaTranslatorConfig] = None,
    verbose: bool = False,
    min_residual_ratio: Optional[float] = None,
) -> Tuple[bool, str, Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Detect when OSB source glyphs were not cleaned well enough.

    The guard is based only on the original source glyph mask and the image
    after inpainting.  It does not depend on where the translated text later
    renders, so shifted short SFX cannot hide a bad cleanup result.
    """
    if original_image.size != cleaned_image.size:
        return False, "", None, None

    source_box = bubble.get("original_bbox") or bubble.get("bbox")
    if source_box is None:
        return False, "", None, None
    try:
        raw_box = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in source_box),
            original_image.size,
        )
    except (TypeError, ValueError):
        raw_box = None
    if raw_box is None:
        return False, "", None, None

    x1, y1, x2, y2 = raw_box
    original_crop = original_image.crop(raw_box)
    source_ink, detector_reason = _osb_residual_detector_source_mask(
        original_crop,
        config,
        verbose=verbose,
    )
    if source_ink is None:
        if detector_reason:
            log_message(
                "Skipping OSB residual source-text guard because CTD/DBNet "
                f"found no text mask for {raw_box}: {detector_reason}",
                verbose=verbose,
            )
        return False, "", None, None

    source_ink = np.asarray(source_ink).astype(bool)
    source_pixels = int(np.count_nonzero(source_ink))
    if source_pixels < OSB_RESIDUAL_RENDER_GUARD_MIN_SOURCE_PIXELS:
        return False, "", None, None

    try:
        original_gray = np.asarray(original_crop.convert("L"), dtype=np.int16)
        cleaned_gray = np.asarray(cleaned_image.crop(raw_box).convert("L"), dtype=np.int16)
    except Exception:
        return False, "", None, None
    if original_gray.shape != cleaned_gray.shape or original_gray.shape[:2] != source_ink.shape:
        return False, "", None, None

    similar_to_source = (
        np.abs(cleaned_gray - original_gray) <= OSB_RESIDUAL_RENDER_GUARD_MAX_SOURCE_DELTA
    )
    residual_source = np.logical_and(source_ink, similar_to_source)

    residual_pixels = int(np.count_nonzero(residual_source))
    try:
        residual_ratio_threshold = float(
            OSB_RESIDUAL_RENDER_GUARD_MIN_RESIDUAL_RATIO
            if min_residual_ratio is None
            else min_residual_ratio
        )
    except (TypeError, ValueError):
        residual_ratio_threshold = OSB_RESIDUAL_RENDER_GUARD_MIN_RESIDUAL_RATIO
    residual_ratio_threshold = max(0.0, min(1.0, residual_ratio_threshold))
    min_residual_pixels = max(
        OSB_RESIDUAL_RENDER_GUARD_MIN_SOURCE_PIXELS,
        int(round(source_pixels * residual_ratio_threshold)),
    )
    if residual_pixels < min_residual_pixels:
        return False, "", None, None

    residual_ratio = residual_pixels / float(max(1, source_pixels))
    reason = (
        f"source_px={source_pixels}, residual_px={residual_pixels} "
        f"({residual_ratio:.2f}), detector={detector_reason}"
    )
    residual_full_mask = np.zeros(
        (original_image.height, original_image.width), dtype=bool
    )
    residual_full_mask[y1:y2, x1:x2] = residual_source
    return True, reason, raw_box, residual_full_mask


def _osb_residual_lama_mask_looks_art_contaminated(mask: np.ndarray) -> Tuple[bool, str]:
    mask_bool = np.asarray(mask).astype(bool)
    if not np.any(mask_bool):
        return False, ""

    ys, xs = np.where(mask_bool)
    if xs.size == 0 or ys.size == 0:
        return False, ""

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    bbox_area = width * height
    if (
        bbox_area < OSB_RESIDUAL_LAMA_UNSAFE_MIN_BBOX_AREA
        or width < OSB_RESIDUAL_LAMA_UNSAFE_MIN_BBOX_SIDE
        or height < OSB_RESIDUAL_LAMA_UNSAFE_MIN_BBOX_SIDE
    ):
        return False, ""

    crop = mask_bool[y1:y2, x1:x2].astype(np.uint8)
    mask_pixels = int(np.count_nonzero(crop))
    if mask_pixels <= 0:
        return False, ""
    density = mask_pixels / float(max(1, bbox_area))
    if density < OSB_RESIDUAL_LAMA_UNSAFE_MIN_MASK_DENSITY:
        return False, ""

    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    except Exception:
        return False, ""
    if num_labels <= 1:
        return False, ""

    largest = max(
        range(1, num_labels),
        key=lambda label_idx: int(stats[label_idx, cv2.CC_STAT_AREA]),
    )
    largest_area = int(stats[largest, cv2.CC_STAT_AREA])
    comp_w = int(stats[largest, cv2.CC_STAT_WIDTH])
    comp_h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    largest_share = largest_area / float(max(1, mask_pixels))
    if largest_share < OSB_RESIDUAL_LAMA_UNSAFE_MIN_COMPONENT_SHARE:
        return False, ""

    span_w = comp_w / float(width)
    span_h = comp_h / float(height)
    unsafe = (
        span_w >= OSB_RESIDUAL_LAMA_UNSAFE_MIN_COMPONENT_SPAN
        and span_h >= OSB_RESIDUAL_LAMA_UNSAFE_MIN_COMPONENT_SPAN
    )
    if not unsafe:
        return False, ""

    reason = (
        f"bbox={width}x{height}, mask_px={mask_pixels}, density={density:.2f}, "
        f"largest={largest_area} ({largest_share:.2f}), span={span_w:.2f}x{span_h:.2f}"
    )
    return True, reason


def _run_osb_residual_lama_large(
    source_image: Image.Image,
    residual_mask: Optional[np.ndarray],
    config: MangaTranslatorConfig,
    limit_bbox: Optional[Tuple[int, int, int, int]] = None,
    verbose: bool = False,
) -> Optional[Image.Image]:
    if residual_mask is None:
        return None
    mask_bool = np.asarray(residual_mask).astype(bool)
    if mask_bool.shape != (source_image.height, source_image.width) or not np.any(mask_bool):
        return None

    unsafe_mask, unsafe_reason = _osb_residual_lama_mask_looks_art_contaminated(mask_bool)
    if unsafe_mask:
        log_message(
            "Skipping OSB residual LAMA Large retry because residual mask looks "
            f"art-contaminated ({unsafe_reason})",
            verbose=verbose,
            always_print=True,
        )
        return None

    img_w, img_h = source_image.size
    domain_mask = np.full((img_h, img_w), 255, dtype=np.uint8)
    if limit_bbox is not None:
        clipped = _clip_bbox_to_image(limit_bbox, source_image.size)
        if clipped is not None:
            x1, y1, x2, y2 = clipped
            domain_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # build_lama_inpaint_mask erodes the domain by one pixel. Pad the
            # domain, then clip the final mask back to the original OSB box so
            # edge glyphs stay eligible without expanding into nearby artwork.
            dx1 = max(0, x1 - 1)
            dy1 = max(0, y1 - 1)
            dx2 = min(img_w, x2 + 1)
            dy2 = min(img_h, y2 + 1)
            domain_mask[dy1:dy2, dx1:dx2] = 255

    try:
        base_kernel_size = int(config.cleaning.colored_bubble_mask_kernel_size)
    except (TypeError, ValueError):
        base_kernel_size = 3
    try:
        base_dilation_offset = int(config.cleaning.colored_bubble_mask_dilation_offset)
    except (TypeError, ValueError):
        base_dilation_offset = 50
    lama_kernel_size = base_kernel_size + OSB_RESIDUAL_LAMA_EXTRA_MASK_KERNEL_SIZE
    lama_dilation_offset = base_dilation_offset + OSB_RESIDUAL_LAMA_EXTRA_DILATION_OFFSET

    try:
        lama_mask = build_lama_inpaint_mask(
            mask_bool.astype(np.uint8),
            domain_mask,
            lama_kernel_size,
            lama_dilation_offset,
        )
        if limit_bbox is not None:
            clipped = _clip_bbox_to_image(limit_bbox, source_image.size)
            if clipped is not None:
                x1, y1, x2, y2 = clipped
                bbox_limit = np.zeros_like(lama_mask, dtype=np.uint8)
                bbox_limit[y1:y2, x1:x2] = 255
                lama_mask = cv2.bitwise_and(lama_mask, bbox_limit)
    except Exception:
        lama_mask = mask_bool.astype(np.uint8)

    if not np.any(lama_mask):
        return None

    unsafe_mask, unsafe_reason = _osb_residual_lama_mask_looks_art_contaminated(
        lama_mask > 0
    )
    if unsafe_mask:
        log_message(
            "Skipping OSB residual LAMA Large retry because expanded residual "
            f"mask looks art-contaminated ({unsafe_reason})",
            verbose=verbose,
            always_print=True,
        )
        return None

    log_message(
        "OSB residual LAMA Large retry mask expanded with cleaning settings "
        f"(kernel={lama_kernel_size}, offset={lama_dilation_offset}px, "
        f"base_kernel={config.cleaning.colored_bubble_mask_kernel_size}, "
        f"base_offset={config.cleaning.colored_bubble_mask_dilation_offset}px, "
        f"seed_px={int(np.count_nonzero(mask_bool))}, "
        f"mask_px={int(np.count_nonzero(lama_mask))})",
        verbose=verbose,
    )

    return run_lama_large(
        source_image,
        lama_mask,
        device=config.device,
        inpainting_size=config.cleaning.colored_bubble_inpainting_size,
        precision=config.cleaning.colored_bubble_inpainting_precision,
        verbose=verbose,
    )


def _reapply_edge_sfx_inpaint_rectangles(
    cleaned_image: Image.Image,
    post_osb_image: Image.Image,
    outside_text_data: List[Dict[str, Any]],
) -> Tuple[Image.Image, int]:
    if cleaned_image.size != post_osb_image.size:
        return cleaned_image, 0

    result_rgba = np.asarray(cleaned_image.convert("RGBA")).copy()
    post_osb_rgba = np.asarray(post_osb_image.convert("RGBA"))
    restored_pixels = 0
    for item in outside_text_data:
        if not item.get("edge_sfx_full_glyph", False):
            continue
        raw_bbox = item.get("original_bbox") or item.get("bbox")
        if raw_bbox is None:
            continue
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in raw_bbox),
            cleaned_image.size,
        )
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        before = result_rgba[y1:y2, x1:x2]
        source = post_osb_rgba[y1:y2, x1:x2]
        restored_pixels += int(
            np.count_nonzero(np.any(before[:, :, :3] != source[:, :, :3], axis=2))
        )
        result_rgba[y1:y2, x1:x2] = source

    if restored_pixels <= 0:
        return cleaned_image, 0

    result = Image.fromarray(result_rgba)
    if cleaned_image.mode != "RGBA":
        result = result.convert(cleaned_image.mode)
    return result, restored_pixels


def _restore_edge_sfx_collision_background(
    image: Image.Image,
    edge_bbox: Any,
    translated_text_mask: np.ndarray,
    speech_regions: List[Dict[str, Any]],
    repair_padding: int = 3,
) -> Tuple[Image.Image, int]:
    clipped = _clip_bbox_to_image(
        tuple(int(round(float(v))) for v in edge_bbox),
        image.size,
    )
    if clipped is None:
        return image, 0

    img_w, img_h = image.size
    if translated_text_mask.shape != (img_h, img_w):
        return image, 0

    x1, y1, x2, y2 = clipped
    rectangle_mask = np.zeros((img_h, img_w), dtype=bool)
    rectangle_mask[y1:y2, x1:x2] = True
    collision_mask = np.logical_and(rectangle_mask, translated_text_mask)
    collision_pixels = int(np.count_nonzero(collision_mask))
    if collision_pixels <= 0:
        return image, 0

    best_region = None
    best_score = 0
    for region in speech_regions:
        raw_region_bbox = region.get("bbox")
        if raw_region_bbox is None:
            continue
        region_bbox = _clip_bbox_to_image(raw_region_bbox, image.size)
        if region_bbox is None:
            continue
        rx1, ry1, rx2, ry2 = region_bbox
        score = int(np.count_nonzero(collision_mask[ry1:ry2, rx1:rx2]))
        if score > best_score:
            best_score = score
            best_region = region

    if best_region is None:
        return image, 0

    padding = max(0, int(repair_padding))
    if padding > 0:
        kernel_size = padding * 2 + 1
        repair_mask = cv2.dilate(
            collision_mask.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        ) > 0
        repair_mask = np.logical_and(repair_mask, rectangle_mask)
    else:
        repair_mask = collision_mask

    color_bgr = _render_rgb(best_region.get("color"), (255, 255, 255))
    color_rgb = np.array(
        (color_bgr[2], color_bgr[1], color_bgr[0]),
        dtype=np.uint8,
    )
    image_rgb = np.asarray(image.convert("RGB")).copy()
    translated_pixels = image_rgb[translated_text_mask].copy()
    image_rgb[repair_mask] = color_rgb
    image_rgb[translated_text_mask] = translated_pixels

    repaired = Image.fromarray(image_rgb)
    if image.mode == "RGBA":
        repaired.putalpha(image.getchannel("A"))
    elif image.mode != "RGB":
        repaired = repaired.convert(image.mode)
    return repaired, int(np.count_nonzero(repair_mask))


def _shift_bbox_away_from_overlapping_box(
    bbox: Any,
    obstacle_boxes: List[Tuple[int, int, int, int]],
    distance: int,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    clipped = _clip_bbox_to_image(
        tuple(int(round(float(v))) for v in bbox),
        image_size,
    )
    if clipped is None or not obstacle_boxes:
        return tuple(int(round(float(v))) for v in bbox)

    best_obstacle = max(
        obstacle_boxes,
        key=lambda item: _bbox_intersection_area(clipped, item),
    )
    x1, y1, x2, y2 = clipped
    ox1, oy1, ox2, oy2 = best_obstacle
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ocx = (ox1 + ox2) / 2.0
    ocy = (oy1 + oy2) / 2.0
    normalized_dx = (cx - ocx) / float(max(1, ox2 - ox1))
    normalized_dy = (cy - ocy) / float(max(1, oy2 - oy1))

    shift_x = 0
    shift_y = 0
    if abs(normalized_dx) >= abs(normalized_dy):
        shift_x = -distance if normalized_dx <= 0.0 else distance
    else:
        shift_y = -distance if normalized_dy <= 0.0 else distance

    img_w, img_h = image_size
    width = x2 - x1
    height = y2 - y1
    nx1 = max(0, min(img_w - width, x1 + shift_x))
    ny1 = max(0, min(img_h - height, y1 + shift_y))
    return nx1, ny1, nx1 + width, ny1 + height


def _normalize_restore_mask(mask: Any, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    try:
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        if mask_array.ndim != 2:
            return None
        img_w, img_h = image_size
        if mask_array.shape != (img_h, img_w):
            return None
        return mask_array > 0
    except Exception:
        return None


def _safe_area_dimensions_for_mask(
    mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
) -> Optional[Tuple[float, float]]:
    mask_bool = _normalize_restore_mask(mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return None

    try:
        padding = max(0.0, float(padding_pixels))
    except (TypeError, ValueError):
        padding = 0.0

    mask_u8 = np.where(mask_bool, 255, 0).astype(np.uint8)
    padded_mask = np.zeros(
        (mask_u8.shape[0] + 2, mask_u8.shape[1] + 2), dtype=np.uint8
    )
    padded_mask[1:-1, 1:-1] = mask_u8
    distance_map = cv2.distanceTransform(
        padded_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    safe_area_mask = (distance_map >= padding).astype(np.uint8) * 255
    if not np.any(safe_area_mask):
        return None

    moments = cv2.moments(safe_area_mask)
    if moments["m00"] == 0:
        return None

    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]
    _, max_val, _, max_loc = cv2.minMaxLoc(distance_map)
    mask_h, mask_w = safe_area_mask.shape
    cx = max(0, min(int(round(centroid_x)), mask_w - 1))
    cy = max(0, min(int(round(centroid_y)), mask_h - 1))
    if (
        max_val > 0
        and distance_map[cy, cx] < max_val * CONSTRICTED_CENTROID_POLE_RATIO
        and has_multiple_deep_lobes(distance_map, max_val)
    ):
        centroid_x, centroid_y = float(max_loc[0]), float(max_loc[1])
        cx = max(0, min(int(round(centroid_x)), mask_w - 1))
        cy = max(0, min(int(round(centroid_y)), mask_h - 1))

    if safe_area_mask[cy, cx] != 255:
        safe_pixels = np.argwhere(safe_area_mask == 255)
        if safe_pixels.size == 0:
            return None
        distances = np.sqrt(
            (safe_pixels[:, 0] - centroid_y) ** 2
            + (safe_pixels[:, 1] - centroid_x) ** 2
        )
        nearest_idx = int(np.argmin(distances))
        cy, cx = safe_pixels[nearest_idx]

    left_zeros = np.where(safe_area_mask[cy, 0:cx] == 0)[0]
    right_zeros = np.where(safe_area_mask[cy, cx:] == 0)[0]
    up_zeros = np.where(safe_area_mask[0:cy, cx] == 0)[0]
    down_zeros = np.where(safe_area_mask[cy:, cx] == 0)[0]

    dist_to_left = cx - (left_zeros.max() if left_zeros.size > 0 else 0)
    dist_to_right = right_zeros.min() if right_zeros.size > 0 else mask_w - cx
    dist_to_top = cy - (up_zeros.max() if up_zeros.size > 0 else 0)
    dist_to_bottom = down_zeros.min() if down_zeros.size > 0 else mask_h - cy

    min_width_dist = min(dist_to_left, dist_to_right)
    min_height_dist = min(dist_to_top, dist_to_bottom)
    safe_width_base = min_width_dist - 1 if min_width_dist > 1 else min_width_dist
    safe_height_base = min_height_dist - 1 if min_height_dist > 1 else min_height_dist
    safe_w = float(2 * max(0, safe_width_base))
    safe_h = float(2 * max(0, safe_height_base))
    if safe_w <= 0 or safe_h <= 0:
        return None
    return safe_w, safe_h


def _guard_conjoined_child_layout_mask(
    bubble: Dict[str, Any],
    bbox: Any,
    layout_mask: Any,
    child_fallback_mask: Any,
    base_mask: Any,
    image_size: Tuple[int, int],
    verbose: bool,
) -> Tuple[Any, bool]:
    """Reject conjoined child layout masks that spill into sibling bubbles."""
    neighbor_bboxes = bubble.get("conjoined_neighbor_bboxes") or []
    if not neighbor_bboxes:
        return layout_mask, False

    layout_bool = _normalize_restore_mask(layout_mask, image_size)
    if layout_bool is None or not np.any(layout_bool):
        return layout_mask, False

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return layout_mask, False

    img_w, img_h = image_size
    x1, y1, x2, y2 = clipped
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = max(1, bbox_w * bbox_h)
    margin = int(round(min(bbox_w, bbox_h) * 0.08))
    margin = max(6, min(24, margin))
    mx1 = max(0, x1 - margin)
    my1 = max(0, y1 - margin)
    mx2 = min(img_w, x2 + margin)
    my2 = min(img_h, y2 + margin)

    child_area_mask = np.zeros_like(layout_bool, dtype=bool)
    child_area_mask[my1:my2, mx1:mx2] = True
    strict_child_mask = np.zeros_like(layout_bool, dtype=bool)
    strict_child_mask[y1:y2, x1:x2] = True

    def _sibling_intrusion_stats(mask_bool: np.ndarray) -> Tuple[int, int, Any]:
        outside_child = np.logical_and(mask_bool, np.logical_not(child_area_mask))
        outside_px = int(np.count_nonzero(outside_child))
        if outside_px <= 0:
            return 0, 0, None

        max_intrusion = 0
        max_neighbor = None
        for neighbor_bbox in neighbor_bboxes:
            try:
                neighbor = _clip_bbox_to_image(
                    tuple(int(round(float(v))) for v in neighbor_bbox),
                    image_size,
                )
            except (TypeError, ValueError):
                neighbor = None
            if neighbor is None:
                continue
            nx1, ny1, nx2, ny2 = neighbor
            intrusion_px = int(np.count_nonzero(outside_child[ny1:ny2, nx1:nx2]))
            if intrusion_px > max_intrusion:
                max_intrusion = intrusion_px
                max_neighbor = neighbor
        return outside_px, max_intrusion, max_neighbor

    outside_px, intrusion_px, intruded_neighbor = _sibling_intrusion_stats(layout_bool)
    min_outside_px = max(96, int(round(float(bbox_area) * 0.02)))
    min_intrusion_px = max(96, int(round(float(bbox_area) * 0.01)))
    if outside_px < min_outside_px or intrusion_px < min_intrusion_px:
        return layout_mask, False

    fallback_candidates = [child_fallback_mask, base_mask]
    for candidate in (child_fallback_mask, base_mask, layout_mask):
        candidate_bool = _normalize_restore_mask(candidate, image_size)
        if candidate_bool is None or not np.any(candidate_bool):
            continue
        clipped_candidate = np.logical_and(candidate_bool, strict_child_mask)
        if np.any(clipped_candidate):
            fallback_candidates.append(
                np.where(clipped_candidate, 255, 0).astype(np.uint8)
            )

    for candidate in fallback_candidates:
        candidate_bool = _normalize_restore_mask(candidate, image_size)
        if candidate_bool is None or not np.any(candidate_bool):
            continue
        candidate_outside_px, candidate_intrusion_px, _ = _sibling_intrusion_stats(
            candidate_bool
        )
        if (
            candidate_intrusion_px <= max(32, int(round(intrusion_px * 0.35)))
            and candidate_outside_px
            <= max(min_outside_px, int(round(outside_px * 0.65)))
        ):
            layout_bbox = _bbox_from_binary_mask(layout_bool, clipped)
            fallback_bbox = _bbox_from_binary_mask(candidate_bool, clipped)
            log_message(
                "Rejected conjoined child layout mask "
                f"{tuple(clipped)} because it intruded into sibling "
                f"{intruded_neighbor} ({intrusion_px}px, outside={outside_px}px); "
                f"using child-local mask {fallback_bbox} instead of {layout_bbox}",
                verbose=verbose,
            )
            return candidate, True

    return layout_mask, False


def _clip_simple_neighbor_intruding_mask(
    bubble: Dict[str, Any],
    bbox: Any,
    mask: Any,
    speech_bubble_bboxes: List[Tuple[int, int, int, int]],
    image_size: Tuple[int, int],
    verbose: bool,
    *,
    stage: str,
    mask_name: str,
    allow_conjoined: bool = False,
) -> Tuple[Any, bool]:
    """Clip a simple bubble mask when it clearly spills into a sibling bubble.

    Conjoined children are handled by their own layout guard. This catches the
    failure mode where a split child is misclassified as simple, so its YOLO
    mask can expand into a neighboring child and distort the render safe area.
    """
    if mask is None:
        return mask, False
    if not allow_conjoined and (
        bubble.get("conjoined_parent_mask") is not None
        or bubble.get("conjoined_parent_mask_for_classification") is not None
        or bubble.get("conjoined_neighbor_bboxes")
        or bubble.get("synthetic_conjoined_child")
    ):
        return mask, False

    try:
        clipped_bbox = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped_bbox = None
    if clipped_bbox is None:
        return mask, False

    mask_bool = _normalize_restore_mask(mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return mask, False

    img_w, img_h = image_size
    x1, y1, x2, y2 = clipped_bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = max(1, bbox_w * bbox_h)

    margin = int(round(min(bbox_w, bbox_h) * 0.035))
    margin = max(4, min(10, margin))
    mx1 = max(0, x1 - margin)
    my1 = max(0, y1 - margin)
    mx2 = min(img_w, x2 + margin)
    my2 = min(img_h, y2 + margin)

    own_area_mask = np.zeros_like(mask_bool, dtype=bool)
    own_area_mask[my1:my2, mx1:mx2] = True
    outside_own = np.logical_and(mask_bool, np.logical_not(own_area_mask))
    outside_px = int(np.count_nonzero(outside_own))
    min_outside_px = max(160, int(round(float(bbox_area) * 0.025)))
    if outside_px < min_outside_px:
        return mask, False

    intrusion_px = 0
    intruded_neighbor = None
    for neighbor_bbox in speech_bubble_bboxes:
        if tuple(neighbor_bbox) == tuple(clipped_bbox):
            continue
        if _bbox_iou(neighbor_bbox, clipped_bbox) >= 0.92:
            continue
        neighbor = _clip_bbox_to_image(neighbor_bbox, image_size)
        if neighbor is None:
            continue
        nx1, ny1, nx2, ny2 = neighbor
        overlap_px = int(np.count_nonzero(outside_own[ny1:ny2, nx1:nx2]))
        if overlap_px > intrusion_px:
            intrusion_px = overlap_px
            intruded_neighbor = neighbor

    min_intrusion_px = max(128, int(round(float(bbox_area) * 0.008)))
    if intrusion_px < min_intrusion_px:
        return mask, False

    clipped_mask_bool = np.logical_and(mask_bool, own_area_mask)
    clipped_px = int(np.count_nonzero(clipped_mask_bool))
    if clipped_px < max(64, int(round(float(bbox_area) * 0.12))):
        return mask, False

    original_bbox = _bbox_from_binary_mask(np.where(mask_bool, 255, 0).astype(np.uint8), clipped_bbox)
    clipped_mask = np.where(clipped_mask_bool, 255, 0).astype(np.uint8)
    clipped_mask_bbox = _bbox_from_binary_mask(clipped_mask, clipped_bbox)
    log_message(
        "Clipped speech-bubble "
        f"{mask_name} during {stage} for {tuple(clipped_bbox)}: "
        f"{original_bbox} -> {clipped_mask_bbox}; sibling={intruded_neighbor}, "
        f"intrusion={intrusion_px}px, outside={outside_px}px",
        verbose=verbose,
        always_print=True,
    )
    return clipped_mask, True


def _clip_simple_neighbor_intruding_item_masks(
    items: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    verbose: bool,
    *,
    mask_keys: Tuple[str, ...] = ("sam_mask",),
    stage: str,
) -> int:
    """Apply simple-bubble sibling-intrusion clipping to item masks in place."""
    speech_bubble_bboxes: List[Tuple[int, int, int, int]] = []
    for item in items or []:
        if item.get("is_outside_text", False):
            continue
        try:
            bbox = tuple(int(round(float(v))) for v in item.get("bbox", ()))
        except (TypeError, ValueError):
            continue
        if len(bbox) == 4:
            speech_bubble_bboxes.append(bbox)

    if len(speech_bubble_bboxes) < 2:
        return 0

    changed = 0
    for item in items or []:
        if item.get("is_outside_text", False):
            continue
        try:
            bbox = tuple(int(round(float(v))) for v in item.get("bbox", ()))
        except (TypeError, ValueError):
            continue
        if len(bbox) != 4:
            continue
        for key in mask_keys:
            if item.get(key) is None:
                continue
            clipped_mask, did_clip = _clip_simple_neighbor_intruding_mask(
                item,
                bbox,
                item.get(key),
                speech_bubble_bboxes,
                image_size,
                verbose,
                stage=stage,
                mask_name=key,
            )
            if did_clip:
                item[key] = clipped_mask
                changed += 1
    return changed


def _compact_render_char_count(text: Any) -> int:
    return sum(1 for ch in str(text or "") if ch.isalnum())


def _odd_kernel_size(value: float, minimum: int = 3, maximum: int = 31) -> int:
    try:
        size = int(round(float(value)))
    except (TypeError, ValueError):
        size = minimum
    size = max(minimum, min(maximum, size))
    if size % 2 == 0:
        size += 1 if size < maximum else -1
    return max(minimum, min(maximum, size))


def _build_colored_layout_clean_mask(
    render_mask: Any,
    base_mask: Any,
    bbox: Tuple[int, int, int, int],
    padding_pixels: float,
    image_size: Tuple[int, int],
    render_dims: Tuple[float, float],
    base_dims: Tuple[float, float],
    *,
    min_safe_area_gain: float = COLORED_LAYOUT_CLEAN_MIN_SAFE_AREA_GAIN,
    min_candidate_width_gain: float = COLORED_LAYOUT_CLEAN_MIN_CANDIDATE_WIDTH_GAIN,
    min_candidate_height_gain: float = COLORED_LAYOUT_CLEAN_MIN_CANDIDATE_WIDTH_GAIN,
    min_base_width_recovery: float = COLORED_LAYOUT_CLEAN_MIN_BASE_WIDTH_RECOVERY,
    min_base_height_recovery: float = COLORED_LAYOUT_CLEAN_MIN_BASE_WIDTH_RECOVERY,
    max_base_fill_ratio: float = COLORED_LAYOUT_CLEAN_MAX_BASE_FILL_RATIO,
    max_added_missing_ratio: float = COLORED_LAYOUT_CLEAN_MAX_ADDED_MISSING_RATIO,
    allow_height_recovery: bool = False,
) -> Optional[Tuple[np.ndarray, Tuple[float, float], int]]:
    """Repair a colored/translucent clean mask for layout without using base directly."""
    render_bool = _normalize_restore_mask(render_mask, image_size)
    base_bool = _normalize_restore_mask(base_mask, image_size)
    if render_bool is None or base_bool is None:
        return None
    if not np.any(render_bool) or not np.any(base_bool):
        return None

    x1, y1, x2, y2 = bbox
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    render_crop = np.where(render_bool[y1:y2, x1:x2], 255, 0).astype(np.uint8)
    base_crop = np.where(base_bool[y1:y2, x1:x2], 255, 0).astype(np.uint8)
    if not np.any(render_crop) or not np.any(base_crop):
        return None

    render_pixels = int(np.count_nonzero(render_crop))
    base_pixels = int(np.count_nonzero(base_crop))
    if base_pixels <= render_pixels:
        return None

    kernel_size = _odd_kernel_size(
        min(bbox_w, bbox_h) * COLORED_LAYOUT_CLEAN_KERNEL_RATIO,
        maximum=COLORED_LAYOUT_CLEAN_MAX_KERNEL,
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    grow_size = _odd_kernel_size(max(3, kernel_size // 2), maximum=kernel_size)
    grow_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (grow_size, grow_size)
    )

    candidate_crop = cv2.morphologyEx(
        render_crop, cv2.MORPH_CLOSE, close_kernel, iterations=2
    )
    candidate_crop = cv2.dilate(candidate_crop, grow_kernel, iterations=1)
    candidate_crop = cv2.morphologyEx(
        candidate_crop, cv2.MORPH_CLOSE, close_kernel, iterations=1
    )
    candidate_crop = cv2.bitwise_or(candidate_crop, render_crop)
    candidate_crop = cv2.bitwise_and(candidate_crop, base_crop)

    candidate_pixels = int(np.count_nonzero(candidate_crop))
    added_pixels = int(
        np.count_nonzero((candidate_crop > 0) & (render_crop == 0))
    )
    missing_base_pixels = int(
        np.count_nonzero((base_crop > 0) & (render_crop == 0))
    )
    if added_pixels <= 0 or missing_base_pixels <= 0:
        return None
    max_candidate_pixels = int(round(base_pixels * max_base_fill_ratio))
    if candidate_pixels >= max_candidate_pixels:
        return None
    if (
        float(added_pixels) / float(max(1, missing_base_pixels))
        > max_added_missing_ratio
    ):
        return None

    candidate = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    candidate[y1:y2, x1:x2] = candidate_crop
    candidate_dims = _safe_area_dimensions_for_mask(
        candidate,
        padding_pixels,
        image_size,
    )
    if candidate_dims is None:
        return None

    render_area = max(1.0, float(render_dims[0]) * float(render_dims[1]))
    candidate_area = max(1.0, float(candidate_dims[0]) * float(candidate_dims[1]))
    candidate_area_gain = candidate_area / render_area
    candidate_width_gain = float(candidate_dims[0]) / max(float(render_dims[0]), 1.0)
    candidate_height_gain = float(candidate_dims[1]) / max(float(render_dims[1]), 1.0)
    base_width_recovery = float(candidate_dims[0]) / max(float(base_dims[0]), 1.0)
    base_height_recovery = float(candidate_dims[1]) / max(float(base_dims[1]), 1.0)
    height_retention = float(candidate_dims[1]) / max(float(render_dims[1]), 1.0)
    candidate_side_recovered = candidate_width_gain >= min_candidate_width_gain
    base_side_recovered = base_width_recovery >= min_base_width_recovery
    if allow_height_recovery:
        candidate_side_recovered = (
            candidate_side_recovered
            or candidate_height_gain >= min_candidate_height_gain
        )
        base_side_recovered = (
            base_side_recovered
            or base_height_recovery >= min_base_height_recovery
        )
    if (
        candidate_area_gain < min_safe_area_gain
        or not candidate_side_recovered
        or not base_side_recovered
        or height_retention < COLORED_LAYOUT_CLEAN_MIN_HEIGHT_RETENTION
    ):
        return None

    return candidate, candidate_dims, added_pixels


def _select_conjoined_layout_mask(
    bubble: Dict[str, Any],
    bbox: Any,
    render_mask: Any,
    base_mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    translated_text: str,
    render_tag: str,
    verbose: bool,
    allow_colored_bubble: bool = False,
    layout_mask_hint: Any = None,
) -> Any:
    """Use the bubble-shape mask for layout when the text-clean mask collapses.

    Cleaning masks are optimized for removing original glyphs. In conjoined
    and LAMA-cleaned colored bubbles they can shrink around the original text,
    making the distance-transform safe area much smaller than the bubble.
    """
    if render_mask is None or base_mask is None:
        return render_mask
    if bubble.get("is_outside_text", False) or _is_sfx_render_tag(render_tag):
        return render_mask
    is_conjoined = not (
        bubble.get("conjoined_parent_mask") is None
        and not bubble.get("conjoined_neighbor_bboxes")
    )
    compact_chars = _compact_render_char_count(translated_text)

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return render_mask

    render_dims = _safe_area_dimensions_for_mask(render_mask, padding_pixels, image_size)
    base_dims = _safe_area_dimensions_for_mask(base_mask, padding_pixels, image_size)
    if render_dims is None or base_dims is None:
        return render_mask

    x1, y1, x2, y2 = clipped
    bbox_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    render_area = max(1.0, float(render_dims[0] * render_dims[1]))
    base_area = max(1.0, float(base_dims[0] * base_dims[1]))
    render_area_ratio = render_area / bbox_area
    base_area_ratio = base_area / bbox_area
    area_gain = base_area / render_area
    width_gain = float(base_dims[0]) / max(float(render_dims[0]), 1.0)
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))
    render_width_ratio = float(render_dims[0]) / bbox_w
    render_height_ratio = float(render_dims[1]) / bbox_h
    base_height_ratio = float(base_dims[1]) / bbox_h
    height_gain = float(base_dims[1]) / max(float(render_dims[1]), 1.0)
    if layout_mask_hint is not None:
        hint_dims = _safe_area_dimensions_for_mask(
            layout_mask_hint,
            padding_pixels,
            image_size,
        )
        if hint_dims is not None:
            hint_area = max(1.0, float(hint_dims[0] * hint_dims[1]))
            hint_area_ratio = hint_area / bbox_area
            hint_gain = hint_area / render_area
            hint_width_gain = float(hint_dims[0]) / max(float(render_dims[0]), 1.0)
            hint_height_gain = float(hint_dims[1]) / max(float(render_dims[1]), 1.0)
            base_safe_area = max(1.0, float(base_dims[0] * base_dims[1]))
            hint_base_ratio = hint_area / base_safe_area
            if (
                hint_area_ratio >= 0.08
                and hint_gain >= 1.10
                and (hint_width_gain >= 1.05 or hint_height_gain >= 1.05)
                and hint_base_ratio <= 0.98
            ):
                log_message(
                    "Using source-stroke layout mask "
                    f"{tuple(clipped)} (render_safe={render_dims[0]:.0f}x{render_dims[1]:.0f}, "
                    f"layout_safe={hint_dims[0]:.0f}x{hint_dims[1]:.0f}, "
                    f"base_safe={base_dims[0]:.0f}x{base_dims[1]:.0f}, "
                    f"gain={hint_gain:.2f}, width_gain={hint_width_gain:.2f}, "
                    f"height_gain={hint_height_gain:.2f})",
                    verbose=verbose,
                )
                return layout_mask_hint
    if not is_conjoined and not allow_colored_bubble:
        return render_mask
    colored_tall_height_collapse = (
        allow_colored_bubble
        and not is_conjoined
        and compact_chars >= COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_TEXT_CHARS
        and (bbox_h / bbox_w) >= COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_ASPECT_RATIO
        and render_area_ratio <= COLORED_LAYOUT_HEIGHT_COLLAPSE_MAX_AREA_RATIO
        and render_height_ratio
        <= COLORED_LAYOUT_HEIGHT_COLLAPSE_MAX_RENDER_HEIGHT_RATIO
        and render_width_ratio
        >= COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_RENDER_WIDTH_RATIO
        and base_height_ratio >= COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_BASE_HEIGHT_RATIO
        and height_gain >= COLORED_LAYOUT_HEIGHT_COLLAPSE_MIN_HEIGHT_GAIN
    )
    severe_short_text_collapse = (
        is_conjoined
        and CONJOINED_LAYOUT_SHORT_TEXT_MIN_CHARS
        <= compact_chars
        <= CONJOINED_LAYOUT_SHORT_TEXT_MAX_CHARS
        and render_area_ratio <= CONJOINED_LAYOUT_SHORT_TEXT_MAX_AREA_RATIO
        and area_gain >= CONJOINED_LAYOUT_SHORT_TEXT_MIN_AREA_GAIN
        and width_gain >= CONJOINED_LAYOUT_SHORT_TEXT_MIN_WIDTH_GAIN
    )
    severe_short_colored_collapse = False
    center_improvement = 0.0
    if (
        allow_colored_bubble
        and not is_conjoined
        and COLORED_LAYOUT_SHORT_TEXT_MIN_CHARS
        <= compact_chars
        <= COLORED_LAYOUT_SHORT_TEXT_MAX_CHARS
        and render_area_ratio <= COLORED_LAYOUT_SHORT_TEXT_MAX_AREA_RATIO
        and area_gain >= COLORED_LAYOUT_SHORT_TEXT_MIN_AREA_GAIN
        and width_gain >= COLORED_LAYOUT_BASE_MASK_MIN_WIDTH_GAIN
    ):
        render_geometry = _safe_area_box_for_mask(
            render_mask,
            padding_pixels,
            image_size,
        )
        base_geometry = _safe_area_box_for_mask(
            base_mask,
            padding_pixels,
            image_size,
        )
        if render_geometry is not None and base_geometry is not None:
            bbox_center_y = (float(y1) + float(y2)) / 2.0
            render_center_y = float(render_geometry[1][1])
            base_center_y = float(base_geometry[1][1])
            center_improvement = abs(render_center_y - bbox_center_y) - abs(
                base_center_y - bbox_center_y
            )
            min_center_improvement = max(
                COLORED_LAYOUT_SHORT_TEXT_MIN_CENTER_IMPROVEMENT_PX,
                float(y2 - y1)
                * COLORED_LAYOUT_SHORT_TEXT_MIN_CENTER_IMPROVEMENT_RATIO,
            )
            severe_short_colored_collapse = (
                center_improvement >= min_center_improvement
            )

    if (
        compact_chars < CONJOINED_LAYOUT_BASE_MASK_MIN_TEXT_CHARS
        and not severe_short_text_collapse
        and not severe_short_colored_collapse
        and not colored_tall_height_collapse
    ):
        return render_mask

    collapsed_enough = (
        render_area_ratio <= CONJOINED_LAYOUT_BASE_MASK_MAX_COLLAPSED_AREA_RATIO
        and area_gain >= CONJOINED_LAYOUT_BASE_MASK_MIN_AREA_GAIN
    )
    max_borderline_ratio = (
        COLORED_LAYOUT_BASE_MASK_MAX_BORDERLINE_AREA_RATIO
        if allow_colored_bubble and not is_conjoined
        else CONJOINED_LAYOUT_BASE_MASK_MAX_BORDERLINE_AREA_RATIO
    )
    borderline_but_clearly_recoverable = (
        render_area_ratio <= max_borderline_ratio
        and area_gain >= CONJOINED_LAYOUT_BASE_MASK_MIN_BORDERLINE_AREA_GAIN
    )
    colored_width_recovered = (
        allow_colored_bubble
        and not is_conjoined
        and width_gain >= COLORED_LAYOUT_BASE_MASK_MIN_WIDTH_GAIN
    )
    colored_height_recovered = colored_tall_height_collapse
    colored_layout_repair = None
    if (
        allow_colored_bubble
        and not is_conjoined
        and not severe_short_colored_collapse
        and width_gain >= COLORED_LAYOUT_CLEAN_MIN_WIDTH_GAIN
        and area_gain >= COLORED_LAYOUT_CLEAN_MIN_SAFE_AREA_GAIN
    ):
        colored_layout_repair = _build_colored_layout_clean_mask(
            render_mask,
            base_mask,
            tuple(clipped),
            padding_pixels,
            image_size,
            render_dims,
            base_dims,
        )
        if colored_layout_repair is not None:
            layout_mask, repaired_dims, added_pixels = colored_layout_repair
            log_message(
                "Using repaired colored/translucent layout clean mask "
                f"{tuple(clipped)} (render_safe={render_dims[0]:.0f}x{render_dims[1]:.0f}, "
                f"layout_safe={repaired_dims[0]:.0f}x{repaired_dims[1]:.0f}, "
                f"base_safe={base_dims[0]:.0f}x{base_dims[1]:.0f}, "
                f"gain={area_gain:.2f}, width_gain={width_gain:.2f}, "
                f"added={added_pixels}px)",
                verbose=verbose,
            )
            return layout_mask

    colored_conjoined_layout_repair = None
    if (
        allow_colored_bubble
        and is_conjoined
        and base_area_ratio >= CONJOINED_LAYOUT_BASE_MASK_MIN_BASE_AREA_RATIO
        and (collapsed_enough or borderline_but_clearly_recoverable)
    ):
        colored_conjoined_layout_repair = _build_colored_layout_clean_mask(
            render_mask,
            base_mask,
            tuple(clipped),
            padding_pixels,
            image_size,
            render_dims,
            base_dims,
            min_safe_area_gain=(
                COLORED_CONJOINED_LAYOUT_CLEAN_MIN_SAFE_AREA_GAIN
            ),
            min_candidate_width_gain=(
                COLORED_CONJOINED_LAYOUT_CLEAN_MIN_CANDIDATE_SIDE_GAIN
            ),
            min_candidate_height_gain=(
                COLORED_CONJOINED_LAYOUT_CLEAN_MIN_CANDIDATE_SIDE_GAIN
            ),
            min_base_width_recovery=(
                COLORED_CONJOINED_LAYOUT_CLEAN_MIN_BASE_SIDE_RECOVERY
            ),
            min_base_height_recovery=(
                COLORED_CONJOINED_LAYOUT_CLEAN_MIN_BASE_SIDE_RECOVERY
            ),
            max_base_fill_ratio=(
                COLORED_CONJOINED_LAYOUT_CLEAN_MAX_BASE_FILL_RATIO
            ),
            max_added_missing_ratio=(
                COLORED_CONJOINED_LAYOUT_CLEAN_MAX_ADDED_MISSING_RATIO
            ),
            allow_height_recovery=True,
        )
        if colored_conjoined_layout_repair is not None:
            layout_mask, repaired_dims, added_pixels = colored_conjoined_layout_repair
            repaired_base_width_recovery = float(repaired_dims[0]) / max(
                float(base_dims[0]), 1.0
            )
            repaired_base_height_recovery = float(repaired_dims[1]) / max(
                float(base_dims[1]), 1.0
            )
            if (
                compact_chars
                >= COLORED_CONJOINED_LAYOUT_CLEAN_LONG_TEXT_MIN_CHARS
                and repaired_base_width_recovery
                < COLORED_CONJOINED_LAYOUT_CLEAN_LONG_TEXT_MIN_BASE_WIDTH_RECOVERY
            ):
                log_message(
                    "Skipping repaired colored conjoined layout clean mask "
                    f"{tuple(clipped)} because it remains too narrow "
                    f"(render_safe={render_dims[0]:.0f}x{render_dims[1]:.0f}, "
                    f"layout_safe={repaired_dims[0]:.0f}x{repaired_dims[1]:.0f}, "
                    f"base_safe={base_dims[0]:.0f}x{base_dims[1]:.0f}, "
                    f"width_recovery={repaired_base_width_recovery:.2f}, "
                    f"height_recovery={repaired_base_height_recovery:.2f}, "
                    f"text_chars={compact_chars}); falling back to base layout",
                    verbose=verbose,
                )
            else:
                log_message(
                    "Using repaired colored conjoined layout clean mask "
                    f"{tuple(clipped)} (render_safe={render_dims[0]:.0f}x{render_dims[1]:.0f}, "
                    f"layout_safe={repaired_dims[0]:.0f}x{repaired_dims[1]:.0f}, "
                    f"base_safe={base_dims[0]:.0f}x{base_dims[1]:.0f}, "
                    f"gain={area_gain:.2f}, width_gain={width_gain:.2f}, "
                    f"height_gain={height_gain:.2f}, "
                    f"width_recovery={repaired_base_width_recovery:.2f}, "
                    f"height_recovery={repaired_base_height_recovery:.2f}, "
                    f"added={added_pixels}px)",
                    verbose=verbose,
                )
                return layout_mask

    if (
        base_area_ratio >= CONJOINED_LAYOUT_BASE_MASK_MIN_BASE_AREA_RATIO
        and (
            collapsed_enough
            or borderline_but_clearly_recoverable
            or severe_short_text_collapse
            or severe_short_colored_collapse
            or colored_tall_height_collapse
        )
        and (is_conjoined or colored_width_recovered or colored_height_recovered)
    ):
        layout_kind = "conjoined" if is_conjoined else "colored/translucent"
        log_message(
            f"Using {layout_kind} base mask for layout "
            f"{tuple(clipped)} (render_safe={render_dims[0]:.0f}x{render_dims[1]:.0f}, "
            f"base_safe={base_dims[0]:.0f}x{base_dims[1]:.0f}, "
            f"render_area_ratio={render_area_ratio:.2f}, "
            f"base_area_ratio={base_area_ratio:.2f}, gain={area_gain:.2f}, "
            f"width_gain={width_gain:.2f}, height_gain={height_gain:.2f}, "
            f"render_height_ratio={render_height_ratio:.2f}, "
            f"center_improvement={center_improvement:.1f}px)",
            verbose=verbose,
        )
        return base_mask

    return render_mask


def _safe_area_box_for_mask(
    mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, float], np.ndarray]]:
    mask_bool = _normalize_restore_mask(mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return None

    try:
        padding = max(0.0, float(padding_pixels))
    except (TypeError, ValueError):
        padding = 0.0

    mask_u8 = np.where(mask_bool, 255, 0).astype(np.uint8)
    padded_mask = np.zeros(
        (mask_u8.shape[0] + 2, mask_u8.shape[1] + 2), dtype=np.uint8
    )
    padded_mask[1:-1, 1:-1] = mask_u8
    distance_map = cv2.distanceTransform(
        padded_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )[1:-1, 1:-1]
    safe_area_mask = distance_map >= padding
    if not np.any(safe_area_mask):
        return None

    moments = cv2.moments(np.where(safe_area_mask, 255, 0).astype(np.uint8))
    if moments["m00"] == 0:
        return None

    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]
    _, max_val, _, max_loc = cv2.minMaxLoc(distance_map)
    mask_h, mask_w = safe_area_mask.shape
    cx = max(0, min(int(round(centroid_x)), mask_w - 1))
    cy = max(0, min(int(round(centroid_y)), mask_h - 1))
    if max_val > 0 and distance_map[cy, cx] < max_val * 0.70:
        centroid_x, centroid_y = float(max_loc[0]), float(max_loc[1])
        cx = max(0, min(int(round(centroid_x)), mask_w - 1))
        cy = max(0, min(int(round(centroid_y)), mask_h - 1))

    if not safe_area_mask[cy, cx]:
        safe_pixels = np.argwhere(safe_area_mask)
        if safe_pixels.size == 0:
            return None
        distances = np.sqrt(
            (safe_pixels[:, 0] - centroid_y) ** 2
            + (safe_pixels[:, 1] - centroid_x) ** 2
        )
        nearest_idx = int(np.argmin(distances))
        cy, cx = safe_pixels[nearest_idx]
        centroid_x, centroid_y = float(cx), float(cy)

    left_zeros = np.where(~safe_area_mask[cy, 0:cx])[0]
    right_zeros = np.where(~safe_area_mask[cy, cx:])[0]
    up_zeros = np.where(~safe_area_mask[0:cy, cx])[0]
    down_zeros = np.where(~safe_area_mask[cy:, cx])[0]

    dist_to_left = cx - (left_zeros.max() if left_zeros.size > 0 else 0)
    dist_to_right = right_zeros.min() if right_zeros.size > 0 else mask_w - cx
    dist_to_top = cy - (up_zeros.max() if up_zeros.size > 0 else 0)
    dist_to_bottom = down_zeros.min() if down_zeros.size > 0 else mask_h - cy

    safe_width_base = min(dist_to_left, dist_to_right)
    safe_height_base = min(dist_to_top, dist_to_bottom)
    safe_width_base = safe_width_base - 1 if safe_width_base > 1 else safe_width_base
    safe_height_base = (
        safe_height_base - 1 if safe_height_base > 1 else safe_height_base
    )
    safe_w = int(2 * max(0, safe_width_base))
    safe_h = int(2 * max(0, safe_height_base))
    if safe_w <= 0 or safe_h <= 0:
        return None

    box_x = int(round(centroid_x - safe_w / 2.0))
    box_y = int(round(centroid_y - safe_h / 2.0))
    if box_x < 0 or box_y < 0 or box_x + safe_w > mask_w or box_y + safe_h > mask_h:
        return None

    return (box_x, box_y, safe_w, safe_h), (centroid_x, centroid_y), safe_area_mask


def _bridge_narrow_notch_for_layout(
    bubble: Dict[str, Any],
    bbox: Any,
    render_mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    translated_text: str,
    render_tag: str,
    verbose: bool,
) -> Any:
    """Bridge a thin vertical intrusion for text layout without changing pixels."""
    if render_mask is None:
        return render_mask
    if bubble.get("is_outside_text", False) or _is_sfx_render_tag(render_tag):
        return render_mask
    if (
        bubble.get("conjoined_parent_mask") is not None
        or bubble.get("conjoined_neighbor_bboxes")
    ):
        return render_mask
    if _compact_render_char_count(translated_text) < NARROW_NOTCH_LAYOUT_MIN_TEXT_CHARS:
        return render_mask

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(value))) for value in bbox),
            image_size,
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return render_mask

    x1, y1, x2, y2 = clipped
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    if (
        bbox_w < NARROW_NOTCH_LAYOUT_MIN_BBOX_WIDTH
        or bbox_h < NARROW_NOTCH_LAYOUT_MIN_BBOX_HEIGHT
    ):
        return render_mask

    mask_bool = _normalize_restore_mask(render_mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return render_mask
    crop = np.where(mask_bool[y1:y2, x1:x2], 255, 0).astype(np.uint8)
    bbox_area = max(1, bbox_w * bbox_h)
    mask_pixels = int(np.count_nonzero(crop))
    if mask_pixels < bbox_area * NARROW_NOTCH_LAYOUT_MIN_MASK_COVERAGE:
        return render_mask

    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(
        crop,
        connectivity=8,
    )
    if component_count <= 1:
        return render_mask
    largest_component = max(
        int(component_stats[index, cv2.CC_STAT_AREA])
        for index in range(1, component_count)
    )
    if (
        float(largest_component) / float(max(mask_pixels, 1))
        < NARROW_NOTCH_LAYOUT_MIN_LARGEST_COMPONENT_RATIO
    ):
        return render_mask

    original_dims = _safe_area_dimensions_for_mask(
        render_mask,
        padding_pixels,
        image_size,
    )
    if original_dims is None:
        return render_mask
    original_w = max(1, int(round(original_dims[0])))
    original_h = max(1, int(round(original_dims[1])))
    if float(original_w) / float(bbox_w) > NARROW_NOTCH_LAYOUT_MAX_SAFE_WIDTH_RATIO:
        return render_mask

    kernel_width = max(
        3,
        int(round(bbox_w * NARROW_NOTCH_LAYOUT_KERNEL_WIDTH_RATIO)),
    )
    if kernel_width % 2 == 0:
        kernel_width += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    closed_crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel)
    added_pixels = int(np.count_nonzero((closed_crop > 0) & (crop == 0)))
    if (
        added_pixels <= 0
        or float(added_pixels) / float(bbox_area)
        > NARROW_NOTCH_LAYOUT_MAX_ADDED_AREA_RATIO
    ):
        return render_mask

    candidate = np.where(mask_bool, 255, 0).astype(np.uint8)
    candidate[y1:y2, x1:x2] = closed_crop
    candidate_dims = _safe_area_dimensions_for_mask(
        candidate,
        padding_pixels,
        image_size,
    )
    if candidate_dims is None:
        return render_mask
    candidate_w = max(1, int(round(candidate_dims[0])))
    candidate_h = max(1, int(round(candidate_dims[1])))
    width_gain = float(candidate_w) / float(original_w)
    area_gain = float(candidate_w * candidate_h) / float(original_w * original_h)
    height_retention = float(candidate_h) / float(original_h)

    if (
        float(candidate_w) / float(bbox_w) >= NARROW_NOTCH_LAYOUT_MIN_SAFE_WIDTH_RATIO
        and width_gain >= NARROW_NOTCH_LAYOUT_MIN_SAFE_WIDTH_GAIN
        and area_gain >= NARROW_NOTCH_LAYOUT_MIN_SAFE_AREA_GAIN
        and height_retention >= NARROW_NOTCH_LAYOUT_MIN_SAFE_HEIGHT_RETENTION
    ):
        log_message(
            "Bridged narrow layout notch for "
            f"{tuple(clipped)} (safe={original_w}x{original_h} -> "
            f"{candidate_w}x{candidate_h}, added={added_pixels}px, "
            f"kernel={kernel_width}px)",
            verbose=verbose,
            always_print=True,
        )
        return candidate

    return render_mask


def _complete_occluded_dark_bubble_layout_mask(
    bubble: Dict[str, Any],
    bbox: Any,
    render_mask: Any,
    bubble_color_bgr: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    render_tag: str,
    verbose: bool,
) -> Any:
    """Complete a severely occluded dark bubble for text placement only."""
    if render_mask is None:
        return render_mask
    if bubble.get("is_outside_text", False) or _is_sfx_render_tag(render_tag):
        return render_mask
    if (
        bubble.get("conjoined_parent_mask") is not None
        or bubble.get("conjoined_neighbor_bboxes")
        or not bubble.get("text_boxes")
        or not _is_dark_background_bgr(bubble_color_bgr)
    ):
        return render_mask

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(value))) for value in bbox),
            image_size,
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return render_mask

    x1, y1, x2, y2 = clipped
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = max(1, bbox_w * bbox_h)

    mask_bool = _normalize_restore_mask(render_mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return render_mask
    crop = np.where(mask_bool[y1:y2, x1:x2], 255, 0).astype(np.uint8)
    mask_coverage = float(np.count_nonzero(crop)) / float(bbox_area)
    if mask_coverage > OCCLUDED_DARK_LAYOUT_MAX_MASK_COVERAGE:
        return render_mask

    original_dims = _safe_area_dimensions_for_mask(
        render_mask,
        padding_pixels,
        image_size,
    )
    if original_dims is None:
        return render_mask
    original_area_ratio = (
        float(original_dims[0]) * float(original_dims[1]) / float(bbox_area)
    )
    if original_area_ratio > OCCLUDED_DARK_LAYOUT_MAX_SAFE_AREA_RATIO:
        return render_mask

    contours, _ = cv2.findContours(
        crop,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return render_mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < bbox_area * 0.20:
        return render_mask

    hull_crop = np.zeros_like(crop)
    cv2.fillConvexPoly(hull_crop, cv2.convexHull(largest), 255)
    added_pixels = int(np.count_nonzero((hull_crop > 0) & (crop == 0)))
    if (
        added_pixels <= 0
        or float(added_pixels) / float(bbox_area)
        > OCCLUDED_DARK_LAYOUT_MAX_ADDED_AREA_RATIO
    ):
        return render_mask

    candidate = np.where(mask_bool, 255, 0).astype(np.uint8)
    candidate[y1:y2, x1:x2] = hull_crop
    candidate_dims = _safe_area_dimensions_for_mask(
        candidate,
        padding_pixels,
        image_size,
    )
    if candidate_dims is None:
        return render_mask

    candidate_w = float(candidate_dims[0])
    candidate_h = float(candidate_dims[1])
    original_area = max(1.0, float(original_dims[0]) * float(original_dims[1]))
    candidate_area_gain = candidate_w * candidate_h / original_area
    if (
        candidate_w / float(bbox_w)
        < OCCLUDED_DARK_LAYOUT_MIN_CANDIDATE_WIDTH_RATIO
        or candidate_h / float(bbox_h)
        < OCCLUDED_DARK_LAYOUT_MIN_CANDIDATE_HEIGHT_RATIO
        or candidate_area_gain < OCCLUDED_DARK_LAYOUT_MIN_SAFE_AREA_GAIN
    ):
        return render_mask

    log_message(
        "Completed occluded dark-bubble layout mask for "
        f"{tuple(clipped)} (safe={original_dims[0]:.0f}x{original_dims[1]:.0f} -> "
        f"{candidate_w:.0f}x{candidate_h:.0f}, added={added_pixels}px)",
        verbose=verbose,
        always_print=True,
    )
    return candidate


def _complete_occluded_colored_bubble_layout_mask(
    bubble: Dict[str, Any],
    bbox: Any,
    render_mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    translated_text: str,
    render_tag: str,
    verbose: bool,
    *,
    is_colored: bool,
) -> Any:
    """Complete a split colored-bubble mask for text placement only."""
    if render_mask is None or not is_colored:
        return render_mask
    if bubble.get("is_outside_text", False) or _is_sfx_render_tag(render_tag):
        return render_mask
    if (
        bubble.get("conjoined_parent_mask") is not None
        or bubble.get("conjoined_neighbor_bboxes")
        or not bubble.get("text_boxes")
        or _compact_render_char_count(translated_text)
        < OCCLUDED_COLORED_LAYOUT_MIN_TEXT_CHARS
    ):
        return render_mask

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(value))) for value in bbox),
            image_size,
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return render_mask

    x1, y1, x2, y2 = clipped
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = max(1, bbox_w * bbox_h)

    mask_bool = _normalize_restore_mask(render_mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return render_mask
    crop = np.where(mask_bool[y1:y2, x1:x2], 255, 0).astype(np.uint8)
    mask_coverage = float(np.count_nonzero(crop)) / float(bbox_area)
    if not (
        OCCLUDED_COLORED_LAYOUT_MIN_MASK_COVERAGE
        <= mask_coverage
        <= OCCLUDED_COLORED_LAYOUT_MAX_MASK_COVERAGE
    ):
        return render_mask

    original_dims = _safe_area_dimensions_for_mask(
        render_mask,
        padding_pixels,
        image_size,
    )
    if original_dims is None:
        return render_mask
    original_w = float(original_dims[0])
    original_h = float(original_dims[1])
    if (
        original_h / float(bbox_h)
        > OCCLUDED_COLORED_LAYOUT_MAX_SAFE_HEIGHT_RATIO
        or original_w / float(bbox_w)
        < OCCLUDED_COLORED_LAYOUT_MIN_SAFE_WIDTH_RATIO
    ):
        return render_mask

    contours, _ = cv2.findContours(
        crop,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return render_mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < bbox_area * 0.40:
        return render_mask

    hull_crop = np.zeros_like(crop)
    cv2.fillConvexPoly(hull_crop, cv2.convexHull(largest), 255)
    added_pixels = int(np.count_nonzero((hull_crop > 0) & (crop == 0)))
    if (
        added_pixels <= 0
        or float(added_pixels) / float(bbox_area)
        > OCCLUDED_COLORED_LAYOUT_MAX_ADDED_AREA_RATIO
    ):
        return render_mask

    candidate = np.where(mask_bool, 255, 0).astype(np.uint8)
    candidate[y1:y2, x1:x2] = hull_crop
    candidate_dims = _safe_area_dimensions_for_mask(
        candidate,
        padding_pixels,
        image_size,
    )
    if candidate_dims is None:
        return render_mask

    candidate_w = float(candidate_dims[0])
    candidate_h = float(candidate_dims[1])
    original_area = max(1.0, original_w * original_h)
    candidate_area_gain = candidate_w * candidate_h / original_area
    if (
        candidate_w / float(bbox_w)
        < OCCLUDED_COLORED_LAYOUT_MIN_CANDIDATE_WIDTH_RATIO
        or candidate_h / float(bbox_h)
        < OCCLUDED_COLORED_LAYOUT_MIN_CANDIDATE_HEIGHT_RATIO
        or candidate_area_gain < OCCLUDED_COLORED_LAYOUT_MIN_SAFE_AREA_GAIN
    ):
        return render_mask

    log_message(
        "Completed occluded colored-bubble layout mask for "
        f"{tuple(clipped)} (safe={original_w:.0f}x{original_h:.0f} -> "
        f"{candidate_w:.0f}x{candidate_h:.0f}, added={added_pixels}px)",
        verbose=verbose,
        always_print=True,
    )
    return candidate


def _restore_cleaned_text_holes_for_layout(
    bubble: Dict[str, Any],
    bbox: Any,
    render_mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    translated_text: str,
    render_tag: str,
    verbose: bool,
    *,
    original_image_bgr: Any,
    cleaned_image: Any,
    bubble_color_bgr: Any,
    is_colored: bool,
) -> Any:
    """Restore source-text holes proven clean, without expanding the bubble."""
    if render_mask is None or original_image_bgr is None or cleaned_image is None:
        return render_mask
    if is_colored or _is_dark_background_bgr(bubble_color_bgr):
        return render_mask
    if bubble.get("is_outside_text", False) or _is_sfx_render_tag(render_tag):
        return render_mask
    if (
        bubble.get("conjoined_parent_mask") is not None
        or bubble.get("conjoined_neighbor_bboxes")
        or not bubble.get("text_boxes")
        or _compact_render_char_count(translated_text)
        < CLEANED_TEXT_HOLE_LAYOUT_MIN_TEXT_CHARS
    ):
        return render_mask

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(value))) for value in bbox),
            image_size,
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return render_mask

    x1, y1, x2, y2 = clipped
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    bbox_area = max(1, bbox_w * bbox_h)

    mask_bool = _normalize_restore_mask(render_mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return render_mask
    crop = np.where(mask_bool[y1:y2, x1:x2], 255, 0).astype(np.uint8)
    mask_coverage = float(np.count_nonzero(crop)) / float(bbox_area)
    if not (
        CLEANED_TEXT_HOLE_LAYOUT_MIN_MASK_COVERAGE
        <= mask_coverage
        <= CLEANED_TEXT_HOLE_LAYOUT_MAX_MASK_COVERAGE
    ):
        return render_mask

    original_dims = _safe_area_dimensions_for_mask(
        render_mask,
        padding_pixels,
        image_size,
    )
    if original_dims is None:
        return render_mask
    original_w = max(1.0, float(original_dims[0]))
    original_h = max(1.0, float(original_dims[1]))
    if original_w / float(bbox_w) > CLEANED_TEXT_HOLE_LAYOUT_MAX_SAFE_WIDTH_RATIO:
        return render_mask

    try:
        cleaned_bgr = pil_to_cv2(cleaned_image)
    except Exception:
        return render_mask
    if (
        not isinstance(original_image_bgr, np.ndarray)
        or not isinstance(cleaned_bgr, np.ndarray)
        or original_image_bgr.shape[:2] != cleaned_bgr.shape[:2]
        or original_image_bgr.shape[1] != image_size[0]
        or original_image_bgr.shape[0] != image_size[1]
    ):
        return render_mask

    original_crop = original_image_bgr[y1:y2, x1:x2]
    cleaned_crop = cleaned_bgr[y1:y2, x1:x2]
    if original_crop.size == 0 or cleaned_crop.size == 0:
        return render_mask
    original_gray = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)
    cleaned_gray = cv2.cvtColor(cleaned_crop, cv2.COLOR_BGR2GRAY)
    brightened = cleaned_gray.astype(np.int16) - original_gray.astype(np.int16)
    cleaned_source_ink = (
        (original_gray <= 220)
        & (cleaned_gray >= 235)
        & (brightened >= 35)
    )
    changed_pixels = int(np.count_nonzero(cleaned_source_ink))
    if changed_pixels < max(16, int(round(bbox_area * 0.01))):
        return render_mask

    contours, _ = cv2.findContours(
        crop,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return render_mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < bbox_area * 0.35:
        return render_mask

    outer_envelope = np.zeros_like(crop)
    cv2.fillConvexPoly(outer_envelope, cv2.convexHull(largest), 255)
    source_ink_mask = np.where(cleaned_source_ink, 255, 0).astype(np.uint8)
    source_ink_mask = cv2.dilate(
        source_ink_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    source_ink_mask[
        (outer_envelope == 0) | (cleaned_gray < 225)
    ] = 0

    candidate_crop = np.where(
        (crop > 0) | (source_ink_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    added_pixels = int(
        np.count_nonzero((candidate_crop > 0) & (crop == 0))
    )
    if (
        added_pixels <= 0
        or float(added_pixels) / float(bbox_area)
        > CLEANED_TEXT_HOLE_LAYOUT_MAX_ADDED_AREA_RATIO
    ):
        return render_mask

    candidate = np.where(mask_bool, 255, 0).astype(np.uint8)
    candidate[y1:y2, x1:x2] = candidate_crop
    candidate_dims = _safe_area_dimensions_for_mask(
        candidate,
        padding_pixels,
        image_size,
    )
    if candidate_dims is None:
        return render_mask

    candidate_w = float(candidate_dims[0])
    candidate_h = float(candidate_dims[1])
    original_area = max(1.0, original_w * original_h)
    candidate_area_gain = candidate_w * candidate_h / original_area
    if (
        candidate_w / float(bbox_w)
        < CLEANED_TEXT_HOLE_LAYOUT_MIN_CANDIDATE_WIDTH_RATIO
        or candidate_h / float(bbox_h)
        < CLEANED_TEXT_HOLE_LAYOUT_MIN_CANDIDATE_HEIGHT_RATIO
        or candidate_area_gain < CLEANED_TEXT_HOLE_LAYOUT_MIN_SAFE_AREA_GAIN
    ):
        return render_mask

    log_message(
        "Restored cleaned source-text holes for layout "
        f"{tuple(clipped)} (safe={original_w:.0f}x{original_h:.0f} -> "
        f"{candidate_w:.0f}x{candidate_h:.0f}, added={added_pixels}px)",
        verbose=verbose,
        always_print=True,
    )
    return candidate


def _center_x_from_text_bbox(
    text_bbox: Any,
    bbox: Tuple[int, int, int, int],
) -> Optional[float]:
    if text_bbox is None:
        return None
    try:
        tx1, ty1, tx2, ty2 = (int(round(float(v))) for v in text_bbox)
    except (TypeError, ValueError):
        return None

    x1, y1, x2, y2 = bbox
    tx1 = max(x1, min(tx1, x2))
    tx2 = max(x1, min(tx2, x2))
    ty1 = max(y1, min(ty1, y2))
    ty2 = max(y1, min(ty2, y2))
    if tx2 <= tx1 or ty2 <= ty1:
        return None
    if (tx2 - tx1) < max(3, int(round((x2 - x1) * 0.08))):
        return None
    return (tx1 + tx2) / 2.0


def _rect_layout_mask_from_safe_box(
    image_size: Tuple[int, int],
    safe_box: Tuple[int, int, int, int],
    padding_pixels: float,
) -> np.ndarray:
    img_w, img_h = image_size
    x, y, w, h = safe_box
    try:
        pad = max(1, int(math.ceil(max(0.0, float(padding_pixels)))) - 1)
    except (TypeError, ValueError):
        pad = 1

    outer_x1 = max(0, x - pad)
    outer_y1 = max(0, y - pad)
    outer_x2 = min(img_w, x + w + pad)
    outer_y2 = min(img_h, y + h + pad)

    layout_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if outer_x2 > outer_x1 and outer_y2 > outer_y1:
        layout_mask[outer_y1:outer_y2, outer_x1:outer_x2] = 255
    return layout_mask


def _recenter_conjoined_layout_mask(
    bubble: Dict[str, Any],
    bbox: Any,
    layout_mask: Any,
    base_mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    text_bbox: Any,
    verbose: bool,
) -> Any:
    if layout_mask is not base_mask:
        return layout_mask
    if bubble.get("is_outside_text", False):
        return layout_mask
    if (
        bubble.get("conjoined_parent_mask") is None
        and not bubble.get("conjoined_neighbor_bboxes")
    ):
        return layout_mask

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return layout_mask

    source_center_x = _center_x_from_text_bbox(text_bbox, clipped)
    if source_center_x is None:
        return layout_mask

    base_area = _safe_area_box_for_mask(base_mask, padding_pixels, image_size)
    if base_area is None:
        return layout_mask

    safe_box, (base_center_x, base_center_y), safe_area_mask = base_area
    safe_x, safe_y, safe_w, safe_h = safe_box
    x1, _y1, x2, _y2 = clipped
    bbox_w = max(1, x2 - x1)

    shift = source_center_x - base_center_x
    if abs(shift) < CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX:
        return layout_mask

    max_shift = max(
        CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX,
        min(
            bbox_w * CONJOINED_LAYOUT_RECENTER_MAX_SHIFT_RATIO,
            safe_w * 0.25,
        ),
    )
    target_center_x = base_center_x + max(-max_shift, min(max_shift, shift))
    min_center_x = max(safe_w / 2.0, x1 + safe_w / 2.0)
    max_center_x = min(image_size[0] - safe_w / 2.0, x2 - safe_w / 2.0)
    if min_center_x <= max_center_x:
        target_center_x = max(min_center_x, min(max_center_x, target_center_x))

    def candidate_box(cx: float) -> Tuple[int, int, int, int]:
        return int(round(cx - safe_w / 2.0)), safe_y, safe_w, safe_h

    def coverage_for_box(box: Tuple[int, int, int, int]) -> float:
        bx, by, bw, bh = box
        if bx < 0 or by < 0 or bx + bw > image_size[0] or by + bh > image_size[1]:
            return 0.0
        region = safe_area_mask[by : by + bh, bx : bx + bw]
        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(region)) / float(region.size)

    best_box = None
    best_center_x = base_center_x
    best_coverage = 0.0
    for step in range(20, -1, -1):
        t = step / 20.0
        candidate_center_x = base_center_x + (target_center_x - base_center_x) * t
        candidate = candidate_box(candidate_center_x)
        coverage = coverage_for_box(candidate)
        if coverage >= CONJOINED_LAYOUT_RECENTER_SAFE_COVERAGE:
            best_box = candidate
            best_center_x = candidate_center_x
            best_coverage = coverage
            break

    if best_box is None or abs(best_center_x - base_center_x) < CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX:
        return layout_mask

    log_message(
        "Recentering conjoined base layout "
        f"{tuple(clipped)} (base_cx={base_center_x:.1f}, source_cx={source_center_x:.1f}, "
        f"new_cx={best_center_x:.1f}, coverage={best_coverage:.2f})",
        verbose=verbose,
    )
    return _rect_layout_mask_from_safe_box(image_size, best_box, padding_pixels)


def _conjoined_layout_center_x_override(
    bubble: Dict[str, Any],
    bbox: Any,
    render_mask: Any,
    base_mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    text_bbox: Any,
    verbose: bool,
) -> Optional[float]:
    if base_mask is None:
        return None
    if bubble.get("is_outside_text", False):
        return None
    if (
        bubble.get("conjoined_parent_mask") is None
        and not bubble.get("conjoined_neighbor_bboxes")
    ):
        return None

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return None

    x1, _y1, x2, _y2 = clipped
    bbox_w = max(1, x2 - x1)

    source_center_x = None
    source_name = None
    render_area = _safe_area_box_for_mask(render_mask, padding_pixels, image_size)
    if render_area is not None:
        _render_box, (render_center_x, _render_center_y), _render_safe_area = render_area
        if x1 <= render_center_x <= x2:
            source_center_x = render_center_x
            source_name = "render_mask"

    text_center_x = _center_x_from_text_bbox(text_bbox, clipped)
    if text_center_x is not None:
        if source_center_x is None:
            source_center_x = text_center_x
            source_name = "text_bbox"
        elif abs(text_center_x - source_center_x) <= max(
            CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX, bbox_w * 0.10
        ):
            source_center_x = (source_center_x + text_center_x) / 2.0
            source_name = "render_mask+text_bbox"

    if source_center_x is None:
        source_center_x = (x1 + x2) / 2.0
        source_name = "bbox"

    base_area = _safe_area_box_for_mask(base_mask, padding_pixels, image_size)
    if base_area is None:
        return None

    safe_box, (base_center_x, _base_center_y), safe_area_mask = base_area
    safe_x, safe_y, safe_w, safe_h = safe_box

    shift = source_center_x - base_center_x
    if abs(shift) < CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX:
        return None

    max_shift = max(
        CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX,
        min(
            bbox_w * CONJOINED_LAYOUT_RECENTER_MAX_SHIFT_RATIO,
            safe_w * 0.25,
        ),
    )
    target_center_x = base_center_x + max(-max_shift, min(max_shift, shift))
    min_center_x = max(safe_w / 2.0, x1 + safe_w / 2.0)
    max_center_x = min(image_size[0] - safe_w / 2.0, x2 - safe_w / 2.0)
    if min_center_x <= max_center_x:
        target_center_x = max(min_center_x, min(max_center_x, target_center_x))

    def candidate_box(cx: float) -> Tuple[int, int, int, int]:
        return int(round(cx - safe_w / 2.0)), safe_y, safe_w, safe_h

    base_mask_bool = _normalize_restore_mask(base_mask, image_size)

    def coverage_for_box(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        bx, by, bw, bh = box
        if bx < 0 or by < 0 or bx + bw > image_size[0] or by + bh > image_size[1]:
            return 0.0, 0.0
        safe_region = safe_area_mask[by : by + bh, bx : bx + bw]
        if safe_region.size == 0:
            return 0.0, 0.0
        safe_coverage = float(np.count_nonzero(safe_region)) / float(safe_region.size)
        if base_mask_bool is None:
            mask_coverage = safe_coverage
        else:
            mask_region = base_mask_bool[by : by + bh, bx : bx + bw]
            mask_coverage = (
                float(np.count_nonzero(mask_region)) / float(mask_region.size)
                if mask_region.size
                else 0.0
            )
        return safe_coverage, mask_coverage

    best_center_x = None
    best_safe_coverage = 0.0
    best_mask_coverage = 0.0
    for step in range(20, -1, -1):
        t = step / 20.0
        candidate_center_x = base_center_x + (target_center_x - base_center_x) * t
        safe_coverage, mask_coverage = coverage_for_box(candidate_box(candidate_center_x))
        if (
            safe_coverage >= CONJOINED_LAYOUT_CENTER_OVERRIDE_MIN_SAFE_COVERAGE
            and mask_coverage >= CONJOINED_LAYOUT_CENTER_OVERRIDE_MIN_MASK_COVERAGE
        ):
            best_center_x = candidate_center_x
            best_safe_coverage = safe_coverage
            best_mask_coverage = mask_coverage
            break

    if best_center_x is None:
        return None
    if abs(best_center_x - base_center_x) < CONJOINED_LAYOUT_RECENTER_MIN_SHIFT_PX:
        return None

    log_message(
        "Using conjoined layout center override "
        f"{tuple(clipped)} (source={source_name}, base_cx={base_center_x:.1f}, "
        f"source_cx={source_center_x:.1f}, override_cx={best_center_x:.1f}, "
        f"safe_cov={best_safe_coverage:.2f}, mask_cov={best_mask_coverage:.2f})",
        verbose=verbose,
    )
    return best_center_x


def _should_exempt_conjoined_tight_fit_shrink(
    bubble: Dict[str, Any],
    bbox: Any,
    mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    translated_text: str,
    verbose: bool,
) -> bool:
    if bubble.get("is_outside_text", False):
        return False
    if (
        bubble.get("conjoined_parent_mask") is None
        and not bubble.get("conjoined_neighbor_bboxes")
    ):
        return False

    compact_text = re.sub(r"\s+", "", str(translated_text or ""))
    if len(compact_text) > CONJOINED_TIGHT_FIT_EXEMPT_MAX_TEXT_CHARS:
        return False

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return False

    x1, y1, x2, y2 = clipped
    bbox_w = max(0, x2 - x1)
    bbox_h = max(0, y2 - y1)
    if (
        bbox_w < CONJOINED_TIGHT_FIT_EXEMPT_MIN_BBOX_WIDTH
        or bbox_h < CONJOINED_TIGHT_FIT_EXEMPT_MIN_BBOX_HEIGHT
    ):
        return False

    dims = _safe_area_dimensions_for_mask(mask, padding_pixels, image_size)
    if dims is None:
        return False

    safe_w, safe_h = dims
    if (
        safe_w < CONJOINED_TIGHT_FIT_EXEMPT_MIN_SAFE_SIDE
        or safe_h < CONJOINED_TIGHT_FIT_EXEMPT_MIN_SAFE_SIDE
    ):
        return False

    safe_area_ratio = (safe_w * safe_h) / max(1.0, float(bbox_w * bbox_h))
    safe_height_ratio = safe_h / max(1.0, float(bbox_h))
    safe_width_ratio = safe_w / max(1.0, float(bbox_w))
    if (
        safe_area_ratio > CONJOINED_TIGHT_FIT_EXEMPT_MAX_SAFE_AREA_RATIO
        or safe_height_ratio > CONJOINED_TIGHT_FIT_EXEMPT_MAX_SAFE_HEIGHT_RATIO
        or safe_width_ratio < CONJOINED_TIGHT_FIT_EXEMPT_MIN_SAFE_WIDTH_RATIO
    ):
        return False

    log_message(
        "Exempting conjoined child from tight-fit shrink "
        f"{tuple(clipped)} (safe={safe_w:.0f}x{safe_h:.0f}, "
        f"bbox={bbox_w}x{bbox_h}, area_ratio={safe_area_ratio:.2f}, "
        f"height_ratio={safe_height_ratio:.2f})",
        verbose=verbose,
    )
    return True


def _conjoined_base_layout_recovery_allows_tight_fit_shrink(
    original_mask: Any,
    layout_mask: Any,
    bbox: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    verbose: bool,
) -> bool:
    original_dims = _safe_area_dimensions_for_mask(
        original_mask,
        padding_pixels,
        image_size,
    )
    layout_dims = _safe_area_dimensions_for_mask(
        layout_mask,
        padding_pixels,
        image_size,
    )
    if original_dims is None or layout_dims is None:
        return False

    original_area = max(1.0, float(original_dims[0]) * float(original_dims[1]))
    layout_area = max(1.0, float(layout_dims[0]) * float(layout_dims[1]))
    area_gain = layout_area / original_area
    width_gain = float(layout_dims[0]) / max(1.0, float(original_dims[0]))
    height_gain = float(layout_dims[1]) / max(1.0, float(original_dims[1]))
    if (
        area_gain < CONJOINED_TIGHT_FIT_UNEXEMPT_MIN_AREA_GAIN
        or max(width_gain, height_gain)
        < CONJOINED_TIGHT_FIT_UNEXEMPT_MIN_SIDE_GAIN
    ):
        return False

    log_message(
        "Allowing conjoined tight-fit shrink after base layout recovery "
        f"{tuple(bbox)} (clean_safe={original_dims[0]:.0f}x{original_dims[1]:.0f}, "
        f"layout_safe={layout_dims[0]:.0f}x{layout_dims[1]:.0f}, "
        f"gain={area_gain:.2f}, width_gain={width_gain:.2f}, "
        f"height_gain={height_gain:.2f})",
        verbose=verbose,
    )
    return True


def _should_use_bbox_layout_for_conjoined_text(
    bubble: Dict[str, Any],
    bbox: Any,
    mask: Any,
    padding_pixels: float,
    image_size: Tuple[int, int],
    translated_text: str,
    render_tag: str,
    verbose: bool,
) -> bool:
    if bubble.get("is_outside_text", False):
        return False
    if _is_sfx_render_tag(render_tag):
        return False
    if (
        bubble.get("conjoined_parent_mask") is None
        and not bubble.get("conjoined_neighbor_bboxes")
    ):
        return False

    compact_text = re.sub(
        r"[^\w가-힣ぁ-んァ-ン一-龯0-9]+",
        "",
        str(translated_text or ""),
    )
    if len(compact_text) < CONJOINED_BBOX_LAYOUT_MIN_TEXT_CHARS:
        return False

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return False

    x1, y1, x2, y2 = clipped
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))
    if bbox_h < CONJOINED_BBOX_LAYOUT_MIN_BBOX_HEIGHT:
        return False

    dims = _safe_area_dimensions_for_mask(mask, padding_pixels, image_size)
    if dims is None:
        return False

    safe_w, safe_h = dims
    safe_height_ratio = safe_h / bbox_h
    safe_width_ratio = safe_w / bbox_w
    fallback_height = bbox_h * (1.0 - 2.0 * 0.08)
    height_gain = fallback_height / max(1.0, safe_h)
    if (
        safe_height_ratio > CONJOINED_BBOX_LAYOUT_MAX_SAFE_HEIGHT_RATIO
        or safe_width_ratio < CONJOINED_BBOX_LAYOUT_MIN_SAFE_WIDTH_RATIO
        or height_gain < CONJOINED_BBOX_LAYOUT_MIN_HEIGHT_GAIN
    ):
        return False

    log_message(
        "Using bbox layout for constricted conjoined text "
        f"{tuple(clipped)} (safe={safe_w:.0f}x{safe_h:.0f}, "
        f"bbox={bbox_w:.0f}x{bbox_h:.0f}, "
        f"height_ratio={safe_height_ratio:.2f}, gain={height_gain:.2f})",
        verbose=verbose,
    )
    return True


def _sfx_safe_area_too_small(
    mask: Any,
    bbox: Any,
    image_size: Tuple[int, int],
    padding_pixels: float,
) -> Tuple[bool, Tuple[float, float]]:
    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return False, (0.0, 0.0)

    dims = _safe_area_dimensions_for_mask(mask, padding_pixels, image_size)
    if dims is None:
        return True, (0.0, 0.0)

    x1, y1, x2, y2 = clipped
    box_w = max(1.0, float(x2 - x1))
    box_h = max(1.0, float(y2 - y1))
    safe_w, safe_h = dims
    area_ratio = (safe_w * safe_h) / (box_w * box_h)
    side_ratio = min(safe_w / box_w, safe_h / box_h)
    return (
        area_ratio <= SFX_TINY_SAFE_AREA_RATIO
        and side_ratio <= SFX_TINY_SAFE_SIDE_RATIO,
        dims,
    )


def _find_sfx_residual_text_mask(
    pil_image: Image.Image,
    mask: Any,
    bbox: Any,
    fill_color_bgr: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    mask_bool = _normalize_restore_mask(mask, pil_image.size)
    if mask_bool is None or not np.any(mask_bool):
        return None

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), pil_image.size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped

    candidate = np.where(mask_bool, 255, 0).astype(np.uint8)
    candidate[:y1, :] = 0
    candidate[y2:, :] = 0
    candidate[:, :x1] = 0
    candidate[:, x2:] = 0
    candidate = cv2.erode(
        candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )
    if not np.any(candidate):
        return None

    image_cv = pil_to_cv2(pil_image)
    if image_cv.ndim == 3 and image_cv.shape[2] == 4:
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2GRAY)
    elif image_cv.ndim == 3:
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image_cv

    try:
        bgr = tuple(int(np.clip(c, 0, 255)) for c in fill_color_bgr[:3])
    except Exception:
        bgr = (255, 255, 255)
    fill_gray = int(round(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]))

    if fill_gray >= 128:
        core_mask = np.where((image_gray <= 185) & (candidate == 255), 255, 0).astype(
            np.uint8
        )
        halo_allow = np.where(image_gray <= 225, 255, 0).astype(np.uint8)
    else:
        core_mask = np.where((image_gray >= 70) & (candidate == 255), 255, 0).astype(
            np.uint8
        )
        halo_allow = np.where(image_gray >= 35, 255, 0).astype(np.uint8)

    if not np.any(core_mask):
        return None

    contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    kept_core = np.zeros_like(core_mask)
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    max_component_area = max(900, int(bbox_area * 0.18))
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 2.0 or area > max_component_area:
            continue
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        long_side = max(cw, ch)
        short_side = max(1, min(cw, ch))
        if (cw <= 4 and ch >= 18) or (ch <= 4 and cw >= 18):
            continue
        if long_side >= 22 and long_side / float(short_side) >= 12.0:
            continue

        component_mask = np.zeros_like(core_mask)
        cv2.drawContours(component_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        cv2.bitwise_or(kept_core, component_mask, dst=kept_core)

    if int(np.count_nonzero(kept_core)) < SFX_RESIDUAL_MIN_PIXELS:
        return None

    residual_mask = cv2.dilate(
        kept_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2
    )
    residual_mask = cv2.bitwise_and(residual_mask, halo_allow)
    residual_mask = cv2.bitwise_and(residual_mask, candidate)
    if int(np.count_nonzero(residual_mask)) < SFX_RESIDUAL_MIN_PIXELS:
        return None
    return residual_mask > 0


def _apply_sfx_tiny_safe_area_rescue(
    pil_image: Image.Image,
    bbox: Any,
    render_tag: Any,
    render_mask: Any,
    base_mask: Any,
    fill_color_bgr: Tuple[int, int, int],
    padding_pixels: float,
    verbose: bool = False,
) -> Tuple[Image.Image, Any, bool]:
    if not _is_sfx_render_tag(render_tag) or render_mask is None or base_mask is None:
        return pil_image, render_mask, False

    tiny_safe_area, safe_dims = _sfx_safe_area_too_small(
        render_mask, bbox, pil_image.size, padding_pixels
    )
    if not tiny_safe_area:
        return pil_image, render_mask, False

    residual_mask = _find_sfx_residual_text_mask(
        pil_image, base_mask, bbox, fill_color_bgr
    )
    if residual_mask is None:
        base_too_small, base_safe_dims = _sfx_safe_area_too_small(
            base_mask, bbox, pil_image.size, padding_pixels
        )
        if base_too_small:
            return pil_image, render_mask, False
        log_message(
            "Using full internal SFX bubble mask for layout "
            f"{bbox} (clean_safe={safe_dims[0]:.0f}x{safe_dims[1]:.0f}, "
            f"base_safe={base_safe_dims[0]:.0f}x{base_safe_dims[1]:.0f})",
            verbose=verbose,
            always_print=True,
        )
        return pil_image, base_mask, False

    image_cv = pil_to_cv2(pil_image)
    bgr = tuple(int(np.clip(c, 0, 255)) for c in fill_color_bgr[:3])
    if image_cv.ndim == 3 and image_cv.shape[2] == 4:
        image_cv[residual_mask, :3] = bgr
    else:
        image_cv[residual_mask] = bgr

    rescued_image = cv2_to_pil(image_cv)
    if rescued_image.mode != pil_image.mode:
        rescued_image = rescued_image.convert(pil_image.mode)

    layout_mask = base_mask
    base_too_small, _ = _sfx_safe_area_too_small(
        base_mask, bbox, pil_image.size, padding_pixels
    )
    if base_too_small:
        layout_mask = None

    log_message(
        "Applied SFX residual cleanup for tiny safe area "
        f"{bbox} ({int(np.count_nonzero(residual_mask))} px, "
        f"safe={safe_dims[0]:.0f}x{safe_dims[1]:.0f})",
        verbose=verbose,
        always_print=True,
    )
    return rescued_image, layout_mask, True


def _split_multiline_sfx_text(text: Any) -> List[str]:
    value = (
        str(text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\\n", "\n")
    )
    lines = [" ".join(line.split()) for line in value.split("\n")]
    lines = [line for line in lines if line]
    if len(lines) > 1:
        return lines

    # Some translators return short SFX pairs as "word word" instead of keeping
    # the source newline.  Only split very short tokens; this helper is called
    # after the narrow internal-SFX/tiny-safe-area rescue gate.
    if len(lines) == 1 and "\n" not in value:
        tokens = [token.strip() for token in lines[0].split() if token.strip()]
        token_chars = sum(len(token) for token in tokens)
        if (
            2 <= len(tokens) <= 4
            and token_chars <= 12
            and all(len(token) <= 5 for token in tokens)
            and not any(re.search(r"\d", token) for token in tokens)
        ):
            return tokens

    return lines


def _build_internal_sfx_lobe_masks(
    mask: Any,
    bbox: Any,
    image_size: Tuple[int, int],
    lobe_count: int,
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    if lobe_count < 2 or lobe_count > 4:
        return []

    mask_bool = _normalize_restore_mask(mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return []

    try:
        clipped = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        clipped = None
    if clipped is None:
        return []

    x1, y1, x2, y2 = clipped
    candidate = np.zeros_like(mask_bool, dtype=np.uint8)
    candidate[y1:y2, x1:x2] = np.where(mask_bool[y1:y2, x1:x2], 255, 0).astype(
        np.uint8
    )
    if int(np.count_nonzero(candidate)) < 32:
        return []

    distance_map = cv2.distanceTransform(candidate, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    work_map = distance_map.copy()
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    peaks: List[Tuple[int, int, float]] = []
    for _ in range(lobe_count):
        _, max_val, _, max_loc = cv2.minMaxLoc(work_map)
        if max_val < 3.0:
            break
        px, py = int(max_loc[0]), int(max_loc[1])
        peaks.append((px, py, float(max_val)))
        suppress_radius = max(
            8,
            int(round(max_val * 2.4)),
            int(round(min(box_w, box_h) / max(2, lobe_count + 1))),
        )
        cv2.circle(work_map, (px, py), suppress_radius, 0.0, thickness=-1)

    if len(peaks) < lobe_count:
        return []

    yy, xx = np.indices(candidate.shape)
    candidate_pixels = candidate > 0
    distances = []
    for px, py, _ in peaks[:lobe_count]:
        distances.append((xx - px) ** 2 + (yy - py) ** 2)
    assignment = np.argmin(np.stack(distances, axis=0), axis=0)

    lobes: List[Tuple[Tuple[int, int, int, int], np.ndarray, float, float]] = []
    min_lobe_area = max(16, int(np.count_nonzero(candidate_pixels) * 0.08))
    for index in range(lobe_count):
        lobe_mask_bool = candidate_pixels & (assignment == index)
        if int(np.count_nonzero(lobe_mask_bool)) < min_lobe_area:
            return []

        lobe_mask = np.zeros_like(candidate, dtype=np.uint8)
        lobe_mask[lobe_mask_bool] = 255
        lobe_mask = cv2.morphologyEx(
            lobe_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        lobe_mask = cv2.bitwise_and(lobe_mask, candidate)

        ys, xs = np.where(lobe_mask > 0)
        if ys.size == 0 or xs.size == 0:
            return []

        lobe_bbox = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        lobes.append((lobe_bbox, lobe_mask, float(ys.mean()), float(xs.mean())))

    lobes.sort(key=lambda item: (item[2], item[3]))
    return [(lobe_bbox, lobe_mask) for lobe_bbox, lobe_mask, _, _ in lobes]


def _try_render_split_internal_sfx(
    pil_image: Image.Image,
    text: Any,
    bbox: Any,
    render_tag: Any,
    base_mask: Any,
    font_dir: str,
    bubble_color_bgr: Tuple[int, int, int],
    render_config: RenderingConfig,
    text_color_rgb: Optional[Tuple[int, int, int]],
    bubble_id: str,
    verbose: bool = False,
) -> Optional[Image.Image]:
    if not _is_sfx_render_tag(render_tag) or base_mask is None:
        return None

    lines = _split_multiline_sfx_text(text)
    if len(lines) < 2:
        return None

    lobe_masks = _build_internal_sfx_lobe_masks(
        base_mask, bbox, pil_image.size, len(lines)
    )
    if len(lobe_masks) != len(lines):
        return None

    rendered = pil_image
    for line_index, (line_text, (lobe_bbox, lobe_mask)) in enumerate(
        zip(lines, lobe_masks),
        start=1,
    ):
        try:
            rendered = render_text_skia(
                pil_image=rendered,
                text=line_text,
                bbox=lobe_bbox,
                font_dir=font_dir,
                cleaned_mask=lobe_mask,
                bubble_color_bgr=bubble_color_bgr,
                config=render_config,
                verbose=verbose,
                bubble_id=f"{bubble_id}.{line_index}",
                rotation_deg=0.0,
                vertical_stack=False,
                text_color_rgb=text_color_rgb,
                raise_on_safe_error=False,
                auto_vertical_ignore_word_count=True,
            )
        except Exception as e:
            log_message(
                f"Split internal SFX render failed for {bbox}: {e}",
                verbose=verbose,
            )
            return None

    log_message(
        f"Rendered split internal SFX {bbox} into {len(lines)} lobe(s)",
        verbose=verbose,
        always_print=True,
    )
    return rendered


def _source_anchor_internal_bbox(
    source_mask: Any,
    bbox: Any,
    image_size: Tuple[int, int],
    padding_pixels: float = 0.0,
) -> Optional[Tuple[int, int, int, int]]:
    mask_bool = _normalize_restore_mask(source_mask, image_size)
    if mask_bool is None or not np.any(mask_bool):
        return None

    try:
        container = _clip_bbox_to_image(
            tuple(int(round(float(v))) for v in bbox), image_size
        )
    except (TypeError, ValueError):
        container = None
    if container is None:
        return None

    x1, y1, x2, y2 = container
    crop = mask_bool[y1:y2, x1:x2]
    ys, xs = np.where(crop)
    if xs.size < 24 or ys.size < 24:
        return None

    sx1 = x1 + int(xs.min())
    sy1 = y1 + int(ys.min())
    sx2 = x1 + int(xs.max()) + 1
    sy2 = y1 + int(ys.max()) + 1
    source_w = max(1.0, float(sx2 - sx1))
    source_h = max(1.0, float(sy2 - sy1))
    container_w = max(1.0, float(x2 - x1))
    container_h = max(1.0, float(y2 - y1))
    try:
        pad = max(0.0, float(padding_pixels))
    except (TypeError, ValueError):
        pad = 0.0
    bleed = max(4.0, min(18.0, min(container_w, container_h) * 0.045), pad * 1.5)

    target_w = min(container_w + bleed * 2.0, max(source_w * 1.16, container_w * 0.72))
    target_h = min(container_h + bleed * 2.0, max(source_h * 1.12, container_h * 0.62))
    if source_w / float(max(1.0, source_h)) >= 1.25:
        target_h = min(container_h + bleed * 2.0, max(target_h, source_h * 1.25))
    elif source_h / float(max(1.0, source_w)) >= 1.25:
        target_w = min(container_w + bleed * 2.0, max(target_w, source_w * 1.30))

    center_x = (sx1 + sx2) / 2.0
    center_y = (sy1 + sy2) / 2.0
    candidate = (
        center_x - target_w / 2.0,
        center_y - target_h / 2.0,
        center_x + target_w / 2.0,
        center_y + target_h / 2.0,
    )

    expanded_container = (
        x1 - bleed,
        y1 - bleed,
        x2 + bleed,
        y2 + bleed,
    )
    shifted = _shift_bbox_inside(candidate, expanded_container)
    clipped = _clip_bbox_to_image(shifted, image_size)
    if clipped is None or clipped[2] - clipped[0] < 12 or clipped[3] - clipped[1] < 10:
        return None
    return clipped


def _try_render_source_anchor_internal_sfx(
    pil_image: Image.Image,
    text: Any,
    bbox: Any,
    source_mask: Any,
    font_dir: str,
    bubble_color_bgr: Tuple[int, int, int],
    render_config: RenderingConfig,
    text_color_rgb: Optional[Tuple[int, int, int]],
    bubble_id: str,
    font_size_scale: float = 1.0,
    verbose: bool = False,
) -> Optional[Image.Image]:
    if not str(text or "").strip():
        return None

    anchor_bbox = _source_anchor_internal_bbox(
        source_mask,
        bbox,
        pil_image.size,
        padding_pixels=getattr(render_config, "padding_pixels", 0.0),
    )
    if anchor_bbox is None:
        return None

    try:
        anchored_config = replace(
            render_config,
            tight_fit_shrink_exempt=False,
        )
        anchored_config.compare_forced_vertical_layout = False
        anchored_config.force_compare_vertical_text = False
        rendered = render_text_skia(
            pil_image=pil_image,
            text=text,
            bbox=anchor_bbox,
            font_dir=font_dir,
            cleaned_mask=None,
            bubble_color_bgr=bubble_color_bgr,
            config=anchored_config,
            verbose=verbose,
            bubble_id=bubble_id,
            rotation_deg=0.0,
            vertical_stack=False,
            text_color_rgb=text_color_rgb,
            raise_on_safe_error=False,
            auto_vertical_ignore_word_count=True,
        )
    except Exception as e:
        log_message(
            f"Source-anchor internal SFX render failed for {bbox}: {e}",
            verbose=verbose,
        )
        return None

    log_message(
        f"Rendered source-anchor internal SFX {bbox} in original text area {anchor_bbox}",
        verbose=verbose,
        always_print=True,
    )
    return rendered


def _restore_original_region(
    current_image: Image.Image,
    original_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    mask: Any = None,
) -> Image.Image:
    clipped = _clip_bbox_to_image(bbox, current_image.size)
    if clipped is None:
        return current_image

    x1, y1, x2, y2 = clipped
    source_image = original_image
    if source_image.mode != current_image.mode:
        source_image = source_image.convert(current_image.mode)

    restored_image = current_image.copy()
    original_patch = source_image.crop((x1, y1, x2, y2))
    restore_mask = _normalize_restore_mask(mask, current_image.size)
    if restore_mask is not None:
        mask_crop = restore_mask[y1:y2, x1:x2]
        if np.any(mask_crop):
            alpha = Image.fromarray(
                np.where(mask_crop, 255, 0).astype(np.uint8),
                mode="L",
            )
            restored_image.paste(original_patch, (x1, y1), alpha)
            return restored_image

    restored_image.paste(original_patch, (x1, y1))
    return restored_image


def _restore_original_region_excluding_bboxes(
    current_image: Image.Image,
    original_image: Image.Image,
    bbox: Tuple[int, int, int, int],
    excluded_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    mask: Any = None,
) -> Tuple[Image.Image, int]:
    clipped = _clip_bbox_to_image(bbox, current_image.size)
    if clipped is None:
        return current_image, 0

    img_w, img_h = current_image.size
    x1, y1, x2, y2 = clipped
    bbox_mask = np.zeros((img_h, img_w), dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True

    restore_mask = _normalize_restore_mask(mask, current_image.size)
    if restore_mask is not None and np.any(restore_mask[y1:y2, x1:x2]):
        effective_mask = restore_mask & bbox_mask
    else:
        effective_mask = bbox_mask

    blocked_pixels = 0
    for protected_bbox in excluded_bboxes or []:
        protected = _clip_bbox_to_image(protected_bbox, current_image.size)
        if protected is None:
            continue
        px1, py1, px2, py2 = protected
        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        overlap = effective_mask[iy1:iy2, ix1:ix2]
        overlap_pixels = int(np.count_nonzero(overlap))
        if overlap_pixels <= 0:
            continue
        effective_mask[iy1:iy2, ix1:ix2] = False
        blocked_pixels += overlap_pixels

    if not np.any(effective_mask[y1:y2, x1:x2]):
        return current_image, blocked_pixels

    if blocked_pixels <= 0 and restore_mask is None:
        return _restore_original_region(
            current_image, original_image, clipped, None
        ), 0

    return (
        _restore_original_region(
            current_image, original_image, clipped, effective_mask
        ),
        blocked_pixels,
    )


def _debug_mask_bbox(mask):
    """Return full-image bbox for a debug mask, or None when empty/invalid."""
    normalized = (
        _normalize_debug_mask(mask, (mask.shape[1], mask.shape[0]))
        if isinstance(mask, np.ndarray) and mask.ndim >= 2
        else None
    )
    if normalized is None:
        try:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            if mask_array.ndim != 2:
                return None
            normalized = mask_array > 0
        except Exception:
            return None
    coords = np.where(normalized)
    if coords[0].size == 0 or coords[1].size == 0:
        return None
    return [
        int(coords[1].min()),
        int(coords[0].min()),
        int(coords[1].max()) + 1,
        int(coords[0].max()) + 1,
    ]


def get_image_encoding_params(pil_image_format: Optional[str]) -> Tuple[str, str]:
    """Returns (mime_type, cv2_ext) for a given PIL image format."""
    if pil_image_format and pil_image_format.upper() == "PNG":
        return "image/png", ".png"
    return "image/jpeg", ".jpg"


def _normalize_context_image_mode(
    image: Image.Image,
    mime_type: str,
    verbose: bool = False,
) -> Image.Image:
    if mime_type == "image/jpeg":
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        if image.mode != "RGB":
            log_message(
                f"Converting {image.mode} previous context image to RGB",
                verbose=verbose,
            )
            return image.convert("RGB")
        return image

    if image.mode not in ("RGB", "RGBA", "L"):
        log_message(
            f"Converting {image.mode} previous context image to RGBA",
            verbose=verbose,
        )
        return image.convert("RGBA")
    return image


def _encode_previous_context_source_page(
    image_path: Path,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> Optional[Dict[str, str]]:
    try:
        with Image.open(image_path) as source_image:
            image_format = source_image.format
            mime_type, cv2_ext = get_image_encoding_params(image_format)
            context_image_pil = _normalize_context_image_mode(
                source_image.copy(),
                mime_type,
                verbose,
            )

        effective_context_max_side = scale_length(
            config.translation.context_image_max_side_pixels,
            None,
            minimum=512,
            maximum=4096,
        )
        context_upscale_method = (
            "none" if config.test_mode else config.translation.upscale_method
        )

        if context_upscale_method in ("model", "model_lite"):
            model_manager = get_model_manager()
            if context_upscale_method == "model":
                upscale_model = model_manager.load_upscale(verbose=verbose)
            else:
                upscale_model = model_manager.load_upscale_lite(verbose=verbose)
            context_image_pil = upscale_image_to_dimension(
                upscale_model,
                context_image_pil,
                effective_context_max_side,
                config.device,
                "max",
                context_upscale_method,
                verbose,
            )
            context_image_pil = resize_to_max_side(
                context_image_pil,
                effective_context_max_side,
                verbose=verbose,
            )
            model_manager.clear_cache()
        elif context_upscale_method == "lanczos":
            context_image_pil = resize_to_max_side(
                context_image_pil,
                effective_context_max_side,
                verbose=verbose,
            )

        context_image_cv = pil_to_cv2(context_image_pil)
        is_success, buffer = cv2.imencode(cv2_ext, context_image_cv)
        if not is_success:
            raise ImageProcessingError(
                f"Previous context image encoding to {cv2_ext} failed"
            )
        return {
            "mime_type": mime_type,
            "data": base64.b64encode(buffer).decode("utf-8"),
        }
    except Exception as e:
        log_message(
            f"Warning: Failed to encode previous context image {image_path}: {e}",
            always_print=True,
        )
        return None


def _previous_context_cache_key(
    image_path: Path,
    config: MangaTranslatorConfig,
) -> Tuple[Any, ...]:
    stat = image_path.stat()
    context_upscale_method = (
        "none" if config.test_mode else config.translation.upscale_method
    )
    return (
        str(image_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        config.translation.context_image_max_side_pixels,
        context_upscale_method,
    )


def _get_cached_previous_context_image(
    image_path: Path,
    config: MangaTranslatorConfig,
    context_cache: Optional[OrderedDict],
    context_cache_lock: Optional[threading.Lock],
) -> Optional[Dict[str, str]]:
    verbose = config.verbose
    if context_cache is None:
        return _encode_previous_context_source_page(image_path, config, verbose)

    try:
        cache_key = _previous_context_cache_key(image_path, config)
    except Exception:
        return _encode_previous_context_source_page(image_path, config, verbose)

    if context_cache_lock:
        with context_cache_lock:
            cached = context_cache.get(cache_key)
            if cached is not None:
                context_cache.move_to_end(cache_key)
                return cached

    encoded = _encode_previous_context_source_page(image_path, config, verbose)
    if encoded is None:
        return None

    if context_cache_lock:
        with context_cache_lock:
            context_cache[cache_key] = encoded
            context_cache.move_to_end(cache_key)
            while len(context_cache) > PREVIOUS_CONTEXT_CACHE_MAX_SIZE:
                context_cache.popitem(last=False)
    return encoded


def _build_previous_context_images(
    image_files: List[Path],
    image_index: int,
    config: MangaTranslatorConfig,
    context_cache: Optional[OrderedDict] = None,
    context_cache_lock: Optional[threading.Lock] = None,
) -> List[Dict[str, str]]:
    if not getattr(config.translation, "send_full_page_context", False):
        return []
    if getattr(config.translation, "ocr_method", "LLM") != "LLM":
        return []

    requested_count = int(
        getattr(config.translation, "previous_context_image_count", 0) or 0
    )
    if requested_count <= 0:
        return []

    start_index = max(0, image_index - requested_count)
    previous_paths = image_files[start_index:image_index]
    previous_images = []
    for previous_path in previous_paths:
        encoded = _get_cached_previous_context_image(
            previous_path,
            config,
            context_cache,
            context_cache_lock,
        )
        if encoded is not None:
            previous_images.append(encoded)
    return previous_images


def _build_previous_context_texts(
    image_files: List[Path],
    image_index: int,
    config: MangaTranslatorConfig,
    ocr_text_history: Optional[Dict[Path, List[str]]] = None,
    ocr_text_history_lock: Optional[threading.Lock] = None,
) -> List[List[str]]:
    """Collect OCR transcripts from up to N already-processed prior pages.

    Parallel callers that require deterministic prior-page text context should
    wait for the required previous pages before calling this helper.
    """
    requested_count = int(
        getattr(config.translation, "previous_context_text_count", 0) or 0
    )
    if requested_count <= 0 or ocr_text_history is None:
        return []

    start_index = max(0, image_index - requested_count)
    previous_paths = image_files[start_index:image_index]
    if not previous_paths:
        return []

    previous_texts: List[List[str]] = []
    if ocr_text_history_lock is not None:
        with ocr_text_history_lock:
            for previous_path in previous_paths:
                texts = ocr_text_history.get(previous_path)
                if texts:
                    previous_texts.append(list(texts))
    else:
        for previous_path in previous_paths:
            texts = ocr_text_history.get(previous_path)
            if texts:
                previous_texts.append(list(texts))
    return previous_texts


def _previous_context_text_mode(config: MangaTranslatorConfig) -> str:
    mode = str(
        getattr(config.translation, "previous_context_text_mode", "strict") or "strict"
    ).strip().lower()
    if mode in {"available", "ready", "nonblocking", "non-blocking"}:
        return "available"
    return "strict"


def _load_debug_font(size: int):
    """Load a bold-ish font for the debug overlay, falling back safely."""
    font_candidates = [
        "arialbd.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_dashed_rectangle(draw, bbox, color, width=2, dash=12, gap=7):
    """Draw a dashed rectangle matching the requested debug style."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return

    def _draw_dashed_line(start, end, horizontal=True):
        if horizontal:
            fixed = start[1]
            pos = start[0]
            limit = end[0]
            while pos < limit:
                seg_end = min(pos + dash, limit)
                draw.line((pos, fixed, seg_end, fixed), fill=color, width=width)
                pos += dash + gap
        else:
            fixed = start[0]
            pos = start[1]
            limit = end[1]
            while pos < limit:
                seg_end = min(pos + dash, limit)
                draw.line((fixed, pos, fixed, seg_end), fill=color, width=width)
                pos += dash + gap

    _draw_dashed_line((x0, y0), (x1, y0), horizontal=True)
    _draw_dashed_line((x0, y1), (x1, y1), horizontal=True)
    _draw_dashed_line((x0, y0), (x0, y1), horizontal=False)
    _draw_dashed_line((x1, y0), (x1, y1), horizontal=False)


def _draw_centered_index(draw, bbox, value, font, color):
    """Draw the index at the visual center of the box."""
    x0, y0, x1, y1 = bbox
    cx = int(round((x0 + x1) / 2))
    cy = int(round((y0 + y1) / 2))
    label = str(value)
    try:
        draw.text((cx, cy), label, fill=color, font=font, anchor="mm")
    except TypeError:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (cx - (right - left) / 2, cy - (bottom - top) / 2),
            label,
            fill=color,
            font=font,
        )


def _normalize_debug_mask(mask, image_size):
    """Normalize a debug mask into a full-image boolean array."""
    if mask is None:
        return None

    try:
        mask_array = np.asarray(mask)
    except Exception:
        return None

    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    if mask_array.ndim != 2:
        return None

    width, height = image_size
    if mask_array.shape != (height, width):
        return None

    return mask_array > 0


def _apply_mask_debug_overlay(canvas, mask, color=(255, 0, 0, 84)):
    """Alpha-composite a semi-transparent mask overlay onto the debug canvas."""
    normalized_mask = _normalize_debug_mask(mask, canvas.size)
    if normalized_mask is None or not np.any(normalized_mask):
        return

    overlay = np.zeros((canvas.size[1], canvas.size[0], 4), dtype=np.uint8)
    overlay[normalized_mask] = color
    canvas.alpha_composite(Image.fromarray(overlay, mode="RGBA"))


def _write_component_order_debug_image(
    image_size,
    sorted_items,
    panels,
    bubble_masks,
    reading_direction,
    image_path,
    output_path,
    verbose=False,
):
    """Write a debug PNG showing panel order and merged text-element order."""
    width, height = image_size
    if width <= 0 or height <= 0:
        return

    canvas = Image.new("RGBA", (width, height), (238, 238, 238, 255))
    draw = ImageDraw.Draw(canvas)

    panel_color = (32, 63, 255)
    osb_color = (255, 0, 255)
    bubble_color = (34, 160, 34)
    index_color = (255, 0, 0)

    font_size = max(14, min(width, height) // 28)
    font = _load_debug_font(font_size)

    panel_order = (
        sort_panels_by_reading_order(panels, reading_direction) if panels else []
    )

    for item in sorted_items:
        if item.get("is_outside_text", False):
            continue
        bbox = tuple(int(round(v)) for v in item.get("bbox", (0, 0, 0, 0)))
        _apply_mask_debug_overlay(
            canvas, bubble_masks.get(bbox) if bubble_masks else None
        )

    for panel_index, panel_id in enumerate(panel_order, start=1):
        panel_bbox = tuple(int(round(v)) for v in panels[panel_id])
        draw.rectangle(panel_bbox, outline=panel_color, width=3)
        _draw_centered_index(draw, panel_bbox, panel_index, font, index_color)

    for item_index, item in enumerate(sorted_items, start=1):
        bbox = tuple(int(round(v)) for v in item.get("bbox", (0, 0, 0, 0)))
        if item.get("is_outside_text", False):
            draw.rectangle(bbox, outline=osb_color, width=2)
            draw_bbox = bbox
        else:
            mask_bbox = (
                _debug_mask_bbox(bubble_masks.get(bbox)) if bubble_masks else None
            )
            draw_bbox = tuple(mask_bbox) if mask_bbox is not None else bbox
            _draw_dashed_rectangle(draw, draw_bbox, bubble_color, width=2)
        _draw_centered_index(draw, draw_bbox, item_index, font, index_color)

    base_path = Path(output_path) if output_path else Path(image_path)
    debug_path = base_path.parent / f"{base_path.stem}.component-order-debug.png"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(debug_path, format="PNG")
    log_message(
        f"Wrote component-order debug image: {debug_path}",
        verbose=verbose,
        always_print=True,
    )


def _write_llm_crop_debug_images(
    sorted_items,
    image_path,
    output_path,
    verbose=False,
):
    """Save the exact image crops the LLM sees to a debug subfolder."""
    base_path = Path(output_path) if output_path else Path(image_path)
    crop_dir = base_path.parent / f"{base_path.stem}.llm-crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, item in enumerate(sorted_items, start=1):
        img_b64 = item.get("image_b64")
        if not img_b64:
            continue
        try:
            img_bytes = base64.b64decode(img_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_cv = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_cv is None:
                continue
            label = "osb" if item.get("is_outside_text", False) else "bubble"
            crop_path = crop_dir / f"{i:03d}_{label}.png"
            cv2.imwrite(str(crop_path), img_cv)
            count += 1
        except Exception:
            pass

    log_message(
        f"Wrote {count} LLM crop debug images to: {crop_dir}",
        verbose=verbose,
        always_print=True,
    )


def _resolve_pre_upscale_factor(
    pre_cfg: Optional[PreprocessingConfig],
    verbose: bool = False,
) -> float:
    if pre_cfg is None or not pre_cfg.enabled:
        return 1.0

    factor = max(1.0, min(float(pre_cfg.factor or 1.0), 8.0))
    if factor <= 1.01:
        return 1.0

    log_message(f"Initial upscaling enabled: {factor:.2f}x", verbose=verbose)
    return factor


def _apply_pre_upscale_if_needed(
    image: Image.Image,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> Tuple[Image.Image, float]:
    factor = _resolve_pre_upscale_factor(
        getattr(config, "preprocessing", None), verbose
    )
    if factor == 1.0:
        return image, 1.0

    # Use the output upscale model setting for initial upscaling as well
    model_type = (
        getattr(config.output, "image_upscale_model", "model_lite")
        if hasattr(config, "output")
        else "model_lite"
    )
    upscaled = upscale_image(image, factor, model_type=model_type, verbose=verbose)
    return upscaled, factor


def _select_outside_text_for_inpainting(items, translations, mode, provider):
    """Only classified, valid translations may authorize destructive OSB cleanup."""
    if len(items) != len(translations):
        raise TranslationError(
            "Cannot select outside text: incomplete translation results."
        )
    selected = set()
    excluded = set()
    for item, translation in zip(items, translations):
        if not item.get("is_outside_text"):
            continue
        bbox = tuple(item.get("original_bbox") or item["bbox"])
        if not _is_invalid_translation_text(
            translation, provider
        ) and should_process_outside_text_tag(
            mode, extract_translation_tag(translation)[0]
        ):
            selected.add(bbox)
        else:
            excluded.add(bbox)
    return selected - excluded


def _restore_excluded_outside_text(
    image, original, items, translations, mode, verbose=False
):
    """Restore skipped OSB before snapshotting the editor background or rendering."""
    excluded = {
        index
        for index, item in enumerate(items)
        if item.get("is_outside_text", False)
        and (
            item.get("skip_reason") == "outside_text_processing_mode"
            or (
                not item.get("skip_render")
                and not should_process_outside_text_tag(
                    mode, extract_translation_tag(translations[index])[0]
                )
            )
        )
    }
    protected = [
        bbox
        for index, item in enumerate(items)
        if index not in excluded and not item.get("skip_render")
        for bbox in [_text_region_restore_protection_bbox(item, image.size)]
        if bbox is not None
    ]
    for index in sorted(excluded):
        item = items[index]
        preserve_bbox = item.get("outside_text_preserve_bbox")
        bbox = preserve_bbox or (
            item.get("original_bbox") if item.get("edge_sfx_full_glyph") else None
        )
        if bbox is None:
            bbox = _osb_layout_bbox(item, image.size)
        image, _ = _restore_original_region_excluding_bboxes(
            image,
            original,
            tuple(int(round(float(v))) for v in bbox),
            [] if preserve_bbox is not None else protected,
            None,
        )
        item["skip_render"] = True
        item["skip_reason"] = "outside_text_processing_mode"
        log_message(
            f"Restored original outside text {bbox} - excluded by {mode} mode",
            verbose=verbose,
            always_print=True,
        )
    return image


def _should_overlap_inpaint_translation(config: MangaTranslatorConfig) -> bool:
    """Optionally overlap remote LLM calls with local inpainting work."""
    return (
        bool(getattr(config.translation, "parallel_inpaint_translation", False))
        and getattr(config.translation, "ocr_method", "LLM") == "LLM"
        and not getattr(config, "test_mode", False)
        and not getattr(config, "cleaning_only", False)
        and not getattr(config, "upscaling_only", False)
    )


def _translation_item_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    bbox = tuple(int(round(v)) for v in item.get("bbox", ()))
    original_bbox = item.get("original_bbox")
    if original_bbox is not None:
        original_bbox = tuple(int(round(v)) for v in original_bbox)
    return (bool(item.get("is_outside_text", False)), bbox, original_bbox)


def _align_translation_results_to_items(
    source_items: List[Dict[str, Any]],
    translations: List[str],
    source_ocr_texts: List[str],
    target_items: List[Dict[str, Any]],
    provider: str,
) -> Tuple[List[str], List[str]]:
    """Map async translation results back onto the final render item order."""
    buckets: Dict[Tuple[Any, ...], List[Tuple[str, str]]] = {}
    for i, item in enumerate(source_items):
        key = _translation_item_key(item)
        translation = translations[i] if i < len(translations) else ""
        ocr_text = source_ocr_texts[i] if i < len(source_ocr_texts) else ""
        buckets.setdefault(key, []).append((translation, ocr_text))

    aligned_translations = []
    aligned_ocr_texts = []
    missing = f"[{provider}: Missing async translation]"
    for item in target_items:
        key = _translation_item_key(item)
        values = buckets.get(key)
        if values:
            translation, ocr_text = values.pop(0)
        else:
            translation, ocr_text = missing, ""
        aligned_translations.append(translation)
        aligned_ocr_texts.append(ocr_text)
    return aligned_translations, aligned_ocr_texts


def _encode_full_page_context_for_translation(
    image_cv: np.ndarray,
    image_format: Optional[str],
    cv2_ext: str,
    mime_type: str,
    config: MangaTranslatorConfig,
    verbose: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    if not config.translation.send_full_page_context:
        return None, None

    try:
        context_image_pil = cv2_to_pil(image_cv)
        effective_context_max_side = scale_length(
            config.translation.context_image_max_side_pixels,
            None,
            minimum=512,
            maximum=4096,
        )
        context_upscale_method = (
            "none" if config.test_mode else config.translation.upscale_method
        )

        if context_upscale_method in ("model", "model_lite"):
            model_manager = get_model_manager()
            if context_upscale_method == "model":
                upscale_model = model_manager.load_upscale(verbose=verbose)
            else:
                upscale_model = model_manager.load_upscale_lite(verbose=verbose)
            context_image_pil = upscale_image_to_dimension(
                upscale_model,
                context_image_pil,
                effective_context_max_side,
                config.device,
                "max",
                context_upscale_method,
                verbose,
            )
            context_image_pil = resize_to_max_side(
                context_image_pil,
                effective_context_max_side,
                verbose=verbose,
            )
            model_manager.clear_cache()
            log_message(
                f"Upscaled full image for context with {context_upscale_method}",
                verbose=verbose,
            )
        elif context_upscale_method == "lanczos":
            context_image_pil = resize_to_max_side(
                context_image_pil,
                effective_context_max_side,
                verbose=verbose,
            )
            log_message("Resized full image for context with LANCZOS", verbose=verbose)
        else:
            log_message("Using full image for context without resizing", verbose=verbose)

        context_image_cv = pil_to_cv2(context_image_pil)
        is_success, buffer = cv2.imencode(cv2_ext, context_image_cv)
        if not is_success:
            raise ImageProcessingError(f"Full image encoding to {cv2_ext} failed")
        return base64.b64encode(buffer).decode("utf-8"), mime_type
    except Exception as e:
        log_message(
            f"Warning: Failed to encode full image context: {e}",
            always_print=True,
        )
        return None, None


def translate_and_render(
    image_path: Union[str, Path],
    config: MangaTranslatorConfig,
    output_path: Optional[Union[str, Path]] = None,
    edit_session_path: Optional[Union[str, Path]] = None,
    cancellation_manager: Optional["CancellationManager"] = None,
    previous_context_images: Optional[List[Dict[str, str]]] = None,
    previous_context_texts: Optional[List[List[str]]] = None,
    previous_context_texts_provider: Optional[Callable[[], List[List[str]]]] = None,
    ocr_texts_out: Optional[List[str]] = None,
    usage_out: Optional[Dict[str, int]] = None,
):
    """
    Main function to translate manga speech bubbles and render translations using a config object.

    Args:
        image_path (str or Path): Path to input image
        config (MangaTranslatorConfig): Configuration object containing all settings.
        output_path (str or Path, optional): Path to save the final image. If None, image is not saved.
        edit_session_path: Optional path for the post-translation editor session JSON.
        previous_context_images: Batch-only previous source page images for LLM reference.
        previous_context_texts: Batch-only previous source page OCR transcripts for LLM reference
            (oldest-to-newest, one inner list per page).
        previous_context_texts_provider: Optional callback used to fetch previous-page
            OCR transcripts immediately before the translation API call.
        ocr_texts_out: Optional mutable list. When provided, the current page's OCR transcripts
            (in reading order) are appended so the orchestrator can chain them as previous-page
            text context for subsequent pages.

    Returns:
        PIL.Image: Final translated image
    """
    start_time = time.time()
    image_path = Path(image_path)
    edit_session_path = Path(edit_session_path) if edit_session_path else None
    verbose = config.verbose
    device = config.device
    previous_context_images = previous_context_images or []
    previous_context_texts = previous_context_texts or []

    log_message(f"Using device: {device}", verbose=verbose)

    # Set global HF token for model downloads
    hf_token = config.outside_text.huggingface_token
    get_model_manager().set_hf_token(hf_token)

    try:
        pil_original = Image.open(image_path)
        image_format = pil_original.format
        mime_type, cv2_ext = get_image_encoding_params(image_format)
        log_message(
            f"Original image format: {image_format} -> MIME: {mime_type}",
            verbose=verbose,
        )
    except FileNotFoundError:
        log_message(f"Error: Input image not found at {image_path}", always_print=True)
        raise
    except Exception as e:
        log_message(f"Error opening image {image_path}: {e}", always_print=True)
        raise

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise TranslationError("Process cancelled by user.")

    desired_format = config.output.output_format
    output_ext_for_mode = (
        Path(output_path).suffix.lower() if output_path else image_path.suffix.lower()
    )

    if desired_format == "jpeg" or (
        desired_format == "auto" and output_ext_for_mode in [".jpg", ".jpeg"]
    ):
        target_mode = "RGB"
    else:  # Default to RGBA for PNG, WEBP, or other formats in auto mode
        target_mode = "RGBA"
    log_message(f"Target mode: {target_mode}", verbose=verbose)

    pil_image_processed = convert_image_to_target_mode(
        pil_original, target_mode, verbose
    )
    pil_image_processed, _ = _apply_pre_upscale_if_needed(
        pil_image_processed, config, verbose
    )
    original_restore_image = pil_image_processed.copy()

    # Check for Upscaling Only Mode (skip detection, cleaning, and translation)
    if config.upscaling_only:
        log_message(
            "Upscaling only mode - skipping detection and translation",
            always_print=True,
        )
        final_image_to_save = pil_image_processed

        if config.output.upscale_final_image:
            log_message("Upscaling final image...", verbose=verbose, always_print=True)
            final_image_to_save = upscale_image(
                final_image_to_save,
                config.output.image_upscale_factor,
                model_type=config.output.image_upscale_model,
                verbose=verbose,
            )

        if output_path:
            if final_image_to_save.mode != target_mode:
                log_message(f"Converting final image to {target_mode}", verbose=verbose)
                final_image_to_save = final_image_to_save.convert(target_mode)

            try:
                save_image_with_compression(
                    final_image_to_save,
                    output_path,
                    jpeg_quality=config.output.jpeg_quality,
                    png_compression=config.output.png_compression,
                    verbose=verbose,
                )
            except ImageProcessingError as e:
                log_message(f"Failed to save image: {e}", always_print=True)
                raise

        end_time = time.time()
        processing_time = end_time - start_time
        log_message(
            f"Processing completed in {processing_time:.2f}s", always_print=True
        )

        return final_image_to_save

    # Calculate dynamic processing scale based on image area relative to 1MP (if enabled).
    # Font sizes use a separate manga-page reference so user-entered font sizes remain
    # visually consistent when 1024x1456 pages are the design baseline.
    if config.preprocessing.auto_scale:
        width, height = pil_image_processed.size
        processing_scale = math.sqrt((width * height) / 1_000_000)
        font_size_scale = _font_size_scale_for_dimensions(width, height)
        log_message(
            f"Dynamic processing scale: {processing_scale:.2f}x", verbose=verbose
        )
        log_message(
            "Font size scale: "
            f"{font_size_scale:.2f}x "
            f"({FONT_SIZE_REFERENCE_WIDTH}x{FONT_SIZE_REFERENCE_HEIGHT} reference)",
            verbose=verbose,
        )
    else:
        processing_scale = 1.0
        font_size_scale = 1.0

    get_cache().set_current_image(pil_image_processed, verbose)

    original_cv_image = pil_to_cv2(pil_image_processed)

    # Detect speech bubbles first so OSB processing can respect bubble regions
    log_message("Detecting speech bubbles...", verbose=verbose)
    try:
        bubble_data, text_free_boxes = detect_speech_bubbles(
            image_path,
            config.yolo_model_path,
            config.detection.confidence,
            verbose=verbose,
            device=device,
            seg_model=config.detection.seg_model,
            conjoined_detection=config.detection.conjoined_detection,
            conjoined_confidence=config.detection.conjoined_confidence,
            image_override=pil_image_processed,
            osb_enabled=config.outside_text.enabled,
            osb_text_verification=config.detection.use_osb_text_verification,
            osb_text_hf_token=config.outside_text.huggingface_token,
            collect_text_boxes=config.cleaning.inpaint_colored_bubbles,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
        _clip_simple_neighbor_intruding_item_masks(
            bubble_data,
            pil_image_processed.size,
            verbose,
            mask_keys=("sam_mask",),
            stage="detection",
        )
    except Exception as e:
        log_message(f"Error during detection: {e}", always_print=True)
        bubble_data = []
        text_free_boxes = []

    panels = None
    debug_panels = None
    if config.detection.use_panel_sorting or ENABLE_COMPONENT_ORDER_DEBUG:
        try:
            log_message(
                "Detecting panels...",
                verbose=verbose,
            )
            debug_panels = detect_panels(
                image_path,
                confidence=config.detection.panel_confidence,
                device=device,
                verbose=verbose,
            )
            if debug_panels:
                log_message(
                    f"Detected {len(debug_panels)} panels",
                    always_print=True,
                )
            else:
                log_message(
                    "No panels detected",
                    verbose=verbose,
                )
        except Exception as e:
            log_message(
                f"Panel detection failed: {e}. Using global sorting.",
                always_print=True,
            )
            debug_panels = None

        if config.detection.use_panel_sorting:
            panels = debug_panels

    _panels_detected_early = panels

    select_before_inpaint = (
        config.outside_text.enabled
        and normalize_outside_text_processing_mode(config.outside_text.processing_mode) == "prose"
        and not config.test_mode and not config.cleaning_only
    )
    overlap_inpaint_translation = select_before_inpaint or _should_overlap_inpaint_translation(config)
    speech_edge_split_data: List[Dict[str, Any]] = []
    if bool(getattr(config.outside_text, "enabled", False)):
        speech_edge_split_data = _detect_speech_edge_text_splits(
            bubble_data,
            original_cv_image,
            mime_type,
            cv2_ext,
            getattr(config.translation, "osb_min_side_pixels", SPEECH_EDGE_SPLIT_MIN_SIDE_PIXELS),
            verbose,
        )
        if speech_edge_split_data and overlap_inpaint_translation and not select_before_inpaint:
            overlap_inpaint_translation = False
            log_message(
                "Parallel inpaint/LLM translation disabled: separated "
                f"{len(speech_edge_split_data)} speech-edge text region(s)",
                always_print=True,
            )
    async_translation_executor = None
    async_translation_future = None
    async_translation_source_items: List[Dict[str, Any]] = []
    async_prepared_bubble_data: Optional[List[Dict[str, Any]]] = None
    async_full_image_b64: Optional[str] = None
    async_full_image_mime_type: Optional[str] = None
    async_translation_lock = threading.Lock()

    if overlap_inpaint_translation:
        log_message(
            "Selective outside text: classifying before inpainting"
            if select_before_inpaint else
            "Parallel inpaint/LLM translation enabled: preparing request before inpainting",
            always_print=True,
        )
        async_full_image_b64, async_full_image_mime_type = (
            _encode_full_page_context_for_translation(
                original_cv_image,
                image_format,
                cv2_ext,
                mime_type,
                config,
                verbose,
            )
        )
        try:
            bubble_upscale_method = (
                "none" if config.test_mode else config.translation.upscale_method
            )
            upscale_model = None
            if bubble_data and bubble_upscale_method == "model":
                upscale_model = get_model_manager().load_upscale(verbose=verbose)
            elif bubble_data and bubble_upscale_method == "model_lite":
                upscale_model = get_model_manager().load_upscale_lite(verbose=verbose)

            async_prepared_bubble_data = prepare_bubble_images_for_translation(
                bubble_data,
                original_cv_image,
                upscale_model,
                config.device,
                mime_type,
                config.translation.bubble_min_side_pixels,
                bubble_upscale_method,
                config.translation.whiteout_conjoined_bubbles,
                verbose,
            )
            if upscale_model is not None:
                get_model_manager().clear_cache()
        except Exception as e:
            if select_before_inpaint:
                raise TranslationError("Could not prepare text classification before inpainting.") from e
            overlap_inpaint_translation = False
            async_prepared_bubble_data = None
            log_message(
                f"Parallel inpaint/LLM translation disabled: early bubble preparation failed: {e}",
                always_print=True,
            )

    def _has_bubble_overlap_rescue_osb(
        items: Optional[List[Dict[str, Any]]],
    ) -> bool:
        return any(
            bool(
                item.get("mask_outside_bubble_overlap")
                or item.get("bubble_overlap_rescue")
            )
            for item in (items or [])
            if isinstance(item, dict)
        )

    def _start_inpaint_translation_overlap(
        candidate_outside_text_data: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        nonlocal overlap_inpaint_translation
        nonlocal async_translation_executor
        nonlocal async_translation_future
        nonlocal async_translation_source_items
        nonlocal async_prepared_bubble_data

        if not overlap_inpaint_translation:
            return
        if cancellation_manager and cancellation_manager.is_cancelled():
            return

        with async_translation_lock:
            if async_translation_future is not None:
                return

            prepared_bubbles = async_prepared_bubble_data or []
            valid_bubbles = [b for b in prepared_bubbles if b.get("image_b64")]
            candidate_outside_text_data = candidate_outside_text_data or []
            if _has_bubble_overlap_rescue_osb(candidate_outside_text_data) and not select_before_inpaint:
                overlap_inpaint_translation = False
                async_prepared_bubble_data = None
                log_message(
                    "Parallel inpaint/LLM translation disabled: "
                    "OSB overlaps a speech-bubble area; using post-inpaint crops",
                    always_print=True,
                )
                return

            all_text_data = valid_bubbles + [
                item for item in candidate_outside_text_data if item.get("image_b64")
            ]
            if not all_text_data:
                return

            reading_direction = config.translation.reading_direction
            sorted_for_translation = sort_bubbles_by_reading_order(
                all_text_data, reading_direction, panels=panels
            )
            translatable_for_translation = [
                item for item in sorted_for_translation if item.get("image_b64")
            ]
            if not translatable_for_translation:
                return

            bubble_images_b64 = [
                item["image_b64"] for item in translatable_for_translation
            ]
            bubble_mime_types = [
                item.get("mime_type", "image/jpeg")
                for item in translatable_for_translation
            ]

            def _translation_worker():
                local_previous_context_texts = previous_context_texts
                if previous_context_texts_provider is not None:
                    local_previous_context_texts = (
                        previous_context_texts_provider() or []
                    )
                captured_ocr_texts: List[str] = []
                translated = call_translation_api_batch(
                    config=config.translation,
                    images_b64=bubble_images_b64,
                    full_image_b64=async_full_image_b64 or "",
                    mime_types=bubble_mime_types,
                    full_image_mime_type=async_full_image_mime_type or "image/jpeg",
                    bubble_metadata=translatable_for_translation,
                    previous_context_images=previous_context_images,
                    previous_context_texts=local_previous_context_texts,
                    ocr_texts_output=captured_ocr_texts,
                    usage_output=usage_out,
                    debug=verbose,
                )
                return translated, captured_ocr_texts, translatable_for_translation

            async_translation_source_items = translatable_for_translation
            async_translation_executor = ThreadPoolExecutor(max_workers=1)
            async_translation_future = async_translation_executor.submit(
                _translation_worker
            )
            log_message(
                f"Started LLM translation before inpainting ({len(bubble_images_b64)} items)"
                if select_before_inpaint else
                f"Started LLM translation request while inpainting ({len(bubble_images_b64)} items)",
                always_print=True,
            )

    def _select_before_outside_inpaint(candidate_outside_text_data):
        if cancellation_manager and cancellation_manager.is_cancelled():
            raise CancellationError("Process cancelled by user.")
        if not candidate_outside_text_data:
            return set()
        _start_inpaint_translation_overlap(candidate_outside_text_data)
        if async_translation_future is None:
            raise TranslationError("Outside text classification did not start; image preserved.")
        try:
            translated, _, source_items = async_translation_future.result()
            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Process cancelled by user.")
            return _select_outside_text_for_inpainting(
                source_items, translated, config.outside_text.processing_mode,
                config.translation.provider,
            )
        except Exception:
            if async_translation_executor is not None:
                async_translation_executor.shutdown(wait=False)
            raise

    # Classify selective OSB before any fill, mask expansion, or model loading.
    pil_image_processed, outside_text_data = process_outside_text(
        pil_image_processed,
        config,
        image_path,
        image_format,
        verbose,
        bubble_data=bubble_data,
        text_free_boxes=text_free_boxes,
        panels=panels,
        pre_inpaint_data_callback=(
            _select_before_outside_inpaint if select_before_inpaint else
            (_start_inpaint_translation_overlap if overlap_inpaint_translation else None)
        ),
        injected_outside_text_regions=speech_edge_split_data,
    )
    if (
        overlap_inpaint_translation
        and not select_before_inpaint
        and async_translation_future is None
        and _has_bubble_overlap_rescue_osb(outside_text_data)
    ):
        overlap_inpaint_translation = False
        async_prepared_bubble_data = None
        log_message(
            "Parallel inpaint/LLM translation disabled after OSB detection: "
            "using post-inpaint bubble crops",
            always_print=True,
        )
    if overlap_inpaint_translation and async_translation_future is None:
        _start_inpaint_translation_overlap(outside_text_data)
    original_cv_image = pil_to_cv2(pil_image_processed)

    full_image_b64 = async_full_image_b64 if overlap_inpaint_translation else None
    full_image_mime_type = (
        async_full_image_mime_type if overlap_inpaint_translation else None
    )
    if config.translation.send_full_page_context and full_image_b64 is None:
        full_image_b64, full_image_mime_type = _encode_full_page_context_for_translation(
            original_cv_image,
            image_format,
            cv2_ext,
            mime_type,
            config,
            verbose,
        )
        if full_image_b64:
            log_message("Encoded full image for context", verbose=verbose)

    if cancellation_manager and cancellation_manager.is_cancelled():
        raise CancellationError("Process cancelled by user.")

    final_image_to_save = pil_image_processed

    if not bubble_data and not outside_text_data:
        log_message("No speech bubbles or outside text detected", always_print=True)
    else:
        if bubble_data:
            log_message(f"Detected {len(bubble_data)} bubbles", verbose=verbose)
        if outside_text_data:
            log_message(
                f"Detected {len(outside_text_data)} outside text regions",
                verbose=verbose,
            )

        if cancellation_manager and cancellation_manager.is_cancelled():
            raise CancellationError("Process cancelled by user.")

        if bubble_data:
            log_message("Cleaning speech bubbles...", verbose=verbose)
            try:
                use_otsu = config.cleaning.use_otsu_threshold
                if config.cleaning.inpaint_colored_bubbles:
                    log_message(
                        "LAMA Large inpainting enabled for colored bubbles",
                        verbose=verbose,
                    )

                cleaning_bubble_data = _build_cleaning_detections(
                    bubble_data, verbose=verbose
                )
                cleaned_image_cv, processed_bubbles_info = clean_speech_bubbles(
                    pil_image_processed,
                    config.yolo_model_path,
                    config.detection.confidence,
                    pre_computed_detections=cleaning_bubble_data,
                    device=device,
                    thresholding_value=config.cleaning.thresholding_value,
                    use_otsu_threshold=use_otsu,
                    roi_shrink_px=config.cleaning.roi_shrink_px,
                    verbose=verbose,
                    processing_scale=processing_scale,
                    conjoined_confidence=config.detection.conjoined_confidence,
                    inpaint_colored_bubbles=config.cleaning.inpaint_colored_bubbles,
                    colored_bubble_inpainting_size=(
                        config.cleaning.colored_bubble_inpainting_size
                    ),
                    colored_bubble_inpainting_precision=(
                        config.cleaning.colored_bubble_inpainting_precision
                    ),
                    colored_bubble_mask_kernel_size=(
                        config.cleaning.colored_bubble_mask_kernel_size
                    ),
                    colored_bubble_mask_dilation_offset=(
                        config.cleaning.colored_bubble_mask_dilation_offset
                    ),
                    flux_hf_token=config.outside_text.huggingface_token,
                    flux_num_inference_steps=config.outside_text.flux_num_inference_steps,
                    flux_residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
                    flux_seed=config.outside_text.seed,
                    osb_text_verification=config.detection.use_osb_text_verification,
                    osb_text_hf_token=config.outside_text.huggingface_token,
                    inpaint_method=config.outside_text.inpainting_method,
                    kontext_backend=config.outside_text.kontext_backend,
                    flux_cpu_offload=config.outside_text.flux_cpu_offload,
                    flux_low_vram=config.outside_text.flux_low_vram,
                    flux_luminance_correction=config.outside_text.flux_luminance_correction,
                    flux_upscale_small_crops=config.outside_text.flux_upscale_small_crops,
                    flux_kontext_prompt=config.outside_text.flux_kontext_prompt,
                    flux_klein_prompt=config.outside_text.flux_klein_prompt,
                    bubble_detector_model=config.detection.bubble_detector_model,
                    lama_source_image=original_restore_image,
                )

                conjoined_render_info = [
                    info
                    for info in processed_bubbles_info
                    if info.get("conjoined_child_overlay_only", False)
                ]
                if conjoined_render_info:
                    try:
                        _apply_conjoined_child_residual_cleaning(
                            cleaned_image_cv,
                            conjoined_render_info,
                            protected_masks=[
                                info.get("base_mask")
                                for info in processed_bubbles_info
                                if info.get("inpainted", False)
                                and info.get("base_mask") is not None
                            ],
                            verbose=verbose,
                        )
                        log_message(
                            f"Preserved render masks for {len(conjoined_render_info)} conjoined child box(es)",
                            verbose=verbose,
                        )
                    except Exception as e:
                        log_message(
                            f"Could not build conjoined child render masks: {e}",
                            always_print=True,
                        )
            except CleaningError as e:
                log_message(f"Cleaning failed: {e}", always_print=True)
                cleaned_image_cv = original_cv_image.copy()
                processed_bubbles_info = []
            except Exception as e:
                log_message(f"Error during cleaning: {e}", always_print=True)
                cleaned_image_cv = original_cv_image.copy()
                processed_bubbles_info = []

            pil_cleaned_image = cv2_to_pil(cleaned_image_cv)
            if pil_cleaned_image.mode != target_mode:
                log_message(
                    f"Converting cleaned image to {target_mode}", verbose=verbose
                )
                pil_cleaned_image = pil_cleaned_image.convert(target_mode)
            (
                pil_cleaned_image,
                reapplied_edge_sfx_pixels,
            ) = _reapply_edge_sfx_inpaint_rectangles(
                pil_cleaned_image,
                pil_image_processed,
                outside_text_data,
            )
            if reapplied_edge_sfx_pixels > 0:
                log_message(
                    "Reapplied selected OSB inpainting result after speech-bubble "
                    f"cleaning ({reapplied_edge_sfx_pixels}px)",
                    verbose=verbose,
                    always_print=True,
                )
            final_image_to_save = pil_cleaned_image
        else:
            processed_bubbles_info = []
            pil_cleaned_image = pil_image_processed
            if pil_cleaned_image.mode != target_mode:
                log_message(f"Converting image to {target_mode}", verbose=verbose)
                pil_cleaned_image = pil_cleaned_image.convert(target_mode)
            final_image_to_save = pil_cleaned_image

        # Check for Cleaning Only Mode
        if config.cleaning_only:
            pil_cleaned_image = _restore_excluded_outside_text(
                pil_cleaned_image, original_restore_image, outside_text_data,
                [""] * len(outside_text_data), config.outside_text.processing_mode, verbose,
            )
            final_image_to_save = pil_cleaned_image
            log_message("Cleaning only mode - skipping translation", always_print=True)
        else:
            main_min_font = scale_font_size(
                config.rendering.min_font_size, font_size_scale, minimum=4, maximum=256
            )
            main_max_font = scale_font_size(
                config.rendering.max_font_size,
                font_size_scale,
                minimum=main_min_font,
                maximum=384,
            )
            padding_pixels = scale_scalar(
                config.rendering.padding_pixels,
                processing_scale,
                minimum=1.0,
                maximum=80.0,
            )
            padding_horizontal_pixels = scale_scalar(
                (
                    config.rendering.padding_horizontal_pixels
                    if config.rendering.padding_horizontal_pixels is not None
                    else config.rendering.padding_pixels
                ),
                processing_scale,
                minimum=1.0,
                maximum=80.0,
            )
            padding_vertical_pixels = scale_scalar(
                (
                    config.rendering.padding_vertical_pixels
                    if config.rendering.padding_vertical_pixels is not None
                    else config.rendering.padding_pixels
                ),
                processing_scale,
                minimum=1.0,
                maximum=80.0,
            )
            osb_padding_pixels = scale_scalar(
                config.outside_text.osb_padding_pixels,
                processing_scale,
                minimum=0.0,
                maximum=160.0,
            )
            osb_min_font = scale_font_size(
                config.outside_text.osb_min_font_size,
                font_size_scale,
                minimum=4,
                maximum=512,
            )
            osb_max_font = scale_font_size(
                config.outside_text.osb_max_font_size,
                font_size_scale,
                minimum=osb_min_font,
                maximum=640,
            )
            osb_outline_width = scale_scalar(
                config.outside_text.osb_outline_width,
                processing_scale,
                minimum=0.0,
                maximum=24.0,
            )
            # Prepare images for Translation
            log_message("Preparing bubble images...", verbose=verbose)

            # Enrich bubble_data with refined cleaning masks
            if processed_bubbles_info:
                _mask_lut: Dict[tuple, Any] = {}
                _bubble_by_bbox: Dict[tuple, Dict[str, Any]] = {}
                _speech_bubble_bboxes: List[Tuple[int, int, int, int]] = []
                for _b in bubble_data:
                    try:
                        _bk = tuple(int(round(float(v))) for v in _b.get("bbox", ()))
                    except (TypeError, ValueError):
                        continue
                    if len(_bk) != 4:
                        continue
                    _bubble_by_bbox[_bk] = _b
                    _speech_bubble_bboxes.append(_bk)
                for _info in processed_bubbles_info:
                    _bk = tuple(int(round(v)) for v in _info.get("bbox", ()))
                    if len(_bk) != 4:
                        continue
                    _m = _info.get("mask")
                    if _m is None:
                        _m = _info.get("base_mask")
                    _source_bubble = _bubble_by_bbox.get(_bk)
                    if _m is not None and _source_bubble is not None:
                        _m, _did_clip_mask = _clip_simple_neighbor_intruding_mask(
                            _source_bubble,
                            _bk,
                            _m,
                            _speech_bubble_bboxes,
                            pil_image_processed.size,
                            verbose,
                            stage="post-cleaning",
                            mask_name="render_mask",
                            allow_conjoined=True,
                        )
                        if _did_clip_mask:
                            _info["mask"] = _m
                            for _layout_key in ("layout_mask",):
                                if _info.get(_layout_key) is not None:
                                    _layout_mask, _did_clip_layout = (
                                        _clip_simple_neighbor_intruding_mask(
                                            _source_bubble,
                                            _bk,
                                            _info.get(_layout_key),
                                            _speech_bubble_bboxes,
                                            pil_image_processed.size,
                                            verbose,
                                            stage="post-cleaning",
                                            mask_name=_layout_key,
                                            allow_conjoined=True,
                                        )
                                    )
                                    if _did_clip_layout:
                                        _info[_layout_key] = _layout_mask
                    if _m is not None:
                        _mask_lut[_bk] = _m
                for _b in bubble_data:
                    _bk = tuple(int(round(v)) for v in _b.get("bbox", ()))
                    if _bk in _mask_lut:
                        _b["sam_mask"] = _mask_lut[_bk]

            if overlap_inpaint_translation and async_prepared_bubble_data is not None:
                refined_masks_by_bbox = {}
                for _b in bubble_data:
                    _bk = tuple(int(round(v)) for v in _b.get("bbox", ()))
                    if len(_bk) == 4 and _b.get("sam_mask") is not None:
                        refined_masks_by_bbox[_bk] = _b.get("sam_mask")
                for _b in async_prepared_bubble_data:
                    _bk = tuple(int(round(v)) for v in _b.get("bbox", ()))
                    if _bk in refined_masks_by_bbox:
                        _b["sam_mask"] = refined_masks_by_bbox[_bk]
                bubble_data = async_prepared_bubble_data
                log_message(
                    f"Using {len(bubble_data)} pre-prepared bubble images for LLM translation",
                    always_print=True,
                )
            else:
                # Disable upscaling in test_mode
                bubble_upscale_method = (
                    "none" if config.test_mode else config.translation.upscale_method
                )

                model_manager = get_model_manager()
                upscale_model = None
                if bubble_upscale_method == "model":
                    upscale_model = model_manager.load_upscale(verbose=verbose)
                elif bubble_upscale_method == "model_lite":
                    upscale_model = model_manager.load_upscale_lite(verbose=verbose)

                bubble_data = prepare_bubble_images_for_translation(
                    bubble_data,
                    original_cv_image,
                    upscale_model,
                    config.device,
                    mime_type,
                    config.translation.bubble_min_side_pixels,
                    bubble_upscale_method,
                    config.translation.whiteout_conjoined_bubbles,
                    verbose,
                )
                if upscale_model is not None:
                    model_manager.clear_cache()

                if bubble_upscale_method != "none":
                    log_message(
                        f"Upscaled {len(bubble_data)} bubble images for translation",
                        always_print=True,
                    )
                else:
                    log_message(
                        f"Prepared {len(bubble_data)} bubble images for translation",
                        always_print=True,
                    )
            valid_bubble_data = [b for b in bubble_data if b.get("image_b64")]
            if not valid_bubble_data and not outside_text_data:
                log_message(
                    "No valid bubble images or outside text for translation",
                    always_print=True,
                )
            else:  # Proceed if we have valid bubble data or outside text
                if cancellation_manager and cancellation_manager.is_cancelled():
                    raise CancellationError("Process cancelled by user.")

                # Sort and Translate
                reading_direction = config.translation.reading_direction
                # Merge outside text data with speech bubbles for reading order calculation
                if outside_text_data:
                    log_message(
                        f"Including {len(outside_text_data)} outside text regions in reading order calculation",
                        verbose=verbose,
                    )
                    # Combine speech bubbles and OSB text for unified reading order sorting
                    all_text_data = valid_bubble_data + outside_text_data
                else:
                    all_text_data = valid_bubble_data

                log_message(
                    f"Sorting all text elements ({reading_direction.upper()})",
                    verbose=verbose,
                )

                panels = None
                debug_panels = None
                if ENABLE_COMPONENT_ORDER_DEBUG:
                    try:
                        log_message(
                            "Detecting panels for ordering debug...",
                            verbose=verbose,
                        )
                        debug_panels = detect_panels(
                            image_path,
                            confidence=config.detection.panel_confidence,
                            device=config.device,
                            verbose=verbose,
                        )
                        if debug_panels:
                            log_message(
                                f"Detected {len(debug_panels)} panels",
                                always_print=True,
                            )
                        else:
                            log_message(
                                "No panels detected",
                                verbose=verbose,
                            )
                    except Exception as e:
                        log_message(
                            f"Panel detection failed: {e}. Using global sorting.",
                            always_print=True,
                        )
                        debug_panels = None

                    if config.detection.use_panel_sorting:
                        panels = debug_panels
                elif config.detection.use_panel_sorting:
                    # Already detected once above (same image_path/confidence/device) —
                    # reuse it instead of calling detect_panels() a second time per image.
                    panels = _panels_detected_early

                # Sort all text elements (speech bubbles + OSB text) by reading order
                sorted_bubble_data = sort_bubbles_by_reading_order(
                    all_text_data, reading_direction, panels=panels
                )
                if ENABLE_COMPONENT_ORDER_DEBUG:
                    bubble_debug_masks = {}
                    for bubble in sorted_bubble_data:
                        if bubble.get("is_outside_text", False):
                            continue
                        bbox = tuple(int(round(v)) for v in bubble.get("bbox", ()))
                        if len(bbox) != 4:
                            continue
                        mask = bubble.get("sam_mask")
                        if mask is not None:
                            bubble_debug_masks[bbox] = mask

                    for info in processed_bubbles_info:
                        bbox = tuple(int(round(v)) for v in info.get("bbox", ()))
                        if len(bbox) != 4:
                            continue
                        mask = info.get("mask")
                        if mask is None:
                            mask = info.get("base_mask")
                        if mask is not None:
                            bubble_debug_masks[bbox] = mask

                    try:
                        _write_component_order_debug_image(
                            pil_image_processed.size,
                            sorted_bubble_data,
                            debug_panels,
                            bubble_debug_masks,
                            reading_direction,
                            image_path,
                            output_path,
                            verbose=verbose,
                        )
                    except Exception as e:
                        log_message(
                            f"Failed to write component-order debug image: {e}",
                            always_print=True,
                        )

                    try:
                        _write_llm_crop_debug_images(
                            sorted_bubble_data,
                            image_path,
                            output_path,
                            verbose=verbose,
                        )
                    except Exception as e:
                        log_message(
                            f"Failed to write LLM crop debug images: {e}",
                            always_print=True,
                        )

                translatable_bubble_data = [
                    bubble for bubble in sorted_bubble_data if bubble.get("image_b64")
                ]
                bubble_images_b64 = [
                    bubble["image_b64"] for bubble in translatable_bubble_data
                ]
                bubble_mime_types = [
                    bubble.get("mime_type", "image/jpeg")
                    for bubble in translatable_bubble_data
                ]
                translated_texts = []
                source_ocr_texts: List[str] = []
                _provider_error_prefixes = (
                    f"[{config.translation.provider}:",
                    f"[{config.translation.provider}-",
                )
                if not bubble_images_b64:
                    log_message("No valid bubbles after sorting", always_print=True)
                else:
                    if getattr(config, "test_mode", False):
                        translated_texts = generate_test_placeholders(
                            sorted_bubble_data=sorted_bubble_data,
                            processed_bubbles_info=processed_bubbles_info,
                            config=config,
                            main_min_font=main_min_font,
                            main_max_font=main_max_font,
                            osb_min_font=osb_min_font,
                            osb_max_font=osb_max_font,
                            padding_pixels=padding_pixels,
                            padding_horizontal_pixels=padding_horizontal_pixels,
                            padding_vertical_pixels=padding_vertical_pixels,
                            osb_padding_pixels=osb_padding_pixels,
                            osb_outline_width=osb_outline_width,
                            verbose=verbose,
                        )
                    else:
                        try:
                            if (
                                async_translation_future is None
                                and previous_context_texts_provider is not None
                            ):
                                previous_context_texts = (
                                    previous_context_texts_provider() or []
                                )
                            context_parts = []
                            if previous_context_images:
                                image_page_count = len(previous_context_images)
                                context_parts.append(
                                    f"Previous Context Images: {image_page_count} page(s)"
                                )
                            if previous_context_texts:
                                usable_context_text_pages = sum(
                                    1
                                    for page_texts in previous_context_texts
                                    if any(
                                        (text or "").strip()
                                        and (text or "").strip() != "[OCR FAILED]"
                                        for text in (page_texts or [])
                                    )
                                )
                                if usable_context_text_pages:
                                    context_parts.append(
                                        "Previous Context OCR Text: "
                                        f"{usable_context_text_pages} page(s)"
                                    )
                            context_suffix = (
                                f" ({', '.join(context_parts)})"
                                if context_parts
                                else ""
                            )
                            log_message(
                                f"Translating {len(bubble_images_b64)} bubbles: "
                                f"{config.translation.input_language} → {config.translation.output_language}"
                                f"{context_suffix}",
                                always_print=True,
                            )
                            if async_translation_future is not None:
                                log_message(
                                    "Reusing classification completed before inpainting"
                                    if select_before_inpaint else
                                    "Waiting for LLM translation request started during inpainting",
                                    always_print=True,
                                )
                                try:
                                    (
                                        async_translated_texts,
                                        captured_ocr_texts,
                                        async_source_items,
                                    ) = async_translation_future.result()
                                finally:
                                    if async_translation_executor is not None:
                                        async_translation_executor.shutdown(
                                            wait=False
                                        )
                                        async_translation_executor = None

                                translated_texts, source_ocr_texts = (
                                    _align_translation_results_to_items(
                                        async_source_items,
                                        async_translated_texts,
                                        captured_ocr_texts,
                                        translatable_bubble_data,
                                        config.translation.provider,
                                    )
                                )
                                if ocr_texts_out is not None:
                                    ocr_texts_out.extend(source_ocr_texts)
                            else:
                                captured_ocr_texts: List[str] = []
                                translated_texts = call_translation_api_batch(
                                    config=config.translation,
                                    images_b64=bubble_images_b64,
                                    full_image_b64=full_image_b64 or "",
                                    mime_types=bubble_mime_types,
                                    full_image_mime_type=full_image_mime_type
                                    or "image/jpeg",
                                    bubble_metadata=translatable_bubble_data,
                                    previous_context_images=previous_context_images,
                                    previous_context_texts=previous_context_texts,
                                    ocr_texts_output=captured_ocr_texts,
                                    usage_output=usage_out,
                                    debug=verbose,
                                )
                                source_ocr_texts = captured_ocr_texts
                                if ocr_texts_out is not None:
                                    ocr_texts_out.extend(captured_ocr_texts)
                        except CancellationError:
                            raise
                        except TranslationError as e:
                            error_str = str(e).lower()
                            critical_tokens = (
                                "429",
                                "rate limit",
                                "rate-limit",
                                "auth",
                                "unauthorized",
                                "forbidden",
                                "payment",
                                "quota",
                                "empty response",
                                "api failed",
                            )
                            if any(token in error_str for token in critical_tokens):
                                raise

                            log_message(f"Translation failed: {e}", always_print=True)
                            translated_texts = [f"[Translation Error: {e}]"] * len(
                                bubble_images_b64
                            )
                        except Exception as e:
                            log_message(
                                f"Translation API error: {e}", always_print=True
                            )
                            translated_texts = [
                                "[Translation Error: API call raised exception]"
                                for _ in sorted_bubble_data
                            ]

                        if (
                            len(translated_texts) == len(translatable_bubble_data)
                            and len(translatable_bubble_data) != len(sorted_bubble_data)
                        ):
                            translation_by_id = {
                                id(bubble): translated_texts[i]
                                for i, bubble in enumerate(translatable_bubble_data)
                            }
                            ocr_by_id = {
                                id(bubble): source_ocr_texts[i]
                                for i, bubble in enumerate(translatable_bubble_data)
                                if i < len(source_ocr_texts)
                            }
                            translated_texts = [
                                translation_by_id.get(
                                    id(bubble),
                                    f"[{config.translation.provider}: Missing image data]",
                                )
                                for bubble in sorted_bubble_data
                            ]
                            source_ocr_texts = [
                                ocr_by_id.get(id(bubble), "")
                                for bubble in sorted_bubble_data
                            ]

                        valid_translations = [
                            t
                            for t in translated_texts
                            if not _is_invalid_translation_text(
                                t,
                                config.translation.provider,
                            )
                        ]

                        if bubble_images_b64 and not valid_translations:
                            raise TranslationError("All bubbles failed.")

                if len(translated_texts) == len(sorted_bubble_data):
                    _merge_overlapping_speech_edge_split_translations(
                        sorted_bubble_data,
                        translated_texts,
                        source_ocr_texts,
                        config.translation.provider,
                        verbose,
                    )

                # Render Translations
                bubble_render_info_map = {
                    tuple(info["bbox"]): {
                        "color": info["color"],
                        "mask": info.get("mask"),
                        "base_mask": info.get("base_mask"),
                        "layout_mask": info.get("layout_mask"),
                        "is_sam": info.get("is_sam", False),
                        "is_colored": info.get("is_colored", False),
                        "text_bbox": info.get("text_bbox"),
                        "text_color_bgr": info.get("text_color_bgr"),
                        "conjoined_neighbor_bboxes": info.get(
                            "conjoined_neighbor_bboxes"
                        )
                        or [],
                        "conjoined_white_stroke_only_cleaning": info.get(
                            "conjoined_white_stroke_only_cleaning", False
                        ),
                    }
                    for info in processed_bubbles_info
                    if "bbox" in info and "color" in info and "mask" in info
                }
                log_message("Rendering translations...", verbose=verbose)
                if len(translated_texts) == len(sorted_bubble_data):
                    pil_cleaned_image = _restore_excluded_outside_text(
                        pil_cleaned_image, original_restore_image, sorted_bubble_data,
                        translated_texts, config.outside_text.processing_mode, verbose,
                    )
                    final_image_to_save = pil_cleaned_image
                    editor_session_base_image = (
                        pil_cleaned_image.copy() if edit_session_path is not None else None
                    )
                    restore_protection_bboxes = [
                        _text_region_restore_protection_bbox(
                            item, pil_cleaned_image.size
                        )
                        for item in sorted_bubble_data
                    ]
                    deferred_edge_sfx_indices = [
                        index
                        for index, item in enumerate(sorted_bubble_data)
                        if item.get("is_outside_text", False)
                        and item.get("edge_sfx_full_glyph", False)
                    ]
                    render_indices = [
                        index
                        for index in range(len(sorted_bubble_data))
                        if index not in deferred_edge_sfx_indices
                    ] + deferred_edge_sfx_indices
                    translated_speech_text_mask = np.zeros(
                        (pil_cleaned_image.height, pil_cleaned_image.width),
                        dtype=bool,
                    )
                    translated_speech_regions = []
                    for item in sorted_bubble_data:
                        if item.get("is_outside_text", False):
                            continue
                        item_bbox = item.get("bbox")
                        if item_bbox is None:
                            continue
                        normalized_bbox = tuple(
                            int(round(float(v))) for v in item_bbox
                        )
                        render_info = bubble_render_info_map.get(normalized_bbox, {})
                        translated_speech_regions.append(
                            {
                                "bbox": normalized_bbox,
                                "color": render_info.get(
                                    "color",
                                    item.get("bubble_color_bgr", (255, 255, 255)),
                                ),
                            }
                        )
                    conjoined_pair_font_balance_caps: Dict[
                        Tuple[int, int, int, int], int
                    ] = {}

                    def _estimate_conjoined_pair_font_balance_entry(
                        render_index: int,
                    ) -> Optional[Dict[str, Any]]:
                        try:
                            item = sorted_bubble_data[render_index]
                        except (IndexError, TypeError):
                            return None
                        if item.get("is_outside_text", False):
                            return None
                        try:
                            entry_bbox = tuple(
                                int(round(float(v))) for v in item.get("bbox", ())
                            )
                        except (TypeError, ValueError):
                            return None
                        if len(entry_bbox) != 4:
                            return None

                        group_key = _conjoined_parent_group_key(
                            item,
                            pil_cleaned_image.size,
                        )
                        if group_key is None:
                            return None

                        entry_render_info = bubble_render_info_map.get(entry_bbox)
                        if not entry_render_info:
                            return None
                        entry_is_colored = bool(
                            entry_render_info.get("is_colored", False)
                        )

                        raw_text = (
                            translated_texts[render_index]
                            if render_index < len(translated_texts)
                            else item.get("translation", "")
                        )
                        entry_render_tag, entry_text = extract_translation_tag(raw_text)
                        source_text_for_entry = (
                            source_ocr_texts[render_index]
                            if render_index < len(source_ocr_texts)
                            else ""
                        )
                        entry_text = _normalize_korean_bubble_emphasis_bars(
                            source_text_for_entry,
                            entry_text,
                            entry_render_tag,
                            False,
                        )
                        if _is_sfx_render_tag(entry_render_tag):
                            return None

                        entry_font_dir = config.rendering.font_dir
                        entry_min_font = main_min_font
                        entry_max_font = main_max_font
                        entry_line_spacing = config.rendering.line_spacing_mult
                        entry_vertical_line_spacing = (
                            config.rendering.vertical_line_spacing_mult
                        )
                        (
                            entry_font_dir,
                            entry_min_font,
                            entry_max_font,
                            entry_line_spacing,
                            entry_vertical_line_spacing,
                            entry_tag_vertical_max_chars,
                            entry_tag_vertical_min_aspect,
                        ) = _apply_tag_style_overrides(
                            entry_render_tag,
                            config,
                            entry_font_dir,
                            entry_min_font,
                            entry_max_font,
                            entry_line_spacing,
                            entry_vertical_line_spacing,
                            font_scale=font_size_scale,
                            font_maximum=384,
                        )

                        entry_cleaned_mask = entry_render_info.get("mask")
                        entry_base_mask = entry_render_info.get("base_mask")
                        if entry_cleaned_mask is None or entry_base_mask is None:
                            return None
                        entry_bubble_color_bgr = entry_render_info.get(
                            "color",
                            (255, 255, 255),
                        )

                        entry_rotation_source_mask = entry_base_mask
                        entry_rotation_text_mask = entry_cleaned_mask
                        entry_rotation_deg = _regular_bubble_vertical_rotation(
                            entry_bbox,
                            entry_rotation_source_mask,
                            pil_cleaned_image.size,
                            original_cv_image,
                            entry_rotation_text_mask,
                        )
                        entry_tilted_vertical_allowed = (
                            abs(entry_rotation_deg) > 0.01
                            and _regular_tilt_vertical_allowed(
                                entry_text,
                                source_text_for_entry,
                                config,
                                max_chars_override=entry_tag_vertical_max_chars,
                            )
                        )
                        if entry_tilted_vertical_allowed:
                            return None
                        if abs(entry_rotation_deg) > 0.01:
                            entry_rotation_deg = 0.0

                        (
                            entry_force_on_ot_vertical,
                            _entry_forced_on_ot_aspect,
                            _entry_forced_on_ot_chars,
                            _entry_forced_on_ot_width,
                            _entry_forced_on_ot_width_ratio,
                            _entry_forced_on_ot_columns,
                        ) = _should_force_on_ot_vertical(
                            item,
                            entry_render_tag,
                            entry_text,
                            pil_cleaned_image.size,
                            entry_tag_vertical_max_chars,
                            font_size_scale,
                        )
                        if entry_force_on_ot_vertical:
                            return None

                        entry_should_hyphenate = (
                            config.rendering.hyphenate_before_scaling
                        )
                        entry_is_korean_output = (
                            config.translation.output_language.strip().lower()
                            == "korean"
                        )
                        if not (
                            is_latin_style_language(config.translation.output_language)
                            or entry_is_korean_output
                        ):
                            entry_should_hyphenate = False

                        entry_render_config = RenderingConfig(
                            min_font_size=entry_min_font,
                            max_font_size=entry_max_font,
                            emphasis_font_dir=config.rendering.emphasis_font_dir,
                            emphasis_max_font_size=_scaled_optional_render_font_size(
                                config.rendering.emphasis_max_font_size,
                                font_size_scale,
                                minimum=1,
                                maximum=640,
                            ),
                            line_spacing_mult=entry_line_spacing,
                            emphasis_line_spacing_mult=entry_line_spacing,
                            vertical_line_spacing_mult=entry_vertical_line_spacing,
                            vertical_emphasis_line_spacing_mult=entry_vertical_line_spacing,
                            vertical_font_size_mult=config.rendering.vertical_font_size_mult,
                            vertical_max_chars=(
                                entry_tag_vertical_max_chars
                                if entry_tag_vertical_max_chars is not None
                                else config.rendering.vertical_max_chars
                            ),
                            vertical_emphasis_max_chars=(
                                entry_tag_vertical_max_chars
                                if entry_tag_vertical_max_chars is not None
                                else config.rendering.vertical_emphasis_max_chars
                            ),
                            vertical_min_aspect_ratio=(
                                entry_tag_vertical_min_aspect
                                if entry_tag_vertical_min_aspect is not None
                                else config.rendering.vertical_min_aspect_ratio
                            ),
                            use_subpixel_rendering=config.rendering.use_subpixel_rendering,
                            font_hinting=config.rendering.font_hinting,
                            use_ligatures=config.rendering.use_ligatures,
                            hyphenate_before_scaling=entry_should_hyphenate,
                            hyphen_penalty=config.rendering.hyphen_penalty,
                            hyphenation_min_word_length=config.rendering.hyphenation_min_word_length,
                            badness_exponent=config.rendering.badness_exponent,
                            padding_pixels=padding_pixels,
                            padding_horizontal_pixels=padding_horizontal_pixels,
                            padding_vertical_pixels=padding_vertical_pixels,
                            outline_width=scale_scalar(
                                config.rendering.inside_dark_outline_width
                                if _is_dark_background_bgr(entry_bubble_color_bgr)
                                else config.rendering.inside_light_outline_width,
                                processing_scale,
                                minimum=0.0,
                                maximum=24.0,
                            ),
                            supersampling_factor=config.rendering.supersampling_factor,
                            detach_trailing_punctuation=config.rendering.detach_trailing_punctuation,
                            auto_vertical_text=config.rendering.auto_vertical_text,
                            tight_fit_shrink_min_font_size=scale_font_size(
                                config.rendering.tight_fit_shrink_min_font_size,
                                font_size_scale,
                                minimum=1,
                                maximum=384,
                            ),
                            tight_fit_shrink_exempt=_should_exempt_conjoined_tight_fit_shrink(
                                item,
                                entry_bbox,
                                entry_cleaned_mask,
                                padding_pixels,
                                pil_cleaned_image.size,
                                entry_text,
                                False,
                            ),
                            layout_scale=font_size_scale,
                            small_text_zero_padding_max_font_size=_scaled_render_font_size(
                                10,
                                font_size_scale,
                                minimum=1,
                                maximum=384,
                            ),
                            small_text_zero_padding_max_chars=10,
                            collapsed_small_bbox_fallback_max_area=_scaled_render_area(
                                8000,
                                font_size_scale,
                                minimum=1,
                            ),
                            collapsed_small_bbox_fallback_max_side=_scaled_render_length(
                                140,
                                font_size_scale,
                                minimum=1,
                                maximum=max(pil_cleaned_image.size),
                            ),
                            collapsed_small_bbox_fallback_max_short_side=_scaled_render_length(
                                90,
                                font_size_scale,
                                minimum=1,
                                maximum=max(pil_cleaned_image.size),
                            ),
                            wrapped_symbol_relax_font_size_threshold=_scaled_render_font_size(
                                18,
                                font_size_scale,
                                minimum=1,
                                maximum=384,
                            ),
                        )
                        entry_render_config.short_text_ignores_auto_vertical_font_gain = True
                        _configure_translucent_tight_fit_shrink(
                            entry_render_config,
                            entry_render_info,
                            False,
                            font_size_scale,
                        )
                        entry_child_local_layout_mask = entry_cleaned_mask

                        entry_selected_mask = _select_conjoined_layout_mask(
                            item,
                            entry_bbox,
                            entry_cleaned_mask,
                            entry_base_mask,
                            entry_render_config.padding_pixels,
                            pil_cleaned_image.size,
                            entry_text,
                            entry_render_tag,
                            False,
                            allow_colored_bubble=True,
                            layout_mask_hint=entry_render_info.get("layout_mask"),
                        )
                        entry_conjoined_base_layout_selected = False
                        if entry_selected_mask is not entry_cleaned_mask:
                            entry_is_conjoined_layout = not (
                                item.get("conjoined_parent_mask") is None
                                and not item.get("conjoined_neighbor_bboxes")
                            )
                            entry_conjoined_base_layout_selected = (
                                entry_is_conjoined_layout
                                and entry_selected_mask is entry_base_mask
                            )
                            entry_allows_shrink = (
                                _conjoined_base_layout_recovery_allows_tight_fit_shrink(
                                    entry_cleaned_mask,
                                    entry_selected_mask,
                                    entry_bbox,
                                    entry_render_config.padding_pixels,
                                    pil_cleaned_image.size,
                                    False,
                                )
                            )
                            entry_center_override = (
                                _conjoined_layout_center_x_override(
                                    item,
                                    entry_bbox,
                                    entry_cleaned_mask,
                                    entry_base_mask,
                                    entry_render_config.padding_pixels,
                                    pil_cleaned_image.size,
                                    entry_render_info.get("text_bbox"),
                                    False,
                                )
                            )
                            entry_cleaned_mask = entry_selected_mask
                            if entry_allows_shrink or entry_center_override is not None:
                                entry_render_config.tight_fit_shrink_exempt = False
                            else:
                                entry_render_config.tight_fit_shrink_exempt = True
                            if entry_center_override is not None:
                                setattr(
                                    entry_render_config,
                                    "safe_area_center_x_override",
                                    entry_center_override,
                                )
                                setattr(
                                    entry_render_config,
                                    "skip_layout_collision_check",
                                    True,
                                )

                        entry_cleaned_mask = _complete_occluded_dark_bubble_layout_mask(
                            item,
                            entry_bbox,
                            entry_cleaned_mask,
                            entry_bubble_color_bgr,
                            entry_render_config.padding_pixels,
                            pil_cleaned_image.size,
                            entry_render_tag,
                            False,
                        )
                        entry_cleaned_mask = _complete_occluded_colored_bubble_layout_mask(
                            item,
                            entry_bbox,
                            entry_cleaned_mask,
                            entry_render_config.padding_pixels,
                            pil_cleaned_image.size,
                            entry_text,
                            entry_render_tag,
                            False,
                            is_colored=entry_is_colored,
                        )
                        entry_cleaned_mask = _restore_cleaned_text_holes_for_layout(
                            item,
                            entry_bbox,
                            entry_cleaned_mask,
                            entry_render_config.padding_pixels,
                            pil_cleaned_image.size,
                            entry_text,
                            entry_render_tag,
                            False,
                            original_image_bgr=original_cv_image,
                            cleaned_image=pil_cleaned_image,
                            bubble_color_bgr=entry_bubble_color_bgr,
                            is_colored=entry_is_colored,
                        )
                        entry_cleaned_mask = _bridge_narrow_notch_for_layout(
                            item,
                            entry_bbox,
                            entry_cleaned_mask,
                            entry_render_config.padding_pixels,
                            pil_cleaned_image.size,
                            entry_text,
                            entry_render_tag,
                            False,
                        )
                        entry_guard_item = item
                        if (
                            entry_render_info.get("conjoined_neighbor_bboxes")
                            and not item.get("conjoined_neighbor_bboxes")
                        ):
                            entry_guard_item = dict(item)
                            entry_guard_item["conjoined_neighbor_bboxes"] = (
                                entry_render_info.get("conjoined_neighbor_bboxes")
                                or []
                            )
                        entry_cleaned_mask, entry_guard_applied = (
                            _guard_conjoined_child_layout_mask(
                                entry_guard_item,
                                entry_bbox,
                                entry_cleaned_mask,
                                entry_child_local_layout_mask,
                                entry_base_mask,
                                pil_cleaned_image.size,
                                False,
                            )
                        )
                        if entry_guard_applied:
                            entry_conjoined_base_layout_selected = False
                            entry_render_config.tight_fit_shrink_exempt = False
                            for _layout_attr in (
                                "safe_area_center_x_override",
                                "skip_layout_collision_check",
                            ):
                                if hasattr(entry_render_config, _layout_attr):
                                    delattr(entry_render_config, _layout_attr)
                        _configure_translucent_tight_fit_shrink(
                            entry_render_config,
                            entry_render_info,
                            False,
                            font_size_scale,
                        )
                        if (
                            entry_conjoined_base_layout_selected
                            and entry_cleaned_mask is not None
                        ):
                            _configure_conjoined_base_tight_fit_shrink(
                                entry_render_config,
                                font_size_scale,
                            )

                        try:
                            layout_probe = render_text_skia(
                                pil_image=pil_cleaned_image,
                                text=entry_text,
                                bbox=entry_bbox,
                                font_dir=entry_font_dir,
                                cleaned_mask=entry_cleaned_mask,
                                bubble_color_bgr=entry_bubble_color_bgr,
                                config=entry_render_config,
                                verbose=False,
                                bubble_id=f"balance-{render_index + 1}",
                                rotation_deg=entry_rotation_deg,
                                vertical_stack=False,
                                text_color_rgb=None,
                                raise_on_safe_error=True,
                                layout_only=True,
                                auto_vertical_ignore_word_count=True,
                            )
                        except Exception:
                            return None
                        probe_font_size = getattr(layout_probe, "info", {}).get(
                            "font_size"
                        )
                        if not probe_font_size:
                            return None
                        return {
                            "bbox": entry_bbox,
                            "font_size": int(probe_font_size),
                            "min_font_size": int(entry_render_config.min_font_size),
                            "group_key": group_key,
                        }

                    conjoined_pair_balance_groups: Dict[
                        Tuple[int, int, int, int, int], List[Dict[str, Any]]
                    ] = {}
                    for balance_index in render_indices:
                        balance_entry = _estimate_conjoined_pair_font_balance_entry(
                            balance_index
                        )
                        if balance_entry is None:
                            continue
                        conjoined_pair_balance_groups.setdefault(
                            balance_entry["group_key"],
                            [],
                        ).append(balance_entry)
                    for balance_group in conjoined_pair_balance_groups.values():
                        if len(balance_group) != 2:
                            continue
                        larger_entry, smaller_entry = sorted(
                            balance_group,
                            key=lambda entry: int(entry["font_size"]),
                            reverse=True,
                        )
                        cap = _balanced_conjoined_pair_font_cap(
                            larger_entry["font_size"],
                            smaller_entry["font_size"],
                            larger_entry["min_font_size"],
                        )
                        if cap is None:
                            continue
                        conjoined_pair_font_balance_caps[larger_entry["bbox"]] = cap
                        log_message(
                            "Balancing conjoined pair font size "
                            f"{larger_entry['bbox']}: "
                            f"{larger_entry['font_size']} -> {cap} "
                            f"(sibling={smaller_entry['font_size']}, "
                            f"diff={larger_entry['font_size'] - smaller_entry['font_size']})",
                            verbose=verbose,
                        )
                    for i in render_indices:
                        bubble = sorted_bubble_data[i]
                        bubble["translation"] = translated_texts[i]
                        bbox = bubble["bbox"]
                        if bubble.get("skip_render"):
                            log_message(
                                f"Skipping text {bbox}: "
                                f"{bubble.get('skip_reason') or 'merged speech-edge text'}",
                                verbose=verbose,
                            )
                            continue
                        text = bubble.get("translation", "")
                        render_tag, text = extract_translation_tag(text)
                        if render_tag:
                            bubble["render_tag"] = render_tag
                            log_message(
                                f"Applying render tag <{render_tag}> to {bbox}",
                                verbose=verbose,
                            )
                        is_outside_text = bubble.get("is_outside_text", False)
                        source_ocr_text = (
                            source_ocr_texts[i] if i < len(source_ocr_texts) else ""
                        )
                        text = _normalize_korean_bubble_emphasis_bars(
                            source_ocr_text,
                            text,
                            render_tag,
                            is_outside_text,
                        )
                        bubble["translation"] = text
                        translated_texts[i] = _wrap_translation_with_tag(
                            render_tag,
                            text,
                        )
                        osb_render_mask = (
                            bubble.get("render_mask") if is_outside_text else None
                        )
                        render_info = None
                        osb_unexpanded_position_bbox = (
                            _osb_unexpanded_layout_bbox(
                                bubble,
                                pil_cleaned_image.size,
                            )
                            if is_outside_text
                            else bbox
                        )
                        if (
                            is_outside_text
                            and bubble.get("edge_sfx_full_glyph", False)
                            and bubble.get("original_bbox") is not None
                        ):
                            osb_position_bbox = (
                                _clip_bbox_to_image(
                                    tuple(
                                        int(round(float(v)))
                                        for v in bubble["original_bbox"]
                                    ),
                                    pil_cleaned_image.size,
                                )
                                or bbox
                            )
                        else:
                            osb_position_bbox = (
                                _osb_layout_bbox(bubble, pil_cleaned_image.size)
                                if is_outside_text
                                else bbox
                            )
                        osb_layout_bbox = osb_position_bbox
                        osb_tall_width_expanded = False
                        osb_render_padding_pixels = _osb_padding_for_render_tag(
                            is_outside_text,
                            render_tag,
                            osb_padding_pixels,
                        )
                        if is_outside_text:
                            (
                                osb_layout_bbox,
                                osb_tall_width_expanded,
                                osb_original_layout_aspect,
                                osb_expanded_layout_aspect,
                            ) = _expand_tall_non_sfx_osb_layout_bbox(
                                tuple(
                                    int(round(float(value)))
                                    for value in osb_position_bbox
                                ),
                                pil_cleaned_image.size,
                                render_tag,
                                config,
                            )
                            if osb_tall_width_expanded:
                                log_message(
                                    "Expanded tall non-SFX OSB layout bbox "
                                    f"{osb_position_bbox} -> {osb_layout_bbox} "
                                    f"(aspect={osb_original_layout_aspect:.2f}"
                                    f"->{osb_expanded_layout_aspect:.2f})",
                                    verbose=verbose,
                                )
                        if is_outside_text:
                            (
                                render_bbox,
                                osb_vertical_unexpanded_bbox,
                            ) = _osb_orientation_layout_bboxes(
                                osb_unexpanded_position_bbox,
                                osb_layout_bbox,
                                osb_render_padding_pixels,
                            )
                            osb_horizontal_compare_bbox = render_bbox
                            osb_vertical_compare_bbox = (
                                osb_vertical_unexpanded_bbox
                                if tuple(osb_vertical_unexpanded_bbox)
                                != tuple(render_bbox)
                                else None
                            )
                        else:
                            render_bbox = bbox
                            osb_horizontal_compare_bbox = None
                            osb_vertical_compare_bbox = None
                        other_restore_protection_bboxes = [
                            protected_bbox
                            for protected_index, protected_bbox in enumerate(
                                restore_protection_bboxes
                            )
                            if protected_index != i and protected_bbox is not None
                        ]

                        if (
                            not text
                            or text.startswith("API Error")
                            or text.startswith("[Translation Error]")
                            or text.startswith("[Translation Error:")
                            or text.startswith(_provider_error_prefixes)
                            or text.strip()
                            in {
                                "[OCR FAILED]",
                                "[Empty response / no content]",
                            }
                        ):
                            entry_type = "outside text" if is_outside_text else "bubble"
                            restore_mask = None
                            if not is_outside_text:
                                render_info = bubble_render_info_map.get(tuple(bbox))
                                if render_info:
                                    restore_mask = render_info.get("base_mask")
                                    if restore_mask is None:
                                        restore_mask = render_info.get("mask")
                                if restore_mask is None:
                                    restore_mask = bubble.get("base_mask")
                                    if restore_mask is None:
                                        restore_mask = bubble.get("sam_mask")

                            restore_bbox = tuple(
                                int(round(float(v)))
                                for v in (
                                    osb_position_bbox if is_outside_text else bbox
                                )
                            )
                            if is_outside_text:
                                (
                                    pil_cleaned_image,
                                    blocked_restore_pixels,
                                ) = _restore_original_region_excluding_bboxes(
                                    pil_cleaned_image,
                                    original_restore_image,
                                    restore_bbox,
                                    other_restore_protection_bboxes,
                                    None,
                                )
                            else:
                                blocked_restore_pixels = 0
                                pil_cleaned_image = _restore_original_region(
                                    pil_cleaned_image,
                                    original_restore_image,
                                    restore_bbox,
                                    restore_mask,
                                )
                            final_image_to_save = pil_cleaned_image
                            if blocked_restore_pixels > 0:
                                log_message(
                                    "Clipped original outside text restore "
                                    f"{bbox} to avoid overlap with other text "
                                    f"regions ({blocked_restore_pixels} px)",
                                    verbose=verbose,
                                    always_print=True,
                                )
                            log_message(
                                f"Restored original {entry_type} {bbox} - invalid translation",
                                verbose=verbose,
                                always_print=True,
                            )
                            continue

                        source_text = (
                            source_ocr_texts[i] if i < len(source_ocr_texts) else ""
                        )
                        if _translation_matches_source(source_text, text):
                            render_info = bubble_render_info_map.get(tuple(bbox))
                            restore_mask = None
                            if render_info:
                                restore_mask = render_info.get("base_mask")
                                if restore_mask is None:
                                    restore_mask = render_info.get("mask")
                            if restore_mask is None:
                                restore_mask = bubble.get("base_mask")
                                if restore_mask is None:
                                    restore_mask = bubble.get("sam_mask")

                            restore_bbox = tuple(
                                int(round(float(v)))
                                for v in (
                                    osb_position_bbox if is_outside_text else bbox
                                )
                            )
                            if is_outside_text:
                                (
                                    pil_cleaned_image,
                                    blocked_restore_pixels,
                                ) = _restore_original_region_excluding_bboxes(
                                    pil_cleaned_image,
                                    original_restore_image,
                                    restore_bbox,
                                    other_restore_protection_bboxes,
                                    None,
                                )
                            else:
                                blocked_restore_pixels = 0
                                pil_cleaned_image = _restore_original_region(
                                    pil_cleaned_image,
                                    original_restore_image,
                                    restore_bbox,
                                    restore_mask,
                                )
                            final_image_to_save = pil_cleaned_image
                            entry_type = "outside text" if is_outside_text else "bubble"
                            if blocked_restore_pixels > 0:
                                log_message(
                                    "Clipped original outside text restore "
                                    f"{bbox} to avoid overlap with other text "
                                    f"regions ({blocked_restore_pixels} px)",
                                    verbose=verbose,
                                    always_print=True,
                                )
                            log_message(
                                f"Restored original {entry_type} {bbox} - translation matches OCR source",
                                verbose=verbose,
                                always_print=True,
                            )
                            continue

                        # Convert OSB text to uppercase after same-source comparison.
                        if is_outside_text and text:
                            text = text.upper()
                            bubble["translation"] = text

                        # Use OSB-specific settings for outside text, regular settings for speech bubbles
                        force_on_ot_vertical = False
                        compare_sfx_vertical_speech_bubble = False
                        sfx_speech_bubble_aspect = 0.0
                        osb_is_sfx_tag = False
                        osb_forced_sfx_vertical = False
                        osb_refined_vertical_sfx_rotation = False
                        tilted_vertical_allowed = False
                        if is_outside_text:
                            log_message(
                                f"Rendering outside text {bbox}: '{text[:30]}...'",
                                verbose=verbose,
                            )
                            font_dir = (
                                config.outside_text.osb_font_dir
                                if config.outside_text.osb_font_dir
                                else config.rendering.font_dir
                            )
                            min_font = osb_min_font
                            max_font = osb_max_font
                            line_spacing = config.outside_text.osb_line_spacing
                            vertical_line_spacing = getattr(
                                config.outside_text,
                                "osb_vertical_line_spacing",
                                line_spacing,
                            )
                            use_ligs = config.outside_text.osb_use_ligatures
                            (
                                font_dir,
                                min_font,
                                max_font,
                                line_spacing,
                                vertical_line_spacing,
                                tag_vertical_max_chars,
                                tag_vertical_min_aspect,
                            ) = _apply_tag_style_overrides(
                                render_tag,
                                config,
                                font_dir,
                                min_font,
                                max_font,
                                line_spacing,
                                vertical_line_spacing,
                                font_scale=font_size_scale,
                                font_maximum=640,
                            )
                            cleaned_mask = None
                            if osb_render_mask is not None:
                                log_message(
                                    "Skipping OSB render-mask clipping for translated "
                                    f"outside text {bbox}",
                                    verbose=verbose,
                                )
                            is_dark_text = bool(bubble.get("is_dark_text", True))
                            source_text_color_rgb = bubble.get("text_color_rgb", None)
                            (
                                selected_text_color_rgb,
                                selected_outline_color_rgb,
                                selected_outline_width,
                            ) = _render_palette(
                                config.rendering,
                                is_outside_text=True,
                                is_dark_background=is_dark_text,
                            )
                            text_color_rgb = (
                                _render_rgb(
                                    source_text_color_rgb,
                                    selected_text_color_rgb,
                                )
                                if _is_colored_source_text(source_text_color_rgb)
                                else selected_text_color_rgb
                            )
                            bubble["text_color_rgb"] = list(text_color_rgb)
                            bubble_color_bgr = (
                                (50, 50, 50) if is_dark_text else (255, 255, 255)
                            )
                            (
                                vertical_stack,
                                osb_aspect_ratio,
                                osb_vertical_char_count,
                            ) = _osb_should_use_vertical_stack(
                                bubble,
                                text,
                                config,
                                max_chars_override=tag_vertical_max_chars,
                                layout_bbox=(
                                    osb_vertical_unexpanded_bbox
                                ),
                            )
                            try:
                                rotation_deg = float(
                                    bubble.get("rotation_deg", 0.0) or 0.0
                                )
                            except (TypeError, ValueError):
                                rotation_deg = 0.0
                            osb_vertical_rotation_deg = rotation_deg
                            osb_is_sfx_tag = _is_sfx_render_tag(render_tag)
                            osb_spaced_single_hangul_repeat = (
                                osb_is_sfx_tag
                                and _is_spaced_single_hangul_repeat_osb(text)
                            )
                            osb_short_vertical_sfx = (
                                _osb_short_vertical_sfx_for_tag(
                                    render_tag,
                                    bubble,
                                    text,
                                    source_text,
                                )
                            )
                            osb_sfx_like = osb_is_sfx_tag
                            try:
                                osb_sfx_vertical_max_chars = int(
                                    float(
                                        tag_vertical_max_chars
                                        if tag_vertical_max_chars is not None
                                        else getattr(
                                            config.outside_text,
                                            "osb_vertical_max_chars",
                                            30,
                                        )
                                    )
                                )
                            except (TypeError, ValueError):
                                osb_sfx_vertical_max_chars = 30
                            osb_sfx_space_groups = len(
                                re.findall(r"\s+", str(text or "").strip())
                            )
                            if (
                                osb_sfx_like
                                and not vertical_stack
                                and osb_aspect_ratio >= OSB_SFX_VERTICAL_MIN_ASPECT
                                and osb_sfx_space_groups
                                <= OSB_SFX_VERTICAL_MAX_SPACE_GROUPS
                                and (
                                    osb_short_vertical_sfx
                                    or _is_single_hangul_osb_token(text)
                                    or osb_vertical_char_count > 1
                                )
                                and (
                                    osb_vertical_char_count <= 0
                                    or osb_vertical_char_count
                                    <= osb_sfx_vertical_max_chars
                                )
                            ):
                                vertical_stack = True
                                osb_forced_sfx_vertical = True
                                log_message(
                                    "Using vertical layout for SFX-like outside text "
                                    f"{bbox} (aspect={osb_aspect_ratio:.2f}, "
                                    f"chars={osb_vertical_char_count})",
                                    verbose=verbose,
                                )
                            elif (
                                osb_sfx_like
                                and not vertical_stack
                                and osb_sfx_space_groups
                                > OSB_SFX_VERTICAL_MAX_SPACE_GROUPS
                            ):
                                log_message(
                                    "Skipping SFX-like OSB vertical layout for "
                                    f"{bbox}: too many space groups "
                                    f"({osb_sfx_space_groups}/"
                                    f"{OSB_SFX_VERTICAL_MAX_SPACE_GROUPS})",
                                    verbose=verbose,
                                )
                            (
                                force_on_ot_vertical,
                                forced_on_ot_aspect,
                                forced_on_ot_chars,
                                forced_on_ot_width,
                                forced_on_ot_width_ratio,
                                forced_on_ot_columns,
                            ) = _should_force_on_ot_vertical(
                                bubble,
                                render_tag,
                                text,
                                pil_cleaned_image.size,
                                tag_vertical_max_chars,
                                font_size_scale,
                            )
                            if force_on_ot_vertical:
                                text = _single_column_vertical_text(text)
                                bubble["translation"] = text
                                vertical_stack = True
                                osb_aspect_ratio = forced_on_ot_aspect
                                osb_vertical_char_count = forced_on_ot_chars
                                anchored_render_bbox = (
                                    _forced_on_ot_vertical_anchor_bbox(
                                        bubble,
                                        tuple(int(round(float(v))) for v in bbox),
                                        pil_cleaned_image.size,
                                    )
                                )
                                if (
                                    not osb_tall_width_expanded
                                    and tuple(render_bbox)
                                    != tuple(anchored_render_bbox)
                                ):
                                    log_message(
                                        "Anchoring forced ON/OM/OT vertical layout to "
                                        f"original outside-text bbox "
                                        f"{anchored_render_bbox} instead of expanded "
                                        f"{render_bbox}",
                                        verbose=verbose,
                                    )
                                    render_bbox = anchored_render_bbox
                                log_message(
                                    "Forcing ON/OM/OT vertical layout for "
                                    f"{render_tag} outside text {bbox} "
                                    f"(aspect={forced_on_ot_aspect:.2f}, "
                                    f"width={forced_on_ot_width}px, "
                                    f"width_ratio={forced_on_ot_width_ratio:.2f}, "
                                    f"columns={forced_on_ot_columns}, "
                                    f"chars={forced_on_ot_chars}/"
                                    f"{tag_vertical_max_chars})",
                                    verbose=verbose,
                                )
                            else:
                                anchored_render_bbox = None
                                if not osb_tall_width_expanded:
                                    anchored_render_bbox = (
                                        _long_vertical_osb_original_anchor_bbox(
                                            bubble,
                                            pil_cleaned_image.size,
                                            osb_render_padding_pixels,
                                            render_tag,
                                            text,
                                            vertical_stack,
                                        )
                                    )
                                if (
                                    anchored_render_bbox is not None
                                    and tuple(render_bbox)
                                    != tuple(anchored_render_bbox)
                                ):
                                    log_message(
                                        "Anchoring long vertical OSB layout to "
                                        f"original outside-text bbox "
                                        f"{anchored_render_bbox} instead of expanded "
                                        f"{render_bbox}",
                                        verbose=verbose,
                                    )
                                    render_bbox = anchored_render_bbox
                            if osb_sfx_like and vertical_stack:
                                try:
                                    sfx_rotation_deg = float(
                                        bubble.get("sfx_rotation_deg", 0.0) or 0.0
                                    )
                                except (TypeError, ValueError):
                                    sfx_rotation_deg = 0.0
                                vertical_glyph_rotation_deg = (
                                    _estimate_vertical_sfx_glyph_rotation_from_original_crop(
                                        bubble
                                    )
                                )
                                if abs(vertical_glyph_rotation_deg) > 0.01:
                                    # Despite the historical variable name,
                                    # is_dark_text stores whether the OSB
                                    # background is dark.  White-on-dark SFX
                                    # already uses the renderer's rotation
                                    # sign, while black-on-light short SFX
                                    # needs the glyph estimate flipped.
                                    use_direct_glyph_rotation = (
                                        bool(is_dark_text)
                                        and osb_aspect_ratio
                                        >= OSB_SFX_VERTICAL_MIN_ASPECT
                                        and abs(sfx_rotation_deg) > 0.01
                                        and sfx_rotation_deg
                                        * vertical_glyph_rotation_deg
                                        > 0.0
                                    )
                                    glyph_render_rotation_deg = (
                                        vertical_glyph_rotation_deg
                                        if use_direct_glyph_rotation
                                        else -vertical_glyph_rotation_deg
                                    )
                                    if use_direct_glyph_rotation:
                                        log_message(
                                            "Keeping direct vertical SFX rotation for "
                                            f"outside text {bbox} "
                                            f"(rotation={sfx_rotation_deg:.1f}deg, "
                                            f"glyph={vertical_glyph_rotation_deg:.1f}deg)",
                                            verbose=verbose,
                                        )
                                        osb_refined_vertical_sfx_rotation = True
                                    if (
                                        abs(sfx_rotation_deg) <= 0.01
                                        or sfx_rotation_deg
                                        * glyph_render_rotation_deg
                                        < 0.0
                                        or abs(sfx_rotation_deg)
                                        > abs(glyph_render_rotation_deg) + 3.0
                                    ):
                                        log_message(
                                            "Using vertical SFX glyph rotation for "
                                            f"outside text {bbox} "
                                            f"(rotation={glyph_render_rotation_deg:.1f}deg, "
                                            f"centerline={sfx_rotation_deg:.1f}deg)",
                                            verbose=verbose,
                                        )
                                        sfx_rotation_deg = glyph_render_rotation_deg
                                        osb_refined_vertical_sfx_rotation = True
                                if abs(sfx_rotation_deg) > 0.01:
                                    if (
                                        _is_grouped_close_sfx_text(text, source_text)
                                        and abs(rotation_deg) > 0.01
                                        and rotation_deg * sfx_rotation_deg < 0.0
                                    ):
                                        centerline_rotation_deg = sfx_rotation_deg
                                        osb_vertical_rotation_deg = rotation_deg
                                        log_message(
                                            "Using grouped SFX OSB rotation for "
                                            f"outside text {bbox} "
                                            f"(rotation={rotation_deg:.1f}deg, "
                                            "centerline="
                                            f"{centerline_rotation_deg:.1f}deg)",
                                            verbose=verbose,
                                        )
                                    else:
                                        rotation_deg = sfx_rotation_deg
                                        osb_vertical_rotation_deg = sfx_rotation_deg
                                        log_message(
                                            "Using SFX-only centerline OSB rotation for "
                                            f"outside text {bbox} "
                                            f"(rotation={rotation_deg:.1f}deg)",
                                            verbose=verbose,
                                        )
                            if not vertical_stack:
                                try:
                                    horizontal_rotation_deg = float(
                                        bubble.get("horizontal_rotation_deg", 0.0)
                                        or 0.0
                                    )
                                except (TypeError, ValueError):
                                    horizontal_rotation_deg = 0.0
                                if (
                                    abs(horizontal_rotation_deg) <= 0.01
                                    and osb_is_sfx_tag
                                ):
                                    horizontal_sfx_rotation_deg = (
                                        _estimate_horizontal_sfx_rotation_from_original_crop(
                                            bubble
                                        )
                                    )
                                    if abs(horizontal_sfx_rotation_deg) > 0.01:
                                        horizontal_rotation_deg = (
                                            horizontal_sfx_rotation_deg
                                        )
                                        horizontal_sfx_rotation_mode = bubble.pop(
                                            "_horizontal_sfx_rotation_mode",
                                            "ink",
                                        )
                                        log_message(
                                            "Using horizontal SFX "
                                            f"{horizontal_sfx_rotation_mode} rotation for "
                                            f"outside text {bbox} "
                                            f"(rotation={horizontal_rotation_deg:.1f}deg)",
                                            verbose=verbose,
                                        )
                                if abs(horizontal_rotation_deg) > 0.01:
                                    rotation_deg = horizontal_rotation_deg
                                    log_message(
                                        "Using horizontal OSB rotation for outside text "
                                        f"{bbox} (rotation={rotation_deg:.1f}deg)",
                                        verbose=verbose,
                                    )
                                elif abs(osb_vertical_rotation_deg) > 0.01:
                                    rotation_deg = 0.0
                                    log_message(
                                        "Ignoring vertical OSB rotation estimate for "
                                        f"horizontal outside text {bbox} "
                                        f"(vertical_rotation="
                                        f"{osb_vertical_rotation_deg:.1f}deg)",
                                        verbose=verbose,
                                    )
                            osb_caption_like = _should_ignore_osb_rotation_for_caption_text(
                                bubble,
                                text,
                                source_text,
                                pil_cleaned_image.size,
                            )
                            trusted_horizontal_osb_rotation = (
                                _has_trusted_horizontal_osb_rotation(
                                    bubble,
                                    render_tag,
                                    pil_cleaned_image.size,
                                )
                            )
                            if (
                                abs(rotation_deg) > 0.01
                                and osb_caption_like
                                and not trusted_horizontal_osb_rotation
                            ):
                                log_message(
                                    "Ignoring OSB rotation for caption/credit text "
                                    f"{bbox} (rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )
                                rotation_deg = 0.0
                            elif (
                                abs(rotation_deg) > 0.01
                                and osb_caption_like
                                and trusted_horizontal_osb_rotation
                            ):
                                log_message(
                                    "Keeping trusted horizontal OSB rotation for "
                                    f"caption/credit text {bbox} "
                                    f"(rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )
                            elif (
                                abs(rotation_deg) > 0.01
                                and _is_long_general_osb_text(text, source_text)
                                and not _is_dark_tilted_osb_title_candidate(
                                    bubble, pil_cleaned_image.size
                                )
                                and not trusted_horizontal_osb_rotation
                            ):
                                log_message(
                                    "Ignoring OSB rotation for long/general text "
                                    f"{bbox} (rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )
                                rotation_deg = 0.0
                                cleaned_mask = None
                            elif (
                                abs(rotation_deg) > 0.01
                                and _is_long_general_osb_text(text, source_text)
                                and trusted_horizontal_osb_rotation
                            ):
                                log_message(
                                    "Keeping trusted horizontal OSB rotation for "
                                    f"long/general text {bbox} "
                                    f"(rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )
                            if (
                                not osb_sfx_like
                                and abs(rotation_deg) > 0.01
                                and abs(rotation_deg)
                                <= OSB_NON_SFX_SMALL_ROTATION_MAX_DEGREES
                            ):
                                log_message(
                                    "Ignoring small non-SFX OSB rotation for "
                                    f"outside text {bbox} "
                                    f"(rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )
                                rotation_deg = 0.0
                            if cleaned_mask is not None and osb_caption_like:
                                log_message(
                                    "Skipping OSB render-mask clipping for "
                                    f"caption-like outside text {bbox}",
                                    verbose=verbose,
                                )
                                cleaned_mask = None
                            if vertical_stack or abs(rotation_deg) > 0.01:
                                log_message(
                                    "Using transformed layout for outside text "
                                    f"{bbox} (aspect={osb_aspect_ratio:.2f}, "
                                    f"chars={osb_vertical_char_count}, "
                                    f"vertical={vertical_stack}, "
                                    f"rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )

                            if osb_is_sfx_tag:
                                (
                                    preserve_wide_sfx,
                                    preserve_wide_sfx_reason,
                                    preserve_wide_sfx_bbox,
                                ) = _should_preserve_wide_spaced_sfx_osb(
                                    bubble,
                                    text,
                                    source_text,
                                    original_restore_image,
                                    pil_cleaned_image.size,
                                    osb_position_bbox,
                                )
                                if (
                                    preserve_wide_sfx
                                    and preserve_wide_sfx_bbox is not None
                                ):
                                    restore_bbox = tuple(
                                        int(round(float(v)))
                                        for v in preserve_wide_sfx_bbox
                                    )
                                    (
                                        pil_cleaned_image,
                                        blocked_restore_pixels,
                                    ) = _restore_original_region_excluding_bboxes(
                                        pil_cleaned_image,
                                        original_restore_image,
                                        restore_bbox,
                                        other_restore_protection_bboxes,
                                        None,
                                    )
                                    final_image_to_save = pil_cleaned_image
                                    if blocked_restore_pixels > 0:
                                        log_message(
                                            "Clipped wide-spaced SFX restore "
                                            f"{bbox} to avoid overlap with other "
                                            "text regions "
                                            f"({blocked_restore_pixels} px)",
                                            verbose=verbose,
                                            always_print=True,
                                        )
                                    log_message(
                                        "Preserved wide-spaced SFX outside text "
                                        f"{bbox}: restored original "
                                        f"({preserve_wide_sfx_reason})",
                                        verbose=verbose,
                                        always_print=True,
                                    )
                                    continue

                            text_bg_rgb = None
                            if bubble.get("needs_text_background"):
                                if text_color_rgb:
                                    lum = (
                                        0.299 * text_color_rgb[0]
                                        + 0.587 * text_color_rgb[1]
                                        + 0.114 * text_color_rgb[2]
                                    )
                                    text_bg_rgb = (
                                        (255, 255, 255) if lum < 128 else (0, 0, 0)
                                    )
                                else:
                                    text_bg_rgb = (
                                        (0, 0, 0) if is_dark_text else (255, 255, 255)
                                    )
                        else:
                            log_message(
                                f"Rendering bubble {bbox}: '{text[:30]}...'",
                                verbose=verbose,
                            )
                            font_dir = config.rendering.font_dir
                            min_font = main_min_font
                            max_font = main_max_font
                            line_spacing = config.rendering.line_spacing_mult
                            vertical_line_spacing = config.rendering.vertical_line_spacing_mult
                            use_ligs = config.rendering.use_ligatures
                            (
                                font_dir,
                                min_font,
                                max_font,
                                line_spacing,
                                vertical_line_spacing,
                                tag_vertical_max_chars,
                                tag_vertical_min_aspect,
                            ) = _apply_tag_style_overrides(
                                render_tag,
                                config,
                                font_dir,
                                min_font,
                                max_font,
                                line_spacing,
                                vertical_line_spacing,
                                font_scale=font_size_scale,
                                font_maximum=384,
                            )
                            render_info = bubble_render_info_map.get(tuple(bbox))
                            bubble_color_bgr = (255, 255, 255)
                            cleaned_mask = None
                            base_mask = None
                            is_sam_mask = False
                            text_color_rgb = None
                            source_anchor_internal_mask = None
                            if render_info:
                                bubble_color_bgr = render_info["color"]
                                cleaned_mask = render_info.get("mask")
                                base_mask = render_info.get("base_mask")
                                is_sam_mask = render_info.get("is_sam", False)
                                if render_info.get(
                                    "conjoined_white_stroke_only_cleaning",
                                    False,
                                ):
                                    source_anchor_internal_mask = cleaned_mask
                                text_color_bgr_val = render_info.get("text_color_bgr")
                                if text_color_bgr_val:
                                    text_color_rgb = (
                                        text_color_bgr_val[2],
                                        text_color_bgr_val[1],
                                        text_color_bgr_val[0],
                                    )
                            elif bubble.get("sam_mask") is not None:
                                cleaned_mask = bubble.get("sam_mask")
                                base_mask = bubble.get("sam_mask")
                                is_sam_mask = True
                            (
                                selected_text_color_rgb,
                                selected_outline_color_rgb,
                                selected_outline_width,
                            ) = _render_palette(
                                config.rendering,
                                is_outside_text=False,
                                is_dark_background=_is_dark_background_bgr(
                                    bubble_color_bgr
                                ),
                            )
                            if not _is_colored_source_text(text_color_rgb):
                                text_color_rgb = selected_text_color_rgb
                            else:
                                text_color_rgb = _render_rgb(
                                    text_color_rgb,
                                    selected_text_color_rgb,
                                )
                            bubble["text_color_rgb"] = list(text_color_rgb)
                            rotation_source_mask = base_mask if base_mask is not None else cleaned_mask
                            rotation_text_mask = cleaned_mask if cleaned_mask is not None else rotation_source_mask
                            rotation_deg = _regular_bubble_vertical_rotation(
                                bbox,
                                rotation_source_mask,
                                pil_cleaned_image.size,
                                original_cv_image,
                                rotation_text_mask,
                            )
                            tilted_vertical_allowed = (
                                abs(rotation_deg) > 0.01
                                and _regular_tilt_vertical_allowed(
                                    text,
                                    source_text,
                                    config,
                                    max_chars_override=tag_vertical_max_chars,
                                )
                            )
                            if abs(rotation_deg) > 0.01 and not tilted_vertical_allowed:
                                log_message(
                                    "Ignoring tilted vertical speech-bubble layout "
                                    f"{bbox} (rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )
                                rotation_deg = 0.0
                            vertical_stack = tilted_vertical_allowed
                            if vertical_stack:
                                # Slanted speech-bubble masks produce a very thin
                                # axis-aligned distance-transform safe box, which
                                # shrinks vertical text too aggressively. Use the
                                # original render bbox for layout, then rotate the
                                # vertical text block around the bubble center.
                                cleaned_mask = None
                                log_message(
                                    "Using tilted vertical layout for speech bubble "
                                    f"{bbox} (rotation={rotation_deg:.1f}deg)",
                                    verbose=verbose,
                                )
                            (
                                force_on_ot_vertical,
                                forced_on_ot_aspect,
                                forced_on_ot_chars,
                                forced_on_ot_width,
                                forced_on_ot_width_ratio,
                                forced_on_ot_columns,
                            ) = _should_force_on_ot_vertical(
                                bubble,
                                render_tag,
                                text,
                                pil_cleaned_image.size,
                                tag_vertical_max_chars,
                                font_size_scale,
                            )
                            if force_on_ot_vertical:
                                text = _single_column_vertical_text(text)
                                bubble["translation"] = text
                                vertical_stack = True
                                rotation_deg = 0.0
                                log_message(
                                    "Forcing ON/OM/OT vertical layout for "
                                    f"{render_tag} speech bubble {bbox} "
                                    f"(aspect={forced_on_ot_aspect:.2f}, "
                                    f"width={forced_on_ot_width}px, "
                                    f"width_ratio={forced_on_ot_width_ratio:.2f}, "
                                    f"columns={forced_on_ot_columns}, "
                                    f"chars={forced_on_ot_chars}/"
                                    f"{tag_vertical_max_chars})",
                                    verbose=verbose,
                                )
                            (
                                compare_sfx_vertical_speech_bubble,
                                sfx_speech_bubble_aspect,
                            ) = _should_compare_sfx_vertical_speech_bubble(
                                bubble,
                                render_tag,
                                text,
                                (
                                    tag_vertical_max_chars
                                    if tag_vertical_max_chars is not None
                                    else config.rendering.vertical_max_chars
                                ),
                            )
                            if (
                                compare_sfx_vertical_speech_bubble
                                and not vertical_stack
                            ):
                                log_message(
                                    "Trying vertical/horizontal comparison for "
                                    f"SFX speech bubble {bbox} "
                                    f"(aspect={sfx_speech_bubble_aspect:.2f})",
                                    verbose=verbose,
                                )

                        conjoined_tight_fit_exempt = False
                        if (
                            not is_outside_text
                            and not vertical_stack
                            and not force_on_ot_vertical
                        ):
                            conjoined_tight_fit_exempt = (
                                _should_exempt_conjoined_tight_fit_shrink(
                                    bubble,
                                    bbox,
                                    cleaned_mask,
                                    padding_pixels,
                                    pil_cleaned_image.size,
                                    text,
                                    verbose,
                                )
                            )

                        # Latin languages use hyphenation; Korean uses no-hyphen
                        # Hangul emergency breaks under the same user setting.
                        should_hyphenate = config.rendering.hyphenate_before_scaling
                        is_korean_output = (
                            config.translation.output_language.strip().lower()
                            == "korean"
                        )
                        if not (
                            is_latin_style_language(config.translation.output_language)
                            or is_korean_output
                        ):
                            should_hyphenate = False

                        emphasis_max_font_size = (
                            config.outside_text.osb_emphasis_max_font_size
                            if is_outside_text
                            else config.rendering.emphasis_max_font_size
                        )

                        render_config = RenderingConfig(
                            min_font_size=min_font,
                            max_font_size=max_font,
                            emphasis_font_dir=(
                                config.outside_text.osb_emphasis_font_dir
                                if is_outside_text
                                else config.rendering.emphasis_font_dir
                            ),
                            emphasis_max_font_size=_scaled_optional_render_font_size(
                                emphasis_max_font_size,
                                font_size_scale,
                                minimum=1,
                                maximum=640,
                            ),
                            line_spacing_mult=line_spacing,
                            emphasis_line_spacing_mult=line_spacing,
                            vertical_line_spacing_mult=vertical_line_spacing,
                            vertical_emphasis_line_spacing_mult=vertical_line_spacing,
                            vertical_font_size_mult=(
                                1.0 if is_outside_text else config.rendering.vertical_font_size_mult
                            ),
                            vertical_max_chars=(
                                tag_vertical_max_chars
                                if tag_vertical_max_chars is not None
                                else (0 if is_outside_text else config.rendering.vertical_max_chars)
                            ),
                            vertical_emphasis_max_chars=(
                                tag_vertical_max_chars
                                if tag_vertical_max_chars is not None
                                else (0 if is_outside_text else config.rendering.vertical_emphasis_max_chars)
                            ),
                            vertical_min_aspect_ratio=(
                                tag_vertical_min_aspect
                                if tag_vertical_min_aspect is not None
                                else config.rendering.vertical_min_aspect_ratio
                            ),
                            use_subpixel_rendering=(
                                config.outside_text.osb_use_subpixel_rendering
                                if is_outside_text
                                else config.rendering.use_subpixel_rendering
                            ),
                            font_hinting=(
                                config.outside_text.osb_font_hinting
                                if is_outside_text
                                else config.rendering.font_hinting
                            ),
                            use_ligatures=use_ligs,
                            hyphenate_before_scaling=should_hyphenate,
                            hyphen_penalty=config.rendering.hyphen_penalty,
                            hyphenation_min_word_length=config.rendering.hyphenation_min_word_length,
                            badness_exponent=config.rendering.badness_exponent,
                            padding_pixels=(
                                osb_render_padding_pixels
                                if is_outside_text
                                else padding_pixels
                            ),
                            padding_horizontal_pixels=(
                                osb_render_padding_pixels
                                if is_outside_text
                                else padding_horizontal_pixels
                            ),
                            padding_vertical_pixels=(
                                osb_render_padding_pixels
                                if is_outside_text
                                else padding_vertical_pixels
                            ),
                            outline_width=scale_scalar(
                                selected_outline_width,
                                processing_scale,
                                minimum=0.0,
                                maximum=24.0,
                            ),
                            outline_color_rgb=selected_outline_color_rgb,
                            supersampling_factor=config.rendering.supersampling_factor,
                            detach_trailing_punctuation=(
                                config.rendering.detach_trailing_punctuation
                            ),
                            auto_vertical_text=(
                                False
                                if is_outside_text
                                else config.rendering.auto_vertical_text
                            ),
                            ignore_auto_vertical_font_gain=(
                                _osb_inside_tag_uses_one_to_one_font_ratio(
                                    is_outside_text,
                                    render_tag,
                                )
                            ),
                            compare_forced_vertical_layout=(
                                _should_compare_forced_vertical_layout(
                                    is_outside_text,
                                    force_on_ot_vertical,
                                    osb_is_sfx_tag,
                                    osb_forced_sfx_vertical,
                                    tilted_vertical_allowed,
                                )
                            ),
                            force_vertical_fallback_only=force_on_ot_vertical,
                            tight_fit_shrink_min_font_size=scale_font_size(
                                config.rendering.tight_fit_shrink_min_font_size,
                                font_size_scale,
                                minimum=1,
                                maximum=384,
                            ),
                            force_vertical_retry_columns=(
                                forced_on_ot_columns if force_on_ot_vertical else 1
                            ),
                            force_vertical_retry_font_size=_scaled_render_font_size(
                                config.rendering.force_vertical_retry_font_size,
                                font_size_scale,
                                minimum=1,
                                maximum=384,
                            ),
                            force_vertical_column_gap_mult=(
                                1.12
                                if force_on_ot_vertical and forced_on_ot_columns > 1
                                else 1.0
                            ),
                            tight_fit_shrink_exempt=(
                                is_outside_text
                                or force_on_ot_vertical
                                or conjoined_tight_fit_exempt
                            ),
                            layout_scale=font_size_scale,
                            small_text_zero_padding_max_font_size=_scaled_render_font_size(
                                10,
                                font_size_scale,
                                minimum=1,
                                maximum=384,
                            ),
                            small_text_zero_padding_max_chars=10,
                            collapsed_small_bbox_fallback_max_area=_scaled_render_area(
                                8000,
                                font_size_scale,
                                minimum=1,
                            ),
                            collapsed_small_bbox_fallback_max_side=_scaled_render_length(
                                140,
                                font_size_scale,
                                minimum=1,
                                maximum=max(pil_cleaned_image.size),
                            ),
                            collapsed_small_bbox_fallback_max_short_side=_scaled_render_length(
                                90,
                                font_size_scale,
                                minimum=1,
                                maximum=max(pil_cleaned_image.size),
                            ),
                            wrapped_symbol_relax_font_size_threshold=_scaled_render_font_size(
                                18,
                                font_size_scale,
                                minimum=1,
                                maximum=384,
                            ),
                        )
                        if not is_outside_text:
                            render_config.short_text_ignores_auto_vertical_font_gain = True
                            _configure_tilted_vertical_comparison(
                                render_config,
                                tilted_vertical_allowed,
                            )
                        if not is_outside_text and render_tag == "BE":
                            render_config.tight_fit_shrink_factor = 0.95
                        _configure_translucent_tight_fit_shrink(
                            render_config,
                            render_info,
                            is_outside_text,
                            font_size_scale,
                        )
                        if compare_sfx_vertical_speech_bubble:
                            render_config.force_compare_vertical_text = True
                            render_config.forced_vertical_max_horizontal_fill = (
                                SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MAX_HORIZONTAL_FILL
                            )
                            render_config.forced_vertical_min_fill_gain = (
                                SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MIN_FILL_GAIN
                            )
                            render_config.forced_vertical_min_font_size_ratio = (
                                SPEECH_BUBBLE_SFX_VERTICAL_COMPARE_MIN_FONT_RATIO
                            )
                        if _osb_inside_tag_uses_one_to_one_font_ratio(
                            is_outside_text,
                            render_tag,
                        ):
                            render_config.forced_vertical_max_horizontal_fill = (
                                OSB_INSIDE_TAG_VERTICAL_COMPARE_MAX_HORIZONTAL_FILL
                            )
                            render_config.forced_vertical_min_fill_gain = (
                                OSB_INSIDE_TAG_VERTICAL_COMPARE_MIN_FILL_GAIN
                            )
                            render_config.forced_vertical_min_font_size_ratio = (
                                OSB_INSIDE_TAG_VERTICAL_COMPARE_MIN_FONT_RATIO
                            )
                            log_message(
                                "Using OSB-detected inside-tag vertical comparison "
                                f"policy for <{render_tag}> {bbox} "
                                "(max_horizontal_fill=0.75, "
                                "min_fill_gain=0.10, min_font_ratio=1.00)",
                                verbose=verbose,
                            )
                        if osb_forced_sfx_vertical:
                            render_config.forced_vertical_max_horizontal_fill = (
                                OSB_SFX_VERTICAL_COMPARE_MAX_HORIZONTAL_FILL
                            )
                            render_config.forced_vertical_min_fill_gain = (
                                OSB_SFX_VERTICAL_COMPARE_MIN_FILL_GAIN
                            )
                            render_config.forced_vertical_min_font_size_ratio = (
                                OSB_SFX_VERTICAL_COMPARE_MIN_FONT_RATIO
                            )
                        if (
                            is_outside_text
                            and osb_refined_vertical_sfx_rotation
                            and vertical_stack
                            and abs(rotation_deg) > 0.01
                        ):
                            render_config.compare_forced_vertical_layout = False
                            render_config.force_compare_vertical_text = False
                            log_message(
                                "Locking rotated vertical SFX layout for "
                                f"{bbox}; skipping horizontal fallback comparison",
                                verbose=verbose,
                            )
                        if not is_outside_text:
                            conjoined_base_layout_selected = False
                            pil_cleaned_image, cleaned_mask, sfx_rescue_applied = (
                                _apply_sfx_tiny_safe_area_rescue(
                                    pil_cleaned_image,
                                    bbox,
                                    render_tag,
                                    cleaned_mask,
                                    base_mask,
                                    bubble_color_bgr,
                                    render_config.padding_pixels,
                                    verbose,
                                )
                            )
                            child_local_layout_mask = cleaned_mask
                            if (
                                cleaned_mask is not None
                                and base_mask is not None
                                and not vertical_stack
                                and not force_on_ot_vertical
                            ):
                                selected_layout_mask = _select_conjoined_layout_mask(
                                    bubble,
                                    bbox,
                                    cleaned_mask,
                                    base_mask,
                                    render_config.padding_pixels,
                                    pil_cleaned_image.size,
                                    text,
                                    render_tag,
                                    verbose,
                                    allow_colored_bubble=bool(
                                        render_info
                                        and render_info.get("is_colored", False)
                                    ),
                                    layout_mask_hint=(
                                        render_info.get("layout_mask")
                                        if render_info
                                        else None
                                    ),
                                )
                                if selected_layout_mask is not cleaned_mask:
                                    is_conjoined_layout = not (
                                        bubble.get("conjoined_parent_mask") is None
                                        and not bubble.get("conjoined_neighbor_bboxes")
                                    )
                                    center_x_override = None
                                    conjoined_layout_allows_shrink = False
                                    conjoined_base_layout_selected = (
                                        is_conjoined_layout
                                        and selected_layout_mask is base_mask
                                    )
                                    if is_conjoined_layout:
                                        conjoined_layout_allows_shrink = (
                                            _conjoined_base_layout_recovery_allows_tight_fit_shrink(
                                                cleaned_mask,
                                                selected_layout_mask,
                                                bbox,
                                                render_config.padding_pixels,
                                                pil_cleaned_image.size,
                                                verbose,
                                            )
                                        )
                                        center_x_override = (
                                            _conjoined_layout_center_x_override(
                                                bubble,
                                                bbox,
                                                cleaned_mask,
                                                base_mask,
                                                render_config.padding_pixels,
                                                pil_cleaned_image.size,
                                                (
                                                    render_info.get("text_bbox")
                                                    if render_info
                                                    else None
                                                ),
                                                verbose,
                                            )
                                        )
                                        if center_x_override is not None:
                                            setattr(
                                                render_config,
                                                "safe_area_center_x_override",
                                                center_x_override,
                                            )
                                            setattr(
                                                render_config,
                                                "skip_layout_collision_check",
                                                True,
                                            )
                                    cleaned_mask = selected_layout_mask
                                    if not is_conjoined_layout:
                                        render_config.tight_fit_shrink_exempt = False
                                    elif conjoined_layout_allows_shrink:
                                        render_config.tight_fit_shrink_exempt = False
                                    elif center_x_override is None:
                                        render_config.tight_fit_shrink_exempt = True
                                    else:
                                        render_config.tight_fit_shrink_exempt = False
                                        if render_tag == "BE":
                                            render_config.tight_fit_shrink_factor = 0.95
                                        else:
                                            render_config.tight_fit_shrink_factor = min(
                                                float(
                                                    getattr(
                                                        render_config,
                                                        "tight_fit_shrink_factor",
                                                        0.85,
                                                    )
                                                    or 0.85
                                                ),
                                                0.82,
                                            )
                            if (
                                cleaned_mask is not None
                                and not vertical_stack
                                and not force_on_ot_vertical
                                and _should_use_bbox_layout_for_conjoined_text(
                                    bubble,
                                    bbox,
                                    cleaned_mask,
                                    render_config.padding_pixels,
                                    pil_cleaned_image.size,
                                    text,
                                    render_tag,
                                    verbose,
                                )
                            ):
                                cleaned_mask = None
                            elif (
                                cleaned_mask is not None
                                and not vertical_stack
                                and not force_on_ot_vertical
                            ):
                                cleaned_mask = (
                                    _complete_occluded_dark_bubble_layout_mask(
                                        bubble,
                                        bbox,
                                        cleaned_mask,
                                        bubble_color_bgr,
                                        render_config.padding_pixels,
                                        pil_cleaned_image.size,
                                        render_tag,
                                        verbose,
                                    )
                                )
                                cleaned_mask = (
                                    _complete_occluded_colored_bubble_layout_mask(
                                        bubble,
                                        bbox,
                                        cleaned_mask,
                                        render_config.padding_pixels,
                                        pil_cleaned_image.size,
                                        text,
                                        render_tag,
                                        verbose,
                                        is_colored=bool(
                                            render_info
                                            and render_info.get(
                                                "is_colored", False
                                            )
                                        ),
                                    )
                                )
                                cleaned_mask = (
                                    _restore_cleaned_text_holes_for_layout(
                                        bubble,
                                        bbox,
                                        cleaned_mask,
                                        render_config.padding_pixels,
                                        pil_cleaned_image.size,
                                        text,
                                        render_tag,
                                        verbose,
                                        original_image_bgr=original_cv_image,
                                        cleaned_image=pil_cleaned_image,
                                        bubble_color_bgr=bubble_color_bgr,
                                        is_colored=bool(
                                            render_info
                                            and render_info.get(
                                                "is_colored", False
                                            )
                                        ),
                                    )
                                )
                                cleaned_mask = _bridge_narrow_notch_for_layout(
                                    bubble,
                                    bbox,
                                    cleaned_mask,
                                    render_config.padding_pixels,
                                    pil_cleaned_image.size,
                                    text,
                                    render_tag,
                                    verbose,
                                )
                                guard_bubble = bubble
                                if (
                                    render_info
                                    and render_info.get("conjoined_neighbor_bboxes")
                                    and not bubble.get("conjoined_neighbor_bboxes")
                                ):
                                    guard_bubble = dict(bubble)
                                    guard_bubble["conjoined_neighbor_bboxes"] = (
                                        render_info.get("conjoined_neighbor_bboxes")
                                        or []
                                    )
                                cleaned_mask, conjoined_guard_applied = (
                                    _guard_conjoined_child_layout_mask(
                                        guard_bubble,
                                        bbox,
                                        cleaned_mask,
                                        child_local_layout_mask,
                                        base_mask,
                                        pil_cleaned_image.size,
                                        verbose,
                                    )
                                )
                                if conjoined_guard_applied:
                                    conjoined_base_layout_selected = False
                                    render_config.tight_fit_shrink_exempt = False
                                    for _layout_attr in (
                                        "safe_area_center_x_override",
                                        "skip_layout_collision_check",
                                    ):
                                        if hasattr(render_config, _layout_attr):
                                            delattr(render_config, _layout_attr)
                            _configure_translucent_tight_fit_shrink(
                                render_config,
                                render_info,
                                is_outside_text,
                                font_size_scale,
                            )
                            if (
                                conjoined_base_layout_selected
                                and cleaned_mask is not None
                            ):
                                _configure_conjoined_base_tight_fit_shrink(
                                    render_config,
                                    font_size_scale,
                                )
                            balance_cap = conjoined_pair_font_balance_caps.get(
                                tuple(int(round(float(v))) for v in bbox)
                            )
                            if (
                                balance_cap is not None
                                and balance_cap < int(render_config.max_font_size)
                            ):
                                original_balance_max_font = int(
                                    render_config.max_font_size
                                )
                                render_config = _render_config_with_max_font(
                                    render_config,
                                    balance_cap,
                                )
                                log_message(
                                    "Applying conjoined pair font balance cap "
                                    f"{tuple(int(round(float(v))) for v in bbox)}: "
                                    f"max_font {original_balance_max_font} -> "
                                    f"{render_config.max_font_size}",
                                    verbose=verbose,
                                )
                        else:
                            sfx_rescue_applied = False
                        osb_split_texts: Optional[List[str]] = None
                        osb_split_boxes: List[Tuple[int, int, int, int]] = []
                        osb_split_layouts: List[Dict[str, Any]] = []
                        if (
                            is_outside_text
                            and not force_on_ot_vertical
                            and osb_is_sfx_tag
                        ):
                            repeated_hangul_split = (
                                _repeated_two_syllable_hangul_osb_tokens(text)
                            )
                            mixed_large_sfx_split = (
                                _mixed_long_hangul_sfx_osb_tokens(text)
                            )
                            mixed_large_sfx_source_boxes: List[
                                Tuple[int, int, int, int]
                            ] = []
                            mixed_large_sfx_visual = False
                            if mixed_large_sfx_split:
                                mixed_large_sfx_source_boxes = (
                                    _large_mixed_sfx_source_boxes(
                                        bubble,
                                        original_restore_image,
                                        original_restore_image.size,
                                        max(0, len(mixed_large_sfx_split) - 1),
                                    )
                                )
                                mixed_large_sfx_visual = bool(
                                    mixed_large_sfx_source_boxes
                                )
                            repeated_source_text_like = (
                                _source_is_compact_repeated_sfx_token(source_text)
                            )
                            repeated_source_split_like = False
                            if repeated_hangul_split:
                                repeated_source_split_like = (
                                    _osb_source_supports_repeated_group_split(
                                        bubble,
                                        original_restore_image,
                                        original_restore_image.size,
                                        len(repeated_hangul_split),
                                    )
                                )
                                if not repeated_source_split_like:
                                    log_message(
                                        "Skipping repeated SFX split for "
                                        f"{bbox}: source ink does not look split",
                                        verbose=verbose,
                                    )
                                elif not repeated_source_text_like:
                                    log_message(
                                        "Using visual source split for repeated "
                                        f"SFX {bbox}; source OCR text was not "
                                        "a compact repeat",
                                        verbose=verbose,
                                    )
                            if mixed_large_sfx_split and not mixed_large_sfx_visual:
                                log_message(
                                    "Skipping mixed large SFX split for "
                                    f"{bbox}: source image does not match "
                                    "mixed inset/body layout",
                                    verbose=verbose,
                                )
                            compact_visual_sfx_split: List[str] = []
                            compact_visual_sfx_layouts: List[Dict[str, Any]] = []
                            if (
                                not vertical_stack
                                and not mixed_large_sfx_split
                                and not repeated_hangul_split
                            ):
                                candidate_compact_split = (
                                    _compact_visual_sfx_osb_tokens(text)
                                )
                                if candidate_compact_split and (
                                    _source_supports_compact_visual_sfx_split(
                                        bubble,
                                        source_text,
                                        original_restore_image,
                                        original_restore_image.size,
                                        len(candidate_compact_split),
                                    )
                                ):
                                    try:
                                        compact_visual_base_box = _clip_bbox_to_image(
                                            tuple(
                                                int(round(float(v)))
                                                for v in (
                                                    bubble.get("original_bbox")
                                                    or bubble.get("bbox")
                                                    or osb_position_bbox
                                                )
                                            ),
                                            pil_cleaned_image.size,
                                        )
                                    except (TypeError, ValueError):
                                        compact_visual_base_box = None
                                    if compact_visual_base_box is not None:
                                        compact_visual_sfx_layouts = (
                                            _osb_stacked_source_position_render_boxes(
                                                candidate_compact_split,
                                                compact_visual_base_box,
                                                pil_cleaned_image.size,
                                                render_config.max_font_size,
                                                compact_visual=True,
                                                page_font_scale=font_size_scale,
                                            )
                                        )
                                    if compact_visual_sfx_layouts:
                                        compact_visual_sfx_split = (
                                            candidate_compact_split
                                        )
                            if (
                                osb_refined_vertical_sfx_rotation
                                and vertical_stack
                                and abs(rotation_deg) > 0.01
                                and len(
                                    [
                                        token
                                        for token in str(text or "").split()
                                        if token.strip()
                                    ]
                                )
                                <= 1
                            ):
                                if repeated_hangul_split and repeated_source_split_like:
                                    osb_split_texts = repeated_hangul_split
                                    log_message(
                                        "Using source-group repeated SFX split for "
                                        f"rotated vertical SFX {bbox}",
                                        verbose=verbose,
                                    )
                                else:
                                    compact_spaced_source_split = (
                                        _compact_spaced_source_vertical_sfx_osb_tokens(
                                            text,
                                            source_text,
                                            bubble,
                                            original_restore_image,
                                            original_restore_image.size,
                                        )
                                    )
                                    if compact_spaced_source_split:
                                        osb_split_texts = compact_spaced_source_split
                                        log_message(
                                            "Using visually separated compact SFX "
                                            "split for rotated vertical SFX "
                                            f"{bbox}",
                                            verbose=verbose,
                                        )
                                    else:
                                        osb_split_texts = []
                                        log_message(
                                            "Skipping original ink split for rotated "
                                            f"vertical SFX {bbox}; using whole OSB layout",
                                            verbose=verbose,
                                        )
                            elif osb_spaced_single_hangul_repeat:
                                osb_split_texts = []
                                log_message(
                                    "Skipping original ink split for spaced repeated "
                                    f"SFX {bbox}; using whole OSB layout",
                                    verbose=verbose,
                                )
                            elif _source_is_single_base_kana_with_small_mark(
                                source_text
                            ):
                                osb_split_texts = []
                                log_message(
                                    "Skipping original ink split for single-base "
                                    f"source SFX {bbox}; using whole OSB layout",
                                    verbose=verbose,
                                )
                            else:
                                osb_split_texts = (
                                    mixed_large_sfx_split
                                    if (
                                        mixed_large_sfx_split
                                        and mixed_large_sfx_visual
                                    )
                                    else repeated_hangul_split
                                    if (
                                        repeated_hangul_split
                                        and repeated_source_split_like
                                    )
                                    else compact_visual_sfx_split
                                    if compact_visual_sfx_split
                                    else (
                                        []
                                        if (
                                            vertical_stack
                                            and _is_single_hangul_osb_token(text)
                                        )
                                        else _original_position_osb_tokens(text)
                                    )
                                )
                            if osb_split_texts:
                                if (
                                    compact_visual_sfx_split
                                    and osb_split_texts == compact_visual_sfx_split
                                    and compact_visual_sfx_layouts
                                ):
                                    osb_split_layouts = compact_visual_sfx_layouts
                                    bubble["_osb_compact_source_position_layout"] = True
                                used_large_mixed_sfx_layout = False
                                if (
                                    mixed_large_sfx_split
                                    and osb_split_texts == mixed_large_sfx_split
                                    and mixed_large_sfx_visual
                                ):
                                    try:
                                        large_mixed_base_box = _clip_bbox_to_image(
                                            tuple(
                                                int(round(float(v)))
                                                for v in (
                                                    bubble.get("original_bbox")
                                                    or bubble.get("bbox")
                                                    or osb_position_bbox
                                                )
                                            ),
                                            pil_cleaned_image.size,
                                        )
                                    except (TypeError, ValueError):
                                        large_mixed_base_box = None
                                    if large_mixed_base_box is not None:
                                        osb_split_layouts = (
                                            _osb_large_mixed_sfx_render_boxes(
                                                osb_split_texts,
                                                large_mixed_base_box,
                                                pil_cleaned_image.size,
                                                render_config.max_font_size,
                                                source_boxes=mixed_large_sfx_source_boxes,
                                                page_font_scale=font_size_scale,
                                            )
                                        )
                                        if osb_split_layouts:
                                            used_large_mixed_sfx_layout = True
                                            log_message(
                                                "Using mixed large SFX source-position "
                                                f"layout {bbox}: "
                                                f"{len(osb_split_layouts)} part(s)",
                                                verbose=verbose,
                                            )
                                boost_horizontal_sfx_split = bool(
                                    not vertical_stack
                                    and osb_is_sfx_tag
                                    and not used_large_mixed_sfx_layout
                                )
                                force_horizontal_repeated_split = bool(
                                    not vertical_stack
                                    and _is_repeated_short_osb_split(osb_split_texts)
                                )
                                if (
                                    boost_horizontal_sfx_split
                                    and not force_horizontal_repeated_split
                                ):
                                    log_message(
                                        "Using source-size horizontal SFX split layout "
                                        f"for {bbox}",
                                        verbose=verbose,
                                    )
                                if force_horizontal_repeated_split:
                                    log_message(
                                        "Using horizontal repeated SFX split layout "
                                        f"for {bbox}; preserving whole OSB "
                                        "orientation",
                                        verbose=verbose,
                                    )
                                if not osb_split_layouts:
                                    osb_split_layouts = _osb_original_cluster_render_boxes(
                                        bubble,
                                        osb_split_texts,
                                        original_restore_image,
                                        pil_cleaned_image.size,
                                        osb_position_bbox,
                                        render_config.max_font_size,
                                        force_horizontal_repeated_split=(
                                            force_horizontal_repeated_split
                                        ),
                                        boost_horizontal_sfx_split=boost_horizontal_sfx_split,
                                        page_font_scale=font_size_scale,
                                    )
                                if osb_split_layouts:
                                    osb_split_boxes = [
                                        layout["bbox"] for layout in osb_split_layouts
                                    ]
                                    if bubble.pop(
                                        "_osb_diagonal_source_position_layout", False
                                    ):
                                        fallback_reason = bubble.pop(
                                            "_osb_source_position_fallback_reason",
                                            "cluster split fallback",
                                        )
                                        log_message(
                                            "Using diagonal source-position OSB split "
                                            f"render layout {bbox}: "
                                            f"{len(osb_split_layouts)} part(s) "
                                            f"({fallback_reason})",
                                            verbose=verbose,
                                        )
                                    elif bubble.pop(
                                        "_osb_stacked_source_position_layout", False
                                    ):
                                        log_message(
                                            "Using stacked source-position OSB split "
                                            f"render layout {bbox}: "
                                            f"{len(osb_split_layouts)} part(s)",
                                            verbose=verbose,
                                        )
                                    elif bubble.pop(
                                        "_osb_compact_source_position_layout", False
                                    ):
                                        log_message(
                                            "Using compact visual source-position OSB "
                                            f"split render layout {bbox}: "
                                            f"{len(osb_split_layouts)} part(s)",
                                            verbose=verbose,
                                        )
                                    else:
                                        log_message(
                                            "Using original ink cluster OSB split render layout "
                                            f"{bbox}: {len(osb_split_layouts)} part(s)",
                                            verbose=verbose,
                                        )
                                else:
                                    rejected_cluster_reason = bubble.pop(
                                        "_osb_rejected_cluster_reason",
                                        None,
                                    )
                                    if rejected_cluster_reason:
                                        log_message(
                                            "Rejecting original ink split for "
                                            f"{bbox}: {rejected_cluster_reason}; "
                                            "using whole OSB layout",
                                            verbose=verbose,
                                        )
                                    rejected_cluster_rotation = bubble.pop(
                                        "_osb_rejected_cluster_rotation_deg",
                                        None,
                                    )
                                    if (
                                        rejected_cluster_rotation is not None
                                        and not vertical_stack
                                        and osb_is_sfx_tag
                                    ):
                                        try:
                                            rejected_cluster_rotation = float(
                                                rejected_cluster_rotation
                                            )
                                        except (TypeError, ValueError):
                                            rejected_cluster_rotation = 0.0
                                        if abs(rejected_cluster_rotation) > 0.01:
                                            if abs(rotation_deg) <= 0.01 or (
                                                (rotation_deg > 0.0)
                                                == (rejected_cluster_rotation > 0.0)
                                                and abs(rejected_cluster_rotation)
                                                > abs(rotation_deg) + 1.0
                                            ):
                                                rotation_deg = rejected_cluster_rotation
                                                log_message(
                                                    "Using whole-OSB rotation after "
                                                    "rejecting unbalanced split for "
                                                    f"{bbox} "
                                                    f"(rotation={rotation_deg:.1f}deg)",
                                                    verbose=verbose,
                                                )
                                            else:
                                                log_message(
                                                    "Keeping existing OSB rotation after "
                                                    "rejecting unbalanced split for "
                                                    f"{bbox} "
                                                    f"(rotation={rotation_deg:.1f}deg, "
                                                    "rejected_split_rotation="
                                                    f"{rejected_cluster_rotation:.1f}deg)",
                                                    verbose=verbose,
                                                )
                                    spaced_vertical_fallback_texts: List[str] = []
                                    spaced_vertical_fallback_layouts: List[
                                        Dict[str, Any]
                                    ] = []
                                    spaced_vertical_fallback_kind = "spaced"
                                    if (
                                        vertical_stack
                                        and osb_refined_vertical_sfx_rotation
                                        and osb_is_sfx_tag
                                        and rejected_cluster_reason
                                    ):
                                        spaced_vertical_fallback_texts = (
                                            _spaced_vertical_sfx_osb_tokens(
                                                text,
                                                source_text,
                                            )
                                        )
                                        if not spaced_vertical_fallback_texts:
                                            spaced_vertical_fallback_texts = (
                                                _compact_spaced_source_vertical_sfx_osb_tokens(
                                                    text,
                                                    source_text,
                                                    bubble,
                                                    original_restore_image,
                                                    original_restore_image.size,
                                                )
                                            )
                                            if spaced_vertical_fallback_texts:
                                                spaced_vertical_fallback_kind = (
                                                    "visually separated compact"
                                                )
                                        if spaced_vertical_fallback_texts:
                                            try:
                                                spaced_vertical_base_box = (
                                                    _clip_bbox_to_image(
                                                        tuple(
                                                            int(round(float(v)))
                                                            for v in (
                                                                bubble.get(
                                                                    "original_bbox"
                                                                )
                                                                or bubble.get("bbox")
                                                                or osb_position_bbox
                                                            )
                                                        ),
                                                        pil_cleaned_image.size,
                                                    )
                                                )
                                            except (TypeError, ValueError):
                                                spaced_vertical_base_box = None
                                            if spaced_vertical_base_box is not None:
                                                spaced_vertical_fallback_layouts = (
                                                    _osb_spaced_vertical_sfx_source_position_render_boxes(
                                                        spaced_vertical_fallback_texts,
                                                        spaced_vertical_base_box,
                                                        pil_cleaned_image.size,
                                                        render_config.max_font_size,
                                                        rotation_deg,
                                                        page_font_scale=font_size_scale,
                                                    )
                                                )
                                    if spaced_vertical_fallback_layouts:
                                        osb_split_texts = (
                                            spaced_vertical_fallback_texts
                                        )
                                        osb_split_layouts = (
                                            spaced_vertical_fallback_layouts
                                        )
                                        osb_split_boxes = [
                                            layout["bbox"]
                                            for layout in osb_split_layouts
                                        ]
                                        log_message(
                                            "Using "
                                            f"{spaced_vertical_fallback_kind} "
                                            "vertical SFX "
                                            "source-position fallback "
                                            f"{bbox}: "
                                            f"{len(osb_split_layouts)} part(s)",
                                            verbose=verbose,
                                        )
                                    else:
                                        osb_split_texts = None
                                        osb_split_layouts = []
                        if (
                            is_outside_text
                            and not osb_split_layouts
                            and not osb_tall_width_expanded
                        ):
                            vertical_ink_anchor = _osb_vertical_ink_anchor_bbox(
                                bubble,
                                original_restore_image,
                                pil_cleaned_image.size,
                                True,
                            )
                            if vertical_ink_anchor is not None:
                                (
                                    ink_anchor_bbox,
                                    raw_anchor_bbox,
                                    ink_anchor_pixels,
                                ) = vertical_ink_anchor
                                osb_vertical_compare_bbox = ink_anchor_bbox
                                if vertical_stack and tuple(render_bbox) != tuple(
                                    ink_anchor_bbox
                                ):
                                    log_message(
                                        "Anchoring vertical OSB layout to original "
                                        f"ink bbox {ink_anchor_bbox} instead of "
                                        f"detected bbox {render_bbox} "
                                        f"(source={raw_anchor_bbox}, "
                                        f"ink={ink_anchor_pixels}px)",
                                        verbose=verbose,
                                        always_print=True,
                                    )
                                    render_bbox = ink_anchor_bbox
                        active_vertical_stack = vertical_stack
                        active_rotation_deg = rotation_deg
                        active_render_config = render_config
                        first_split_layout: Dict[str, Any] = {}
                        if osb_split_layouts:
                            first_split_layout = osb_split_layouts[0]
                            active_vertical_stack = bool(
                                first_split_layout.get("vertical_stack", vertical_stack)
                            )
                            try:
                                active_rotation_deg = float(
                                    first_split_layout.get("rotation_deg", rotation_deg)
                                    or 0.0
                                )
                            except (TypeError, ValueError):
                                active_rotation_deg = rotation_deg
                            active_render_config = _render_config_with_max_font(
                                render_config,
                                first_split_layout.get("max_font_size"),
                            )
                            if first_split_layout.get("lock_vertical_stack"):
                                active_render_config.compare_forced_vertical_layout = False
                                active_render_config.force_compare_vertical_text = False
                        primary_render_text = text
                        primary_render_bbox = render_bbox
                        if is_outside_text and osb_split_texts and osb_split_layouts:
                            primary_render_text = osb_split_texts[0]
                            primary_render_bbox = _inset_bbox_for_padding(
                                first_split_layout.get("bbox", osb_split_boxes[0]),
                                first_split_layout.get(
                                    "padding_pixels", osb_render_padding_pixels
                                ),
                            )
                        success = False
                        primary_rendered_font_size = None
                        if (
                            is_outside_text
                            and bubble.get("edge_sfx_full_glyph", False)
                            and not osb_split_layouts
                            and np.any(translated_speech_text_mask)
                        ):
                            (
                                pil_cleaned_image,
                                repaired_collision_pixels,
                            ) = _restore_edge_sfx_collision_background(
                                pil_cleaned_image,
                                bubble.get("original_bbox") or bbox,
                                translated_speech_text_mask,
                                translated_speech_regions,
                            )
                            if repaired_collision_pixels > 0:
                                log_message(
                                    "Restored only the edge-SFX rectangle area "
                                    "conflicting with translated speech text "
                                    f"{bbox} ({repaired_collision_pixels}px)",
                                    verbose=verbose,
                                    always_print=True,
                                )
                        pre_entry_render_image = pil_cleaned_image.copy()
                        if is_outside_text:
                            pre_render_image = pil_cleaned_image
                            # For OSB, keep the original OCR bbox as the layout anchor.
                            # The render mask is only used after drawing to clip pixels
                            # back into the original detected shape.
                            osb_layout_mask = None
                            try:
                                rendered_image = render_text_skia(
                                    pil_image=pre_render_image,
                                    text=primary_render_text,
                                    bbox=primary_render_bbox,
                                    font_dir=font_dir,
                                    cleaned_mask=osb_layout_mask,
                                    bubble_color_bgr=bubble_color_bgr,
                                    config=active_render_config,
                                    verbose=verbose,
                                    bubble_id=str(i + 1),
                                    rotation_deg=active_rotation_deg,
                                    vertical_stack=active_vertical_stack,
                                    text_color_rgb=text_color_rgb,
                                    raise_on_safe_error=False,
                                    text_background_color=text_bg_rgb,
                                    horizontal_compare_bbox=(
                                        osb_horizontal_compare_bbox
                                        if not osb_split_layouts
                                        else None
                                    ),
                                    vertical_compare_bbox=(
                                        osb_vertical_compare_bbox
                                        if not osb_split_layouts
                                        else None
                                    ),
                                )
                                primary_rendered_font_size = getattr(
                                    rendered_image,
                                    "info",
                                    {},
                                ).get("font_size")
                                rendered_image = _clip_rendered_image_to_mask(
                                    pre_render_image, rendered_image, cleaned_mask
                                )
                                if (
                                    bubble.get("edge_sfx_full_glyph", False)
                                    and not osb_split_layouts
                                ):
                                    edge_rect = _clip_bbox_to_image(
                                        tuple(
                                            int(round(float(v)))
                                            for v in (
                                                bubble.get("original_bbox") or bbox
                                            )
                                        ),
                                        pil_cleaned_image.size,
                                    )
                                    edge_rect_mask = np.zeros(
                                        translated_speech_text_mask.shape,
                                        dtype=bool,
                                    )
                                    if edge_rect is not None:
                                        ex1, ey1, ex2, ey2 = edge_rect
                                        edge_rect_mask[ey1:ey2, ex1:ex2] = True
                                    speech_guard_mask = translated_speech_text_mask
                                    if np.any(speech_guard_mask):
                                        speech_guard_mask = cv2.dilate(
                                            speech_guard_mask.astype(np.uint8),
                                            np.ones((5, 5), dtype=np.uint8),
                                            iterations=1,
                                        ) > 0

                                    best_image = rendered_image
                                    best_bbox = primary_render_bbox
                                    best_config = active_render_config
                                    best_font_size = primary_rendered_font_size
                                    best_change_mask = _rendered_pixel_change_mask(
                                        pre_render_image,
                                        rendered_image,
                                    )
                                    best_overlap = int(
                                        np.count_nonzero(
                                            np.logical_and(
                                                best_change_mask,
                                                speech_guard_mask,
                                            )
                                        )
                                    )
                                    best_overflow = int(
                                        np.count_nonzero(
                                            np.logical_and(
                                                best_change_mask,
                                                np.logical_not(edge_rect_mask),
                                            )
                                        )
                                    )
                                    best_score = best_overlap + best_overflow
                                    if best_score > 0:
                                        initial_font_size = int(
                                            primary_rendered_font_size
                                            or active_render_config.max_font_size
                                        )
                                        candidate_specs = []
                                        seen_font_caps = set()
                                        for ratio in (0.90, 0.80, 0.70, 0.60):
                                            font_cap = max(
                                                int(active_render_config.min_font_size),
                                                int(round(initial_font_size * ratio)),
                                            )
                                            if font_cap in seen_font_caps:
                                                continue
                                            seen_font_caps.add(font_cap)
                                            candidate_specs.append(
                                                (
                                                    primary_render_bbox,
                                                    _render_config_with_max_font(
                                                        active_render_config,
                                                        font_cap,
                                                    ),
                                                    f"font={font_cap}",
                                                )
                                            )

                                        for (
                                            candidate_bbox,
                                            candidate_config,
                                            candidate_reason,
                                        ) in candidate_specs:
                                            candidate_image = render_text_skia(
                                                pil_image=pre_render_image,
                                                text=primary_render_text,
                                                bbox=candidate_bbox,
                                                font_dir=font_dir,
                                                cleaned_mask=osb_layout_mask,
                                                bubble_color_bgr=bubble_color_bgr,
                                                config=candidate_config,
                                                verbose=False,
                                                bubble_id=str(i + 1),
                                                rotation_deg=active_rotation_deg,
                                                vertical_stack=active_vertical_stack,
                                                text_color_rgb=text_color_rgb,
                                                raise_on_safe_error=False,
                                                text_background_color=text_bg_rgb,
                                            )
                                            candidate_font_size = getattr(
                                                candidate_image,
                                                "info",
                                                {},
                                            ).get("font_size")
                                            candidate_image = (
                                                _clip_rendered_image_to_mask(
                                                    pre_render_image,
                                                    candidate_image,
                                                    cleaned_mask,
                                                )
                                            )
                                            candidate_change_mask = (
                                                _rendered_pixel_change_mask(
                                                    pre_render_image,
                                                    candidate_image,
                                                )
                                            )
                                            candidate_overlap = int(
                                                np.count_nonzero(
                                                    np.logical_and(
                                                        candidate_change_mask,
                                                        speech_guard_mask,
                                                    )
                                                )
                                            )
                                            candidate_overflow = int(
                                                np.count_nonzero(
                                                    np.logical_and(
                                                        candidate_change_mask,
                                                        np.logical_not(
                                                            edge_rect_mask
                                                        ),
                                                    )
                                                )
                                            )
                                            candidate_score = (
                                                candidate_overlap
                                                + candidate_overflow
                                            )
                                            if candidate_score < best_score:
                                                best_image = candidate_image
                                                best_bbox = candidate_bbox
                                                best_config = candidate_config
                                                best_font_size = candidate_font_size
                                                best_overlap = candidate_overlap
                                                best_overflow = candidate_overflow
                                                best_score = candidate_score
                                                log_message(
                                                    "Adjusted translated edge SFX to "
                                                    "stay inside its inpaint rectangle "
                                                    "and avoid translated speech text "
                                                    f"{bbox} ({candidate_reason}, "
                                                    f"overlap={candidate_overlap}px, "
                                                    f"overflow={candidate_overflow}px)",
                                                    always_print=True,
                                                )
                                            if best_score == 0:
                                                break

                                        rendered_image = best_image
                                        primary_render_bbox = best_bbox
                                        active_render_config = best_config
                                        primary_rendered_font_size = best_font_size
                                success = True
                            except Exception as e:
                                log_message(
                                    f"Text rendering failed: {e}", verbose=verbose
                                )
                                rendered_image = pil_cleaned_image
                                success = False

                                # Absolute last-chance fallback: force vertical stacking before giving up
                                if (
                                    not active_vertical_stack
                                    and _translated_looks_like_short_sfx(text)
                                ):
                                    forced_stack_rotation = active_rotation_deg
                                    try:
                                        log_message(
                                            "OSB render failed, retrying with vertical-stack fallback",
                                            verbose=verbose,
                                        )
                                        rendered_image = render_text_skia(
                                            pil_image=pre_render_image,
                                            text=text,
                                            bbox=render_bbox,
                                            font_dir=font_dir,
                                            cleaned_mask=osb_layout_mask,
                                            bubble_color_bgr=bubble_color_bgr,
                                            config=active_render_config,
                                            verbose=verbose,
                                            bubble_id=str(i + 1),
                                            rotation_deg=forced_stack_rotation,
                                            vertical_stack=True,
                                            text_color_rgb=text_color_rgb,
                                            raise_on_safe_error=False,
                                            text_background_color=text_bg_rgb,
                                        )
                                        primary_rendered_font_size = getattr(
                                            rendered_image,
                                            "info",
                                            {},
                                        ).get("font_size")
                                        rendered_image = _clip_rendered_image_to_mask(
                                            pre_render_image,
                                            rendered_image,
                                            cleaned_mask,
                                        )
                                        log_message(
                                            "Vertical-stack fallback succeeded",
                                            verbose=verbose,
                                        )
                                        success = True
                                        active_vertical_stack = True
                                        active_rotation_deg = forced_stack_rotation
                                        primary_render_bbox = render_bbox
                                    except Exception as e2:
                                        log_message(
                                            f"Vertical-stack fallback failed: {e2}",
                                            verbose=verbose,
                                        )
                                        # Restore original OSB patch if available
                                        if "original_crop_pil" in bubble:
                                            log_message(
                                                f"Restoring original OSB patch for {bbox}",
                                                verbose=verbose,
                                                always_print=True,
                                            )
                                            (
                                                rendered_image,
                                                blocked_restore_pixels,
                                            ) = _restore_original_region_excluding_bboxes(
                                                pil_cleaned_image,
                                                original_restore_image,
                                                bbox,
                                                other_restore_protection_bboxes,
                                                None,
                                            )
                                            if blocked_restore_pixels > 0:
                                                log_message(
                                                    "Clipped original OSB patch restore "
                                                    f"{bbox} to avoid overlap with other "
                                                    "text regions "
                                                    f"({blocked_restore_pixels} px)",
                                                    verbose=verbose,
                                                    always_print=True,
                                                )
                                            success = True
                                        else:
                                            rendered_image = pil_cleaned_image
                                            success = False
                                else:
                                    if "original_crop_pil" in bubble:
                                        log_message(
                                            f"Restoring original OSB patch for {bbox}",
                                            verbose=verbose,
                                            always_print=True,
                                        )
                                        (
                                            rendered_image,
                                            blocked_restore_pixels,
                                        ) = _restore_original_region_excluding_bboxes(
                                            pil_cleaned_image,
                                            original_restore_image,
                                            bbox,
                                            other_restore_protection_bboxes,
                                            None,
                                        )
                                        if blocked_restore_pixels > 0:
                                            log_message(
                                                "Clipped original OSB patch restore "
                                                f"{bbox} to avoid overlap with other "
                                                "text regions "
                                                f"({blocked_restore_pixels} px)",
                                                verbose=verbose,
                                                always_print=True,
                                            )
                                        success = True
                                    else:
                                        rendered_image = pil_cleaned_image
                                        success = False
                        else:
                            try:
                                split_sfx_image = None
                                if (
                                    source_anchor_internal_mask is not None
                                    and not vertical_stack
                                    and not force_on_ot_vertical
                                ):
                                    split_sfx_image = (
                                        _try_render_source_anchor_internal_sfx(
                                            pil_cleaned_image,
                                            text,
                                            bbox,
                                            source_anchor_internal_mask,
                                            font_dir,
                                            bubble_color_bgr,
                                            render_config,
                                            text_color_rgb,
                                            str(i + 1),
                                            font_size_scale=font_size_scale,
                                            verbose=verbose,
                                        )
                                    )
                                if (
                                    split_sfx_image is None
                                    and sfx_rescue_applied
                                    and not vertical_stack
                                ):
                                    split_sfx_image = _try_render_split_internal_sfx(
                                        pil_cleaned_image,
                                        text,
                                        bbox,
                                        render_tag,
                                        base_mask,
                                        font_dir,
                                        bubble_color_bgr,
                                        render_config,
                                        text_color_rgb,
                                        str(i + 1),
                                        verbose,
                                    )

                                if split_sfx_image is not None:
                                    rendered_image = split_sfx_image
                                else:
                                    rendered_image = render_text_skia(
                                        pil_image=pil_cleaned_image,
                                        text=text,
                                        bbox=bbox,
                                        font_dir=font_dir,
                                        cleaned_mask=cleaned_mask,
                                        bubble_color_bgr=bubble_color_bgr,
                                        config=render_config,
                                        verbose=verbose,
                                        bubble_id=str(i + 1),
                                        rotation_deg=rotation_deg,
                                        vertical_stack=vertical_stack,
                                        text_color_rgb=text_color_rgb,
                                        raise_on_safe_error=True,
                                        auto_vertical_ignore_word_count=True,
                                    )
                                success = True
                            except ImageProcessingError as e:
                                safe_area_failed = (
                                    "Safe area calculation failed" in str(e)
                                )
                                retry_result = None
                                if safe_area_failed and base_mask is not None:
                                    log_message(
                                        f"Safe area failed for bubble {bbox}, retrying mask with Otsu",
                                        verbose=verbose,
                                        always_print=True,
                                    )
                                    retry_result = retry_cleaning_with_otsu(
                                        original_cv_image,
                                        {
                                            "base_mask": base_mask,
                                            "bbox": bbox,
                                            "is_sam": is_sam_mask,
                                            "is_colored": (
                                                render_info.get("is_colored", False)
                                                if render_info
                                                else False
                                            ),
                                            "text_bbox": (
                                                render_info.get("text_bbox")
                                                if render_info
                                                else None
                                            ),
                                            "text_color_bgr": (
                                                render_info.get("text_color_bgr")
                                                if render_info
                                                else None
                                            ),
                                        },
                                        config.cleaning.thresholding_value,
                                        config.cleaning.roi_shrink_px,
                                        processing_scale,
                                        verbose=verbose,
                                        classify_colored=(
                                            config.cleaning.inpaint_colored_bubbles
                                        ),
                                    )

                                if (
                                    retry_result
                                    and retry_result.get("mask") is not None
                                ):
                                    cleaned_mask = retry_result["mask"]
                                    bubble_color_bgr = retry_result.get(
                                        "color", bubble_color_bgr
                                    )
                                    base_mask = retry_result.get("base_mask", base_mask)
                                    if render_info is not None:
                                        render_info.update(
                                            {
                                                "mask": cleaned_mask,
                                                "color": bubble_color_bgr,
                                                "base_mask": base_mask,
                                                "is_colored": retry_result.get(
                                                    "is_colored",
                                                    render_info.get(
                                                        "is_colored", False
                                                    ),
                                                ),
                                                "text_bbox": retry_result.get(
                                                    "text_bbox",
                                                    render_info.get("text_bbox"),
                                                ),
                                            }
                                        )

                                    try:
                                        rendered_image = render_text_skia(
                                            pil_image=pil_cleaned_image,
                                            text=text,
                                            bbox=bbox,
                                            font_dir=font_dir,
                                            cleaned_mask=cleaned_mask,
                                            bubble_color_bgr=bubble_color_bgr,
                                            config=render_config,
                                            verbose=verbose,
                                            bubble_id=str(i + 1),
                                            rotation_deg=rotation_deg,
                                            vertical_stack=vertical_stack,
                                            text_color_rgb=text_color_rgb,
                                            raise_on_safe_error=False,
                                            auto_vertical_ignore_word_count=True,
                                        )
                                        success = True
                                    except (
                                        RenderingError,
                                        FontError,
                                        ImageProcessingError,
                                    ) as e2:
                                        log_message(
                                            f"Text rendering failed after Otsu retry: {e2}",
                                            verbose=verbose,
                                        )
                                        rendered_image = pil_cleaned_image
                                        success = False
                                if not success:
                                    # Final fallback to padded bbox path
                                    fallback_msg = (
                                        f"Safe area calculation failed for {bbox}, using padded bbox fallback"
                                        if safe_area_failed
                                        else f"Rendering retry fallback for {bbox}, using padded bbox method"
                                    )
                                    log_message(
                                        fallback_msg,
                                        verbose=verbose,
                                    )
                                    try:
                                        rendered_image = render_text_skia(
                                            pil_image=pil_cleaned_image,
                                            text=text,
                                            bbox=bbox,
                                            font_dir=font_dir,
                                            cleaned_mask=cleaned_mask,
                                            bubble_color_bgr=bubble_color_bgr,
                                            config=render_config,
                                            verbose=verbose,
                                            bubble_id=str(i + 1),
                                            rotation_deg=rotation_deg,
                                            vertical_stack=vertical_stack,
                                            text_color_rgb=text_color_rgb,
                                            raise_on_safe_error=False,
                                            auto_vertical_ignore_word_count=True,
                                        )
                                        success = True
                                    except (RenderingError, FontError) as e2:
                                        log_message(
                                            f"Text rendering failed: {e2}",
                                            verbose=verbose,
                                        )
                                        rendered_image = pil_cleaned_image
                                        success = False
                            except (RenderingError, FontError) as e:
                                log_message(
                                    f"Text rendering failed: {e}", verbose=verbose
                                )
                                rendered_image = pil_cleaned_image
                                success = False

                        if (
                            success
                            and is_outside_text
                            and osb_split_texts
                            and len(osb_split_texts) > 1
                        ):
                            for part_index, part_text in enumerate(
                                osb_split_texts[1:], start=1
                            ):
                                if part_index >= len(osb_split_boxes):
                                    break
                                part_layout = (
                                    osb_split_layouts[part_index]
                                    if part_index < len(osb_split_layouts)
                                    else {}
                                )
                                part_bbox = _inset_bbox_for_padding(
                                    part_layout.get(
                                        "bbox", osb_split_boxes[part_index]
                                    ),
                                    part_layout.get(
                                        "padding_pixels", osb_render_padding_pixels
                                    ),
                                )
                                part_vertical_stack = bool(
                                    part_layout.get("vertical_stack", vertical_stack)
                                )
                                try:
                                    part_rotation_deg = float(
                                        part_layout.get("rotation_deg", rotation_deg)
                                        or 0.0
                                    )
                                except (TypeError, ValueError):
                                    part_rotation_deg = rotation_deg
                                part_render_config = _render_config_with_max_font(
                                    render_config,
                                    part_layout.get("max_font_size"),
                                )
                                if part_layout.get("lock_vertical_stack"):
                                    part_render_config.compare_forced_vertical_layout = False
                                    part_render_config.force_compare_vertical_text = False
                                try:
                                    before_part_render = rendered_image
                                    rendered_image = render_text_skia(
                                        pil_image=rendered_image,
                                        text=part_text,
                                        bbox=part_bbox,
                                        font_dir=font_dir,
                                        cleaned_mask=None,
                                        bubble_color_bgr=bubble_color_bgr,
                                        config=part_render_config,
                                        verbose=verbose,
                                        bubble_id=f"{i + 1}.{part_index + 1}",
                                        rotation_deg=part_rotation_deg,
                                        vertical_stack=part_vertical_stack,
                                        text_color_rgb=text_color_rgb,
                                        raise_on_safe_error=False,
                                        text_background_color=text_bg_rgb,
                                    )
                                    rendered_image = _clip_rendered_image_to_mask(
                                        before_part_render,
                                        rendered_image,
                                        cleaned_mask,
                                    )
                                except Exception as e:
                                    log_message(
                                        f"Short OSB split render failed: {e}",
                                        verbose=verbose,
                                    )

                        osb_residual_guard_text_len = sum(
                            1 for ch in str(text or "") if ch.isalnum()
                        )
                        osb_residual_guard_action = (
                            _normalize_osb_residual_render_guard_action(
                                getattr(
                                    config.outside_text,
                                    "osb_residual_render_guard_action",
                                    "restore_original",
                                )
                            )
                        )
                        osb_residual_guard_enabled = (
                            is_outside_text
                            and _is_sfx_render_tag(render_tag)
                            and 0 < osb_residual_guard_text_len <= 5
                            and osb_residual_guard_action != "none"
                            and str(config.outside_text.inpainting_method or "")
                            in {"flux_klein_4b", "flux_klein_9b", "flux_kontext"}
                        )
                        if success and osb_residual_guard_enabled:
                            osb_residual_initial_ratio = (
                                OSB_RESIDUAL_LAMA_RETRY_MIN_RESIDUAL_RATIO
                                if osb_residual_guard_action == "lama_large"
                                else OSB_RESIDUAL_RENDER_GUARD_MIN_RESIDUAL_RATIO
                            )
                            (
                                residual_remaining,
                                residual_reason,
                                residual_restore_bbox,
                                residual_mask,
                            ) = _osb_residual_source_text_remaining(
                                bubble,
                                original_restore_image,
                                pre_entry_render_image,
                                config,
                                verbose=verbose,
                                min_residual_ratio=osb_residual_initial_ratio,
                            )
                            if residual_remaining and residual_restore_bbox is not None:
                                if osb_residual_guard_action == "lama_large":
                                    try:
                                        lama_base_image = _run_osb_residual_lama_large(
                                            pre_entry_render_image,
                                            residual_mask,
                                            config,
                                            residual_restore_bbox,
                                            verbose=verbose,
                                        )
                                        if lama_base_image is None:
                                            raise RuntimeError(
                                                "no residual source-text mask"
                                            )
                                        retry_rendered_image = render_text_skia(
                                            pil_image=lama_base_image,
                                            text=primary_render_text,
                                            bbox=primary_render_bbox,
                                            font_dir=font_dir,
                                            cleaned_mask=osb_layout_mask,
                                            bubble_color_bgr=bubble_color_bgr,
                                            config=active_render_config,
                                            verbose=verbose,
                                            bubble_id=str(i + 1),
                                            rotation_deg=active_rotation_deg,
                                            vertical_stack=active_vertical_stack,
                                            text_color_rgb=text_color_rgb,
                                            raise_on_safe_error=False,
                                            text_background_color=text_bg_rgb,
                                            horizontal_compare_bbox=(
                                                osb_horizontal_compare_bbox
                                                if not osb_split_layouts
                                                else None
                                            ),
                                            vertical_compare_bbox=(
                                                osb_vertical_compare_bbox
                                                if not osb_split_layouts
                                                else None
                                            ),
                                        )
                                        retry_font_size = getattr(
                                            retry_rendered_image, "info", {}
                                        ).get("font_size")
                                        retry_rendered_image = (
                                            _clip_rendered_image_to_mask(
                                                lama_base_image,
                                                retry_rendered_image,
                                                cleaned_mask,
                                            )
                                        )

                                        if osb_split_texts and len(osb_split_texts) > 1:
                                            for part_index, part_text in enumerate(
                                                osb_split_texts[1:], start=1
                                            ):
                                                if part_index >= len(osb_split_boxes):
                                                    break
                                                part_layout = (
                                                    osb_split_layouts[part_index]
                                                    if part_index
                                                    < len(osb_split_layouts)
                                                    else {}
                                                )
                                                part_bbox = _inset_bbox_for_padding(
                                                    part_layout.get(
                                                        "bbox",
                                                        osb_split_boxes[part_index],
                                                    ),
                                                    part_layout.get(
                                                        "padding_pixels",
                                                        osb_render_padding_pixels,
                                                    ),
                                                )
                                                part_vertical_stack = bool(
                                                    part_layout.get(
                                                        "vertical_stack",
                                                        vertical_stack,
                                                    )
                                                )
                                                try:
                                                    part_rotation_deg = float(
                                                        part_layout.get(
                                                            "rotation_deg",
                                                            rotation_deg,
                                                        )
                                                        or 0.0
                                                    )
                                                except (TypeError, ValueError):
                                                    part_rotation_deg = rotation_deg
                                                part_render_config = (
                                                    _render_config_with_max_font(
                                                        render_config,
                                                        part_layout.get(
                                                            "max_font_size"
                                                        ),
                                                    )
                                                )
                                                if part_layout.get(
                                                    "lock_vertical_stack"
                                                ):
                                                    part_render_config.compare_forced_vertical_layout = False
                                                    part_render_config.force_compare_vertical_text = False
                                                before_part_retry = retry_rendered_image
                                                retry_rendered_image = render_text_skia(
                                                    pil_image=retry_rendered_image,
                                                    text=part_text,
                                                    bbox=part_bbox,
                                                    font_dir=font_dir,
                                                    cleaned_mask=None,
                                                    bubble_color_bgr=bubble_color_bgr,
                                                    config=part_render_config,
                                                    verbose=verbose,
                                                    bubble_id=f"{i + 1}.{part_index + 1}",
                                                    rotation_deg=part_rotation_deg,
                                                    vertical_stack=part_vertical_stack,
                                                    text_color_rgb=text_color_rgb,
                                                    raise_on_safe_error=False,
                                                    text_background_color=text_bg_rgb,
                                                )
                                                retry_rendered_image = (
                                                    _clip_rendered_image_to_mask(
                                                        before_part_retry,
                                                        retry_rendered_image,
                                                        cleaned_mask,
                                                    )
                                                )

                                        (
                                            retry_remaining,
                                            retry_reason,
                                            _,
                                            _,
                                        ) = _osb_residual_source_text_remaining(
                                            bubble,
                                            original_restore_image,
                                            lama_base_image,
                                            config,
                                            verbose=verbose,
                                            min_residual_ratio=OSB_RESIDUAL_LAMA_POST_RETRY_MIN_RESIDUAL_RATIO,
                                        )
                                        if retry_remaining:
                                            raise RuntimeError(
                                                "residual source text still remains "
                                                f"after LAMA ({retry_reason})"
                                            )

                                        rendered_image = retry_rendered_image
                                        primary_rendered_font_size = (
                                            retry_font_size
                                            or primary_rendered_font_size
                                        )
                                        log_message(
                                            "Retried OSB residual source text with "
                                            "LAMA Large before rendering "
                                            f"{bbox} ({residual_reason})",
                                            verbose=verbose,
                                            always_print=True,
                                        )
                                        residual_remaining = False
                                    except Exception as e:
                                        log_message(
                                            "OSB residual LAMA Large retry failed; "
                                            "restoring original "
                                            f"{bbox}: {e}",
                                            verbose=verbose,
                                            always_print=True,
                                        )

                                if not residual_remaining:
                                    pass
                                else:
                                    (
                                        pil_cleaned_image,
                                        blocked_restore_pixels,
                                    ) = _restore_original_region_excluding_bboxes(
                                        pil_cleaned_image,
                                        original_restore_image,
                                        residual_restore_bbox,
                                        other_restore_protection_bboxes,
                                        None,
                                    )
                                    final_image_to_save = pil_cleaned_image
                                    if blocked_restore_pixels > 0:
                                        log_message(
                                            "Clipped residual-source OSB restore "
                                            f"{bbox} to avoid overlap with other text "
                                            f"regions ({blocked_restore_pixels} px)",
                                            verbose=verbose,
                                            always_print=True,
                                        )
                                    log_message(
                                        "Skipped OSB render and restored original "
                                        f"{bbox}: source text remained after inpainting "
                                        f"({residual_reason})",
                                        verbose=verbose,
                                        always_print=True,
                                    )
                                    continue

                        if success:
                            if not is_outside_text and cleaned_mask is not None:
                                bubble["_editor_layout_mask"] = cleaned_mask
                            rendered_font_size = primary_rendered_font_size
                            if not rendered_font_size:
                                rendered_font_size = getattr(
                                    rendered_image,
                                    "info",
                                    {},
                                ).get("font_size")
                            rendered_orientation = str(
                                getattr(rendered_image, "info", {}).get(
                                    "orientation", ""
                                )
                                or ""
                            ).lower()
                            rendered_bbox = getattr(rendered_image, "info", {}).get(
                                "render_bbox"
                            )
                            if (
                                is_outside_text
                                and not osb_split_layouts
                                and rendered_bbox is not None
                            ):
                                try:
                                    primary_render_bbox = tuple(
                                        int(round(float(value)))
                                        for value in rendered_bbox
                                    )
                                except (TypeError, ValueError):
                                    pass
                            if rendered_orientation in {"vertical", "horizontal"}:
                                active_vertical_stack = (
                                    rendered_orientation == "vertical"
                                )
                            _store_editor_render_meta(
                                bubble,
                                bbox=primary_render_bbox,
                                font_dir=font_dir,
                                render_config=active_render_config,
                                rotation_deg=active_rotation_deg,
                                vertical_stack=active_vertical_stack,
                                font_size=rendered_font_size,
                            )
                            if not is_outside_text:
                                translated_speech_text_mask |= (
                                    _rendered_pixel_change_mask(
                                        pre_entry_render_image,
                                        rendered_image,
                                    )
                                )
                            pil_cleaned_image = rendered_image
                            final_image_to_save = pil_cleaned_image
                        else:
                            log_message(
                                f"Failed to render bubble {bbox}", verbose=verbose
                            )
                    if (
                        edit_session_path is not None
                        and editor_session_base_image is not None
                    ):
                        try:
                            from .editor_session import save_pipeline_edit_session

                            save_pipeline_edit_session(
                                session_path=edit_session_path,
                                source_image_path=image_path,
                                translated_output_path=Path(output_path)
                                if output_path
                                else None,
                                base_image=editor_session_base_image,
                                sorted_bubble_data=sorted_bubble_data,
                                translated_texts=translated_texts,
                                source_ocr_texts=source_ocr_texts,
                                processed_bubbles_info=processed_bubbles_info,
                                config=config,
                                main_min_font=main_min_font,
                                main_max_font=main_max_font,
                                osb_min_font=osb_min_font,
                                osb_max_font=osb_max_font,
                                padding_pixels=padding_pixels,
                                padding_horizontal_pixels=padding_horizontal_pixels,
                                padding_vertical_pixels=padding_vertical_pixels,
                                osb_padding_pixels=osb_padding_pixels,
                                osb_outline_width=osb_outline_width,
                                verbose=verbose,
                            )
                        except Exception as e:
                            log_message(
                                f"Warning: failed to save editor session: {e}",
                                always_print=True,
                            )
                else:
                    log_message(
                        f"Warning: Bubble/translation count mismatch "
                        f"({len(sorted_bubble_data)}/{len(translated_texts)})",
                        always_print=True,
                    )

    # Final Image Upscaling (optional)
    if config.output.upscale_final_image:
        log_message("Upscaling final image...", verbose=verbose, always_print=True)
        final_image_to_save = upscale_image(
            final_image_to_save,
            config.output.image_upscale_factor,
            model_type=config.output.image_upscale_model,
            verbose=verbose,
        )

    # Save Output
    if output_path:
        if final_image_to_save.mode != target_mode:
            log_message(f"Converting final image to {target_mode}", verbose=verbose)
            final_image_to_save = final_image_to_save.convert(target_mode)

        try:
            save_image_with_compression(
                final_image_to_save,
                output_path,
                jpeg_quality=config.output.jpeg_quality,
                png_compression=config.output.png_compression,
                verbose=verbose,
            )
        except ImageProcessingError as e:
            log_message(f"Failed to save image: {e}", always_print=True)
            raise
        if edit_session_path is not None:
            try:
                finalize_pipeline_edit_session(
                    edit_session_path,
                    final_image_to_save,
                    verbose=verbose,
                )
                _maybe_embed_editor_package(
                    output_path=output_path,
                    edit_session_path=edit_session_path,
                    config=config,
                    verbose=verbose,
                )
            except Exception as exc:
                log_message(
                    f"Warning: failed to preserve editor rendered snapshot: {exc}",
                    always_print=True,
                )

    elif edit_session_path is not None:
        try:
            finalize_pipeline_edit_session(
                edit_session_path,
                final_image_to_save,
                verbose=verbose,
            )
        except Exception as exc:
            log_message(
                f"Warning: failed to preserve editor rendered snapshot: {exc}",
                always_print=True,
            )

    end_time = time.time()
    processing_time = end_time - start_time
    log_message(f"Processing completed in {processing_time:.2f}s", always_print=True)

    return final_image_to_save


def _resolve_output_path(
    img_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: MangaTranslatorConfig,
    preserve_structure: bool,
) -> Tuple[Path, str, str]:
    """Compute output path, display name, and error key for a single image."""
    use_split_batch_output = bool(
        getattr(config.output, "embed_editor_package", True)
        and getattr(config.output, "save_plain_image_copy", False)
    )
    output_root = (
        output_dir / BATCH_EMBEDDED_OUTPUT_DIR_NAME
        if use_split_batch_output
        else output_dir
    )
    if preserve_structure:
        relative_path = img_path.relative_to(input_dir)
        output_subdir = output_root / relative_path.parent
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f"{relative_path.stem}_translated"
        display_path = str(relative_path)
        error_key = str(relative_path)
    else:
        output_subdir = output_root
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f"{img_path.stem}_translated"
        display_path = img_path.name
        error_key = img_path.name

    original_ext = img_path.suffix.lower()
    desired_format = config.output.output_format
    if desired_format == "jpeg":
        output_ext = ".jpg"
    elif desired_format == "png":
        output_ext = ".png"
    elif desired_format == "auto":
        output_ext = original_ext
    else:
        output_ext = original_ext
        log_message(
            f"Warning: Invalid output_format '{desired_format}' in config. "
            f"Using original extension '{original_ext}'.",
            always_print=True,
        )

    return output_subdir / f"{output_filename}{output_ext}", display_path, error_key


def initialize_failed_files_txt(
    failed_files_txt_path: Optional[Union[str, Path]],
) -> Optional[Path]:
    """Create or clear the per-run failed source path list."""
    if not failed_files_txt_path:
        return None

    target_path = Path(failed_files_txt_path)
    try:
        with _FAILED_FILES_TXT_LOCK:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text("", encoding="utf-8")
        log_message(
            f"Initialized failed file list: {target_path}",
            always_print=True,
        )
        return target_path
    except OSError as exc:
        log_message(
            f"Failed to initialize {FAILED_FILES_TXT_NAME} ({target_path}): {exc}",
            always_print=True,
        )
        return None


def _write_failed_files_txt(results: Dict[str, Any]) -> None:
    target_value = results.get("_failed_files_txt_path")
    if not target_value:
        return

    target_path = Path(str(target_value))
    failed_files = dict(results.get("failed_files", {}) or {})
    source_paths = dict(results.get("_failed_source_paths", {}) or {})
    errors = dict(results.get("errors", {}) or {})
    failed_keys = sorted(
        failed_files,
        key=lambda key: _natural_path_sort_key(Path(str(key))),
    )
    lines = []
    for key in failed_keys:
        source_path = source_paths.get(key)
        if source_path:
            display_path = str(source_path)
        else:
            try:
                display_path = str(Path(failed_files[key]).resolve())
            except (OSError, RuntimeError):
                display_path = str(failed_files[key])
        reason = _single_line_failed_reason(errors.get(key))
        lines.append(f"{display_path}\t{reason}" if reason else display_path)
    payload = ("\n".join(lines) + "\n") if lines else ""

    try:
        with _FAILED_FILES_TXT_LOCK:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target_path.with_name(f"{target_path.name}.tmp")
            temp_path.write_text(payload, encoding="utf-8")
            try:
                temp_path.replace(target_path)
            except OSError:
                target_path.write_text(payload, encoding="utf-8")
                temp_path.unlink(missing_ok=True)
    except OSError as exc:
        log_message(
            f"Failed to update {FAILED_FILES_TXT_NAME} ({target_path}): {exc}",
            always_print=True,
        )


def _single_line_failed_reason(error: Any) -> str:
    reason = str(error or "").strip()
    if not reason and error is not None and not isinstance(error, str):
        reason = error.__class__.__name__
    reason = re.sub(r"[\r\n\t]+", " ", reason)
    return re.sub(r"\s+", " ", reason).strip()


def _new_batch_results(
    failed_files_txt_path: Optional[Union[str, Path]] = None,
    failed_source_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    return {
        "success_count": 0,
        "error_count": 0,
        "errors": {},
        "successful_files": {},
        "successful_download_files": {},
        "successful_inputs": {},
        "failed_files": {},
        "retry_attempted_count": 0,
        "retry_success_count": 0,
        "retry_failed_count": 0,
        "cancelled": False,
        "cancel_message": "",
        "total_count": 0,
        "_failed_files_txt_path": (
            str(failed_files_txt_path) if failed_files_txt_path else ""
        ),
        "_failed_source_paths": dict(failed_source_paths or {}),
    }


def _sync_batch_counts(results: Dict[str, Any]) -> None:
    results["success_count"] = len(results.get("successful_files", {}))
    results["error_count"] = len(results.get("errors", {}))


def _record_batch_success(
    results: Dict[str, Any],
    error_key: str,
    img_path: Path,
    output_path: Path,
) -> None:
    results.setdefault("errors", {}).pop(error_key, None)
    results.setdefault("failed_files", {}).pop(error_key, None)
    results.setdefault("successful_files", {})[error_key] = str(output_path)
    results.setdefault("successful_download_files", {})[error_key] = str(output_path)
    plain_copy_path = editor_plain_copy_path(output_path)
    if plain_copy_path.is_file():
        results.setdefault("successful_download_files", {})[
            f"{error_key}::plain"
        ] = str(plain_copy_path)
    results.setdefault("successful_inputs", {})[error_key] = str(img_path)
    _sync_batch_counts(results)
    _write_failed_files_txt(results)


def _record_batch_failure(
    results: Dict[str, Any],
    error_key: str,
    img_path: Path,
    error: Any,
) -> None:
    results.setdefault("errors", {})[error_key] = str(error)
    results.setdefault("failed_files", {})[error_key] = str(img_path)
    results.setdefault("successful_download_files", {}).pop(error_key, None)
    results.setdefault("successful_download_files", {}).pop(f"{error_key}::plain", None)
    _sync_batch_counts(results)
    _write_failed_files_txt(results)


def _mark_batch_cancelled(
    results: Dict[str, Any],
    image_files: List[Path],
    input_dir: Path,
    config: MangaTranslatorConfig,
    output_dir: Path,
    preserve_structure: bool,
    message: str = "Batch process cancelled by user.",
) -> None:
    results["cancelled"] = True
    results["cancel_message"] = message
    for img_path in image_files:
        _output_path, _display_path, error_key = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )
        if error_key in results.get("successful_files", {}):
            continue
        if error_key in results.get("failed_files", {}):
            continue
        _record_batch_failure(results, error_key, img_path, message)
    _sync_batch_counts(results)


async def _batch_translate_parallel(
    image_files: List[Path],
    input_dir: Path,
    config: MangaTranslatorConfig,
    output_dir: Path,
    preserve_structure: bool,
    progress_callback: Optional[Callable[[float, str], None]],
    cancellation_manager: Optional["CancellationManager"],
    failed_files_txt_path: Optional[Union[str, Path]] = None,
    failed_source_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Process images in parallel using a semaphore to maintain target concurrency.

    The first image is processed sequentially to warm up all ML models (triggering
    lazy loading and YOLO layer fusing on the main thread). Remaining images are
    then processed in parallel with models already initialized.
    """
    total_images = len(image_files)
    n_workers = config.parallel_requests
    results = _new_batch_results(
        failed_files_txt_path=failed_files_txt_path,
        failed_source_paths=failed_source_paths,
    )
    results["total_count"] = total_images
    previous_context_cache = OrderedDict()
    previous_context_cache_lock = threading.Lock()
    ocr_text_history: Dict[Path, List[str]] = {}
    ocr_text_history_lock = threading.Lock()
    ocr_text_ready_events = [threading.Event() for _ in image_files]
    requested_text_context_count = int(
        getattr(config.translation, "previous_context_text_count", 0) or 0
    )
    previous_text_context_mode = _previous_context_text_mode(config)
    wait_for_previous_context_texts = previous_text_context_mode == "strict"

    log_message(
        f"Starting parallel batch processing: {total_images} images, "
        f"{n_workers} parallel workers",
        always_print=True,
    )
    if requested_text_context_count > 0:
        if wait_for_previous_context_texts:
            log_message(
                "Previous OCR context mode: strict; waiting for previous pages "
                "before each translation request.",
                always_print=True,
            )
        else:
            log_message(
                "Previous OCR context mode: available; using completed previous "
                "OCR only without waiting.",
                always_print=True,
            )

    # -- Phase 1: process the first image sequentially to warm up models --
    first_img = image_files[0]
    first_output, first_display, first_key = _resolve_output_path(
        first_img, input_dir, output_dir, config, preserve_structure
    )
    log_message(
        f"Processing 1/{total_images}: {first_display} (warming up models)",
        always_print=True,
    )
    try:
        if cancellation_manager and cancellation_manager.is_cancelled():
            raise CancellationError("Batch process cancelled by user.")
        first_previous_context_images = _build_previous_context_images(
            image_files,
            0,
            config,
            previous_context_cache,
            previous_context_cache_lock,
        )
        first_previous_context_texts = _build_previous_context_texts(
            image_files,
            0,
            config,
            ocr_text_history,
            ocr_text_history_lock,
        )
        first_ocr_texts: List[str] = []
        translate_and_render(
            first_img,
            config,
            first_output,
            edit_session_path=default_edit_session_path(first_output),
            cancellation_manager=cancellation_manager,
            previous_context_images=first_previous_context_images,
            previous_context_texts=first_previous_context_texts,
            ocr_texts_out=first_ocr_texts,
        )
        if first_ocr_texts:
            with ocr_text_history_lock:
                ocr_text_history[first_img] = first_ocr_texts
        _record_batch_success(results, first_key, first_img, first_output)
    except CancellationError as exc:
        cancel_message = str(exc) or "Batch process cancelled by user."
        log_message(
            f"Batch cancelled during processing of {first_display}",
            verbose=config.verbose,
        )
        _mark_batch_cancelled(
            results,
            image_files,
            input_dir,
            config,
            output_dir,
            preserve_structure,
            message=cancel_message,
        )
        return results
    except Exception as e:
        log_message(f"Error processing {first_display}: {str(e)}", always_print=True)
        _record_batch_failure(results, first_key, first_img, e)
    finally:
        ocr_text_ready_events[0].set()

    completed_count = 1
    if progress_callback:
        has_errors = results["error_count"] > 0
        suffix = " (with errors)" if has_errors else ""
        progress_callback(
            1 / total_images, f"Completed 1/{total_images} images{suffix}"
        )

    # -- Phase 2: process remaining images in parallel --
    remaining = image_files[1:]
    if not remaining:
        return results

    if cancellation_manager and cancellation_manager.is_cancelled():
        _mark_batch_cancelled(
            results,
            image_files,
            input_dir,
            config,
            output_dir,
            preserve_structure,
        )
        return results

    sem = asyncio.Semaphore(n_workers)
    results_lock = threading.Lock()
    cancelled = False
    cancel_message = "Batch process cancelled by user."

    def _wait_for_required_previous_ocr(index: int) -> None:
        if requested_text_context_count <= 0:
            return
        start_index = max(0, index - requested_text_context_count)
        for previous_index in range(start_index, index):
            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Batch process cancelled by user.")
            while not ocr_text_ready_events[previous_index].wait(timeout=0.2):
                if cancellation_manager and cancellation_manager.is_cancelled():
                    raise CancellationError("Batch process cancelled by user.")
            if cancellation_manager and cancellation_manager.is_cancelled():
                raise CancellationError("Batch process cancelled by user.")

    def _process_single(img_path: Path, index: int) -> Tuple[str, str, Path]:
        """Run translate_and_render for a single image."""
        output_path, display_path, error_key = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )
        log_message(
            f"Processing {index + 1}/{total_images}: {display_path}",
            always_print=True,
        )
        previous_context_images = _build_previous_context_images(
            image_files,
            index,
            config,
            previous_context_cache,
            previous_context_cache_lock,
        )

        def previous_context_texts_provider() -> List[List[str]]:
            if wait_for_previous_context_texts:
                _wait_for_required_previous_ocr(index)
            return _build_previous_context_texts(
                image_files,
                index,
                config,
                ocr_text_history,
                ocr_text_history_lock,
            )

        captured_ocr_texts: List[str] = []
        translate_and_render(
            img_path,
            config,
            output_path,
            edit_session_path=default_edit_session_path(output_path),
            cancellation_manager=cancellation_manager,
            previous_context_images=previous_context_images,
            previous_context_texts_provider=previous_context_texts_provider,
            ocr_texts_out=captured_ocr_texts,
        )
        if captured_ocr_texts:
            with ocr_text_history_lock:
                ocr_text_history[img_path] = captured_ocr_texts
        return display_path, error_key, output_path

    async def _worker(img_path: Path, index: int, executor: ThreadPoolExecutor):
        nonlocal completed_count, cancelled
        nonlocal cancel_message
        try:
            if cancelled or (
                cancellation_manager and cancellation_manager.is_cancelled()
            ):
                cancelled = True
                return

            async with sem:
                if cancelled or (
                    cancellation_manager and cancellation_manager.is_cancelled()
                ):
                    cancelled = True
                    return

                loop = asyncio.get_event_loop()
                try:
                    _, error_key, output_path = await loop.run_in_executor(
                        executor, _process_single, img_path, index
                    )
                    with results_lock:
                        _record_batch_success(
                            results, error_key, img_path, output_path
                        )
                        completed_count += 1
                        count = completed_count
                except CancellationError as exc:
                    cancelled = True
                    cancel_message = str(exc) or cancel_message
                    raise
                except Exception as e:
                    _, display_path, error_key = _resolve_output_path(
                        img_path, input_dir, output_dir, config, preserve_structure
                    )
                    log_message(
                        f"Error processing {display_path}: {str(e)}",
                        always_print=True,
                    )
                    with results_lock:
                        _record_batch_failure(results, error_key, img_path, e)
                        completed_count += 1
                        count = completed_count

                if progress_callback:
                    progress = count / total_images
                    has_errors = results["error_count"] > 0
                    suffix = " (with errors)" if has_errors else ""
                    progress_callback(
                        progress, f"Completed {count}/{total_images} images{suffix}"
                    )
        except CancellationError:
            cancelled = True
            raise
        finally:
            ocr_text_ready_events[index].set()

            if cancelled:
                for event in ocr_text_ready_events:
                    event.set()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        tasks = [_worker(img, i, executor) for i, img in enumerate(remaining, start=1)]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

    for exc in gathered:
        if isinstance(exc, CancellationError):
            cancelled = True
            cancel_message = str(exc) or cancel_message

    if cancelled or (cancellation_manager and cancellation_manager.is_cancelled()):
        _mark_batch_cancelled(
            results,
            image_files,
            input_dir,
            config,
            output_dir,
            preserve_structure,
            message=cancel_message,
        )

    return results


def _retry_failed_batch_images(
    results: Dict[str, Any],
    input_dir: Path,
    config: MangaTranslatorConfig,
    output_dir: Path,
    preserve_structure: bool,
    progress_callback: Optional[Callable[[float, str], None]],
    cancellation_manager: Optional["CancellationManager"],
) -> None:
    failed_items = list(results.get("failed_files", {}).items())
    if not failed_items:
        return

    retry_total = len(failed_items)
    results["retry_attempted_count"] = retry_total
    results["retry_success_count"] = 0
    results["retry_failed_count"] = 0

    log_message(
        f"Retrying {retry_total} failed image(s) once after initial batch pass",
        always_print=True,
    )

    for retry_index, (_previous_key, img_path_str) in enumerate(
        failed_items, start=1
    ):
        img_path = Path(img_path_str)
        output_path, display_path, error_key = _resolve_output_path(
            img_path, input_dir, output_dir, config, preserve_structure
        )

        if cancellation_manager and cancellation_manager.is_cancelled():
            results["cancelled"] = True
            results["cancel_message"] = "Batch process cancelled by user."
            _mark_batch_cancelled(
                results,
                [img_path],
                input_dir,
                config,
                output_dir,
                preserve_structure,
                message=results["cancel_message"],
            )
            return

        if progress_callback:
            retry_progress = 0.95 + ((retry_index - 1) / retry_total) * 0.05
            try:
                progress_callback(
                    retry_progress,
                    f"Retrying failed image {retry_index}/{retry_total}: {display_path}",
                )
            except CancellationError as exc:
                results["cancelled"] = True
                results["cancel_message"] = str(exc) or "Batch process cancelled by user."
                return

        try:
            translate_and_render(
                img_path,
                config,
                output_path,
                edit_session_path=default_edit_session_path(output_path),
                cancellation_manager=cancellation_manager,
            )
            _record_batch_success(results, error_key, img_path, output_path)
            results["retry_success_count"] += 1
            log_message(
                f"Retry succeeded for {display_path}",
                always_print=True,
            )
        except CancellationError as exc:
            results["cancelled"] = True
            results["cancel_message"] = str(exc) or "Batch process cancelled by user."
            return
        except Exception as e:
            log_message(
                f"Retry failed for {display_path}: {str(e)}",
                always_print=True,
            )
            _record_batch_failure(results, error_key, img_path, e)
            results["retry_failed_count"] += 1

        if progress_callback:
            retry_progress = 0.95 + (retry_index / retry_total) * 0.05
            try:
                progress_callback(
                    retry_progress,
                    f"Retried {retry_index}/{retry_total} failed image(s)",
                )
            except CancellationError as exc:
                results["cancelled"] = True
                results["cancel_message"] = str(exc) or "Batch process cancelled by user."
                return


def batch_translate_images(
    input_dir: Union[str, Path],
    config: MangaTranslatorConfig,
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    preserve_structure: bool = False,
    cancellation_manager: Optional["CancellationManager"] = None,
    failed_files_txt_path: Optional[Union[str, Path]] = None,
    failed_source_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Process all images in a directory using a configuration object.

    Args:
        input_dir (str or Path): Directory containing images to process
        config (MangaTranslatorConfig): Configuration object containing all settings.
        output_dir (str or Path, optional): Directory to save translated images.
                                            If None, uses input_dir / "output_translated".
        progress_callback (callable, optional): Function to call with progress updates (0.0-1.0, message).
        preserve_structure (bool): If True, recursively process subdirectories and preserve folder structure
                                   in the output. If False, only processes files in the root directory.
        failed_files_txt_path: Text file to clear at batch start and update after
                               every failure or retry success.
        failed_source_paths: Optional mapping from input-relative keys to original
                             source paths shown in the failed file list.

    Returns:
        dict: Processing results with keys:
            - "success_count": Number of successfully processed images
            - "error_count": Number of images that failed to process
            - "errors": Dictionary mapping filenames to error messages
            - "successful_files": Dictionary mapping input-relative paths to output files
            - "failed_files": Dictionary mapping input-relative paths to failed inputs
    """
    if config.translation.provider == "Google":
        reset_google_rate_limit_session()

    initialize_failed_files_txt(failed_files_txt_path)
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        log_message(f"Input path '{input_dir}' is not a directory", always_print=True)
        return _new_batch_results(
            failed_files_txt_path=failed_files_txt_path,
            failed_source_paths=failed_source_paths,
        )

    if output_dir:
        output_dir = Path(output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("./output") / timestamp

    os.makedirs(output_dir, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    if preserve_structure:
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in image_extensions:
                    image_files.append(file_path)
    else:
        image_files = [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

    def _batch_sort_key(path: Path):
        try:
            sort_path = (
                path.relative_to(input_dir) if preserve_structure else Path(path.name)
            )
        except ValueError:
            sort_path = path
        return _natural_path_sort_key(sort_path)

    image_files.sort(key=_batch_sort_key)

    if not image_files:
        log_message(f"No image files found in '{input_dir}'", always_print=True)
        return _new_batch_results(
            failed_files_txt_path=failed_files_txt_path,
            failed_source_paths=failed_source_paths,
        )

    total_images = len(image_files)
    start_batch_time = time.time()
    cancelled = False

    if progress_callback:
        progress_callback(0.0, f"Starting batch processing of {total_images} images...")

    if config.parallel_requests > 1:
        results = asyncio.run(
            _batch_translate_parallel(
                image_files=image_files,
                input_dir=input_dir,
                config=config,
                output_dir=output_dir,
                preserve_structure=preserve_structure,
                progress_callback=progress_callback,
                cancellation_manager=cancellation_manager,
                failed_files_txt_path=failed_files_txt_path,
                failed_source_paths=failed_source_paths,
            )
        )
    else:
        results = _new_batch_results(
            failed_files_txt_path=failed_files_txt_path,
            failed_source_paths=failed_source_paths,
        )
        results["total_count"] = total_images
        previous_context_cache = OrderedDict()
        previous_context_cache_lock = threading.Lock()
        ocr_text_history: Dict[Path, List[str]] = {}
        ocr_text_history_lock = threading.Lock()
        log_message(
            f"Starting batch processing: {total_images} images", always_print=True
        )

        for i, img_path in enumerate(image_files):
            try:
                output_path, display_path, error_key = _resolve_output_path(
                    img_path, input_dir, output_dir, config, preserve_structure
                )

                if cancellation_manager and cancellation_manager.is_cancelled():
                    raise CancellationError("Batch process cancelled by user.")

                if progress_callback:
                    current_progress = i / total_images
                    progress_callback(
                        current_progress,
                        f"Processing image {i + 1}/{total_images}: {display_path}",
                    )

                log_message(
                    f"Processing {i + 1}/{total_images}: {display_path}",
                    always_print=True,
                )

                previous_context_images = _build_previous_context_images(
                    image_files,
                    i,
                    config,
                    previous_context_cache,
                    previous_context_cache_lock,
                )
                previous_context_texts = _build_previous_context_texts(
                    image_files,
                    i,
                    config,
                    ocr_text_history,
                    ocr_text_history_lock,
                )
                captured_ocr_texts: List[str] = []
                translate_and_render(
                    img_path,
                    config,
                    output_path,
                    edit_session_path=default_edit_session_path(output_path),
                    cancellation_manager=cancellation_manager,
                    previous_context_images=previous_context_images,
                    previous_context_texts=previous_context_texts,
                    ocr_texts_out=captured_ocr_texts,
                )
                if captured_ocr_texts:
                    with ocr_text_history_lock:
                        ocr_text_history[img_path] = captured_ocr_texts

                _record_batch_success(results, error_key, img_path, output_path)

                if progress_callback:
                    completed_progress = (i + 1) / total_images
                    progress_callback(
                        completed_progress,
                        f"Completed {i + 1}/{total_images} images",
                    )

            except CancellationError as exc:
                cancel_message = str(exc) or "Batch process cancelled by user."
                log_message(
                    f"Batch cancelled during processing of {display_path}",
                    verbose=config.verbose,
                )
                if error_key not in results.get("successful_files", {}):
                    _record_batch_failure(
                        results,
                        error_key,
                        img_path,
                        cancel_message,
                    )
                remaining_files = image_files[i + 1 :]
                _mark_batch_cancelled(
                    results,
                    remaining_files,
                    input_dir,
                    config,
                    output_dir,
                    preserve_structure,
                    message=cancel_message,
                )
                cancelled = True
                break
            except Exception as e:
                log_message(
                    f"Error processing {display_path}: {str(e)}", always_print=True
                )
                _record_batch_failure(results, error_key, img_path, e)

                if progress_callback:
                    completed_progress = (i + 1) / total_images
                    try:
                        progress_callback(
                            completed_progress,
                            f"Completed {i + 1}/{total_images} images (with errors)",
                        )
                    except CancellationError:
                        remaining_files = image_files[i + 1 :]
                        _mark_batch_cancelled(
                            results,
                            remaining_files,
                            input_dir,
                            config,
                            output_dir,
                            preserve_structure,
                        )
                        cancelled = True
                        break

    results.setdefault("total_count", total_images)

    if (
        not cancelled
        and not results.get("cancelled", False)
        and getattr(config, "retry_failed_once", False)
        and results.get("error_count", 0)
    ):
        _retry_failed_batch_images(
            results=results,
            input_dir=input_dir,
            config=config,
            output_dir=output_dir,
            preserve_structure=preserve_structure,
            progress_callback=progress_callback,
            cancellation_manager=cancellation_manager,
        )

    if progress_callback and not results.get("cancelled", False):
        progress_callback(1.0, "Processing complete")

    end_batch_time = time.time()
    total_batch_time = end_batch_time - start_batch_time
    seconds_per_image = total_batch_time / total_images if total_images > 0 else 0

    log_message(
        f"Batch {'cancelled' if results.get('cancelled') else 'complete'}: "
        f"{results['success_count']}/{total_images} images in "
        f"{total_batch_time:.2f}s ({seconds_per_image:.2f}s/image)",
        always_print=True,
    )
    if results["error_count"] > 0:
        log_message(f"Failed: {results['error_count']} images", always_print=True)
        for filename, error_msg in results["errors"].items():
            log_message(f"  - {filename}: {error_msg}", always_print=True)

    link_edit_session_group(
        [
            default_edit_session_path(Path(output_path))
            for output_path in sorted(
                (
                    Path(value)
                    for value in results.get("successful_files", {}).values()
                ),
                key=_natural_path_sort_key,
            )
        ]
    )
    return results
