from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

from core.scaling import scale_area, scale_kernel, scale_scalar
from utils.exceptions import CleaningError, ImageProcessingError, ValidationError
from utils.logging import log_message

from .detection import detect_speech_bubbles
from .image_utils import pil_to_cv2
from .lama_inpainting import (
    build_lama_contrast_outline_seed,
    build_detector_text_hole_seed_mask,
    build_lama_inpaint_mask,
    build_lama_text_seed_mask,
    run_lama_large,
)

# Cleaning parameters
GRAYSCALE_MIDPOINT = 128  # Threshold for determining black vs white bubbles
MIN_CONTOUR_AREA = 50  # Minimum area threshold for filtering small contours
DILATION_KERNEL_SIZE = (7, 7)  # Kernel size for morphological dilation
EROSION_KERNEL_SIZE = (5, 5)  # Kernel size for morphological erosion
DISTANCE_TRANSFORM_MASK_SIZE = 5  # Mask size for distance transform

# Classification thresholds for colored bubbles
BRIGHT_RATIO_THRESHOLD = 0.65
DARK_RATIO_THRESHOLD = 0.65
BRIGHT_DOM_RATIO_MIN = 0.40
DARK_DOM_RATIO_MIN = 0.40
BRIGHT_DARK_RATIO_MAX = 0.10
DARK_BRIGHT_RATIO_MAX = 0.10

# Conservative fallback for small gray/colored bubbles whose bright text outline
# can otherwise be mistaken for a white bubble after text masking.
MIDTONE_MIN_VALUE = 32
MIDTONE_MAX_VALUE = 235
MIDTONE_MIN_RATIO = 0.50
MIDTONE_MAX_BRIGHT_RATIO = 0.35
MIDTONE_MAX_DARK_RATIO = 0.20
MIDTONE_MIN_MODE_RATIO = 0.04
MIDTONE_MIN_MEDIAN = 48
MIDTONE_MAX_MEDIAN = 230

# A narrow ring just inside the detected outline is less affected by large
# white text strokes or decorative white centers than the full bubble interior.
MIDTONE_RING_INNER_RATIO = 0.08
MIDTONE_RING_OUTER_RATIO = 0.25
MIDTONE_RING_MIN_PIXELS = 128
MIDTONE_RING_MIN_RATIO = 0.85
MIDTONE_RING_MAX_BRIGHT_RATIO = 0.12
MIDTONE_RING_MAX_DARK_RATIO = 0.05
MIDTONE_RING_MIN_MEDIAN = 48
MIDTONE_RING_MAX_MEDIAN = 230
SOLID_MIDTONE_RESIDUAL_DIFF = 24
FLUX_CONDITIONING_DARK_DIFF = 35
FLUX_CONDITIONING_MIN_COMPONENT_AREA = 2
FLUX_CONDITIONING_MAX_COMPONENT_RATIO = 0.25

# Conservative translucent-bubble rescue. This runs only after the normal
# classifier selected white and inspects deep interior pixels outside trusted
# text boxes, so ordinary solid-white bubbles keep the existing fast fill.
TRANSLUCENT_DEEP_INTERIOR_RATIO = 0.12
TRANSLUCENT_DEEP_INTERIOR_MIN_PX = 5.0
TRANSLUCENT_TEXT_BOX_PADDING = 4
TRANSLUCENT_TEXT_STROKE_PADDING = 3
TRANSLUCENT_MIN_SAMPLE_PIXELS = 256
TRANSLUCENT_MIN_SAMPLE_RATIO = 0.10
TRANSLUCENT_MIDTONE_MIN = 32
TRANSLUCENT_MIDTONE_MAX = 235
TRANSLUCENT_BROAD_MIDTONE_RATIO = 0.18
TRANSLUCENT_BROAD_STD_MIN = 12.0
TRANSLUCENT_BROAD_EDGE_RATIO_MIN = 0.025
TRANSLUCENT_SPARSE_MIDTONE_RATIO = 0.075
TRANSLUCENT_SPARSE_STD_MIN = 14.0
TRANSLUCENT_SPARSE_EDGE_RATIO_MIN = 0.075
TRANSLUCENT_MAX_BRIGHT_RATIO = 0.93
TRANSLUCENT_BRIGHT_PALETTE_MIN_RATIO = 0.78
TRANSLUCENT_BRIGHT_PALETTE_MIN_MEDIAN = 245.0
TRANSLUCENT_BRIGHT_PALETTE_MAX_SAMPLE = 96
CONJOINED_WHITE_CORE_MIN_OVERFLOW_RATIO = 0.12
CONJOINED_WHITE_CORE_TEXT_PADDING = 8
CONJOINED_WHITE_CORE_MIN_PIXELS = 512
CONJOINED_WHITE_CORE_MIN_MEDIAN = 250.0
CONJOINED_WHITE_CORE_MIN_BRIGHT_RATIO = 0.94
CONJOINED_WHITE_CORE_MAX_MIDTONE_RATIO = 0.04
CONJOINED_WHITE_CORE_MAX_DARK_RATIO = 0.04
CONJOINED_WHITE_INTERIOR_MIN_PIXELS = 512
CONJOINED_WHITE_INTERIOR_MIN_MEDIAN = 245.0
CONJOINED_WHITE_INTERIOR_MIN_BRIGHT_RATIO = 0.65
CONJOINED_WHITE_INTERIOR_MAX_MIDTONE_RATIO = 0.28
CONJOINED_WHITE_INTERIOR_MAX_DARK_RATIO = 0.12
TRANSLUCENT_LAMA_MASK_LIMIT_TRIGGER_RATIO = 0.68
TRANSLUCENT_LAMA_MASK_MAX_RATIO = 0.56
TRANSLUCENT_LAMA_MASK_MAX_SEED_RATIO = 0.42
ACHROMATIC_TEXT_CONTRAST_MARGIN = 12.0
CONJOINED_CHILD_LAMA_MAX_MEAN = 200.0
CONJOINED_PARENT_WHITE_MIN_PIXELS = 256
CONJOINED_PARENT_WHITE_MIN_MEDIAN = 245.0
CONJOINED_PARENT_WHITE_MIN_BRIGHT_RATIO = 0.75
CONJOINED_PARENT_WHITE_MAX_DARK_RATIO = 0.08
CONJOINED_PARENT_FULL_SEED_MAX_RATIO = 0.20
CONJOINED_PARENT_FULL_SEED_MIN_MISSING_PIXELS = 16
BRIGHT_TRANSLUCENT_COMPLETE_MIN_BRIGHT_RATIO = 0.78
BRIGHT_TRANSLUCENT_COMPLETE_MIN_MEDIAN = 245.0
BRIGHT_TRANSLUCENT_COMPLETE_MAX_FULL_SEED_RATIO = 0.20
BRIGHT_TRANSLUCENT_COMPLETE_MAX_GUIDED_COVERAGE = 0.72
BRIGHT_TRANSLUCENT_COMPLETE_MIN_MISSING_PIXELS = 64
DARK_TEXT_ONLY_MAX_MASK_RATIO = 0.55
DARK_TEXT_ONLY_MIN_SEED_PIXELS = 16
DARK_TEXT_ONLY_BOX_PADDING = 8
DARK_TEXT_ONLY_HALO_MIN_VALUE = 32
DARK_TEXT_ONLY_HALO_GROW_ITERATIONS = 7
DARK_TEXT_ONLY_SOLID_BLACK_MAX_MEDIAN = 8
DARK_TEXT_ONLY_SOLID_BLACK_DARK_LIMIT = 32
DARK_TEXT_ONLY_SOLID_BLACK_MIN_DARK_RATIO = 0.75
DARK_TEXT_ONLY_SOLID_BLACK_MIN_BRIGHT_RATIO = 0.02

DARK_BUBBLE_RESIDUAL_BRIGHT_CORE = 145
DARK_BUBBLE_RESIDUAL_BRIGHT_HALO = 75
DARK_BUBBLE_RESIDUAL_RING_DARK_MAX = 85
DARK_BUBBLE_RESIDUAL_MIN_RING_DARK_RATIO = 0.35
DARK_BUBBLE_RESIDUAL_MAX_SIDE_PX = 72
DARK_BUBBLE_RESIDUAL_MAX_AREA = 1800.0
DARK_BUBBLE_RESIDUAL_MIN_BORDER_DISTANCE = 2.0


# Adaptive shrink defaults for conjoined junction zones (at 1.0x scale)
JUNCTION_ADJACENCY_MARGIN = 10  # Px margin around bbox intersection for junction zone
JUNCTION_MIN_SHRINK = 1.0  # Minimal shrink applied inside junction zones

def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is uint8 binary (0/255).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _is_confident_white_conjoined_parent(
    image_gray: np.ndarray, parent_mask: Optional[np.ndarray]
) -> bool:
    """Recognize a genuinely white parent despite dark glyphs in a child crop."""
    if parent_mask is None or image_gray is None:
        return False

    try:
        normalized = _normalize_mask(np.asarray(parent_mask))
    except (TypeError, ValueError):
        return False
    if normalized.shape != image_gray.shape:
        return False

    interior = cv2.erode(normalized, np.ones((3, 3), np.uint8), iterations=1)
    pixels = image_gray[interior == 255]
    if pixels.size < CONJOINED_PARENT_WHITE_MIN_PIXELS:
        pixels = image_gray[normalized == 255]
    if pixels.size < CONJOINED_PARENT_WHITE_MIN_PIXELS:
        return False

    median = float(np.median(pixels))
    bright_ratio = float(np.mean(pixels >= 245))
    dark_ratio = float(np.mean(pixels <= 15))
    return bool(
        median >= CONJOINED_PARENT_WHITE_MIN_MEDIAN
        and bright_ratio >= CONJOINED_PARENT_WHITE_MIN_BRIGHT_RATIO
        and dark_ratio <= CONJOINED_PARENT_WHITE_MAX_DARK_RATIO
    )


def _limit_overbroad_translucent_lama_mask(
    lama_mask: np.ndarray,
    text_seed: np.ndarray,
    bubble_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Keep LAMA context without replacing most of a translucent bubble."""
    expanded = _normalize_mask(lama_mask)
    seed = _normalize_mask(text_seed)
    bubble = _normalize_mask(bubble_mask)
    safe_bubble = cv2.erode(
        bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if not np.any(safe_bubble):
        safe_bubble = bubble

    safe_pixels = safe_bubble > 0
    bubble_area = int(np.count_nonzero(safe_pixels))
    if bubble_area <= 0:
        return expanded, {}

    expanded = cv2.bitwise_and(expanded, safe_bubble)
    seed = cv2.bitwise_and(seed, expanded)
    expanded_count = int(np.count_nonzero(expanded))
    seed_count = int(np.count_nonzero(seed))
    expanded_ratio = expanded_count / float(bubble_area)
    seed_ratio = seed_count / float(bubble_area)
    stats = {
        "before_ratio": expanded_ratio,
        "after_ratio": expanded_ratio,
        "seed_ratio": seed_ratio,
    }
    if (
        expanded_ratio <= TRANSLUCENT_LAMA_MASK_LIMIT_TRIGGER_RATIO
        or seed_ratio >= TRANSLUCENT_LAMA_MASK_MAX_SEED_RATIO
    ):
        return expanded, stats

    target_count = max(
        seed_count,
        int(round(bubble_area * TRANSLUCENT_LAMA_MASK_MAX_RATIO)),
    )
    if target_count >= expanded_count:
        return expanded, stats

    distance = cv2.distanceTransform(
        np.where(seed > 0, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    expansion_domain = (expanded > 0) & (seed == 0)
    expansion_distances = distance[expansion_domain]
    keep_expansion_count = target_count - seed_count
    if keep_expansion_count <= 0 or expansion_distances.size == 0:
        limited = seed
    elif keep_expansion_count >= expansion_distances.size:
        limited = expanded
    else:
        threshold_index = max(0, keep_expansion_count - 1)
        threshold = float(
            np.partition(expansion_distances, threshold_index)[threshold_index]
        )
        limited = np.where(
            (seed > 0)
            | (expansion_domain & (distance <= threshold)),
            255,
            0,
        ).astype(np.uint8)

    limited = cv2.bitwise_and(limited, expanded)
    stats["after_ratio"] = int(np.count_nonzero(limited)) / float(bubble_area)
    return limited, stats


def _resolve_achromatic_text_color(
    sampled_bgr: tuple[int, ...],
    background_bgr: tuple[int, ...],
    *,
    is_black_bubble: bool,
) -> tuple[int, int, int]:
    """Resolve grayscale text polarity relative to its bubble background."""
    text_value = float(np.mean(sampled_bgr[:3]))
    background_value = float(np.mean(background_bgr[:3]))
    if text_value <= background_value - ACHROMATIC_TEXT_CONTRAST_MARGIN:
        return (0, 0, 0)
    if text_value >= background_value + ACHROMATIC_TEXT_CONTRAST_MARGIN:
        return (255, 255, 255)
    return (255, 255, 255) if is_black_bubble else (0, 0, 0)


def _detect_confident_midtone_background(
    pixel_values: np.ndarray,
) -> tuple[bool, int, dict[str, float]]:
    """
    Detect a clearly midtone bubble interior without weakening white detection.

    This is only used as a fallback after the existing classifier has selected
    white. It requires most of the original interior to be midtone and limits
    both pure-white and pure-black contamination from text and outlines.
    """
    values = np.asarray(pixel_values, dtype=np.uint8).reshape(-1)
    if values.size < 64:
        return False, 255, {}

    total = float(values.size)
    bright_ratio = float(np.count_nonzero(values >= 245)) / total
    dark_ratio = float(np.count_nonzero(values <= 15)) / total
    midtone_values = values[
        (values >= MIDTONE_MIN_VALUE) & (values <= MIDTONE_MAX_VALUE)
    ]
    midtone_ratio = float(midtone_values.size) / total
    median_value = float(np.median(values))

    if midtone_values.size:
        midtone_hist = np.bincount(midtone_values, minlength=256)
        midtone_mode = int(midtone_hist.argmax())
        midtone_mode_ratio = float(midtone_hist[midtone_mode]) / total
    else:
        midtone_mode = int(round(median_value))
        midtone_mode_ratio = 0.0

    stats = {
        "midtone_ratio": midtone_ratio,
        "bright_ratio": bright_ratio,
        "dark_ratio": dark_ratio,
        "median": median_value,
        "mode_ratio": midtone_mode_ratio,
    }
    is_midtone = (
        midtone_ratio >= MIDTONE_MIN_RATIO
        and bright_ratio <= MIDTONE_MAX_BRIGHT_RATIO
        and dark_ratio <= MIDTONE_MAX_DARK_RATIO
        and MIDTONE_MIN_MEDIAN <= median_value <= MIDTONE_MAX_MEDIAN
        and midtone_mode_ratio >= MIDTONE_MIN_MODE_RATIO
    )
    return is_midtone, midtone_mode, stats


def _detect_confident_midtone_inner_ring(
    image_gray: np.ndarray,
    bubble_mask: np.ndarray,
) -> tuple[bool, int, dict[str, float]]:
    """
    Detect a gray/colored inner border while preserving true white bubbles.

    The test is deliberately strict and only accepts a ring that is almost
    entirely midtone. This catches gray SFX bubbles with a large white center,
    without promoting ordinary white or mixed conjoined bubbles to Flux.
    """
    mask = _normalize_mask(bubble_mask)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE)
    max_distance = float(distance.max())
    if max_distance < 4.0:
        return False, 255, {}

    ring_mask = (
        (distance >= max_distance * MIDTONE_RING_INNER_RATIO)
        & (distance < max_distance * MIDTONE_RING_OUTER_RATIO)
    )
    values = np.asarray(image_gray[ring_mask], dtype=np.uint8).reshape(-1)
    if values.size < MIDTONE_RING_MIN_PIXELS:
        return False, 255, {}

    total = float(values.size)
    bright_ratio = float(np.count_nonzero(values >= 245)) / total
    dark_ratio = float(np.count_nonzero(values <= 15)) / total
    midtone_values = values[
        (values >= MIDTONE_MIN_VALUE) & (values <= MIDTONE_MAX_VALUE)
    ]
    midtone_ratio = float(midtone_values.size) / total
    median_value = float(np.median(values))
    sample_value = int(round(np.median(midtone_values))) if midtone_values.size else 255

    stats = {
        "midtone_ratio": midtone_ratio,
        "bright_ratio": bright_ratio,
        "dark_ratio": dark_ratio,
        "median": median_value,
        "pixels": float(values.size),
    }
    is_midtone = (
        midtone_ratio >= MIDTONE_RING_MIN_RATIO
        and bright_ratio <= MIDTONE_RING_MAX_BRIGHT_RATIO
        and dark_ratio <= MIDTONE_RING_MAX_DARK_RATIO
        and MIDTONE_RING_MIN_MEDIAN <= median_value <= MIDTONE_RING_MAX_MEDIAN
    )
    return is_midtone, sample_value, stats


def _detect_confident_translucent_background(
    image_gray: np.ndarray,
    bubble_mask: np.ndarray,
    text_boxes: Optional[list],
) -> tuple[bool, int, dict[str, float]]:
    """Detect artwork or tone showing through an otherwise bright bubble."""
    gray = np.asarray(image_gray, dtype=np.uint8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    mask = _normalize_mask(bubble_mask)
    if gray.shape != mask.shape:
        return False, 255, {}

    distance = cv2.distanceTransform(
        mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE
    )
    max_distance = float(distance.max())
    if max_distance < TRANSLUCENT_DEEP_INTERIOR_MIN_PX:
        return False, 255, {}

    deep_threshold = max(
        TRANSLUCENT_DEEP_INTERIOR_MIN_PX,
        max_distance * TRANSLUCENT_DEEP_INTERIOR_RATIO,
    )
    deep_interior = distance >= deep_threshold
    deep_count = int(np.count_nonzero(deep_interior))
    if deep_count < TRANSLUCENT_MIN_SAMPLE_PIXELS:
        return False, 255, {}

    excluded = np.zeros_like(mask)
    height, width = gray.shape[:2]
    for box in text_boxes or []:
        try:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        except (TypeError, ValueError):
            continue
        pad = TRANSLUCENT_TEXT_BOX_PADDING
        x0 = max(0, min(width, x0 - pad))
        y0 = max(0, min(height, y0 - pad))
        x1 = max(x0, min(width, x1 + pad))
        y1 = max(y0, min(height, y1 + pad))
        excluded[y0:y1, x0:x1] = 255

    sample_mask = deep_interior & (excluded == 0)
    values = gray[sample_mask]
    sample_ratio = float(values.size) / float(max(deep_count, 1))
    used_stroke_fallback = False
    if (
        values.size < TRANSLUCENT_MIN_SAMPLE_PIXELS
        or sample_ratio < TRANSLUCENT_MIN_SAMPLE_RATIO
    ):
        stroke_mask = np.zeros_like(mask)
        for text_is_dark in (True, False):
            stroke_mask = cv2.bitwise_or(
                stroke_mask,
                build_lama_text_seed_mask(
                    gray,
                    mask,
                    text_boxes,
                    text_is_dark=text_is_dark,
                    protect_background_artwork=True,
                ),
            )
        if np.any(stroke_mask):
            pad = TRANSLUCENT_TEXT_STROKE_PADDING
            stroke_mask = cv2.dilate(
                stroke_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (pad * 2 + 1, pad * 2 + 1),
                ),
                iterations=1,
            )
            sample_mask = deep_interior & (stroke_mask == 0)
            values = gray[sample_mask]
            sample_ratio = float(values.size) / float(max(deep_count, 1))
            used_stroke_fallback = True

        if (
            values.size < TRANSLUCENT_MIN_SAMPLE_PIXELS
            or sample_ratio < TRANSLUCENT_MIN_SAMPLE_RATIO
        ):
            return False, 255, {}

    total = float(values.size)
    bright_ratio = float(np.count_nonzero(values >= 245)) / total
    midtone_values = values[
        (values >= TRANSLUCENT_MIDTONE_MIN)
        & (values <= TRANSLUCENT_MIDTONE_MAX)
    ]
    midtone_ratio = float(midtone_values.size) / total
    stddev = float(np.std(values))

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    edge_ratio = float(np.count_nonzero(gradient[sample_mask] >= 40.0)) / total

    broad_detail = (
        midtone_ratio >= TRANSLUCENT_BROAD_MIDTONE_RATIO
        and stddev >= TRANSLUCENT_BROAD_STD_MIN
        and edge_ratio >= TRANSLUCENT_BROAD_EDGE_RATIO_MIN
    )
    sparse_detail = (
        midtone_ratio >= TRANSLUCENT_SPARSE_MIDTONE_RATIO
        and stddev >= TRANSLUCENT_SPARSE_STD_MIN
        and edge_ratio >= TRANSLUCENT_SPARSE_EDGE_RATIO_MIN
    )
    detected = (
        bright_ratio <= TRANSLUCENT_MAX_BRIGHT_RATIO
        and (broad_detail or sparse_detail)
    )

    sample_value = (
        int(round(float(np.median(midtone_values))))
        if midtone_values.size
        else int(round(float(np.median(values))))
    )
    stats = {
        "sample_ratio": sample_ratio,
        "midtone_ratio": midtone_ratio,
        "bright_ratio": bright_ratio,
        "stddev": stddev,
        "edge_ratio": edge_ratio,
        "median": float(np.median(values)),
        "stroke_fallback": float(used_stroke_fallback),
    }
    return detected, sample_value, stats


def _detect_conjoined_white_text_core(
    image_gray: np.ndarray,
    bubble_mask: np.ndarray,
    text_boxes: Optional[list],
    detection_bbox,
    neighbor_bboxes: Optional[list],
) -> tuple[bool, dict[str, float]]:
    """Reject LAMA when conjoined-mask overflow contaminates a white text core."""
    if not text_boxes or not neighbor_bboxes or detection_bbox is None:
        return False, {}

    gray = np.asarray(image_gray, dtype=np.uint8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    mask = _normalize_mask(bubble_mask)
    if gray.shape != mask.shape or not np.any(mask):
        return False, {}

    height, width = gray.shape[:2]
    try:
        bx0, by0, bx1, by1 = [
            int(round(float(value))) for value in detection_bbox[:4]
        ]
    except (TypeError, ValueError):
        return False, {}
    bx0 = max(0, min(width, bx0))
    by0 = max(0, min(height, by0))
    bx1 = max(bx0, min(width, bx1))
    by1 = max(by0, min(height, by1))
    if bx1 <= bx0 or by1 <= by0:
        return False, {}

    declared_box_mask = np.zeros_like(mask)
    declared_box_mask[by0:by1, bx0:bx1] = 255
    mask_pixels = max(1, int(np.count_nonzero(mask)))
    overflow_pixels = int(
        np.count_nonzero(
            cv2.bitwise_and(mask, cv2.bitwise_not(declared_box_mask))
        )
    )
    overflow_ratio = overflow_pixels / float(mask_pixels)
    if overflow_ratio < CONJOINED_WHITE_CORE_MIN_OVERFLOW_RATIO:
        return False, {"overflow_ratio": overflow_ratio}

    safe_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    if not np.any(safe_mask):
        safe_mask = mask

    local_mask = np.zeros_like(mask)
    for box in text_boxes:
        try:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        except (TypeError, ValueError):
            continue
        pad = CONJOINED_WHITE_CORE_TEXT_PADDING
        x0 = max(bx0, min(bx1, x0 - pad))
        y0 = max(by0, min(by1, y0 - pad))
        x1 = max(x0, min(bx1, x1 + pad))
        y1 = max(y0, min(by1, y1 + pad))
        local_mask[y0:y1, x0:x1] = 255
    local_mask = cv2.bitwise_and(local_mask, safe_mask)
    if not np.any(local_mask):
        return False, {"overflow_ratio": overflow_ratio}

    text_seed = build_lama_text_seed_mask(
        gray,
        mask,
        text_boxes,
        text_is_dark=True,
    )
    if np.any(text_seed):
        text_seed = cv2.dilate(
            text_seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        local_mask[text_seed > 0] = 0

    values = gray[local_mask > 0]
    if values.size < CONJOINED_WHITE_CORE_MIN_PIXELS:
        return False, {
            "overflow_ratio": overflow_ratio,
            "pixels": float(values.size),
        }

    total = float(values.size)
    median = float(np.median(values))
    bright_ratio = float(np.count_nonzero(values >= 245)) / total
    midtone_ratio = float(
        np.count_nonzero(
            (values >= TRANSLUCENT_MIDTONE_MIN)
            & (values <= TRANSLUCENT_MIDTONE_MAX)
        )
    ) / total
    dark_ratio = float(np.count_nonzero(values <= 15)) / total
    stats = {
        "overflow_ratio": overflow_ratio,
        "pixels": total,
        "median": median,
        "bright_ratio": bright_ratio,
        "midtone_ratio": midtone_ratio,
        "dark_ratio": dark_ratio,
    }
    detected = (
        median >= CONJOINED_WHITE_CORE_MIN_MEDIAN
        and bright_ratio >= CONJOINED_WHITE_CORE_MIN_BRIGHT_RATIO
        and midtone_ratio <= CONJOINED_WHITE_CORE_MAX_MIDTONE_RATIO
        and dark_ratio <= CONJOINED_WHITE_CORE_MAX_DARK_RATIO
    )
    return detected, stats


def _detect_conjoined_white_interior(
    image_gray: np.ndarray,
    bubble_mask: np.ndarray,
    detection_bbox,
    neighbor_bboxes: Optional[list],
) -> tuple[bool, dict[str, float]]:
    """Keep bright conjoined children out of translucent/LAMA cleanup.

    Dense vertical glyphs and overlap seams can make the translucent detector's
    edge/midtone sample look busy even when the actual bubble fill is white.
    This check looks at the current child bubble's own interior instead of the
    translucent detail sample.
    """
    if not neighbor_bboxes or detection_bbox is None:
        return False, {}

    gray = np.asarray(image_gray, dtype=np.uint8)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    mask = _normalize_mask(bubble_mask)
    if gray.shape != mask.shape or not np.any(mask):
        return False, {}

    height, width = gray.shape[:2]
    try:
        bx0, by0, bx1, by1 = [
            int(round(float(value))) for value in detection_bbox[:4]
        ]
    except (TypeError, ValueError):
        return False, {}
    bx0 = max(0, min(width, bx0))
    by0 = max(0, min(height, by0))
    bx1 = max(bx0, min(width, bx1))
    by1 = max(by0, min(height, by1))
    if bx1 <= bx0 or by1 <= by0:
        return False, {}

    box_mask = np.zeros_like(mask)
    box_mask[by0:by1, bx0:bx1] = 255
    child_mask = cv2.bitwise_and(mask, box_mask)
    if not np.any(child_mask):
        return False, {}

    safe_mask = cv2.erode(
        child_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if np.count_nonzero(safe_mask) < CONJOINED_WHITE_INTERIOR_MIN_PIXELS:
        safe_mask = child_mask

    values = gray[safe_mask > 0]
    if values.size < CONJOINED_WHITE_INTERIOR_MIN_PIXELS:
        return False, {"pixels": float(values.size)}

    total = float(values.size)
    median = float(np.median(values))
    bright_ratio = float(np.count_nonzero(values >= 245)) / total
    midtone_ratio = float(
        np.count_nonzero(
            (values >= TRANSLUCENT_MIDTONE_MIN)
            & (values <= TRANSLUCENT_MIDTONE_MAX)
        )
    ) / total
    dark_ratio = float(np.count_nonzero(values <= 15)) / total
    stats = {
        "pixels": total,
        "median": median,
        "bright_ratio": bright_ratio,
        "midtone_ratio": midtone_ratio,
        "dark_ratio": dark_ratio,
    }
    detected = (
        median >= CONJOINED_WHITE_INTERIOR_MIN_MEDIAN
        and bright_ratio >= CONJOINED_WHITE_INTERIOR_MIN_BRIGHT_RATIO
        and midtone_ratio <= CONJOINED_WHITE_INTERIOR_MAX_MIDTONE_RATIO
        and dark_ratio <= CONJOINED_WHITE_INTERIOR_MAX_DARK_RATIO
    )
    return detected, stats


def _resolve_translucent_render_sample(
    sample_value: int,
    stats: dict[str, float],
) -> int:
    """Keep text polarity stable when sparse dark strokes contaminate white."""
    if (
        float(stats.get("bright_ratio", 0.0))
        >= TRANSLUCENT_BRIGHT_PALETTE_MIN_RATIO
        and float(stats.get("median", 0.0))
        >= TRANSLUCENT_BRIGHT_PALETTE_MIN_MEDIAN
        and int(sample_value) <= TRANSLUCENT_BRIGHT_PALETTE_MAX_SAMPLE
    ):
        return 255
    return int(sample_value)


def _complete_solid_midtone_text_mask(
    clean_mask: np.ndarray,
    bubble_mask: np.ndarray,
    image_gray: np.ndarray,
    target_value: int,
    constraint_erosion_kernel: np.ndarray,
) -> np.ndarray:
    """Complete adjacent text strokes while protecting a solid bubble's outline."""
    normalized_clean = _normalize_mask(clean_mask)
    safe_bubble = cv2.erode(
        _normalize_mask(bubble_mask),
        constraint_erosion_kernel,
        iterations=1,
    )
    constrained = cv2.bitwise_and(normalized_clean, safe_bubble)
    contours, _ = cv2.findContours(
        constrained,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    completed = np.zeros_like(constrained)
    if contours:
        cv2.drawContours(completed, contours, -1, 255, thickness=cv2.FILLED)
        text_seed = cv2.dilate(
            cv2.bitwise_or(constrained, completed),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        )
        residual = np.where(
            np.abs(image_gray.astype(np.int16) - int(target_value))
            >= SOLID_MIDTONE_RESIDUAL_DIFF,
            255,
            0,
        ).astype(np.uint8)
        adjacent_residual = cv2.bitwise_and(residual, text_seed)
        completed = cv2.bitwise_or(completed, adjacent_residual)
        completed = cv2.dilate(
            completed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        completed = cv2.bitwise_and(completed, safe_bubble)
    return cv2.bitwise_or(normalized_clean, completed)


def _build_flux_conditioning_text_mask(
    image_gray: np.ndarray,
    bubble_mask: np.ndarray,
    clean_mask: np.ndarray,
    sample_value: int,
    use_solid_midtone_fill: bool,
) -> np.ndarray:
    """Build a text-only conditioning mask so Flux cannot copy source glyphs."""
    normalized_clean = _normalize_mask(clean_mask)
    normalized_bubble = _normalize_mask(bubble_mask)
    if use_solid_midtone_fill:
        return normalized_clean

    safe_bubble = cv2.erode(
        normalized_bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    deeper_bubble = cv2.erode(
        safe_bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    boundary_band = cv2.subtract(safe_bubble, deeper_bubble)
    dark_limit = max(15, int(sample_value) - FLUX_CONDITIONING_DARK_DIFF)
    dark_candidates = np.where(
        (image_gray <= dark_limit)
        & (safe_bubble == 255)
        & (normalized_clean == 255),
        255,
        0,
    ).astype(np.uint8)

    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_candidates,
        connectivity=8,
    )
    kept = np.zeros_like(dark_candidates)
    bubble_area = max(1, int(np.count_nonzero(normalized_bubble)))
    max_component_area = max(
        64,
        int(bubble_area * FLUX_CONDITIONING_MAX_COMPONENT_RATIO),
    )
    for label_id in range(1, label_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < FLUX_CONDITIONING_MIN_COMPONENT_AREA:
            continue
        if area > max_component_area:
            continue
        component = labels == label_id
        if np.any(boundary_band[component] > 0):
            continue
        kept[component] = 255

    if not np.any(kept):
        return kept

    kept = cv2.dilate(
        kept,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    kept = cv2.bitwise_and(kept, safe_bubble)
    return cv2.bitwise_and(kept, normalized_clean)


def _prepare_flux_conditioning_image(
    image_pil: Image.Image,
    conditioning_mask: np.ndarray,
    fill_color_bgr: tuple[int, int, int],
    use_solid_midtone_fill: bool,
) -> Image.Image:
    """Remove source text from the image supplied to an image-to-image Flux model."""
    mask = _normalize_mask(conditioning_mask)
    if not np.any(mask):
        return image_pil

    image_rgb = np.asarray(image_pil.convert("RGB")).copy()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if use_solid_midtone_fill:
        image_bgr[mask == 255] = tuple(int(c) for c in fill_color_bgr[:3])
    else:
        image_bgr = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)

    conditioned = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return conditioned.convert(image_pil.mode) if image_pil.mode != "RGB" else conditioned


def _merge_conjoined_parent_clean_mask(
    clean_mask: Optional[np.ndarray],
    parent_mask: Optional[np.ndarray],
    fallback_mask: Optional[np.ndarray],
    child_mask: Optional[np.ndarray],
    img_gray: np.ndarray,
    fill_color_bgr,
    is_colored_bubble: bool,
    thresholding_value: int,
    constraint_erosion_kernel: np.ndarray,
    min_contour_area: float,
    detection_bbox,
    verbose: bool,
) -> Optional[np.ndarray]:
    """
    Use the original conjoined speech-bubble mask as a search area only.

    Rendering still uses the split child mask. The parent mask must not be filled
    directly, otherwise bubble outlines and nearby panel art can be painted over.
    """
    if parent_mask is None or is_colored_bubble:
        return clean_mask

    base_mask = clean_mask if clean_mask is not None else fallback_mask
    if base_mask is None:
        return clean_mask

    try:
        parent_clean_mask = _normalize_mask(parent_mask)
    except Exception:
        return clean_mask

    if parent_clean_mask.shape != base_mask.shape or not np.any(parent_clean_mask):
        return clean_mask

    safe_parent = cv2.erode(parent_clean_mask, constraint_erosion_kernel, iterations=1)
    if not np.any(safe_parent):
        return clean_mask

    seed_mask = child_mask if child_mask is not None else fallback_mask
    try:
        seed_mask = _normalize_mask(seed_mask)
    except Exception:
        seed_mask = fallback_mask

    if seed_mask is None or seed_mask.shape != safe_parent.shape:
        seed_mask = base_mask

    seed_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    seed_mask = cv2.dilate(seed_mask, seed_kernel, iterations=1)

    if fill_color_bgr == (0, 0, 0):
        bg_core = np.where((img_gray <= 25) & (safe_parent == 255), 255, 0).astype(
            np.uint8
        )
        bg_growth = np.where((img_gray <= 55) & (safe_parent == 255), 255, 0).astype(
            np.uint8
        )
    else:
        bg_core = np.where((img_gray >= 245) & (safe_parent == 255), 255, 0).astype(
            np.uint8
        )
        bg_growth = np.where((img_gray >= 230) & (safe_parent == 255), 255, 0).astype(
            np.uint8
        )

    seed_hits = cv2.bitwise_and(seed_mask, bg_growth)
    if not np.any(seed_hits):
        seed_hits = cv2.bitwise_and(seed_mask, bg_core)

    num_labels, labels = cv2.connectedComponents(bg_growth)
    allowed_parent = np.zeros_like(safe_parent)
    if num_labels > 1 and np.any(seed_hits):
        hit_labels = np.unique(labels[seed_hits > 0])
        for label_id in hit_labels:
            if label_id == 0:
                continue
            allowed_parent[labels == label_id] = 255

    if not np.any(allowed_parent):
        return clean_mask

    allow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    allowed_parent = cv2.dilate(allowed_parent, allow_kernel, iterations=2)
    allowed_parent = cv2.bitwise_and(allowed_parent, safe_parent)

    if fill_color_bgr == (0, 0, 0):
        core_limit = max(55, min(210, 255 - int(thresholding_value) + 25))
        antialias_limit = max(core_limit + 20, 180)
        text_core = np.where(
            (img_gray >= antialias_limit) & (allowed_parent == 255), 255, 0
        )
        text_halo_limit = max(core_limit, antialias_limit - 25)
        antialias_mask = np.where(img_gray >= text_halo_limit, 255, 0).astype(np.uint8)
    else:
        core_limit = min(200, max(90, int(thresholding_value) - 35))
        antialias_limit = min(245, max(core_limit + 35, int(thresholding_value) + 5))
        text_core = np.where(
            (img_gray <= core_limit) & (allowed_parent == 255), 255, 0
        )
        antialias_mask = np.where(img_gray <= antialias_limit, 255, 0).astype(np.uint8)

    text_core = text_core.astype(np.uint8)
    contours, _ = cv2.findContours(text_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_core = np.zeros_like(text_core)
    max_fragment_area = max(min_contour_area * 60.0, 3000.0)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(2.0, min_contour_area * 0.15):
            continue
        if area > max_fragment_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 96 and h < 5:
            continue
        if h > 96 and w < 5:
            continue

        component_mask = np.zeros_like(text_core)
        cv2.drawContours(component_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        ring_mask = cv2.dilate(component_mask, ring_kernel, iterations=2)
        ring_mask[component_mask == 255] = 0
        ring_mask = cv2.bitwise_and(ring_mask, allowed_parent)
        ring_values = img_gray[ring_mask == 255]
        if ring_values.size >= 4:
            if fill_color_bgr == (0, 0, 0):
                bg_like_ratio = float(np.count_nonzero(ring_values <= 70)) / float(
                    ring_values.size
                )
                bg_like_median = float(np.median(ring_values))
                if bg_like_ratio < 0.35 and bg_like_median > 95.0:
                    continue
            else:
                bg_like_ratio = float(np.count_nonzero(ring_values >= 235)) / float(
                    ring_values.size
                )
                bg_like_median = float(np.median(ring_values))
                if bg_like_ratio < 0.35 and bg_like_median < 225.0:
                    continue

        cv2.bitwise_or(filtered_core, component_mask, dst=filtered_core)

    if not np.any(filtered_core):
        return clean_mask

    halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    parent_text_mask = cv2.dilate(filtered_core, halo_kernel, iterations=2)
    parent_text_mask = cv2.bitwise_and(parent_text_mask, antialias_mask)
    parent_text_mask = cv2.bitwise_and(parent_text_mask, allowed_parent)

    merged = cv2.bitwise_or(base_mask, parent_text_mask)
    added_px = int(np.count_nonzero((parent_text_mask > 0) & (base_mask == 0)))
    if added_px:
        log_message(
            f"Detection {detection_bbox}: added conjoined parent text clean mask ({added_px} px)",
            verbose=verbose,
        )
    return merged


def _add_dark_bubble_bright_residual_mask(
    clean_mask: np.ndarray,
    base_mask: np.ndarray,
    img_gray: np.ndarray,
    constraint_erosion_kernel: np.ndarray,
    min_contour_area: float,
    detection_bbox,
    verbose: bool,
) -> np.ndarray:
    """Add missed white source-text flecks inside dark/gradient bubbles only."""
    if clean_mask is None or base_mask is None:
        return clean_mask

    base_mask = _normalize_mask(base_mask)
    clean_mask = _normalize_mask(clean_mask)
    if base_mask.shape != clean_mask.shape or not np.any(base_mask):
        return clean_mask

    # Use a lighter erosion than the outline-protection mask. Residual glyphs
    # often sit near spiky dark-bubble edges, but real outlines should stay out.
    small_safe_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    safe_mask = cv2.erode(base_mask, small_safe_kernel, iterations=1)
    if not np.any(safe_mask):
        return clean_mask

    dist_map = cv2.distanceTransform(base_mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE)
    bright_core = np.where(
        (img_gray >= DARK_BUBBLE_RESIDUAL_BRIGHT_CORE)
        & (safe_mask == 255)
        & (clean_mask == 0),
        255,
        0,
    ).astype(np.uint8)
    if not np.any(bright_core):
        return clean_mask

    contours, _ = cv2.findContours(
        bright_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return clean_mask

    filtered_core = np.zeros_like(clean_mask)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    max_area = max(DARK_BUBBLE_RESIDUAL_MAX_AREA, min_contour_area * 24.0)
    min_area = max(2.0, min_contour_area * 0.08)
    max_side = int(round(DARK_BUBBLE_RESIDUAL_MAX_SIDE_PX))
    min_border_distance = float(DARK_BUBBLE_RESIDUAL_MIN_BORDER_DISTANCE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) > max_side:
            continue
        if w > 48 and h < 5:
            continue
        if h > 48 and w < 5:
            continue

        component_mask = np.zeros_like(clean_mask)
        cv2.drawContours(component_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        component_dist = dist_map[component_mask == 255]
        if component_dist.size == 0:
            continue
        if float(np.median(component_dist)) < min_border_distance:
            continue

        ring_mask = cv2.dilate(component_mask, ring_kernel, iterations=2)
        ring_mask[component_mask == 255] = 0
        ring_mask = cv2.bitwise_and(ring_mask, safe_mask)
        ring_values = img_gray[ring_mask == 255]
        if ring_values.size >= 4:
            dark_ratio = float(
                np.count_nonzero(ring_values <= DARK_BUBBLE_RESIDUAL_RING_DARK_MAX)
            ) / float(ring_values.size)
            ring_median = float(np.median(ring_values))
            if (
                dark_ratio < DARK_BUBBLE_RESIDUAL_MIN_RING_DARK_RATIO
                and ring_median > 120.0
            ):
                continue

        cv2.bitwise_or(filtered_core, component_mask, dst=filtered_core)

    if not np.any(filtered_core):
        return clean_mask

    halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    residual_mask = cv2.dilate(filtered_core, halo_kernel, iterations=2)
    halo_allowed = np.where(
        (img_gray >= DARK_BUBBLE_RESIDUAL_BRIGHT_HALO) & (safe_mask == 255),
        255,
        0,
    ).astype(np.uint8)
    residual_mask = cv2.bitwise_and(residual_mask, halo_allowed)
    residual_mask = cv2.bitwise_and(residual_mask, safe_mask)

    added_px = int(np.count_nonzero((residual_mask > 0) & (clean_mask == 0)))
    if added_px <= 0:
        return clean_mask

    merged = cv2.bitwise_or(clean_mask, residual_mask)
    log_message(
        f"Detection {detection_bbox}: added dark-bubble residual text clean mask ({added_px} px)",
        verbose=verbose,
    )
    return merged


def _add_detector_guided_text_residual_mask(
    clean_mask: np.ndarray,
    base_mask: np.ndarray,
    img_gray: np.ndarray,
    text_boxes: Optional[list],
    fill_color_bgr: tuple[int, int, int],
    detection_bbox,
    verbose: bool,
) -> np.ndarray:
    """Add missed source glyphs using trusted AnimeText boxes."""
    if (
        clean_mask is None
        or base_mask is None
        or not text_boxes
        or fill_color_bgr not in ((0, 0, 0), (255, 255, 255))
    ):
        return clean_mask

    clean_mask = _normalize_mask(clean_mask)
    base_mask = _normalize_mask(base_mask)
    text_is_dark = fill_color_bgr == (255, 255, 255)
    seed_mask = build_lama_text_seed_mask(
        img_gray,
        base_mask,
        text_boxes,
        text_is_dark=text_is_dark,
    )
    if not np.any(seed_mask):
        return clean_mask

    uncovered_seed = cv2.bitwise_and(seed_mask, cv2.bitwise_not(clean_mask))
    if not np.any(uncovered_seed):
        return clean_mask

    # Minimal hgmzhn-style refinement covers glyph antialiasing without using
    # the much wider configurable LAMA expansion intended for colored bubbles.
    residual_mask = build_lama_inpaint_mask(
        uncovered_seed,
        base_mask,
        kernel_size=1,
        dilation_offset=0,
    )
    added_px = int(np.count_nonzero((residual_mask > 0) & (clean_mask == 0)))
    if added_px <= 0:
        return clean_mask

    log_message(
        f"Detection {detection_bbox}: added detector-guided source-text "
        f"clean mask ({added_px} px)",
        verbose=verbose,
    )
    return cv2.bitwise_or(clean_mask, residual_mask)


def _build_detector_guided_layout_mask(
    render_mask: Optional[np.ndarray],
    previous_clean_mask: Optional[np.ndarray],
    clean_mask: Optional[np.ndarray],
    detection_bbox,
    fill_color_bgr: tuple[int, int, int],
    is_colored_bubble: bool,
    verbose: bool,
) -> Optional[np.ndarray]:
    """Let source-text pixels that were actually cleaned count as layout space."""
    if (
        render_mask is None
        or previous_clean_mask is None
        or clean_mask is None
        or is_colored_bubble
        or fill_color_bgr != (255, 255, 255)
    ):
        return None

    try:
        render = _normalize_mask(render_mask)
        before = _normalize_mask(previous_clean_mask)
        after = _normalize_mask(clean_mask)
    except Exception:
        return None

    if render.shape != before.shape or render.shape != after.shape:
        return None

    added = cv2.bitwise_and(after, cv2.bitwise_not(before))
    if not np.any(added):
        return None

    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in detection_bbox]
    except (TypeError, ValueError):
        return None

    img_h, img_w = render.shape[:2]
    x0 = max(0, min(img_w, x0))
    x1 = max(0, min(img_w, x1))
    y0 = max(0, min(img_h, y0))
    y1 = max(0, min(img_h, y1))
    if x1 <= x0 or y1 <= y0:
        return None

    bbox_mask = np.zeros_like(render)
    bbox_mask[y0:y1, x0:x1] = 255
    added = cv2.bitwise_and(added, bbox_mask)
    if not np.any(added):
        return None

    # Keep this tied to the original render mask so a stray detector box cannot
    # turn distant art into layout space.  Once a cleaned glyph component touches
    # the render mask, keep the whole component so thick source strokes do not
    # remain as holes in the layout mask.
    join_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_render = cv2.dilate(render, join_kernel, iterations=1)
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats((added > 0).astype(np.uint8), 8)
    )
    connected_added = np.zeros_like(added)
    for component_id in range(1, component_count):
        component = component_labels == component_id
        if int(component_stats[component_id, cv2.CC_STAT_AREA]) <= 0:
            continue
        if np.any(component & (near_render > 0)):
            connected_added[component] = 255

    if not np.any(connected_added):
        return None

    candidate = cv2.bitwise_or(render, connected_added)
    added_px = int(np.count_nonzero((candidate > 0) & (render == 0)))
    if added_px <= 0:
        return None

    log_message(
        f"Detection {detection_bbox}: included detector-guided cleaned "
        f"source text in render layout mask ({added_px} px)",
        verbose=verbose,
    )
    return candidate


def _build_conjoined_white_text_only_clean_mask(
    base_mask: np.ndarray,
    img_gray: np.ndarray,
    text_boxes: Optional[list],
) -> Optional[np.ndarray]:
    """Build a trusted glyph-only mask without repainting bubble outlines."""
    if base_mask is None:
        return None

    base = _normalize_mask(base_mask)
    text_seed = np.zeros_like(base)
    if text_boxes:
        text_seed = build_lama_text_seed_mask(
            img_gray,
            base,
            text_boxes,
            text_is_dark=True,
        )

    # Conjoined detection can occasionally return only one child OCR box.
    # The parent has already passed a strict solid-white test, so a second
    # boundary-protected full-bubble seed can safely recover omitted glyphs.
    full_seed = build_lama_text_seed_mask(
        img_gray,
        base,
        None,
        text_is_dark=True,
    )
    base_pixels = max(1, int(np.count_nonzero(base)))
    full_seed_pixels = int(np.count_nonzero(full_seed))
    missing_seed = cv2.bitwise_and(full_seed, cv2.bitwise_not(text_seed))
    if (
        full_seed_pixels > 0
        and full_seed_pixels / float(base_pixels)
        <= CONJOINED_PARENT_FULL_SEED_MAX_RATIO
        and int(np.count_nonzero(missing_seed))
        >= CONJOINED_PARENT_FULL_SEED_MIN_MISSING_PIXELS
    ):
        text_seed = cv2.bitwise_or(text_seed, full_seed)

    text_mask = np.zeros_like(base)
    if np.any(text_seed):
        text_mask = build_lama_inpaint_mask(
            text_seed,
            base,
            kernel_size=1,
            dilation_offset=0,
        )

    safe_base = cv2.erode(
        base,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    if not np.any(safe_base):
        safe_base = base

    trusted_boxes = np.zeros_like(base)
    image_height, image_width = base.shape
    for box in text_boxes or []:
        try:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        except (TypeError, ValueError):
            continue
        x0 = max(0, min(image_width - 1, x0))
        y0 = max(0, min(image_height - 1, y0))
        x1 = max(x0 + 1, min(image_width, x1))
        y1 = max(y0 + 1, min(image_height, y1))
        trusted_boxes[y0:y1, x0:x1] = 255

    if np.any(trusted_boxes):
        # The generic seed deliberately removes long lines to protect artwork.
        # In a pixel-confirmed white parent, trusted OCR boxes can safely recover
        # those Japanese glyph strokes and their pale antialiasing.
        pale_glyphs = np.where(
            (img_gray <= 250)
            & (trusted_boxes == 255)
            & (safe_base == 255),
            255,
            0,
        ).astype(np.uint8)
        if np.any(pale_glyphs):
            pale_glyphs = cv2.dilate(
                pale_glyphs,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            pale_glyphs = cv2.bitwise_and(pale_glyphs, safe_base)
            text_mask = cv2.bitwise_or(text_mask, pale_glyphs)

    text_mask = cv2.bitwise_and(text_mask, safe_base)
    if not np.any(text_mask):
        return None
    return text_mask


def _complete_bright_translucent_text_seed(
    img_gray: np.ndarray,
    lama_domain: np.ndarray,
    guided_seed: np.ndarray,
    translucent_stats: dict[str, float],
    *,
    text_is_dark: bool,
    detection_bbox,
    verbose: bool,
) -> np.ndarray:
    """Recover omitted dark glyph columns in an otherwise bright LAMA bubble."""
    seed = _normalize_mask(guided_seed)
    domain = _normalize_mask(lama_domain)
    if (
        not text_is_dark
        or float(translucent_stats.get("bright_ratio", 0.0))
        < BRIGHT_TRANSLUCENT_COMPLETE_MIN_BRIGHT_RATIO
        or float(translucent_stats.get("median", 0.0))
        < BRIGHT_TRANSLUCENT_COMPLETE_MIN_MEDIAN
    ):
        return seed

    full_seed = build_lama_text_seed_mask(
        img_gray,
        domain,
        None,
        text_is_dark=True,
    )
    full_pixels = int(np.count_nonzero(full_seed))
    if full_pixels <= 0:
        return seed

    domain_pixels = max(1, int(np.count_nonzero(domain)))
    guided_overlap = int(np.count_nonzero(cv2.bitwise_and(seed, full_seed)))
    missing_seed = cv2.bitwise_and(full_seed, cv2.bitwise_not(seed))
    missing_pixels = int(np.count_nonzero(missing_seed))
    if (
        full_pixels / float(domain_pixels)
        > BRIGHT_TRANSLUCENT_COMPLETE_MAX_FULL_SEED_RATIO
        or guided_overlap / float(full_pixels)
        > BRIGHT_TRANSLUCENT_COMPLETE_MAX_GUIDED_COVERAGE
        or missing_pixels < BRIGHT_TRANSLUCENT_COMPLETE_MIN_MISSING_PIXELS
    ):
        return seed

    log_message(
        f"Detection {detection_bbox}: completed bright translucent source-text "
        f"seed ({missing_pixels} missing px)",
        verbose=verbose,
    )
    return cv2.bitwise_or(seed, full_seed)


def _build_dark_bubble_text_only_clean_mask(
    base_mask: np.ndarray,
    img_gray: np.ndarray,
    text_boxes: Optional[list],
    detection_bbox,
    verbose: bool,
) -> Optional[np.ndarray]:
    """Erase trusted white glyphs without repainting occluding dark artwork."""
    if base_mask is None or not text_boxes:
        return None

    base = _normalize_mask(base_mask)
    if not np.any(base):
        return None

    seed = build_lama_text_seed_mask(
        img_gray,
        base,
        text_boxes,
        text_is_dark=False,
        protect_background_artwork=True,
    )
    seed_pixels = int(np.count_nonzero(seed))
    if seed_pixels < DARK_TEXT_ONLY_MIN_SEED_PIXELS:
        return None

    safe_base = cv2.erode(
        base,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if not np.any(safe_base):
        safe_base = base

    trusted_boxes = np.zeros_like(base)
    solid_black_residual = np.zeros_like(base)
    image_height, image_width = base.shape
    for box in text_boxes or []:
        try:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        except (TypeError, ValueError):
            continue
        x0 = max(0, min(image_width, x0))
        y0 = max(0, min(image_height, y0))
        x1 = max(x0, min(image_width, x1))
        y1 = max(y0, min(image_height, y1))
        box_safe = safe_base[y0:y1, x0:x1] == 255
        box_values = img_gray[y0:y1, x0:x1][box_safe]
        if (
            box_values.size >= 64
            and float(np.median(box_values))
            <= DARK_TEXT_ONLY_SOLID_BLACK_MAX_MEDIAN
            and float(
                np.mean(box_values <= DARK_TEXT_ONLY_SOLID_BLACK_DARK_LIMIT)
            )
            >= DARK_TEXT_ONLY_SOLID_BLACK_MIN_DARK_RATIO
            and float(np.mean(box_values >= DARK_BUBBLE_RESIDUAL_BRIGHT_CORE))
            >= DARK_TEXT_ONLY_SOLID_BLACK_MIN_BRIGHT_RATIO
        ):
            residual_roi = solid_black_residual[y0:y1, x0:x1]
            residual_roi[box_safe] = 255

        pad = DARK_TEXT_ONLY_BOX_PADDING
        x0 = max(0, min(image_width, x0 - pad))
        y0 = max(0, min(image_height, y0 - pad))
        x1 = max(x0, min(image_width, x1 + pad))
        y1 = max(y0, min(image_height, y1 + pad))
        trusted_boxes[y0:y1, x0:x1] = 255

    # White source glyphs on black bubbles often have a gray antialiasing or
    # outline band. Grow only from trusted bright cores, through sufficiently
    # bright connected pixels, and only inside padded OCR boxes.
    halo_support = np.where(
        (img_gray >= DARK_TEXT_ONLY_HALO_MIN_VALUE)
        & (safe_base == 255)
        & (trusted_boxes == 255),
        255,
        0,
    ).astype(np.uint8)
    grown_seed = seed.copy()
    grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(DARK_TEXT_ONLY_HALO_GROW_ITERATIONS):
        next_seed = cv2.bitwise_and(
            cv2.dilate(grown_seed, grow_kernel, iterations=1),
            halo_support,
        )
        next_seed = cv2.bitwise_or(next_seed, seed)
        if np.array_equal(next_seed, grown_seed):
            break
        grown_seed = next_seed

    refined = build_lama_inpaint_mask(
        grown_seed,
        base,
        kernel_size=1,
        dilation_offset=0,
    )
    refined = cv2.bitwise_or(refined, solid_black_residual)
    refined = cv2.bitwise_and(refined, safe_base)
    refined_pixels = int(np.count_nonzero(refined))
    base_pixels = max(1, int(np.count_nonzero(base)))
    if (
        refined_pixels < int(np.count_nonzero(grown_seed))
        or float(refined_pixels) / float(base_pixels)
        > DARK_TEXT_ONLY_MAX_MASK_RATIO
    ):
        return None

    log_message(
        f"Detection {detection_bbox}: using text-only dark-bubble clean mask "
        f"({refined_pixels} px, {refined_pixels / float(base_pixels):.2f} of mask)",
        verbose=verbose,
    )
    return refined


def _build_adaptive_shrink_mask(
    roi_mask: np.ndarray,
    roi_shrink_px: float,
    detection_bbox: tuple,
    neighbor_bboxes: list,
    processing_scale: float = 1.0,
) -> np.ndarray:
    """
    Shrink ROI uniformly except near conjoined neighbor edges, where only
    JUNCTION_MIN_SHRINK is applied.  Without this, the distance-transform
    shrink pinches off narrow junction passages and leaves remnant text.
    """
    adjacency_margin = max(1, int(round(JUNCTION_ADJACENCY_MARGIN * processing_scale)))
    junction_min_shrink = max(1.0, JUNCTION_MIN_SHRINK * processing_scale)

    dist_map = cv2.distanceTransform(
        roi_mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE
    )
    shrunk = np.where(dist_map >= roi_shrink_px, 255, 0).astype(np.uint8)

    x1, y1, x2, y2 = detection_bbox
    h, w = roi_mask.shape[:2]

    for other_bbox in neighbor_bboxes:
        ox1, oy1, ox2, oy2 = other_bbox

        if (
            x1 - adjacency_margin > ox2
            or ox1 - adjacency_margin > x2
            or y1 - adjacency_margin > oy2
            or oy1 - adjacency_margin > y2
        ):
            continue

        zone_x1 = max(0, max(x1, ox1) - adjacency_margin)
        zone_y1 = max(0, max(y1, oy1) - adjacency_margin)
        zone_x2 = min(w, min(x2, ox2) + adjacency_margin)
        zone_y2 = min(h, min(y2, oy2) + adjacency_margin)

        if zone_x2 <= zone_x1 or zone_y2 <= zone_y1:
            continue

        junction_slice = dist_map[zone_y1:zone_y2, zone_x1:zone_x2]
        restored = np.where(junction_slice >= junction_min_shrink, 255, 0).astype(
            np.uint8
        )
        np.bitwise_or(
            shrunk[zone_y1:zone_y2, zone_x1:zone_x2],
            restored,
            out=shrunk[zone_y1:zone_y2, zone_x1:zone_x2],
        )

    return shrunk


def process_single_bubble(
    base_mask,
    img_gray,
    img_height,
    img_width,
    thresholding_value,
    use_otsu_threshold,
    roi_shrink_px,
    verbose,
    detection_bbox=None,
    is_sam=False,
    dilation_kernel=None,
    constraint_erosion_kernel=None,
    min_contour_area: float = MIN_CONTOUR_AREA,
    classify_colored: bool = False,
    neighbor_bboxes: Optional[list] = None,
    processing_scale: float = 1.0,
    image_bgr: Optional[np.ndarray] = None,
    text_boxes: Optional[list] = None,
):
    """
    Process a single speech bubble mask to extract text regions and determine fill color.

    Args:
        base_mask (numpy.ndarray): The base mask (SAM or YOLO) for the bubble
        img_gray (numpy.ndarray): Grayscale image
        img_height (int): Image height
        img_width (int): Image width
        thresholding_value (int): Fixed threshold value for text detection
        use_otsu_threshold (bool): Whether to use Otsu's method for thresholding
        roi_shrink_px (int): Pixels to shrink ROI inwards
        verbose (bool): Whether to print verbose messages
        detection_bbox: Bounding box for logging (optional)
        is_sam (bool): Whether this is a SAM mask (for logging)
        dilation_kernel: Kernel for dilation
        constraint_erosion_kernel: Kernel for erosion
        min_contour_area: Min area for contours
        classify_colored: Whether to classify colored bubbles
        neighbor_bboxes: Neighboring bboxes for shrink logic
        processing_scale: Processing scale multiplier
        image_bgr (numpy.ndarray): BGR image for text color sampling

    Returns:
        tuple: (render_mask, clean_mask, fill_color_bgr, is_colored,
                use_solid_midtone_fill, sample_color_bgr, text_bbox, text_color_bgr)

    Raises:
        CleaningError: If processing fails
    """
    try:
        base_mask = _normalize_mask(base_mask)

        if dilation_kernel is None:
            dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, DILATION_KERNEL_SIZE
            )
        if constraint_erosion_kernel is None:
            constraint_erosion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, EROSION_KERNEL_SIZE
            )
        masked_pixels = img_gray[base_mask == 255]
        if masked_pixels.size == 0:
            log_message(
                f"{'[SAM]' if is_sam else ''}Skipping detection {detection_bbox}: empty mask",
                verbose=verbose,
            )
            raise CleaningError(f"Empty mask for detection {detection_bbox}")

        mean_pixel_value = np.mean(masked_pixels)
        is_black_bubble = mean_pixel_value < GRAYSCALE_MIDPOINT
        fill_color_bgr = (0, 0, 0) if is_black_bubble else (255, 255, 255)
        is_colored_bubble = False
        use_solid_midtone_fill = False
        sample_color_bgr: tuple[int, int, int] = fill_color_bgr

        log_message(
            f"{'[SAM]' if is_sam else ''}Detection {detection_bbox}: "
            f"{'Black' if is_black_bubble else 'White'} bubble (mean={mean_pixel_value:.1f})",
            verbose=verbose,
        )

        roi_mask = cv2.dilate(base_mask, dilation_kernel, iterations=1)
        roi_gray = np.zeros_like(img_gray)
        roi_indices = roi_mask == 255
        roi_gray[roi_indices] = img_gray[roi_indices]

        # Invert for black bubbles to detect text properly
        roi_for_thresholding = (
            cv2.bitwise_not(roi_gray) if is_black_bubble else roi_gray
        )
        thresholded_roi = np.zeros_like(img_gray)

        if use_otsu_threshold:
            roi_pixels_for_otsu = roi_for_thresholding[roi_indices]
            thresh_val, _ = cv2.threshold(
                roi_pixels_for_otsu, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            log_message(
                f"{'[SAM]' if is_sam else ''}  Otsu threshold: {thresh_val}",
                verbose=verbose,
            )
            _, thresholded_roi = cv2.threshold(
                roi_for_thresholding, thresh_val, 255, cv2.THRESH_BINARY
            )
        else:
            _, thresholded_roi = cv2.threshold(
                roi_for_thresholding, thresholding_value, 255, cv2.THRESH_BINARY
            )

        thresholded_roi = cv2.bitwise_and(thresholded_roi, roi_mask)

        if neighbor_bboxes and detection_bbox is not None:
            shrunk_roi_mask = _build_adaptive_shrink_mask(
                roi_mask,
                float(roi_shrink_px),
                detection_bbox,
                neighbor_bboxes,
                processing_scale=processing_scale,
            )
        else:
            dist_map = cv2.distanceTransform(
                roi_mask, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK_SIZE
            )
            shrunk_roi_mask = np.where(dist_map >= float(roi_shrink_px), 255, 0).astype(
                np.uint8
            )
        thresholded_roi = cv2.bitwise_and(thresholded_roi, shrunk_roi_mask)

        # Use eroded mask to avoid erasing bubble outlines
        eroded_constraint_mask = cv2.erode(
            base_mask, constraint_erosion_kernel, iterations=1
        )

        contours, _ = cv2.findContours(
            thresholded_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= min_contour_area:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            if (
                0 <= cx < img_width
                and 0 <= cy < img_height
                and eroded_constraint_mask[cy, cx] == 255
            ):
                valid_contours.append(cnt)

        log_message(
            f"{'[SAM]' if is_sam else ''}Detection {detection_bbox}: {len(valid_contours)} text fragments found",
            verbose=verbose,
        )

        text_bbox = None
        if valid_contours:
            validated_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.drawContours(
                validated_mask, valid_contours, -1, 255, thickness=cv2.FILLED
            )
            clean_mask = validated_mask.copy()

            # Re-contour to get clean boundary from validated mask
            boundary_contours, _ = cv2.findContours(
                validated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if boundary_contours:
                largest_contour = max(boundary_contours, key=cv2.contourArea)
                final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                cv2.drawContours(
                    final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED
                )
                x, y, w, h = cv2.boundingRect(largest_contour)
                text_bbox = (x, y, x + w, y + h)

                if classify_colored:
                    # Sample bubble interior excluding exact text pixels and outline to determine if colored
                    interior_sampling_mask = cv2.erode(
                        base_mask, constraint_erosion_kernel, iterations=2
                    )
                    raw_interior_values = img_gray[interior_sampling_mask == 255]
                    sampling_mask = interior_sampling_mask.copy()
                    text_mask = cv2.bitwise_and(
                        cv2.bitwise_not(thresholded_roi), shrunk_roi_mask
                    )
                    text_mask_dilated = cv2.dilate(
                        text_mask, np.ones((3, 3), np.uint8), iterations=1
                    )
                    sampling_mask[text_mask_dilated == 255] = 0

                    sample_pixels = img_gray[sampling_mask == 255]
                    if sample_pixels.size == 0:
                        sample_pixels = masked_pixels

                    sample_values = sample_pixels.astype(np.uint8).flatten()
                    hist = np.bincount(sample_values, minlength=256)
                    dominant_val = (
                        int(hist.argmax()) if hist.size > 0 else int(mean_pixel_value)
                    )
                    dominant_count = int(hist.max()) if hist.size > 0 else 0
                    total_count = max(int(sample_values.size), 1)
                    dominant_ratio = dominant_count / float(total_count)
                    bright_ratio = float(
                        np.count_nonzero(sample_values >= 245)
                    ) / float(total_count)
                    dark_ratio = float(np.count_nonzero(sample_values <= 15)) / float(
                        total_count
                    )

                    log_prefix = "[SAM] " if is_sam else ""
                    if bright_ratio >= BRIGHT_RATIO_THRESHOLD or (
                        dominant_val >= 245
                        and dominant_ratio >= BRIGHT_DOM_RATIO_MIN
                        and dark_ratio <= BRIGHT_DARK_RATIO_MAX
                    ):
                        (
                            has_midtone_background,
                            midtone_mode,
                            midtone_stats,
                        ) = _detect_confident_midtone_background(
                            raw_interior_values
                        )
                        (
                            has_midtone_ring,
                            ring_midtone_value,
                            ring_stats,
                        ) = _detect_confident_midtone_inner_ring(
                            img_gray,
                            base_mask,
                        )
                        if has_midtone_ring:
                            has_midtone_background = True
                            midtone_mode = ring_midtone_value
                            midtone_stats = ring_stats
                            midtone_stats["source"] = "inner-ring"
                            use_solid_midtone_fill = True
                        if has_midtone_background:
                            is_colored_bubble = True
                            fill_color_bgr = (
                                midtone_mode,
                                midtone_mode,
                                midtone_mode,
                            )
                            sample_color_bgr = (
                                midtone_mode,
                                midtone_mode,
                                midtone_mode,
                            )
                            log_message(
                                f"{log_prefix}Detection {detection_bbox}: "
                                f"colored/gray override (mode={midtone_mode}, "
                                f"source={midtone_stats.get('source', 'interior')}, "
                                f"midtone_ratio={midtone_stats['midtone_ratio']:.2f}, "
                                f"bright_ratio={midtone_stats['bright_ratio']:.2f}, "
                                f"dark_ratio={midtone_stats['dark_ratio']:.2f}, "
                                f"median={midtone_stats['median']:.1f})",
                                verbose=verbose,
                            )
                        else:
                            (
                                has_translucent_background,
                                translucent_sample,
                                translucent_stats,
                            ) = _detect_confident_translucent_background(
                                img_gray,
                                base_mask,
                                text_boxes,
                            )
                            (
                                has_conjoined_white_core,
                                white_core_stats,
                            ) = _detect_conjoined_white_text_core(
                                img_gray,
                                base_mask,
                                text_boxes,
                                detection_bbox,
                                neighbor_bboxes,
                            )
                            (
                                has_conjoined_white_interior,
                                white_interior_stats,
                            ) = _detect_conjoined_white_interior(
                                img_gray,
                                base_mask,
                                detection_bbox,
                                neighbor_bboxes,
                            )
                            if (
                                has_translucent_background
                                and (
                                    has_conjoined_white_core
                                    or has_conjoined_white_interior
                                )
                            ):
                                has_translucent_background = False
                                if has_conjoined_white_core:
                                    log_message(
                                        f"{log_prefix}Detection {detection_bbox}: "
                                        "kept white; conjoined mask overflow caused "
                                        "the translucent sample "
                                        f"(overflow={white_core_stats['overflow_ratio']:.2f}, "
                                        f"core_bright={white_core_stats['bright_ratio']:.2f}, "
                                        f"core_midtone={white_core_stats['midtone_ratio']:.2f}, "
                                        f"core_median={white_core_stats['median']:.1f})",
                                        verbose=verbose,
                                    )
                                else:
                                    log_message(
                                        f"{log_prefix}Detection {detection_bbox}: "
                                        "kept white; bright conjoined interior "
                                        "overrode the translucent sample "
                                        f"(interior_bright={white_interior_stats['bright_ratio']:.2f}, "
                                        f"interior_midtone={white_interior_stats['midtone_ratio']:.2f}, "
                                        f"interior_dark={white_interior_stats['dark_ratio']:.2f}, "
                                        f"interior_median={white_interior_stats['median']:.1f})",
                                        verbose=verbose,
                                    )
                            if has_translucent_background:
                                is_colored_bubble = True
                                render_sample = _resolve_translucent_render_sample(
                                    translucent_sample,
                                    translucent_stats,
                                )
                                sample_color_bgr = (
                                    render_sample,
                                    render_sample,
                                    render_sample,
                                )
                                log_message(
                                    f"{log_prefix}Detection {detection_bbox}: "
                                    "colored/translucent "
                                    f"(sample={translucent_sample}, "
                                    f"render_sample={render_sample}, "
                                    f"sample_ratio={translucent_stats['sample_ratio']:.2f}, "
                                    f"midtone_ratio={translucent_stats['midtone_ratio']:.2f}, "
                                    f"bright_ratio={translucent_stats['bright_ratio']:.2f}, "
                                    f"std={translucent_stats['stddev']:.1f}, "
                                    f"edge_ratio={translucent_stats['edge_ratio']:.2f})",
                                    verbose=verbose,
                                )
                            else:
                                is_colored_bubble = False
                                fill_color_bgr = (255, 255, 255)
                                sample_color_bgr = (255, 255, 255)
                                log_message(
                                    f"{log_prefix}Detection {detection_bbox}: white "
                                    f"(mode={dominant_val}, dom_ratio={dominant_ratio:.2f}, "
                                    f"bright_ratio={bright_ratio:.2f}, dark_ratio={dark_ratio:.2f})",
                                    verbose=verbose,
                                )
                    elif dark_ratio >= DARK_RATIO_THRESHOLD or (
                        dominant_val <= 15
                        and dominant_ratio >= DARK_DOM_RATIO_MIN
                        and bright_ratio <= DARK_BRIGHT_RATIO_MAX
                    ):
                        is_colored_bubble = False
                        fill_color_bgr = (0, 0, 0)
                        sample_color_bgr = (0, 0, 0)
                        log_message(
                            f"{log_prefix}Detection {detection_bbox}: black "
                            f"(mode={dominant_val}, dom_ratio={dominant_ratio:.2f}, "
                            f"bright_ratio={bright_ratio:.2f}, dark_ratio={dark_ratio:.2f})",
                            verbose=verbose,
                        )
                    else:
                        is_colored_bubble = True
                        sample_color_bgr = (dominant_val, dominant_val, dominant_val)
                        log_message(
                            f"{log_prefix}Detection {detection_bbox}: "
                            f"colored/gradient (mode={dominant_val}, "
                            f"dom_ratio={dominant_ratio:.2f}, "
                            f"bright_ratio={bright_ratio:.2f}, "
                            f"dark_ratio={dark_ratio:.2f})",
                                verbose=verbose,
                            )

                if use_solid_midtone_fill:
                    clean_mask = _complete_solid_midtone_text_mask(
                        clean_mask,
                        base_mask,
                        img_gray,
                        sample_color_bgr[0],
                        constraint_erosion_kernel,
                    )

                text_color_bgr = None
                if image_bgr is not None:
                    text_mask = cv2.bitwise_and(
                        cv2.bitwise_not(thresholded_roi), shrunk_roi_mask
                    )
                    sample_mask = cv2.erode(
                        text_mask, np.ones((3, 3), np.uint8), iterations=1
                    )
                    text_pixels_bgr = image_bgr[sample_mask == 255]
                    # Fallback if erosion obliterates thin text
                    if text_pixels_bgr.size == 0:
                        text_pixels_bgr = image_bgr[text_mask == 255]

                    if text_pixels_bgr.size > 0:
                        sampled_bgr = tuple(
                            np.median(text_pixels_bgr, axis=0).astype(int)
                        )
                        hsv = cv2.cvtColor(
                            np.uint8([[sampled_bgr]]), cv2.COLOR_BGR2HSV
                        )[0][0]
                        if hsv[1] < 25:
                            text_color_bgr = _resolve_achromatic_text_color(
                                sampled_bgr,
                                sample_color_bgr,
                                is_black_bubble=is_black_bubble,
                            )
                        else:
                            text_color_bgr = sampled_bgr

                if is_black_bubble:
                    text_only_clean_mask = _build_dark_bubble_text_only_clean_mask(
                        base_mask,
                        img_gray,
                        text_boxes,
                        detection_bbox,
                        verbose,
                    )
                    if text_only_clean_mask is not None:
                        clean_mask = text_only_clean_mask
                    else:
                        clean_mask = _add_dark_bubble_bright_residual_mask(
                            clean_mask,
                            base_mask,
                            img_gray,
                            constraint_erosion_kernel,
                            min_contour_area,
                            detection_bbox,
                            verbose,
                        )

                return (
                    final_mask,
                    clean_mask,
                    fill_color_bgr,
                    is_colored_bubble,
                    use_solid_midtone_fill,
                    sample_color_bgr,
                    text_bbox,
                    text_color_bgr,
                )

        raise CleaningError("Failed to process bubble mask")

    except Exception as e:
        log_message(
            f"Failed to process {'SAM' if is_sam else 'YOLO'} mask for {detection_bbox}",
            always_print=True,
        )
        raise CleaningError("Failed to process bubble mask") from e


def clean_speech_bubbles(
    image_input: Union[str, Path, Image.Image],
    model_path,
    confidence=0.6,
    pre_computed_detections=None,
    device=None,
    thresholding_value: int = 200,
    use_otsu_threshold: bool = False,
    roi_shrink_px: int = 5,
    verbose: bool = False,
    processing_scale: float = 1.0,
    conjoined_confidence=0.35,
    inpaint_colored_bubbles: bool = False,
    colored_bubble_inpainting_size: int = 2048,
    colored_bubble_inpainting_precision: str = "fp32",
    colored_bubble_mask_kernel_size: int = 3,
    colored_bubble_mask_dilation_offset: int = 50,
    flux_hf_token: str = "",
    flux_num_inference_steps: int = 8,
    flux_residual_diff_threshold: float = 0.15,
    flux_seed: int = 1,
    osb_text_verification: bool = False,
    osb_text_hf_token: str = "",
    inpaint_method: str = "flux_kontext",
    kontext_backend: str = "nunchaku",
    flux_cpu_offload: bool = True,
    flux_low_vram: bool = False,
    flux_luminance_correction: bool = True,
    flux_upscale_small_crops: bool = True,
    flux_kontext_prompt: str = "",
    flux_klein_prompt: str = "",
    bubble_detector_model: str = "yolo_1",
    lama_source_image: Optional[Union[str, Path, Image.Image]] = None,
):
    """
    Clean speech bubbles using YOLO/SAM masks and optional LAMA Large inpainting.

    Args:
        image_input (str, Path, or PIL.Image.Image): Path to input image or a PIL Image object.
        model_path (str): Path to YOLO model.
        confidence (float): Confidence threshold for detections.
        pre_computed_detections (list, optional): Pre-computed detections from previous call.
        device (torch.device, optional): The device to run detection model on if needed.
        thresholding_value (int): Fixed threshold value for text detection (0-255). Lower values (e.g., 180)
                                 are useful for uncleaned text close to bubble's edges.
        use_otsu_threshold (bool): If True, use Otsu's method for thresholding instead of the fixed value.
        roi_shrink_px (int): Number of pixels to shrink the ROI inwards before identification/fill.
        inpaint_colored_bubbles (bool): Use LAMA Large for non-white/black bubbles.
        colored_bubble_inpainting_size (int): Maximum LAMA inference image side.
        colored_bubble_inpainting_precision (str): LAMA torch precision.
        colored_bubble_mask_kernel_size (int): Text-mask dilation kernel size.
        colored_bubble_mask_dilation_offset (int): Additional text-mask expansion.
        osb_text_verification (bool): When True, expand bubble boxes to fully cover OSB text detections.
        osb_text_hf_token (str): Optional token for OSB text model downloads.
        lama_source_image: Optional pre-OSB image used for bubble-mask analysis
                           and LAMA context. The final pixels are still composed
                           onto the post-OSB cleaning image.

    Returns:
        numpy.ndarray: Cleaned image with text removed.
        list[dict]: A list of dictionaries per bubble containing:
                    - 'mask' (np.ndarray): validated text mask (0/255)
                    - 'base_mask' (np.ndarray): normalized detection mask used for processing
                    - 'color' (tuple BGR): sampled bubble color
                    - 'bbox' (tuple): detection bounding box
                    - 'is_colored' (bool): whether bubble interior was classified colored
                    - 'text_bbox' (tuple|None): bounding box of detected text mask
                    - 'is_sam' (bool): whether detection originated from SAM
    Raises:
        ValueError: If the image cannot be loaded or if an image object is passed without pre-computed detections.
        RuntimeError: If model loading or bubble detection fails.
    """
    try:
        if isinstance(image_input, (str, Path)):
            pil_image = Image.open(image_input)
            image_path = image_input
        else:
            pil_image = image_input
            image_path = None  # In-memory image has no path

        image = pil_to_cv2(pil_image)
        img_height, img_width = image.shape[:2]

        cleaned_image = image.copy()
        lama_source_cv = image
        if lama_source_image is not None:
            if isinstance(lama_source_image, (str, Path)):
                with Image.open(lama_source_image) as source_pil:
                    lama_source_cv = pil_to_cv2(source_pil)
            else:
                lama_source_cv = pil_to_cv2(lama_source_image)
            if lama_source_cv.shape[:2] != (img_height, img_width):
                raise ValidationError(
                    "LAMA source image dimensions must match the cleaning image."
                )
        img_gray = cv2.cvtColor(lama_source_cv, cv2.COLOR_BGR2GRAY)

        if pre_computed_detections is not None:
            detections = pre_computed_detections
        elif image_path is not None:
            detection_result = detect_speech_bubbles(
                image_path,
                model_path,
                confidence,
                device=device,
                conjoined_confidence=conjoined_confidence,
                osb_text_verification=osb_text_verification,
                osb_text_hf_token=osb_text_hf_token,
                bubble_detector_model=bubble_detector_model,
                collect_text_boxes=inpaint_colored_bubbles,
            )
            detections = (
                detection_result[0]
                if isinstance(detection_result, tuple)
                else detection_result
            )
        else:
            raise ValidationError(
                "Bubble detection requires an image path, but an image object "
                "was provided without pre-computed detections."
            )

        processed_bubbles = []

        effective_roi_shrink_px = float(
            scale_scalar(
                roi_shrink_px,
                processing_scale,
                minimum=0.0,
                maximum=64.0,
            )
        )
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, scale_kernel(DILATION_KERNEL_SIZE, processing_scale)
        )
        constraint_erosion_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, scale_kernel(EROSION_KERNEL_SIZE, processing_scale)
        )
        min_contour_area = scale_area(
            MIN_CONTOUR_AREA,
            processing_scale,
            minimum=MIN_CONTOUR_AREA,
            maximum=5000,
        )
        for detection in detections:
            final_mask = None
            clean_mask = None
            layout_mask = None
            fill_color_bgr = None
            is_colored_bubble = False
            use_solid_midtone_fill = False
            sample_color_bgr: Optional[tuple[int, int, int]] = None
            text_bbox: Optional[tuple[int, int, int, int]] = None
            text_color_bgr: Optional[tuple[int, int, int]] = None
            base_mask = None
            is_sam_mask = False

            sam_mask = detection.get("sam_mask")
            if sam_mask is not None:
                base_mask = _normalize_mask(sam_mask)
                is_sam_mask = True
                try:
                    (
                        final_mask,
                        clean_mask,
                        fill_color_bgr,
                        is_colored_bubble,
                        use_solid_midtone_fill,
                        sample_color_bgr,
                        text_bbox,
                        text_color_bgr,
                    ) = process_single_bubble(
                        base_mask,
                        img_gray,
                        img_height,
                        img_width,
                        thresholding_value,
                        use_otsu_threshold,
                        effective_roi_shrink_px,
                        verbose,
                        detection.get("bbox"),
                        is_sam=True,
                        dilation_kernel=dilation_kernel,
                        constraint_erosion_kernel=constraint_erosion_kernel,
                        min_contour_area=min_contour_area,
                        classify_colored=inpaint_colored_bubbles,
                        neighbor_bboxes=detection.get("conjoined_neighbor_bboxes"),
                        processing_scale=processing_scale,
                        image_bgr=lama_source_cv,
                        text_boxes=detection.get("text_boxes"),
                    )
                except Exception as e:
                    retry_success = False
                    if not use_otsu_threshold and base_mask is not None:
                        log_message(
                            f"Standard cleaning failed for {detection.get('bbox')}, retrying with Otsu...",
                            verbose=verbose,
                        )
                        retry_res = retry_cleaning_with_otsu(
                            lama_source_cv,
                            {
                                "base_mask": base_mask,
                                "bbox": detection.get("bbox"),
                                "is_sam": True,
                                "neighbor_bboxes": detection.get(
                                    "conjoined_neighbor_bboxes"
                                ),
                                "text_boxes": detection.get("text_boxes"),
                            },
                            thresholding_value,
                            roi_shrink_px,
                            processing_scale,
                            verbose,
                            inpaint_colored_bubbles,
                        )
                        if retry_res:
                            final_mask = retry_res["mask"]
                            clean_mask = retry_res.get("clean_mask", final_mask)
                            fill_color_bgr = retry_res["color"]
                            sample_color_bgr = retry_res["color"]
                            is_colored_bubble = retry_res["is_colored"]
                            use_solid_midtone_fill = retry_res.get(
                                "use_solid_midtone_fill", False
                            )
                            text_bbox = retry_res["text_bbox"]
                            text_color_bgr = retry_res.get("text_color_bgr")
                            retry_success = True
                            log_message(
                                f"Otsu retry successful for {detection.get('bbox')}",
                                verbose=verbose,
                            )
                        else:
                            log_message(
                                f"Otsu retry failed for {detection.get('bbox')}",
                                verbose=verbose,
                            )

                    if not retry_success:
                        error_msg = f"Error processing SAM mask for detection {detection.get('bbox')}: {e}"
                        log_message(error_msg, always_print=True)
                        continue
            else:
                if "mask_points" not in detection or not detection["mask_points"]:
                    log_message(
                        f"Skipping detection {detection.get('bbox')}: no mask points",
                        verbose=verbose,
                    )
                    continue

                try:
                    points_list = detection["mask_points"]
                    points = np.array(points_list, dtype=np.float32)

                    if len(points.shape) == 3 and points.shape[1] == 1:
                        points_int = np.round(points).astype(int)
                    elif len(points.shape) == 2 and points.shape[1] == 2:
                        points_int = np.round(points).astype(int).reshape((-1, 1, 2))
                    else:
                        log_message(
                            f"Skipping detection {detection.get('bbox')}: invalid mask format",
                            verbose=verbose,
                        )
                        continue

                    yolo_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    cv2.fillPoly(yolo_mask, [points_int], 255)
                    base_mask = _normalize_mask(yolo_mask)

                    (
                        final_mask,
                        clean_mask,
                        fill_color_bgr,
                        is_colored_bubble,
                        use_solid_midtone_fill,
                        sample_color_bgr,
                        text_bbox,
                        text_color_bgr,
                    ) = process_single_bubble(
                        base_mask,
                        img_gray,
                        img_height,
                        img_width,
                        thresholding_value,
                        use_otsu_threshold,
                        effective_roi_shrink_px,
                        verbose,
                        detection.get("bbox"),
                        is_sam=False,
                        dilation_kernel=dilation_kernel,
                        constraint_erosion_kernel=constraint_erosion_kernel,
                        min_contour_area=min_contour_area,
                        classify_colored=inpaint_colored_bubbles,
                        neighbor_bboxes=detection.get("conjoined_neighbor_bboxes"),
                        processing_scale=processing_scale,
                        image_bgr=lama_source_cv,
                        text_boxes=detection.get("text_boxes"),
                    )

                except Exception as e:
                    retry_success = False
                    if not use_otsu_threshold and base_mask is not None:
                        log_message(
                            f"Standard cleaning failed for {detection.get('bbox')}, retrying with Otsu...",
                            verbose=verbose,
                        )
                        retry_res = retry_cleaning_with_otsu(
                            lama_source_cv,
                            {
                                "base_mask": base_mask,
                                "bbox": detection.get("bbox"),
                                "is_sam": False,
                                "neighbor_bboxes": detection.get(
                                    "conjoined_neighbor_bboxes"
                                ),
                                "text_boxes": detection.get("text_boxes"),
                            },
                            thresholding_value,
                            roi_shrink_px,
                            processing_scale,
                            verbose,
                            inpaint_colored_bubbles,
                        )
                        if retry_res:
                            final_mask = retry_res["mask"]
                            clean_mask = retry_res.get("clean_mask", final_mask)
                            fill_color_bgr = retry_res["color"]
                            sample_color_bgr = retry_res["color"]
                            is_colored_bubble = retry_res["is_colored"]
                            use_solid_midtone_fill = retry_res.get(
                                "use_solid_midtone_fill", False
                            )
                            text_bbox = retry_res["text_bbox"]
                            text_color_bgr = retry_res.get("text_color_bgr")
                            retry_success = True
                            log_message(
                                f"Otsu retry successful for {detection.get('bbox')}",
                                verbose=verbose,
                            )
                        else:
                            log_message(
                                f"Otsu retry failed for {detection.get('bbox')}",
                                verbose=verbose,
                            )

                    if not retry_success:
                        error_msg = f"Error processing YOLO mask for detection {detection.get('bbox')}: {e}"
                        log_message(error_msg, always_print=True)
                        continue

            if final_mask is not None and fill_color_bgr is not None:
                if clean_mask is None:
                    clean_mask = final_mask
                clean_mask = _merge_conjoined_parent_clean_mask(
                    clean_mask,
                    detection.get("conjoined_parent_mask"),
                    final_mask,
                    base_mask,
                    img_gray,
                    fill_color_bgr,
                    is_colored_bubble,
                    thresholding_value,
                    constraint_erosion_kernel,
                    min_contour_area,
                    detection.get("bbox"),
                    verbose,
                )
                conjoined_parent_cleaning = bool(
                    detection.get("conjoined_parent_cleaning", False)
                )
                conjoined_parent_confident_white = False
                if conjoined_parent_cleaning and base_mask is not None:
                    conjoined_parent_confident_white = (
                        _is_confident_white_conjoined_parent(
                            img_gray,
                            base_mask,
                        )
                    )
                    if conjoined_parent_confident_white:
                        is_colored_bubble = False
                        use_solid_midtone_fill = False
                        fill_color_bgr = (255, 255, 255)
                        sample_color_bgr = (255, 255, 255)
                        text_color_bgr = (0, 0, 0)
                        text_only_mask = (
                            _build_conjoined_white_text_only_clean_mask(
                                base_mask,
                                img_gray,
                                detection.get("text_boxes"),
                            )
                        )
                        if text_only_mask is not None:
                            clean_mask = text_only_mask
                            log_message(
                                f"Detection {detection.get('bbox')}: using "
                                "trusted text-only clean mask for confident "
                                f"white conjoined parent "
                                f"({int(np.count_nonzero(clean_mask))} px)",
                                verbose=verbose,
                            )
                        else:
                            boundary_safe_mask = cv2.erode(
                                _normalize_mask(base_mask),
                                cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (5, 5)
                                ),
                                iterations=1,
                            )
                            clean_mask = cv2.bitwise_and(
                                _normalize_mask(clean_mask),
                                boundary_safe_mask,
                            )
                            log_message(
                                f"Detection {detection.get('bbox')}: protected "
                                "conjoined parent boundary; trusted text mask "
                                "was unavailable",
                                verbose=verbose,
                            )
                if (
                    not is_colored_bubble
                    and detection.get("conjoined_parent_mask") is None
                    and not conjoined_parent_confident_white
                ):
                    clean_mask_before_detector_guided = (
                        _normalize_mask(clean_mask).copy()
                        if clean_mask is not None
                        else None
                    )
                    clean_mask = _add_detector_guided_text_residual_mask(
                        clean_mask,
                        base_mask,
                        img_gray,
                        detection.get("text_boxes"),
                        fill_color_bgr,
                        detection.get("bbox"),
                        verbose,
                    )
                    layout_mask = _build_detector_guided_layout_mask(
                        final_mask,
                        clean_mask_before_detector_guided,
                        clean_mask,
                        detection.get("bbox"),
                        fill_color_bgr,
                        is_colored_bubble,
                        verbose,
                    )
                conjoined_child_overlay_only = bool(
                    detection.get("conjoined_child_overlay_only", False)
                )
                conjoined_child_lama_eligible = True
                parent_is_confident_white = False
                if conjoined_child_overlay_only and base_mask is not None:
                    parent_is_confident_white = (
                        _is_confident_white_conjoined_parent(
                            img_gray,
                            detection.get(
                                "conjoined_parent_mask_for_classification"
                            ),
                        )
                    )
                    if parent_is_confident_white:
                        conjoined_child_lama_eligible = False
                        is_colored_bubble = False
                        use_solid_midtone_fill = False
                        fill_color_bgr = (255, 255, 255)
                        sample_color_bgr = (255, 255, 255)
                        text_color_bgr = (0, 0, 0)
                        log_message(
                            f"Detection {detection.get('bbox')}: inherited "
                            "confident white conjoined parent cleaning",
                            verbose=verbose,
                        )
                    else:
                        child_pixels = img_gray[base_mask == 255]
                        conjoined_child_lama_eligible = bool(
                            child_pixels.size > 0
                            and float(np.mean(child_pixels))
                            <= CONJOINED_CHILD_LAMA_MAX_MEAN
                        )

                processed_bubbles.append(
                    {
                        "mask": final_mask,
                        "clean_mask": clean_mask,
                        "base_mask": base_mask,
                        "layout_mask": layout_mask,
                        "color": (
                            sample_color_bgr if sample_color_bgr else fill_color_bgr
                        ),
                        "bbox": detection.get("bbox"),
                        "is_colored": is_colored_bubble,
                        "use_solid_midtone_fill": use_solid_midtone_fill,
                        "text_bbox": text_bbox,
                        "text_color_bgr": text_color_bgr,
                        "text_boxes": detection.get("text_boxes") or [],
                        "is_sam": is_sam_mask,
                        "inpainted": False,
                        "conjoined_neighbor_bboxes": detection.get(
                            "conjoined_neighbor_bboxes"
                        )
                        or [],
                        "conjoined_child_overlay_only": conjoined_child_overlay_only,
                        "conjoined_child_lama_eligible": (
                            conjoined_child_lama_eligible
                        ),
                        "conjoined_parent_confident_white": (
                            parent_is_confident_white
                            or conjoined_parent_confident_white
                        ),
                    }
                )
                log_message(
                    f"Detection {detection.get('bbox')}: processed successfully",
                    verbose=verbose,
                )

        # LAMA receives a real mask, so source glyphs cannot be copied back into
        # the result as they can with image-to-image Flux compositing.
        combined_lama_mask = np.zeros(cleaned_image.shape[:2], dtype=np.uint8)
        combined_lama_protection_mask = np.zeros(
            cleaned_image.shape[:2], dtype=np.uint8
        )
        if inpaint_colored_bubbles:
            colored_bubbles = [
                bubble
                for bubble in processed_bubbles
                if bubble.get("is_colored", False)
                and (
                    not bubble.get("conjoined_child_overlay_only", False)
                    or bubble.get("conjoined_child_lama_eligible", False)
                )
            ]
            if colored_bubbles:
                for bubble_info in colored_bubbles:
                    is_translucent = False
                    detector_hole_seed = None
                    if bubble_info.get("use_solid_midtone_fill", False):
                        text_mask = bubble_info.get(
                            "clean_mask", bubble_info["mask"]
                        )
                        lama_domain = bubble_info.get(
                            "base_mask", bubble_info["mask"]
                        )
                    else:
                        text_color = bubble_info.get("text_color_bgr")
                        text_is_dark = (
                            text_color is None
                            or float(np.mean(text_color[:3])) < 128.0
                        )
                        (
                            is_translucent,
                            _translucent_sample,
                            translucent_stats,
                        ) = _detect_confident_translucent_background(
                            img_gray,
                            bubble_info.get("base_mask", bubble_info["mask"]),
                            bubble_info.get("text_boxes"),
                        )
                        base_lama_domain = np.where(
                            np.asarray(
                                bubble_info.get("base_mask", bubble_info["mask"])
                            )
                            > 0,
                            255,
                            0,
                        ).astype(np.uint8)
                        if base_lama_domain.ndim == 3:
                            base_lama_domain = base_lama_domain[..., 0]
                        lama_domain = base_lama_domain
                        detector_hole_seed = build_detector_text_hole_seed_mask(
                            base_lama_domain,
                            bubble_info.get("mask", base_lama_domain),
                            bubble_info.get("text_boxes"),
                            bubble_info.get("bbox"),
                        )
                        text_mask = build_lama_text_seed_mask(
                            img_gray,
                            lama_domain,
                            bubble_info.get("text_boxes"),
                            text_is_dark=text_is_dark,
                            protect_background_artwork=is_translucent,
                        )
                        if is_translucent:
                            opposite_mask = build_lama_text_seed_mask(
                                img_gray,
                                lama_domain,
                                bubble_info.get("text_boxes"),
                                text_is_dark=not text_is_dark,
                                protect_background_artwork=True,
                            )
                            text_mask = cv2.bitwise_or(text_mask, opposite_mask)
                            text_mask = _complete_bright_translucent_text_seed(
                                img_gray,
                                lama_domain,
                                text_mask,
                                translucent_stats,
                                text_is_dark=text_is_dark,
                                detection_bbox=bubble_info.get("bbox"),
                                verbose=verbose,
                            )
                        if np.any(detector_hole_seed):
                            text_mask = cv2.bitwise_or(
                                text_mask,
                                detector_hole_seed,
                            )
                        outline_mask = build_lama_contrast_outline_seed(
                            img_gray,
                            lama_domain,
                            text_mask,
                            text_is_dark=text_is_dark,
                        )
                        text_mask = cv2.bitwise_or(text_mask, outline_mask)
                        if not np.any(text_mask):
                            text_mask = bubble_info.get(
                                "clean_mask", bubble_info["mask"]
                            )
                    lama_dilation_offset = colored_bubble_mask_dilation_offset
                    if detector_hole_seed is not None and np.any(
                        detector_hole_seed
                    ):
                        lama_dilation_offset = 0
                        log_message(
                            "Using detector-guided source-hole LAMA seed for "
                            f"{tuple(bubble_info.get('bbox') or ())} "
                            f"({np.count_nonzero(detector_hole_seed)} px)",
                            always_print=True,
                        )
                    lama_mask = build_lama_inpaint_mask(
                        text_mask,
                        lama_domain,
                        colored_bubble_mask_kernel_size,
                        lama_dilation_offset,
                    )
                    if is_translucent:
                        lama_mask, mask_limit_stats = (
                            _limit_overbroad_translucent_lama_mask(
                                lama_mask,
                                text_mask,
                                lama_domain,
                            )
                        )
                        if (
                            mask_limit_stats
                            and mask_limit_stats["after_ratio"]
                            < mask_limit_stats["before_ratio"] - 0.01
                        ):
                            log_message(
                                "Limited overbroad translucent LAMA mask for "
                                f"{tuple(bubble_info.get('bbox') or ())} "
                                f"({mask_limit_stats['before_ratio']:.2f} -> "
                                f"{mask_limit_stats['after_ratio']:.2f}, "
                                f"seed={mask_limit_stats['seed_ratio']:.2f})",
                                verbose=verbose,
                            )
                    bubble_info["lama_inpaint_mask"] = lama_mask
                    combined_lama_mask = cv2.bitwise_or(
                        combined_lama_mask, lama_mask
                    )

                if np.any(combined_lama_mask):
                    log_message(
                        f"Inpainting {len(colored_bubbles)} colored bubbles "
                        "with LAMA Large",
                        always_print=True,
                    )
                    try:
                        pil_working = Image.fromarray(
                            cv2.cvtColor(lama_source_cv, cv2.COLOR_BGR2RGB)
                        )
                        pil_working = run_lama_large(
                            pil_working,
                            combined_lama_mask,
                            device=device,
                            inpainting_size=colored_bubble_inpainting_size,
                            precision=colored_bubble_inpainting_precision,
                            verbose=verbose,
                        )
                        lama_result = cv2.cvtColor(
                            np.asarray(pil_working.convert("RGB")),
                            cv2.COLOR_RGB2BGR,
                        )
                        lama_pixels = combined_lama_mask > 0
                        if cleaned_image.shape[2] == 4:
                            cleaned_image[lama_pixels, :3] = lama_result[lama_pixels]
                        else:
                            cleaned_image[lama_pixels] = lama_result[lama_pixels]
                        for bubble_info in colored_bubbles:
                            bubble_info["inpainted"] = True
                            if bubble_info.get(
                                "conjoined_child_overlay_only", False
                            ):
                                child_base_mask = bubble_info.get("base_mask")
                                if child_base_mask is not None:
                                    combined_lama_protection_mask = cv2.bitwise_or(
                                        combined_lama_protection_mask,
                                        _normalize_mask(child_base_mask),
                                    )
                            mask_bool = bubble_info["lama_inpaint_mask"] > 0
                            masked_after = cleaned_image[mask_bool]
                            if masked_after.ndim == 2 and masked_after.shape[1] > 3:
                                masked_after = masked_after[:, :3]
                            if masked_after.size > 0:
                                mean_bgr = np.clip(
                                    np.mean(masked_after, axis=0), 0, 255
                                ).astype(int)
                                bubble_info["color"] = tuple(
                                    int(value) for value in mean_bgr
                                )
                        combined_lama_protection_mask = cv2.bitwise_or(
                            combined_lama_protection_mask, combined_lama_mask
                        )
                    except Exception as error:
                        log_message(
                            "LAMA Large colored-bubble inpainting failed; "
                            f"falling back to sampled fill: {error}",
                            always_print=True,
                        )

        # Group masks by color for efficient batch processing (skip already inpainted regions)
        if processed_bubbles:
            color_groups = {}
            for bubble_info in processed_bubbles:
                if bubble_info.get("inpainted", False):
                    continue
                if bubble_info.get("conjoined_child_overlay_only", False):
                    continue
                color_key = bubble_info["color"]
                if color_key not in color_groups:
                    color_groups[color_key] = []
                color_groups[color_key].append(
                    bubble_info.get("clean_mask", bubble_info["mask"])
                )

            for color_bgr, masks in color_groups.items():
                combined_mask = np.bitwise_or.reduce(masks)
                if np.any(combined_lama_protection_mask):
                    combined_mask = cv2.bitwise_and(
                        combined_mask,
                        cv2.bitwise_not(combined_lama_protection_mask),
                    )

                if cleaned_image.shape[2] == 4:
                    cleaned_image[combined_mask == 255, :3] = (
                        color_bgr  # Preserve alpha channel
                    )
                else:
                    cleaned_image[combined_mask == 255] = color_bgr

        log_message(
            f"Cleaned {len(processed_bubbles)} speech bubbles", always_print=True
        )
        return cleaned_image, processed_bubbles
    except IOError as e:
        raise ImageProcessingError(f"Error loading image {image_input}: {str(e)}")
    except Exception as e:
        raise CleaningError(f"Error cleaning speech bubbles: {str(e)}")


def retry_cleaning_with_otsu(
    image_bgr: np.ndarray,
    bubble_info: dict,
    thresholding_value: int,
    roi_shrink_px: int,
    processing_scale: float = 1.0,
    verbose: bool = False,
    classify_colored: bool = False,
) -> Optional[dict]:
    """
    Retry cleaning for a single bubble using Otsu thresholding.

    Returns a bubble-info dict compatible with clean_speech_bubbles output,
    or None if retry fails.
    """
    base_mask = bubble_info.get("base_mask")
    if base_mask is None:
        log_message(
            f"Otsu retry skipped for {bubble_info.get('bbox')}: missing base_mask",
            verbose=verbose,
        )
        return None

    try:
        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 4:
            img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        log_message(
            f"Otsu retry failed to convert image to grayscale: {e}",
            always_print=True,
        )
        return None

    img_height, img_width = img_gray.shape[:2]

    effective_roi_shrink_px = float(
        scale_scalar(
            roi_shrink_px,
            processing_scale,
            minimum=0.0,
            maximum=64.0,
        )
    )
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, scale_kernel(DILATION_KERNEL_SIZE, processing_scale)
    )
    constraint_erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, scale_kernel(EROSION_KERNEL_SIZE, processing_scale)
    )
    min_contour_area = scale_area(
        MIN_CONTOUR_AREA,
        processing_scale,
        minimum=MIN_CONTOUR_AREA,
        maximum=5000,
    )

    try:
        result = process_single_bubble(
            base_mask,
            img_gray,
            img_height,
            img_width,
            thresholding_value,
            True,  # force Otsu
            effective_roi_shrink_px,
            verbose,
            bubble_info.get("bbox"),
            bubble_info.get("is_sam", False),
            dilation_kernel=dilation_kernel,
            constraint_erosion_kernel=constraint_erosion_kernel,
            min_contour_area=min_contour_area,
            classify_colored=classify_colored,
            neighbor_bboxes=bubble_info.get("neighbor_bboxes"),
            processing_scale=processing_scale,
            image_bgr=image_bgr,
            text_boxes=bubble_info.get("text_boxes"),
        )
    except CleaningError as e:
        log_message(
            f"Otsu retry cleaning failed for {bubble_info.get('bbox')}: {e}",
            always_print=True,
        )
        return None
    except Exception as e:
        log_message(
            f"Otsu retry cleaning unexpected error for {bubble_info.get('bbox')}: {e}",
            always_print=True,
        )
        return None

    if not result:
        return None

    (
        final_mask,
        clean_mask,
        fill_color_bgr,
        is_colored_bubble,
        use_solid_midtone_fill,
        sample_color_bgr,
        text_bbox,
        text_color_bgr,
    ) = result

    bubble_color = sample_color_bgr if sample_color_bgr else fill_color_bgr

    log_message(
        f"Otsu retry succeeded for {bubble_info.get('bbox')}",
        verbose=verbose,
    )

    return {
        "mask": final_mask,
        "clean_mask": clean_mask,
        "base_mask": _normalize_mask(base_mask),
        "color": bubble_color,
        "bbox": bubble_info.get("bbox"),
        "is_colored": is_colored_bubble,
        "use_solid_midtone_fill": use_solid_midtone_fill,
        "text_bbox": text_bbox,
        "text_color_bgr": text_color_bgr,
        "is_sam": bubble_info.get("is_sam", False),
    }
