import base64
import gc
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from core.config import MangaTranslatorConfig, normalize_outside_text_processing_mode
from core.image.image_utils import cv2_to_pil, pil_to_cv2, process_bubble_image_cached
from core.image.inpaint_composite import finish_outside_inpaint
from core.image.inpainting import FluxKleinInpainter, FluxKontextInpainter
from core.image.ocr_detection import OutsideTextDetector, extract_text_with_manga_ocr
from core.ml.model_manager import get_model_manager
from utils.exceptions import CancellationError, TranslationError
from utils.logging import log_message

# OSB Expansion Parameters
OSB_EXPANSION_PIXEL_BUFFER = 5  # for bubbles, nearby OSB regions, panels
OSB_BUBBLE_IOA_FILTER_THRESHOLD = 0.35
OSB_BUBBLE_MASK_COVERAGE_THRESHOLD = 0.35
OSB_BUBBLE_MASK_OUTSIDE_RESCUE_MAX_COVERAGE = 0.12
OSB_BUBBLE_MASK_OUTSIDE_RESCUE_MIN_IOA = 0.10
OSB_PARTIAL_SFX_RESCUE_MAX_MASK_COVERAGE = 0.45
OSB_PARTIAL_SFX_RESCUE_MIN_BUBBLE_IOA = 0.10
OSB_PARTIAL_SFX_RESCUE_MAX_BUBBLE_IOA = 0.55
OSB_PARTIAL_SFX_RESCUE_MIN_ASPECT = 1.65
OSB_PARTIAL_SFX_RESCUE_MAX_HEIGHT_RATIO = 0.12
OSB_EDGE_SFX_RESCUE_MIN_BUBBLE_IOA = 0.35
OSB_EDGE_SFX_RESCUE_MAX_AREA_RATIO = 0.16
OSB_EDGE_SFX_RESCUE_MAX_WIDTH_RATIO = 0.46
OSB_EDGE_SFX_RESCUE_MAX_HEIGHT_RATIO = 0.50
OSB_EDGE_SFX_RESCUE_EDGE_BAND_RATIO = 0.24
OSB_EDGE_SFX_RESCUE_MAX_EDGE_GAP_RATIO = 0.12
OSB_EDGE_SFX_RESCUE_MAX_EDGE_GAP_PX = 14.0
OSB_EDGE_SFX_RESCUE_MAX_MASK_COVERAGE = 0.45
OSB_EDGE_SFX_GLYPH_BASE_PADDING = 3
OSB_EDGE_SFX_GLYPH_INWARD_PADDING = 10
OSB_EDGE_SFX_GLYPH_OBSTACLE_GAP = 2
OSB_ROTATION_MIN_ABS_DEGREES = 2.0
OSB_HORIZONTAL_ROTATION_PROJECTION_MIN_GAIN = 1.06
OSB_HORIZONTAL_ROTATION_PROJECTION_MIN_PEAK_GAIN = 1.03
OSB_HORIZONTAL_ROTATION_PEAK_MIN_SCORE_GAIN = 1.0
OSB_SFX_STRONG_AXIS_MIN_ABS_DEGREES = 10.0
OSB_SFX_STRONG_AXIS_MIN_DELTA_DEGREES = 3.0
OSB_BUFFER_CONFLICT_FLUX_DILATION = 10
OSB_BUFFER_CONFLICT_CLIP_PADDING = 4
OSB_BUFFER_CONFLICT_WIDE_SFX_MIN_ASPECT = 2.4
OSB_BUFFER_CONFLICT_WIDE_SFX_MIN_WIDTH_RATIO = 0.12
OSB_BUFFER_CONFLICT_WIDE_SFX_MAX_HEIGHT_RATIO = 0.12
OSB_BUFFER_CONFLICT_WIDE_SFX_MAX_AREA_RATIO = 0.08
OSB_LARGE_COMPLEX_FILL_MIN_AREA_RATIO = 0.12
OSB_LARGE_COMPLEX_FILL_MIN_WIDTH_RATIO = 0.30
OSB_LARGE_COMPLEX_FILL_MIN_HEIGHT_RATIO = 0.25
OSB_LARGE_COMPLEX_FILL_MAX_STROKE_COVERAGE = 0.16
OSB_SPEECH_EDGE_RECT_MIN_PIXELS = 32
OSB_SPEECH_EDGE_RECT_MIN_RAW_MASK_GAIN = 1.15
OSB_SPEECH_EDGE_RECT_MIN_BBOX_COVERAGE = 0.08
OSB_SPEECH_EDGE_RECT_PAD_X = 3
OSB_SPEECH_EDGE_RECT_PAD_Y = 3
OSB_COLORED_POLARITY_MIN_SAT_P75 = 28.0
OSB_COLORED_POLARITY_MIN_CHROMA_MEAN = 14.0
OSB_COLORED_POLARITY_SCORE_RATIO = 2.25
OSB_COLORED_POLARITY_SCORE_DELTA = 0.18
OSB_HORIZONTAL_CENTERLINE_MIN_COMPONENTS = 8
OSB_HORIZONTAL_CENTERLINE_MIN_ABS_DEGREES = 5.0
OSB_HORIZONTAL_CENTERLINE_MAX_ABS_DEGREES = 15.0
OSB_HORIZONTAL_CENTERLINE_MIN_ASPECT = 1.8
OSB_HORIZONTAL_CENTERLINE_MIN_SPAN_RATIO = 0.55


def _normalize_osb_rotation(angle_deg: float) -> float:
    angle = float(angle_deg)
    while angle <= -90.0:
        angle += 180.0
    while angle > 90.0:
        angle -= 180.0
    return angle


def _clamp_osb_rotation(angle_deg: float, max_abs_degrees: float) -> float:
    try:
        max_abs = max(0.0, min(45.0, float(max_abs_degrees)))
    except (TypeError, ValueError):
        max_abs = 0.0
    if max_abs <= 0.0:
        return 0.0

    angle = _normalize_osb_rotation(angle_deg)
    if abs(angle) < OSB_ROTATION_MIN_ABS_DEGREES:
        return 0.0
    return float(max(-max_abs, min(max_abs, angle)))


def _osb_horizontal_projection_metrics(mask: np.ndarray) -> Tuple[float, float]:
    if mask is None or mask.size == 0:
        return 0.0, 0.0

    row_counts = np.count_nonzero(mask > 0, axis=1).astype(np.float64)
    total = float(row_counts.sum())
    if total <= 0.0:
        return 0.0, 0.0

    concentration = float(np.sum(row_counts * row_counts) / (total * total))
    peak_ratio = float(row_counts.max() / total)
    return concentration, peak_ratio


def _rotate_osb_binary_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return mask
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    return cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _horizontal_projection_supports_osb_rotation(
    mask: np.ndarray,
    rotation_deg: float,
) -> bool:
    if mask is None or mask.size == 0 or abs(float(rotation_deg)) < OSB_ROTATION_MIN_ABS_DEGREES:
        return False

    base_score, base_peak = _osb_horizontal_projection_metrics(mask)
    if base_score <= 0.0:
        return False

    # Try both signs because the renderer's rotation sign and the image-space
    # deskew sign differ between text families. A real tilted horizontal line
    # should become more row-concentrated in one of these two directions.
    rotated_scores = [
        _osb_horizontal_projection_metrics(
            _rotate_osb_binary_mask(mask, -float(rotation_deg))
        ),
        _osb_horizontal_projection_metrics(
            _rotate_osb_binary_mask(mask, float(rotation_deg))
        ),
    ]
    best_score, best_peak = max(rotated_scores, key=lambda item: (item[0], item[1]))
    if best_score >= base_score * OSB_HORIZONTAL_ROTATION_PROJECTION_MIN_GAIN:
        return True

    return (
        best_peak >= base_peak * OSB_HORIZONTAL_ROTATION_PROJECTION_MIN_PEAK_GAIN
        and best_score >= base_score * OSB_HORIZONTAL_ROTATION_PEAK_MIN_SCORE_GAIN
    )


def _fill_osb_mask_holes(mask: np.ndarray) -> np.ndarray:
    filled = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled


def _large_complex_osb_stroke_fill_mask(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    total_bubble_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    raw_w = x2 - x1
    raw_h = y2 - y1
    if raw_w <= 0 or raw_h <= 0:
        return None
    area_ratio = (raw_w * raw_h) / float(max(1, img_w * img_h))
    if area_ratio < OSB_LARGE_COMPLEX_FILL_MIN_AREA_RATIO:
        return None
    if raw_w / float(max(1, img_w)) < OSB_LARGE_COMPLEX_FILL_MIN_WIDTH_RATIO:
        return None
    if raw_h / float(max(1, img_h)) < OSB_LARGE_COMPLEX_FILL_MIN_HEIGHT_RATIO:
        return None

    try:
        gray = np.asarray(image.crop((x1, y1, x2, y2)).convert("L"))
    except Exception:
        return None
    if gray.shape[:2] != (raw_h, raw_w) or gray.size == 0:
        return None

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
        candidate = cv2.morphologyEx(
            candidate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    except Exception:
        return None

    crop_area = float(max(1, raw_w * raw_h))
    min_area = max(60.0, crop_area * 0.00035)
    max_area = crop_area * 0.04
    kept = np.zeros((raw_h, raw_w), dtype=np.uint8)
    kept_count = 0
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
        if width <= 16 and height >= raw_h * 0.20:
            continue
        if height <= 12 and width >= raw_w * 0.20:
            continue
        try:
            component = labels == label_idx
            ring = np.logical_and(
                cv2.dilate(
                    component.astype(np.uint8),
                    np.ones((13, 13), np.uint8),
                    iterations=1,
                ).astype(bool),
                np.logical_not(component),
            )
            ring_size = int(np.count_nonzero(ring))
            if ring_size > 0:
                bright_ring_ratio = float(
                    np.count_nonzero(np.logical_and(ring, bright))
                ) / float(ring_size)
                if bright_ring_ratio < 0.30:
                    continue
        except Exception:
            pass
        touches_edge = (
            left <= 1 or top <= 1 or left + width >= raw_w - 1 or top + height >= raw_h - 1
        )
        if touches_edge and (width <= 14 or height <= 14):
            continue
        cx = left + width / 2.0
        cy = top + height / 2.0
        if area < 180 and (cx < raw_w * 0.18 or cy > raw_h * 0.88):
            continue
        kept[labels == label_idx] = 1
        kept_count += 1

    if kept_count < 2:
        return None
    try:
        kept = cv2.dilate(kept, np.ones((5, 5), np.uint8), iterations=1)
        kept = cv2.morphologyEx(
            kept, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )
    except Exception:
        pass

    coverage = np.count_nonzero(kept) / crop_area
    if coverage <= 0.002 or coverage > OSB_LARGE_COMPLEX_FILL_MAX_STROKE_COVERAGE:
        return None

    full_mask = np.zeros((img_h, img_w), dtype=bool)
    full_mask[y1:y2, x1:x2] = kept.astype(bool)
    if total_bubble_mask is not None and total_bubble_mask.shape == full_mask.shape:
        full_mask = np.logical_and(full_mask, np.logical_not(total_bubble_mask))
    if not np.any(full_mask):
        return None
    return full_mask


def _build_osb_render_mask(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    raw_bbox: Tuple[int, int, int, int],
    dark_background: bool,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    clipped = _to_int_bbox(bbox, img_w, img_h)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None

    if dark_background:
        threshold = max(45.0, min(125.0, float(np.percentile(gray, 42))))
        local_mask = (gray <= threshold).astype(np.uint8)
        if np.count_nonzero(local_mask) < max(20, int(w * h * 0.01)):
            local_mask = (gray <= 145).astype(np.uint8)
    else:
        threshold = min(245.0, max(165.0, float(np.percentile(gray, 58))))
        local_mask = (gray >= threshold).astype(np.uint8)
        if np.count_nonzero(local_mask) < max(20, int(w * h * 0.01)):
            local_mask = (gray >= 145).astype(np.uint8)

    local_mask = cv2.morphologyEx(
        local_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    local_mask = cv2.morphologyEx(
        local_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(local_mask, 8)
    if num_labels <= 1:
        return None

    rx1, ry1, rx2, ry2 = raw_bbox
    local_raw = (
        max(0, rx1 - x1),
        max(0, ry1 - y1),
        min(w, rx2 - x1),
        min(h, ry2 - y1),
    )
    crop_area = float(max(1, w * h))
    best_label = None
    best_score = -1.0
    largest_label = None
    largest_area = 0.0

    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if area < max(12.0, crop_area * 0.001) or area > crop_area * 0.98:
            continue
        if area > largest_area:
            largest_area = area
            largest_label = label_idx

        overlap = 0
        if local_raw[2] > local_raw[0] and local_raw[3] > local_raw[1]:
            overlap = int(
                np.count_nonzero(
                    labels[
                        local_raw[1] : local_raw[3],
                        local_raw[0] : local_raw[2],
                    ]
                    == label_idx
                )
            )
        label_cx = float(stats[label_idx, cv2.CC_STAT_LEFT]) + float(
            stats[label_idx, cv2.CC_STAT_WIDTH]
        ) / 2.0
        label_cy = float(stats[label_idx, cv2.CC_STAT_TOP]) + float(
            stats[label_idx, cv2.CC_STAT_HEIGHT]
        ) / 2.0
        target_cx = ((rx1 + rx2) / 2.0) - x1
        target_cy = ((ry1 + ry2) / 2.0) - y1
        distance_penalty = (
            ((label_cx - target_cx) ** 2 + (label_cy - target_cy) ** 2) ** 0.5
        ) / max(1.0, (w * w + h * h) ** 0.5)
        score = overlap * 1000.0 + area * (1.0 - min(0.9, distance_penalty))
        if score > best_score:
            best_score = score
            best_label = label_idx

    if best_label is None:
        best_label = largest_label
    if best_label is None:
        return None

    selected = (labels == best_label).astype(np.uint8)
    selected = _fill_osb_mask_holes(selected)
    if np.count_nonzero(selected) < max(20, int(w * h * 0.01)):
        return None

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = selected * 255
    return full_mask


def _estimate_osb_vertical_band_angle(
    mask: np.ndarray,
    max_abs_degrees: float,
    use_stroke_angle: bool = False,
) -> float:
    mask = np.asarray(mask) > 0
    ys, xs = np.where(mask)
    if xs.size < 20:
        return 0.0

    h, w = mask.shape[:2]
    candidates: List[float] = []

    try:
        band_count = max(4, min(12, int(round(h / 14.0))))
        band_edges = np.linspace(0, h, band_count + 1).astype(int)
        centers: List[Tuple[float, float, float]] = []
        for band_idx in range(band_count):
            y0 = int(band_edges[band_idx])
            y1 = int(band_edges[band_idx + 1])
            if y1 <= y0:
                continue
            band = mask[y0:y1, :]
            by, bx = np.where(band)
            if bx.size < max(4, int((y1 - y0) * w * 0.006)):
                continue
            centers.append(
                (
                    float(np.median(bx)),
                    float(y0 + np.median(by)),
                    float(max(1, bx.size)),
                )
            )
        if len(centers) >= 3:
            center_arr = np.asarray(centers, dtype=np.float32)
            center_x = center_arr[:, 0]
            center_y = center_arr[:, 1]
            center_w = np.sqrt(center_arr[:, 2])
            span_y = float(center_y.max() - center_y.min())
            span_x = float(center_x.max() - center_x.min())
            if span_y >= max(12.0, h * 0.35):
                slope, intercept = np.polyfit(center_y, center_x, 1, w=center_w)
                predicted = center_y * slope + intercept
                residual = float(np.median(np.abs(center_x - predicted)))
                if residual <= max(6.0, span_x * 0.45):
                    candidates.append(float(np.degrees(np.arctan(float(slope)))))
    except Exception:
        pass

    try:
        top_limit = max(1, int(round(h * 0.38)))
        bottom_start = min(h - 1, int(round(h * 0.62)))
        top_y, top_x = np.where(mask[:top_limit, :])
        bottom_y, bottom_x = np.where(mask[bottom_start:, :])
        if top_x.size >= 8 and bottom_x.size >= 8:
            top_cx = float(np.median(top_x))
            top_cy = float(np.median(top_y))
            bottom_cx = float(np.median(bottom_x))
            bottom_cy = float(bottom_start + np.median(bottom_y))
            dy = bottom_cy - top_cy
            if dy >= max(12.0, h * 0.35):
                candidates.append(float(np.degrees(np.arctan((bottom_cx - top_cx) / dy))))
    except Exception:
        pass

    if use_stroke_angle:
        try:
            aspect_ratio = h / float(max(1, w))
            if 1.20 <= aspect_ratio <= 2.60:
                component_mask = np.zeros_like(mask, dtype=np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), 8
                )
                crop_area = float(max(1, w * h))
                for label_idx in range(1, num_labels):
                    area = float(stats[label_idx, cv2.CC_STAT_AREA])
                    left = int(stats[label_idx, cv2.CC_STAT_LEFT])
                    top = int(stats[label_idx, cv2.CC_STAT_TOP])
                    comp_w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
                    comp_h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
                    right = left + comp_w
                    bottom = top + comp_h
                    touches_border = (
                        left <= 1 or top <= 1 or right >= w - 1 or bottom >= h - 1
                    )
                    if area < 4.0:
                        continue
                    if touches_border and area > max(30.0, crop_area * 0.004):
                        continue
                    component_mask[labels == label_idx] = 255

                if int(np.count_nonzero(component_mask)) >= 30:
                    edges = cv2.Canny(component_mask, 50, 150, apertureSize=3)
                    lines = cv2.HoughLinesP(
                        edges,
                        1,
                        np.pi / 180.0,
                        threshold=8,
                        minLineLength=max(8, int(round(min(h, w) * 0.12))),
                        maxLineGap=3,
                    )
                    stroke_candidates: List[Tuple[float, float]] = []
                    if lines is not None:
                        for line in lines[:, 0, :]:
                            lx1, ly1, lx2, ly2 = [float(v) for v in line]
                            length = float(((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2) ** 0.5)
                            if length < 8.0:
                                continue
                            line_angle = float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
                            while line_angle <= -90.0:
                                line_angle += 180.0
                            while line_angle > 90.0:
                                line_angle -= 180.0
                            # Hough gives the visual stroke direction. Convert it to
                            # the renderer's vertical-text rotation sign; this is the
                            # opposite of the centerline slope sign used above.
                            source_angle = _normalize_osb_rotation(line_angle - 90.0)
                            if 6.0 <= abs(source_angle) <= max_abs_degrees:
                                stroke_candidates.append((source_angle, length))
                    if stroke_candidates:
                        total_weight = sum(weight for _, weight in stroke_candidates)
                        if total_weight > 0.0:
                            stroke_angle = sum(
                                angle * weight for angle, weight in stroke_candidates
                            ) / total_weight
                            candidates.append(stroke_angle)
        except Exception:
            pass

    try:
        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        if points.shape[0] >= 20:
            line_points = points.reshape(-1, 1, 2)
            vx, vy, _, _ = cv2.fitLine(
                line_points, cv2.DIST_L2, 0, 0.01, 0.01
            ).flatten()
            if abs(float(vy)) > 1e-4:
                candidates.append(
                    float(np.degrees(np.arctan2(float(vx), float(vy))))
                )
    except Exception:
        pass

    try:
        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        centered = points - np.mean(points, axis=0)
        if centered.shape[0] >= 20:
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] >= 1.08:
                major = eigvecs[:, order[0]]
                angle_from_x = float(np.degrees(np.arctan2(major[1], major[0])))
                if angle_from_x < 0.0:
                    angle_from_x += 180.0
                candidates.append(_normalize_osb_rotation(90.0 - angle_from_x))
    except Exception:
        pass

    if not candidates:
        return 0.0
    clamped_candidates = [
        _clamp_osb_rotation(float(candidate), max_abs_degrees)
        for candidate in candidates
    ]
    clamped_candidates = [
        candidate for candidate in clamped_candidates if candidate != 0.0
    ]
    if not clamped_candidates:
        return 0.0
    positives = [candidate for candidate in clamped_candidates if candidate > 0]
    negatives = [candidate for candidate in clamped_candidates if candidate < 0]
    if positives and negatives:
        return max(clamped_candidates, key=lambda candidate: abs(candidate))
    median_angle = float(np.median(clamped_candidates))
    strongest_angle = max(clamped_candidates, key=lambda candidate: abs(candidate))
    if (
        use_stroke_angle
        and abs(strongest_angle) >= 8.0
        and abs(strongest_angle) >= abs(median_angle) + 6.0
    ):
        return strongest_angle
    if abs(median_angle) < 4.0 and abs(strongest_angle) >= 6.0:
        return strongest_angle
    return median_angle


def _estimate_osb_multicolumn_vertical_rotation(
    mask: np.ndarray,
    max_abs_degrees: float,
) -> Optional[float]:
    """Estimate vertical text tilt per column before trying a whole-mask fit.

    Multi-column upright Japanese text can look diagonal when all ink pixels are
    fitted at once: row medians jump from one column to the next.  In that case
    the reliable signal is the slope inside each separated column.
    """
    try:
        mask_bool = np.asarray(mask) > 0
        h, w = mask_bool.shape[:2]
    except Exception:
        return None

    if h < 24 or w < 16 or int(np.count_nonzero(mask_bool)) < 40:
        return None

    col_counts = np.count_nonzero(mask_bool, axis=0)
    min_col_pixels = max(1, int(round(h * 0.012)))
    active_cols = col_counts >= min_col_pixels
    runs: List[List[int]] = []
    start: Optional[int] = None
    for col, active in enumerate(active_cols):
        if active and start is None:
            start = col
        is_last = col == w - 1
        if start is not None and ((not active) or is_last):
            end = col if not active else col + 1
            runs.append([start, end])
            start = None

    if not runs:
        return None

    max_merge_gap = max(2, int(round(w * 0.045)))
    merged: List[List[int]] = []
    for run in runs:
        if not merged or run[0] - merged[-1][1] > max_merge_gap:
            merged.append(run)
        else:
            merged[-1][1] = run[1]

    columns: List[Tuple[float, float, float, float]] = []
    for col_start, col_end in merged:
        col_start = max(0, int(col_start))
        col_end = min(w, int(col_end))
        if col_end <= col_start:
            continue
        column = mask_bool[:, col_start:col_end]
        ys, xs = np.where(column)
        pixel_count = int(xs.size)
        if pixel_count < max(20, int(h * w * 0.004)):
            continue

        span_h = int(ys.max() - ys.min() + 1)
        span_w = int(xs.max() - xs.min() + 1)
        if span_h < max(18, int(round(h * 0.28))) or span_w < 2:
            continue

        ys_for_fit: List[float] = []
        xs_for_fit: List[float] = []
        min_row_pixels = max(1, int(round(max(1, span_w) * 0.05)))
        for row in range(max(0, int(ys.min())), min(h, int(ys.max()) + 1)):
            row_xs = np.where(column[row] > 0)[0]
            if row_xs.size < min_row_pixels:
                continue
            ys_for_fit.append(float(row))
            xs_for_fit.append(float(col_start + np.median(row_xs)))

        if len(xs_for_fit) < max(8, int(round(span_h * 0.12))):
            continue

        try:
            y_arr = np.asarray(ys_for_fit, dtype=np.float32)
            x_arr = np.asarray(xs_for_fit, dtype=np.float32)
            slope, intercept = np.polyfit(y_arr, x_arr, 1)
            predicted = y_arr * slope + intercept
            residual = float(np.median(np.abs(x_arr - predicted)))
        except Exception:
            continue

        residual_limit = max(2.5, min(8.0, float(max(1, span_w)) * 0.35))
        if residual > residual_limit:
            continue

        angle = float(np.degrees(np.arctan(float(slope))))
        columns.append((angle, float(pixel_count), float(col_start), float(col_end)))

    if len(columns) < 2:
        return None

    # Require a real gutter between at least two usable columns.  Without this,
    # a single wide glyph or connected SFX stroke should fall back to the old
    # whole-mask estimators.
    separated_pair = False
    sorted_columns = sorted(columns, key=lambda item: item[2])
    min_gutter = max(4.0, float(w) * 0.10)
    for left, right in zip(sorted_columns, sorted_columns[1:]):
        if right[2] - left[3] >= min_gutter:
            separated_pair = True
            break
    if not separated_pair:
        return None

    angles = np.asarray([item[0] for item in columns], dtype=np.float32)
    weights = np.asarray([max(1.0, item[1]) for item in columns], dtype=np.float32)
    median_angle = float(np.median(angles))
    max_deviation = float(np.max(np.abs(angles - median_angle)))

    if abs(median_angle) < OSB_ROTATION_MIN_ABS_DEGREES:
        return 0.0

    if max_deviation > 7.0:
        return 0.0

    weighted_angle = float(np.average(angles, weights=weights))
    return _clamp_osb_rotation(weighted_angle, max_abs_degrees)


def _osb_crop_has_colored_rotation_background(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0:
        return False
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        sat_p75 = float(np.percentile(saturation, 75))
        chroma = (
            np.max(crop.astype(np.int16), axis=2)
            - np.min(crop.astype(np.int16), axis=2)
        ).astype(np.float32)
        chroma_mean = float(np.mean(chroma))
        return (
            sat_p75 >= OSB_COLORED_POLARITY_MIN_SAT_P75
            and chroma_mean >= OSB_COLORED_POLARITY_MIN_CHROMA_MEAN
        )
    except Exception:
        return False


def _build_osb_rotation_mask_candidate(
    gray: np.ndarray,
    bright_text: bool,
) -> Tuple[np.ndarray, float, Dict[str, float]]:
    h, w = gray.shape[:2]
    if bright_text:
        threshold = max(145.0, float(np.percentile(gray, 78)))
        mask = (gray >= threshold).astype(np.uint8)
        if np.count_nonzero(mask) < max(20, int(w * h * 0.002)):
            mask = (gray >= 130).astype(np.uint8)
    else:
        threshold = min(115.0, float(np.percentile(gray, 22)))
        mask = (gray <= threshold).astype(np.uint8)
        if np.count_nonzero(mask) < max(20, int(w * h * 0.002)):
            mask = (gray <= 135).astype(np.uint8)

    if np.count_nonzero(mask) < max(20, int(w * h * 0.001)):
        return mask, 0.0, {
            "pixel_ratio": 0.0,
            "span_y_ratio": 0.0,
            "dominant_share": 1.0,
            "dominant_edge": 1.0,
        }

    raw_mask = mask
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if np.count_nonzero(opened_mask) >= max(20, int(w * h * 0.001)):
        mask = opened_mask
    else:
        mask = raw_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    dominant_share = 0.0
    dominant_edge = 0.0
    if num_labels > 1:
        kept = np.zeros_like(mask)
        crop_area = float(max(1, w * h))
        min_area = max(4.0, crop_area * 0.00008)
        max_area = crop_area * 0.35
        kept_areas: List[float] = []
        kept_edge_flags: List[float] = []
        for label_idx in range(1, num_labels):
            area = float(stats[label_idx, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                kept[labels == label_idx] = 1
                kept_areas.append(area)
                comp_y = float(stats[label_idx, cv2.CC_STAT_TOP])
                comp_h = float(stats[label_idx, cv2.CC_STAT_HEIGHT])
                edge_touch = comp_y <= h * 0.05 or (comp_y + comp_h) >= h * 0.95
                short_edge_blob = comp_h <= h * 0.28 and edge_touch
                kept_edge_flags.append(1.0 if short_edge_blob else 0.0)
        if np.count_nonzero(kept) >= max(20, int(w * h * 0.001)):
            mask = kept
        if kept_areas:
            total_area = max(1.0, float(sum(kept_areas)))
            max_idx = int(np.argmax(np.asarray(kept_areas, dtype=np.float32)))
            dominant_share = float(kept_areas[max_idx] / total_area)
            dominant_edge = float(kept_edge_flags[max_idx])

    ys, xs = np.where(mask > 0)
    if xs.size < max(20, int(w * h * 0.001)):
        return mask, 0.0, {
            "pixel_ratio": 0.0,
            "span_y_ratio": 0.0,
            "dominant_share": dominant_share,
            "dominant_edge": dominant_edge,
        }

    pixel_ratio = float(xs.size) / float(max(1, w * h))
    span_y_ratio = float(ys.max() - ys.min() + 1) / float(max(1, h))
    span_x_ratio = float(xs.max() - xs.min() + 1) / float(max(1, w))

    row_count = 0
    row_residual_ratio = 1.0
    ys_for_fit: List[float] = []
    xs_for_fit: List[float] = []
    min_row_pixels = max(2, int(w * 0.015))
    for row in range(h):
        row_xs = np.where(mask[row] > 0)[0]
        if row_xs.size < min_row_pixels:
            continue
        ys_for_fit.append(float(row))
        xs_for_fit.append(float(np.median(row_xs)))
    if len(xs_for_fit) >= max(8, int(h * 0.08)):
        row_count = len(xs_for_fit)
        try:
            slope, intercept = np.polyfit(
                np.asarray(ys_for_fit, dtype=np.float32),
                np.asarray(xs_for_fit, dtype=np.float32),
                1,
            )
            predicted = np.asarray(ys_for_fit, dtype=np.float32) * slope + intercept
            residual = float(
                np.median(
                    np.abs(np.asarray(xs_for_fit, dtype=np.float32) - predicted)
                )
            )
            row_residual_ratio = min(1.0, residual / max(1.0, float(w) * 0.18))
        except Exception:
            row_residual_ratio = 1.0

    vertical_score = min(1.0, span_y_ratio / 0.75)
    pixel_score = min(1.0, pixel_ratio / 0.055)
    row_score = min(1.0, float(row_count) / max(1.0, float(h) * 0.45))
    residual_score = 1.0 - row_residual_ratio
    score = (
        0.38 * vertical_score
        + 0.22 * pixel_score
        + 0.24 * row_score
        + 0.16 * residual_score
    )
    if dominant_share >= 0.72 and dominant_edge >= 1.0:
        score *= 0.30
    elif dominant_share >= 0.82:
        score *= 0.55
    if span_y_ratio < 0.35:
        score *= 0.45
    if span_x_ratio > 0.92 and span_y_ratio < 0.45:
        score *= 0.35

    return mask, float(score), {
        "pixel_ratio": pixel_ratio,
        "span_y_ratio": span_y_ratio,
        "dominant_share": dominant_share,
        "dominant_edge": dominant_edge,
    }


def _estimate_osb_rotation_from_original(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    dark_background: bool,
    max_abs_degrees: float,
    img_w: int,
    img_h: int,
    allow_centerline_fallback: bool = False,
) -> float:
    def _to_render_rotation(source_angle: float, mask_dark_background: bool) -> float:
        clamped = _clamp_osb_rotation(source_angle, max_abs_degrees)
        if clamped == 0.0:
            return 0.0
        # Dark text on a light background and bright/outlined text on a dark
        # background produce opposite source-line signs in this detector. Match
        # the renderer to the detected source appearance instead of flipping
        # every OSB/SFX case globally.
        return clamped if mask_dark_background else -clamped

    clipped = _to_int_bbox(bbox, img_w, img_h)
    if clipped is None:
        return 0.0
    x1, y1, x2, y2 = clipped
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 16 or w < 8:
        return 0.0

    dark_text_mask, dark_text_score, dark_text_metrics = (
        _build_osb_rotation_mask_candidate(gray, bright_text=False)
    )
    bright_text_mask, bright_text_score, bright_text_metrics = (
        _build_osb_rotation_mask_candidate(gray, bright_text=True)
    )

    current_score = bright_text_score if dark_background else dark_text_score
    opposite_score = dark_text_score if dark_background else bright_text_score
    opposite_metrics = dark_text_metrics if dark_background else bright_text_metrics
    current_metrics = bright_text_metrics if dark_background else dark_text_metrics
    use_opposite_polarity = False
    if _osb_crop_has_colored_rotation_background(crop):
        enough_gain = (
            opposite_score >= current_score * OSB_COLORED_POLARITY_SCORE_RATIO
            and opposite_score >= current_score + OSB_COLORED_POLARITY_SCORE_DELTA
        )
        opposite_spans_text = opposite_metrics.get("span_y_ratio", 0.0) >= 0.50
        current_is_suspicious = (
            current_metrics.get("span_y_ratio", 0.0) < 0.38
            or (
                current_metrics.get("dominant_share", 0.0) >= 0.72
                and current_metrics.get("dominant_edge", 0.0) >= 1.0
            )
        )
        use_opposite_polarity = bool(
            enough_gain and opposite_spans_text and current_is_suspicious
        )

    mask_dark_background = bool(dark_background)
    if use_opposite_polarity:
        mask_dark_background = not mask_dark_background

    mask = bright_text_mask if mask_dark_background else dark_text_mask

    if np.count_nonzero(mask) < max(20, int(w * h * 0.001)):
        return 0.0

    centerline_source_mask = mask.copy()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    centerline_angle: Optional[float] = None
    if num_labels > 1:
        kept = np.zeros_like(mask)
        crop_area = float(max(1, w * h))
        min_area = max(4.0, crop_area * 0.00008)
        max_area = crop_area * 0.35
        component_centers: List[Tuple[float, float, float]] = []
        for label_idx in range(1, num_labels):
            area = float(stats[label_idx, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                kept[labels == label_idx] = 1
                if allow_centerline_fallback:
                    component_centers.append(
                        (
                            float(centroids[label_idx][0]),
                            float(centroids[label_idx][1]),
                            max(1.0, area),
                        )
                    )
        if np.count_nonzero(kept) >= max(20, int(w * h * 0.001)):
            mask = kept
        if allow_centerline_fallback and len(component_centers) >= 2:
            try:
                center_arr = np.asarray(component_centers, dtype=np.float32)
                center_x = center_arr[:, 0]
                center_y = center_arr[:, 1]
                center_w = np.sqrt(center_arr[:, 2])
                center_span_y = float(center_y.max() - center_y.min())
                center_span_x = float(center_x.max() - center_x.min())
                if center_span_y >= max(10.0, h * 0.22):
                    slope, intercept = np.polyfit(
                        center_y,
                        center_x,
                        1,
                        w=center_w,
                    )
                    predicted = center_y * slope + intercept
                    residual = float(np.median(np.abs(center_x - predicted)))
                    if residual <= max(5.0, center_span_x * 0.35):
                        candidate = float(np.degrees(np.arctan(float(slope))))
                        if _clamp_osb_rotation(candidate, max_abs_degrees) != 0.0:
                            centerline_angle = candidate
            except Exception:
                centerline_angle = None

    multicolumn_angle = _estimate_osb_multicolumn_vertical_rotation(
        mask,
        max_abs_degrees,
    )
    if multicolumn_angle is not None:
        return _to_render_rotation(multicolumn_angle, mask_dark_background)

    ys_for_fit: List[float] = []
    xs_for_fit: List[float] = []
    min_row_pixels = max(2, int(w * 0.015))
    for row in range(h):
        row_xs = np.where(mask[row] > 0)[0]
        if row_xs.size < min_row_pixels:
            continue
        ys_for_fit.append(float(row))
        xs_for_fit.append(float(np.median(row_xs)))

    row_render_angle = 0.0
    if len(xs_for_fit) >= max(8, int(h * 0.08)):
        try:
            slope, _ = np.polyfit(
                np.asarray(ys_for_fit, dtype=np.float32),
                np.asarray(xs_for_fit, dtype=np.float32),
                1,
            )
            angle = float(np.degrees(np.arctan(float(slope))))
            render_angle = _to_render_rotation(angle, mask_dark_background)
            if not allow_centerline_fallback:
                return render_angle
            row_render_angle = render_angle
        except Exception:
            pass

    if allow_centerline_fallback:
        sfx_axis_candidates: List[float] = []
        if centerline_angle is not None:
            centerline_render_angle = _to_render_rotation(
                centerline_angle,
                mask_dark_background,
            )
            if centerline_render_angle != 0.0:
                sfx_axis_candidates.append(centerline_render_angle)
        band_angle = _estimate_osb_vertical_band_angle(
            centerline_source_mask,
            max_abs_degrees,
            use_stroke_angle=True,
        )
        band_render_angle = _to_render_rotation(band_angle, mask_dark_background)
        if band_render_angle != 0.0:
            sfx_axis_candidates.append(band_render_angle)
        if sfx_axis_candidates:
            if len(sfx_axis_candidates) == 1:
                return sfx_axis_candidates[0]
            positives = [angle for angle in sfx_axis_candidates if angle > 0]
            negatives = [angle for angle in sfx_axis_candidates if angle < 0]
            if positives and negatives:
                return max(sfx_axis_candidates, key=lambda angle: abs(angle))
            median_angle = float(np.median(sfx_axis_candidates))
            strongest_angle = max(sfx_axis_candidates, key=lambda angle: abs(angle))
            if (
                abs(strongest_angle) >= OSB_SFX_STRONG_AXIS_MIN_ABS_DEGREES
                and abs(strongest_angle)
                >= abs(median_angle) + OSB_SFX_STRONG_AXIS_MIN_DELTA_DEGREES
            ):
                return strongest_angle
            return float(np.median(sfx_axis_candidates))
        if row_render_angle != 0.0:
            return row_render_angle

    ys, xs = np.where(mask > 0)
    if xs.size < max(20, int(w * h * 0.001)):
        return 0.0
    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    centered = coords - np.mean(coords, axis=0)
    try:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] < 1.12:
            return 0.0
        major = eigvecs[:, order[0]]
        angle_from_x = float(np.degrees(np.arctan2(major[1], major[0])))
        if angle_from_x < 0.0:
            angle_from_x += 180.0
        angle = _normalize_osb_rotation(90.0 - angle_from_x)
        return _to_render_rotation(angle, mask_dark_background)
    except Exception:
        return 0.0


def _estimate_osb_horizontal_rotation_from_original(
    image_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    dark_background: bool,
    max_abs_degrees: float,
    img_w: int,
    img_h: int,
) -> float:
    clipped = _to_int_bbox(bbox, img_w, img_h)
    if clipped is None:
        return 0.0
    x1, y1, x2, y2 = clipped
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 8 or w < 16:
        return 0.0

    if dark_background:
        threshold = max(145.0, float(np.percentile(gray, 78)))
        mask = (gray >= threshold).astype(np.uint8)
        if np.count_nonzero(mask) < max(20, int(w * h * 0.002)):
            mask = (gray >= 130).astype(np.uint8)
    else:
        threshold = min(115.0, float(np.percentile(gray, 22)))
        mask = (gray <= threshold).astype(np.uint8)
        if np.count_nonzero(mask) < max(20, int(w * h * 0.002)):
            mask = (gray <= 135).astype(np.uint8)

    if np.count_nonzero(mask) < max(20, int(w * h * 0.001)):
        return 0.0

    raw_mask = mask
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if np.count_nonzero(opened_mask) >= max(20, int(w * h * 0.001)):
        mask = opened_mask
    else:
        mask = raw_mask

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels > 1:
        kept = np.zeros_like(mask)
        crop_area = float(max(1, w * h))
        min_area = max(4.0, crop_area * 0.00008)
        max_area = crop_area * 0.35
        for label_idx in range(1, num_labels):
            area = float(stats[label_idx, cv2.CC_STAT_AREA])
            if min_area <= area <= max_area:
                kept[labels == label_idx] = 1
        if np.count_nonzero(kept) >= max(20, int(w * h * 0.001)):
            mask = kept
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, 8
            )

    ys, xs = np.where(mask > 0)
    if xs.size < max(20, int(w * h * 0.001)):
        return 0.0

    span_x = float(xs.max() - xs.min() + 1)
    span_y = float(ys.max() - ys.min() + 1)
    if span_x < max(12.0, span_y * 1.05):
        return 0.0

    angle_candidates: List[float] = []
    centerline_candidate: Optional[float] = None
    centerline_component_count = 0

    component_centers: List[Tuple[float, float]] = []
    crop_area = float(max(1, w * h))
    min_area = max(4.0, crop_area * 0.00008)
    max_area = crop_area * 0.35
    for label_idx in range(1, num_labels):
        area = float(stats[label_idx, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        component_centers.append(
            (float(centroids[label_idx][0]), float(centroids[label_idx][1]))
        )

    if len(component_centers) >= 2:
        try:
            center_arr = np.asarray(component_centers, dtype=np.float32)
            center_span_x = float(center_arr[:, 0].max() - center_arr[:, 0].min())
            if center_span_x >= max(10.0, w * 0.22):
                slope, intercept = np.polyfit(center_arr[:, 0], center_arr[:, 1], 1)
                predicted = center_arr[:, 0] * slope + intercept
                residual = float(np.median(np.abs(center_arr[:, 1] - predicted)))
                if residual <= max(5.0, span_y * 0.35):
                    center_angle = float(np.degrees(np.arctan(float(slope))))
                    angle_candidates.append(center_angle)
                    if (
                        abs(center_angle)
                        >= OSB_HORIZONTAL_CENTERLINE_MIN_ABS_DEGREES
                        and abs(center_angle)
                        <= OSB_HORIZONTAL_CENTERLINE_MAX_ABS_DEGREES
                    ):
                        centerline_candidate = center_angle
                        centerline_component_count = len(component_centers)
        except Exception:
            pass

    xs_for_fit: List[float] = []
    ys_for_fit: List[float] = []
    min_col_pixels = max(2, int(h * 0.015))
    for col in range(w):
        col_ys = np.where(mask[:, col] > 0)[0]
        if col_ys.size < min_col_pixels:
            continue
        xs_for_fit.append(float(col))
        ys_for_fit.append(float(np.median(col_ys)))

    if len(xs_for_fit) >= max(8, int(span_x * 0.16)):
        try:
            x_arr = np.asarray(xs_for_fit, dtype=np.float32)
            y_arr = np.asarray(ys_for_fit, dtype=np.float32)
            if float(x_arr.max() - x_arr.min()) >= max(10.0, w * 0.22):
                slope, intercept = np.polyfit(x_arr, y_arr, 1)
                predicted = x_arr * slope + intercept
                residual = float(np.median(np.abs(y_arr - predicted)))
                if residual <= max(5.0, span_y * 0.40):
                    angle_candidates.append(
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
        if eigvals[order[1]] > 0 and eigvals[order[0]] / eigvals[order[1]] >= 1.18:
            major = eigvecs[:, order[0]]
            angle_candidates.append(
                _normalize_osb_rotation(
                    float(np.degrees(np.arctan2(major[1], major[0])))
                )
            )
    except Exception:
        pass

    if not angle_candidates:
        return 0.0

    rotation = _clamp_osb_rotation(float(np.median(angle_candidates)), max_abs_degrees)
    if rotation and not _horizontal_projection_supports_osb_rotation(mask, rotation):
        rotation = 0.0
    if rotation == 0.0 and centerline_candidate is not None:
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = width / float(max(1, height))
        span_ratio = span_x / float(max(1, width))
        center_rotation = _clamp_osb_rotation(centerline_candidate, max_abs_degrees)
        if (
            center_rotation != 0.0
            and centerline_component_count >= OSB_HORIZONTAL_CENTERLINE_MIN_COMPONENTS
            and aspect_ratio >= OSB_HORIZONTAL_CENTERLINE_MIN_ASPECT
            and span_ratio >= OSB_HORIZONTAL_CENTERLINE_MIN_SPAN_RATIO
            and _horizontal_projection_supports_osb_rotation(mask, center_rotation)
        ):
            return center_rotation
    return rotation


def _to_int_bbox(
    box: Any, img_w: Optional[int] = None, img_h: Optional[int] = None
) -> Optional[Tuple[int, int, int, int]]:
    if box is None:
        return None
    if hasattr(box, "detach"):
        box = box.detach().cpu().numpy()
    if hasattr(box, "tolist"):
        box = box.tolist()
    if len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(c))) for c in box]
    except (TypeError, ValueError):
        return None
    if img_w is not None:
        x1 = max(0, min(img_w, x1))
        x2 = max(0, min(img_w, x2))
    if img_h is not None:
        y1 = max(0, min(img_h, y1))
        y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_area(box: Tuple[int, int, int, int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _bbox_intersection_area(
    box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_ioa(
    inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int]
) -> float:
    area = _bbox_area(inner_box)
    if area <= 0.0:
        return 0.0
    return _bbox_intersection_area(inner_box, outer_box) / area


def _bbox_center_inside(
    inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int]
) -> bool:
    cx = (inner_box[0] + inner_box[2]) / 2.0
    cy = (inner_box[1] + inner_box[3]) / 2.0
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


def _mask_coverage(
    mask: np.ndarray, bbox: Tuple[int, int, int, int], img_w: int, img_h: int
) -> float:
    x1, y1, x2, y2 = _to_int_bbox(bbox, img_w, img_h) or (0, 0, 0, 0)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.count_nonzero(region)) / float(region.size)


def _osb_buffer_conflict_is_wide_sfx_like(
    group: Dict[str, Any],
    raw_mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[bool, str]:
    """Detect wide, shallow OSB where expanded fill can erase nearby bubble ink."""
    original_bbox = group.get("original_bbox") or group.get("bbox")
    if not original_bbox:
        return False, ""

    try:
        x1 = int(original_bbox.get("x", 0))
        y1 = int(original_bbox.get("y", 0))
        width = int(original_bbox.get("width", 0))
        height = int(original_bbox.get("height", 0))
    except (AttributeError, TypeError, ValueError):
        return False, ""
    if width <= 0 or height <= 0:
        return False, ""

    x2 = x1 + width
    y2 = y1 + height
    bbox_tuple = _to_int_bbox((x1, y1, x2, y2), img_w, img_h)
    if bbox_tuple is None:
        return False, ""
    x1, y1, x2, y2 = bbox_tuple
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    aspect = width / float(height)
    width_ratio = width / float(max(1, img_w))
    height_ratio = height / float(max(1, img_h))
    area_ratio = (width * height) / float(max(1, img_w * img_h))
    if aspect < OSB_BUFFER_CONFLICT_WIDE_SFX_MIN_ASPECT:
        return False, ""
    if width_ratio < OSB_BUFFER_CONFLICT_WIDE_SFX_MIN_WIDTH_RATIO:
        return False, ""
    if height_ratio > OSB_BUFFER_CONFLICT_WIDE_SFX_MAX_HEIGHT_RATIO:
        return False, ""
    if area_ratio > OSB_BUFFER_CONFLICT_WIDE_SFX_MAX_AREA_RATIO:
        return False, ""

    crop = raw_mask[y1:y2, x1:x2]
    if crop.size == 0:
        return False, ""
    coverage = float(np.count_nonzero(crop)) / float(crop.size)
    if coverage <= 0.0:
        return False, ""

    reason = (
        f"aspect={aspect:.2f}, width_ratio={width_ratio:.2f}, "
        f"height_ratio={height_ratio:.2f}, mask_coverage={coverage:.3f}"
    )
    return True, reason


def _osb_bbox_is_mask_outside_bubble(
    bbox: Any,
    bubble_boxes: List[Tuple[int, int, int, int]],
    bubble_mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> bool:
    """Detect OSB that only overlaps a bubble bbox, not the actual bubble mask."""
    bbox_tuple = _to_int_bbox(bbox, img_w, img_h)
    if bbox_tuple is None:
        return False
    if bubble_mask is None or not np.any(bubble_mask):
        return False

    cx = int((bbox_tuple[0] + bbox_tuple[2]) / 2.0)
    cy = int((bbox_tuple[1] + bbox_tuple[3]) / 2.0)
    center_in_mask = 0 <= cx < img_w and 0 <= cy < img_h and bubble_mask[cy, cx]
    if center_in_mask:
        return False

    coverage = _mask_coverage(bubble_mask, bbox_tuple, img_w, img_h)
    if coverage > OSB_BUBBLE_MASK_OUTSIDE_RESCUE_MAX_COVERAGE:
        return False

    return any(
        _bbox_center_inside(bbox_tuple, bubble_box)
        or _bbox_ioa(bbox_tuple, bubble_box)
        >= OSB_BUBBLE_MASK_OUTSIDE_RESCUE_MIN_IOA
        for bubble_box in bubble_boxes
    )


def _osb_bbox_is_partial_sfx_outside_bubble(
    bbox: Any,
    bubble_boxes: List[Tuple[int, int, int, int]],
    bubble_mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> bool:
    """Keep wide SFX crossing a speech-bubble bbox edge without rescuing dialogue."""
    bbox_tuple = _to_int_bbox(bbox, img_w, img_h)
    if bbox_tuple is None:
        return False
    if bubble_mask is None or not np.any(bubble_mask):
        return False

    width = max(0, bbox_tuple[2] - bbox_tuple[0])
    height = max(0, bbox_tuple[3] - bbox_tuple[1])
    if width <= 0 or height <= 0:
        return False
    if width / float(height) < OSB_PARTIAL_SFX_RESCUE_MIN_ASPECT:
        return False
    if height > img_h * OSB_PARTIAL_SFX_RESCUE_MAX_HEIGHT_RATIO:
        return False

    coverage = _mask_coverage(bubble_mask, bbox_tuple, img_w, img_h)
    if coverage > OSB_PARTIAL_SFX_RESCUE_MAX_MASK_COVERAGE:
        return False

    return any(
        OSB_PARTIAL_SFX_RESCUE_MIN_BUBBLE_IOA
        <= _bbox_ioa(bbox_tuple, bubble_box)
        <= OSB_PARTIAL_SFX_RESCUE_MAX_BUBBLE_IOA
        for bubble_box in bubble_boxes
    )


def _osb_bbox_is_edge_sfx_inside_bubble(
    bbox: Any,
    bubble_boxes: List[Tuple[int, int, int, int]],
    bubble_mask: Optional[np.ndarray],
    img_w: int,
    img_h: int,
) -> bool:
    """Keep compact SFX marks that sit on a speech-bubble edge."""
    bbox_tuple = _to_int_bbox(bbox, img_w, img_h)
    if bbox_tuple is None:
        return False

    text_w = max(0, bbox_tuple[2] - bbox_tuple[0])
    text_h = max(0, bbox_tuple[3] - bbox_tuple[1])
    if text_w <= 0 or text_h <= 0:
        return False

    for bubble_box in bubble_boxes:
        if _bbox_ioa(bbox_tuple, bubble_box) < OSB_EDGE_SFX_RESCUE_MIN_BUBBLE_IOA:
            continue

        bubble_w = max(0, bubble_box[2] - bubble_box[0])
        bubble_h = max(0, bubble_box[3] - bubble_box[1])
        if bubble_w <= 0 or bubble_h <= 0:
            continue

        text_area = float(text_w * text_h)
        bubble_area = float(bubble_w * bubble_h)
        if text_area / max(1.0, bubble_area) > OSB_EDGE_SFX_RESCUE_MAX_AREA_RATIO:
            continue
        if text_w / float(bubble_w) > OSB_EDGE_SFX_RESCUE_MAX_WIDTH_RATIO:
            continue
        if text_h / float(bubble_h) > OSB_EDGE_SFX_RESCUE_MAX_HEIGHT_RATIO:
            continue

        center_x = (bbox_tuple[0] + bbox_tuple[2]) / 2.0
        center_y = (bbox_tuple[1] + bbox_tuple[3]) / 2.0
        if bubble_mask is not None and np.any(bubble_mask):
            cx = int(round(center_x))
            cy = int(round(center_y))
            center_in_mask = 0 <= cx < img_w and 0 <= cy < img_h and bubble_mask[cy, cx]
            if center_in_mask:
                continue
            if (
                _mask_coverage(bubble_mask, bbox_tuple, img_w, img_h)
                > OSB_EDGE_SFX_RESCUE_MAX_MASK_COVERAGE
            ):
                continue

        rel_x = (center_x - bubble_box[0]) / float(bubble_w)
        rel_y = (center_y - bubble_box[1]) / float(bubble_h)
        edge_band = OSB_EDGE_SFX_RESCUE_EDGE_BAND_RATIO
        if not (
            rel_x <= edge_band
            or rel_x >= 1.0 - edge_band
            or rel_y <= edge_band
            or rel_y >= 1.0 - edge_band
        ):
            continue

        max_edge_gap = min(
            OSB_EDGE_SFX_RESCUE_MAX_EDGE_GAP_PX,
            max(
                6.0,
                min(bubble_w, bubble_h) * OSB_EDGE_SFX_RESCUE_MAX_EDGE_GAP_RATIO,
            ),
        )
        edge_gap = min(
            abs(bbox_tuple[0] - bubble_box[0]),
            abs(bbox_tuple[2] - bubble_box[2]),
            abs(bbox_tuple[1] - bubble_box[1]),
            abs(bbox_tuple[3] - bubble_box[3]),
        )
        if edge_gap <= max_edge_gap:
            return True

    return False


def _expand_edge_sfx_bbox_for_full_glyph(
    bbox: Any,
    bubble_boxes: List[Tuple[int, int, int, int]],
    bubble_mask: Optional[np.ndarray],
    other_bboxes: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Add a small inward margin so a rescued edge SFX owns its full glyph."""
    original = _to_int_bbox(bbox, img_w, img_h)
    if original is None or not _osb_bbox_is_edge_sfx_inside_bubble(
        original,
        bubble_boxes,
        bubble_mask,
        img_w,
        img_h,
    ):
        return original

    matching_bubbles = [
        bubble_box
        for bubble_box in bubble_boxes
        if _bbox_ioa(original, bubble_box) >= OSB_EDGE_SFX_RESCUE_MIN_BUBBLE_IOA
    ]
    if not matching_bubbles:
        return original

    bubble_box = max(
        matching_bubbles,
        key=lambda item: (_bbox_ioa(original, item), _bbox_area(item)),
    )
    x1, y1, x2, y2 = original
    base_pad = OSB_EDGE_SFX_GLYPH_BASE_PADDING
    inward_pad = OSB_EDGE_SFX_GLYPH_INWARD_PADDING
    nx1 = max(0, x1 - base_pad)
    ny1 = max(0, y1 - base_pad)
    nx2 = min(img_w, x2 + base_pad)
    ny2 = min(img_h, y2 + base_pad)

    edge_distances = {
        "left": abs(x1 - bubble_box[0]),
        "right": abs(x2 - bubble_box[2]),
        "top": abs(y1 - bubble_box[1]),
        "bottom": abs(y2 - bubble_box[3]),
    }
    nearest_edge = min(edge_distances, key=edge_distances.get)
    if nearest_edge == "left":
        nx2 = min(img_w, x2 + inward_pad)
    elif nearest_edge == "right":
        nx1 = max(0, x1 - inward_pad)
    elif nearest_edge == "top":
        ny2 = min(img_h, y2 + inward_pad)
    else:
        ny1 = max(0, y1 - inward_pad)

    gap = OSB_EDGE_SFX_GLYPH_OBSTACLE_GAP
    for other in other_bboxes:
        other_box = _to_int_bbox(other, img_w, img_h)
        if other_box is None or other_box == original:
            continue
        ox1, oy1, ox2, oy2 = other_box
        original_overlaps = not (
            x2 <= ox1 or x1 >= ox2 or y2 <= oy1 or y1 >= oy2
        )
        expanded_overlaps = not (
            nx2 <= ox1 - gap
            or nx1 >= ox2 + gap
            or ny2 <= oy1 - gap
            or ny1 >= oy2 + gap
        )
        if original_overlaps or not expanded_overlaps:
            continue

        if x2 <= ox1:
            nx2 = min(nx2, max(x2, ox1 - gap))
        elif x1 >= ox2:
            nx1 = max(nx1, min(x1, ox2 + gap))
        if y2 <= oy1:
            ny2 = min(ny2, max(y2, oy1 - gap))
        elif y1 >= oy2:
            ny1 = max(ny1, min(y1, oy2 + gap))

    nx1 = min(nx1, x1)
    ny1 = min(ny1, y1)
    nx2 = max(nx2, x2)
    ny2 = max(ny2, y2)
    return nx1, ny1, nx2, ny2


def _dilate_bool_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    try:
        size = max(1, int(kernel_size))
    except (TypeError, ValueError):
        size = 1
    if size <= 1:
        return mask.astype(bool)
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _bbox_from_mask(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    padding: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    try:
        pad = max(0, int(padding))
    except (TypeError, ValueError):
        pad = 0
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(img_w, int(xs.max()) + 1 + pad)
    y2 = min(img_h, int(ys.max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _speech_edge_rect_mask_for_buffer_conflict(
    group: Dict[str, Any],
    raw_mask: np.ndarray,
    total_bubble_mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> Optional[np.ndarray]:
    """Build a rectangular speech-edge erase mask.

    Speech-edge regions are injected from text components close to a speech
    bubble border.  The component mask is safe but can leave anti-aliased or
    missed source glyph fragments.  For real outside text we prefer the OSB
    rectangle behavior.  These candidates have already passed the speech-edge
    split guards, so clipping them against the YOLO/SAM bubble mask can be too
    conservative: the bubble mask often covers the border-adjacent outside text
    itself and leaves the original glyphs behind.
    """
    bbox_dict = group.get("original_bbox") or group.get("bbox")
    if not bbox_dict:
        return None

    try:
        x = int(round(float(bbox_dict.get("x", 0))))
        y = int(round(float(bbox_dict.get("y", 0))))
        w = int(round(float(bbox_dict.get("width", 0))))
        h = int(round(float(bbox_dict.get("height", 0))))
    except (AttributeError, TypeError, ValueError):
        return None

    if w <= 0 or h <= 0:
        return None

    x1 = max(0, min(img_w, x - OSB_SPEECH_EDGE_RECT_PAD_X))
    y1 = max(0, min(img_h, y - OSB_SPEECH_EDGE_RECT_PAD_Y))
    x2 = max(0, min(img_w, x + w + OSB_SPEECH_EDGE_RECT_PAD_X))
    y2 = max(0, min(img_h, y + h + OSB_SPEECH_EDGE_RECT_PAD_Y))
    if x2 <= x1 or y2 <= y1:
        return None

    rect_mask = np.zeros((img_h, img_w), dtype=bool)
    rect_mask[y1:y2, x1:x2] = True
    rect_pixels = int(np.count_nonzero(rect_mask))
    if rect_pixels <= 0:
        return None

    rect_after_protection = int(np.count_nonzero(rect_mask))
    if rect_after_protection <= 0:
        return None

    raw_pixels = int(np.count_nonzero(raw_mask))
    min_pixels = max(
        OSB_SPEECH_EDGE_RECT_MIN_PIXELS,
        int(round(raw_pixels * OSB_SPEECH_EDGE_RECT_MIN_RAW_MASK_GAIN)),
        int(round(rect_pixels * OSB_SPEECH_EDGE_RECT_MIN_BBOX_COVERAGE)),
    )
    if rect_after_protection < min_pixels:
        return None

    return rect_mask


def _build_bubble_exclusion_data(
    bubble_data: Optional[List[Dict[str, Any]]],
    img_w: int,
    img_h: int,
    verbose: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    bubble_boxes: List[Tuple[int, int, int, int]] = []
    bubble_mask = np.zeros((img_h, img_w), dtype=bool)
    if not bubble_data:
        return bubble_boxes, bubble_mask

    for bubble in bubble_data:
        try:
            bbox = bubble.get("bbox") if isinstance(bubble, dict) else bubble
            bbox_tuple = _to_int_bbox(bbox, img_w, img_h)
            if bbox_tuple is not None:
                bubble_boxes.append(bbox_tuple)

            mask = bubble.get("sam_mask") if isinstance(bubble, dict) else None
            if mask is not None:
                mask_np = np.asarray(mask)
                if mask_np.ndim == 3:
                    mask_np = mask_np[..., 0]
                mask_bool = mask_np > 0
                if mask_bool.shape[0] == img_h and mask_bool.shape[1] == img_w:
                    bubble_mask |= mask_bool
                    continue

            if bbox_tuple is not None:
                x1, y1, x2, y2 = bbox_tuple
                bubble_mask[y1:y2, x1:x2] = True
        except Exception as e:
            log_message(
                f"Warning: Failed to apply bubble mask for OSB exclusion: {e}",
                verbose=verbose,
            )

    return bubble_boxes, bubble_mask


def _osb_bbox_overlaps_bubble(
    expanded_bbox: Any,
    raw_bbox: Any,
    bubble_boxes: List[Tuple[int, int, int, int]],
    bubble_mask: np.ndarray,
    img_w: int,
    img_h: int,
) -> bool:
    raw_bbox_tuple = _to_int_bbox(raw_bbox, img_w, img_h)
    if _osb_bbox_is_mask_outside_bubble(
        raw_bbox_tuple,
        bubble_boxes,
        bubble_mask,
        img_w,
        img_h,
    ):
        return False
    if _osb_bbox_is_partial_sfx_outside_bubble(
        raw_bbox_tuple,
        bubble_boxes,
        bubble_mask,
        img_w,
        img_h,
    ):
        return False
    if _osb_bbox_is_edge_sfx_inside_bubble(
        raw_bbox_tuple,
        bubble_boxes,
        bubble_mask,
        img_w,
        img_h,
    ):
        return False

    for bbox in (raw_bbox_tuple, _to_int_bbox(expanded_bbox, img_w, img_h)):
        if bbox is None:
            continue

        cx = int((bbox[0] + bbox[2]) / 2.0)
        cy = int((bbox[1] + bbox[3]) / 2.0)
        if 0 <= cx < img_w and 0 <= cy < img_h and bubble_mask[cy, cx]:
            return True
        if (
            _mask_coverage(bubble_mask, bbox, img_w, img_h)
            >= OSB_BUBBLE_MASK_COVERAGE_THRESHOLD
        ):
            return True

        for bubble_box in bubble_boxes:
            if _bbox_center_inside(bbox, bubble_box):
                return True
            if _bbox_ioa(bbox, bubble_box) >= OSB_BUBBLE_IOA_FILTER_THRESHOLD:
                return True

    return False


def _build_outside_text_translation_data(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_format: Optional[str],
    outside_text_results: List[Tuple[List[int], float]],
    raw_outside_text_results: List[Tuple[List[int], float]],
    original_text_colors: Dict[Tuple[int, int, int, int], bool],
    bubble_boxes: List[Tuple[int, int, int, int]],
    bubble_core_mask: np.ndarray,
    img_w: int,
    img_h: int,
    verbose: bool = False,
    group_render_bboxes: Optional[
        Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]]
    ] = None,
    extracted_text_colors: Optional[
        Dict[Tuple[int, int, int, int], Tuple[int, int, int]]
    ] = None,
    none_skipped_clip_bboxes: Optional[set] = None,
    allowed_bubble_overlap_bboxes: Optional[set] = None,
    edge_sfx_full_glyph_bboxes: Optional[set] = None,
    injected_metadata_by_bbox: Optional[
        Dict[Tuple[int, int, int, int], Dict[str, Any]]
    ] = None,
) -> List[Dict[str, Any]]:
    """Build OSB crop metadata for translation without requiring inpainting output."""
    group_render_bboxes = group_render_bboxes or {}
    extracted_text_colors = extracted_text_colors or {}
    none_skipped_clip_bboxes = none_skipped_clip_bboxes or set()
    allowed_bubble_overlap_bboxes = allowed_bubble_overlap_bboxes or set()
    edge_sfx_full_glyph_bboxes = edge_sfx_full_glyph_bboxes or set()
    injected_metadata_by_bbox = injected_metadata_by_bbox or {}

    outside_text_data = []
    original_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    mime_type = (
        "image/png"
        if image_format and image_format.upper() == "PNG"
        else "image/jpeg"
    )
    cv2_ext = ".png" if image_format and image_format.upper() == "PNG" else ".jpg"

    for expanded_res, raw_res in zip(outside_text_results, raw_outside_text_results):
        bbox_coords, conf = expanded_res
        raw_coords, _ = raw_res

        x1, y1, x2, y2 = [int(c) for c in bbox_coords]
        bbox_tuple = (x1, y1, x2, y2)
        raw_bbox_tuple = tuple(int(c) for c in raw_coords)
        render_bbox_tuple = group_render_bboxes.get(raw_bbox_tuple)
        forced_bubble_overlap_rescue = raw_bbox_tuple in allowed_bubble_overlap_bboxes
        mask_outside_bubble_overlap = _osb_bbox_is_mask_outside_bubble(
            raw_bbox_tuple,
            bubble_boxes,
            bubble_core_mask,
            img_w,
            img_h,
        )
        bubble_overlap_rescue = bool(
            forced_bubble_overlap_rescue
            or mask_outside_bubble_overlap
            or _osb_bbox_is_partial_sfx_outside_bubble(
                raw_bbox_tuple,
                bubble_boxes,
                bubble_core_mask,
                img_w,
                img_h,
            )
            or _osb_bbox_is_edge_sfx_inside_bubble(
                raw_bbox_tuple,
                bubble_boxes,
                bubble_core_mask,
                img_w,
                img_h,
            )
        )
        if _osb_bbox_overlaps_bubble(
            bbox_tuple,
            raw_bbox_tuple,
            bubble_boxes,
            bubble_core_mask,
            img_w,
            img_h,
        ) and not forced_bubble_overlap_rescue:
            log_message(
                f"Skipping OSB render data inside speech bubble: {raw_bbox_tuple}",
                verbose=verbose,
            )
            continue

        outside_text_image_cv = original_cv_image[y1:y2, x1:x2].copy()
        outside_text_image_pil = cv2_to_pil(outside_text_image_cv)
        original_crop_pil = outside_text_image_pil.copy()

        osb_upscale_method = (
            "none" if config.test_mode else config.translation.upscale_method
        )
        if osb_upscale_method == "model":
            model_manager = get_model_manager()
            upscale_model = model_manager.load_upscale(verbose=verbose)
            final_text_pil = process_bubble_image_cached(
                outside_text_image_pil,
                upscale_model,
                config.device,
                config.translation.osb_min_side_pixels,
                "min",
                "model",
                verbose,
            )
            model_manager.clear_cache()
        elif osb_upscale_method == "model_lite":
            model_manager = get_model_manager()
            upscale_model = model_manager.load_upscale_lite(verbose=verbose)
            final_text_pil = process_bubble_image_cached(
                outside_text_image_pil,
                upscale_model,
                config.device,
                config.translation.osb_min_side_pixels,
                "min",
                "model_lite",
                verbose,
            )
            model_manager.clear_cache()
        elif osb_upscale_method == "lanczos":
            w, h = outside_text_image_pil.size
            min_side = min(w, h)
            if min_side < config.translation.osb_min_side_pixels:
                scale_factor = config.translation.osb_min_side_pixels / min_side
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                final_text_pil = outside_text_image_pil.resize(
                    (new_w, new_h), Image.LANCZOS
                )
            else:
                final_text_pil = outside_text_image_pil
        else:
            final_text_pil = outside_text_image_pil

        outside_text_image_cv = pil_to_cv2(final_text_pil)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = float(h) / float(w)
        is_dark_text_val = original_text_colors.get(raw_bbox_tuple, True)
        osb_rotation_max_degrees = getattr(
            config.outside_text, "osb_rotation_max_degrees", 0.0
        )
        try:
            osb_rotation_max_degrees = float(osb_rotation_max_degrees)
        except (TypeError, ValueError):
            osb_rotation_max_degrees = 0.0
        sfx_rotation_max_degrees = (
            max(osb_rotation_max_degrees, 25.0)
            if osb_rotation_max_degrees > 0.0
            else 0.0
        )
        rotation_deg = _estimate_osb_rotation_from_original(
            original_cv_image,
            raw_bbox_tuple,
            dark_background=bool(is_dark_text_val),
            max_abs_degrees=osb_rotation_max_degrees,
            img_w=img_w,
            img_h=img_h,
        )
        sfx_rotation_deg = _estimate_osb_rotation_from_original(
            original_cv_image,
            raw_bbox_tuple,
            dark_background=bool(is_dark_text_val),
            max_abs_degrees=sfx_rotation_max_degrees,
            img_w=img_w,
            img_h=img_h,
            allow_centerline_fallback=True,
        )
        horizontal_rotation_deg = _estimate_osb_horizontal_rotation_from_original(
            original_cv_image,
            raw_bbox_tuple,
            dark_background=bool(is_dark_text_val),
            max_abs_degrees=osb_rotation_max_degrees,
            img_w=img_w,
            img_h=img_h,
        )
        render_mask = None
        if is_dark_text_val:
            render_mask = _build_osb_render_mask(
                original_cv_image,
                bbox_tuple,
                raw_bbox_tuple,
                dark_background=True,
                img_w=img_w,
                img_h=img_h,
            )

        needs_text_bg = False
        if none_skipped_clip_bboxes:
            bcx = (x1 + x2) / 2
            bcy = (y1 + y2) / 2
            for clip_bbox in none_skipped_clip_bboxes:
                cx1, cy1, cx2, cy2 = clip_bbox
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    needs_text_bg = True
                    break

        try:
            is_success, buffer = cv2.imencode(cv2_ext, outside_text_image_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                item = {
                    "bbox": bbox_tuple,
                    "original_bbox": raw_bbox_tuple,
                    "confidence": conf,
                    "is_outside_text": True,
                    "image_b64": image_b64,
                    "mime_type": mime_type,
                    "is_dark_text": is_dark_text_val,
                    "text_color_rgb": extracted_text_colors.get(raw_bbox_tuple),
                    "aspect_ratio": aspect_ratio,
                    "rotation_deg": rotation_deg,
                    "sfx_rotation_deg": sfx_rotation_deg,
                    "horizontal_rotation_deg": horizontal_rotation_deg,
                    "render_bbox": render_bbox_tuple,
                    "render_mask": render_mask,
                    "needs_text_background": needs_text_bg,
                    "mask_outside_bubble_overlap": mask_outside_bubble_overlap,
                    "bubble_overlap_rescue": bubble_overlap_rescue,
                    "edge_sfx_full_glyph": (
                        raw_bbox_tuple in edge_sfx_full_glyph_bboxes
                    ),
                    "original_crop_pil": original_crop_pil,
                }
                injected_metadata = injected_metadata_by_bbox.get(raw_bbox_tuple)
                if injected_metadata:
                    for key in (
                        "speech_edge_split",
                        "parent_bbox",
                        "speech_edge_side",
                        "mask_outside_bubble_overlap",
                        "bubble_overlap_rescue",
                    ):
                        if key in injected_metadata:
                            item[key] = injected_metadata[key]
                    if item.get("speech_edge_split"):
                        item["mask_outside_bubble_overlap"] = True
                        item["bubble_overlap_rescue"] = True
                outside_text_data.append(item)
        except Exception as e:
            log_message(
                f"Error encoding outside text bbox {(x1, y1, x2, y2)}: {e}",
                verbose=verbose,
            )

    return outside_text_data


def process_outside_text(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_path: Union[str, Path],
    image_format: Optional[str],
    verbose: bool = False,
    bubble_data: Optional[List[Dict[str, Any]]] = None,
    text_free_boxes: Optional[List[List[float]]] = None,
    panels: Optional[List[Tuple[int, int, int, int]]] = None,
    pre_inpaint_data_callback: Optional[
        Callable[[List[Dict[str, Any]]], Optional[set]]
    ] = None,
    injected_outside_text_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Process outside text detection, inpainting, and prepare data for translation.

    In prose mode the callback returns selected original bboxes. Missing
    classification preserves all outside text; it never authorizes a fill.

    This function handles the complete outside text processing pipeline:
    1. Detects text outside speech bubbles using OCR
    2. Inpaints the detected text regions using FluxKontext
    3. Prepares the outside text data for translation API calls

    Args:
        pil_image: The PIL image to process
        config: MangaTranslatorConfig containing all settings
        image_path: Path to the original image file
        image_format: Original image format (PNG, JPEG, etc.)
        processing_scale: The scale factor for image processing
        verbose: Whether to print detailed logging

    Returns:
        Tuple containing:
        - processed_pil_image: The image after outside text inpainting
        - outside_text_data: List of dicts with outside text information for translation
    """
    if not config.outside_text.enabled or normalize_outside_text_processing_mode(
        config.outside_text.processing_mode
    ) == "off":
        return pil_image, []

    log_message("Detecting text outside speech bubbles...", verbose=verbose)

    try:
        outside_detector = OutsideTextDetector(
            device=config.device, hf_token=config.outside_text.huggingface_token
        )
        outside_text_results = outside_detector.detect_outside_text(
            str(image_path),
            yolo_model_path=config.yolo_model_path,
            confidence=config.outside_text.osb_confidence,
            conjoined_confidence=config.detection.conjoined_confidence,
            verbose=verbose,
            image_override=pil_image,
            existing_bubbles=bubble_data,
            text_free_boxes=text_free_boxes,
            bubble_detector_model=config.detection.bubble_detector_model,
        )

        img_w, img_h = pil_image.size
        injected_results: List[Tuple[List[int], float]] = []
        injected_bboxes: set = set()
        injected_masks_by_bbox: Dict[Tuple[int, int, int, int], Any] = {}
        injected_metadata_by_bbox: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
        for injected in injected_outside_text_regions or []:
            try:
                coords = [
                    int(round(float(v)))
                    for v in (injected.get("original_bbox") or injected.get("bbox") or ())
                ]
            except (TypeError, ValueError):
                continue
            if len(coords) != 4:
                continue
            x1, y1, x2, y2 = coords
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            bbox_tuple = (x1, y1, x2, y2)
            if any(
                _bbox_ioa(bbox_tuple, tuple(int(c) for c in existing[0])) >= 0.70
                for existing in outside_text_results
            ):
                continue
            conf = float(injected.get("confidence", 0.50) or 0.50)
            injected_results.append(([x1, y1, x2, y2], conf))
            injected_bboxes.add(bbox_tuple)
            injected_metadata_by_bbox[bbox_tuple] = dict(injected)
            if injected.get("speech_edge_erase_mask") is not None:
                injected_masks_by_bbox[bbox_tuple] = injected.get(
                    "speech_edge_erase_mask"
                )

        if not outside_text_results and not injected_results:
            log_message("No outside text regions found", verbose=verbose)
            return pil_image, []

        # Filter out probable page numbers
        # Only run OCR on "suspicious" detections (small & in margin)
        if config.outside_text.enable_page_number_filtering and outside_text_results:
            suspicious_crops = []
            suspicious_indices = []
            safe_results = []

            margin_threshold = max(
                0.0, min(0.3, config.outside_text.page_filter_margin_threshold)
            )
            min_area_threshold = max(
                0.0, min(0.2, config.outside_text.page_filter_min_area_ratio)
            )

            for i, res in enumerate(outside_text_results):
                bbox, _ = res
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cy = (y1 + y2) / 2

                is_in_margin = (cy < img_h * margin_threshold) or (
                    cy > img_h * (1 - margin_threshold)
                )

                area = (x2 - x1) * (y2 - y1)
                is_small = area < (img_w * img_h * min_area_threshold)

                if is_in_margin and is_small:
                    suspicious_crops.append(pil_image.crop((x1, y1, x2, y2)))
                    suspicious_indices.append(i)
                else:
                    safe_results.append(res)

            if suspicious_crops:
                log_message(
                    f"Verifying {len(suspicious_crops)} suspicious OSB regions with OCR...",
                    verbose=verbose,
                )
                suspicious_texts = extract_text_with_manga_ocr(
                    suspicious_crops, verbose=verbose
                )

                kept_suspicious_count = 0
                for i, text in enumerate(suspicious_texts):
                    # Regex for page numbers: digits, "Page 20", "p. 20", etc.
                    is_page_number = bool(
                        re.match(
                            r"^\s*(?:page\.?|p\.?)?\s*\d+\s*$", text, re.IGNORECASE
                        )
                    )

                    if not is_page_number:
                        safe_results.append(outside_text_results[suspicious_indices[i]])
                        kept_suspicious_count += 1
                    else:
                        log_message(
                            f"Filtered out page number: '{text}'", verbose=verbose
                        )

                outside_text_results = safe_results
                log_message(
                    f"Remaining OSB regions after filtering: {len(outside_text_results)}",
                    verbose=verbose,
                )

        if injected_results:
            outside_text_results.extend(injected_results)
            log_message(
                f"Injected {len(injected_results)} separated speech-edge text region(s) into OSB inpainting",
                verbose=verbose,
                always_print=True,
            )

        raw_outside_text_results = outside_text_results.copy()

        # Apply OSB render expansion for shapes that tend to render too small.
        narrow_expansion_mult = getattr(
            config.outside_text, "osb_render_expansion_narrow_multiplier", 1.0
        )
        tiny_expansion_mult = getattr(
            config.outside_text, "osb_render_expansion_tiny_multiplier", 1.0
        )
        aspect_ratio_threshold = getattr(
            config.outside_text, "osb_render_expansion_aspect_ratio_threshold", 0.4
        )
        area_ratio_threshold = getattr(
            config.outside_text, "osb_render_expansion_area_ratio_threshold", 0.005
        )
        if max(narrow_expansion_mult, tiny_expansion_mult) > 1.0:
            log_message(
                "Expanding OSB bboxes "
                f"(narrow/tall: {narrow_expansion_mult}x, tiny: {tiny_expansion_mult}x)...",
                verbose=verbose,
            )
            expanded_results = []
            for i, res in enumerate(outside_text_results):
                bbox, conf = res
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1

                aspect_ratio = float(w) / float(max(1, h))
                area_ratio = (w * h) / float(max(1, img_w * img_h))
                is_narrow_tall = aspect_ratio <= aspect_ratio_threshold
                is_tiny = area_ratio < area_ratio_threshold

                expansion_mult = 1.0
                if is_narrow_tall:
                    expansion_mult = max(expansion_mult, narrow_expansion_mult)
                if is_tiny:
                    expansion_mult = max(expansion_mult, tiny_expansion_mult)

                if expansion_mult <= 1.0:
                    expanded_results.append(
                        ([int(x1), int(y1), int(x2), int(y2)], conf)
                    )
                    continue

                cx = x1 + w / 2
                cy = y1 + h / 2

                new_w = w * expansion_mult
                new_h = h * expansion_mult

                nx1 = int(cx - new_w / 2)
                ny1 = int(cy - new_h / 2)
                nx2 = int(cx + new_w / 2)
                ny2 = int(cy + new_h / 2)

                nx1 = max(0, nx1)
                ny1 = max(0, ny1)
                nx2 = min(img_w, nx2)
                ny2 = min(img_h, ny2)

                if panels:
                    associated_panel = None
                    for p in panels:
                        px1, py1, px2, py2 = p
                        if px1 <= cx <= px2 and py1 <= cy <= py2:
                            associated_panel = p
                            break
                    if associated_panel:
                        px1, py1, px2, py2 = associated_panel
                        panel_buffer = OSB_EXPANSION_PIXEL_BUFFER
                        c_px1 = min(int(px1) + panel_buffer, int(px2))
                        c_py1 = min(int(py1) + panel_buffer, int(py2))
                        c_px2 = max(int(px2) - panel_buffer, int(px1))
                        c_py2 = max(int(py2) - panel_buffer, int(py1))

                        nx1 = max(c_px1, nx1)
                        ny1 = max(c_py1, ny1)
                        nx2 = min(c_px2, nx2)
                        ny2 = min(c_py2, ny2)

                buffer = OSB_EXPANSION_PIXEL_BUFFER
                obstacles = []
                if bubble_data:
                    for b in bubble_data:
                        bb = b.get("bbox")
                        if bb and len(bb) == 4:
                            bx1, by1, bx2, by2 = [int(c) for c in bb]
                            obstacles.append(
                                (
                                    max(0, bx1 - buffer),
                                    max(0, by1 - buffer),
                                    min(img_w, bx2 + buffer),
                                    min(img_h, by2 + buffer),
                                )
                            )

                for j, other_res in enumerate(outside_text_results):
                    if i == j:
                        continue

                    if j < i:
                        ob, _ = expanded_results[j]
                    else:
                        ob, _ = other_res
                    obx1, oby1, obx2, oby2 = [int(c) for c in ob]

                    obstacles.append(
                        (
                            max(0, obx1 - buffer),
                            max(0, oby1 - buffer),
                            min(img_w, obx2 + buffer),
                            min(img_h, oby2 + buffer),
                        )
                    )

                for ox1, oy1, ox2, oy2 in obstacles:
                    if not (nx2 <= ox1 or nx1 >= ox2 or ny2 <= oy1 or ny1 >= oy2):
                        can_retract_nx2 = (nx2 - ox1) if (ox1 >= x2) else float("inf")
                        can_retract_nx1 = (ox2 - nx1) if (ox2 <= x1) else float("inf")
                        can_retract_ny2 = (ny2 - oy1) if (oy1 >= y2) else float("inf")
                        can_retract_ny1 = (oy2 - ny1) if (oy2 <= y1) else float("inf")

                        min_retract = min(
                            can_retract_nx2,
                            can_retract_nx1,
                            can_retract_ny2,
                            can_retract_ny1,
                        )
                        if min_retract != float("inf"):
                            if min_retract == can_retract_nx2:
                                nx2 = ox1
                            elif min_retract == can_retract_nx1:
                                nx1 = ox2
                            elif min_retract == can_retract_ny2:
                                ny2 = oy1
                            elif min_retract == can_retract_ny1:
                                ny1 = oy2

                nx1 = min(nx1, int(x1))
                ny1 = min(ny1, int(y1))
                nx2 = max(nx2, int(x2))
                ny2 = max(ny2, int(y2))

                expanded_results.append(([nx1, ny1, nx2, ny2], conf))

            outside_text_results = expanded_results

        # Build speech-bubble exclusion data before inpainting and rendering.
        bubble_boxes, bubble_core_mask = _build_bubble_exclusion_data(
            bubble_data, img_w, img_h, verbose=verbose
        )
        edge_sfx_expanded_bboxes: set = set()
        if bubble_boxes and outside_text_results:
            raw_boxes_for_obstacles = [
                tuple(int(c) for c in result[0])
                for result in raw_outside_text_results
            ]
            adjusted_expanded_results = []
            adjusted_raw_results = []
            for expanded_res, raw_res in zip(
                outside_text_results, raw_outside_text_results
            ):
                expanded_bbox, expanded_conf = expanded_res
                raw_bbox, raw_conf = raw_res
                raw_bbox_tuple = tuple(int(c) for c in raw_bbox)
                adjusted_raw_bbox = _expand_edge_sfx_bbox_for_full_glyph(
                    raw_bbox_tuple,
                    bubble_boxes,
                    bubble_core_mask,
                    raw_boxes_for_obstacles,
                    img_w,
                    img_h,
                )
                if (
                    adjusted_raw_bbox is not None
                    and adjusted_raw_bbox != raw_bbox_tuple
                ):
                    ex1, ey1, ex2, ey2 = [int(c) for c in expanded_bbox]
                    ax1, ay1, ax2, ay2 = adjusted_raw_bbox
                    adjusted_expanded_bbox = (
                        min(ex1, ax1),
                        min(ey1, ay1),
                        max(ex2, ax2),
                        max(ey2, ay2),
                    )
                    edge_sfx_expanded_bboxes.add(adjusted_raw_bbox)
                    adjusted_expanded_results.append(
                        ([int(c) for c in adjusted_expanded_bbox], expanded_conf)
                    )
                    adjusted_raw_results.append(
                        ([int(c) for c in adjusted_raw_bbox], raw_conf)
                    )
                    log_message(
                        "Expanded rescued edge SFX bbox to include the full glyph: "
                        f"{raw_bbox_tuple} -> {adjusted_raw_bbox}",
                        always_print=True,
                    )
                else:
                    adjusted_expanded_results.append(expanded_res)
                    adjusted_raw_results.append(raw_res)
            outside_text_results = adjusted_expanded_results
            raw_outside_text_results = adjusted_raw_results

        allowed_bubble_overlap_bboxes = set(injected_bboxes)
        allowed_bubble_overlap_bboxes.update(edge_sfx_expanded_bboxes)
        if bubble_boxes or np.any(bubble_core_mask):
            filtered_expanded_results = []
            filtered_raw_results = []
            removed_overlap_count = 0
            for expanded_res, raw_res in zip(
                outside_text_results, raw_outside_text_results
            ):
                expanded_bbox, _ = expanded_res
                raw_bbox, _ = raw_res
                raw_bbox_tuple = tuple(int(c) for c in raw_bbox)
                if raw_bbox_tuple in allowed_bubble_overlap_bboxes:
                    filtered_expanded_results.append(expanded_res)
                    filtered_raw_results.append(raw_res)
                    continue
                if _osb_bbox_overlaps_bubble(
                    expanded_bbox,
                    raw_bbox,
                    bubble_boxes,
                    bubble_core_mask,
                    img_w,
                    img_h,
                ):
                    removed_overlap_count += 1
                    continue
                filtered_expanded_results.append(expanded_res)
                filtered_raw_results.append(raw_res)

            if removed_overlap_count:
                log_message(
                    "Filtered out "
                    f"{removed_overlap_count} OSB region(s) overlapping speech bubbles",
                    always_print=True,
                )
            outside_text_results = filtered_expanded_results
            raw_outside_text_results = filtered_raw_results

        if not outside_text_results:
            log_message(
                "No outside text regions left after speech bubble overlap filtering",
                verbose=verbose,
            )
            return pil_image, []

        total_bubble_mask = bubble_core_mask
        if np.any(total_bubble_mask):
            # Dilate the bubble mask to provide a safe buffer for OSB fill.
            kernel = np.ones((11, 11), np.uint8)
            total_bubble_mask = cv2.dilate(
                total_bubble_mask.astype(np.uint8), kernel, iterations=1
            ).astype(bool)

        mime_type = (
            "image/png"
            if image_format and image_format.upper() == "PNG"
            else "image/jpeg"
        )
        cv2_ext = ".png" if image_format and image_format.upper() == "PNG" else ".jpg"

        # Probe original text color for OSB rendering
        original_text_colors = {}
        for ocr_result in raw_outside_text_results:
            bbox_coords, conf = ocr_result
            x1, y1, x2, y2 = [int(c) for c in bbox_coords]
            bbox_tuple = (x1, y1, x2, y2)

            bbox_area_img = pil_image.crop((x1, y1, x2, y2))
            bbox_array = np.array(bbox_area_img)

            if bbox_array.shape[-1] == 4:
                bbox_array = bbox_array[..., :3]

            pixels = bbox_array.reshape(-1, 3)

            # Use K-Means to find 2 dominant colors
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(pixels)

            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster_idx = unique[np.argmax(counts)]

            # Dominant cluster is usually the background (text pixels are sparse)
            bg_color_rgb = centers[dominant_cluster_idx]
            # Use proper luminance calculation (ITU-R BT.601)
            bg_brightness = (
                0.299 * bg_color_rgb[0]
                + 0.587 * bg_color_rgb[1]
                + 0.114 * bg_color_rgb[2]
            )
            is_dark_text = (
                bg_brightness < 128
            )  # passed downstream; renderer inverts for text color
            original_text_colors[bbox_tuple] = is_dark_text

            log_message(
                f"OSB bbox {bbox_tuple}: "
                f"{'Dark' if is_dark_text else 'Light'} background detected "
                f"(luminance={bg_brightness:.1f})",
                verbose=verbose,
            )

        selective = normalize_outside_text_processing_mode(
            config.outside_text.processing_mode
        ) == "prose"
        selected_bboxes = set() if selective else None
        pre_inpaint_outside_text_data = []
        if pre_inpaint_data_callback is not None or selective:
            try:
                pre_inpaint_outside_text_data = _build_outside_text_translation_data(
                    pil_image=pil_image,
                    config=config,
                    image_format=image_format,
                    outside_text_results=outside_text_results,
                    raw_outside_text_results=raw_outside_text_results,
                    original_text_colors=original_text_colors,
                    bubble_boxes=bubble_boxes,
                    bubble_core_mask=bubble_core_mask,
                    img_w=img_w,
                    img_h=img_h,
                    verbose=verbose,
                    allowed_bubble_overlap_bboxes=allowed_bubble_overlap_bboxes,
                    edge_sfx_full_glyph_bboxes=edge_sfx_expanded_bboxes,
                    injected_metadata_by_bbox=injected_metadata_by_bbox,
                )
                if pre_inpaint_data_callback is not None:
                    selection = pre_inpaint_data_callback(pre_inpaint_outside_text_data)
                    if selective:
                        if selection is None:
                            raise TranslationError(
                                "Outside text classification returned no selection."
                            )
                        selected_bboxes = set(selection)
            except CancellationError:
                raise
            except Exception as e:
                if selective:
                    raise TranslationError(
                        f"Outside text classification failed; inpainting skipped: {e}"
                    ) from e
                log_message(
                    f"Warning: Failed to prepare pre-inpaint OSB translation data: {e}",
                    always_print=True,
                )

        selected_indices = [
            i for i, result in enumerate(raw_outside_text_results)
            # Crop metadata truncates YOLO float coordinates to integer pixels.
            # Compare using that same key so valid prose is not silently skipped.
            if selected_bboxes is None
            or tuple(int(c) for c in result[0]) in selected_bboxes
        ]
        preserve_mask = np.zeros((img_h, img_w), dtype=bool)
        for i, (coords, _) in enumerate(outside_text_results):
            if (
                selected_bboxes is not None
                and tuple(int(c) for c in raw_outside_text_results[i][0])
                not in selected_bboxes
            ):
                x1, y1, x2, y2 = [int(c) for c in coords]
                preserve_mask[
                    max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)
                ] = True
        if not selected_indices:
            for item in pre_inpaint_outside_text_data:
                item.update(
                    skip_render=True, skip_reason="outside_text_processing_mode",
                    outside_text_preserve_bbox=item["bbox"],
                )
            return pil_image, pre_inpaint_outside_text_data

        log_message("Inpainting outside text regions...", verbose=verbose)

        # Create inpainter based on selected method
        inpainting_method = config.outside_text.inpainting_method
        inpainter = None

        if inpainting_method == "flux_klein_9b":
            try:
                inpainter = FluxKleinInpainter(
                    variant="9b",
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    cpu_offload=config.outside_text.flux_cpu_offload,
                    low_vram=config.outside_text.flux_low_vram,
                    luminance_correction=config.outside_text.flux_luminance_correction,
                    upscale_small_crops=config.outside_text.flux_upscale_small_crops,
                    verbose=verbose,
                    prompt=config.outside_text.flux_klein_prompt,
                )
                log_message("Using Flux.2 Klein 9B for inpainting", verbose=verbose)
            except Exception as e:
                log_message(
                    f"Flux Klein 9B unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "flux_klein_4b":
            try:
                inpainter = FluxKleinInpainter(
                    variant="4b",
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    cpu_offload=config.outside_text.flux_cpu_offload,
                    low_vram=config.outside_text.flux_low_vram,
                    luminance_correction=config.outside_text.flux_luminance_correction,
                    upscale_small_crops=config.outside_text.flux_upscale_small_crops,
                    verbose=verbose,
                    prompt=config.outside_text.flux_klein_prompt,
                )
                log_message("Using Flux.2 Klein 4B for inpainting", verbose=verbose)
            except Exception as e:
                log_message(
                    f"Flux Klein 4B unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "flux_kontext":
            try:
                # Determine backend from config
                backend = config.outside_text.kontext_backend
                inpainter = FluxKontextInpainter(
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
                    backend=backend,
                    cpu_offload=config.outside_text.flux_cpu_offload,
                    low_vram=config.outside_text.flux_low_vram,
                    prompt=config.outside_text.flux_kontext_prompt,
                )
                backend_label = "SDNQ" if backend == "sdnq" else "Nunchaku"
                log_message(
                    f"Using Flux.1 Kontext ({backend_label}) for inpainting",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(
                    f"Flux Kontext unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "none":
            inpainter = None
            log_message(
                "Using text background mode (no inpainting for non-solid regions)",
                verbose=verbose,
            )
        elif inpainting_method == "opencv" or inpainter is None:
            inpainter = None
            log_message("Using OpenCV simple fill for inpainting", verbose=verbose)

        mask_groups, _ = outside_detector.get_text_masks(
            str(image_path),
            bbox_expansion_percent=config.outside_text.bbox_expansion_percent,
            text_box_proximity_ratio=config.outside_text.text_box_proximity_ratio,
            verbose=verbose,
            image_override=pil_image,
            existing_results=[raw_outside_text_results[i] for i in selected_indices],
        )
        for group in mask_groups or []:
            group["mask_indices"] = [
                selected_indices[i] for i in group.get("mask_indices", [])
            ]
            bbox_dict = group.get("original_bbox") or group.get("bbox") or {}
            try:
                group_bbox = (
                    int(bbox_dict.get("x", 0)),
                    int(bbox_dict.get("y", 0)),
                    int(bbox_dict.get("x", 0)) + int(bbox_dict.get("width", 0)),
                    int(bbox_dict.get("y", 0)) + int(bbox_dict.get("height", 0)),
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if group_bbox in edge_sfx_expanded_bboxes:
                group["edge_sfx_full_glyph"] = True

        injected_masks_by_index: Dict[int, Any] = {}
        for idx, raw_res in enumerate(raw_outside_text_results):
            if idx not in selected_indices:
                continue
            try:
                raw_bbox_tuple = tuple(int(c) for c in raw_res[0])
            except Exception:
                continue
            if raw_bbox_tuple in injected_masks_by_bbox:
                injected_masks_by_index[idx] = injected_masks_by_bbox[raw_bbox_tuple]

        def build_injected_mask_group(idx: int) -> Optional[Dict[str, Any]]:
            if idx < 0 or idx >= len(raw_outside_text_results):
                return None
            try:
                raw_bbox_tuple = tuple(int(c) for c in raw_outside_text_results[idx][0])
            except Exception:
                return None
            if len(raw_bbox_tuple) != 4:
                return None
            x1, y1, x2, y2 = raw_bbox_tuple
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))
            if x2 <= x1 or y2 <= y1:
                return None

            local_mask = injected_masks_by_index.get(idx)
            if local_mask is None:
                mask_bool = np.ones((y2 - y1, x2 - x1), dtype=bool)
            else:
                try:
                    mask_arr = np.asarray(local_mask)
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr[..., 0]
                    if mask_arr.shape[:2] != (y2 - y1, x2 - x1):
                        mask_arr = cv2.resize(
                            mask_arr.astype(np.uint8),
                            (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask_bool = mask_arr > 0
                except Exception:
                    mask_bool = np.ones((y2 - y1, x2 - x1), dtype=bool)
            if not np.any(mask_bool):
                return None

            combined_mask = np.zeros((img_h, img_w), dtype=bool)
            combined_mask[y1:y2, x1:x2] = mask_bool
            injected_metadata = injected_metadata_by_bbox.get(raw_bbox_tuple, {})
            is_speech_edge_split = bool(
                injected_metadata.get("speech_edge_split")
            )
            return {
                "combined_mask": combined_mask,
                "bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
                "original_bbox": {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                },
                "individual_masks": [combined_mask],
                "mask_indices": [idx],
                "confidence": float(raw_outside_text_results[idx][1]),
                "speech_edge_split": is_speech_edge_split,
            }

        if injected_masks_by_index:
            replaced_injected_indices = set()
            patched_mask_groups = []
            for group in mask_groups or []:
                mask_indices = list(group.get("mask_indices", []) or [])
                injected_indices = [
                    idx for idx in mask_indices if idx in injected_masks_by_index
                ]
                if len(mask_indices) == 1 and injected_indices:
                    replacement = build_injected_mask_group(injected_indices[0])
                    if replacement is not None:
                        patched_mask_groups.append(replacement)
                        replaced_injected_indices.add(injected_indices[0])
                        continue
                patched_mask_groups.append(group)

            for idx in injected_masks_by_index:
                if idx in replaced_injected_indices:
                    continue
                extra_group = build_injected_mask_group(idx)
                if extra_group is not None:
                    patched_mask_groups.append(extra_group)

            mask_groups = patched_mask_groups
            log_message(
                "Prepared separated speech-edge region(s) for standard OSB inpainting: "
                f"{len(injected_masks_by_index)}",
                verbose=verbose,
                always_print=True,
            )

        current_image = pil_image
        refinement_masks = []
        temp_files = []
        none_skipped_clip_bboxes = set()
        extracted_text_colors = {}
        group_render_bboxes: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}

        def remember_group_render_bbox(group_info, render_box):
            mask_indices = group_info.get("mask_indices", [])
            if not mask_indices:
                return
            try:
                rx1, ry1, rx2, ry2 = [int(round(float(v))) for v in render_box]
            except (TypeError, ValueError):
                return
            rx1 = max(0, min(img_w, rx1))
            ry1 = max(0, min(img_h, ry1))
            rx2 = max(0, min(img_w, rx2))
            ry2 = max(0, min(img_h, ry2))
            if rx2 <= rx1 or ry2 <= ry1:
                return
            for idx in mask_indices:
                if 0 <= idx < len(raw_outside_text_results):
                    raw_box = tuple(int(c) for c in raw_outside_text_results[idx][0])
                    group_render_bboxes[raw_box] = (rx1, ry1, rx2, ry2)

        try:
            if mask_groups:
                base_seed = (
                    random.randint(1, 999999)
                    if config.outside_text.seed == -1
                    else config.outside_text.seed
                )

                extracted_text_colors = {}
                flux_inpaints = 0
                cv2_inpaints = 0
                none_skips = 0
                group_flux_regions = bool(config.outside_text.flux_group_regions)
                grouped_flux_candidates = []
                for i, group in enumerate(mask_groups):
                    log_message(
                        f"Inpainting outside text region {i + 1}/{len(mask_groups)}",
                        verbose=verbose,
                    )
                    raw_combined_mask = group["combined_mask"].astype(bool)
                    expanded_buffer_conflict_mask = _dilate_bool_mask(
                        raw_combined_mask,
                        OSB_BUFFER_CONFLICT_FLUX_DILATION,
                    )
                    buffer_conflict = bool(
                        np.any(total_bubble_mask)
                        and np.any(
                            np.logical_and(
                                expanded_buffer_conflict_mask,
                                total_bubble_mask,
                            )
                        )
                    )
                    buffer_conflict_clip_bbox = None
                    speech_edge_mask_group = bool(group.get("speech_edge_split"))
                    speech_edge_rect_buffer_conflict = False
                    edge_sfx_exact_rect_buffer_conflict = False
                    guarded_wide_sfx_buffer_conflict = False
                    if buffer_conflict and speech_edge_mask_group:
                        speech_edge_rect_mask = (
                            _speech_edge_rect_mask_for_buffer_conflict(
                                group,
                                raw_combined_mask,
                                total_bubble_mask,
                                img_w,
                                img_h,
                            )
                        )
                        if speech_edge_rect_mask is not None:
                            combined_mask = speech_edge_rect_mask
                            speech_edge_rect_buffer_conflict = True
                            buffer_conflict_clip_bbox = _bbox_from_mask(
                                combined_mask,
                                img_w,
                                img_h,
                                padding=OSB_BUFFER_CONFLICT_CLIP_PADDING,
                            )
                            log_message(
                                "Speech-edge OSB mask touches speech-bubble protection buffer; "
                                "using rectangular bbox mask",
                                always_print=True,
                            )
                        else:
                            combined_mask = raw_combined_mask
                            buffer_conflict_clip_bbox = _bbox_from_mask(
                                combined_mask,
                                img_w,
                                img_h,
                                padding=0,
                            )
                            log_message(
                                "Speech-edge OSB mask touches speech-bubble protection buffer; "
                                "using exact injected text mask",
                                always_print=True,
                            )
                    elif buffer_conflict:
                        if group.get("edge_sfx_full_glyph"):
                            combined_mask = raw_combined_mask
                            edge_sfx_exact_rect_buffer_conflict = True
                            buffer_conflict_clip_bbox = _bbox_from_mask(
                                combined_mask,
                                img_w,
                                img_h,
                                padding=0,
                            )
                            log_message(
                                "Edge SFX touches speech-bubble protection; "
                                "using exact full-glyph rectangle without extra dilation",
                                always_print=True,
                            )
                        else:
                            (
                                guarded_wide_sfx_buffer_conflict,
                                wide_sfx_guard_reason,
                            ) = _osb_buffer_conflict_is_wide_sfx_like(
                                group,
                                raw_combined_mask,
                                img_w,
                                img_h,
                            )
                        if (
                            not edge_sfx_exact_rect_buffer_conflict
                            and guarded_wide_sfx_buffer_conflict
                        ):
                            exact_mask = np.logical_and(
                                raw_combined_mask, np.logical_not(total_bubble_mask)
                            )
                            combined_mask = (
                                exact_mask if np.any(exact_mask) else raw_combined_mask
                            )
                            buffer_conflict_clip_bbox = _bbox_from_mask(
                                combined_mask,
                                img_w,
                                img_h,
                                padding=0,
                            )
                            log_message(
                                "Wide OSB/SFX mask touches speech-bubble protection "
                                "buffer; using exact protected text mask "
                                f"({wide_sfx_guard_reason})",
                                always_print=True,
                            )
                        elif not edge_sfx_exact_rect_buffer_conflict:
                            combined_mask = expanded_buffer_conflict_mask
                            buffer_conflict_clip_bbox = _bbox_from_mask(
                                combined_mask,
                                img_w,
                                img_h,
                                padding=OSB_BUFFER_CONFLICT_CLIP_PADDING,
                            )
                            log_message(
                                "OSB mask touches speech-bubble protection buffer; "
                                "using expanded unprotected mask for this region",
                                always_print=True,
                            )
                    else:
                        combined_mask = np.logical_and(
                            raw_combined_mask, np.logical_not(total_bubble_mask)
                        )
                    combined_mask = np.logical_and(combined_mask, ~preserve_mask)
                    if not np.any(combined_mask):
                        log_message(
                            "Skipping outside text region after bubble masking (no remaining area)",
                            verbose=verbose,
                        )
                        continue
                    region_seed = base_seed + i if base_seed > 0 else base_seed

                    original_bbox_dict = group.get("original_bbox")
                    composite_clip_bbox = None
                    fill_color = None
                    fallback_fill_color = None
                    ox0 = oy0 = ox1 = oy1 = None
                    if original_bbox_dict:
                        ox = int(original_bbox_dict.get("x", 0))
                        oy = int(original_bbox_dict.get("y", 0))
                        ow = int(original_bbox_dict.get("width", 0))
                        oh = int(original_bbox_dict.get("height", 0))
                        if ow > 0 and oh > 0:
                            ox0 = max(0, min(img_w, ox))
                            oy0 = max(0, min(img_h, oy))
                            ox1 = max(0, min(img_w, ox + ow))
                            oy1 = max(0, min(img_h, oy + oh))
                            composite_clip_bbox = (ox, oy, ox + ow, oy + oh)
                            if (
                                buffer_conflict
                                and buffer_conflict_clip_bbox is not None
                            ):
                                composite_clip_bbox = buffer_conflict_clip_bbox

                            # Determine detected text color for this region to ensure contrast
                            group_bg_is_dark = None
                            if original_text_colors:
                                votes_dark = 0
                                votes_light = 0
                                gx1, gy1, gx2, gy2 = ox, oy, ox + ow, oy + oh

                                for (
                                    bx1,
                                    by1,
                                    bx2,
                                    by2,
                                ), t_dark in original_text_colors.items():
                                    # Check if center of OCR box is inside group box
                                    bcx = (bx1 + bx2) / 2
                                    bcy = (by1 + by2) / 2
                                    if (
                                        bcx >= gx1
                                        and bcx <= gx2
                                        and bcy >= gy1
                                        and bcy <= gy2
                                    ):
                                        if t_dark:
                                            votes_dark += 1
                                        else:
                                            votes_light += 1
                                if votes_dark > 0 or votes_light > 0:
                                    group_bg_is_dark = votes_dark >= votes_light

                                    # Detected value represents background brightness
                                    fallback_fill_color = (
                                        (0, 0, 0)
                                        if group_bg_is_dark
                                        else (255, 255, 255)
                                    )

                                    t_type = "Dark" if group_bg_is_dark else "Light"
                                    f_col = (
                                        "White"
                                        if fallback_fill_color == (255, 255, 255)
                                        else "Black"
                                    )
                                    log_message(
                                        f"OSB Region {i + 1}: Detected {t_type} background. "
                                        f"Fallback fill: {f_col}.",
                                        verbose=verbose,
                                    )

                            # Expanded sampling around the original bbox to find background color
                            mask_indices = group.get("mask_indices", [])
                            if mask_indices and raw_outside_text_results:
                                rx0 = int(
                                    min(
                                        [
                                            raw_outside_text_results[idx][0][0]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                                ry0 = int(
                                    min(
                                        [
                                            raw_outside_text_results[idx][0][1]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                                rx1 = int(
                                    max(
                                        [
                                            raw_outside_text_results[idx][0][2]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                                ry1 = int(
                                    max(
                                        [
                                            raw_outside_text_results[idx][0][3]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                            else:
                                rx0, ry0, rx1, ry1 = ox, oy, ox + ow, oy + oh

                            expansion_px = 6
                            sx1 = max(0, rx0 - expansion_px)
                            sy1 = max(0, ry0 - expansion_px)
                            sx2 = min(img_w, rx1 + expansion_px)
                            sy2 = min(img_h, ry1 + expansion_px)

                            if sx2 > sx1 and sy2 > sy1:
                                mask_h, mask_w = sy2 - sy1, sx2 - sx1
                                local_mask = np.ones((mask_h, mask_w), dtype=bool)

                                lx0 = max(0, rx0 - sx1)
                                ly0 = max(0, ry0 - sy1)
                                lx1 = min(mask_w, rx1 - sx1)
                                ly1 = min(mask_h, ry1 - sy1)

                                if lx1 > lx0 and ly1 > ly0:
                                    local_mask[ly0:ly1, lx0:lx1] = False

                                border_pixels = None
                                min_border_pixels = 20
                                if np.count_nonzero(local_mask) >= min_border_pixels:
                                    sampling_crop = current_image.crop(
                                        (sx1, sy1, sx2, sy2)
                                    )
                                    crop_np = np.array(sampling_crop.convert("RGB"))
                                    border_pixels = crop_np[local_mask]

                                if border_pixels is not None and border_pixels.size > 0:
                                    # Calculate text color using LAB contrast thresholding
                                    bg_rgb = np.median(border_pixels, axis=0).astype(
                                        np.uint8
                                    )
                                    bg_lab = cv2.cvtColor(
                                        np.uint8([[bg_rgb]]), cv2.COLOR_RGB2LAB
                                    )[0][0]

                                    crop_rgb = np.array(
                                        pil_image.crop((rx0, ry0, rx1, ry1)).convert(
                                            "RGB"
                                        )
                                    )
                                    crop_lab = cv2.cvtColor(
                                        crop_rgb, cv2.COLOR_RGB2LAB
                                    ).astype(np.float32)
                                    dist_map = np.linalg.norm(
                                        crop_lab - bg_lab.astype(np.float32), axis=2
                                    )

                                    robust_max_dist = np.percentile(dist_map, 95)
                                    CONTRAST_THRESHOLD = max(30, robust_max_dist * 0.6)
                                    contrast_mask = (
                                        dist_map > CONTRAST_THRESHOLD
                                    ).astype(np.uint8) * 255

                                    kernel_3 = np.ones((3, 3), np.uint8)
                                    contrast_mask = cv2.morphologyEx(
                                        contrast_mask, cv2.MORPH_CLOSE, kernel_3
                                    )
                                    contrast_mask = cv2.erode(
                                        contrast_mask,
                                        np.ones((2, 2), np.uint8),
                                        iterations=1,
                                    )

                                    contours, _ = cv2.findContours(
                                        contrast_mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )
                                    clean_mask = np.zeros_like(contrast_mask)
                                    MIN_COMPONENT_AREA = 4
                                    for cnt in contours:
                                        if cv2.contourArea(cnt) >= MIN_COMPONENT_AREA:
                                            cv2.drawContours(
                                                clean_mask, [cnt], -1, 255, cv2.FILLED
                                            )

                                    text_pixels_rgb = crop_rgb[clean_mask == 255]
                                    if len(text_pixels_rgb) >= 10:
                                        text_color_rgb = tuple(
                                            np.median(text_pixels_rgb, axis=0).astype(
                                                int
                                            )
                                        )
                                        hsv = cv2.cvtColor(
                                            np.uint8([[text_color_rgb]]),
                                            cv2.COLOR_RGB2HSV,
                                        )[0][0]
                                        if hsv[1] < 25:
                                            text_color_rgb = (
                                                (0, 0, 0)
                                                if hsv[2] < 128
                                                else (255, 255, 255)
                                            )
                                        extracted_text_colors[composite_clip_bbox] = (
                                            text_color_rgb
                                        )

                                    white_thresh = 250
                                    black_thresh = 5
                                    ratio_threshold = 0.99

                                    white_ratio = np.mean(
                                        np.all(border_pixels >= white_thresh, axis=1)
                                    )
                                    black_ratio = np.mean(
                                        np.all(border_pixels <= black_thresh, axis=1)
                                    )

                                    if fallback_fill_color is None:
                                        fallback_fill_color = (
                                            (255, 255, 255)
                                            if white_ratio >= black_ratio
                                            else (0, 0, 0)
                                        )

                                    force_fill = inpainting_method == "opencv"

                                    # Get expanded bounds for this group to check solid color and for cv2 fill
                                    p_x0, p_y0, p_x1, p_y1 = ox, oy, ox + ow, oy + oh
                                    mask_indices = group.get("mask_indices", [])
                                    if mask_indices and outside_text_results:
                                        p_x0 = max(
                                            0,
                                            int(
                                                min(
                                                    [
                                                        outside_text_results[idx][0][0]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )
                                        p_y0 = max(
                                            0,
                                            int(
                                                min(
                                                    [
                                                        outside_text_results[idx][0][1]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )
                                        p_x1 = min(
                                            img_w,
                                            int(
                                                max(
                                                    [
                                                        outside_text_results[idx][0][2]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )
                                        p_y1 = min(
                                            img_h,
                                            int(
                                                max(
                                                    [
                                                        outside_text_results[idx][0][3]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )

                                    force_fill = inpainting_method == "opencv"

                                    # Simply check if the expanded boundary is solid color
                                    expanded_is_solid = False
                                    if not force_fill:
                                        ex_sx1 = max(0, p_x0 - expansion_px)
                                        ex_sy1 = max(0, p_y0 - expansion_px)
                                        ex_sx2 = min(img_w, p_x1 + expansion_px)
                                        ex_sy2 = min(img_h, p_y1 + expansion_px)

                                        if ex_sx2 > ex_sx1 and ex_sy2 > ex_sy1:
                                            # Grab boundary pixels using Numpy directly for speed
                                            crop_img = current_image.crop(
                                                (ex_sx1, ex_sy1, ex_sx2, ex_sy2)
                                            )
                                            ecrop_np = np.array(crop_img.convert("RGB"))
                                            elocal_mask = np.ones(
                                                ecrop_np.shape[:2], dtype=bool
                                            )

                                            ix1 = max(0, p_x0 - ex_sx1)
                                            iy1 = max(0, p_y0 - ex_sy1)
                                            ix2 = min(ecrop_np.shape[1], p_x1 - ex_sx1)
                                            iy2 = min(ecrop_np.shape[0], p_y1 - ex_sy1)

                                            if ix2 > ix1 and iy2 > iy1:
                                                elocal_mask[iy1:iy2, ix1:ix2] = False

                                            eborder_pixels = ecrop_np[elocal_mask]
                                            if eborder_pixels.size > 0:
                                                ewhite_ratio = np.mean(
                                                    np.all(
                                                        eborder_pixels >= white_thresh,
                                                        axis=1,
                                                    )
                                                )
                                                eblack_ratio = np.mean(
                                                    np.all(
                                                        eborder_pixels <= black_thresh,
                                                        axis=1,
                                                    )
                                                )
                                                if (
                                                    ewhite_ratio >= ratio_threshold
                                                    or eblack_ratio >= ratio_threshold
                                                ):
                                                    expanded_is_solid = True

                                    should_simple_fill = expanded_is_solid or force_fill

                                    if should_simple_fill:
                                        fill_color = fallback_fill_color

                                        if force_fill and not (
                                            white_ratio >= ratio_threshold
                                            or black_ratio >= ratio_threshold
                                        ):
                                            log_message(
                                                "Forcing CV2 fill: defaulting to "
                                                f"{'white' if fill_color == (255, 255, 255) else 'black'} background",
                                                verbose=verbose,
                                            )
                                        else:
                                            log_message(
                                                "Skipping Flux for OSB region: detected pure "
                                                f"{'white' if fill_color == (255, 255, 255) else 'black'} background",
                                                verbose=verbose,
                                            )

                    def apply_simple_fill(color_to_use):
                        new_img = current_image.copy()
                        if buffer_conflict:
                            paint_bbox = _bbox_from_mask(
                                combined_mask,
                                img_w,
                                img_h,
                                padding=0,
                            )
                            if paint_bbox is not None:
                                px0, py0, px1, py1 = paint_bbox
                                remember_group_render_bbox(
                                    group, (px0, py0, px1, py1)
                                )
                                refined_mask = None
                                if not (
                                    group.get("speech_edge_split")
                                    or edge_sfx_exact_rect_buffer_conflict
                                ):
                                    refined_mask = _large_complex_osb_stroke_fill_mask(
                                        current_image,
                                        (px0, py0, px1, py1),
                                        img_w,
                                        img_h,
                                        total_bubble_mask,
                                    )
                                if refined_mask is not None:
                                    region_mask = refined_mask[py0:py1, px0:px1]
                                    log_message(
                                        "Refined large OSB simple fill to text-stroke mask "
                                        f"for {(px0, py0, px1, py1)}",
                                        always_print=True,
                                    )
                                else:
                                    region_mask = combined_mask[py0:py1, px0:px1]
                                region_mask = np.logical_and(
                                    region_mask, ~preserve_mask[py0:py1, px0:px1]
                                )
                                if np.any(region_mask):
                                    mask_pil = Image.fromarray(
                                        (region_mask * 255).astype(np.uint8),
                                        mode="L",
                                    )
                                    patch = Image.new(
                                        "RGB", (px1 - px0, py1 - py0), color_to_use
                                    )
                                    new_img.paste(patch, (px0, py0), mask=mask_pil)
                                    if speech_edge_rect_buffer_conflict:
                                        log_message(
                                            "Applied rectangular speech-edge "
                                            "OSB mask with OpenCV fill",
                                            always_print=True,
                                        )
                                    elif group.get("speech_edge_split"):
                                        log_message(
                                            "Applied exact speech-edge OSB mask "
                                            "with OpenCV fill",
                                            always_print=True,
                                        )
                                    elif guarded_wide_sfx_buffer_conflict:
                                        log_message(
                                            "Applied exact wide OSB/SFX buffer-conflict "
                                            "mask with OpenCV fill",
                                            always_print=True,
                                        )
                                    elif edge_sfx_exact_rect_buffer_conflict:
                                        log_message(
                                            "Applied exact full-glyph edge-SFX rectangle "
                                            "with OpenCV fill",
                                            always_print=True,
                                        )
                                    else:
                                        log_message(
                                            "Applied expanded OSB buffer-conflict mask "
                                            "with OpenCV fill",
                                            always_print=True,
                                        )
                            return new_img

                        mask_indices = group.get("mask_indices", [])
                        if mask_indices and outside_text_results:
                            p_x0 = max(
                                0,
                                int(
                                    min(
                                        [
                                            outside_text_results[idx][0][0]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                            p_y0 = max(
                                0,
                                int(
                                    min(
                                        [
                                            outside_text_results[idx][0][1]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                            p_x1 = min(
                                img_w,
                                int(
                                    max(
                                        [
                                            outside_text_results[idx][0][2]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                            p_y1 = min(
                                img_h,
                                int(
                                    max(
                                        [
                                            outside_text_results[idx][0][3]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                        elif (
                            original_bbox_dict
                            and ox1 is not None
                            and ox0 is not None
                            and oy1 is not None
                            and oy0 is not None
                        ):
                            p_x0, p_y0, p_x1, p_y1 = ox0, oy0, ox1, oy1
                        else:
                            # Full mask fill fallback
                            mask_pil = Image.fromarray(
                                (combined_mask * 255).astype(np.uint8), mode="L"
                            )
                            patch = Image.new("RGB", new_img.size, color_to_use)
                            new_img.paste(patch, (0, 0), mask=mask_pil)
                            return new_img

                        if p_x1 > p_x0 and p_y1 > p_y0:
                            remember_group_render_bbox(
                                group, (p_x0, p_y0, p_x1, p_y1)
                            )
                            refined_mask = _large_complex_osb_stroke_fill_mask(
                                current_image,
                                (p_x0, p_y0, p_x1, p_y1),
                                img_w,
                                img_h,
                                total_bubble_mask,
                            )
                            if refined_mask is not None:
                                rect_mask = refined_mask
                                log_message(
                                    "Refined large OSB rectangle fill to text-stroke mask "
                                    f"for {(p_x0, p_y0, p_x1, p_y1)}",
                                    always_print=True,
                                )
                            else:
                                # Create a solid rectangle mask for the expanded bounds, but exclude speech bubbles
                                rect_mask = np.zeros((img_h, img_w), dtype=bool)
                                rect_mask[p_y0:p_y1, p_x0:p_x1] = True
                                rect_mask = np.logical_and(
                                    rect_mask, np.logical_not(total_bubble_mask)
                                )

                            region_mask = np.logical_and(
                                rect_mask[p_y0:p_y1, p_x0:p_x1],
                                ~preserve_mask[p_y0:p_y1, p_x0:p_x1],
                            )
                            if np.any(region_mask):
                                mask_pil = Image.fromarray(
                                    (region_mask * 255).astype(np.uint8), mode="L"
                                )
                                patch = Image.new(
                                    "RGB", (p_x1 - p_x0, p_y1 - p_y0), color_to_use
                                )
                                new_img.paste(patch, (p_x0, p_y0), mask=mask_pil)

                        return new_img

                    if (
                        buffer_conflict
                        and fill_color is not None
                        and inpainter is not None
                        and inpainting_method != "opencv"
                    ):
                        mask_description = (
                            "rectangular speech-edge mask"
                            if speech_edge_rect_buffer_conflict
                            else (
                                "exact full-glyph rectangle"
                                if edge_sfx_exact_rect_buffer_conflict
                                else "expanded mask"
                            )
                        )
                        log_message(
                            "Buffer-conflict OSB would normally use simple fill, "
                            f"but Flux is available; forcing Flux with {mask_description}",
                            always_print=True,
                        )
                        fill_color = None

                    if fill_color is not None:
                        current_image = apply_simple_fill(fill_color)
                        cv2_inpaints += 1
                        continue

                    if inpainting_method == "none":
                        if composite_clip_bbox:
                            none_skipped_clip_bboxes.add(composite_clip_bbox)
                        none_skips += 1
                        log_message(
                            f"Skipping inpaint for non-solid OSB region {i + 1} (none mode)",
                            verbose=verbose,
                        )
                        continue

                    if group_flux_regions and inpainter is not None:
                        grouped_mask = combined_mask.copy()
                        if composite_clip_bbox is not None:
                            clip_x1, clip_y1, clip_x2, clip_y2 = composite_clip_bbox
                            clip_x1 = max(0, min(img_w, clip_x1))
                            clip_x2 = max(0, min(img_w, clip_x2))
                            clip_y1 = max(0, min(img_h, clip_y1))
                            clip_y2 = max(0, min(img_h, clip_y2))

                            clipped_grouped_mask = np.zeros_like(grouped_mask)
                            if clip_x2 > clip_x1 and clip_y2 > clip_y1:
                                clipped_grouped_mask[
                                    clip_y1:clip_y2, clip_x1:clip_x2
                                ] = grouped_mask[clip_y1:clip_y2, clip_x1:clip_x2]
                            grouped_mask = clipped_grouped_mask

                        if not np.any(grouped_mask):
                            log_message(
                                f"Skipping OSB region {i + 1} after grouped clip (no remaining area)",
                                verbose=verbose,
                            )
                            continue

                        grouped_flux_candidates.append(
                            {
                                "index": i + 1,
                                "mask": grouped_mask,
                                "refine_boundary": not buffer_conflict,
                                "bbox": _bbox_from_mask(
                                    grouped_mask, img_w, img_h, padding=0
                                ),
                                "fallback_color": (
                                    fallback_fill_color
                                    if fallback_fill_color
                                    else (255, 255, 255)
                                ),
                            }
                        )
                        log_message(
                            f"Queued OSB region {i + 1} for grouped Flux inpainting",
                            verbose=verbose,
                        )
                        continue

                    flux_failed = False
                    flux_fail_reason = None
                    inpainted_image = None

                    if inpainter is None:
                        flux_failed = True
                        flux_fail_reason = "Flux inpainter unavailable"
                    else:
                        try:
                            inpainted_image = inpainter.inpaint_mask(
                                current_image,
                                combined_mask,
                                seed=region_seed,
                                verbose=verbose,
                                strict_mask_clipping=True,
                                composite_clip_bbox=composite_clip_bbox,
                            )
                            if inpainted_image is current_image:
                                flux_failed = True
                                flux_fail_reason = (
                                    "Flux returned original image (no inpaint)"
                                )
                        except Exception as e:
                            flux_failed = True
                            flux_fail_reason = f"Flux inpainting error: {e}"

                    if flux_failed:
                        fallback_color_to_use = (
                            fallback_fill_color
                            if fallback_fill_color
                            else (255, 255, 255)
                        )
                        log_message(
                            f"Flux failed for OSB region {i + 1}"
                            + (f" ({flux_fail_reason})" if flux_fail_reason else "")
                            + f"; falling back to CV2 fill ({fallback_color_to_use})",
                            always_print=True,
                        )
                        current_image = apply_simple_fill(fallback_color_to_use)
                        cv2_inpaints += 1
                        continue

                    flux_inpaints += 1
                    if not buffer_conflict:
                        refinement_mask = combined_mask.copy()
                        if composite_clip_bbox is not None:
                            cx1, cy1, cx2, cy2 = composite_clip_bbox
                            clipped = np.zeros_like(refinement_mask)
                            clipped[max(0, cy1):min(img_h, cy2), max(0, cx1):min(img_w, cx2)] = refinement_mask[
                                max(0, cy1):min(img_h, cy2), max(0, cx1):min(img_w, cx2)
                            ]
                            refinement_mask = clipped
                        refinement_masks.append(refinement_mask)
                    # Save to disk if more regions remain to reduce memory usage
                    if i < len(mask_groups) - 1:
                        temp_file = None
                        try:
                            temp_fd, temp_file = tempfile.mkstemp(suffix=".png")
                            os.close(temp_fd)
                            inpainted_image.save(temp_file, format="PNG")
                            temp_files.append(temp_file)

                            with Image.open(temp_file) as img_tmp:
                                img_tmp.load()
                                current_image = img_tmp.copy()

                            del inpainted_image
                            gc.collect()
                            log_message(
                                "Saved intermediate inpainting result to disk",
                                verbose=verbose,
                            )
                        except Exception as e:
                            log_message(
                                "Warning: Failed to save intermediate image to disk: "
                                f"{e}. Continuing with in-memory processing.",
                                verbose=verbose,
                            )
                            # Fallback to in-memory if disk save fails
                            current_image = inpainted_image
                            if temp_file and temp_file in temp_files:
                                temp_files.remove(temp_file)
                    else:
                        current_image = inpainted_image

                if grouped_flux_candidates:
                    grouped_flux_batches = [grouped_flux_candidates]
                    total_grouped_flux_batches = len(grouped_flux_batches)
                    for batch_index, flux_batch in enumerate(
                        grouped_flux_batches, start=1
                    ):
                        grouped_mask = np.zeros((img_h, img_w), dtype=bool)
                        for candidate in flux_batch:
                            grouped_mask = np.logical_or(
                                grouped_mask, candidate["mask"]
                            )

                        if not np.any(grouped_mask):
                            continue

                        batch_suffix = (
                            f" batch {batch_index}/{total_grouped_flux_batches}"
                            if total_grouped_flux_batches > 1
                            else ""
                        )
                        log_message(
                            "Running grouped Flux inpainting"
                            f"{batch_suffix} for {len(flux_batch)} OSB regions",
                            verbose=verbose,
                        )
                        try:
                            group_seed = base_seed if base_seed > 0 else base_seed
                            grouped_luminance_correction = isinstance(
                                inpainter, FluxKleinInpainter
                            ) and bool(
                                getattr(inpainter, "luminance_correction", False)
                            )
                            if grouped_luminance_correction:
                                log_message(
                                    "Using per-region luminance correction for grouped Klein Flux",
                                    verbose=verbose,
                                )

                            inpainted_image = inpainter.inpaint_mask(
                                current_image,
                                grouped_mask,
                                seed=group_seed,
                                verbose=verbose,
                                strict_mask_clipping=True,
                                ocr_params={
                                    "type": "outside_text_group",
                                    "regions": len(flux_batch),
                                    "batch": batch_index,
                                    "batches": total_grouped_flux_batches,
                                    "group_luminance": (
                                        "per_region"
                                        if grouped_luminance_correction
                                        else "off"
                                    ),
                                },
                                luminance_match_masks=(
                                    [candidate["mask"] for candidate in flux_batch]
                                    if grouped_luminance_correction
                                    else None
                                ),
                            )

                            if inpainted_image is current_image:
                                raise RuntimeError(
                                    "Flux returned original image (no inpaint)"
                                )
                            current_image = inpainted_image
                            refinement_masks.extend(
                                candidate["mask"] for candidate in flux_batch
                                if candidate["refine_boundary"]
                            )
                            flux_inpaints += len(flux_batch)
                        except Exception as e:
                            log_message(
                                "Grouped Flux inpainting failed "
                                f"({e}); falling back to CV2 fill",
                                always_print=True,
                            )
                            for candidate in flux_batch:
                                mask_pil = Image.fromarray(
                                    (candidate["mask"] * 255).astype(np.uint8),
                                    mode="L",
                                )
                                patch = Image.new(
                                    "RGB",
                                    current_image.size,
                                    candidate["fallback_color"],
                                )
                                next_image = current_image.copy()
                                next_image.paste(patch, (0, 0), mask=mask_pil)
                                current_image = next_image
                            cv2_inpaints += len(flux_batch)

                log_message("Outside text inpainting completed", verbose=verbose)
                parts = [
                    f"Flux: {flux_inpaints}",
                    f"CV2: {cv2_inpaints}",
                ]
                if none_skips:
                    parts.append(f"Skipped (none): {none_skips}")
                log_message(
                    f"Inpainted {len(mask_groups)} outside text regions ({', '.join(parts)})",
                    always_print=True,
                )
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass

        current_image = finish_outside_inpaint(
            pil_image, current_image, refinement_masks,
            protected=np.logical_or(
                total_bubble_mask, _dilate_bool_mask(preserve_mask, 15)
            ),
        )
        outside_text_data = _build_outside_text_translation_data(
            pil_image=pil_image,
            config=config,
            image_format=image_format,
            outside_text_results=outside_text_results,
            raw_outside_text_results=raw_outside_text_results,
            original_text_colors=original_text_colors,
            bubble_boxes=bubble_boxes,
            bubble_core_mask=bubble_core_mask,
            img_w=img_w,
            img_h=img_h,
            verbose=verbose,
            group_render_bboxes=group_render_bboxes,
            extracted_text_colors=extracted_text_colors,
            none_skipped_clip_bboxes=none_skipped_clip_bboxes,
            allowed_bubble_overlap_bboxes=allowed_bubble_overlap_bboxes,
            edge_sfx_full_glyph_bboxes=edge_sfx_expanded_bboxes,
            injected_metadata_by_bbox=injected_metadata_by_bbox,
        )

        if np.any(preserve_mask):
            # Last shared compositor guard covers grouped Flux and every fallback.
            preserved = np.asarray(current_image.convert(pil_image.mode)).copy()
            source = np.asarray(pil_image)
            preserved[preserve_mask] = source[preserve_mask]
            current_image = Image.fromarray(preserved)
        if selected_bboxes is not None:
            for item in outside_text_data:
                if tuple(item["original_bbox"]) not in selected_bboxes:
                    item.update(
                        skip_render=True, skip_reason="outside_text_processing_mode",
                        outside_text_preserve_bbox=item["bbox"],
                    )
        return current_image, outside_text_data

    except (TranslationError, CancellationError):
        raise
    except Exception as e:
        log_message(
            f"Error during outside text detection/inpainting: {e}",
            always_print=True,
        )
        return pil_image, []
