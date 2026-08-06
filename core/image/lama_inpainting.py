from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from utils.logging import log_message

_HGMZHN_RUNTIME = (
    Path(__file__).resolve().parents[2] / "advanced_editor" / "hgmzhn_runtime"
)

DETECTOR_HOLE_SEED_MIN_ASPECT = 2.5
DETECTOR_HOLE_SEED_MIN_BASE_COVERAGE = 0.70
DETECTOR_HOLE_SEED_MIN_RATIO = 0.20
DETECTOR_HOLE_SEED_MAX_RATIO = 0.45
DETECTOR_HOLE_SEED_MIN_PIXELS = 256

# hgmzhn keeps one global inpainter instance per backend. Parallel batch
# workers must not unload that shared instance while another worker dispatches.
_LAMA_RUNTIME_LOCK = threading.RLock()


def build_detector_text_hole_seed_mask(
    bubble_mask: np.ndarray,
    processed_mask: np.ndarray,
    text_boxes: list | tuple | None,
    detection_bbox: tuple | list | None,
) -> np.ndarray:
    """Recover source glyph holes only for a strict tall-bubble failure case."""
    bubble = np.where(np.asarray(bubble_mask) > 0, 255, 0).astype(np.uint8)
    if bubble.ndim == 3:
        bubble = bubble[..., 0]
    processed = np.where(np.asarray(processed_mask) > 0, 255, 0).astype(np.uint8)
    if processed.ndim == 3:
        processed = processed[..., 0]
    empty = np.zeros_like(bubble)
    if (
        bubble.shape != processed.shape
        or len(text_boxes or []) != 1
        or not np.any(bubble)
    ):
        return empty

    x, y, width, height = cv2.boundingRect((bubble > 0).astype(np.uint8))
    if width <= 0 or height <= 0:
        return empty
    if detection_bbox and len(detection_bbox) >= 4:
        try:
            box_width = max(
                1.0,
                float(detection_bbox[2]) - float(detection_bbox[0]),
            )
            box_height = max(
                1.0,
                float(detection_bbox[3]) - float(detection_bbox[1]),
            )
        except (TypeError, ValueError):
            return empty
    else:
        box_width = float(width)
        box_height = float(height)
    if box_height / box_width < DETECTOR_HOLE_SEED_MIN_ASPECT:
        return empty

    bubble_area = int(np.count_nonzero(bubble))
    bounds_area = max(1, width * height)
    if (
        bubble_area / float(bounds_area)
        < DETECTOR_HOLE_SEED_MIN_BASE_COVERAGE
    ):
        return empty

    boxes_mask = np.zeros_like(bubble)
    image_height, image_width = bubble.shape[:2]
    for box in text_boxes:
        try:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        except (TypeError, ValueError):
            continue
        x0 = max(0, min(image_width, x0))
        y0 = max(0, min(image_height, y0))
        x1 = max(x0, min(image_width, x1))
        y1 = max(y0, min(image_height, y1))
        if x1 > x0 and y1 > y0:
            boxes_mask[y0:y1, x0:x1] = 255
    if not np.any(boxes_mask):
        return empty

    safe_bubble = cv2.erode(
        bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    if not np.any(safe_bubble):
        safe_bubble = bubble
    holes = np.where(
        (bubble > 0)
        & (processed == 0)
        & (boxes_mask > 0)
        & (safe_bubble > 0),
        255,
        0,
    ).astype(np.uint8)
    hole_count = int(np.count_nonzero(holes))
    hole_ratio = hole_count / float(max(1, bubble_area))
    if (
        hole_count < DETECTOR_HOLE_SEED_MIN_PIXELS
        or hole_ratio < DETECTOR_HOLE_SEED_MIN_RATIO
        or hole_ratio > DETECTOR_HOLE_SEED_MAX_RATIO
    ):
        return empty
    return holes


def build_lama_inpaint_mask(
    text_mask: np.ndarray,
    bubble_mask: np.ndarray,
    kernel_size: int,
    dilation_offset: int,
) -> np.ndarray:
    """Expand text strokes using hgmzhn's adaptive mask-refinement sizing."""
    text = np.where(np.asarray(text_mask) > 0, 255, 0).astype(np.uint8)
    bubble = np.where(np.asarray(bubble_mask) > 0, 255, 0).astype(np.uint8)
    if text.ndim == 3:
        text = text[..., 0]
    if bubble.ndim == 3:
        bubble = bubble[..., 0]
    if text.shape != bubble.shape:
        raise ValueError("Text and bubble masks must have the same dimensions.")

    safe_bubble = cv2.erode(
        bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if not np.any(safe_bubble):
        safe_bubble = bubble

    size = max(1, min(31, int(kernel_size)))
    if size % 2 == 0:
        size += 1
    offset = max(0, min(256, int(dilation_offset)))

    expanded = np.zeros_like(text)
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        text,
        connectivity=8,
    )
    for label_id in range(1, label_count):
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue

        minor_side = min(width, height)
        major_side = max(width, height)
        if major_side >= minor_side * 2.5:
            # Long, thin glyphs such as 一 need context based on their full
            # stroke length; using only thickness leaves source-shaped edges.
            text_size = max(3, min(major_side, 64))
        else:
            text_size = max(3, min(minor_side, 64))
        adaptive_size = max(
            (int((text_size + offset) * 0.3) // 2) * 2 + 1,
            3,
        )
        adaptive_size = min(adaptive_size, 127)
        component = np.where(labels == label_id, 255, 0).astype(np.uint8)
        component = cv2.dilate(
            component,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (adaptive_size, adaptive_size),
            ),
            iterations=1,
        )
        expanded = cv2.bitwise_or(expanded, component)

    if size > 1 and np.any(expanded):
        expanded = cv2.dilate(
            expanded,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=1,
        )

    return cv2.bitwise_and(expanded, safe_bubble)


def build_lama_text_seed_mask(
    image_gray: np.ndarray,
    bubble_mask: np.ndarray,
    text_boxes: list | tuple | None,
    *,
    text_is_dark: bool,
    protect_background_artwork: bool = False,
) -> np.ndarray:
    """Extract text strokes while excluding bubble outlines and crossing artwork."""
    gray = np.asarray(image_gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray, dtype=np.uint8)

    bubble = np.where(np.asarray(bubble_mask) > 0, 255, 0).astype(np.uint8)
    if bubble.ndim == 3:
        bubble = bubble[..., 0]
    if gray.shape != bubble.shape:
        raise ValueError("Image and bubble mask must have the same dimensions.")

    safe_bubble = cv2.erode(
        bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    if not np.any(safe_bubble):
        safe_bubble = bubble

    bubble_area = max(1, int(np.count_nonzero(safe_bubble)))
    bubble_x, bubble_y, bubble_w, bubble_h = cv2.boundingRect(
        (safe_bubble > 0).astype(np.uint8)
    )
    normalized_boxes: list[tuple[int, int, int, int]] = []
    for box in text_boxes or []:
        try:
            x0, y0, x1, y1 = [int(round(float(value))) for value in box[:4]]
        except (TypeError, ValueError):
            continue
        x0 = max(0, min(gray.shape[1] - 1, x0))
        y0 = max(0, min(gray.shape[0] - 1, y0))
        x1 = max(x0 + 1, min(gray.shape[1], x1))
        y1 = max(y0 + 1, min(gray.shape[0], y1))
        box_area = max(1, (x1 - x0) * (y1 - y0))
        inside_area = int(np.count_nonzero(safe_bubble[y0:y1, x0:x1]))
        inside_ratio = inside_area / float(box_area)
        if inside_ratio < 0.25:
            continue
        if box_area > bubble_area * 2.0:
            continue
        if box_area > bubble_area * 1.25 and inside_ratio < 0.50:
            continue
        center_x = min(gray.shape[1] - 1, max(0, (x0 + x1) // 2))
        center_y = min(gray.shape[0] - 1, max(0, (y0 + y1) // 2))
        if (
            safe_bubble[center_y, center_x] == 0
            and inside_ratio < 0.65
        ):
            continue
        normalized_boxes.append((x0, y0, x1, y1))

    detector_guided = bool(normalized_boxes)
    if not normalized_boxes and bubble_w > 0 and bubble_h > 0:
        normalized_boxes.append(
            (bubble_x, bubble_y, bubble_x + bubble_w, bubble_y + bubble_h)
        )

    candidate = np.zeros_like(gray, dtype=np.uint8)
    for x0, y0, x1, y1 in normalized_boxes:
        roi_gray = gray[y0:y1, x0:x1]
        roi_safe = safe_bubble[y0:y1, x0:x1]
        sample_values = roi_gray[roi_safe > 0]
        if sample_values.size < 16:
            continue

        threshold, _ = cv2.threshold(
            sample_values,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        median = float(np.median(sample_values))
        if text_is_dark:
            ink_limit = min(float(threshold), median - 18.0, 210.0)
            roi_candidate = roi_gray <= max(8.0, ink_limit)
        else:
            ink_limit = max(float(threshold), median + 18.0, 200.0)
            roi_candidate = roi_gray >= min(247.0, ink_limit)

        roi_out = candidate[y0:y1, x0:x1]
        roi_out[(roi_candidate) & (roi_safe > 0)] = 255

    if not np.any(candidate):
        return candidate

    # A fixed short kernel mistakes wide Japanese/Korean glyph strokes for
    # crossing artwork on small bubbles. Only remove lines that span most of
    # the trusted detector box (or a substantial part of the bubble).
    widest_text_box = max(
        (x1 - x0 for x0, _y0, x1, _y1 in normalized_boxes),
        default=0,
    )
    horizontal_size = max(
        15,
        int(round(max(1, bubble_w) * 0.18)),
        int(round(max(1, widest_text_box) * 0.65)),
    )
    horizontal_lines = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        np.ones((1, horizontal_size), dtype=np.uint8),
    )
    candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(horizontal_lines))

    boundary_band = cv2.subtract(
        safe_bubble,
        cv2.erode(
            safe_bubble,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ),
    )
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )
    kept = np.zeros_like(candidate)
    max_component_area = max(96, int(bubble_area * 0.18))
    for label_id in range(1, label_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area < 2 or area > max_component_area:
            continue
        if max(width, height) / float(max(1, min(width, height))) > 12.0:
            continue
        component = labels == label_id
        if (
            not detector_guided
            and np.any(boundary_band[component] > 0)
            and max(width, height) > 8
        ):
            continue
        kept[component] = 255

    if protect_background_artwork and detector_guided and np.any(kept):
        kept_count, kept_labels, kept_stats, _ = cv2.connectedComponentsWithStats(
            kept,
            connectivity=8,
        )
        components = []
        for label_id in range(1, kept_count):
            area = int(kept_stats[label_id, cv2.CC_STAT_AREA])
            width = int(kept_stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(kept_stats[label_id, cv2.CC_STAT_HEIGHT])
            components.append((area, width, height, label_id))

        components.sort(reverse=True)
        if len(components) >= 2:
            largest_area, largest_width, largest_height, largest_id = components[0]
            second_area = max(1, components[1][0])
            largest_bbox_area = largest_width * largest_height
            dominant_artwork = (
                largest_area >= max(512, int(bubble_area * 0.06))
                and largest_bbox_area >= int(bubble_area * 0.10)
                and largest_area >= second_area * 4
            )
            if dominant_artwork:
                kept[kept_labels == largest_id] = 0

    return cv2.bitwise_and(kept, safe_bubble)


def build_lama_contrast_outline_seed(
    image_gray: np.ndarray,
    bubble_mask: np.ndarray,
    text_seed: np.ndarray,
    *,
    text_is_dark: bool,
    padding: int = 4,
) -> np.ndarray:
    """Keep a contrasting text outline only next to trusted text strokes."""
    gray = np.asarray(image_gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = np.asarray(gray, dtype=np.uint8)
    bubble = np.where(np.asarray(bubble_mask) > 0, 255, 0).astype(np.uint8)
    seed = np.where(np.asarray(text_seed) > 0, 255, 0).astype(np.uint8)
    if bubble.ndim == 3:
        bubble = bubble[..., 0]
    if seed.ndim == 3:
        seed = seed[..., 0]
    if gray.shape != bubble.shape or gray.shape != seed.shape:
        raise ValueError("Image, bubble, and text seed masks must match.")
    if not np.any(seed):
        return np.zeros_like(gray)

    radius = max(1, min(12, int(padding)))
    nearby = cv2.dilate(
        seed,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        ),
        iterations=1,
    )
    safe_bubble = cv2.erode(
        bubble,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    if not np.any(safe_bubble):
        safe_bubble = bubble

    if text_is_dark:
        contrast = np.where(gray >= 242, 255, 0).astype(np.uint8)
    else:
        contrast = np.where(gray <= 24, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(contrast, cv2.bitwise_and(nearby, safe_bubble))


def _ensure_hgmzhn_runtime() -> None:
    if not _HGMZHN_RUNTIME.is_dir():
        raise RuntimeError(
            "LAMA Large runtime is missing. Reinstall the advanced editor files."
        )
    runtime = str(_HGMZHN_RUNTIME)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)


def _run_async(coro: Any) -> Any:
    """Run an async hgmzhn inference call from a synchronous pipeline worker."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def run_lama_large(
    image: Image.Image,
    mask: np.ndarray,
    *,
    device: Any,
    inpainting_size: int,
    precision: str,
    verbose: bool = False,
) -> Image.Image:
    """Run hgmzhn LAMA Large with a CUDA attempt and a CPU ONNX fallback."""
    _ensure_hgmzhn_runtime()
    from manga_translator.config import Inpainter, InpainterConfig, InpaintPrecision
    from manga_translator.inpainting import dispatch, prepare, unload

    image_rgb = image.convert("RGB")
    mask_u8 = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[..., 0]
    if not np.any(mask_u8):
        return image.copy()

    size = max(256, min(4096, int(inpainting_size)))
    precision_name = str(precision or "fp32").lower()
    if precision_name not in {"bf16", "fp16", "fp32"}:
        precision_name = "fp32"
    precision_value = InpaintPrecision(precision_name)

    requested_device = str(device or "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"

    async def infer(run_device: str) -> np.ndarray:
        force_torch = run_device.startswith("cuda") or run_device == "mps"
        config = InpainterConfig(
            inpainter=Inpainter.lama_large,
            inpainting_size=size,
            inpainting_precision=precision_value,
            force_use_torch_inpainting=force_torch,
        )
        try:
            await prepare(
                Inpainter.lama_large,
                device=run_device,
                force_torch=force_torch,
            )
            return await dispatch(
                inpainter_key=Inpainter.lama_large,
                image=np.asarray(image_rgb),
                mask=mask_u8,
                config=config,
                inpainting_size=size,
                device=run_device,
                verbose=verbose,
            )
        finally:
            await unload(Inpainter.lama_large)

    with _LAMA_RUNTIME_LOCK:
        try:
            result = _run_async(infer(requested_device))
        except Exception as first_error:
            if requested_device == "cpu":
                raise RuntimeError(f"LAMA Large failed: {first_error}") from first_error
            log_message(
                "LAMA Large GPU processing failed; retrying with CPU ONNX: "
                f"{first_error}",
                always_print=True,
            )
            result = _run_async(infer("cpu"))

    return Image.fromarray(np.asarray(result, dtype=np.uint8), mode="RGB")
