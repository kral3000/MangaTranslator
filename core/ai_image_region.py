"""Immutable editor image patches shared by preview, export and portable sessions."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Reference weight from the outermost pixel inward. The first two pixels can
# contain both the bright and dark lobes of the padded-canvas resize.
_SEAM_WEIGHTS = (255, 255, 191, 64)

AI_FILE_KEYS = ("ai_original_path", "ai_processed_path", "ai_mask_path",
                "ai_background_path", "ai_lettering_path", "ai_erase_mask_path")


def preserve_boundary_artwork(before, after, allowed):
    """Keep matching edge-connected artwork instead of a resampled copy of it."""
    old_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    new_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    old_ink = old_gray < 224
    new_ink = new_gray < 224
    union = ((old_ink | new_ink) & (allowed > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(union)
    if count <= 1:
        return after
    # Only objects crossing the selection edge can create a visible join.
    inner = cv2.erode(
        allowed, np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    edge_labels = np.unique(labels[(allowed > 0) & (inner == 0)])
    overlap = np.bincount(labels[(old_ink & new_ink) & (allowed > 0)], minlength=count)
    matches = np.zeros(count, dtype=bool)
    matches[edge_labels] = True
    matches[0] = False
    matches &= stats[:, cv2.CC_STAT_AREA] >= 8
    matches &= overlap >= stats[:, cv2.CC_STAT_AREA] * 0.90
    # Edge protection must not resurrect erased glyphs joined to otherwise matching art.
    inset = cv2.erode(allowed, np.ones((7, 7), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0)
    erased = (old_gray < 64) & (new_gray > 240) & (inset > 0)
    erased_counts = np.bincount(labels[erased], minlength=count)
    matches &= erased_counts < 4
    # A changed glyph inside a dark object must never be mistaken for unchanged art.
    interior = cv2.erode(
        union, np.ones((7, 7), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    difference = np.max(np.abs(after.astype(np.int16) - before.astype(np.int16)), axis=2)
    matches[np.unique(labels[(interior > 0) & (difference > 64)])] = False
    protected = cv2.dilate(matches[labels].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    protected &= allowed > 0
    result = after.copy()
    result[protected] = before[protected]
    return result


def _blend_patch_edges(page, box, patch, saved_original=None):
    """Join the patch to the page without resampling its interior."""
    left, top, right, bottom = box
    width, height = patch.size
    result = patch
    for axis, length, margins in (
        (1, width, (left, page.width - right)),
        (0, height, (top, page.height - bottom)),
    ):
        depth = min(len(_SEAM_WEIGHTS), (length - 1) // 2)
        if depth < 1:
            continue
        for reverse, margin in enumerate(margins):
            if not margin:
                continue  # There is no join at the physical page edge.
            start, stop = (length - depth - 1, length) if reverse else (0, depth + 1)
            sample_box = (start, 0, stop, height) if axis == 1 else (0, start, width, stop)
            page_box = tuple(v + offset for v, offset in zip(sample_box, (left, top, left, top)))
            current = page.crop(page_box)
            if current.mode != "RGBA":
                current = current.convert("RGBA")
            sample = patch.crop(sample_box)
            reference_bytes, sample_bytes = current.tobytes(), sample.tobytes()
            if reference_bytes == sample_bytes:
                if saved_original is None:
                    continue
                current = saved_original.crop(sample_box)
                reference_bytes = current.tobytes()
                if reference_bytes == sample_bytes:
                    continue

            # Compare each pixel with its inward neighbor. Keep strong ink or
            # erasure that extends inward, including strokes along a box edge.
            shape = (sample.height, sample.width, 4)
            edited = np.frombuffer(sample_bytes, dtype=np.uint8).reshape(shape)[..., :3]
            reference = np.frombuffer(reference_bytes, dtype=np.uint8).reshape(shape)[..., :3]
            delta = np.subtract(edited, reference, dtype=np.int16)
            delta = np.moveaxis(delta, axis, 1)
            if reverse:
                delta = delta[:, ::-1]
            lighter, darker = delta >= 48, delta <= -48
            connected = (
                (lighter[:, :-1] & lighter[:, 1:]) | (darker[:, :-1] & darker[:, 1:])
            ).any(axis=-1)
            protected = connected.copy()
            protected[:, 1:] |= connected[:, :-1]
            # The last sample only establishes continuity. It is never blended.
            alpha = np.zeros(delta.shape[:2], dtype=np.uint8)
            np.copyto(alpha[:, :depth], np.array(_SEAM_WEIGHTS[:depth], dtype=np.uint8), where=~protected)
            if reverse:
                alpha = alpha[:, ::-1]
            if axis == 0:
                alpha = alpha.T
            if alpha.any():
                if result is patch:
                    result = patch.copy()
                result.paste(current, sample_box, Image.fromarray(alpha))
    return result


@lru_cache(maxsize=32)
def _read_asset(path, stamp, size, mode, reference_path=None, reference_stamp=None):
    with Image.open(path) as image:
        result = image.convert(mode)
        if mode == "L":
            result = result.point(lambda value: 255 if value else 0)
        if result.size != size:
            raise ValueError("AI 식자 파일 크기가 저장된 영역과 다릅니다.")
        if reference_path is not None:
            # Older active patches did not require a readable original to render.
            try:
                with Image.open(reference_path) as reference:
                    if reference.size != size:
                        return result
                    before = np.array(reference.convert("RGB"))
            except (OSError, ValueError):
                return result
            repaired = preserve_boundary_artwork(
                before, np.array(result.convert("RGB")),
                np.full((size[1], size[0]), 255, np.uint8),
            )
            corrected = Image.fromarray(repaired).convert(mode)
            if "A" in result.getbands():
                corrected.putalpha(result.getchannel("A"))
            result = corrected
        return result


def apply_ai_image_regions(image, session, errors=None, *, has_baked_ai=False):
    result = image
    for index, region in enumerate(session.get("regions", []), 1):
        metadata = region.get("ai_image")
        if not metadata:
            continue
        try:
            box = tuple(int(v) for v in metadata["box"])
            size = (box[2] - box[0], box[3] - box[1])
            if (
                box[0] < 0
                or box[1] < 0
                or box[2] > image.width
                or box[3] > image.height
            ):
                raise ValueError("AI 식자 영역이 페이지 바깥에 있습니다.")
            key = (
                "ai_processed_path"
                if region.get("enabled", True)
                else "ai_original_path"
            )

            def asset(field, mode):
                path = Path(region[field])
                reference = Path(region.get("ai_original_path") or "")
                if (
                    field == "ai_processed_path"
                    and not metadata.get("boundary_preserved")
                    and reference.is_file()
                ):
                    return _read_asset(
                        str(path), path.stat().st_mtime_ns, size, mode,
                        str(reference), reference.stat().st_mtime_ns,
                    )
                return _read_asset(str(path), path.stat().st_mtime_ns, size, mode)

            separated = key == "ai_processed_path" and metadata.get("pipeline") == "separated_v1"
            patch = asset("ai_background_path" if separated else key, "RGBA")
            if key == "ai_processed_path":
                saved_original = None
                if has_baked_ai:
                    try:
                        saved_original = asset("ai_original_path", "RGBA")
                    except (OSError, KeyError, ValueError, TypeError):
                        pass
                patch = _blend_patch_edges(result, box, patch, saved_original)
                if separated:
                    patch = Image.alpha_composite(patch, asset("ai_lettering_path", "RGBA"))
            mask = asset("ai_mask_path", "L")
            if result is image:
                result = image.convert("RGBA")
            result.paste(patch, box, mask)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            if errors is not None:
                errors.append(
                    f"영역 {index}: AI 식자 파일을 불러오지 못했습니다. {exc}"
                )
    return result
