"""Immutable editor image patches shared by preview, export and portable sessions."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

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


def repair_padding_fringe(before, after):
    """Restore a coherent one-pixel bright fringe beside the square canvas padding."""
    width, height = after.size
    if width == height or min(width, height) < 3 or max(width, height) < 11:
        return after

    original_patch = after
    vertical = width < height
    if not vertical:
        before = before.transpose(Image.Transpose.TRANSPOSE)
        after = after.transpose(Image.Transpose.TRANSPOSE)
        width, height = after.size

    repaired = after
    for x, direction in ((0, 1), (width - 1, -1)):
        left = min(x, x + 2 * direction)
        strip = (left, 0, left + 3, height)
        old = before.crop(strip).convert("L")
        new = after.crop(strip).convert("L")
        delta = np.asarray(new, dtype=np.int16) - np.asarray(old, dtype=np.int16)
        edge, inner, deeper = delta[:, ::direction].T
        edge_luma = np.asarray(new)[:, ::direction][:, 0]
        bright = (edge >= 16) & (edge >= np.maximum(inner, deeper) + 16)
        bright &= edge_luma < 240
        coherent = np.convolve(bright, np.ones(11, dtype=np.uint8), mode="same") >= 6
        repair = bright & coherent
        if repair.any():
            if repaired is after:
                repaired = after.copy()
            edge_box = (x, 0, x + 1, height)
            mask = Image.fromarray(repair[:, None].astype(np.uint8) * 255)
            repaired.paste(before.crop(edge_box), edge_box, mask)
    if repaired is after:
        return original_patch
    return repaired if vertical else repaired.transpose(Image.Transpose.TRANSPOSE)


@lru_cache(maxsize=32)
def _read_asset(path, stamp, size, mode, reference_path=None, reference_stamp=None):
    with Image.open(path) as image:
        result = image.convert(mode)
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


def apply_ai_image_regions(image, session, errors=None):
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

            if region.get("enabled", True) and metadata.get("pipeline") == "separated_v1":
                patch = Image.alpha_composite(asset("ai_background_path", "RGBA"),
                                              asset("ai_lettering_path", "RGBA"))
            else:
                patch = asset(key, "RGBA")
                if key == "ai_processed_path" and not metadata.get("padding_fringe_fixed"):
                    patch = repair_padding_fringe(result.crop(box), patch)
            mask = asset("ai_mask_path", "L").point(lambda value: 255 if value else 0)
            if result is image:
                result = image.convert("RGBA").copy()
            result.paste(patch, box, mask)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            if errors is not None:
                errors.append(
                    f"영역 {index}: AI 식자 파일을 불러오지 못했습니다. {exc}"
                )
    return result
