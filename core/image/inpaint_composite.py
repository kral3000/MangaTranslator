"""Conservative, local finishing for outside-text inpainting; no model calls."""

import cv2
import numpy as np
from PIL import Image


def finish_outside_inpaint(original, cleaned, region_masks, protected=None):
    """Match only supported smooth background rows without spreading boundary ink."""
    if not region_masks:
        return cleaned
    source = np.array(original.convert("RGB"))
    result = np.array(cleaned.convert("RGB"))
    height, width = source.shape[:2]
    protected = (
        np.zeros((height, width), bool)
        if protected is None
        else np.asarray(protected, bool)
    )
    for region_mask in region_masks:
        mask = np.asarray(region_mask, bool) & ~protected
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        area = (x2 - x1) * (y2 - y1)
        if min(x2 - x1, y2 - y1) < 8 or area > 262144 or mask.sum() < area * 0.95:
            continue
        left, top = max(0, x1 - 8), max(0, y1 - 8)
        right, bottom = min(width, x2 + 8), min(height, y2 + 8)
        before = source[top:bottom, left:right]
        after = result[top:bottom, left:right].copy()
        if np.array_equal(before, after):
            continue
        local_mask = mask[top:bottom, left:right]
        blocked = protected[top:bottom, left:right]
        lx, rx = x1 - left, x2 - left
        # Reference strips sit beyond the possible old glyph outline. Never sample old ink inside the box.
        strips = [
            (0, max(0, lx - 5), lx + 2, lx + 5),
            (min(before.shape[1], rx + 5), before.shape[1], rx - 5, rx - 2),
        ]
        references, samples = [], []
        valid = local_mask.any(axis=1)
        for a, b, c, d in strips:
            if b - a < 2:
                continue
            reference = before[:, a:b].astype(np.float32)
            generated = after[:, c:d].astype(np.float32)
            valid &= ~blocked[:, a:b].any(axis=1)
            valid &= np.max(np.ptp(reference, axis=1), axis=1) <= 10
            valid &= np.max(np.ptp(generated, axis=1), axis=1) <= 16
            references.append(np.median(reference, axis=1))
            samples.append(np.median(generated, axis=1))
        if not references:
            continue
        ref_l, ref_r = references[0], references[-1]
        gen_l, gen_r = samples[0], samples[-1]
        valid &= np.max(np.abs(ref_l - ref_r), axis=1) <= 24
        valid &= np.max(np.abs(gen_l - gen_r), axis=1) <= 32
        delta_l, delta_r = ref_l - gen_l, ref_r - gen_r
        valid &= (
            np.maximum(np.max(np.abs(delta_l), axis=1), np.max(np.abs(delta_r), axis=1))
            <= 32
        )
        # Isolated supported rows in a textured area are not enough evidence to recolor it.
        valid = (
            cv2.morphologyEx(
                valid.astype(np.uint8)[:, None],
                cv2.MORPH_OPEN,
                np.ones((3, 1), np.uint8),
            )[:, 0]
            > 0
        )
        if not np.any(valid):
            continue
        weight = np.clip(
            (np.arange(after.shape[1], dtype=np.float32) - lx - 2)
            / max(1, rx - lx - 5),
            0,
            1,
        )[None, :, None]
        delta = delta_l[:, None, :] * (1 - weight) + delta_r[:, None, :] * weight
        corrected = np.clip(np.rint(after.astype(np.float32) + delta), 0, 255).astype(
            np.uint8
        )
        # Only a bounded offset is allowed: generated lines and contrast are retained.
        generated_background = (
            gen_l[:, None, :] * (1 - weight) + gen_r[:, None, :] * weight
        )
        background_like = (
            np.max(np.abs(after.astype(np.float32) - generated_background), axis=2)
            <= 24
        )
        # Never lift near-black ink, even when it resembles a very dark background.
        background_like &= ~(
            (np.max(after, axis=2) <= 16) & (np.max(delta, axis=2) > 0)
        )
        apply = local_mask & valid[:, None] & background_like
        output = result[top:bottom, left:right]
        output[apply] = corrected[apply]
        gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        ink = ((gray < 160) & local_mask).astype(np.uint8)
        nearby_ink = cv2.dilate(ink, np.ones((19, 19), np.uint8)) > 0
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
        ring = (
            cv2.dilate(local_mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        ) & ~local_mask
        fringe = (
            (gray > 220)
            & (cv2.subtract(gray, opening) > 5)
            & nearby_ink
            & ring
            & ~blocked
            & valid[:, None]
        )
        if np.any(fringe):
            fringe = (
                (
                    cv2.dilate(local_mask.astype(np.uint8), np.ones((15, 15), np.uint8))
                    > 0
                )
                & ~local_mask
                & (gray > 160)
                & nearby_ink
                & ~blocked
                & valid[:, None]
            )
            background = np.clip(
                np.rint(ref_l[:, None, :] * (1 - weight) + ref_r[:, None, :] * weight),
                0,
                255,
            ).astype(np.uint8)
            output[fringe] = background[fringe]
    image = Image.fromarray(result)
    if cleaned.mode == "RGBA":
        image.putalpha(cleaned.getchannel("A"))
    return image if cleaned.mode == "RGBA" else image.convert(cleaned.mode)
