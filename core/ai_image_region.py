"""Immutable editor image patches shared by preview, export and portable sessions."""

from functools import lru_cache
from pathlib import Path

from PIL import Image

AI_FILE_KEYS = ("ai_original_path", "ai_processed_path", "ai_mask_path")


@lru_cache(maxsize=32)
def _read_asset(path, stamp, size, mode):
    with Image.open(path) as image:
        result = image.convert(mode)
        if result.size != size:
            raise ValueError("AI 식자 파일 크기가 저장된 영역과 다릅니다.")
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
                return _read_asset(str(path), path.stat().st_mtime_ns, size, mode)

            patch = asset(key, "RGBA")
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
