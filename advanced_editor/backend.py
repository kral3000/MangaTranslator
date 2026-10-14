from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import fields
from io import BytesIO
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
HGMZHN_RUNTIME = Path(__file__).resolve().parent / "hgmzhn_runtime"


def _device_name(requested: str = "auto", *, prefer_cpu: bool = False) -> str:
    requested = str(requested or "auto").lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없습니다.")
        return "cuda"
    if prefer_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_hgmzhn_runtime() -> None:
    runtime = str(HGMZHN_RUNTIME)
    if runtime not in sys.path:
        sys.path.insert(0, runtime)


def _normalize_mask_for_image(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    mask_u8 = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if mask_u8.ndim > 2:
        mask_u8 = mask_u8[..., 0]
    width, height = image.size
    if mask_u8.shape[:2] != (height, width):
        mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(np.where(mask_u8 > 0, 255, 0).astype(np.uint8))


def _int_setting(value: Any, default: int) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def run_inpainting(
    image: Image.Image,
    mask: np.ndarray,
    settings: dict[str, Any],
) -> Image.Image:
    mask_u8 = _normalize_mask_for_image(mask, image)
    if not np.any(mask_u8):
        raise ValueError("인페인팅 마스크가 비어 있습니다.")

    method = str(settings.get("method") or "opencv")
    image_rgb = image.convert("RGB")
    if method == "opencv":
        radius = max(1.0, float(settings.get("opencv_radius") or 3.0))
        algorithm = (
            cv2.INPAINT_NS
            if str(settings.get("opencv_algorithm") or "telea") == "ns"
            else cv2.INPAINT_TELEA
        )
        result = cv2.inpaint(np.asarray(image_rgb), mask_u8, radius, algorithm)
        return Image.fromarray(result, mode="RGB").convert("RGBA")

    if method == "lama_large":
        _ensure_hgmzhn_runtime()
        from manga_translator.config import InpaintPrecision, Inpainter, InpainterConfig
        from manga_translator.inpainting import dispatch, prepare

        requested_device = str(settings.get("device") or "auto")
        device = _device_name(requested_device, prefer_cpu=requested_device == "auto")
        force_torch = bool(settings.get("force_torch", False))
        inpainting_size = max(256, int(settings.get("inpainting_size") or 1024))
        precision_name = str(settings.get("precision") or "bf16")
        precision = InpaintPrecision(precision_name)
        config = InpainterConfig(
            inpainter=Inpainter.lama_large,
            inpainting_size=inpainting_size,
            inpainting_precision=precision,
            force_use_torch_inpainting=force_torch,
        )

        async def _run(run_device: str, run_force_torch: bool) -> np.ndarray:
            await prepare(
                Inpainter.lama_large,
                device=run_device,
                force_torch=run_force_torch,
            )
            return await dispatch(
                inpainter_key=Inpainter.lama_large,
                image=np.asarray(image_rgb),
                mask=mask_u8,
                config=config,
                inpainting_size=inpainting_size,
                device=run_device,
                verbose=True,
            )

        try:
            result = asyncio.run(_run(device, force_torch))
        except Exception as first_exc:
            if device == "cpu" and not force_torch:
                raise
            try:
                config.force_use_torch_inpainting = False
                result = asyncio.run(_run("cpu", False))
            except Exception as second_exc:
                raise RuntimeError(
                    "LAMA Large 실행에 실패했습니다. "
                    f"1차 오류: {first_exc}\nCPU 재시도 오류: {second_exc}"
                ) from second_exc
        return Image.fromarray(np.asarray(result, dtype=np.uint8)).convert("RGBA")

    from core.image.inpainting import FluxKleinInpainter, FluxKontextInpainter

    device = torch.device(_device_name(str(settings.get("device") or "auto")))
    common = {
        "device": device,
        "huggingface_token": str(settings.get("huggingface_token") or ""),
        "num_inference_steps": max(1, int(settings.get("steps") or 4)),
        "low_vram": bool(settings.get("low_vram", True)),
        "prompt": str(settings.get("prompt") or "").strip(),
    }
    if method == "flux_kontext":
        inpainter = FluxKontextInpainter(
            **common,
            residual_diff_threshold=float(
                settings.get("residual_diff_threshold") or 0.15
            ),
            backend=str(settings.get("kontext_backend") or "sdnq"),
        )
    elif method in {"flux_klein_4b", "flux_klein_9b"}:
        inpainter = FluxKleinInpainter(
            **common,
            variant="9b" if method.endswith("9b") else "4b",
            luminance_correction=bool(settings.get("luminance_correction", True)),
            upscale_small_crops=bool(settings.get("upscale_small_crops", True)),
            verbose=True,
        )
    else:
        raise ValueError(f"지원하지 않는 인페인팅 방식입니다: {method}")

    try:
        return inpainter.inpaint_mask(
            image_rgb,
            mask_u8 > 0,
            seed=_int_setting(settings.get("seed"), 1),
            verbose=True,
            strict_mask_clipping=bool(settings.get("strict_mask_clipping", True)),
        ).convert("RGBA")
    finally:
        inpainter.unload_models()


def _intersects_roi(box: list[float], roi: list[float]) -> bool:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    return min(x2, rx2) > max(x1, rx1) and min(y2, ry2) > max(y1, ry1)


def _merge_boxes_in_roi(
    boxes: list[list[float]],
    roi: list[float],
    image_size: tuple[int, int],
) -> list[list[float]]:
    if not boxes:
        return []

    width, height = image_size
    rx1, ry1, rx2, ry2 = [float(value) for value in roi[:4]]
    rx1, rx2 = sorted((max(0.0, min(float(width), rx1)), max(0.0, min(float(width), rx2))))
    ry1, ry2 = sorted((max(0.0, min(float(height), ry1)), max(0.0, min(float(height), ry2))))
    if rx2 <= rx1 or ry2 <= ry1:
        return []

    clipped_boxes: list[list[float]] = []
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        cx1 = max(rx1, min(float(width), x1))
        cy1 = max(ry1, min(float(height), y1))
        cx2 = min(rx2, max(0.0, x2))
        cy2 = min(ry2, max(0.0, y2))
        if cx2 > cx1 and cy2 > cy1:
            clipped_boxes.append([cx1, cy1, cx2, cy2])

    if not clipped_boxes:
        return []

    return [[
        min(box[0] for box in clipped_boxes),
        min(box[1] for box in clipped_boxes),
        max(box[2] for box in clipped_boxes),
        max(box[3] for box in clipped_boxes),
    ]]


def _merge_recognized_results(
    results: list[dict[str, Any]],
    merged_boxes: list[list[float]],
) -> list[dict[str, Any]]:
    if not results or not merged_boxes:
        return []
    texts = [
        str(result.get("text") or "").strip()
        for result in results
        if str(result.get("text") or "").strip()
    ]
    confidences = [
        float(result.get("confidence") or 0.0)
        for result in results
        if result.get("confidence") is not None
    ]
    return [
        {
            "bbox": merged_boxes[0],
            "text": "\n".join(texts),
            "confidence": (
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
        }
    ]


def detect_text_boxes(
    image: Image.Image,
    roi: list[float],
    detector: str,
    confidence: float,
    *,
    hf_token: str = "",
) -> list[list[float]]:
    detector = str(detector or "default")
    if detector == "default":
        _ensure_hgmzhn_runtime()
        from manga_translator.config import Detector
        from manga_translator.detection import dispatch, get_detector, prepare

        width, height = image.size
        x1 = max(0, min(width - 1, int(round(roi[0]))))
        y1 = max(0, min(height - 1, int(round(roi[1]))))
        x2 = max(x1 + 1, min(width, int(round(roi[2]))))
        y2 = max(y1 + 1, min(height, int(round(roi[3]))))
        crop = image.convert("RGB").crop((x1, y1, x2, y2))
        longest_side = max(crop.size)
        detect_size = min(
            2048,
            max(1024, ((longest_side + 31) // 32) * 32),
        )
        threshold = max(0.01, min(0.99, float(confidence)))

        async def _run(run_device: str):
            await prepare(Detector.default)
            return await dispatch(
                Detector.default,
                np.asarray(crop),
                detect_size,
                threshold,
                threshold,
                1.5,
                device=run_device,
                verbose=False,
            )

        device = _device_name("auto")
        try:
            textlines, _mask, _raw = asyncio.run(_run(device))
        except Exception:
            if device != "cuda":
                raise

            async def _retry_cpu():
                detector_instance = get_detector(Detector.default)
                await detector_instance.unload()
                return await _run("cpu")

            textlines, _mask, _raw = asyncio.run(_retry_cpu())

        boxes: list[list[float]] = []
        for textline in textlines:
            points = np.asarray(getattr(textline, "pts", []), dtype=np.float32)
            if points.size < 8:
                continue
            local_x1 = float(np.min(points[:, 0]))
            local_y1 = float(np.min(points[:, 1]))
            local_x2 = float(np.max(points[:, 0]))
            local_y2 = float(np.max(points[:, 1]))
            if local_x2 <= local_x1 or local_y2 <= local_y1:
                continue
            boxes.append(
                [
                    max(0.0, min(float(width), local_x1 + x1)),
                    max(0.0, min(float(height), local_y1 + y1)),
                    max(0.0, min(float(width), local_x2 + x1)),
                    max(0.0, min(float(height), local_y2 + y1)),
                ]
            )
        return boxes

    from core.ml.model_manager import ModelType, get_model_manager

    manager = get_model_manager()
    device = _device_name("auto")
    if detector in {"yolo_1", "yolo_2"}:
        model_type = (
            ModelType.YOLO_SPEECH_BUBBLE_2
            if detector == "yolo_2"
            else ModelType.YOLO_SPEECH_BUBBLE
        )
        model_path = manager.model_paths[model_type]
        model = manager.load_yolo_speech_bubble(str(model_path), verbose=True)
        result = model(
            image.convert("RGB"),
            conf=float(confidence),
            device=device,
            verbose=False,
            imgsz=1600 if detector == "yolo_2" else 640,
            retina_masks=True,
        )[0]
    elif detector == "osb":
        model = manager.load_yolo_osbtext(token=hf_token, verbose=True)
        result = model(
            image.convert("RGB"),
            conf=float(confidence),
            device=device,
            verbose=False,
            imgsz=640,
        )[0]
    else:
        raise ValueError(f"지원하지 않는 검출 방식입니다: {detector}")

    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
    return [[float(value) for value in box[:4]] for box in boxes if _intersects_roi(box, roi)]


def _recognize_hgmzhn(
    image: Image.Image,
    boxes: list[list[float]],
    model_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    _ensure_hgmzhn_runtime()
    from manga_translator.config import Ocr, OcrConfig
    from manga_translator.ocr import dispatch, prepare, unload
    from manga_translator.utils import Quadrilateral

    model_map = {
        "32px": Ocr.ocr32px,
        "48px": Ocr.ocr48px,
        "48px_ctc": Ocr.ocr48px_ctc,
    }
    ocr_key = model_map[model_name]
    rgb = np.asarray(image.convert("RGB"))
    quads = []
    for x1, y1, x2, y2 in boxes:
        points = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        quads.append(Quadrilateral(pts=points, text="", prob=1.0))
    config = OcrConfig(ocr=ocr_key, prob=float(threshold), ignore_bubble=0.0)
    device = _device_name("auto")

    async def _run():
        await prepare(ocr_key, device=device)
        try:
            return await dispatch(
                ocr_key,
                rgb,
                quads,
                config,
                device=device,
                verbose=True,
            )
        finally:
            await unload(ocr_key)

    recognized = asyncio.run(_run())
    output = []
    for box, item in zip(boxes, recognized):
        output.append(
            {
                "bbox": box,
                "text": str(getattr(item, "text", "") or ""),
                "confidence": float(getattr(item, "prob", 0.0) or 0.0),
            }
        )
    return output


def recognize_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    model_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    if not boxes:
        return []
    if model_name in {"32px", "48px", "48px_ctc"}:
        return _recognize_hgmzhn(image, boxes, model_name, threshold)

    if model_name == "llm_ocr":
        return _recognize_llm_ocr(image, boxes)

    if model_name not in {"first_stage", "manga_ocr"}:
        raise ValueError(f"지원하지 않는 OCR 방식입니다: {model_name}")
    from core.image.ocr_detection import extract_text_with_manga_ocr

    width, height = image.size
    crops = []
    normalized_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        normalized = [
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(1, min(width, int(round(x2)))),
            max(1, min(height, int(round(y2)))),
        ]
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            continue
        normalized_boxes.append([float(value) for value in normalized])
        crops.append(image.convert("RGB").crop(tuple(normalized)))
    texts = extract_text_with_manga_ocr(crops, verbose=True)
    return [
        {"bbox": box, "text": text, "confidence": 1.0 if text.strip() else 0.0}
        for box, text in zip(normalized_boxes, texts)
    ]


def _recognize_llm_ocr(
    image: Image.Image,
    boxes: list[list[float]],
) -> list[dict[str, Any]]:
    if not boxes:
        return []

    from core.services.translation import (
        _build_system_prompt_ocr,
        _call_llm_endpoint,
        _parse_llm_response_unified,
    )

    config = _build_translation_config({})
    config.send_full_page_context = False
    width, height = image.size
    parts: list[dict[str, Any]] = []
    normalized_boxes: list[list[float]] = []
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        normalized = [
            max(0, min(width - 1, int(round(x1)))),
            max(0, min(height - 1, int(round(y1)))),
            max(1, min(width, int(round(x2)))),
            max(1, min(height, int(round(y2)))),
        ]
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            continue
        crop = image.convert("RGB").crop(tuple(normalized))
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        parts.append({"text": f"INPUT OCR CROP {index}. Return this crop as JSON key \"{index}\"."})
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                }
            }
        )
        normalized_boxes.append([float(value) for value in normalized])

    if not normalized_boxes:
        return []

    prompt = (
        f"There are exactly {len(normalized_boxes)} manga text crop images above. "
        "Perform OCR transcription only. Do not translate. Return a single JSON object "
        "with numeric string keys \"1\", \"2\", ... matching the crop order. "
        "If a crop is unreadable, return \"[OCR FAILED]\" for that key."
    )
    response = _call_llm_endpoint(
        config,
        parts,
        prompt,
        debug=False,
        system_prompt=_build_system_prompt_ocr(
            config.input_language,
            config.reading_direction,
        ),
    )
    texts = _parse_llm_response_unified(
        response,
        len(normalized_boxes),
        f"{config.provider}-EditorOCR",
        debug=False,
    )
    return [
        {
            "bbox": box,
            "text": text,
            "confidence": 1.0 if str(text).strip() and str(text).strip() != "[OCR FAILED]" else 0.0,
        }
        for box, text in zip(normalized_boxes, texts)
    ]


def detect_and_recognize(
    image: Image.Image,
    roi: list[float],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    detector_confidence = settings.get("detector_confidence")
    ocr_threshold = settings.get("ocr_threshold")
    boxes = detect_text_boxes(
        image,
        roi,
        str(settings.get("detector") or "default"),
        float(0.3 if detector_confidence is None else detector_confidence),
        hf_token=str(settings.get("huggingface_token") or ""),
    )
    ocr_model = str(settings.get("ocr_model") or "manga_ocr")
    merge_detected = bool(settings.get("merge_detected_boxes", True))
    merged_boxes = _merge_boxes_in_roi(boxes, roi, image.size) if merge_detected else []
    if merge_detected and ocr_model in {"32px", "48px", "48px_ctc"}:
        recognized = recognize_boxes(
            image,
            boxes,
            ocr_model,
            float(0.3 if ocr_threshold is None else ocr_threshold),
        )
        return _merge_recognized_results(recognized, merged_boxes)
    if merged_boxes:
        boxes = merged_boxes
    return recognize_boxes(
        image,
        boxes,
        ocr_model,
        float(0.3 if ocr_threshold is None else ocr_threshold),
    )


_TRANSLATION_PROVIDERS = [
    "Google",
    "OpenAI",
    "Anthropic",
    "xAI",
    "DeepSeek",
    "Z.ai",
    "Moonshot AI",
    "OpenRouter",
    "OpenAI-Compatible",
]

_PROVIDER_API_KEY_FIELDS = {
    "Google": "google_api_key",
    "OpenAI": "openai_api_key",
    "Anthropic": "anthropic_api_key",
    "xAI": "xai_api_key",
    "DeepSeek": "deepseek_api_key",
    "Z.ai": "zai_api_key",
    "Moonshot AI": "moonshot_api_key",
    "OpenRouter": "openrouter_api_key",
    "OpenAI-Compatible": "openai_compatible_api_key",
}

_ENV_KEY_NAMES = {
    "google_api_key": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openai_api_key": ("OPENAI_API_KEY",),
    "anthropic_api_key": ("ANTHROPIC_API_KEY",),
    "xai_api_key": ("XAI_API_KEY",),
    "deepseek_api_key": ("DEEPSEEK_API_KEY",),
    "zai_api_key": ("ZAI_API_KEY",),
    "moonshot_api_key": ("MOONSHOT_API_KEY",),
    "openrouter_api_key": ("OPENROUTER_API_KEY",),
    "openai_compatible_api_key": ("OPENAI_COMPATIBLE_API_KEY",),
}


def get_translation_defaults() -> dict[str, Any]:
    try:
        from ui import settings_manager

        saved = settings_manager.get_saved_settings()
        provider_models = saved.get("provider_models") or {}
    except Exception:
        saved = {}
        provider_models = {}

    provider = str(saved.get("provider") or "Google")
    model_name = str(
        saved.get("model_name")
        or provider_models.get(provider)
        or "gemini-3.1-flash-lite"
    )
    return {
        "providers": list(_TRANSLATION_PROVIDERS),
        "provider": provider,
        "model_name": model_name,
        "input_language": str(saved.get("input_language") or "Japanese"),
        "output_language": str(saved.get("output_language") or "Korean"),
        "reading_direction": str(saved.get("reading_direction") or "rtl"),
        "openai_compatible_url": str(
            saved.get("openai_compatible_url") or "http://localhost:8080/v1"
        ),
    }


def _load_saved_translation_settings() -> dict[str, Any]:
    try:
        from ui import settings_manager

        return dict(settings_manager.get_saved_settings())
    except Exception:
        return {}


def _build_translation_config(settings: dict[str, Any]):
    from core.config import TranslationConfig

    saved = _load_saved_translation_settings()
    merged = dict(saved)
    for key, value in (settings or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        merged[key] = value

    config = TranslationConfig()
    valid_fields = {field.name for field in fields(TranslationConfig)}
    for key in valid_fields:
        if key in merged:
            setattr(config, key, merged[key])

    provider = str(merged.get("provider") or config.provider or "Google")
    config.provider = provider
    provider_models = merged.get("provider_models") or {}
    model_name = merged.get("model_name") or provider_models.get(provider)
    if model_name:
        config.model_name = str(model_name)

    api_key_override = str(merged.get("api_key_override") or "").strip()
    api_key_field = _PROVIDER_API_KEY_FIELDS.get(provider)
    if api_key_override and api_key_field:
        setattr(config, api_key_field, api_key_override)

    for key_field, env_names in _ENV_KEY_NAMES.items():
        if getattr(config, key_field, ""):
            continue
        for env_name in env_names:
            value = os.environ.get(env_name, "")
            if value:
                setattr(config, key_field, value)
                break

    config.translation_mode = "two-step"
    config.send_full_page_context = False
    return config


def translate_texts(texts: list[str], settings: dict[str, Any]) -> list[str]:
    cleaned_texts = [str(text or "").strip() for text in texts]
    if not cleaned_texts or not any(cleaned_texts):
        raise ValueError("번역할 텍스트가 비어 있습니다.")

    from core.services.translation import (
        _build_system_prompt_translation,
        _call_llm_endpoint,
        _format_indexed_json,
        _format_special_instructions,
        _parse_llm_response_unified,
    )

    config = _build_translation_config(settings)
    indexed_input = _format_indexed_json(cleaned_texts)
    special_instructions = _format_special_instructions(config)
    prompt = f"""
## CONTEXT
You are translating already-OCRed manga text segments inside an editor.
The source text is provided as numbered JSON values, in reading order.
Do not perform OCR. Do not merge, skip, renumber, or add extra items.

## INPUT DATA
{indexed_input}

## TASK
Translate each value into {config.output_language}.
Return a single JSON object with exactly the same numeric string keys.
Each value must contain only the translated text for that same key.
Preserve intentional line breaks when they are present in the source text.
Use natural manga localization wording and keep sound effects concise.{special_instructions}
"""
    system_prompt = _build_system_prompt_translation(
        config.output_language,
        mode="two-step",
        reading_direction=config.reading_direction,
        full_page_context=False,
    )
    response = _call_llm_endpoint(
        config,
        [],
        prompt,
        debug=bool(settings.get("debug", False)),
        system_prompt=system_prompt,
    )
    from core.text.text_processing import strip_translation_terminal_periods

    return [
        strip_translation_terminal_periods(text)
        for text in _parse_llm_response_unified(
            response,
            len(cleaned_texts),
            f"{config.provider}-EditorTranslate",
            debug=bool(settings.get("debug", False)),
        )
    ]
