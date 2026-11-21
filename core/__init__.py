"""Public translation APIs, imported when their subsystem is used."""

from importlib import import_module

from ._version import __version__, __version_info__

_MODULES = {
    "UnifiedCache": ".caching",
    "get_cache": ".caching",
    "clean_speech_bubbles": ".image.cleaning",
    "detect_speech_bubbles": ".image.detection",
    "cv2_to_pil": ".image.image_utils",
    "pil_to_cv2": ".image.image_utils",
    "save_image_with_compression": ".image.image_utils",
    "FluxKleinInpainter": ".image.inpainting",
    "FluxKontextInpainter": ".image.inpainting",
    "OutsideTextDetector": ".image.ocr_detection",
    "sort_bubbles_by_reading_order": ".image.sorting",
    "ModelManager": ".ml.model_manager",
    "get_model_manager": ".ml.model_manager",
    "batch_translate_images": ".pipeline",
    "translate_and_render": ".pipeline",
    "call_translation_api_batch": ".services.translation",
    "render_text_skia": ".text.text_renderer",
}
__all__ = ["__version__", "__version_info__", *_MODULES]


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
