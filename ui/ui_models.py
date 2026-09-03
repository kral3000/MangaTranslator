from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, Optional

import torch

from core.tag_styles import default_tag_style_rows, normalize_tag_style_table, tag_styles_to_table
from core.config import (
    CleaningConfig,
    DEFAULT_FLUX_KLEIN_PROMPT,
    DEFAULT_FLUX_KONTEXT_PROMPT,
    DetectionConfig,
    MangaTranslatorConfig,
    OutputConfig,
    OutsideTextConfig,
    PreprocessingConfig,
    RenderingConfig,
    TagStyleConfig,
    TranslationConfig,
    normalize_outside_text_processing_mode,
)


def normalize_flux_offload_mode(
    value: Any,
    *,
    cpu_offload: bool = True,
    low_vram: bool = False,
) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"device", "model", "sequential"}:
        return mode
    if not cpu_offload:
        return "device"
    return "sequential" if low_vram else "model"


def flux_offload_flags(value: Any) -> tuple[bool, bool]:
    mode = normalize_flux_offload_mode(value)
    return mode != "device", mode == "sequential"


def _normalize_hex_color(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if text.startswith("#") and len(text) in {4, 7, 9}:
        try:
            int(text[1:], 16)
            if len(text) == 4:
                return "#" + "".join(channel * 2 for channel in text[1:]).lower()
            return text[:7].lower()
        except ValueError:
            pass

    color_match = re.fullmatch(
        r"rgba?\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,"
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,"
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"(?:\s*,\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)\s*)?\)",
        text,
        flags=re.IGNORECASE,
    )
    if color_match:
        channels = [
            max(0, min(255, int(round(float(channel)))))
            for channel in color_match.groups()
        ]
        return "#{:02x}{:02x}{:02x}".format(*channels)
    return fallback


def _hex_color_to_rgb(value: Any, fallback: str) -> tuple[int, int, int]:
    normalized = _normalize_hex_color(value, fallback)
    return tuple(int(normalized[index : index + 2], 16) for index in (1, 3, 5))


@dataclass
class UIDetectionSettings:
    """UI state for detection settings."""

    confidence: float = 0.6
    conjoined_confidence: float = 0.35
    panel_confidence: float = 0.25
    seg_model: str = "yolo"  # "sam3", "sam2", or "yolo"
    bubble_detector_model: str = "yolo_1"  # "yolo_1" or "yolo_2"
    conjoined_detection: bool = True
    use_panel_sorting: bool = True
    use_osb_text_verification: bool = True


@dataclass
class UICleaningSettings:
    """UI state for cleaning settings."""

    thresholding_value: int = 200
    use_otsu_threshold: bool = False
    roi_shrink_px: int = 5
    inpaint_colored_bubbles: bool = False
    colored_bubble_inpainting_size: int = 2048
    colored_bubble_inpainting_precision: str = "fp32"
    colored_bubble_mask_kernel_size: int = 3
    colored_bubble_mask_dilation_offset: int = 50


@dataclass
class UITranslationProviderSettings:
    """UI state for translation provider settings."""

    provider: str = "Google"
    google_api_key: Optional[str] = ""
    openai_api_key: Optional[str] = ""
    anthropic_api_key: Optional[str] = ""
    xai_api_key: Optional[str] = ""
    deepseek_api_key: Optional[str] = ""
    zai_api_key: Optional[str] = ""
    moonshot_api_key: Optional[str] = ""
    openrouter_api_key: Optional[str] = ""
    openai_compatible_url: str = "http://localhost:8080/v1"
    openai_compatible_api_key: Optional[str] = ""


@dataclass
class UIGoogleRateLimitSettings:
    """UI state for Google/Vertex request throttling settings."""

    enabled: bool = False
    request_interval_seconds: float = 0.0
    requests_per_minute: int = 0
    burst_requests: int = 0
    burst_cooldown_seconds: float = 60.0
    cooldown_429_seconds: float = 60.0
    max_retries: int = 3
    hourly_429_limit: int = 0
    hourly_429_action: str = "stop"
    pause_429_minutes: float = 5.0


@dataclass
class UITranslationLLMSettings:
    """UI state for LLM-specific translation settings."""

    model_name: Optional[str] = None
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: Optional[int] = None
    translation_mode: str = "one-step"
    reading_direction: str = "rtl"
    send_full_page_context: bool = True
    parallel_inpaint_translation: bool = False
    whiteout_conjoined_bubbles: bool = True
    upscale_method: str = "model_lite"
    bubble_min_side_pixels: int = 128
    context_image_max_side_pixels: int = 1024
    osb_min_side_pixels: int = 128
    special_instructions: Optional[str] = None
    ocr_method: str = "LLM"  # "LLM", "manga-ocr", or "paddleocr-vl"


@dataclass
class UIRenderingSettings:
    """UI state for rendering settings."""

    emphasis_font_dir: str = ""  # Empty = use main font bold variant
    max_font_size: int = 16
    emphasis_max_font_size: int = 16
    min_font_size: int = 8
    line_spacing_mult: float = 1.0
    emphasis_line_spacing_mult: float = 1.0
    vertical_line_spacing_mult: float = 1.0
    vertical_emphasis_line_spacing_mult: float = 1.0
    vertical_font_size_mult: float = 1.0
    vertical_max_chars: int = 4
    vertical_emphasis_max_chars: int = 4
    tag_style_table: Any = field(default_factory=default_tag_style_rows)
    use_subpixel_rendering: bool = True
    font_hinting: str = "none"
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True
    hyphen_penalty: float = 1000.0
    hyphenation_min_word_length: int = 8
    badness_exponent: float = 3.0
    padding_pixels: float = 5.0
    padding_horizontal_pixels: float = 5.0
    padding_vertical_pixels: float = 5.0
    supersampling_factor: int = 4
    detach_trailing_punctuation: bool = True
    auto_vertical_text: bool = False
    inside_light_text_color: str = "#000000"
    inside_light_outline_color: str = "#ffffff"
    inside_light_outline_width: float = 0.0
    inside_dark_text_color: str = "#ffffff"
    inside_dark_outline_color: str = "#000000"
    inside_dark_outline_width: float = 0.0
    outside_light_text_color: str = "#000000"
    outside_light_outline_color: str = "#ffffff"
    outside_light_outline_width: float = 3.0
    outside_dark_text_color: str = "#ffffff"
    outside_dark_outline_color: str = "#000000"
    outside_dark_outline_width: float = 3.0


@dataclass
class UIOutputSettings:
    """UI state for output settings."""

    output_directory: str = "output"
    disable_auto_save: bool = False
    output_format: str = "png"
    jpeg_quality: int = 95
    png_compression: int = 2
    embed_editor_package: bool = True
    save_plain_image_copy: bool = False
    image_upscale_mode: str = "off"  # "off", "initial", "final"
    image_upscale_factor: float = 2.0
    image_upscale_model: str = "model_lite"  # "model" or "model_lite"


@dataclass
class UIOutsideTextSettings:
    """UI state for outside speech bubble text removal settings."""

    enabled: bool = False
    processing_mode: str = "prose"
    enable_page_number_filtering: bool = False
    page_filter_margin_threshold: float = 0.1
    page_filter_min_area_ratio: float = 0.05
    seed: int = 1  # -1 = random
    huggingface_token: str = ""
    inpainting_method: str = (
        "flux_klein_4b"  # flux_klein_9b, flux_klein_4b, flux_kontext, opencv, none
    )
    kontext_backend: str = "sdnq"  # "sdnq" (cross-platform) or "nunchaku" (CUDA)
    flux_cpu_offload: bool = True
    flux_low_vram: bool = False  # Use sequential CPU offload for Klein/Kontext SDNQ
    flux_num_inference_steps: int = 8
    flux_kontext_prompt: str = DEFAULT_FLUX_KONTEXT_PROMPT
    flux_klein_prompt: str = DEFAULT_FLUX_KLEIN_PROMPT
    flux_luminance_correction: bool = (
        True  # Match patch luminance to surrounding context
    )
    flux_upscale_small_crops: bool = True
    flux_group_regions: bool = False
    flux_residual_diff_threshold: float = 0.15
    osb_residual_render_guard_action: str = "restore_original"
    osb_confidence: float = 0.6
    osb_font_dir: str = ""  # Empty = use main font
    osb_emphasis_font_dir: str = ""  # Empty = use OSB font bold variant
    osb_max_font_size: int = 64
    osb_emphasis_max_font_size: int = 64
    osb_min_font_size: int = 10
    osb_use_ligatures: bool = False
    osb_outline_width: float = 3.0
    osb_padding_pixels: float = 1.0
    osb_line_spacing: float = 1.0
    osb_vertical_line_spacing: float = 1.0
    osb_auto_vertical: bool = True
    osb_vertical_aspect_ratio: float = 1.45
    osb_vertical_max_chars: int = 30
    osb_tall_box_expand_aspect_ratio: float = 2.0
    osb_tall_box_expand_width_percent: float = 20.0
    osb_rotation_max_degrees: float = 15.0
    osb_use_subpixel_rendering: bool = True
    osb_font_hinting: str = "none"
    bbox_expansion_percent: float = 0.1
    osb_render_expansion_narrow_multiplier: float = 1.0
    osb_render_expansion_tiny_multiplier: float = 1.0
    osb_render_expansion_aspect_ratio_threshold: float = 0.4
    osb_render_expansion_area_ratio_threshold: float = 0.005
    text_box_proximity_ratio: float = 0.02


@dataclass
class UIGeneralSettings:
    """UI state for general application settings."""

    verbose: bool = False
    cleaning_only: bool = False
    upscaling_only: bool = False
    test_mode: bool = False
    enable_web_search: bool = (
        False  # Enable model's built-in web search for up-to-date information.
    )
    enable_code_execution: bool = (
        False  # Enable Gemini's code execution for image zoom/inspection.
    )
    image_detail: str = "auto"  # OpenAI image detail (auto/high/low/original)
    media_resolution: str = (
        "auto"  # Only available via Google provider (auto/high/medium/low)
    )
    media_resolution_bubbles: str = "auto"  # Gemini 3 models
    media_resolution_context: str = "auto"  # Gemini 3 models
    reasoning_effort: Optional[str] = None
    effort: Optional[str] = (
        None  # Opus 4.5/4.6/4.7, Sonnet 4.6 only: token spending eagerness
    )
    verbosity: Optional[str] = (
        None  # GPT-5 series only: controls response verbosity (high/medium/low)
    )
    auto_scale: bool = True


@dataclass
class UIConfigState:
    """Represents the complete configuration state managed by the UI."""

    detection: UIDetectionSettings = field(default_factory=UIDetectionSettings)
    cleaning: UICleaningSettings = field(default_factory=UICleaningSettings)
    provider_settings: UITranslationProviderSettings = field(
        default_factory=UITranslationProviderSettings
    )
    google_rate_limit: UIGoogleRateLimitSettings = field(
        default_factory=UIGoogleRateLimitSettings
    )
    llm_settings: UITranslationLLMSettings = field(
        default_factory=UITranslationLLMSettings
    )
    rendering: UIRenderingSettings = field(default_factory=UIRenderingSettings)
    output: UIOutputSettings = field(default_factory=UIOutputSettings)
    outside_text: UIOutsideTextSettings = field(default_factory=UIOutsideTextSettings)
    general: UIGeneralSettings = field(default_factory=UIGeneralSettings)

    # Specific UI elements state (saved in config.json)
    input_language: str = "Japanese"
    output_language: str = "English"
    font_pack: Optional[str] = None
    batch_input_language: str = "Japanese"
    batch_output_language: str = "English"
    batch_font_pack: Optional[str] = None
    batch_special_instructions: Optional[str] = None
    batch_parallel_requests: int = 1
    batch_retry_failed_once: bool = False
    batch_previous_context_image_count: int = 0
    batch_previous_context_text_count: int = 3
    batch_previous_context_text_mode: str = "strict"
    character_profile_name: str = ""

    def to_save_dict(self) -> Dict[str, Any]:
        """Converts the UI state into a dictionary suitable for saving to config.json."""
        data = {
            "confidence": self.detection.confidence,
            "conjoined_confidence": self.detection.conjoined_confidence,
            "panel_confidence": self.detection.panel_confidence,
            "seg_model": self.detection.seg_model,
            "bubble_detector_model": self.detection.bubble_detector_model,
            "conjoined_detection": self.detection.conjoined_detection,
            "use_panel_sorting": self.detection.use_panel_sorting,
            "use_osb_text_verification": self.detection.use_osb_text_verification,
            "reading_direction": self.llm_settings.reading_direction,
            "thresholding_value": self.cleaning.thresholding_value,
            "use_otsu_threshold": self.cleaning.use_otsu_threshold,
            "roi_shrink_px": self.cleaning.roi_shrink_px,
            "inpaint_colored_bubbles": self.cleaning.inpaint_colored_bubbles,
            "colored_bubble_inpainting_size": (
                self.cleaning.colored_bubble_inpainting_size
            ),
            "colored_bubble_inpainting_precision": (
                self.cleaning.colored_bubble_inpainting_precision
            ),
            "colored_bubble_mask_kernel_size": (
                self.cleaning.colored_bubble_mask_kernel_size
            ),
            "colored_bubble_mask_dilation_offset": (
                self.cleaning.colored_bubble_mask_dilation_offset
            ),
            "provider": self.provider_settings.provider,
            "google_api_key": self.provider_settings.google_api_key,
            "google_rate_limit_enabled": self.google_rate_limit.enabled,
            "google_request_interval_seconds": self.google_rate_limit.request_interval_seconds,
            "google_requests_per_minute": self.google_rate_limit.requests_per_minute,
            "google_burst_requests": self.google_rate_limit.burst_requests,
            "google_burst_cooldown_seconds": self.google_rate_limit.burst_cooldown_seconds,
            "google_429_cooldown_seconds": self.google_rate_limit.cooldown_429_seconds,
            "google_max_retries": self.google_rate_limit.max_retries,
            "google_429_hourly_limit": self.google_rate_limit.hourly_429_limit,
            "google_429_hourly_action": self.google_rate_limit.hourly_429_action,
            "google_429_pause_minutes": self.google_rate_limit.pause_429_minutes,
            "openai_api_key": self.provider_settings.openai_api_key,
            "anthropic_api_key": self.provider_settings.anthropic_api_key,
            "xai_api_key": self.provider_settings.xai_api_key,
            "deepseek_api_key": self.provider_settings.deepseek_api_key,
            "zai_api_key": self.provider_settings.zai_api_key,
            "moonshot_api_key": self.provider_settings.moonshot_api_key,
            "openrouter_api_key": self.provider_settings.openrouter_api_key,
            "openai_compatible_url": self.provider_settings.openai_compatible_url,
            "openai_compatible_api_key": self.provider_settings.openai_compatible_api_key,
            "model_name": self.llm_settings.model_name,
            "temperature": self.llm_settings.temperature,
            "top_p": self.llm_settings.top_p,
            "top_k": self.llm_settings.top_k,
            "max_tokens": self.llm_settings.max_tokens,
            "translation_mode": self.llm_settings.translation_mode,
            "ocr_method": self.llm_settings.ocr_method,
            "send_full_page_context": self.llm_settings.send_full_page_context,
            "parallel_inpaint_translation": self.llm_settings.parallel_inpaint_translation,
            "whiteout_conjoined_bubbles": self.llm_settings.whiteout_conjoined_bubbles,
            "upscale_method": self.llm_settings.upscale_method,
            "bubble_min_side_pixels": self.llm_settings.bubble_min_side_pixels,
            "context_image_max_side_pixels": self.llm_settings.context_image_max_side_pixels,
            "osb_min_side_pixels": self.llm_settings.osb_min_side_pixels,
            "special_instructions": self.llm_settings.special_instructions or "",
            "font_pack": self.font_pack,
            "emphasis_font_pack": self.rendering.emphasis_font_dir,
            "max_font_size": self.rendering.max_font_size,
            "emphasis_max_font_size": self.rendering.emphasis_max_font_size,
            "min_font_size": self.rendering.min_font_size,
            "line_spacing_mult": self.rendering.line_spacing_mult,
            "emphasis_line_spacing_mult": self.rendering.emphasis_line_spacing_mult,
            "vertical_line_spacing_mult": self.rendering.vertical_line_spacing_mult,
            "vertical_emphasis_line_spacing_mult": self.rendering.vertical_emphasis_line_spacing_mult,
            "vertical_font_size_mult": self.rendering.vertical_font_size_mult,
            "vertical_max_chars": self.rendering.vertical_max_chars,
            "vertical_emphasis_max_chars": self.rendering.vertical_emphasis_max_chars,
            "tag_styles": normalize_tag_style_table(self.rendering.tag_style_table),
            "tag_style_table": tag_styles_to_table(self.rendering.tag_style_table),
            "use_subpixel_rendering": self.rendering.use_subpixel_rendering,
            "font_hinting": self.rendering.font_hinting,
            "use_ligatures": self.rendering.use_ligatures,
            "hyphenate_before_scaling": self.rendering.hyphenate_before_scaling,
            "hyphen_penalty": self.rendering.hyphen_penalty,
            "hyphenation_min_word_length": self.rendering.hyphenation_min_word_length,
            "badness_exponent": self.rendering.badness_exponent,
            "padding_pixels": self.rendering.padding_pixels,
            "padding_horizontal_pixels": self.rendering.padding_horizontal_pixels,
            "padding_vertical_pixels": self.rendering.padding_vertical_pixels,
            "supersampling_factor": self.rendering.supersampling_factor,
            "detach_trailing_punctuation": self.rendering.detach_trailing_punctuation,
            "auto_vertical_text": self.rendering.auto_vertical_text,
            "inside_light_text_color": self.rendering.inside_light_text_color,
            "inside_light_outline_color": self.rendering.inside_light_outline_color,
            "inside_light_outline_width": self.rendering.inside_light_outline_width,
            "inside_dark_text_color": self.rendering.inside_dark_text_color,
            "inside_dark_outline_color": self.rendering.inside_dark_outline_color,
            "inside_dark_outline_width": self.rendering.inside_dark_outline_width,
            "outside_light_text_color": self.rendering.outside_light_text_color,
            "outside_light_outline_color": self.rendering.outside_light_outline_color,
            "outside_light_outline_width": self.rendering.outside_light_outline_width,
            "outside_dark_text_color": self.rendering.outside_dark_text_color,
            "outside_dark_outline_color": self.rendering.outside_dark_outline_color,
            "outside_dark_outline_width": self.rendering.outside_dark_outline_width,
            "outside_text_enabled": self.outside_text.enabled,
            "outside_text_processing_mode": self.outside_text.processing_mode,
            "outside_text_seed": self.outside_text.seed,
            "outside_text_huggingface_token": self.outside_text.huggingface_token,
            "outside_text_inpainting_method": self.outside_text.inpainting_method,
            "outside_text_kontext_backend": self.outside_text.kontext_backend,
            "outside_text_flux_offload_mode": normalize_flux_offload_mode(
                None,
                cpu_offload=self.outside_text.flux_cpu_offload,
                low_vram=self.outside_text.flux_low_vram,
            ),
            "outside_text_flux_cpu_offload": self.outside_text.flux_cpu_offload,
            "outside_text_flux_low_vram": self.outside_text.flux_low_vram,
            "outside_text_flux_num_inference_steps": self.outside_text.flux_num_inference_steps,
            "outside_text_flux_kontext_prompt": self.outside_text.flux_kontext_prompt,
            "outside_text_flux_klein_prompt": self.outside_text.flux_klein_prompt,
            "outside_text_flux_luminance_correction": self.outside_text.flux_luminance_correction,
            "outside_text_flux_upscale_small_crops": self.outside_text.flux_upscale_small_crops,
            "outside_text_flux_group_regions": self.outside_text.flux_group_regions,
            "outside_text_flux_residual_diff_threshold": self.outside_text.flux_residual_diff_threshold,
            "outside_text_osb_residual_render_guard_action": self.outside_text.osb_residual_render_guard_action,
            "outside_text_osb_confidence": self.outside_text.osb_confidence,
            "outside_text_enable_page_number_filtering": self.outside_text.enable_page_number_filtering,
            "outside_text_page_filter_margin_threshold": self.outside_text.page_filter_margin_threshold,
            "outside_text_page_filter_min_area_ratio": self.outside_text.page_filter_min_area_ratio,
            "outside_text_osb_font_pack": self.outside_text.osb_font_dir,
            "outside_text_osb_emphasis_font_pack": self.outside_text.osb_emphasis_font_dir,
            "outside_text_osb_max_font_size": self.outside_text.osb_max_font_size,
            "outside_text_osb_emphasis_max_font_size": self.outside_text.osb_emphasis_max_font_size,
            "outside_text_osb_min_font_size": self.outside_text.osb_min_font_size,
            "outside_text_osb_use_ligatures": self.outside_text.osb_use_ligatures,
            "outside_text_osb_outline_width": self.outside_text.osb_outline_width,
            "outside_text_osb_padding_pixels": self.outside_text.osb_padding_pixels,
            "outside_text_osb_line_spacing": self.outside_text.osb_line_spacing,
            "outside_text_osb_vertical_line_spacing": (
                self.outside_text.osb_vertical_line_spacing
            ),
            "outside_text_osb_auto_vertical": self.outside_text.osb_auto_vertical,
            "outside_text_osb_vertical_aspect_ratio": self.outside_text.osb_vertical_aspect_ratio,
            "outside_text_osb_vertical_max_chars": self.outside_text.osb_vertical_max_chars,
            "outside_text_osb_tall_box_expand_aspect_ratio": (
                self.outside_text.osb_tall_box_expand_aspect_ratio
            ),
            "outside_text_osb_tall_box_expand_width_percent": (
                self.outside_text.osb_tall_box_expand_width_percent
            ),
            "outside_text_osb_rotation_max_degrees": self.outside_text.osb_rotation_max_degrees,
            "outside_text_osb_use_subpixel_rendering": self.outside_text.osb_use_subpixel_rendering,
            "outside_text_osb_font_hinting": self.outside_text.osb_font_hinting,
            "outside_text_bbox_expansion_percent": self.outside_text.bbox_expansion_percent,
            "outside_text_osb_render_expansion_narrow_multiplier": (
                self.outside_text.osb_render_expansion_narrow_multiplier
            ),
            "outside_text_osb_render_expansion_tiny_multiplier": (
                self.outside_text.osb_render_expansion_tiny_multiplier
            ),
            "outside_text_osb_render_expansion_aspect_ratio_threshold": (
                self.outside_text.osb_render_expansion_aspect_ratio_threshold
            ),
            "outside_text_osb_render_expansion_area_ratio_threshold": (
                self.outside_text.osb_render_expansion_area_ratio_threshold
            ),
            "outside_text_text_box_proximity_ratio": self.outside_text.text_box_proximity_ratio,
            "output_directory": self.output.output_directory,
            "disable_auto_save": self.output.disable_auto_save,
            "output_format": self.output.output_format,
            "jpeg_quality": self.output.jpeg_quality,
            "png_compression": self.output.png_compression,
            "embed_editor_package": self.output.embed_editor_package,
            "save_plain_image_copy": self.output.save_plain_image_copy,
            "image_upscale_mode": self.output.image_upscale_mode,
            "image_upscale_factor": self.output.image_upscale_factor,
            "image_upscale_model": self.output.image_upscale_model,
            "verbose": self.general.verbose,
            "cleaning_only": self.general.cleaning_only,
            "upscaling_only": self.general.upscaling_only,
            "test_mode": self.general.test_mode,
            "enable_web_search": self.general.enable_web_search,
            "enable_code_execution": self.general.enable_code_execution,
            "image_detail": self.general.image_detail,
            "media_resolution": self.general.media_resolution,
            "media_resolution_bubbles": self.general.media_resolution_bubbles,
            "media_resolution_context": self.general.media_resolution_context,
            "reasoning_effort": self.general.reasoning_effort,
            "effort": self.general.effort,
            "verbosity": self.general.verbosity,
            "auto_scale": self.general.auto_scale,
            "input_language": self.input_language,
            "output_language": self.output_language,
            "batch_input_language": self.batch_input_language,
            "batch_output_language": self.batch_output_language,
            "batch_font_pack": self.batch_font_pack,
            "batch_special_instructions": self.batch_special_instructions or "",
            "batch_parallel_requests": self.batch_parallel_requests,
            "batch_retry_failed_once": self.batch_retry_failed_once,
            "batch_previous_context_image_count": (
                self.batch_previous_context_image_count
                if (
                    self.llm_settings.send_full_page_context
                    and self.llm_settings.ocr_method == "LLM"
                )
                else 0
            ),
            "batch_previous_context_text_count": self.batch_previous_context_text_count,
            "batch_previous_context_text_mode": self.batch_previous_context_text_mode,
        }
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "UIConfigState":
        """Creates a UIConfigState instance from a dictionary (e.g., loaded from config.json)."""

        from . import (  # Local import to avoid circular dependency issues
            settings_manager,
        )

        defaults = settings_manager.DEFAULT_SETTINGS.copy()
        defaults.update(settings_manager.DEFAULT_BATCH_SETTINGS)
        legacy_padding_pixels = data.get(
            "padding_pixels", defaults.get("padding_pixels", 5.0)
        )
        padding_horizontal_pixels = data.get(
            "padding_horizontal_pixels", legacy_padding_pixels
        )
        padding_vertical_pixels = data.get(
            "padding_vertical_pixels", legacy_padding_pixels
        )
        legacy_outside_text_enabled = bool(
            data.get(
                "outside_text_enabled",
                defaults.get("outside_text_enabled", False),
            )
        )
        outside_text_processing_mode = normalize_outside_text_processing_mode(
            data.get("outside_text_processing_mode")
            if "outside_text_processing_mode" in data
            else (
                "all"
                if legacy_outside_text_enabled
                else "off"
                if "outside_text_enabled" in data
                else defaults.get("outside_text_processing_mode", "prose")
            ),
            enabled=legacy_outside_text_enabled,
        )

        return UIConfigState(
            detection=UIDetectionSettings(
                confidence=data.get("confidence", defaults["confidence"]),
                conjoined_confidence=data.get(
                    "conjoined_confidence", defaults.get("conjoined_confidence", 0.35)
                ),
                panel_confidence=data.get(
                    "panel_confidence", defaults.get("panel_confidence", 0.25)
                ),
                seg_model=data.get("seg_model", defaults.get("seg_model", "yolo")),
                bubble_detector_model=data.get(
                    "bubble_detector_model",
                    defaults.get("bubble_detector_model", "yolo_1"),
                ),
                conjoined_detection=data.get(
                    "conjoined_detection",
                    defaults.get("conjoined_detection", True),
                ),
                use_osb_text_verification=data.get(
                    "use_osb_text_verification",
                    defaults.get("use_osb_text_verification", True),
                ),
                use_panel_sorting=data.get(
                    "use_panel_sorting",
                    defaults.get("use_panel_sorting", True),
                ),
            ),
            cleaning=UICleaningSettings(
                thresholding_value=data.get(
                    "thresholding_value", defaults["thresholding_value"]
                ),
                use_otsu_threshold=data.get(
                    "use_otsu_threshold", defaults["use_otsu_threshold"]
                ),
                roi_shrink_px=data.get(
                    "roi_shrink_px", defaults.get("roi_shrink_px", 5)
                ),
                inpaint_colored_bubbles=data.get(
                    "inpaint_colored_bubbles",
                    defaults.get("inpaint_colored_bubbles", False),
                ),
                colored_bubble_inpainting_size=data.get(
                    "colored_bubble_inpainting_size",
                    defaults.get("colored_bubble_inpainting_size", 2048),
                ),
                colored_bubble_inpainting_precision=data.get(
                    "colored_bubble_inpainting_precision",
                    defaults.get("colored_bubble_inpainting_precision", "fp32"),
                ),
                colored_bubble_mask_kernel_size=data.get(
                    "colored_bubble_mask_kernel_size",
                    defaults.get("colored_bubble_mask_kernel_size", 3),
                ),
                colored_bubble_mask_dilation_offset=data.get(
                    "colored_bubble_mask_dilation_offset",
                    defaults.get("colored_bubble_mask_dilation_offset", 50),
                ),
            ),
            outside_text=UIOutsideTextSettings(
                enabled=outside_text_processing_mode != "off",
                processing_mode=outside_text_processing_mode,
                enable_page_number_filtering=data.get(
                    "outside_text_enable_page_number_filtering",
                    defaults.get("outside_text_enable_page_number_filtering", False),
                ),
                page_filter_margin_threshold=data.get(
                    "outside_text_page_filter_margin_threshold",
                    defaults.get("outside_text_page_filter_margin_threshold", 0.1),
                ),
                page_filter_min_area_ratio=data.get(
                    "outside_text_page_filter_min_area_ratio",
                    defaults.get("outside_text_page_filter_min_area_ratio", 0.05),
                ),
                seed=data.get("outside_text_seed", 1),
                huggingface_token=data.get("outside_text_huggingface_token", ""),
                inpainting_method=data.get(
                    "outside_text_inpainting_method",
                    defaults.get("outside_text_inpainting_method", "flux_klein_4b"),
                ),
                kontext_backend=data.get(
                    "outside_text_kontext_backend",
                    defaults.get("outside_text_kontext_backend", "sdnq"),
                ),
                flux_cpu_offload=flux_offload_flags(
                    normalize_flux_offload_mode(
                        data.get("outside_text_flux_offload_mode"),
                        cpu_offload=data.get(
                            "outside_text_flux_cpu_offload",
                            defaults.get("outside_text_flux_cpu_offload", True),
                        ),
                        low_vram=data.get("outside_text_flux_low_vram", False),
                    )
                )[0],
                flux_low_vram=flux_offload_flags(
                    normalize_flux_offload_mode(
                        data.get("outside_text_flux_offload_mode"),
                        cpu_offload=data.get(
                            "outside_text_flux_cpu_offload",
                            defaults.get("outside_text_flux_cpu_offload", True),
                        ),
                        low_vram=data.get("outside_text_flux_low_vram", False),
                    )
                )[1],
                flux_num_inference_steps=data.get(
                    "outside_text_flux_num_inference_steps", 8
                ),
                flux_kontext_prompt=data.get(
                    "outside_text_flux_kontext_prompt",
                    defaults.get(
                        "outside_text_flux_kontext_prompt",
                        DEFAULT_FLUX_KONTEXT_PROMPT,
                    ),
                ),
                flux_klein_prompt=data.get(
                    "outside_text_flux_klein_prompt",
                    defaults.get(
                        "outside_text_flux_klein_prompt",
                        DEFAULT_FLUX_KLEIN_PROMPT,
                    ),
                ),
                flux_luminance_correction=data.get(
                    "outside_text_flux_luminance_correction", True
                ),
                flux_upscale_small_crops=data.get(
                    "outside_text_flux_upscale_small_crops", True
                ),
                flux_group_regions=data.get("outside_text_flux_group_regions", False),
                flux_residual_diff_threshold=data.get(
                    "outside_text_flux_residual_diff_threshold", 0.15
                ),
                osb_residual_render_guard_action=data.get(
                    "outside_text_osb_residual_render_guard_action",
                    defaults.get(
                        "outside_text_osb_residual_render_guard_action",
                        "restore_original",
                    ),
                ),
                osb_confidence=data.get("outside_text_osb_confidence", 0.6),
                osb_font_dir=data.get(
                    "outside_text_osb_font_pack",
                    defaults.get("outside_text_osb_font_pack", ""),
                ),
                osb_emphasis_font_dir=data.get(
                    "outside_text_osb_emphasis_font_pack",
                    defaults.get("outside_text_osb_emphasis_font_pack", ""),
                ),
                osb_max_font_size=data.get("outside_text_osb_max_font_size", 64),
                osb_emphasis_max_font_size=data.get(
                    "outside_text_osb_emphasis_max_font_size",
                    defaults.get("outside_text_osb_emphasis_max_font_size", 64),
                ),
                osb_min_font_size=data.get("outside_text_osb_min_font_size", 10),
                osb_use_ligatures=data.get("outside_text_osb_use_ligatures", False),
                osb_outline_width=data.get("outside_text_osb_outline_width", 3.0),
                osb_padding_pixels=data.get(
                    "outside_text_osb_padding_pixels",
                    defaults.get("outside_text_osb_padding_pixels", 1.0),
                ),
                osb_line_spacing=data.get("outside_text_osb_line_spacing", 1.0),
                osb_vertical_line_spacing=data.get(
                    "outside_text_osb_vertical_line_spacing",
                    data.get("outside_text_osb_line_spacing", 1.0),
                ),
                osb_auto_vertical=data.get(
                    "outside_text_osb_auto_vertical",
                    defaults.get("outside_text_osb_auto_vertical", True),
                ),
                osb_vertical_aspect_ratio=data.get(
                    "outside_text_osb_vertical_aspect_ratio",
                    defaults.get("outside_text_osb_vertical_aspect_ratio", 1.45),
                ),
                osb_vertical_max_chars=data.get(
                    "outside_text_osb_vertical_max_chars",
                    defaults.get("outside_text_osb_vertical_max_chars", 30),
                ),
                osb_tall_box_expand_aspect_ratio=data.get(
                    "outside_text_osb_tall_box_expand_aspect_ratio",
                    defaults.get(
                        "outside_text_osb_tall_box_expand_aspect_ratio", 2.0
                    ),
                ),
                osb_tall_box_expand_width_percent=data.get(
                    "outside_text_osb_tall_box_expand_width_percent",
                    defaults.get(
                        "outside_text_osb_tall_box_expand_width_percent", 20.0
                    ),
                ),
                osb_rotation_max_degrees=data.get(
                    "outside_text_osb_rotation_max_degrees",
                    defaults.get("outside_text_osb_rotation_max_degrees", 15.0),
                ),
                osb_use_subpixel_rendering=data.get(
                    "outside_text_osb_use_subpixel_rendering", True
                ),
                osb_font_hinting=data.get("outside_text_osb_font_hinting", "none"),
                bbox_expansion_percent=data.get(
                    "outside_text_bbox_expansion_percent", 0.1
                ),
                osb_render_expansion_narrow_multiplier=data.get(
                    "outside_text_osb_render_expansion_narrow_multiplier",
                    defaults.get(
                        "outside_text_osb_render_expansion_narrow_multiplier", 1.0
                    ),
                ),
                osb_render_expansion_tiny_multiplier=data.get(
                    "outside_text_osb_render_expansion_tiny_multiplier",
                    defaults.get(
                        "outside_text_osb_render_expansion_tiny_multiplier", 1.0
                    ),
                ),
                osb_render_expansion_aspect_ratio_threshold=data.get(
                    "outside_text_osb_render_expansion_aspect_ratio_threshold", 0.4
                ),
                osb_render_expansion_area_ratio_threshold=data.get(
                    "outside_text_osb_render_expansion_area_ratio_threshold", 0.005
                ),
                text_box_proximity_ratio=data.get(
                    "outside_text_text_box_proximity_ratio", 0.02
                ),
            ),
            provider_settings=UITranslationProviderSettings(
                provider=data.get("provider", defaults["provider"]),
                google_api_key=data.get(
                    "google_api_key", defaults.get("google_api_key", "")
                ),
                openai_api_key=data.get("openai_api_key", defaults["openai_api_key"]),
                anthropic_api_key=data.get(
                    "anthropic_api_key", defaults["anthropic_api_key"]
                ),
                xai_api_key=data.get("xai_api_key", defaults["xai_api_key"]),
                deepseek_api_key=data.get(
                    "deepseek_api_key", defaults.get("deepseek_api_key", "")
                ),
                zai_api_key=data.get("zai_api_key", defaults.get("zai_api_key", "")),
                moonshot_api_key=data.get(
                    "moonshot_api_key", defaults.get("moonshot_api_key", "")
                ),
                openrouter_api_key=data.get(
                    "openrouter_api_key", defaults["openrouter_api_key"]
                ),
                openai_compatible_url=data.get(
                    "openai_compatible_url", defaults["openai_compatible_url"]
                ),
                openai_compatible_api_key=data.get(
                    "openai_compatible_api_key", defaults["openai_compatible_api_key"]
                ),
            ),
            google_rate_limit=UIGoogleRateLimitSettings(
                enabled=data.get(
                    "google_rate_limit_enabled",
                    defaults.get("google_rate_limit_enabled", False),
                ),
                request_interval_seconds=data.get(
                    "google_request_interval_seconds",
                    defaults.get("google_request_interval_seconds", 0.0),
                ),
                requests_per_minute=data.get(
                    "google_requests_per_minute",
                    defaults.get("google_requests_per_minute", 0),
                ),
                burst_requests=data.get(
                    "google_burst_requests",
                    defaults.get("google_burst_requests", 0),
                ),
                burst_cooldown_seconds=data.get(
                    "google_burst_cooldown_seconds",
                    defaults.get("google_burst_cooldown_seconds", 60.0),
                ),
                cooldown_429_seconds=data.get(
                    "google_429_cooldown_seconds",
                    defaults.get("google_429_cooldown_seconds", 60.0),
                ),
                max_retries=data.get(
                    "google_max_retries", defaults.get("google_max_retries", 3)
                ),
                hourly_429_limit=data.get(
                    "google_429_hourly_limit",
                    defaults.get("google_429_hourly_limit", 0),
                ),
                hourly_429_action=(
                    data.get(
                        "google_429_hourly_action",
                        defaults.get("google_429_hourly_action", "stop"),
                    )
                    if str(
                        data.get(
                            "google_429_hourly_action",
                            defaults.get("google_429_hourly_action", "stop"),
                        )
                    ).strip().lower()
                    in {"stop", "pause"}
                    else "stop"
                ),
                pause_429_minutes=data.get(
                    "google_429_pause_minutes",
                    defaults.get("google_429_pause_minutes", 5.0),
                ),
            ),
            llm_settings=UITranslationLLMSettings(
                model_name=data.get("model_name"),
                temperature=data.get("temperature", defaults["temperature"]),
                top_p=data.get("top_p", defaults["top_p"]),
                top_k=data.get("top_k", defaults["top_k"]),
                max_tokens=data.get("max_tokens", defaults.get("max_tokens")),
                translation_mode=data.get(
                    "translation_mode", defaults["translation_mode"]
                ),
                ocr_method=data.get("ocr_method", defaults.get("ocr_method", "LLM")),
                reading_direction=data.get(
                    "reading_direction", defaults["reading_direction"]
                ),
                send_full_page_context=data.get("send_full_page_context", True),
                parallel_inpaint_translation=data.get(
                    "parallel_inpaint_translation", False
                ),
                whiteout_conjoined_bubbles=data.get("whiteout_conjoined_bubbles", True),
                upscale_method=data.get(
                    "upscale_method", defaults.get("upscale_method", "model_lite")
                ),
                bubble_min_side_pixels=data.get("bubble_min_side_pixels", 128),
                context_image_max_side_pixels=data.get(
                    "context_image_max_side_pixels", 1024
                ),
                osb_min_side_pixels=data.get("osb_min_side_pixels", 128),
                special_instructions=data.get("special_instructions") or None,
            ),
            rendering=UIRenderingSettings(
                emphasis_font_dir=data.get(
                    "emphasis_font_pack", defaults.get("emphasis_font_pack", "")
                ),
                max_font_size=data.get("max_font_size", defaults["max_font_size"]),
                emphasis_max_font_size=data.get(
                    "emphasis_max_font_size",
                    defaults.get("emphasis_max_font_size", defaults["max_font_size"]),
                ),
                min_font_size=data.get("min_font_size", defaults["min_font_size"]),
                line_spacing_mult=data.get(
                    "line_spacing_mult", defaults["line_spacing_mult"]
                ),
                emphasis_line_spacing_mult=data.get(
                    "emphasis_line_spacing_mult",
                    data.get(
                        "line_spacing_mult",
                        defaults.get(
                            "emphasis_line_spacing_mult",
                            defaults["line_spacing_mult"],
                        ),
                    ),
                ),
                vertical_line_spacing_mult=data.get(
                    "vertical_line_spacing_mult",
                    data.get(
                        "line_spacing_mult",
                        defaults.get(
                            "vertical_line_spacing_mult",
                            defaults["line_spacing_mult"],
                        ),
                    ),
                ),
                vertical_emphasis_line_spacing_mult=data.get(
                    "vertical_emphasis_line_spacing_mult",
                    data.get(
                        "emphasis_line_spacing_mult",
                        data.get(
                            "vertical_line_spacing_mult",
                            defaults.get("vertical_emphasis_line_spacing_mult", 1.0),
                        ),
                    ),
                ),
                vertical_font_size_mult=data.get(
                    "vertical_font_size_mult",
                    defaults.get("vertical_font_size_mult", 1.0),
                ),
                vertical_max_chars=data.get(
                    "vertical_max_chars",
                    data.get(
                        "vertical_min_chars",  # migrate old test builds that used the wrong setting name
                        defaults.get("vertical_max_chars", 4),
                    ),
                ),
                vertical_emphasis_max_chars=data.get(
                    "vertical_emphasis_max_chars",
                    data.get(
                        "vertical_emphasis_min_chars",  # migrate old test builds that used the wrong setting name
                        data.get(
                            "vertical_max_chars",
                            data.get("vertical_min_chars", defaults.get("vertical_emphasis_max_chars", 4)),
                        ),
                    ),
                ),
                tag_style_table=tag_styles_to_table(
                    data.get("tag_styles", data.get("tag_style_table", defaults.get("tag_styles")))
                ),
                use_subpixel_rendering=data.get(
                    "use_subpixel_rendering", defaults["use_subpixel_rendering"]
                ),
                font_hinting=data.get("font_hinting", defaults["font_hinting"]),
                use_ligatures=data.get("use_ligatures", defaults["use_ligatures"]),
                hyphenate_before_scaling=data.get(
                    "hyphenate_before_scaling",
                    defaults.get("hyphenate_before_scaling", True),
                ),
                hyphen_penalty=data.get(
                    "hyphen_penalty", defaults.get("hyphen_penalty", 1000.0)
                ),
                hyphenation_min_word_length=data.get(
                    "hyphenation_min_word_length",
                    defaults.get("hyphenation_min_word_length", 8),
                ),
                badness_exponent=data.get(
                    "badness_exponent", defaults.get("badness_exponent", 3.0)
                ),
                padding_pixels=data.get(
                    "padding_pixels", defaults.get("padding_pixels", 5.0)
                ),
                padding_horizontal_pixels=padding_horizontal_pixels,
                padding_vertical_pixels=padding_vertical_pixels,
                supersampling_factor=data.get(
                    "supersampling_factor", defaults.get("supersampling_factor", 4)
                ),
                detach_trailing_punctuation=data.get(
                    "detach_trailing_punctuation",
                    defaults.get("detach_trailing_punctuation", True),
                ),
                auto_vertical_text=data.get(
                    "auto_vertical_text",
                    defaults.get("auto_vertical_text", False),
                ),
                inside_light_text_color=_normalize_hex_color(
                    data.get("inside_light_text_color"), "#000000"
                ),
                inside_light_outline_color=_normalize_hex_color(
                    data.get("inside_light_outline_color"), "#ffffff"
                ),
                inside_light_outline_width=float(
                    data.get("inside_light_outline_width", 0.0)
                ),
                inside_dark_text_color=_normalize_hex_color(
                    data.get("inside_dark_text_color"), "#ffffff"
                ),
                inside_dark_outline_color=_normalize_hex_color(
                    data.get("inside_dark_outline_color"), "#000000"
                ),
                inside_dark_outline_width=float(
                    data.get("inside_dark_outline_width", 0.0)
                ),
                outside_light_text_color=_normalize_hex_color(
                    data.get("outside_light_text_color"), "#000000"
                ),
                outside_light_outline_color=_normalize_hex_color(
                    data.get("outside_light_outline_color"), "#ffffff"
                ),
                outside_light_outline_width=float(
                    data.get(
                        "outside_light_outline_width",
                        data.get("outside_text_osb_outline_width", 3.0),
                    )
                ),
                outside_dark_text_color=_normalize_hex_color(
                    data.get("outside_dark_text_color"), "#ffffff"
                ),
                outside_dark_outline_color=_normalize_hex_color(
                    data.get("outside_dark_outline_color"), "#000000"
                ),
                outside_dark_outline_width=float(
                    data.get(
                        "outside_dark_outline_width",
                        data.get("outside_text_osb_outline_width", 3.0),
                    )
                ),
            ),
            output=UIOutputSettings(
                output_directory=data.get(
                    "output_directory", defaults.get("output_directory", "output")
                ),
                disable_auto_save=data.get(
                    "disable_auto_save",
                    defaults.get("disable_auto_save", False),
                ),
                output_format=data.get("output_format", defaults["output_format"]),
                jpeg_quality=data.get("jpeg_quality", defaults["jpeg_quality"]),
                png_compression=data.get(
                    "png_compression", defaults["png_compression"]
                ),
                embed_editor_package=data.get(
                    "embed_editor_package",
                    defaults.get("embed_editor_package", True),
                ),
                save_plain_image_copy=data.get(
                    "save_plain_image_copy",
                    defaults.get("save_plain_image_copy", False),
                ),
                image_upscale_mode=data.get(
                    "image_upscale_mode", defaults.get("image_upscale_mode", "off")
                ),
                image_upscale_factor=data.get(
                    "image_upscale_factor",
                    defaults.get("image_upscale_factor", 2.0),
                ),
                image_upscale_model=data.get(
                    "image_upscale_model",
                    defaults.get("image_upscale_model", "model_lite"),
                ),
            ),
            general=UIGeneralSettings(
                verbose=data.get("verbose", defaults["verbose"]),
                cleaning_only=data.get("cleaning_only", defaults["cleaning_only"]),
                upscaling_only=data.get(
                    "upscaling_only", defaults.get("upscaling_only", False)
                ),
                test_mode=data.get("test_mode", defaults.get("test_mode", False)),
                enable_web_search=data.get(
                    "enable_web_search", defaults.get("enable_web_search", False)
                ),
                enable_code_execution=data.get(
                    "enable_code_execution",
                    defaults.get("enable_code_execution", False),
                ),
                image_detail=data.get(
                    "image_detail",
                    defaults.get("image_detail", "auto"),
                ),
                media_resolution=data.get(
                    "media_resolution",
                    defaults.get("media_resolution", "auto"),
                ),
                media_resolution_bubbles=data.get(
                    "media_resolution_bubbles",
                    defaults.get("media_resolution_bubbles", "auto"),
                ),
                media_resolution_context=data.get(
                    "media_resolution_context",
                    defaults.get("media_resolution_context", "auto"),
                ),
                reasoning_effort=data.get(
                    "reasoning_effort", defaults.get("reasoning_effort")
                ),
                effort=data.get("effort", defaults.get("effort", "medium")),
                verbosity=data.get("verbosity", defaults.get("verbosity", "low")),
                auto_scale=data.get("auto_scale", defaults.get("auto_scale", True)),
            ),
            input_language=data.get("input_language", defaults["input_language"]),
            output_language=data.get("output_language", defaults["output_language"]),
            font_pack=data.get("font_pack"),
            batch_input_language=data.get(
                "batch_input_language", defaults["batch_input_language"]
            ),
            batch_output_language=data.get(
                "batch_output_language", defaults["batch_output_language"]
            ),
            batch_font_pack=data.get("batch_font_pack"),
            batch_special_instructions=data.get("batch_special_instructions") or None,
            batch_parallel_requests=int(data.get("batch_parallel_requests", 1)),
            batch_retry_failed_once=bool(
                data.get("batch_retry_failed_once", defaults.get("batch_retry_failed_once", False))
            ),
            batch_previous_context_image_count=int(
                data.get("batch_previous_context_image_count", 0)
            ),
            batch_previous_context_text_count=int(
                data.get("batch_previous_context_text_count", 3)
            ),
            batch_previous_context_text_mode=(
                data.get("batch_previous_context_text_mode", "strict")
                if data.get("batch_previous_context_text_mode", "strict")
                in {"strict", "available"}
                else "strict"
            ),
        )


def map_ui_to_backend_config(
    ui_state: UIConfigState,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
    is_batch: bool = False,
) -> MangaTranslatorConfig:
    """Maps the UIConfigState to the backend MangaTranslatorConfig."""

    yolo_path = ""
    font_pack_name = ui_state.batch_font_pack if is_batch else ui_state.font_pack
    font_dir_path = fonts_base_dir / font_pack_name if font_pack_name else ""
    emphasis_font_path = (
        fonts_base_dir / ui_state.rendering.emphasis_font_dir
        if ui_state.rendering.emphasis_font_dir
        else None
    )
    input_lang = ui_state.batch_input_language if is_batch else ui_state.input_language
    output_lang = (
        ui_state.batch_output_language if is_batch else ui_state.output_language
    )

    detection_cfg = DetectionConfig(
        confidence=ui_state.detection.confidence,
        panel_confidence=ui_state.detection.panel_confidence,
    )
    detection_cfg.conjoined_confidence = ui_state.detection.conjoined_confidence
    detection_cfg.seg_model = ui_state.detection.seg_model
    detection_cfg.bubble_detector_model = ui_state.detection.bubble_detector_model
    detection_cfg.conjoined_detection = ui_state.detection.conjoined_detection
    detection_cfg.use_panel_sorting = ui_state.detection.use_panel_sorting
    detection_cfg.use_osb_text_verification = (
        ui_state.detection.use_osb_text_verification
    )

    cleaning_cfg = CleaningConfig(
        thresholding_value=ui_state.cleaning.thresholding_value,
        use_otsu_threshold=ui_state.cleaning.use_otsu_threshold,
        roi_shrink_px=ui_state.cleaning.roi_shrink_px,
        inpaint_colored_bubbles=ui_state.cleaning.inpaint_colored_bubbles,
        colored_bubble_inpainting_size=(
            ui_state.cleaning.colored_bubble_inpainting_size
        ),
        colored_bubble_inpainting_precision=(
            ui_state.cleaning.colored_bubble_inpainting_precision
        ),
        colored_bubble_mask_kernel_size=(
            ui_state.cleaning.colored_bubble_mask_kernel_size
        ),
        colored_bubble_mask_dilation_offset=(
            ui_state.cleaning.colored_bubble_mask_dilation_offset
        ),
    )

    translation_cfg = TranslationConfig(
        provider=ui_state.provider_settings.provider,
        google_api_key=ui_state.provider_settings.google_api_key or "",
        openai_api_key=ui_state.provider_settings.openai_api_key or "",
        anthropic_api_key=ui_state.provider_settings.anthropic_api_key or "",
        xai_api_key=ui_state.provider_settings.xai_api_key or "",
        deepseek_api_key=ui_state.provider_settings.deepseek_api_key or "",
        zai_api_key=ui_state.provider_settings.zai_api_key or "",
        moonshot_api_key=ui_state.provider_settings.moonshot_api_key or "",
        openrouter_api_key=ui_state.provider_settings.openrouter_api_key or "",
        openai_compatible_url=ui_state.provider_settings.openai_compatible_url,
        openai_compatible_api_key=ui_state.provider_settings.openai_compatible_api_key,
        model_name=ui_state.llm_settings.model_name or "",
        temperature=ui_state.llm_settings.temperature,
        top_p=ui_state.llm_settings.top_p,
        top_k=ui_state.llm_settings.top_k,
        max_tokens=ui_state.llm_settings.max_tokens,
        input_language=input_lang,
        output_language=output_lang,
        reading_direction=ui_state.llm_settings.reading_direction,
        translation_mode=ui_state.llm_settings.translation_mode,
        ocr_method=ui_state.llm_settings.ocr_method,
        enable_web_search=ui_state.general.enable_web_search,
        enable_code_execution=ui_state.general.enable_code_execution,
        image_detail=ui_state.general.image_detail,
        media_resolution=ui_state.general.media_resolution,
        media_resolution_bubbles=ui_state.general.media_resolution_bubbles,
        media_resolution_context=ui_state.general.media_resolution_context,
        send_full_page_context=ui_state.llm_settings.send_full_page_context,
        parallel_inpaint_translation=(
            ui_state.llm_settings.parallel_inpaint_translation
        ),
        whiteout_conjoined_bubbles=ui_state.llm_settings.whiteout_conjoined_bubbles,
        upscale_method=ui_state.llm_settings.upscale_method,
        bubble_min_side_pixels=ui_state.llm_settings.bubble_min_side_pixels,
        context_image_max_side_pixels=ui_state.llm_settings.context_image_max_side_pixels,
        previous_context_image_count=(
            ui_state.batch_previous_context_image_count
            if (
                is_batch
                and ui_state.llm_settings.send_full_page_context
                and ui_state.llm_settings.ocr_method == "LLM"
            )
            else 0
        ),
        previous_context_text_count=(
            ui_state.batch_previous_context_text_count if is_batch else 0
        ),
        previous_context_text_mode=(
            ui_state.batch_previous_context_text_mode if is_batch else "strict"
        ),
        osb_min_side_pixels=ui_state.llm_settings.osb_min_side_pixels,
        special_instructions=ui_state.llm_settings.special_instructions,
        reasoning_effort=ui_state.general.reasoning_effort,
        effort=ui_state.general.effort,
        verbosity=ui_state.general.verbosity,
    )

    tag_style_cfgs = {}
    for tag_name, tag_data in normalize_tag_style_table(ui_state.rendering.tag_style_table).items():
        font_pack = (tag_data.get("font_pack") or "").strip()
        tag_font_path = fonts_base_dir / font_pack if font_pack else None
        tag_style_cfgs[tag_name] = TagStyleConfig(
            font_dir=str(tag_font_path) if tag_font_path else None,
            min_font_size=int(tag_data.get("min_font_size", ui_state.rendering.min_font_size)),
            max_font_size=int(tag_data.get("max_font_size", ui_state.rendering.max_font_size)),
            line_spacing_mult=float(tag_data.get("line_spacing_mult", ui_state.rendering.line_spacing_mult)),
            vertical_line_spacing_mult=float(tag_data.get("vertical_line_spacing_mult", ui_state.rendering.vertical_line_spacing_mult)),
            vertical_max_chars=int(tag_data.get("vertical_max_chars", ui_state.rendering.vertical_max_chars)),
            vertical_min_aspect_ratio=float(tag_data["vertical_min_aspect_ratio"])
            if tag_data.get("vertical_min_aspect_ratio") is not None
            else None,
        )

    rendering_cfg = RenderingConfig(
        font_dir=str(font_dir_path),
        emphasis_font_dir=str(emphasis_font_path) if emphasis_font_path else None,
        max_font_size=ui_state.rendering.max_font_size,
        emphasis_max_font_size=ui_state.rendering.emphasis_max_font_size,
        min_font_size=ui_state.rendering.min_font_size,
        line_spacing_mult=ui_state.rendering.line_spacing_mult,
        emphasis_line_spacing_mult=ui_state.rendering.emphasis_line_spacing_mult,
        vertical_line_spacing_mult=ui_state.rendering.vertical_line_spacing_mult,
        vertical_emphasis_line_spacing_mult=ui_state.rendering.vertical_emphasis_line_spacing_mult,
        vertical_font_size_mult=ui_state.rendering.vertical_font_size_mult,
        vertical_max_chars=ui_state.rendering.vertical_max_chars,
        vertical_emphasis_max_chars=ui_state.rendering.vertical_emphasis_max_chars,
        tag_styles=tag_style_cfgs,
        use_subpixel_rendering=ui_state.rendering.use_subpixel_rendering,
        font_hinting=ui_state.rendering.font_hinting,
        use_ligatures=ui_state.rendering.use_ligatures,
        hyphenate_before_scaling=ui_state.rendering.hyphenate_before_scaling,
        hyphen_penalty=ui_state.rendering.hyphen_penalty,
        hyphenation_min_word_length=(ui_state.rendering.hyphenation_min_word_length),
        badness_exponent=ui_state.rendering.badness_exponent,
        padding_pixels=ui_state.rendering.padding_pixels,
        padding_horizontal_pixels=ui_state.rendering.padding_horizontal_pixels,
        padding_vertical_pixels=ui_state.rendering.padding_vertical_pixels,
        supersampling_factor=ui_state.rendering.supersampling_factor,
        detach_trailing_punctuation=ui_state.rendering.detach_trailing_punctuation,
        auto_vertical_text=ui_state.rendering.auto_vertical_text,
        inside_light_text_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.inside_light_text_color, "#000000"
        ),
        inside_light_outline_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.inside_light_outline_color, "#ffffff"
        ),
        inside_light_outline_width=ui_state.rendering.inside_light_outline_width,
        inside_dark_text_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.inside_dark_text_color, "#ffffff"
        ),
        inside_dark_outline_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.inside_dark_outline_color, "#000000"
        ),
        inside_dark_outline_width=ui_state.rendering.inside_dark_outline_width,
        outside_light_text_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.outside_light_text_color, "#000000"
        ),
        outside_light_outline_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.outside_light_outline_color, "#ffffff"
        ),
        outside_light_outline_width=ui_state.rendering.outside_light_outline_width,
        outside_dark_text_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.outside_dark_text_color, "#ffffff"
        ),
        outside_dark_outline_color_rgb=_hex_color_to_rgb(
            ui_state.rendering.outside_dark_outline_color, "#000000"
        ),
        outside_dark_outline_width=ui_state.rendering.outside_dark_outline_width,
    )

    upscale_mode = ui_state.output.image_upscale_mode
    upscale_factor = ui_state.output.image_upscale_factor

    output_cfg = OutputConfig(
        output_format=ui_state.output.output_format,
        jpeg_quality=ui_state.output.jpeg_quality,
        png_compression=ui_state.output.png_compression,
        embed_editor_package=ui_state.output.embed_editor_package,
        save_plain_image_copy=ui_state.output.save_plain_image_copy,
        upscale_final_image=upscale_mode == "final",
        image_upscale_factor=upscale_factor,
        image_upscale_model=ui_state.output.image_upscale_model,
    )

    # Determine OSB font (use main font if not specified)
    osb_font_pack = (
        ui_state.outside_text.osb_font_dir
        if ui_state.outside_text.osb_font_dir
        else ui_state.font_pack
    )
    osb_font_path = fonts_base_dir / osb_font_pack if osb_font_pack else None
    osb_emphasis_font_pack = (
        ui_state.outside_text.osb_emphasis_font_dir
        if ui_state.outside_text.osb_emphasis_font_dir
        else None
    )
    osb_emphasis_font_path = (
        fonts_base_dir / osb_emphasis_font_pack if osb_emphasis_font_pack else None
    )

    outside_text_cfg = OutsideTextConfig(
        enabled=ui_state.outside_text.enabled,
        processing_mode=ui_state.outside_text.processing_mode,
        enable_page_number_filtering=ui_state.outside_text.enable_page_number_filtering,
        page_filter_margin_threshold=ui_state.outside_text.page_filter_margin_threshold,
        page_filter_min_area_ratio=ui_state.outside_text.page_filter_min_area_ratio,
        seed=ui_state.outside_text.seed,
        huggingface_token=ui_state.outside_text.huggingface_token,
        inpainting_method=ui_state.outside_text.inpainting_method,
        kontext_backend=ui_state.outside_text.kontext_backend,
        flux_cpu_offload=ui_state.outside_text.flux_cpu_offload,
        flux_low_vram=ui_state.outside_text.flux_low_vram,
        flux_num_inference_steps=ui_state.outside_text.flux_num_inference_steps,
        flux_kontext_prompt=ui_state.outside_text.flux_kontext_prompt,
        flux_klein_prompt=ui_state.outside_text.flux_klein_prompt,
        flux_luminance_correction=ui_state.outside_text.flux_luminance_correction,
        flux_upscale_small_crops=ui_state.outside_text.flux_upscale_small_crops,
        flux_group_regions=ui_state.outside_text.flux_group_regions,
        flux_residual_diff_threshold=ui_state.outside_text.flux_residual_diff_threshold,
        osb_residual_render_guard_action=ui_state.outside_text.osb_residual_render_guard_action,
        osb_confidence=ui_state.outside_text.osb_confidence,
        osb_font_dir=str(osb_font_path) if osb_font_path else None,
        osb_emphasis_font_dir=(
            str(osb_emphasis_font_path) if osb_emphasis_font_path else None
        ),
        osb_max_font_size=ui_state.outside_text.osb_max_font_size,
        osb_emphasis_max_font_size=ui_state.outside_text.osb_emphasis_max_font_size,
        osb_min_font_size=ui_state.outside_text.osb_min_font_size,
        osb_use_ligatures=ui_state.outside_text.osb_use_ligatures,
        osb_outline_width=ui_state.outside_text.osb_outline_width,
        osb_padding_pixels=ui_state.outside_text.osb_padding_pixels,
        osb_line_spacing=ui_state.outside_text.osb_line_spacing,
        osb_vertical_line_spacing=ui_state.outside_text.osb_vertical_line_spacing,
        osb_auto_vertical=ui_state.outside_text.osb_auto_vertical,
        osb_vertical_aspect_ratio=ui_state.outside_text.osb_vertical_aspect_ratio,
        osb_vertical_max_chars=ui_state.outside_text.osb_vertical_max_chars,
        osb_tall_box_expand_aspect_ratio=(
            ui_state.outside_text.osb_tall_box_expand_aspect_ratio
        ),
        osb_tall_box_expand_width_percent=(
            ui_state.outside_text.osb_tall_box_expand_width_percent
        ),
        osb_rotation_max_degrees=ui_state.outside_text.osb_rotation_max_degrees,
        osb_use_subpixel_rendering=ui_state.outside_text.osb_use_subpixel_rendering,
        osb_font_hinting=ui_state.outside_text.osb_font_hinting,
        bbox_expansion_percent=ui_state.outside_text.bbox_expansion_percent,
        osb_render_expansion_narrow_multiplier=ui_state.outside_text.osb_render_expansion_narrow_multiplier,
        osb_render_expansion_tiny_multiplier=ui_state.outside_text.osb_render_expansion_tiny_multiplier,
        osb_render_expansion_aspect_ratio_threshold=ui_state.outside_text.osb_render_expansion_aspect_ratio_threshold,
        osb_render_expansion_area_ratio_threshold=ui_state.outside_text.osb_render_expansion_area_ratio_threshold,
        text_box_proximity_ratio=ui_state.outside_text.text_box_proximity_ratio,
    )

    preprocessing_cfg = PreprocessingConfig(
        enabled=upscale_mode == "initial",
        factor=upscale_factor,
        auto_scale=ui_state.general.auto_scale,
    )

    backend_config = MangaTranslatorConfig(
        yolo_model_path=str(yolo_path),
        verbose=ui_state.general.verbose,
        device=target_device,
        detection=detection_cfg,
        cleaning=cleaning_cfg,
        translation=translation_cfg,
        rendering=rendering_cfg,
        output=output_cfg,
        outside_text=outside_text_cfg,
        preprocessing=preprocessing_cfg,
        cleaning_only=ui_state.general.cleaning_only,
        upscaling_only=ui_state.general.upscaling_only,
        test_mode=ui_state.general.test_mode,
        parallel_requests=ui_state.batch_parallel_requests if is_batch else 1,
        retry_failed_once=ui_state.batch_retry_failed_once if is_batch else False,
        editor_character_profile_name=ui_state.character_profile_name,
    )

    return backend_config
