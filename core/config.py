from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from core.device import get_best_device
from core.llm_defaults import DEFAULT_LLM_PROVIDER, get_provider_sampling_defaults


DEFAULT_FLUX_KONTEXT_PROMPT = "Remove all text."
DEFAULT_FLUX_KLEIN_PROMPT = (
    "Remove all text. Preserve all character line art, screentones, panel borders, "
    "and background details exactly as they appear. Maintain the original black-and-white "
    "contrast and shading, ensuring character expressions and environmental textures "
    "remain unchanged while leaving the text areas completely blank."
)


@dataclass
class DetectionConfig:
    """Configuration for speech bubble detection."""

    confidence: float = 0.6
    conjoined_confidence: float = 0.35
    panel_confidence: float = 0.25
    seg_model: str = "yolo"  # "sam3", "sam2", or "yolo"
    bubble_detector_model: str = "yolo_1"  # "yolo_1" or "yolo_2"
    conjoined_detection: bool = True
    use_panel_sorting: bool = True
    use_osb_text_verification: bool = True


@dataclass
class CleaningConfig:
    """Configuration for speech bubble cleaning."""

    thresholding_value: int = 200
    use_otsu_threshold: bool = False
    roi_shrink_px: int = 5
    inpaint_colored_bubbles: bool = False
    colored_bubble_inpainting_size: int = 2048
    colored_bubble_inpainting_precision: str = "fp32"
    colored_bubble_mask_kernel_size: int = 3
    colored_bubble_mask_dilation_offset: int = 50


_DEFAULT_TRANSLATION_PROVIDER = DEFAULT_LLM_PROVIDER
_DEFAULT_SAMPLING = get_provider_sampling_defaults(_DEFAULT_TRANSLATION_PROVIDER)


@dataclass
class TranslationConfig:
    """Configuration for text translation."""

    provider: str = _DEFAULT_TRANSLATION_PROVIDER
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    deepseek_api_key: str = ""
    zai_api_key: str = ""
    moonshot_api_key: str = ""
    openrouter_api_key: str = ""
    openai_compatible_url: str = "http://localhost:8080/v1"
    openai_compatible_api_key: Optional[str] = ""
    model_name: str = "gemini-3.1-flash-lite"
    provider_models: dict[str, Optional[str]] = field(default_factory=dict)
    temperature: float = float(_DEFAULT_SAMPLING["temperature"])
    top_p: float = float(_DEFAULT_SAMPLING["top_p"])
    top_k: int = int(_DEFAULT_SAMPLING["top_k"])
    max_tokens: Optional[int] = (
        None  # None = use default logic (16384 for reasoning, 4096 otherwise)
    )
    input_language: str = "Japanese"
    output_language: str = "English"
    reading_direction: str = "rtl"
    translation_mode: str = "one-step"
    reasoning_effort: Optional[str] = (
        None  # Default: Google uses "auto", Anthropic uses "none", others use "medium"
    )
    effort: Optional[str] = (
        None  # Opus 4.5/4.6/4.7, Sonnet 4.6 only: token spending eagerness (xhigh/high/medium/low)
    )
    verbosity: Optional[str] = (
        None  # GPT-5 series only: controls response verbosity (high/medium/low)
    )
    send_full_page_context: bool = True
    parallel_inpaint_translation: bool = False
    whiteout_conjoined_bubbles: bool = True
    upscale_method: str = "model_lite"  # "model", "model_lite", "lanczos", or "none"
    enable_web_search: bool = (
        False  # Enable model's built-in web search for up-to-date information. OpenRouter uses its own web search tool.
    )
    enable_code_execution: bool = (
        False  # Enable Gemini's code execution tool for image zoom/inspection (Gemini 3 Flash only)
    )
    image_detail: str = "auto"  # OpenAI image detail (auto/high/low/original)
    media_resolution: str = (
        "auto"  # Only available via Google provider (auto/high/medium/low)
    )
    media_resolution_bubbles: str = "auto"  # Gemini 3 and xAI models
    media_resolution_context: str = "auto"  # Gemini 3 and xAI models
    bubble_min_side_pixels: int = 128
    context_image_max_side_pixels: int = 1024
    previous_context_image_count: int = 0
    previous_context_text_count: int = 0
    previous_context_text_mode: str = "strict"  # "strict" waits; "available" does not
    osb_min_side_pixels: int = 128
    special_instructions: Optional[str] = None
    ocr_method: str = "LLM"  # "LLM", "manga-ocr", or "paddleocr-vl"


@dataclass
class TagStyleConfig:
    """Rendering overrides for tagged translated text such as <BN>...</BN>."""

    font_dir: Optional[str] = None
    min_font_size: Optional[int] = None
    max_font_size: Optional[int] = None
    line_spacing_mult: Optional[float] = None
    vertical_line_spacing_mult: Optional[float] = None
    vertical_max_chars: Optional[int] = None
    vertical_min_aspect_ratio: Optional[float] = None


@dataclass
class RenderingConfig:
    """Configuration for rendering translated text."""

    font_dir: str = "./fonts"
    emphasis_font_dir: Optional[str] = None  # None = use font_dir bold variant
    max_font_size: int = 16
    emphasis_max_font_size: Optional[int] = None  # None = use max_font_size
    min_font_size: int = 8
    line_spacing_mult: float = 1.0
    emphasis_line_spacing_mult: Optional[float] = None  # None = use line_spacing_mult
    vertical_line_spacing_mult: Optional[float] = None  # None = use active horizontal line spacing
    vertical_emphasis_line_spacing_mult: Optional[float] = None  # None = use vertical_line_spacing_mult
    vertical_font_size_mult: float = 1.0  # Applies only to vertical speech-bubble layout
    vertical_max_chars: int = 4  # Maximum alphanumeric chars before auto vertical is blocked for regular text
    vertical_emphasis_max_chars: Optional[int] = None  # None = use vertical_max_chars
    # How much taller than wide a box must be before vertical layout is even tried.
    # Per-tag because the right answer differs by kind of text: Korean dialogue reads
    # horizontally unless the bubble is a genuine column, while sound effects follow
    # the original artwork and go vertical at much milder ratios.
    # None = fall back to AUTO_VERTICAL_MIN_ASPECT_RATIO.
    vertical_min_aspect_ratio: Optional[float] = None
    use_subpixel_rendering: bool = False
    font_hinting: str = "none"
    use_ligatures: bool = False
    hyphenate_before_scaling: bool = True
    hyphen_penalty: float = 1000.0
    hyphenation_min_word_length: int = 8
    badness_exponent: float = 3.0
    padding_pixels: float = 5.0  # Legacy/default inside-bubble padding
    padding_horizontal_pixels: Optional[float] = None  # None = use padding_pixels
    padding_vertical_pixels: Optional[float] = None  # None = use padding_pixels
    outline_width: float = 0.0
    outline_color_rgb: Optional[tuple[int, int, int]] = None
    inside_light_text_color_rgb: tuple[int, int, int] = (0, 0, 0)
    inside_light_outline_color_rgb: tuple[int, int, int] = (255, 255, 255)
    inside_light_outline_width: float = 0.0
    inside_dark_text_color_rgb: tuple[int, int, int] = (255, 255, 255)
    inside_dark_outline_color_rgb: tuple[int, int, int] = (0, 0, 0)
    inside_dark_outline_width: float = 0.0
    outside_light_text_color_rgb: tuple[int, int, int] = (0, 0, 0)
    outside_light_outline_color_rgb: tuple[int, int, int] = (255, 255, 255)
    outside_light_outline_width: float = 3.0
    outside_dark_text_color_rgb: tuple[int, int, int] = (255, 255, 255)
    outside_dark_outline_color_rgb: tuple[int, int, int] = (0, 0, 0)
    outside_dark_outline_width: float = 3.0
    text_alignment: str = "center"  # "left", "center", or "right"
    supersampling_factor: int = 4
    detach_trailing_punctuation: bool = True
    auto_vertical_text: bool = False
    ignore_auto_vertical_font_gain: bool = False
    compare_forced_vertical_layout: bool = False
    force_vertical_fallback_only: bool = False
    force_vertical_retry_columns: int = 1
    force_vertical_retry_font_size: int = 12
    force_vertical_column_gap_mult: float = 1.0
    tight_fit_shrink_enabled: bool = True
    tight_fit_shrink_min_font_size: int = 18
    tight_fit_shrink_width_ratio: float = 0.88
    tight_fit_shrink_height_ratio: float = 0.86
    tight_fit_shrink_factor: float = 0.85
    tight_fit_shrink_max_retries: int = 1
    tight_fit_shrink_exempt: bool = False
    layout_scale: float = 1.0
    small_text_zero_padding_max_font_size: int = 10
    small_text_zero_padding_max_chars: int = 10
    collapsed_small_bbox_fallback_max_area: int = 8000
    collapsed_small_bbox_fallback_max_side: int = 140
    collapsed_small_bbox_fallback_max_short_side: int = 90
    wrapped_symbol_relax_font_size_threshold: int = 18
    tag_styles: Dict[str, TagStyleConfig] = field(default_factory=dict)


@dataclass
class OutsideTextConfig:
    """Configuration for outside speech bubble text detection and removal."""

    enabled: bool = False
    enable_page_number_filtering: bool = False
    page_filter_margin_threshold: float = 0.1
    page_filter_min_area_ratio: float = 0.05
    seed: int = 1  # -1 = random
    huggingface_token: str = ""  # Required for Flux Kontext model downloads
    inpainting_method: str = (
        "flux_klein_4b"  # flux_klein_9b, flux_klein_4b, flux_kontext, opencv, none
    )
    kontext_backend: str = "sdnq"  # "sdnq" (cross-platform) or "nunchaku" (CUDA-only)
    flux_cpu_offload: bool = True  # Keep most Flux weights in system RAM
    flux_low_vram: bool = False  # Use sequential CPU offload for Klein/Kontext SDNQ
    flux_num_inference_steps: int = 8
    flux_luminance_correction: bool = (
        True  # Match patch luminance to surrounding context
    )
    flux_upscale_small_crops: bool = True
    flux_group_regions: bool = False
    flux_residual_diff_threshold: float = 0.15
    osb_residual_render_guard_action: str = "restore_original"
    osb_confidence: float = 0.6
    osb_font_dir: Optional[str] = None  # None = use main font as fallback
    osb_emphasis_font_dir: Optional[str] = None  # None = use OSB font bold variant
    osb_max_font_size: int = 64
    osb_emphasis_max_font_size: Optional[int] = None  # None = use osb_max_font_size
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
    osb_use_subpixel_rendering: bool = False
    osb_font_hinting: str = "none"
    bbox_expansion_percent: float = 0.1
    osb_render_expansion_narrow_multiplier: float = 1.0
    osb_render_expansion_tiny_multiplier: float = 1.0
    osb_render_expansion_aspect_ratio_threshold: float = 0.4
    osb_render_expansion_area_ratio_threshold: float = 0.005
    text_box_proximity_ratio: float = 0.02  # 2% of image dimension
    flux_guidance_scale: float = 2.5
    flux_kontext_prompt: str = DEFAULT_FLUX_KONTEXT_PROMPT
    flux_klein_prompt: str = DEFAULT_FLUX_KLEIN_PROMPT


@dataclass
class OutputConfig:
    """Configuration for saving output images."""

    jpeg_quality: int = 95
    png_compression: int = 2
    output_format: str = "png"
    embed_editor_package: bool = True
    save_plain_image_copy: bool = False
    upscale_final_image: bool = False
    image_upscale_factor: float = 2.0
    image_upscale_model: str = "model_lite"  # "model" or "model_lite"


@dataclass
class MangaTranslatorConfig:
    """Main configuration for the MangaTranslator pipeline."""

    yolo_model_path: str
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    outside_text: OutsideTextConfig = field(default_factory=OutsideTextConfig)
    preprocessing: "PreprocessingConfig" = field(
        default_factory=lambda: PreprocessingConfig()
    )
    verbose: bool = False
    device: Optional[torch.device] = None
    cleaning_only: bool = False
    upscaling_only: bool = False
    test_mode: bool = False
    processing_scale: float = 1.0
    parallel_requests: int = 1
    retry_failed_once: bool = False

    def __post_init__(self):
        # Load API keys from environment variables if not already set
        import os

        if not self.translation.google_api_key:
            self.translation.google_api_key = os.environ.get(
                "GOOGLE_API_KEY"
            ) or os.environ.get("GEMINI_API_KEY", "")
        if not self.translation.openai_api_key:
            self.translation.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.translation.anthropic_api_key:
            self.translation.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.translation.xai_api_key:
            self.translation.xai_api_key = os.environ.get("XAI_API_KEY", "")
        if not self.translation.deepseek_api_key:
            self.translation.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.translation.zai_api_key:
            self.translation.zai_api_key = os.environ.get("ZAI_API_KEY", "")
        if not self.translation.moonshot_api_key:
            self.translation.moonshot_api_key = os.environ.get("MOONSHOT_API_KEY", "")
        if not self.translation.openrouter_api_key:
            self.translation.openrouter_api_key = os.environ.get(
                "OPENROUTER_API_KEY", ""
            )
        if (
            not self.translation.openai_compatible_api_key
        ):  # Check if it's None or empty string
            self.translation.openai_compatible_api_key = os.environ.get(
                "OPENAI_COMPATIBLE_API_KEY", ""
            )
        if not self.outside_text.huggingface_token:
            self.outside_text.huggingface_token = os.environ.get("HF_TOKEN", "")

        # Autodetect device if not specified
        if self.device is None:
            self.device = get_best_device()


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing before detection/cleaning."""

    enabled: bool = False
    factor: float = 2.0
    auto_scale: bool = True


def calculate_reasoning_budget(total_tokens: int, effort_level: str) -> int:
    """
    Calculate reasoning token budget based on effort level.

    Args:
        total_tokens: Total available tokens (typically max_tokens)
        effort_level: Reasoning effort level ("high", "medium", "low", "minimal", "auto", or "none")

    Returns:
        int: Calculated budget in tokens
        - "high": 80% of total_tokens
        - "medium": 50% of total_tokens
        - "low": 20% of total_tokens
        - "minimal": 10% of total_tokens
        - "auto" or "none": Returns 0 (caller should handle these cases separately)
    """
    if effort_level == "high":
        return int(total_tokens * 0.8)
    elif effort_level == "medium":
        return int(total_tokens * 0.5)
    elif effort_level == "low":
        return int(total_tokens * 0.2)
    elif effort_level == "minimal":
        return int(total_tokens * 0.1)
    else:
        # "auto" or "none" - return 0, caller should handle these cases
        return 0
