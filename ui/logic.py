import base64
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import uuid4

from PIL import Image

from core.config import MangaTranslatorConfig, RenderingConfig
from core.editor_session import default_edit_session_path
from core.pipeline import (
    FAILED_FILES_TXT_NAME,
    batch_translate_images,
    initialize_failed_files_txt,
    reset_google_rate_limit_session,
    translate_and_render,
)
from core.validation import validate_batch_input_path, validate_core_inputs
from utils.exceptions import (
    CancellationError,
    CleaningError,
    FontError,
    ImageProcessingError,
    RenderingError,
    TranslationError,
    ValidationError,
)
from utils.logging import log_message

if TYPE_CHECKING:
    from ui.cancellation import CancellationManager


class LogicError(Exception):
    pass


_UPLOADED_RELATIVE_PATH_PREFIX = "__MT_RELATIVE_PATH__"


def _decode_uploaded_relative_path(file_name: str) -> Optional[Path]:
    """Decode a browser folder upload path embedded in a safe temporary filename."""
    if not file_name.startswith(_UPLOADED_RELATIVE_PATH_PREFIX):
        return None

    suffix = Path(file_name).suffix
    encoded_end = -len(suffix) if suffix else None
    encoded = file_name[len(_UPLOADED_RELATIVE_PATH_PREFIX) : encoded_end]
    try:
        padding = "=" * (-len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(encoded + padding).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None

    clean_parts = [
        part
        for part in PurePosixPath(decoded.replace("\\", "/")).parts
        if part not in ("", ".", "..", "/")
    ]
    return Path(*clean_parts) if clean_parts else None


def _resolve_output_base_dir(output_base_dir: Union[str, Path, None]) -> Path:
    raw_path = str(output_base_dir or "output").strip() or "output"
    try:
        resolved = Path(os.path.expandvars(raw_path)).expanduser()
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved
        resolved = resolved.resolve()
        if resolved.exists() and not resolved.is_dir():
            raise ValidationError(f"Output path '{resolved}' is not a directory.")
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(
            f"Could not create or access output directory '{raw_path}': {exc}"
        ) from exc


def _safe_archive_name(raw_name: Any, fallback: str) -> str:
    name = str(raw_name or fallback).replace("\\", "/").strip("/")
    clean_parts = []
    for part in PurePosixPath(name).parts:
        if part in ("", ".", ".."):
            continue
        clean_parts.append(part)
    return "/".join(clean_parts) if clean_parts else fallback


def _dedupe_archive_name(archive_name: str, used_names: set[str]) -> str:
    archive_name = _safe_archive_name(archive_name, "file")
    if archive_name not in used_names:
        used_names.add(archive_name)
        return archive_name

    path = PurePosixPath(archive_name)
    stem = path.stem or "file"
    suffix = path.suffix
    parent = "" if str(path.parent) == "." else f"{path.parent}/"
    counter = 2
    while True:
        candidate = f"{parent}{stem}_{counter}{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def _common_parent(paths: List[Path]) -> Path:
    parents = [str(path.parent.resolve()) for path in paths]
    if not parents:
        return Path(".")
    try:
        return Path(os.path.commonpath(parents))
    except ValueError:
        return paths[0].parent


def _looks_like_generated_temp_name(name: str) -> bool:
    compact = "".join(ch for ch in name.lower() if ch not in "-_{}()[]")
    if not compact:
        return True
    if compact in {"tmp", "temp", "gradio"}:
        return True
    if compact.startswith("tmp") and len(compact) >= 8:
        return True
    ascii_alnum = "0123456789abcdefghijklmnopqrstuvwxyz"
    if len(compact) >= 24 and all(ch in ascii_alnum for ch in compact):
        return True
    return False


def _upload_root_label(image_files: List[Path]) -> Optional[str]:
    """Best-effort top folder label for Gradio directory uploads."""
    common_parent = _common_parent(image_files)
    name = common_parent.name
    if not name or _looks_like_generated_temp_name(name):
        return None
    return _safe_archive_name(name, name)


def _batch_source_folder_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("files") or value.get("zip") or ""
    if isinstance(value, (list, tuple)):
        paths = [Path(str(item)) for item in value if item]
        decoded = [_decode_uploaded_relative_path(path.name) for path in paths]
        roots = {
            relative.parts[0]
            for relative in decoded
            if relative is not None and len(relative.parts) > 1
        }
        if len(roots) == 1:
            return next(iter(roots))
        if paths:
            parent_name = _common_parent(paths).name
            if parent_name and not _looks_like_generated_temp_name(parent_name):
                return parent_name
        return ""
    path = Path(str(value or "").strip())
    if path.is_dir():
        return path.name
    if path.is_file() and path.suffix.lower() == ".zip":
        return path.stem
    return ""


def _gradio_upload_base(image_files: List[Path]) -> Optional[Path]:
    """Return the Gradio upload-id folder so browser directory paths stay intact."""
    bases: List[Path] = []
    for img_file in image_files:
        parts = img_file.resolve().parts
        lowered = [part.lower() for part in parts]
        try:
            gradio_index = lowered.index("gradio")
        except ValueError:
            return None
        if gradio_index + 2 >= len(parts):
            return None
        bases.append(Path(*parts[: gradio_index + 2]))

    first_base = bases[0] if bases else None
    if first_base and all(base == first_base for base in bases):
        return first_base
    return None


def _dedupe_relative_path(relative_path: Path, used_names: set[str]) -> Path:
    return Path(_dedupe_archive_name(relative_path.as_posix(), used_names))


def _should_flatten_encoded_upload_paths(relative_paths: List[Optional[Path]]) -> bool:
    decoded = [path for path in relative_paths if path is not None]
    if len(decoded) != len(relative_paths) or not decoded:
        return False
    if all(len(path.parts) == 1 for path in decoded):
        return True
    if all(len(path.parts) == 2 for path in decoded):
        first_parts = {path.parts[0] for path in decoded}
        return len(first_parts) == 1
    return False


def _copy_images_preserving_relative_paths(
    image_files: List[Path],
    destination: Path,
    root_label: Optional[str] = None,
    include_root_folder: bool = False,
    flatten_single_folder_images: bool = False,
) -> Dict[str, str]:
    common_parent = _common_parent(image_files)
    gradio_base = _gradio_upload_base(image_files)
    relative_base = gradio_base or common_parent
    source_paths: Dict[str, str] = {}
    if (
        root_label is None
        and include_root_folder
        and gradio_base is None
        and not flatten_single_folder_images
    ):
        root_label = _upload_root_label(image_files)

    decoded_paths = [
        _decode_uploaded_relative_path(img_file.name) for img_file in image_files
    ]
    flatten_encoded_paths = (
        flatten_single_folder_images
        and _should_flatten_encoded_upload_paths(decoded_paths)
    )
    used_relative_paths: set[str] = set()

    for img_file, decoded_path in zip(image_files, decoded_paths):
        relative_path = decoded_path
        has_encoded_relative_path = relative_path is not None
        if has_encoded_relative_path and flatten_encoded_paths:
            relative_path = Path(relative_path.name)
        elif not has_encoded_relative_path and flatten_single_folder_images:
            relative_path = Path(img_file.name)
        elif not has_encoded_relative_path:
            try:
                relative_path = img_file.resolve().relative_to(relative_base)
            except ValueError:
                relative_path = Path(img_file.name)
        if has_encoded_relative_path or gradio_base is not None:
            source_path = str(relative_path)
        else:
            source_path = str(img_file.resolve())
        if root_label and not has_encoded_relative_path:
            relative_path = Path(root_label) / relative_path
        relative_path = _dedupe_relative_path(relative_path, used_relative_paths)
        dest_path = destination / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_file, dest_path)
        source_paths[str(relative_path)] = source_path

    return source_paths


def _zip_source_path_map(zip_path: Path, extracted_root: Path) -> Dict[str, str]:
    """Map extracted paths back to their archive and original inner path."""
    source_paths: Dict[str, str] = {}
    zip_source = str(zip_path.resolve())
    for extracted_path in extracted_root.rglob("*"):
        if not extracted_path.is_file():
            continue
        relative_path = extracted_path.relative_to(extracted_root)
        source_paths[str(relative_path)] = (
            f"{zip_source}::{relative_path.as_posix()}"
        )
    return source_paths


def _create_zip_from_files(
    file_map: Dict[str, str],
    zip_path: Path,
    archive_base_dir: Optional[Path] = None,
    use_map_keys: bool = False,
) -> Optional[Path]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    added_count = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_ref:
        for key, file_path_str in file_map.items():
            file_path = Path(file_path_str)
            if not file_path.is_file():
                continue

            archive_name = None
            if not use_map_keys and archive_base_dir is not None:
                try:
                    archive_name = str(file_path.relative_to(archive_base_dir))
                except ValueError:
                    archive_name = None
            if not archive_name:
                archive_name = _safe_archive_name(key, file_path.name)

            archive_name = _dedupe_archive_name(
                archive_name.replace("\\", "/"), used_names
            )
            zip_ref.write(file_path, archive_name)
            added_count += 1

    if added_count == 0:
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    return zip_path


def _batch_download_zip_path(timestamp: str, filename: str) -> Path:
    """Return a WebUI download-only ZIP path outside the configured output folder."""
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "editor_sessions"
        / "batch_download_zips"
        / f"{timestamp}_{uuid4().hex[:6]}"
        / filename
    )


def extract_zip_to_temp(
    zip_file_path: Union[str, Path],
) -> tuple[Path, tempfile.TemporaryDirectory]:
    """
    Extracts a ZIP archive to a temporary directory, preserving folder structure.

    Args:
        zip_file_path: Path to the ZIP file to extract.

    Returns:
        Tuple of (Path to extracted directory, TemporaryDirectory object).
        The caller must keep a reference to the TemporaryDirectory object
        to prevent premature cleanup, and call cleanup() when done.

    Raises:
        ValidationError: If the ZIP file is invalid or cannot be extracted.
        FileNotFoundError: If the ZIP file does not exist.
    """
    zip_path = Path(zip_file_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file '{zip_file_path}' does not exist.")

    if not zip_path.is_file():
        raise ValidationError(f"Path '{zip_file_path}' is not a file.")

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir_obj.name)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir_path)

        log_message(
            f"Extracted ZIP archive '{zip_path.name}' to temporary directory",
            verbose=True,
        )

        return temp_dir_path, temp_dir_obj

    except zipfile.BadZipFile:
        temp_dir_obj.cleanup()
        raise ValidationError(f"File '{zip_file_path}' is not a valid ZIP archive.")
    except Exception as e:
        temp_dir_obj.cleanup()
        raise ValidationError(f"Failed to extract ZIP archive: {str(e)}")


def translate_manga_logic(
    image: Union[str, Path, Image.Image],
    config: MangaTranslatorConfig,
    selected_font_pack_name: str,
    models_dir: Path,
    fonts_base_dir: Path,
    output_base_dir: Union[str, Path] = Path("./output"),
    disable_auto_save: bool = False,
    cancellation_manager: Optional["CancellationManager"] = None,
    usage_out: Optional[Dict[str, int]] = None,
) -> tuple[Image.Image, Optional[Path], Optional[Path]]:
    """
    Processes a single manga image. Handles core validation, calls the core pipeline,
    and returns the processed image or raises standard exceptions.

    Args:
        image: Input image (path or PIL object).
        config: The main configuration object.
        selected_font_pack_name: Name of the font pack directory.
        models_dir: Path to the directory containing YOLO models.
        fonts_base_dir: Path to the base directory containing font pack subdirectories.
        output_base_dir: Base directory to save the output image.
        cancellation_manager: Optional manager to handle cancellation.

    Returns:
        A tuple containing:
            - The translated PIL Image object.
            - The Path object where the image was saved.
            - The Path object for the editor session, if one was created.

    Raises:
        FileNotFoundError: If required models or fonts are not found.
        ValidationError: If core configuration validation fails.
        ValueError: For general configuration issues.
        LogicError: For processing failures within this function.
        Exception: For unexpected errors during processing.
    """
    display_name = (
        Path(image).name if isinstance(image, (str, Path)) else "uploaded image"
    )
    try:
        # Create temporary config for validation since we only have the font pack name
        rendering_cfg_for_val = RenderingConfig(
            font_dir=selected_font_pack_name,
            max_font_size=config.rendering.max_font_size,
            min_font_size=config.rendering.min_font_size,
            line_spacing_mult=config.rendering.line_spacing_mult,
            font_hinting=config.rendering.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=config.translation,
            rendering_cfg=rendering_cfg_for_val,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
        config.yolo_model_path = str(yolo_model_path)
        config.rendering.font_dir = str(font_dir_path)

    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise e

    temp_image_path = None
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if isinstance(image, (str, Path)):
            input_path = Path(image)
            original_ext = input_path.suffix.lower()
            image_path_for_processing = str(input_path)
        elif isinstance(image, Image.Image):
            # PIL objects need to be saved to disk for processing
            original_ext = ".png"
            temp_image_path = Path(tempfile.mktemp(suffix=".png"))
            image.save(temp_image_path)
            image_path_for_processing = str(temp_image_path)

        output_format = config.output.output_format
        if output_format == "auto":
            output_ext = (
                original_ext
                if original_ext in [".jpg", ".jpeg", ".png", ".webp"]
                else ".png"
            )
        elif output_format == "png":
            output_ext = ".png"
        elif output_format == "jpeg":
            output_ext = ".jpg"
        else:
            output_ext = ".png"

        if isinstance(image, (str, Path)):
            original_name = input_path.stem
        else:
            # PIL.Image object - no original filename available
            original_name = "MangaTranslator"

        if disable_auto_save:
            repo_root = Path(__file__).resolve().parents[1]
            safe_name = "".join(
                "_" if char in '<>:"/\\|?*\x00' else char
                for char in original_name
            ).strip(" ._")
            safe_name = safe_name or "MangaTranslator"
            session_dir = (
                repo_root
                / "editor_sessions"
                / "unsaved"
                / f"{timestamp}_{uuid4().hex[:6]}"
            )
            session_dir.mkdir(parents=True, exist_ok=True)
            save_path = None
            output_path_for_pipeline = None
            edit_session_path = session_dir / f"{safe_name}_edit_session.json"
        else:
            output_base_dir = _resolve_output_base_dir(output_base_dir)
            save_path = (
                output_base_dir / f"{original_name}_translated_{timestamp}{output_ext}"
            )
            output_path_for_pipeline = save_path
            edit_session_path = default_edit_session_path(save_path)

        if config.translation.provider == "Google":
            reset_google_rate_limit_session()

        translated_image = translate_and_render(
            image_path=image_path_for_processing,
            config=config,
            output_path=output_path_for_pipeline,
            edit_session_path=edit_session_path,
            cancellation_manager=cancellation_manager,
            usage_out=usage_out,
        )

        if not translated_image:
            raise LogicError("Translation process failed to return an image.")

        return (
            translated_image,
            save_path,
            edit_session_path if edit_session_path.exists() else None,
        )

    except (FontError, RenderingError) as e:
        raise LogicError(f"Text rendering failed: {str(e)}") from e
    except CleaningError as e:
        raise LogicError(f"Bubble cleaning failed: {str(e)}") from e
    except TranslationError as e:
        raise LogicError(f"Translation failed: {str(e)}") from e
    except ImageProcessingError as e:
        raise LogicError(f"Image processing failed: {str(e)}") from e
    except CancellationError as e:
        log_message(f"Translation cancelled for {display_name}", verbose=config.verbose)
        raise e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise LogicError(
            f"An unexpected error occurred during translation: {str(e)}"
        ) from e
    finally:
        if temp_image_path and temp_image_path.exists():
            try:
                os.remove(temp_image_path)
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up temporary file: {e_clean}", always_print=True
                )


def process_batch_logic(
    input_dir_or_files: Union[str, List[str], Dict[str, Any]],
    config: MangaTranslatorConfig,
    selected_font_pack_name: str,
    models_dir: Path,
    fonts_base_dir: Path,
    output_base_dir: Union[str, Path] = Path("./output"),
    disable_auto_save: bool = False,
    gradio_progress: Any = None,
    cancellation_manager: Optional["CancellationManager"] = None,
) -> Dict[str, Any]:
    """
    Processes a batch of manga images. Handles core validation, calls the core batch pipeline,
    and returns results or raises standard exceptions.

    Args:
        input_dir_or_files: Can be one of:
            - Path to input directory (str)
            - Path to ZIP archive (str ending in .zip)
            - List of file paths (List[str])
            - Dictionary with both "zip" and "files" keys to process both simultaneously (Dict[str, Any])
        config: The main configuration object.
        selected_font_pack_name: Name of the font pack directory.
        models_dir: Path to the directory containing YOLO models.
        fonts_base_dir: Path to the base directory containing font pack subdirectories.
        output_base_dir: Base directory to save the output images.
        gradio_progress: Optional Gradio Progress object for UI updates.

    Returns:
        A dictionary containing processing results:
        {
            "success_count": int,
            "error_count": int,
            "errors": Dict[str, str],
            "output_path": Path,
            "processing_time": float
        }

    Raises:
        FileNotFoundError: If required models or fonts are not found, or input dir is invalid.
        ValidationError: If core configuration validation fails.
        ValueError: For general configuration issues or invalid inputs.
        LogicError: For processing failures within this function.
        Exception: For unexpected errors during processing.
    """
    start_time = time.time()
    config.editor_source_folder_name = _batch_source_folder_name(input_dir_or_files)

    try:
        # Create temporary config for validation since we only have the font pack name
        rendering_cfg_for_val = RenderingConfig(
            font_dir=selected_font_pack_name,
            max_font_size=config.rendering.max_font_size,
            min_font_size=config.rendering.min_font_size,
            line_spacing_mult=config.rendering.line_spacing_mult,
            font_hinting=config.rendering.font_hinting,
        )
        yolo_model_path, font_dir_path = validate_core_inputs(
            translation_cfg=config.translation,
            rendering_cfg=rendering_cfg_for_val,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            bubble_detector_model=config.detection.bubble_detector_model,
        )
        config.yolo_model_path = str(yolo_model_path)
        config.rendering.font_dir = str(font_dir_path)
    except (FileNotFoundError, ValidationError, ValueError) as e:
        raise e

    configured_output_base_dir = _resolve_output_base_dir(output_base_dir)
    failed_files_txt_path = configured_output_base_dir / FAILED_FILES_TXT_NAME
    initialize_failed_files_txt(failed_files_txt_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if disable_auto_save:
        repo_root = Path(__file__).resolve().parents[1]
        output_base_dir = (
            repo_root
            / "editor_sessions"
            / "unsaved_batch"
            / f"{timestamp}_{uuid4().hex[:6]}"
        )
    else:
        output_base_dir = configured_output_base_dir
    batch_output_path = Path(output_base_dir) / timestamp

    def _batch_progress_callback(value, desc="Processing..."):
        if gradio_progress is not None:
            gradio_progress(value, desc=desc)
        elif config.verbose:
            log_message(f"Progress: {desc} [{value * 100:.1f}%]", verbose=True)

    temp_dir_path_obj = None
    zip_temp_dir_obj = None
    try:
        process_dir = None
        preserve_structure = False
        failed_source_paths: Dict[str, str] = {}

        if (
            isinstance(input_dir_or_files, dict)
            and "zip" in input_dir_or_files
            and "files" in input_dir_or_files
        ):
            zip_path_str = input_dir_or_files["zip"]
            files_list = input_dir_or_files["files"]

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            combined_temp_dir = Path(temp_dir_path_obj.name)

            zip_path = Path(zip_path_str)
            if not zip_path.exists():
                raise FileNotFoundError(f"ZIP file '{zip_path_str}' does not exist.")

            extracted_path, zip_temp_dir_obj = extract_zip_to_temp(zip_path)
            failed_source_paths.update(
                _zip_source_path_map(zip_path, extracted_path)
            )

            for item in extracted_path.iterdir():
                dest = combined_temp_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

            if files_list:
                image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
                image_files_to_copy = []
                skipped_files = []
                for f_path_str in files_list:
                    p = Path(f_path_str)
                    if p.is_file() and p.suffix.lower() in image_extensions:
                        try:
                            with Image.open(p) as img:
                                img.verify()
                            image_files_to_copy.append(p)
                        except Exception as img_err:
                            skipped_files.append(f"{p.name} (Invalid: {img_err})")
                    else:
                        skipped_files.append(f"{p.name} (Not a supported image file)")

                if image_files_to_copy:
                    try:
                        failed_source_paths.update(
                            _copy_images_preserving_relative_paths(
                                image_files_to_copy,
                                combined_temp_dir,
                                include_root_folder=True,
                                flatten_single_folder_images=True,
                            )
                        )
                    except Exception as copy_err:
                        log_message(
                            f"Failed to copy uploaded image files: {copy_err}",
                            always_print=True,
                        )

                if skipped_files:
                    log_message(
                        f"Skipped {len(skipped_files)} invalid files from directory upload",
                        verbose=config.verbose,
                    )

            process_dir = combined_temp_dir
            preserve_structure = True

        elif isinstance(input_dir_or_files, list):
            if not input_dir_or_files:
                raise ValidationError("No files provided for batch processing.")

            temp_dir_path_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_path_obj.name)

            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            image_files_to_copy = []
            skipped_files = []
            for f_path_str in input_dir_or_files:
                p = Path(f_path_str)
                if p.is_file() and p.suffix.lower() in image_extensions:
                    try:
                        with Image.open(p) as img:
                            img.verify()
                        image_files_to_copy.append(p)
                    except Exception as img_err:
                        skipped_files.append(f"{p.name} (Invalid: {img_err})")
                else:
                    skipped_files.append(f"{p.name} (Not a supported image file)")

            if not image_files_to_copy:
                raise ValidationError(
                    f"No valid image files found in the selection. "
                    f"Skipped: {', '.join(skipped_files) if skipped_files else 'None'}"
                )

            if skipped_files:
                log_message(
                    f"Preparing {len(image_files_to_copy)} files (skipped {len(skipped_files)} invalid files)",
                    verbose=config.verbose,
                )
            else:
                log_message(
                    f"Preparing {len(image_files_to_copy)} files for batch processing",
                    verbose=config.verbose,
                )

            try:
                failed_source_paths.update(
                    _copy_images_preserving_relative_paths(
                        image_files_to_copy,
                        temp_dir_path,
                        include_root_folder=True,
                        flatten_single_folder_images=True,
                    )
                )
            except Exception as copy_err:
                log_message(
                    f"Failed to copy uploaded image files: {copy_err}",
                    always_print=True,
                )
            process_dir = temp_dir_path
            preserve_structure = True

        elif isinstance(input_dir_or_files, str):
            input_path = validate_batch_input_path(input_dir_or_files)

            if input_path.is_file() and input_path.suffix.lower() == ".zip":
                extracted_path, zip_temp_dir_obj = extract_zip_to_temp(input_path)
                process_dir = extracted_path
                preserve_structure = True
                failed_source_paths.update(
                    _zip_source_path_map(input_path, extracted_path)
                )
            elif input_path.is_dir():
                process_dir = input_path
                preserve_structure = True

        if not process_dir:
            raise LogicError("Could not determine processing directory.")

        results = batch_translate_images(
            input_dir=process_dir,
            config=config,
            output_dir=batch_output_path,
            progress_callback=_batch_progress_callback,
            preserve_structure=preserve_structure,
            cancellation_manager=cancellation_manager,
            failed_files_txt_path=failed_files_txt_path,
            failed_source_paths=failed_source_paths,
        )

        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["output_path"] = batch_output_path
        results["auto_save_disabled"] = bool(disable_auto_save)
        results["failed_files_txt_path"] = failed_files_txt_path
        results["success_zip_path"] = _create_zip_from_files(
            results.get("successful_download_files") or results.get("successful_files", {}),
            _batch_download_zip_path(timestamp, "successful_translations.zip"),
            archive_base_dir=batch_output_path,
            use_map_keys=False,
        )
        results["failed_zip_path"] = _create_zip_from_files(
            results.get("failed_files", {}),
            batch_output_path / "failed_original_files.zip",
            use_map_keys=True,
        )

        log_message(
            f"Batch completed: {results['success_count']} success, "
            f"{results['error_count']} errors in {processing_time:.2f}s",
            verbose=config.verbose,
        )
        if results["errors"]:
            log_message("Processing errors:", verbose=config.verbose)
            for fname, err in results["errors"].items():
                log_message(f"  {fname}: {err}", verbose=config.verbose)

        return results

    except (FontError, RenderingError) as e:
        raise LogicError(
            f"Text rendering failed during batch processing: {str(e)}"
        ) from e
    except CleaningError as e:
        raise LogicError(
            f"Bubble cleaning failed during batch processing: {str(e)}"
        ) from e
    except TranslationError as e:
        raise LogicError(f"Translation failed during batch processing: {str(e)}") from e
    except ImageProcessingError as e:
        raise LogicError(
            f"Image processing failed during batch processing: {str(e)}"
        ) from e
    except CancellationError as e:
        raise e
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise LogicError(
            f"An unexpected error occurred during batch processing: {str(e)}"
        ) from e

    finally:
        if temp_dir_path_obj:
            try:
                temp_dir_path_obj.cleanup()
                log_message("Cleaned up temp directory", verbose=config.verbose)
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up temp directory: {e_clean}",
                    always_print=True,
                )
        if zip_temp_dir_obj:
            try:
                zip_temp_dir_obj.cleanup()
                log_message(
                    "Cleaned up ZIP extraction temp directory", verbose=config.verbose
                )
            except Exception as e_clean:
                log_message(
                    f"Failed to clean up ZIP extraction temp directory: {e_clean}",
                    always_print=True,
                )
