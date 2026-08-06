import argparse
import os
import socket
import sys
import tempfile
from pathlib import Path
from threading import Thread


def custom_except_hook(gr, exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, gr.Error):
        print(f"Gradio-handled Error: {exc_value}")
    else:
        import traceback

        print("--- Uncaught Exception ---")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("--------------------------")


def _get_pytorch_version_tuple(version_str):
    """Convert version string like '2.9.0' to tuple (2, 9, 0) for comparison."""
    parts = version_str.split("+")[0].split("-")[0].split(".")
    return tuple(int(part) for part in parts[:3])


def _find_available_port(preferred_port, max_attempts=50):
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return preferred_port


def _build_gradio_allowed_paths() -> list[str]:
    """Allow Gradio to serve result files saved outside the repo output folder."""
    paths: list[Path] = [
        Path(__file__).resolve().parent,
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path.home().resolve(),
    ]

    if os.name == "nt":
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            drive = Path(f"{letter}:\\")
            if drive.exists():
                paths.append(drive)
    else:
        paths.append(Path("/"))

    allowed: list[str] = []
    seen: set[str] = set()
    for path in paths:
        try:
            resolved = str(path.resolve())
        except OSError:
            continue
        key = resolved.lower() if os.name == "nt" else resolved
        if key not in seen:
            seen.add(key)
            allowed.append(resolved)
    return allowed


def main():
    parser = argparse.ArgumentParser(description="MangaTranslator")
    parser.add_argument(
        "--models",
        type=str,
        default="./models",
        help="Directory containing YOLO model files",
    )
    parser.add_argument(
        "--fonts",
        type=str,
        default="./fonts",
        help="Base directory containing font pack subdirectories",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open in the default web browser",
    )
    parser.add_argument(
        "--port", type=int, default=7676, help="Port number for the web UI"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage even if CUDA is available"
    )
    args = parser.parse_args()

    import gradio as gr
    import torch

    from core._version import __version__
    from core.device import get_best_device
    from ui import layout
    from utils.update_checker import check_for_update

    sys.excepthook = lambda exc_type, exc_value, exc_traceback: custom_except_hook(
        gr, exc_type, exc_value, exc_traceback
    )

    # PyTorch 2.9.0+ uses PYTORCH_ALLOC_CONF, older versions use PYTORCH_CUDA_ALLOC_CONF
    _pytorch_version = _get_pytorch_version_tuple(torch.__version__)
    if _pytorch_version >= (2, 9, 0):
        # Helps prevent fragmentation OOM errors on some GPUs
        os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:512"
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    MODELS_DIR = Path(args.models)
    FONTS_BASE_DIR = Path(args.fonts)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FONTS_BASE_DIR, exist_ok=True)

    target_device = torch.device("cpu") if args.cpu else get_best_device()

    device_info_str = "CPU"
    if target_device.type == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            device_info_str = f"CUDA ({gpu_name}, ID: 0)"
        except Exception:
            device_info_str = "CUDA (Unknown GPU Name)"
    elif target_device.type == "xpu":
        try:
            gpu_name = torch.xpu.get_device_name(0)
            device_info_str = f"XPU ({gpu_name}, ID: 0)"
        except Exception:
            device_info_str = "XPU (Unknown GPU Name)"
    elif target_device.type == "mps":
        device_info_str = "MPS (Apple Silicon)"
    print(f"Using device: {device_info_str.upper()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MangaTranslator version: v{__version__}")

    def _update_notice():
        available, latest = check_for_update(
            __version__, repo="meangrinch/MangaTranslator", timeout=3.0
        )
        if available and latest:
            print(f"UPDATE AVAILABLE: v{latest.lstrip('v')}")

    Thread(target=_update_notice, daemon=True).start()

    app = layout.create_layout(
        models_dir=MODELS_DIR,
        fonts_base_dir=FONTS_BASE_DIR,
        target_device=target_device,
    )

    app.queue()
    launch_port = _find_available_port(args.port)
    if launch_port != args.port:
        print(f"Port {args.port} is already in use. Launching on port {launch_port} instead.")
    app.launch(
        inbrowser=args.open_browser,
        server_port=launch_port,
        show_error=True,
        allowed_paths=_build_gradio_allowed_paths(),
    )


if __name__ == "__main__":
    main()
