from __future__ import annotations

import os
import re
from pathlib import Path


RUNTIME_LOG_PATH_ENV = "MANGATL_RUNTIME_LOG_PATH"
MAX_DISPLAY_BYTES = 1_000_000
MAX_DISPLAY_LINES = 2_000

_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"authorization|password|client[_ -]?secret|private[_ -]?key)\b"
    r"\s*[\"']?\s*[:=]\s*[\"']?)([^\s,;\"']+)"
)


def redact_sensitive(text: str) -> str:
    """Remove common credentials before logs are written, shown, or copied."""
    redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", str(text))
    return _ASSIGNMENT_PATTERN.sub(r"\1[REDACTED]", redacted)


def configured_log_path() -> Path | None:
    value = os.environ.get(RUNTIME_LOG_PATH_ENV, "").strip()
    return Path(value).expanduser().resolve() if value else None


def log_directory() -> Path:
    configured = configured_log_path()
    if configured:
        return configured.parent
    return Path(__file__).resolve().parent / "output" / "logs"


def read_log_tail(
    max_bytes: int = MAX_DISPLAY_BYTES,
    max_lines: int = MAX_DISPLAY_LINES,
) -> str:
    path = configured_log_path()
    if not path:
        return "데스크톱 앱 실행 로그가 없습니다. 콘솔 버전에서는 콘솔 창을 확인하세요."
    if not path.is_file():
        return f"로그 파일을 준비하는 중입니다.\n{path}"

    try:
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
        text = data.decode("utf-8", errors="replace")
        if size > max_bytes:
            text = text.split("\n", 1)[-1]
        lines = text.splitlines()[-max_lines:]
        return redact_sensitive("\n".join(lines))
    except OSError as exc:
        return f"로그 파일을 읽을 수 없습니다: {exc}"


def open_log_directory() -> Path:
    directory = log_directory()
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(directory)  # type: ignore[attr-defined]
    return directory
