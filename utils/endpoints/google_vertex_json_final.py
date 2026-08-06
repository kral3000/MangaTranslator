from collections import deque
import json
import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.exceptions import CancellationError, TranslationError, ValidationError
from utils.logging import log_message
from utils.token_usage import format_usage, parse_google_usage


# ---- Google request throttling helpers ----

_GOOGLE_RATE_LIMIT_LOCK = threading.Lock()
_GOOGLE_RATE_LIMIT_TIMESTAMPS = deque()
_GOOGLE_RATE_LIMIT_LAST_REQUEST_AT = 0.0
_GOOGLE_RATE_LIMIT_BURST_COUNT = 0
_GOOGLE_429_WINDOW_STARTED_AT = 0.0
_GOOGLE_429_COUNT = 0
_GOOGLE_429_PAUSE_UNTIL = 0.0
_GOOGLE_429_STOP_MESSAGE = ""
_GOOGLE_429_WINDOW_SECONDS = 3600.0


def _google_rate_config_path() -> Path:
    return (
        Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")))
        / "MangaTranslator"
        / "config.json"
    )


def _bool_from_any(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _float_from_any(value, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(value)
    except Exception:
        value = float(default)
    return min(max(value, minimum), maximum)


def _int_from_any(value, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(float(value))
    except Exception:
        value = int(default)
    return min(max(value, minimum), maximum)


def _load_google_rate_limit_settings() -> Dict[str, Any]:
    """Load Google/Vertex throttling settings from config.json and env overrides."""
    settings = {
        "google_rate_limit_enabled": False,
        "google_request_interval_seconds": 0.0,
        "google_requests_per_minute": 0,
        "google_burst_requests": 0,
        "google_burst_cooldown_seconds": 60.0,
        "google_429_cooldown_seconds": 60.0,
        "google_max_retries": 3,
        "google_429_hourly_limit": 0,
        "google_429_hourly_action": "stop",
        "google_429_pause_minutes": 5.0,
    }

    config_path = _google_rate_config_path()
    if config_path.exists():
        try:
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for key in settings:
                    if key in saved:
                        settings[key] = saved[key]
        except Exception:
            pass

    env_map = {
        "google_rate_limit_enabled": "MANGATL_GOOGLE_RATE_LIMIT_ENABLED",
        "google_request_interval_seconds": "MANGATL_GOOGLE_REQUEST_INTERVAL_SECONDS",
        "google_requests_per_minute": "MANGATL_GOOGLE_REQUESTS_PER_MINUTE",
        "google_burst_requests": "MANGATL_GOOGLE_BURST_REQUESTS",
        "google_burst_cooldown_seconds": "MANGATL_GOOGLE_BURST_COOLDOWN_SECONDS",
        "google_429_cooldown_seconds": "MANGATL_GOOGLE_429_COOLDOWN_SECONDS",
        "google_max_retries": "MANGATL_GOOGLE_MAX_RETRIES",
        "google_429_hourly_limit": "MANGATL_GOOGLE_429_HOURLY_LIMIT",
        "google_429_hourly_action": "MANGATL_GOOGLE_429_HOURLY_ACTION",
        "google_429_pause_minutes": "MANGATL_GOOGLE_429_PAUSE_MINUTES",
    }
    for key, env_name in env_map.items():
        if os.environ.get(env_name) not in (None, ""):
            settings[key] = os.environ.get(env_name)

    settings["google_rate_limit_enabled"] = _bool_from_any(
        settings.get("google_rate_limit_enabled"), False
    )
    settings["google_request_interval_seconds"] = _float_from_any(
        settings.get("google_request_interval_seconds"), 0.0, 0.0, 3600.0
    )
    settings["google_requests_per_minute"] = _int_from_any(
        settings.get("google_requests_per_minute"), 0, 0, 100000
    )
    settings["google_burst_requests"] = _int_from_any(
        settings.get("google_burst_requests"), 0, 0, 100000
    )
    settings["google_burst_cooldown_seconds"] = _float_from_any(
        settings.get("google_burst_cooldown_seconds"), 60.0, 0.0, 86400.0
    )
    settings["google_429_cooldown_seconds"] = _float_from_any(
        settings.get("google_429_cooldown_seconds"), 60.0, 0.0, 86400.0
    )
    settings["google_max_retries"] = _int_from_any(
        settings.get("google_max_retries"), 3, 0, 50
    )
    settings["google_429_hourly_limit"] = _int_from_any(
        settings.get("google_429_hourly_limit"), 0, 0, 100000
    )
    action = str(settings.get("google_429_hourly_action") or "stop").strip().lower()
    settings["google_429_hourly_action"] = action if action in {"stop", "pause"} else "stop"
    settings["google_429_pause_minutes"] = _float_from_any(
        settings.get("google_429_pause_minutes"), 5.0, 0.0, 10080.0
    )
    return settings


def reset_google_rate_limit_session() -> None:
    """Start a fresh Google/Vertex 429 tracking window for a translation run."""
    global _GOOGLE_429_WINDOW_STARTED_AT
    global _GOOGLE_429_COUNT
    global _GOOGLE_429_PAUSE_UNTIL
    global _GOOGLE_429_STOP_MESSAGE

    with _GOOGLE_RATE_LIMIT_LOCK:
        _GOOGLE_429_WINDOW_STARTED_AT = time.monotonic()
        _GOOGLE_429_COUNT = 0
        _GOOGLE_429_PAUSE_UNTIL = 0.0
        _GOOGLE_429_STOP_MESSAGE = ""


def _check_google_429_guard_locked(now: float) -> None:
    if _GOOGLE_429_STOP_MESSAGE:
        raise CancellationError(_GOOGLE_429_STOP_MESSAGE)

    if _GOOGLE_429_PAUSE_UNTIL > now:
        wait_seconds = _GOOGLE_429_PAUSE_UNTIL - now
        log_message(
            f"Google / Vertex 429 guard: resting {wait_seconds:.1f}s before next request",
            always_print=True,
        )
        _GOOGLE_RATE_LIMIT_LOCK.release()
        try:
            time.sleep(wait_seconds)
        finally:
            _GOOGLE_RATE_LIMIT_LOCK.acquire()


def _record_google_429_response(
    settings: Dict[str, Any],
    request_name: str,
) -> float:
    """Record a Google/Vertex 429 response and return extra pause seconds if needed."""
    global _GOOGLE_429_WINDOW_STARTED_AT
    global _GOOGLE_429_COUNT
    global _GOOGLE_429_PAUSE_UNTIL
    global _GOOGLE_429_STOP_MESSAGE

    if not settings.get("google_rate_limit_enabled", False):
        return 0.0

    limit = int(settings.get("google_429_hourly_limit", 0) or 0)
    if limit <= 0:
        return 0.0

    now = time.monotonic()
    with _GOOGLE_RATE_LIMIT_LOCK:
        if (
            _GOOGLE_429_WINDOW_STARTED_AT <= 0
            or now - _GOOGLE_429_WINDOW_STARTED_AT >= _GOOGLE_429_WINDOW_SECONDS
        ):
            _GOOGLE_429_WINDOW_STARTED_AT = now
            _GOOGLE_429_COUNT = 0

        _GOOGLE_429_COUNT += 1
        current_count = _GOOGLE_429_COUNT
        log_message(
            f"{request_name} 429 count: {current_count}/{limit} in current 1-hour window",
            always_print=True,
        )

        if current_count < limit:
            return 0.0

        action = str(settings.get("google_429_hourly_action") or "stop").lower()
        if action == "pause":
            pause_seconds = float(settings.get("google_429_pause_minutes", 0.0) or 0.0) * 60.0
            _GOOGLE_429_COUNT = 0
            _GOOGLE_429_WINDOW_STARTED_AT = now
            _GOOGLE_429_PAUSE_UNTIL = max(_GOOGLE_429_PAUSE_UNTIL, now + pause_seconds)
            log_message(
                f"{request_name} 429 limit reached ({limit}/hour); resting for "
                f"{pause_seconds / 60.0:.1f} minute(s).",
                always_print=True,
            )
            return pause_seconds

        _GOOGLE_429_STOP_MESSAGE = (
            f"{request_name} 429 limit reached: {current_count} error(s) within "
            "the current 1-hour translation window. Translation stopped."
        )
        raise CancellationError(_GOOGLE_429_STOP_MESSAGE)


def _apply_google_request_rate_limit(
    settings: Dict[str, Any],
    request_name: str,
    debug: bool = False,
) -> None:
    """Serialize and throttle outbound Google/Vertex generateContent requests."""
    global _GOOGLE_RATE_LIMIT_LAST_REQUEST_AT
    global _GOOGLE_RATE_LIMIT_BURST_COUNT

    if not settings.get("google_rate_limit_enabled", False):
        return

    interval_seconds = float(settings.get("google_request_interval_seconds", 0.0) or 0.0)
    requests_per_minute = int(settings.get("google_requests_per_minute", 0) or 0)
    burst_requests = int(settings.get("google_burst_requests", 0) or 0)
    burst_cooldown = float(settings.get("google_burst_cooldown_seconds", 0.0) or 0.0)

    with _GOOGLE_RATE_LIMIT_LOCK:
        now = time.monotonic()
        _check_google_429_guard_locked(now)
        now = time.monotonic()
        wait_seconds = 0.0

        if interval_seconds > 0 and _GOOGLE_RATE_LIMIT_LAST_REQUEST_AT > 0:
            wait_seconds = max(
                wait_seconds,
                (_GOOGLE_RATE_LIMIT_LAST_REQUEST_AT + interval_seconds) - now,
            )

        if requests_per_minute > 0:
            while (
                _GOOGLE_RATE_LIMIT_TIMESTAMPS
                and now - _GOOGLE_RATE_LIMIT_TIMESTAMPS[0] >= 60.0
            ):
                _GOOGLE_RATE_LIMIT_TIMESTAMPS.popleft()

            if len(_GOOGLE_RATE_LIMIT_TIMESTAMPS) >= requests_per_minute:
                oldest = _GOOGLE_RATE_LIMIT_TIMESTAMPS[0]
                wait_seconds = max(wait_seconds, 60.0 - (now - oldest))

        if burst_requests > 0 and _GOOGLE_RATE_LIMIT_BURST_COUNT >= burst_requests:
            wait_seconds = max(wait_seconds, burst_cooldown)
            _GOOGLE_RATE_LIMIT_BURST_COUNT = 0

        if wait_seconds > 0:
            log_message(
                f"{request_name} throttling: waiting {wait_seconds:.1f}s",
                verbose=debug,
            )
            time.sleep(wait_seconds)
            now = time.monotonic()
            if requests_per_minute > 0:
                while (
                    _GOOGLE_RATE_LIMIT_TIMESTAMPS
                    and now - _GOOGLE_RATE_LIMIT_TIMESTAMPS[0] >= 60.0
                ):
                    _GOOGLE_RATE_LIMIT_TIMESTAMPS.popleft()

        _GOOGLE_RATE_LIMIT_LAST_REQUEST_AT = now
        _GOOGLE_RATE_LIMIT_TIMESTAMPS.append(now)
        _GOOGLE_RATE_LIMIT_BURST_COUNT += 1


def _google_retry_delay(current_delay: float, settings: Dict[str, Any]) -> float:
    if not settings.get("google_rate_limit_enabled", False):
        return current_delay
    cooldown = float(settings.get("google_429_cooldown_seconds", 0.0) or 0.0)
    return max(current_delay, cooldown)



try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2 import service_account
except Exception:
    GoogleAuthRequest = None
    service_account = None


def _json_from_text_or_path(value: str) -> Optional[Dict[str, Any]]:
    text = (value or "").strip()

    # Full JSON pasted into the Google API key field.
    if text.startswith("{") and text.endswith("}"):
        try:
            info = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError("Invalid Google service-account JSON.") from exc

        if info.get("type") == "service_account":
            return info
        return None

    # Optional convenience: path to a JSON file.
    possible_path = text.strip('"')
    if possible_path.lower().endswith(".json"):
        p = Path(possible_path)
        if p.exists() and p.is_file():
            try:
                info = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValidationError(f"Could not read service-account JSON: {p}") from exc

            if info.get("type") == "service_account":
                return info

    # Optional env var containing full JSON.
    env_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if env_json.startswith("{") and env_json.endswith("}"):
        try:
            info = json.loads(env_json)
        except json.JSONDecodeError as exc:
            raise ValidationError("Invalid GOOGLE_SERVICE_ACCOUNT_JSON.") from exc

        if info.get("type") == "service_account":
            return info

    return None


def _validate_service_account_info(info: Dict[str, Any]) -> None:
    required = ["project_id", "client_email", "private_key", "token_uri"]
    missing = [key for key in required if not info.get(key)]
    if missing:
        raise ValidationError(
            "Google service-account JSON is missing required fields: "
            + ", ".join(missing)
        )


def _use_vertex_ai(api_key: str) -> bool:
    if _json_from_text_or_path(api_key):
        return True
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _get_vertex_credentials_and_project(api_key: str) -> Tuple[Any, str]:
    if service_account is None or GoogleAuthRequest is None:
        raise ValidationError(
            "google-auth is not installed. Run install_vertex_json_webui_final.bat again."
        )

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]

    info = _json_from_text_or_path(api_key)
    if info:
        _validate_service_account_info(info)
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=scopes,
        )
        return credentials, info["project_id"]

    key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip().strip('"')
    if key_path:
        p = Path(key_path)
        if not p.exists():
            raise ValidationError(
                f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {p}"
            )
        info = json.loads(p.read_text(encoding="utf-8"))
        _validate_service_account_info(info)
        credentials = service_account.Credentials.from_service_account_file(
            str(p),
            scopes=scopes,
        )
        return credentials, info["project_id"]

    raise ValidationError(
        "Vertex AI mode requires service-account JSON pasted into the Google field, "
        "a path to service-account.json, GOOGLE_SERVICE_ACCOUNT_JSON, or "
        "GOOGLE_APPLICATION_CREDENTIALS."
    )


def _get_vertex_access_token_and_project(api_key: str) -> Tuple[str, str]:
    credentials, project_from_credentials = _get_vertex_credentials_and_project(api_key)
    credentials.refresh(GoogleAuthRequest())

    project = (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or project_from_credentials
        or os.environ.get("GOOGLE_PROJECT_ID")
    )
    if not project:
        raise ValidationError(
            "Could not determine Google Cloud project. "
            "Set GOOGLE_CLOUD_PROJECT or include project_id in service-account.json."
        )

    return credentials.token, project


def _vertex_base_url(location: str) -> str:
    if location == "global":
        return "https://aiplatform.googleapis.com"
    return f"https://{location}-aiplatform.googleapis.com"


def _convert_to_vertex_json(obj):
    key_map = {
        "inline_data": "inlineData",
        "mime_type": "mimeType",
        "media_resolution": "mediaResolution",
        "code_execution": "codeExecution",
        "function_call": "functionCall",
        "function_response": "functionResponse",
        "executable_code": "executableCode",
        "code_execution_result": "codeExecutionResult",
    }

    if isinstance(obj, list):
        return [_convert_to_vertex_json(item) for item in obj]

    if isinstance(obj, dict):
        return {
            key_map.get(key, key): _convert_to_vertex_json(value)
            for key, value in obj.items()
        }

    return obj


def call_gemini_endpoint(
    api_key: str,
    model_name: str,
    parts: List[Dict[str, Any]],
    generation_config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    debug: bool = False,
    timeout: int = 150,
    max_retries: int = 3,
    base_delay: float = 1.0,
    enable_web_search: bool = False,
    enable_code_execution: bool = False,
    usage_output: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    use_vertex = _use_vertex_ai(api_key)

    if not api_key and not use_vertex:
        raise ValidationError("API key is required for Google endpoint")

    headers = {"Content-Type": "application/json"}

    if use_vertex:
        access_token, project = _get_vertex_access_token_and_project(api_key)
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        base_url = _vertex_base_url(location)
        url = (
            f"{base_url}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model_name}:generateContent"
        )
        headers["Authorization"] = f"Bearer {access_token}"
        request_name = "Google Vertex AI"
    else:
        is_gemini_3 = "gemini-3" in (model_name or "").lower()
        api_version = "v1alpha" if is_gemini_3 else "v1beta"
        url = (
            f"https://generativelanguage.googleapis.com/{api_version}"
            f"/models/{model_name}:generateContent?key={api_key}"
        )
        request_name = "Google API"

    safety_settings_config = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    # Vertex AI requires role=user/model. Developer API also accepts role=user.
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": generation_config,
        "safetySettings": safety_settings_config,
    }

    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    tools = []
    if enable_web_search:
        tools.append({"googleSearch": {}})
    if enable_code_execution:
        tools.append({"code_execution": {}})
    if tools:
        payload["tools"] = tools

    if use_vertex:
        payload = _convert_to_vertex_json(payload)

    rate_limit_settings = _load_google_rate_limit_settings()
    max_retries = int(rate_limit_settings.get("google_max_retries", max_retries))

    for attempt in range(max_retries + 1):
        current_delay = min(base_delay * (2**attempt), 16.0)

        try:
            log_message(
                f"{request_name} request (attempt {attempt + 1}/{max_retries + 1})",
                verbose=debug,
            )

            _apply_google_request_rate_limit(
                rate_limit_settings,
                request_name=request_name,
                debug=debug,
            )

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

            log_message("Processing Google response", verbose=debug)

            try:
                result = response.json()

                call_usage = parse_google_usage(result)
                if call_usage:
                    if usage_output is not None:
                        usage_output.update(call_usage)
                    log_message(
                        f"Google token usage: {format_usage(call_usage)}",
                        verbose=debug,
                    )

                prompt_feedback = result.get("promptFeedback")
                if prompt_feedback and prompt_feedback.get("blockReason"):
                    return None

                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    content_parts = candidate.get("content", {}).get("parts", [{}])

                    if content_parts:
                        for part in content_parts:
                            if "text" in part and not part.get("thought", False):
                                return part.get("text", "").strip()

                        if "text" in content_parts[0]:
                            return content_parts[0].get("text", "").strip()

                    finish_reason = candidate.get("finishReason") or "unknown"
                    log_message(
                        f"No text content in Google response. Finish reason: {finish_reason}",
                        always_print=True,
                    )
                    log_message(
                        f"Full response: {json.dumps(result, indent=2)}",
                        verbose=debug,
                    )
                    return ""

                block_reason = (
                    prompt_feedback.get("blockReason")
                    if prompt_feedback
                    else "unknown"
                )
                log_message(
                    f"No candidates in Google response. Block reason: {block_reason}",
                    always_print=True,
                )
                log_message(
                    f"Full response: {json.dumps(result, indent=2)}",
                    verbose=debug,
                )
                return None

            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                raise TranslationError(
                    f"Error processing successful Google API response: {str(exc)}"
                ) from exc

        except requests.exceptions.HTTPError as exc:
            status_code = (
                exc.response.status_code if exc.response is not None else "unknown"
            )
            error_text = (
                exc.response.text[:1200]
                if exc.response is not None and exc.response.text
                else str(exc)
            )

            extra_pause_seconds = 0.0
            if status_code == 429:
                extra_pause_seconds = _record_google_429_response(
                    rate_limit_settings,
                    request_name,
                )

            if status_code == 429 and attempt < max_retries:
                retry_delay = _google_retry_delay(current_delay, rate_limit_settings)
                retry_delay = max(retry_delay, extra_pause_seconds)
                log_message(
                    f"Rate limited, retrying in {retry_delay:.1f}s",
                    verbose=debug,
                )
                time.sleep(retry_delay)
                continue

            error_reason = f"Status {status_code}: {error_text}"
            if status_code == 429 and attempt == max_retries:
                error_reason = (
                    f"Rate limited after {max_retries + 1} attempts: {error_text}"
                )

            raise TranslationError(f"Google API HTTP Error: {error_reason}") from exc

        except requests.exceptions.RequestException as exc:
            if attempt < max_retries:
                log_message(
                    f"Connection error, retrying in {current_delay:.1f}s: {str(exc)}",
                    verbose=debug,
                )
                time.sleep(current_delay)
                continue

            raise TranslationError(
                f"Google API Connection Error after retries: {str(exc)}"
            ) from exc

    raise TranslationError(
        f"Failed to get response from Google API after {max_retries + 1} attempts."
    )
