from __future__ import annotations

import py_compile
import re
import shutil
from pathlib import Path

ROOT = Path.cwd()
BACKUP_SUFFIX = ".bak_vertex_json_fixed"


def log(message: str) -> None:
    print(message, flush=True)


def backup(path: Path) -> Path:
    backup_path = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
        log(f"Backup created: {backup_path}")
    else:
        log(f"Backup already exists: {backup_path}")
    return backup_path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def compile_or_restore(path: Path, backup_path: Path) -> bool:
    try:
        py_compile.compile(str(path), doraise=True)
        return True
    except Exception as exc:
        log(f"Syntax check failed for {path}: {exc}")
        shutil.copy2(backup_path, path)
        log(f"Restored from backup: {backup_path}")
        return False


def install_endpoint() -> None:
    endpoint = ROOT / "utils" / "endpoints" / "google_vertex_json_final.py"
    if not endpoint.exists():
        raise SystemExit(
            "Missing utils/endpoints/google_vertex_json_final.py. "
            "Extract the ZIP into the MangaTranslator root folder with folders preserved."
        )
    py_compile.compile(str(endpoint), doraise=True)
    log(f"Endpoint ready: {endpoint}")


def patch_endpoint_import() -> None:
    path = ROOT / "utils" / "endpoints" / "__init__.py"
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    backup_path = backup(path)
    text = read_text(path)

    if "google_vertex_json_final import call_gemini_endpoint" in text:
        log("Endpoint import already patched.")
        return

    new_text = re.sub(
        r"from\s+\.google\s+import\s+call_gemini_endpoint",
        "from .google_vertex_json_final import call_gemini_endpoint",
        text,
        count=1,
    )
    if new_text == text:
        raise SystemExit(
            "Could not find 'from .google import call_gemini_endpoint' in utils/endpoints/__init__.py."
        )

    write_text(path, new_text)
    if not compile_or_restore(path, backup_path):
        raise SystemExit("Failed to patch utils/endpoints/__init__.py.")
    log(f"Patched endpoint import: {path}")


def patch_ui_utils() -> None:
    path = ROOT / "ui" / "utils.py"
    if not path.exists():
        raise SystemExit(f"Missing file: {path}")

    backup_path = backup(path)
    text = read_text(path)

    start_marker = "# VERTEX_JSON_FINAL_VALIDATE_PATCH_BEGIN"
    end_marker = "# VERTEX_JSON_FINAL_VALIDATE_PATCH_END"

    fixed_block = r'''
# VERTEX_JSON_FINAL_VALIDATE_PATCH_BEGIN
# Allows the Google API Key field to accept full Google service-account JSON.
# Important: original signature is validate_api_key(api_key, provider).
if "_MANGATL_ORIGINAL_VALIDATE_API_KEY" not in globals():
    _MANGATL_ORIGINAL_VALIDATE_API_KEY = validate_api_key


def _mangatl_vertex_json_final_is_service_account(value: str) -> bool:
    import json
    from pathlib import Path

    text = (value or "").strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            info = json.loads(text)
        except Exception:
            return False
        return (
            info.get("type") == "service_account"
            and bool(info.get("project_id"))
            and bool(info.get("client_email"))
            and bool(info.get("private_key"))
            and bool(info.get("token_uri"))
        )

    path_text = text.strip('"')
    if path_text.lower().endswith(".json"):
        try:
            p = Path(path_text)
            if not p.exists() or not p.is_file():
                return False
            info = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            info.get("type") == "service_account"
            and bool(info.get("project_id"))
            and bool(info.get("client_email"))
            and bool(info.get("private_key"))
            and bool(info.get("token_uri"))
        )

    return False


def validate_api_key(api_key: str, provider: str):
    if provider == "Google" and _mangatl_vertex_json_final_is_service_account(api_key):
        return True, "Google service-account JSON format looks valid"
    return _MANGATL_ORIGINAL_VALIDATE_API_KEY(api_key, provider)


# VERTEX_JSON_FINAL_VALIDATE_PATCH_END
'''

    if start_marker in text and end_marker in text:
        pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
        text = pattern.sub(fixed_block.strip(), text, count=1)
        log("Replaced existing Vertex JSON validation patch with fixed version.")
    else:
        text = text.rstrip() + "\n\n" + fixed_block.strip() + "\n"
        log("Added Vertex JSON validation patch.")

    write_text(path, text)
    if not compile_or_restore(path, backup_path):
        raise SystemExit("Failed to patch ui/utils.py.")
    log(f"Patched API key validation wrapper: {path}")


def write_helper_files() -> None:
    test_script = '''from pathlib import Path
import importlib.util
import sys

root = Path(__file__).resolve().parent
json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "keys" / "service-account.json"

if not json_path.exists():
    raise SystemExit(f"Missing JSON file: {json_path}")

json_text = json_path.read_text(encoding="utf-8")

endpoint_path = root / "utils" / "endpoints" / "google_vertex_json_final.py"
spec = importlib.util.spec_from_file_location("google_vertex_json_final_test", endpoint_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

print(module.call_gemini_endpoint(
    json_text,
    "gemini-2.5-flash",
    [{"text": "한국어로 OK만 출력해줘"}],
    {"temperature": 0.1, "maxOutputTokens": 20},
    debug=True,
))
'''
    write_text(ROOT / "test_vertex_json_endpoint.py", test_script)

    copy_bat = '''@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "keys\\service-account.json" (
    echo ERROR: keys\\service-account.json not found.
    echo Put service-account.json here:
    echo   %CD%\\keys\\service-account.json
    pause
    exit /b 1
)

set PY=python
if exist "runtime\\python.exe" set PY=runtime\\python.exe

%PY% -c "import json, pathlib, subprocess; p=pathlib.Path('keys/service-account.json'); s=json.dumps(json.loads(p.read_text(encoding='utf-8')), separators=(',',':')); subprocess.run('clip', input=s, text=True); print('Copied service-account.json as one-line JSON. Paste it into the Google API Key field if multiline paste is inconvenient.')"
pause
'''
    write_text(ROOT / "copy_service_account_json_to_clipboard.bat", copy_bat)

    uninstall_bat = '''@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Restoring Vertex JSON fixed patch backups...
echo.

if exist "utils\\endpoints\\__init__.py.bak_vertex_json_fixed" copy /Y "utils\\endpoints\\__init__.py.bak_vertex_json_fixed" "utils\\endpoints\\__init__.py" >nul
if exist "ui\\utils.py.bak_vertex_json_fixed" copy /Y "ui\\utils.py.bak_vertex_json_fixed" "ui\\utils.py" >nul
if exist "utils\\endpoints\\google_vertex_json_final.py" del /F /Q "utils\\endpoints\\google_vertex_json_final.py"

echo Done.
pause
'''
    write_text(ROOT / "uninstall_vertex_json_webui_fixed.bat", uninstall_bat)
    log("Installed helper scripts.")


def main() -> None:
    log("MangaTranslator Vertex JSON Web UI fixed patch")
    log(f"Root: {ROOT}")

    required = [
        ROOT / "utils" / "endpoints" / "__init__.py",
        ROOT / "ui" / "utils.py",
        ROOT / "utils" / "endpoints" / "google_vertex_json_final.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))

    install_endpoint()
    patch_endpoint_import()
    patch_ui_utils()
    write_helper_files()

    log("")
    log("Patch completed successfully.")
    log("Restart MangaTranslator Web UI.")
    log("Use Provider=Google, Model=gemini-2.5-flash, and paste service-account.json into the Google API Key field.")


if __name__ == "__main__":
    main()
