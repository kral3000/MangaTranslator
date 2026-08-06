@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "keys\service-account.json" (
    echo ERROR: keys\service-account.json not found.
    echo Put service-account.json here:
    echo   %CD%\keys\service-account.json
    pause
    exit /b 1
)

set PY=python
if exist "runtime\python.exe" set PY=runtime\python.exe

%PY% -c "import json, pathlib, subprocess; p=pathlib.Path('keys/service-account.json'); s=json.dumps(json.loads(p.read_text(encoding='utf-8')), separators=(',',':')); subprocess.run('clip', input=s, text=True); print('Copied service-account.json as one-line JSON. Paste it into the Google API Key field if multiline paste is inconvenient.')"
pause
