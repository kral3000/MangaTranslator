@echo off
setlocal
cd /d "%~dp0"

if not exist "runtime\pythonw.exe" (
    echo [ERROR] runtime\pythonw.exe was not found.
    echo Keep this BAT inside the MangaTranslatorKo folder.
    pause
    exit /b 1
)

start "" "runtime\pythonw.exe" -m advanced_editor.ui_preview
