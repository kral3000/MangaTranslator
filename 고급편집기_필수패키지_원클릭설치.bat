@echo off
setlocal

set "APP_DIR=%~dp0"
if not exist "%APP_DIR%runtime\python.exe" (
    if exist "%APP_DIR%MangaTranslator\runtime\python.exe" (
        set "APP_DIR=%APP_DIR%MangaTranslator\"
    )
)

set "PYTHON=%APP_DIR%runtime\python.exe"
set "BASE_REQUIREMENTS=%APP_DIR%requirements.txt"
set "EDITOR_REQUIREMENTS=%APP_DIR%advanced_editor\hgmzhn_runtime_requirements.txt"

if not exist "%PYTHON%" (
    echo [ERROR] runtime\python.exe was not found.
    echo Keep this BAT inside the MangaTranslator folder.
    pause
    exit /b 1
)

if not exist "%BASE_REQUIREMENTS%" (
    echo [ERROR] requirements.txt was not found.
    pause
    exit /b 1
)

if not exist "%EDITOR_REQUIREMENTS%" (
    echo [ERROR] advanced_editor\hgmzhn_runtime_requirements.txt was not found.
    pause
    exit /b 1
)

cd /d "%APP_DIR%"
echo ============================================================
echo Installing MangaTranslator Advanced Editor dependencies
echo ============================================================
echo Internet access is required. Already installed packages are skipped.
echo.

"%PYTHON%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Preparing pip...
    "%PYTHON%" -m ensurepip --upgrade
    if errorlevel 1 goto install_failed
)

"%PYTHON%" -m pip install --disable-pip-version-check --no-warn-script-location --prefer-binary -r "%BASE_REQUIREMENTS%" -r "%EDITOR_REQUIREMENTS%"
if errorlevel 1 goto install_failed

echo.
echo Verifying Advanced Editor, text-box detection, and OCR modules...
"%PYTHON%" -c "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / 'advanced_editor' / 'hgmzhn_runtime')); import colorama, dotenv, einops, freetype, huggingface_hub, kornia, langcodes, manga_ocr, natsort, networkx, omegaconf, onnxruntime, py3langid, pyclipper, shapely, timm, tqdm, ultralytics; from PyQt6.QtCore import QT_VERSION_STR; from PyQt6.QtWidgets import QApplication; from advanced_editor.backend import detect_and_recognize, run_inpainting, translate_texts; from core.ml.model_manager import ModelType; from manga_translator.detection import DefaultDetector; from manga_translator.ocr import Model48pxOCR; print('Verification passed: Qt ' + QT_VERSION_STR + ', ONNX Runtime ' + onnxruntime.__version__)"
if errorlevel 1 goto verify_failed

echo.
echo ============================================================
echo Installation completed successfully.
echo OCR model files will download automatically on first use.
echo You can now run the fast launcher.
echo ============================================================
pause
exit /b 0

:install_failed
echo.
echo [ERROR] Package installation failed.
echo Check the internet connection, free disk space, and antivirus quarantine.
pause
exit /b 1

:verify_failed
echo.
echo [ERROR] Packages were installed, but verification failed.
echo Check runtime\Lib\site-packages and antivirus quarantine.
pause
exit /b 1
