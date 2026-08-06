@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo MangaTranslator Vertex JSON Web UI FIXED patch installer
echo ======================================================
echo.
echo Please close MangaTranslator Web UI before running this installer.
echo This fixed version corrects validate_api_key argument order.
echo.

set PY=python
if exist "runtime\python.exe" set PY=runtime\python.exe

echo Using Python: %PY%
echo.

%PY% -m pip install google-auth
if errorlevel 1 (
    echo.
    echo Failed to install google-auth.
    pause
    exit /b 1
)

echo.
echo Applying fixed patch...
%PY% patch_vertex_json_webui_fixed.py
if errorlevel 1 (
    echo.
    echo Patch failed.
    pause
    exit /b 1
)

echo.
echo Patch completed successfully.
echo Restart MangaTranslator Web UI.
pause
