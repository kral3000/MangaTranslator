@echo off
setlocal EnableExtensions DisableDelayedExpansion

REM MangaTranslator public config installer.
REM Applies both config.json and advanced_editor_settings.json.
REM This file is ASCII-only and has no UTF-8 BOM, so it works in Windows CMD.

set "APP_DIR=%LOCALAPPDATA%\MangaTranslator"
set "CONFIG_DEST=%APP_DIR%\config.json"
set "EDITOR_DEST=%APP_DIR%\advanced_editor_settings.json"

if "%~1"=="" (
    set "CONFIG_SRC=%~dp0config.public.json"
) else (
    set "CONFIG_SRC=%~1"
)

if "%~2"=="" (
    set "EDITOR_SRC=%~dp0advanced_editor_settings.public.json"
) else (
    set "EDITOR_SRC=%~2"
)

echo.
echo [MangaTranslator Config Installer]
echo Config source: "%CONFIG_SRC%"
echo Config target: "%CONFIG_DEST%"
echo Editor source: "%EDITOR_SRC%"
echo Editor target: "%EDITOR_DEST%"
echo.

if not exist "%CONFIG_SRC%" (
    echo [ERROR] Source config file not found.
    echo Put config.public.json in the same folder as this BAT,
    echo or pass a JSON config file as the first argument.
    echo.
    pause
    exit /b 1
)

if not exist "%EDITOR_SRC%" (
    echo [ERROR] Source advanced editor settings file not found.
    echo Put advanced_editor_settings.public.json in the same folder as this BAT,
    echo or pass a JSON settings file as the second argument.
    echo.
    pause
    exit /b 1
)

REM Validate both JSON files before changing existing settings.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Get-Content -Raw -LiteralPath $env:CONFIG_SRC | ConvertFrom-Json | Out-Null; Get-Content -Raw -LiteralPath $env:EDITOR_SRC | ConvertFrom-Json | Out-Null; exit 0 } catch { Write-Host '[ERROR] Invalid JSON file.'; Write-Host $_.Exception.Message; exit 1 }"
if errorlevel 1 (
    echo.
    echo [ERROR] JSON validation failed. Settings were not changed.
    echo.
    pause
    exit /b 1
)

if not exist "%APP_DIR%" (
    mkdir "%APP_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create target folder.
        echo Target folder: "%APP_DIR%"
        echo.
        pause
        exit /b 1
    )
)

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "TS=%%I"

if exist "%CONFIG_DEST%" (
    copy /Y "%CONFIG_DEST%" "%APP_DIR%\config.backup.%TS%.json" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to back up config.json.
        echo.
        pause
        exit /b 1
    )
    echo Config backup created.
)

if exist "%EDITOR_DEST%" (
    copy /Y "%EDITOR_DEST%" "%APP_DIR%\advanced_editor_settings.backup.%TS%.json" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to back up advanced_editor_settings.json.
        echo.
        pause
        exit /b 1
    )
    echo Advanced editor settings backup created.
)

copy /Y "%CONFIG_SRC%" "%CONFIG_DEST%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to apply config.json.
    echo.
    pause
    exit /b 1
)

copy /Y "%EDITOR_SRC%" "%EDITOR_DEST%" >nul
if errorlevel 1 (
    echo [ERROR] Failed to apply advanced_editor_settings.json.
    if exist "%APP_DIR%\config.backup.%TS%.json" copy /Y "%APP_DIR%\config.backup.%TS%.json" "%CONFIG_DEST%" >nul
    echo config.json was restored from its backup when possible.
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Both settings files were applied successfully.
echo Restart MangaTranslator.
echo.
pause
exit /b 0
