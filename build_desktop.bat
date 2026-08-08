@echo off
setlocal
cd /d "%~dp0"

runtime\python.exe -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name MangaTranslatorKo ^
  --icon "%CD%\assets\alya-kujo-app-icon-v2.ico" ^
  --distpath "." ^
  --workpath "build\desktop" ^
  --specpath "build\desktop" ^
  desktop_app.py

if errorlevel 1 exit /b %errorlevel%
echo.
echo Built: %CD%\MangaTranslatorKo.exe
endlocal
