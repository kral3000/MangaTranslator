@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Restoring Vertex JSON fixed patch backups...
echo.

if exist "utils\endpoints\__init__.py.bak_vertex_json_fixed" copy /Y "utils\endpoints\__init__.py.bak_vertex_json_fixed" "utils\endpoints\__init__.py" >nul
if exist "ui\utils.py.bak_vertex_json_fixed" copy /Y "ui\utils.py.bak_vertex_json_fixed" "ui\utils.py" >nul
if exist "utils\endpoints\google_vertex_json_final.py" del /F /Q "utils\endpoints\google_vertex_json_final.py"

echo Done.
pause
