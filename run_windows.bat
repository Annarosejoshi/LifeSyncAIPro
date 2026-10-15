@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    call setup_windows.bat
    if errorlevel 1 exit /b 1
)
"venv\Scripts\python.exe" --version >nul 2>nul
if errorlevel 1 (
    call setup_windows.bat
    if errorlevel 1 exit /b 1
)

"venv\Scripts\python.exe" -c "import django, joblib, pandas, sklearn, xgboost" >nul 2>nul
if errorlevel 1 (
    echo Installing missing project dependencies into the project environment...
    call setup_windows.bat
    if errorlevel 1 exit /b 1
)

"venv\Scripts\python.exe" manage.py check
if errorlevel 1 exit /b 1
"venv\Scripts\python.exe" manage.py check_models
if errorlevel 1 exit /b 1

if /I "%~1"=="--check" exit /b 0

"venv\Scripts\python.exe" manage.py migrate --noinput
if errorlevel 1 exit /b 1
echo.
echo Open http://127.0.0.1:8000/ in your browser.
echo Keep this terminal open while using LifeSync AI. Press Ctrl+C to stop.
"venv\Scripts\python.exe" manage.py runserver 127.0.0.1:8000
