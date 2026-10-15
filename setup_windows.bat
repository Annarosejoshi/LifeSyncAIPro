@echo off
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo LifeSync AI Pro - Windows Setup
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found in PATH.
    echo Install Python 3.13 and try again.
    exit /b 1
)

echo Python:
python --version
echo.

if not exist "manage.py" (
    echo ERROR: manage.py was not found in %CD%
    echo Run this script from the LifeSyncAIPro project folder.
    exit /b 1
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found.
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [1/7] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 exit /b 1
) else (
    "venv\Scripts\python.exe" --version >nul 2>nul
    if errorlevel 1 (
        echo [1/7] Repairing the copied virtual environment with this computer's Python...
        python -m venv venv
        if errorlevel 1 exit /b 1
    ) else (
        echo [1/7] Existing virtual environment works.
    )
)

echo.
echo [2/7] Upgrading pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo.
echo [3/7] Installing project dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
"venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
"venv\Scripts\python.exe" manage.py check_models
if errorlevel 1 exit /b 1

echo.
echo [4/7] Applying database migrations...
"venv\Scripts\python.exe" manage.py migrate
if errorlevel 1 exit /b 1

echo.
echo [5/7] Collecting static files...
"venv\Scripts\python.exe" manage.py collectstatic --noinput
if errorlevel 1 exit /b 1

echo.
echo [6/7] Running Django system check...
"venv\Scripts\python.exe" manage.py check
if errorlevel 1 exit /b 1

echo.
echo [7/7] Running automated tests...
"venv\Scripts\python.exe" manage.py test
if errorlevel 1 (
    echo.
    echo ERROR: Automated tests failed.
    exit /b 1
)

echo.
echo ============================================================
echo SETUP COMPLETE
echo ============================================================
echo.
echo Start the development server with:
echo.
echo     run_windows.bat
echo.
echo Then open:
echo     http://127.0.0.1:8000/
echo.

endlocal
