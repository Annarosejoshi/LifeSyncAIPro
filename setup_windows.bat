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
    echo [1/7] Virtual environment already exists.
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    exit /b 1
)

echo.
echo [2/7] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo.
echo [3/7] Installing project dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo [4/7] Applying database migrations...
python manage.py migrate
if errorlevel 1 exit /b 1

echo.
echo [5/7] Collecting static files...
python manage.py collectstatic --noinput
if errorlevel 1 exit /b 1

echo.
echo [6/7] Running Django system check...
python manage.py check
if errorlevel 1 exit /b 1

echo.
echo [7/7] Running automated tests...
python manage.py test
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
echo     venv\Scripts\activate
echo     python manage.py runserver
echo.
echo Then open:
echo     http://127.0.0.1:8000/
echo.

endlocal
