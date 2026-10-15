@echo off
setlocal

cd /d "%~dp0"

if not exist "backups" mkdir "backups"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

if not exist "db.sqlite3" (
    echo ERROR: db.sqlite3 was not found in %CD%
    exit /b 1
)

copy /Y "db.sqlite3" "backups\db_%TS%.sqlite3" >nul

if errorlevel 1 (
    echo ERROR: Database backup failed.
    exit /b 1
)

echo Backup created successfully:
echo backups\db_%TS%.sqlite3

endlocal
