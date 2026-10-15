@echo off
setlocal

cd /d "%~dp0"

echo.
echo ============================================================
echo 1. Django production deployment checks
echo ============================================================
echo.

REM Use a nested environment scope so production-only settings do not
REM leak into the normal Django test suite.
setlocal

set DJANGO_DEBUG=False
set DJANGO_SECRET_KEY=temporary-production-check-key-change-this-before-real-deployment-123456789
set DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
set DJANGO_CSRF_TRUSTED_ORIGINS=https://example.com
set DJANGO_SECURE_SSL_REDIRECT=True

python manage.py check --deploy
set DEPLOY_CHECK_EXIT=%ERRORLEVEL%

endlocal

echo.
echo ============================================================
echo 2. Normal project test suite
echo ============================================================
echo.

REM Tests are intentionally run outside the production environment above.
REM Django's test client uses normal HTTP requests, so forcing HTTPS here
REM would create redirect-related test failures.
python manage.py test
set TEST_EXIT=%ERRORLEVEL%

echo.

if not "%TEST_EXIT%"=="0" (
    echo ERROR: Test suite failed.
    exit /b %TEST_EXIT%
)

echo Test suite passed.

endlocal
