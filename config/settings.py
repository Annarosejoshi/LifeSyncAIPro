"""
Django settings for LifeSync AI Pro.

Development works out of the box.
Production-sensitive values can be supplied with environment variables.
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured


BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def env_bool(name, default=False):
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_list(name, default=None):
    value = os.getenv(name)

    if value is None:
        return list(default or [])

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


# ============================================================
# CORE SECURITY
# ============================================================

DEBUG = env_bool(
    "DJANGO_DEBUG",
    True,
)

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY"
)

if not SECRET_KEY:
    if DEBUG:
        # Development-only fallback.
        # Production refuses to start without DJANGO_SECRET_KEY.
        SECRET_KEY = (
            "django-insecure-development-only-"
            "replace-before-production"
        )
    else:
        raise ImproperlyConfigured(
            "DJANGO_SECRET_KEY must be set when DEBUG=False."
        )


ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    [
        "127.0.0.1",
        "localhost",
    ],
)

CSRF_TRUSTED_ORIGINS = env_list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
    [],
)


# ============================================================
# APPLICATIONS
# ============================================================

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "accounts.apps.AccountsConfig",
    "dashboard",
    "lifestyle",
    "mental_health",
    "predictions",
    "recommendations",
    "analytics",
    "notifications",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = "config.urls"


TEMPLATES = [
    {
        "BACKEND": (
            "django.template.backends.django."
            "DjangoTemplates"
        ),
        "DIRS": [
            BASE_DIR / "templates",
        ],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                (
                    "django.template.context_processors."
                    "request"
                ),
                (
                    "django.contrib.auth.context_processors."
                    "auth"
                ),
                (
                    "django.contrib.messages.context_processors."
                    "messages"
                ),
            ],
        },
    },
]


WSGI_APPLICATION = "config.wsgi.application"


# ============================================================
# DATABASE
# ============================================================

# SQLite is appropriate for the current MCA project/demo setup.
# Keep the database file out of Git and back it up separately.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# ============================================================
# PASSWORD VALIDATION
# ============================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "MinimumLengthValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "CommonPasswordValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth.password_validation."
            "NumericPasswordValidator"
        ),
    },
]


# ============================================================
# INTERNATIONALIZATION
# ============================================================

LANGUAGE_CODE = "en-us"

# Daily records use timezone.localdate(), so this should match
# the project's operating timezone.
TIME_ZONE = os.getenv(
    "DJANGO_TIME_ZONE",
    "Asia/Kolkata",
)

USE_I18N = True
USE_TZ = True


# ============================================================
# STATIC / MEDIA FILES
# ============================================================

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

STATIC_SOURCE_DIR = BASE_DIR / "static"

STATICFILES_DIRS = (
    [STATIC_SOURCE_DIR]
    if STATIC_SOURCE_DIR.exists()
    else []
)

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# ============================================================
# AUTHENTICATION
# ============================================================

LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "dashboard:home"
LOGOUT_REDIRECT_URL = "dashboard:landing"


# ============================================================
# EMAIL
# ============================================================

# Console backend is safe for local development.
# Replace through EMAIL_BACKEND in production.
EMAIL_BACKEND = os.getenv(
    "EMAIL_BACKEND",
    "django.core.mail.backends.console.EmailBackend",
)

DEFAULT_FROM_EMAIL = os.getenv(
    "DEFAULT_FROM_EMAIL",
    "LifeSync AI <noreply@localhost>",
)


# ============================================================
# WEB SECURITY
# ============================================================

SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"
CSRF_COOKIE_SAMESITE = "Lax"

X_FRAME_OPTIONS = "DENY"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"

# Secure cookies are automatically enabled when DEBUG=False.
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# Enable HTTPS redirection only when the deployment is actually
# configured for HTTPS. This avoids breaking local development.
SECURE_SSL_REDIRECT = env_bool(
    "DJANGO_SECURE_SSL_REDIRECT",
    False,
)

if not DEBUG and SECURE_SSL_REDIRECT:
    SECURE_HSTS_SECONDS = int(
        os.getenv(
            "DJANGO_HSTS_SECONDS",
            "31536000",
        )
    )
    SECURE_HSTS_INCLUDE_SUBDOMAINS = env_bool(
        "DJANGO_HSTS_INCLUDE_SUBDOMAINS",
        True,
    )
    SECURE_HSTS_PRELOAD = env_bool(
        "DJANGO_HSTS_PRELOAD",
        False,
    )

# Common reverse-proxy HTTPS header. Only meaningful when a trusted
# reverse proxy is configured to set this header correctly.
if env_bool(
    "DJANGO_TRUST_PROXY_SSL_HEADER",
    False,
):
    SECURE_PROXY_SSL_HEADER = (
        "HTTP_X_FORWARDED_PROTO",
        "https",
    )


# ============================================================
# UPLOAD SAFETY
# ============================================================

# Conservative request/upload limits for this health-tracking web app.
# Increase through code later only if a feature genuinely needs it.
DATA_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv(
    "DJANGO_LOG_LEVEL",
    "INFO",
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": [
            "console",
        ],
        "level": LOG_LEVEL,
    },
    "loggers": {
        "django": {
            "handlers": [
                "console",
            ],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
}


# ============================================================
# DEFAULT PRIMARY KEY
# ============================================================

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
