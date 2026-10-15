from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path(
        "admin/",
        admin.site.urls,
    ),
    path(
        "accounts/",
        include("accounts.urls"),
    ),
    path(
        "lifestyle/",
        include("lifestyle.urls"),
    ),
    path(
        "mental-health/",
        include("mental_health.urls"),
    ),
    path(
        "recommendations/",
        include("recommendations.urls"),
    ),
    path(
        "predictions/",
        include("predictions.urls"),
    ),
    path(
        "medications/",
        include("notifications.urls"),
    ),
   
    path(
        "",
        include("dashboard.urls"),
    )
   
   ]



# Django serves uploaded media only in development.
# In production, the web server/platform should serve media safely.
if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT,
    )
