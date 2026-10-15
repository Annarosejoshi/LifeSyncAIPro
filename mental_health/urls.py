from django.urls import path
from . import views

app_name = "mental_health"

urlpatterns = [
    path(
        "",
        views.assessment_create,
        name="assessment_create"
    ),

    path(
        "history/",
        views.history,
        name="history"
    ),

    path(
        "<int:pk>/edit/",
        views.assessment_edit,
        name="assessment_edit"
    ),
]