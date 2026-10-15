from django.urls import path

from . import views


app_name = "predictions"


urlpatterns = [
    path(
        "diabetes/",
        views.diabetes_risk,
        name="diabetes_risk"
    ),
    
path(
    "history/",
    views.prediction_history,
    name="history"
),
]