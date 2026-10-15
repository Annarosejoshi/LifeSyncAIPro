from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("check-in/", views.dashboard_home, {"page": "checkin"}, name="checkin"),
    path("bmi/", views.dashboard_home, {"page": "bmi"}, name="bmi"),
    path("risks/", views.dashboard_home, {"page": "risks"}, name="risks"),
    path("reports/", views.dashboard_home, {"page": "reports"}, name="reports"),
    path("welcome/", views.landing_page, name="landing"),
    path("", views.dashboard_home, name="home"),
]