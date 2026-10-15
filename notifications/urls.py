from django.urls import path

from . import views


app_name = "notifications"

urlpatterns = [
    path("", views.medication_list, name="medication_list"),
    path("add/", views.medication_create, name="medication_create"),
    path("<int:pk>/edit/", views.medication_edit, name="medication_edit"),
    path("<int:pk>/delete/", views.medication_delete, name="medication_delete"),
    path("<int:pk>/toggle/", views.medication_toggle, name="medication_toggle"),
]
