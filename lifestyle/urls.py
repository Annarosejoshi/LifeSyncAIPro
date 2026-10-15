from django.urls import path

from . import views


app_name = "lifestyle"


urlpatterns = [

    path(
        "check-in/",
        views.checkin_create,
        name="checkin_create"
    ),

    path(
        "check-in/<int:pk>/edit/",
        views.checkin_edit,
        name="checkin_edit"
    ),

    path(
        "check-in/<int:pk>/delete/",
        views.checkin_delete,
        name="checkin_delete"
    ),

    path(
        "history/",
        views.checkin_history,
        name="checkin_history"
    ),
    
    path(
    "habits/",
    views.habit_list,
    name="habit_list"
),

path(
    "habits/add/",
    views.habit_create,
    name="habit_create"
),

path(
    "habits/<int:pk>/delete/",
    views.habit_delete,
    name="habit_delete"
),
]