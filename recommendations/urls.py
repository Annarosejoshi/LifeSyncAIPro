from django.urls import path

from . import views


app_name = "recommendations"


urlpatterns = [
    path("chat/", views.chatbot_reply, name="chat"),

    path(
        "",
        views.recommendation_home,
        name="home",
    ),

    path(
        "history/",
        views.recommendation_history,
        name="history",
    ),

]
