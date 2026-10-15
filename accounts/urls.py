from django.urls import path
from django.contrib.auth import views as auth_views

from . import forms, views


app_name = "accounts"


urlpatterns = [
    path(
        "register/",
        views.register_view,
        name="register",
    ),

    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="accounts/login.html",
            authentication_form=forms.LoginForm,
            redirect_authenticated_user=True,
        ),
        name="login",
    ),

    path(
        "logout/",
        auth_views.LogoutView.as_view(),
        name="logout",
    ),

    path(
        "profile/",
        views.profile_view,
        name="profile",
    ),

    path(
        "profile/edit/",
        views.profile_edit_view,
        name="profile_edit",
    ),
]