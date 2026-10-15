
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

from .forms import RegisterForm, UserProfileForm
from .models import UserProfile


def register_view(request):
    # If the user is already logged in, go to the dashboard.
    if request.user.is_authenticated:
        return redirect("dashboard:home")

    if request.method == "POST":
        form = RegisterForm(request.POST)

        if form.is_valid():
            # RegisterForm saves the user and their health profile.
            user = form.save()

            messages.success(
                request,
                "Your LifeSync AI account has been created successfully. Please sign in."
            )

            # Send the new user to the login page so they can sign in.
            return redirect("accounts:login")

    else:
        form = RegisterForm()

    return render(
        request,
        "accounts/register.html",
        {"form": form}
    )


@login_required
def profile_view(request):
    # Get the user's profile, creating one only if it does not exist.
    profile, created = UserProfile.objects.get_or_create(
        user=request.user
    )

    return render(
        request,
        "accounts/profile.html",
        {"profile": profile}
    )


@login_required
def profile_edit_view(request):
    # Load the existing profile for editing.
    profile, created = UserProfile.objects.get_or_create(
        user=request.user
    )

    if request.method == "POST":
        form = UserProfileForm(
            request.POST,
            instance=profile
        )

        if form.is_valid():
            form.save()

            messages.success(
                request,
                "Your health profile has been updated successfully."
            )

            return redirect("accounts:profile")

    else:
        form = UserProfileForm(instance=profile)

    return render(
        request,
        "accounts/profile_edit.html",
        {
            "form": form,
            "profile": profile,
        }
    )
