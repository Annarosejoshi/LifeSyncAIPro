from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from recommendations.progress_service import (
    evaluate_user_recommendation_progress,
)

from .forms import DailyCheckInForm, HabitGoalForm
from .health_score import (
    calculate_daily_health_score,
    get_health_score_category,
)
from .insights import generate_daily_insights
from .models import DailyCheckIn, HabitGoal
from ml.services.risk_prediction_service import get_all_health_risks
from ml.services.risk_history_service import save_all_health_risk_history


@login_required
def checkin_create(request):

    today = timezone.localdate()

    existing_checkin = (
        DailyCheckIn.objects
        .filter(
            user=request.user,
            date=today,
        )
        .first()
    )

    if existing_checkin:
        return redirect(
            "lifestyle:checkin_edit",
            pk=existing_checkin.pk,
        )

    if request.method == "POST":

        form = DailyCheckInForm(
            request.POST
        )

        if form.is_valid():

            checkin = form.save(
                commit=False
            )

            checkin.user = request.user
            checkin.date = today
            checkin.save()
            save_all_health_risk_history(request.user, get_all_health_risks(request.user))

            evaluate_user_recommendation_progress(
                request.user
            )

            messages.success(
                request,
                "Daily health check-in saved successfully.",
            )

            return redirect(
                "recommendations:home"
            )

        messages.error(
            request,
            "Please correct the highlighted fields and save again.",
        )

    else:
        form = DailyCheckInForm()

    return render(
        request,
        "lifestyle/checkin_form.html",
        {
            "form": form,
            "page_title": "Daily Health Check-In",
        },
    )


@login_required
def checkin_edit(request, pk):

    checkin = get_object_or_404(
        DailyCheckIn,
        pk=pk,
        user=request.user,
    )

    if request.method == "POST":

        form = DailyCheckInForm(
            request.POST,
            instance=checkin,
        )

        if form.is_valid():

            # Date is excluded from the form, so the original
            # check-in date is preserved automatically.
            form.save()
            if checkin.date == timezone.localdate():
                save_all_health_risk_history(request.user, get_all_health_risks(request.user))

            evaluate_user_recommendation_progress(
                request.user
            )

            messages.success(
                request,
                "Daily check-in updated successfully.",
            )

            return redirect(
                "lifestyle:checkin_history"
            )

    else:

        form = DailyCheckInForm(
            instance=checkin
        )

    return render(
        request,
        "lifestyle/checkin_form.html",
        {
            "form": form,
            "page_title": "Edit Daily Check-In",
        },
    )


@login_required
@require_POST
def checkin_delete(request, pk):
    checkin = get_object_or_404(
        DailyCheckIn,
        pk=pk,
        user=request.user,
    )
    checkin.delete()
    messages.success(request, "Daily check-in deleted.")
    return redirect("lifestyle:checkin_history")


@login_required
def checkin_history(request):

    checkins = DailyCheckIn.objects.filter(
        user=request.user
    )

    checkin_data = []

    for checkin in checkins:

        scores = calculate_daily_health_score(
            checkin
        )

        category = get_health_score_category(
            scores["overall"]
        )

        insights = generate_daily_insights(
            checkin,
            scores,
        )

        checkin_data.append(
            {
                "checkin": checkin,
                "scores": scores,
                "category": category,
                "insights": insights,
            }
        )

    return render(
        request,
        "lifestyle/checkin_history.html",
        {
            "checkin_data": checkin_data,
        },
    )


@login_required
def habit_list(request):

    goals = HabitGoal.objects.filter(
        user=request.user,
        active=True,
    )

    today = timezone.localdate()

    today_checkin = (
        DailyCheckIn.objects
        .filter(
            user=request.user,
            date=today,
        )
        .first()
    )

    habit_results = []

    if today_checkin:

        from .habit_engine import evaluate_habit

        for goal in goals:

            result = evaluate_habit(
                goal,
                today_checkin,
            )

            if result:
                habit_results.append(
                    result
                )

    return render(
        request,
        "lifestyle/habit_list.html",
        {
            "goals": goals,
            "habit_results": habit_results,
            "today_checkin": today_checkin,
        },
    )


@login_required
def habit_create(request):

    if request.method == "POST":

        form = HabitGoalForm(
            request.POST
        )

        if form.is_valid():

            goal = form.save(
                commit=False
            )

            goal.user = request.user
            goal.save()

            messages.success(
                request,
                "Habit goal created successfully.",
            )

            return redirect(
                "lifestyle:habit_list"
            )

    else:

        form = HabitGoalForm()

    return render(
        request,
        "lifestyle/habit_form.html",
        {
            "form": form,
        },
    )


@login_required
def habit_delete(request, pk):

    goal = get_object_or_404(
        HabitGoal,
        pk=pk,
        user=request.user,
    )

    if request.method == "POST":

        goal.delete()

        messages.success(
            request,
            "Habit goal removed.",
        )

    return redirect(
        "lifestyle:habit_list"
    )
