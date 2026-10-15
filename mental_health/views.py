from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from recommendations.progress_service import (
    evaluate_user_recommendation_progress,
)

from .forms import MentalWellnessAssessmentForm
from .insights import generate_mental_wellness_insights
from .models import MentalWellnessAssessment
from .wellness_score import (
    calculate_mental_wellness_score,
    get_mental_wellness_category,
)


@login_required
def assessment_create(request):

    today = timezone.localdate()

    existing = (
        MentalWellnessAssessment.objects
        .filter(
            user=request.user,
            date=today,
        )
        .first()
    )

    if existing and request.method != "POST":
        return redirect(
            "mental_health:assessment_edit",
            pk=existing.pk,
        )

    if request.method == "POST":

        form = MentalWellnessAssessmentForm(
            request.POST,
            instance=existing,
        )

        if form.is_valid():

            assessment = form.save(
                commit=False
            )

            assessment.user = request.user
            assessment.date = today
            assessment.save()

            evaluate_user_recommendation_progress(
                request.user
            )

            messages.success(
                request,
                f"Mental wellness saved: {calculate_mental_wellness_score(assessment):.1f}/100.",
            )

            return redirect(
                "dashboard:risks"
            )

    else:

        form = MentalWellnessAssessmentForm()

    return render(
        request,
        "mental_health/assessment_form.html",
        {
            "form": form,
            "page_title": "Mental Wellness Assessment",
        },
    )


@login_required
def assessment_edit(request, pk):

    assessment = get_object_or_404(
        MentalWellnessAssessment,
        pk=pk,
        user=request.user,
    )

    if request.method == "POST":

        form = MentalWellnessAssessmentForm(
            request.POST,
            instance=assessment,
        )

        if form.is_valid():

            # Date is excluded from the form, so the original
            # assessment date is preserved automatically.
            form.save()

            evaluate_user_recommendation_progress(
                request.user
            )

            messages.success(
                request,
                f"Mental wellness updated: {calculate_mental_wellness_score(assessment):.1f}/100.",
            )

            return redirect(
                "dashboard:risks" if assessment.date == timezone.localdate() else "mental_health:history"
            )

    else:

        form = MentalWellnessAssessmentForm(
            instance=assessment
        )

    return render(
        request,
        "mental_health/assessment_form.html",
        {
            "form": form,
            "page_title": "Edit Mental Wellness Assessment",
        },
    )


@login_required
def history(request):

    assessments = MentalWellnessAssessment.objects.filter(
        user=request.user
    )

    assessment_data = []

    for assessment in assessments:

        score = calculate_mental_wellness_score(
            assessment
        )

        category = get_mental_wellness_category(
            score
        )

        insights = generate_mental_wellness_insights(
            assessment
        )

        assessment_data.append(
            {
                "assessment": assessment,
                "score": score,
                "category": category,
                "insights": insights,
            }
        )

    return render(
        request,
        "mental_health/history.html",
        {
            "assessment_data": assessment_data,
        },
    )
