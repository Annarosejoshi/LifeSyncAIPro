from datetime import timedelta

from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.utils import timezone

from lifestyle.health_score import (
    calculate_ai_health_score,
)
from lifestyle.models import DailyCheckIn
from mental_health.models import MentalWellnessAssessment
from predictions.models import PredictionHistory


def _average(values):
    values = list(values)

    if not values:
        return 0

    return round(
        sum(values) / len(values),
        1,
    )


def _trend_from_values(values):
    """
    Return a simple week trend from the first and last recorded values.
    """
    values = list(values)

    if len(values) < 2:
        return {
            "direction": None,
            "difference": None,
        }

    difference = round(
        values[-1] - values[0],
        1,
    )

    if difference >= 3:
        direction = "improving"

    elif difference <= -3:
        direction = "declining"

    else:
        direction = "stable"

    return {
        "direction": direction,
        "difference": difference,
    }


def _latest_prediction_for_type(
    user,
    prediction_type,
):
    return (
        PredictionHistory.objects
        .filter(
            user=user,
            prediction_type=prediction_type,
        )
        .order_by("-created_at")
        .first()
    )


@login_required
def weekly_report(request):

    today = timezone.localdate()
    week_start = today - timedelta(days=6)

    # =====================================================
    # CHECK-INS
    # =====================================================

    checkins = list(
        DailyCheckIn.objects
        .filter(
            user=request.user,
            date__range=[
                week_start,
                today,
            ],
        )
        .order_by("date")
    )

    checkin_count = len(checkins)

    # =====================================================
    # MENTAL WELLNESS ASSESSMENTS
    # =====================================================

    mental_records = list(
        MentalWellnessAssessment.objects
        .filter(
            user=request.user,
            date__range=[
                week_start,
                today,
            ],
        )
        .order_by("date")
    )

    mental_count = len(
        mental_records
    )

    mental_by_date = {
        item.date: item
        for item in mental_records
    }

    # =====================================================
    # BASIC WEEKLY AVERAGES
    # =====================================================

    avg_sleep = _average(
        float(item.sleep_hours or 0)
        for item in checkins
    )

    avg_water = _average(
        float(item.water_liters or 0)
        for item in checkins
    )

    avg_exercise = _average(
        float(item.exercise_minutes or 0)
        for item in checkins
    )

    avg_steps = round(
        _average(
            float(item.steps or 0)
            for item in checkins
        )
    ) if checkins else 0

    avg_stress = _average(
        float(item.stress_level or 0)
        for item in checkins
    )

    avg_energy = _average(
        float(item.energy_level or 0)
        for item in checkins
    )

    avg_screen = _average(
        float(item.screen_time_hours or 0)
        for item in checkins
    )

    # =====================================================
    # MENTAL WELLNESS WEEKLY AVERAGES
    # =====================================================

    avg_mood = _average(
        float(item.mood or 0)
        for item in mental_records
    )

    avg_anxiety = _average(
        float(item.anxiety or 0)
        for item in mental_records
    )

    avg_mental_energy = _average(
        float(item.energy or 0)
        for item in mental_records
    )

    # =====================================================
    # AI HEALTH SCORE V2 - HISTORICAL DAILY SCORES
    #
    # IMPORTANT:
    # Historical days do NOT reuse today's live ML probabilities.
    # Each past score uses that day's check-in and same-day mental
    # assessment if available. ML risk modifiers remain 0 unless
    # historically dated predictions are explicitly added later.
    # =====================================================

    daily_ai_rows = []

    for checkin in checkins:

        same_day_mental = (
            mental_by_date.get(
                checkin.date
            )
        )

        ai_score = (
            calculate_ai_health_score(
                checkin=checkin,
                mental_assessment=same_day_mental,
                health_risks={},
            )
        )

        components = (
            ai_score.get(
                "components",
                {},
            )
        )

        daily_ai_rows.append({
            "date":
                checkin.date,

            "label":
                checkin.date.strftime(
                    "%a"
                ),

            "overall":
                ai_score.get(
                    "overall",
                    0,
                ),

            "category":
                ai_score.get(
                    "category",
                    "No Data",
                ),

            "physical":
                components.get(
                    "physical_lifestyle",
                    0,
                ),

            "mental":
                components.get(
                    "mental_wellness",
                    0,
                ),

            "sleep":
                components.get(
                    "sleep",
                    0,
                ),

            "hydration":
                components.get(
                    "hydration",
                    0,
                ),

            "exercise":
                components.get(
                    "exercise",
                    0,
                ),

            "nutrition":
                components.get(
                    "nutrition",
                    0,
                ),

            "screen":
                components.get(
                    "screen",
                    0,
                ),

            "mental_source":
                ai_score.get(
                    "mental_source"
                ),
        })

    ai_score_values = [
        item["overall"]
        for item in daily_ai_rows
    ]

    weekly_score = _average(
        ai_score_values
    )

    if weekly_score >= 85:
        weekly_category = "Excellent"

    elif weekly_score >= 70:
        weekly_category = "Good"

    elif weekly_score >= 55:
        weekly_category = "Fair"

    elif weekly_score > 0:
        weekly_category = "Needs Attention"

    else:
        weekly_category = "No Data"

    weekly_trend = (
        _trend_from_values(
            ai_score_values
        )
    )

    weekly_score_trend = (
        weekly_trend[
            "direction"
        ]
    )

    weekly_score_difference = (
        weekly_trend[
            "difference"
        ]
    )

    # =====================================================
    # AI SCORE CHART SERIES
    # =====================================================

    ai_chart_labels = [
        item["label"]
        for item in daily_ai_rows
    ]

    ai_score_chart = [
        item["overall"]
        for item in daily_ai_rows
    ]

    physical_chart = [
        item["physical"]
        for item in daily_ai_rows
    ]

    mental_score_chart = [
        item["mental"]
        for item in daily_ai_rows
    ]

    hydration_score_chart = [
        item["hydration"]
        for item in daily_ai_rows
    ]

    exercise_score_chart = [
        item["exercise"]
        for item in daily_ai_rows
    ]

    # =====================================================
    # RAW LIFESTYLE CHART SERIES
    # =====================================================

    chart_labels = [
        item.date.strftime("%a")
        for item in checkins
    ]

    sleep_chart = [
        float(
            item.sleep_hours or 0
        )
        for item in checkins
    ]

    water_chart = [
        float(
            item.water_liters or 0
        )
        for item in checkins
    ]

    exercise_chart = [
        item.exercise_minutes or 0
        for item in checkins
    ]

    stress_chart = [
        item.stress_level or 0
        for item in checkins
    ]

    # =====================================================
    # FOUR MODEL RISK CONTEXT
    # =====================================================

    risk_types = [
        (
            "diabetes",
            "Diabetes",
        ),
        (
            "hypertension",
            "Hypertension",
        ),
        (
            "cardiovascular",
            "Cardiovascular",
        ),
        (
            "mental_health",
            "Mental Wellness",
        ),
    ]

    latest_risks = {}

    for key, label in risk_types:

        latest_risks[key] = {
            "label":
                label,

            "record":
                _latest_prediction_for_type(
                    request.user,
                    key,
                ),
        }

    # Keep old diabetes variables for backward template compatibility.
    latest_risk = (
        latest_risks[
            "diabetes"
        ][
            "record"
        ]
    )

    diabetes_history = list(
        PredictionHistory.objects
        .filter(
            user=request.user,
            prediction_type="diabetes",
        )
        .order_by("-created_at")[:2]
    )

    previous_risk = (
        diabetes_history[1]
        if len(diabetes_history) >= 2
        else None
    )

    risk_trend = None
    risk_difference = None

    if latest_risk and previous_risk:

        risk_difference = round(
            (
                latest_risk.probability
                - previous_risk.probability
            )
            * 100,
            1,
        )

        if risk_difference <= -2:
            risk_trend = "improving"

        elif risk_difference >= 2:
            risk_trend = "increasing"

        else:
            risk_trend = "stable"

    # =====================================================
    # WEEKLY INSIGHTS
    # =====================================================

    insights = []

    if checkin_count == 0:

        insights.append(
            "No lifestyle check-ins were recorded during this reporting period."
        )

    else:

        if avg_sleep < 7:
            insights.append(
                "Average sleep was below the current LifeSync target."
            )

        if avg_water < 2:
            insights.append(
                "Hydration was lower than the current LifeSync target."
            )

        if avg_exercise < 30:
            insights.append(
                "Average daily exercise was below 30 minutes."
            )

        if avg_steps < 6000:
            insights.append(
                "Daily movement was relatively low this week."
            )

        if avg_stress >= 4:
            insights.append(
                "Average reported stress was elevated."
            )

        if avg_screen > 8:
            insights.append(
                "Average screen time was high."
            )

        if daily_ai_rows:

            weakest_components = {
                "Sleep":
                    _average(
                        item["sleep"]
                        for item
                        in daily_ai_rows
                    ),

                "Hydration":
                    _average(
                        item["hydration"]
                        for item
                        in daily_ai_rows
                    ),

                "Exercise":
                    _average(
                        item["exercise"]
                        for item
                        in daily_ai_rows
                    ),

                "Nutrition":
                    _average(
                        item["nutrition"]
                        for item
                        in daily_ai_rows
                    ),
            }

            weakest_name = min(
                weakest_components,
                key=weakest_components.get,
            )

            weakest_value = (
                weakest_components[
                    weakest_name
                ]
            )

            insights.append(
                f"{weakest_name} was the weakest AI Health Score component this week "
                f"at an average of {weakest_value}/100."
            )

        if weekly_score_trend == "improving":
            insights.append(
                f"Your AI Health Score improved by {abs(weekly_score_difference):.1f} points "
                "from the first to the latest recorded day."
            )

        elif weekly_score_trend == "declining":
            insights.append(
                f"Your AI Health Score decreased by {abs(weekly_score_difference):.1f} points "
                "from the first to the latest recorded day."
            )

        elif weekly_score_trend == "stable":
            insights.append(
                "Your AI Health Score remained relatively stable across the recorded days."
            )

    # =====================================================
    # CONTEXT
    # =====================================================

    context = {
        "today":
            today,

        "week_start":
            week_start,

        "checkin_count":
            checkin_count,

        "weekly_score":
            weekly_score,

        "weekly_category":
            weekly_category,

        "weekly_score_trend":
            weekly_score_trend,

        "weekly_score_difference":
            weekly_score_difference,

        "avg_sleep":
            avg_sleep,

        "avg_water":
            avg_water,

        "avg_exercise":
            avg_exercise,

        "avg_steps":
            avg_steps,

        "avg_stress":
            avg_stress,

        "avg_energy":
            avg_energy,

        "avg_screen":
            avg_screen,

        "mental_count":
            mental_count,

        "avg_mood":
            avg_mood,

        "avg_anxiety":
            avg_anxiety,

        "avg_mental_energy":
            avg_mental_energy,

        "latest_risk":
            latest_risk,

        "previous_risk":
            previous_risk,

        "risk_trend":
            risk_trend,

        "risk_difference":
            risk_difference,

        "latest_risks":
            latest_risks,

        "insights":
            insights,

        "daily_ai_rows":
            daily_ai_rows,

        "chart_labels":
            chart_labels,

        "sleep_chart":
            sleep_chart,

        "water_chart":
            water_chart,

        "exercise_chart":
            exercise_chart,

        "stress_chart":
            stress_chart,

        "ai_chart_labels":
            ai_chart_labels,

        "ai_score_chart":
            ai_score_chart,

        "physical_chart":
            physical_chart,

        "mental_score_chart":
            mental_score_chart,

        "hydration_score_chart":
            hydration_score_chart,

        "exercise_score_chart":
            exercise_score_chart,
    }

    return render(
        request,
        "analytics/weekly_report.html",
        context,
    )
