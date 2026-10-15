from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils import timezone
from datetime import timedelta

from .models import RecommendationHistory

from lifestyle.models import DailyCheckIn
from lifestyle.health_score import calculate_ai_health_score
from mental_health.models import MentalWellnessAssessment

from ml.services.risk_prediction_service import (
    get_all_health_risks,
)

from .engine import generate_recommendations
from .coach import answer_with_optional_local_model
from .daily_actions import build_daily_actions

from .services import (
    save_recommendation_history,
)


@login_required
@require_POST
def chatbot_reply(request):
    question = request.POST.get("question", "").strip()
    if not question or len(question) > 500:
        return JsonResponse({"error": "Enter a question of 1 to 500 characters."}, status=400)
    checkin = DailyCheckIn.objects.filter(user=request.user, date=timezone.localdate()).first()
    answer, _ = answer_with_optional_local_model(question, checkin=checkin)
    return JsonResponse({"answer": answer})


@login_required
def recommendation_home(request):

    today = timezone.localdate()

    # ========================================================
    # TODAY'S LIFESTYLE DATA
    # ========================================================

    checkin = (
        DailyCheckIn.objects
        .filter(
            user=request.user,
            date=today,
        )
        .first()
    )

    # ========================================================
    # TODAY'S MENTAL WELLNESS ASSESSMENT
    # ========================================================

    mental_assessment = (
        MentalWellnessAssessment.objects
        .filter(
            user=request.user,
            date=today,
        )
        .first()
    )

    # ========================================================
    # HEALTH PROFILE
    # ========================================================

    try:
        profile = request.user.profile
    except Exception:
        profile = None

    # ========================================================
    # MACHINE-LEARNING RISK RESULTS
    # ========================================================

    try:
        health_risks = get_all_health_risks(
            request.user
        )

    except Exception:
        health_risks = {}

    # ========================================================
    # 7-DAY TREND CONTEXT FOR RECOMMENDATION ENGINE V2
    # ========================================================

    week_start = today - timedelta(days=6)

    weekly_checkins = list(
        DailyCheckIn.objects
        .filter(
            user=request.user,
            date__range=[week_start, today],
        )
        .order_by("date")
    )

    weekly_mental_records = list(
        MentalWellnessAssessment.objects
        .filter(
            user=request.user,
            date__range=[week_start, today],
        )
        .order_by("date")
    )

    mental_by_date = {
        item.date: item
        for item in weekly_mental_records
    }

    def average(values):
        values = list(values)
        if not values:
            return None
        return round(sum(values) / len(values), 1)

    weekly_scores = []
    component_history = {
        "sleep": [],
        "hydration": [],
        "exercise": [],
        "nutrition": [],
    }

    for historical_checkin in weekly_checkins:
        historical_mental = mental_by_date.get(
            historical_checkin.date
        )

        # Do not reuse today's live ML probabilities for historical dates.
        historical_score = calculate_ai_health_score(
            checkin=historical_checkin,
            mental_assessment=historical_mental,
            health_risks={},
        )

        if historical_score.get("available"):
            overall = historical_score.get("overall")
            if overall is not None:
                weekly_scores.append(float(overall))

            components = historical_score.get("components", {})

            for component_name in component_history:
                value = components.get(component_name)
                if value is not None:
                    component_history[component_name].append(
                        float(value)
                    )

    score_trend = None
    score_difference = None

    if len(weekly_scores) >= 2:
        score_difference = round(
            weekly_scores[-1] - weekly_scores[0],
            1,
        )

        if score_difference >= 3:
            score_trend = "improving"
        elif score_difference <= -3:
            score_trend = "declining"
        else:
            score_trend = "stable"

    component_averages = {
        name: average(values)
        for name, values in component_history.items()
        if values
    }

    weakest_component = None

    if component_averages:
        weakest_component = min(
            component_averages,
            key=component_averages.get,
        )

    weekly_trends = {
        "score_trend": score_trend,
        "score_difference": score_difference,
        "avg_sleep": average(
            float(item.sleep_hours or 0)
            for item in weekly_checkins
        ),
        "avg_water": average(
            float(item.water_liters or 0)
            for item in weekly_checkins
        ),
        "avg_exercise": average(
            float(item.exercise_minutes or 0)
            for item in weekly_checkins
        ),
        "avg_stress": average(
            float(item.stress_level or 0)
            for item in weekly_checkins
        ),
        "weakest_component": weakest_component,
        "recorded_days": len(weekly_checkins),
    }

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = []

    if checkin:

        recommendations = generate_recommendations(
            checkin=checkin,
            mental_assessment=mental_assessment,
            health_risks=health_risks,
            profile=profile,
            weekly_trends=weekly_trends,
        )
        
        save_recommendation_history(
    user=request.user,
    recommendations=recommendations,
    checkin=checkin,
    mental_assessment=mental_assessment,
)

    coach_question = ""
    coach_answer = None
    coach_mode = ""
    if request.method == "POST":
        coach_question = request.POST.get(
            "coach_question",
            "",
        ).strip()
        if len(coach_question) > 500:
            coach_answer = "Please keep your question to 500 characters or fewer."
        elif coach_question:
            coach_answer, coach_mode = answer_with_optional_local_model(
                coach_question,
                checkin=checkin,
            )
        else:
            coach_mode = ""
    else:
        coach_mode = "Reviewed local guidance"

    # ========================================================
    # TEMPLATE
    # ========================================================

    return render(
        request,
        "recommendations/home.html",
        {
            "daily_actions": build_daily_actions(checkin),
            "checkin": checkin,
            "mental_assessment": mental_assessment,
            "recommendations": recommendations,
            "health_risks": health_risks,
            "profile": profile,
            "today": today,
            "weekly_trends": weekly_trends,
            "coach_question": coach_question,
            "coach_answer": coach_answer,
            "coach_mode": coach_mode,
        },
    )
    
@login_required
def recommendation_history(request):

    history = (
        RecommendationHistory.objects
        .filter(user=request.user)
        .order_by(
            "-recommendation_date",
            "-created_at",
        )
    )

    active_count = history.filter(
        status="active"
    ).count()

    improving_count = history.filter(
        status="improving"
    ).count()

    completed_count = history.filter(
        status="completed"
    ).count()

    total_count = history.count()

    measurable_count = history.exclude(
        baseline_value__isnull=True
    ).count()

    context = {
        "history": history,
        "active_count": active_count,
        "improving_count": improving_count,
        "completed_count": completed_count,
        "total_count": total_count,
        "measurable_count": measurable_count,
    }

    return render(
        request,
        "recommendations/history.html",
        context,
    )
