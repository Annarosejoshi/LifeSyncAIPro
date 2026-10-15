from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.cache import never_cache

from accounts.models import UserProfile
from accounts.forms import UserProfileForm
from ml.services.risk_input_mapper import build_common_risk_input
from ml.services.risk_history_service import (
    save_all_health_risk_history,
)

from lifestyle.forms import BMIForm, DailyCheckInForm
from lifestyle.health_score import calculate_ai_health_score
from lifestyle.models import BMIRecord, DailyCheckIn, HabitGoal
from mental_health.models import MentalWellnessAssessment
from mental_health.wellness_score import calculate_mental_wellness_score, get_mental_wellness_category
from predictions.models import PredictionHistory
from recommendations.models import RecommendationHistory
from recommendations.engine import generate_recommendations
from recommendations.progress_service import (
    evaluate_user_recommendation_progress,
)
from recommendations.services import save_recommendation_history
from recommendations.coach import answer_with_optional_local_model
from recommendations.daily_actions import build_daily_actions
from predictions.presentation import prediction_input_details, overall_screening_summary
from notifications.models import MedicationRecord

from ml.services.risk_prediction_service import get_all_health_risks


def landing_page(request):
    return render(request, "dashboard/landing.html")


def generate_daily_insights(checkin):
    insights = []

    if not checkin:
        return insights

    sleep_hours = float(
        checkin.sleep_hours or 0
    )

    water = float(
        checkin.water_liters or 0
    )

    exercise_minutes = float(
        checkin.exercise_minutes or 0
    )

    stress = float(
        checkin.stress_level or 0
    )

    screen_time = float(
        checkin.screen_time_hours or 0
    )

    if sleep_hours < 7:
        insights.append(
            "Your sleep duration is below the recommended target."
        )

    if water < 2:
        insights.append(
            "Your recorded hydration is relatively low today."
        )

    if exercise_minutes < 30:
        insights.append(
            "You recorded less than 30 minutes of exercise today."
        )

    if stress >= 4:
        insights.append(
            "Your stress level is elevated today."
        )

    if screen_time > 8:
        insights.append(
            "Your screen time is high today."
        )

    if not insights:
        insights.append(
            "Your recorded lifestyle indicators look well balanced today."
        )

    return insights


def build_habit_results(active_goals, active_checkin):
    """
    Compare each active habit goal with the latest/today check-in.

    Returns:
    [
        {
            "habit": "Water Intake",
            "achieved": True,
            "current": 2.5,
            "target": 2.0,
        },
        ...
    ]
    """

    habit_results = []

    for goal in active_goals:

        achieved = False
        current = 0

        target = float(goal.target_value or 0)

        if active_checkin:

            # -----------------------------------------
            # WATER
            # Higher value is better
            # -----------------------------------------
            if goal.habit_type == "water":

                current = float(
                    active_checkin.water_liters or 0
                )

                achieved = current >= target

            # -----------------------------------------
            # SLEEP
            # Higher value is better
            # -----------------------------------------
            elif goal.habit_type == "sleep":

                current = float(
                    active_checkin.sleep_hours or 0
                )

                achieved = current >= target

            # -----------------------------------------
            # EXERCISE
            # Higher value is better
            # -----------------------------------------
            elif goal.habit_type == "exercise":

                current = float(
                    active_checkin.exercise_minutes or 0
                )

                achieved = current >= target

            # -----------------------------------------
            # STEPS
            # Higher value is better
            # -----------------------------------------
            elif goal.habit_type == "steps":

                current = float(
                    active_checkin.steps or 0
                )

                achieved = current >= target

            # -----------------------------------------
            # NUTRITION
            # Fruit + vegetable servings
            # Higher value is better
            # -----------------------------------------
            elif goal.habit_type == "nutrition":

                current = float(
                    active_checkin.fruit_vegetable_servings or 0
                )

                achieved = current >= target

            # -----------------------------------------
            # SCREEN TIME
            # Lower value is better
            # -----------------------------------------
            elif goal.habit_type == "screen_time":

                current = float(
                    active_checkin.screen_time_hours or 0
                )

                achieved = current <= target

            # -----------------------------------------
            # STRESS
            # Lower value is better
            # -----------------------------------------
            elif goal.habit_type == "stress":

                current = float(
                    active_checkin.stress_level or 0
                )

                achieved = current <= target

        habit_results.append(
            {
                "habit": goal.get_habit_type_display(),
                "habit_type": goal.habit_type,
                "achieved": achieved,
                "current": current,
                "target": target,
            }
        )

    return habit_results

@login_required
@never_cache
def dashboard_home(request, page="home"):
    page_templates = {name: f"dashboard/{name}.html" for name in ("home", "checkin", "bmi", "risks", "reports")}

    # =====================================================
    # DATE RANGE
    # =====================================================

    today = timezone.localdate()
    today_medications = MedicationRecord.objects.filter(
        user=request.user,
        scheduled_date=today,
    )

    seven_days_ago = (
        today - timedelta(days=6)
    )


    # =====================================================
    # TODAY / LATEST CHECK-IN
    # =====================================================

    today_checkin = (
        DailyCheckIn.objects
        .filter(
            user=request.user,
            date=today,
        )
        .first()
    )

    checkin_form = DailyCheckInForm(
        instance=today_checkin,
    )

    if (
        request.method == "POST"
        and request.POST.get("action") == "daily_checkin"
    ):
        checkin_form = DailyCheckInForm(
            request.POST,
            instance=today_checkin,
        )

        if checkin_form.is_valid():
            checkin = checkin_form.save(commit=False)
            checkin.user = request.user
            checkin.date = today
            checkin.save()

            profile, _ = UserProfile.objects.get_or_create(
                user=request.user,
            )
            mental_assessment = (
                MentalWellnessAssessment.objects
                .filter(
                    user=request.user,
                    date=today,
                )
                .first()
            )

            try:
                health_risks = get_all_health_risks(
                    request.user,
                )
                save_all_health_risk_history(request.user, health_risks)
            except Exception:
                health_risks = {}

            recommendations = generate_recommendations(
                checkin=checkin,
                mental_assessment=mental_assessment,
                health_risks=health_risks,
                profile=profile,
            )
            save_recommendation_history(
                user=request.user,
                recommendations=recommendations,
                checkin=checkin,
                mental_assessment=mental_assessment,
            )

            evaluate_user_recommendation_progress(
                request.user,
            )

            messages.success(
                request,
                "Today's check-in was saved to your dashboard.",
            )
            return redirect("dashboard:risks")

        messages.error(
            request,
            "Please correct the check-in fields marked below.",
        )

    latest_checkin = (
        DailyCheckIn.objects
        .filter(
            user=request.user
        )
        .order_by("-date")
        .first()
    )

    active_checkin = today_checkin

    coach_question = ""
    coach_answer = None
    coach_mode = ""
    if (
        request.method == "POST"
        and request.POST.get("action") == "coach_question"
    ):
        coach_question = request.POST.get("coach_question", "").strip()
        if len(coach_question) > 500:
            coach_answer = "Please keep your question to 500 characters or fewer."
            coach_mode = "Question limit"
        elif coach_question:
            coach_answer, coach_mode = answer_with_optional_local_model(
                coach_question,
                checkin=active_checkin,
            )

    profile, _ = UserProfile.objects.get_or_create(
        user=request.user
    )

    # Let users supply missing prediction inputs on the results page.
    # Preserve all existing fields rather than replacing them with defaults.
    common_inputs = build_common_risk_input(request.user)
    profile_feature_fields = {
        "age": ("date_of_birth",), "sex": ("gender",),
        "bmi": tuple(field for field in ("height_cm", "weight_kg") if not getattr(profile, field)),
        "smoking_status": ("smoking_status",),
    }
    missing_profile_fields = [
        field for feature, fields in profile_feature_fields.items()
        if common_inputs.get(feature) is None for field in fields
    ]
    updating_prediction_profile = request.method == "POST" and request.POST.get("action") == "prediction_profile"
    prediction_profile_form = UserProfileForm(
        request.POST if updating_prediction_profile else None, instance=profile,
    )
    for field in list(prediction_profile_form.fields):
        if field not in missing_profile_fields:
            del prediction_profile_form.fields[field]
        else:
            prediction_profile_form.fields[field].required = True
    if "gender" in prediction_profile_form.fields:
        prediction_profile_form.fields["gender"].help_text = "These predictions support the male and female profile categories."
    if updating_prediction_profile and prediction_profile_form.is_valid():
        prediction_profile_form.save()
        messages.success(request, "Profile updated. Your predictions have been refreshed.")
        return redirect("dashboard:risks")

    profile_age = None
    if profile.date_of_birth:
        profile_age = (
            today.year - profile.date_of_birth.year
            - ((today.month, today.day) < (
                profile.date_of_birth.month,
                profile.date_of_birth.day,
            ))
        )

    bmi_form = BMIForm(initial={
        "height_cm": profile.height_cm,
        "weight_kg": profile.weight_kg,
    })

    if (
        request.method == "POST"
        and request.POST.get("action") == "bmi_calculation"
    ):
        bmi_form = BMIForm(request.POST)

        if bmi_form.is_valid():
            height_cm = bmi_form.cleaned_data["height_cm"]
            weight_kg = bmi_form.cleaned_data["weight_kg"]
            birth_date = profile.date_of_birth
            age = None

            if birth_date:
                age = (
                    today.year - birth_date.year
                    - ((today.month, today.day) < (birth_date.month, birth_date.day))
                )

            if age is None:
                category = "Age not available"
            elif age < 18:
                category = "Age-specific chart required"
            else:
                bmi_value = weight_kg / ((height_cm / 100) ** 2)
                if bmi_value < 18.5:
                    category = "Below usual adult range"
                elif bmi_value < 25:
                    category = "Usual adult range"
                elif bmi_value < 30:
                    category = "Above usual adult range"
                else:
                    category = "High adult BMI range"

            BMIRecord.objects.update_or_create(
                user=request.user,
                date=today,
                defaults={
                    "height_cm": height_cm,
                    "weight_kg": weight_kg,
                    "category": category,
                },
            )

            profile.height_cm = height_cm
            profile.weight_kg = weight_kg
            profile.save(update_fields=["height_cm", "weight_kg"])

            messages.success(request, "BMI result saved to your health history.")
            return redirect("dashboard:bmi")

        messages.error(request, "Please enter valid height and weight values.")

    bmi_history = list(
        BMIRecord.objects
        .filter(user=request.user)
        .order_by("-date", "-created_at")[:7]
    )
    latest_bmi = bmi_history[0] if bmi_history else None
    bmi_chart_history = list(reversed(bmi_history))

    display_name = (
        request.user.get_full_name().strip()
        or request.user.username
    )


    # =====================================================
    # DAILY INSIGHTS
    # =====================================================

    insights = generate_daily_insights(
        active_checkin
    )


    # =====================================================
    # RECENT 7-DAY CHECK-INS
    # =====================================================

    recent_checkins = list(
        DailyCheckIn.objects
        .filter(
            user=request.user,
            date__range=(
                seven_days_ago,
                today,
            ),
        )
        .order_by("date")
    )

    recent_mental_assessments = list(
        MentalWellnessAssessment.objects
        .filter(
            user=request.user,
            date__range=(
                seven_days_ago,
                today,
            ),
        )
        .order_by("date")
    )

    mental_by_date = {
        assessment.date: assessment
        for assessment in recent_mental_assessments
    }


    # =====================================================
    # WEEKLY VALUES
    # =====================================================

    weekly_average_score = 0
    weekly_average_sleep = 0
    weekly_average_water = 0
    weekly_exercise = 0

    total_scores = []
    score_values = []

    # Apply each day's own saved screening results to its score.
    # Today's results are refreshed below before being used in the chart.
    historical_risks = {}
    for prediction in PredictionHistory.objects.filter(
        user=request.user, created_at__date__range=(seven_days_ago, today),
        model_version__in=("lifestyle_v2", "lifestyle_v6", "lifestyle_v10"),
    ).order_by("created_at"):
        prediction_date = timezone.localtime(prediction.created_at).date()
        historical_risks.setdefault(prediction_date, {})[prediction.prediction_type] = {
            "prediction_available": True,
            "prediction": prediction.predicted_class,
            "probability": prediction.probability,
            "threshold": prediction.threshold,
        }

    if recent_checkins:

        sleep_values_week = [
            float(
                item.sleep_hours or 0
            )
            for item in recent_checkins
        ]

        water_values_week = [
            float(
                item.water_liters or 0
            )
            for item in recent_checkins
        ]

        exercise_values_week = [
            float(
                item.exercise_minutes or 0
            )
            for item in recent_checkins
        ]

        weekly_average_sleep = round(
            sum(sleep_values_week)
            / len(sleep_values_week),
            1,
        )

        weekly_average_water = round(
            sum(water_values_week)
            / len(water_values_week),
            1,
        )

        weekly_exercise = round(
            sum(exercise_values_week),
            1,
        )

        for item in recent_checkins:

            historical_ai_health = (
                calculate_ai_health_score(
                    checkin=item,
                    mental_assessment=mental_by_date.get(
                        item.date
                    ),
                    health_risks=historical_risks.get(item.date, {}),
                )
            )

            historical_score = (
                historical_ai_health.get(
                    "overall"
                )
            )

            if historical_score is not None:
                score_values.append(historical_score)
                total_scores.append(
                    historical_score
                )
            else:
                score_values.append(None)

        if total_scores:
            weekly_average_score = round(
                sum(total_scores)
                / len(total_scores),
                1,
            )


    # =====================================================
    # CHART DATA
    # =====================================================

    chart_labels = [
        item.date.strftime("%a")
        for item in recent_checkins
    ]

    health_scores = total_scores

    sleep_values = [
        float(
            item.sleep_hours or 0
        )
        for item in recent_checkins
    ]

    water_values = [
        float(
            item.water_liters or 0
        )
        for item in recent_checkins
    ]

    exercise_values = [
        float(
            item.exercise_minutes or 0
        )
        for item in recent_checkins
    ]

    mood_values = [
        float(
            item.mood or 0
        )
        for item in recent_checkins
    ]

    stress_values = [
        float(
            item.stress_level or 0
        )
        for item in recent_checkins
    ]


    # =====================================================
    # DIABETES AI PREDICTION
    # =====================================================

    latest_diabetes_prediction = (
        PredictionHistory.objects
        .filter(
            user=request.user,
            prediction_type="diabetes",
        )
        .order_by("-created_at")
        .first()
    )

    
        # =====================================================
    # LIVE ML HEALTH RISK PREDICTIONS
    # =====================================================

    risk_predictions = {}
    predictions_calculated_at = None

    diabetes_risk = None
    hypertension_risk = None
    cardiovascular_risk = None
    mental_health_risk = None

    try:
        risk_predictions = get_all_health_risks(
            request.user
        )
        if any(result and result.get("prediction_available") for result in risk_predictions.values()):
            predictions_calculated_at = timezone.now()
        
        save_all_health_risk_history(
            request.user,
            risk_predictions,
        )

        diabetes_risk = risk_predictions.get(
            "diabetes"
        )

        hypertension_risk = risk_predictions.get(
            "hypertension"
        )

        cardiovascular_risk = risk_predictions.get(
            "cardiovascular"
        )

        mental_health_risk = risk_predictions.get(
            "mental_health"
        )

    except Exception as exc:
        print(
            "Dashboard health risk prediction error:",
            exc,
        )
        
        

    diabetes_readiness = (
        diabetes_risk.get("readiness", {})
        if diabetes_risk
        else {}
    )
    risk_input_labels = {
        "age": "date of birth",
        "sex": "gender",
        "bmi": "height and weight",
        "smoking_status": "smoking status",
        "sleep_hours": "a recent sleep check-in",
    }
    risk_missing_inputs = [
        risk_input_labels.get(feature, feature.replace("_", " "))
        for feature in diabetes_readiness.get("missing_features", [])
    ]
    risk_cards = [
        ("Diabetes screening", diabetes_risk),
        ("Hypertension screening", hypertension_risk),
        ("Cardiovascular screening", cardiovascular_risk),
        ("Mental wellness screening", mental_health_risk),
    ]
    risk_ready_count = sum(bool(result and result.get("prediction_available")) for _, result in risk_cards)
    for _, result in risk_cards:
        if result:
            result["display_inputs"] = prediction_input_details(result)


    # =====================================================
    # HABIT ADHERENCE
    # =====================================================

    active_goals = list(
        HabitGoal.objects
        .filter(
            user=request.user,
            active=True,
        )
    )

    habit_results_ai = (
        build_habit_results(
            active_goals,
            active_checkin,
        )
    )


    # =====================================================
    # AI HEALTH SCORE V2
    # =====================================================

    # Prefer today's dedicated mental assessment. If there is no
    # assessment today, calculate_ai_health_score() automatically
    # falls back to mood/stress/energy from the active check-in.
    today_mental_assessment = (
        mental_by_date.get(today)
    )
    mental_assessment_result = None
    if today_mental_assessment is not None:
        assessment_score = calculate_mental_wellness_score(today_mental_assessment)
        mental_assessment_result = {
            "score": assessment_score,
            "category": get_mental_wellness_category(assessment_score),
        }

    ai_health = calculate_ai_health_score(
        checkin=active_checkin,
        mental_assessment=today_mental_assessment,
        health_risks=risk_predictions,
    )

    ai_health_score = ai_health.get("overall")
    for index, checkin in enumerate(recent_checkins):
        if checkin.date == today:
            score_values[index] = ai_health_score
    total_scores[:] = [value for value in score_values if value is not None]
    weekly_average_score = (
        round(sum(total_scores) / len(total_scores), 1) if total_scores else 0
    )
    ai_health_category = ai_health.get("category", "No Data")
    ai_health_components = ai_health.get("components", {})
    ai_health_base_score = ai_health.get("base_score")
    ai_health_risk_penalty = ai_health.get("risk_penalty", 0)
    ai_health_risk_details = ai_health.get("risk_details", {})
    ai_health_mental_source = ai_health.get("mental_source")
    ai_health_explanation = ai_health.get("explanation", "")



    # =====================================================
    # RECOMMENDATION PROGRESS SUMMARY
    # =====================================================

    recommendation_history = (
        RecommendationHistory.objects
        .filter(user=request.user)
    )

    recommendation_active_count = (
        recommendation_history
        .filter(status="active")
        .count()
    )

    recommendation_improving_count = (
        recommendation_history
        .filter(status="improving")
        .count()
    )

    recommendation_completed_count = (
        recommendation_history
        .filter(status="completed")
        .count()
    )

    recommendation_total_count = (
        recommendation_history.count()
    )

    # Prioritize improving recommendations first, then active items.
    # Within each status, higher-priority recommendations appear first.
    priority_rank = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    status_rank = {
        "improving": 0,
        "active": 1,
    }

    recommendation_candidates = list(
        recommendation_history
        .filter(
            status__in=[
                "active",
                "improving",
            ]
        )
        .order_by(
            "-recommendation_date",
            "-created_at",
        )
    )

    recommendation_candidates.sort(
        key=lambda item: (
            status_rank.get(
                item.status,
                9,
            ),
            priority_rank.get(
                item.priority,
                9,
            ),
            -item.recommendation_date.toordinal(),
        )
    )

    top_recommendations = (
        recommendation_candidates[:3]
    )


    dashboard_data = {
        "name": display_name,
        "hasCheckin": active_checkin is not None,
        "healthScore": ai_health_score,
        "healthCategory": ai_health_category,
        "sleep": active_checkin.sleep_hours if active_checkin else None,
        "water": active_checkin.water_liters if active_checkin else None,
        "exercise": active_checkin.exercise_minutes if active_checkin else None,
        "nutrition": active_checkin.fruit_vegetable_servings if active_checkin else None,
        "mealsCompleted": (
            sum((
                active_checkin.breakfast_taken,
                active_checkin.lunch_taken,
                active_checkin.dinner_taken,
            ))
            if active_checkin else None
        ),
        "mood": active_checkin.get_mood_display() if active_checkin else None,
        "stress": active_checkin.stress_level if active_checkin else None,
        "screen": active_checkin.screen_time_hours if active_checkin else None,
        "height": profile.height_cm,
        "weight": profile.weight_kg,
        "bmiAdultEligible": (
            None if profile_age is None else profile_age >= 18
        ),
        "chartLabels": chart_labels,
        "sleepRaw": sleep_values,
        "waterRaw": water_values,
        "exerciseRaw": exercise_values,
        "scoreTrend": score_values,
        "bmiChartLabels": [
            record.date.strftime("%b %d")
            for record in bmi_chart_history
        ],
        "bmiChartValues": [record.bmi for record in bmi_chart_history],
        "sleepTrend": [
            min(value / 8 * 100, 100)
            for value in sleep_values
        ],
        "waterTrend": [
            min(value / 2.5 * 100, 100)
            for value in water_values
        ],
        "exerciseTrend": [
            min(value / 30 * 100, 100)
            for value in exercise_values
        ],
    }


    # =====================================================
    # CONTEXT
    # =====================================================

    checkin_sections = [
        (
            "checkin-recovery",
            "😴 Sleep & hydration",
            "Log sleep duration and water intake.",
            [
                checkin_form["sleep_hours"],
                checkin_form["water_liters"],
            ],
        ),
        (
            "checkin-movement",
            "🚶 Exercise",
            "Enter exercise duration as hours and extra minutes.",
            [
                checkin_form["exercise_hours"],
                checkin_form["exercise_extra_minutes"],
            ],
        ),
        (
            "checkin-nutrition",
            "🥗 Food variety",
            "Choose the fruits and vegetables you ate today.",
            [
                checkin_form["fruit_vegetable_items"],
            ],
        ),
        (
            "checkin-meals",
            "🍽️ Meals",
            "Tick the meals you had today.",
            [
                checkin_form["breakfast_taken"],
                checkin_form["lunch_taken"],
                checkin_form["dinner_taken"],
            ],
        ),
        (
            "checkin-wellness",
            "💛 Mood & energy",
            "Record how you felt, your stress, energy, and an optional note.",
            [
                checkin_form["mood"],
                checkin_form["stress_level"],
                checkin_form["energy_level"],
                checkin_form["screen_time_hours"],
                checkin_form["notes"],
            ],
        ),
    ]

    context = {
        "daily_actions": build_daily_actions(today_checkin),
        "prediction_profile_form": prediction_profile_form if missing_profile_fields else None,
        "risk_ready_count": risk_ready_count,
        "predictions_calculated_at": predictions_calculated_at,
        "overall_screening": overall_screening_summary(risk_cards),
        "display_name": display_name,
        "profile": profile,
        "recent_checkins": recent_checkins,
        "dashboard_data": dashboard_data,
        "checkin_form": checkin_form,
        "checkin_sections": checkin_sections,
        "bmi_form": bmi_form,
        "bmi_history": bmi_history,
        "latest_bmi": latest_bmi,
        "profile_age": profile_age,
        "today": today,
        "today_checkin": today_checkin,
        "active_checkin": active_checkin,
        "today_medications": today_medications,

        "insights": insights,

        "weekly_average_score": weekly_average_score,
        "weekly_average_sleep": weekly_average_sleep,
        "weekly_average_water": weekly_average_water,
        "weekly_exercise": weekly_exercise,

        "chart_labels": chart_labels,
        "health_scores": health_scores,
        "sleep_values": sleep_values,
        "water_values": water_values,
        "exercise_values": exercise_values,
        "mood_values": mood_values,
        "stress_values": stress_values,

        "latest_diabetes_prediction": latest_diabetes_prediction,

        "diabetes_risk": diabetes_risk,
        "risk_missing_inputs": risk_missing_inputs,
        "risk_cards": risk_cards,
        "coach_question": coach_question,
        "coach_answer": coach_answer,
        "coach_mode": coach_mode,
        "hypertension_risk": hypertension_risk,
        "cardiovascular_risk": cardiovascular_risk,
        "mental_health_risk": mental_health_risk,

        "ai_health_score": ai_health_score,
        "ai_health_category": ai_health_category,
        "ai_health_components": ai_health_components,
        "ai_health_base_score": ai_health_base_score,
        "ai_health_risk_penalty": ai_health_risk_penalty,
        "ai_health_risk_details": ai_health_risk_details,
        "ai_health_mental_source": ai_health_mental_source,
        "ai_health_explanation": ai_health_explanation,
        "today_mental_assessment": today_mental_assessment,
        "mental_assessment_result": mental_assessment_result,

        "habit_results_ai": habit_results_ai,

        "top_recommendations": top_recommendations,
        "recommendation_active_count": recommendation_active_count,
        "recommendation_improving_count": recommendation_improving_count,
        "recommendation_completed_count": recommendation_completed_count,
        "recommendation_total_count": recommendation_total_count,
    }

    # =====================================================
    # RENDER
    # =====================================================

    return render(
        request,
        page_templates[page],
        context,
    )
