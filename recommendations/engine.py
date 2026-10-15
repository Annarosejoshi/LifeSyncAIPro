def generate_recommendations(
    checkin,
    mental_assessment=None,
    health_risks=None,
    profile=None,
    weekly_trends=None,
):

    recommendations = []

    health_risks = health_risks or {}

    # Sleep
    if checkin.sleep_hours < 6:
        recommendations.append({
            "category": "sleep",
            "title": "Prioritize Sleep",
            "message": (
                "Your recorded sleep is quite low. "
                "Try moving toward a more consistent sleep schedule "
                "and gradually increasing sleep duration."
            ),
            "priority": "high",
        })

    elif checkin.sleep_hours < 7:
        recommendations.append({
            "category": "sleep",
            "title": "Improve Sleep Duration",
            "message": (
                "Your sleep duration is below the general target range. "
                "Consider going to bed a little earlier tonight."
            ),
            "priority": "medium",
        })

    # Hydration
    if checkin.water_liters < 1.5:
        recommendations.append({
            "category": "hydration",
            "title": "Increase Hydration",
            "message": (
                "Your recorded water intake is relatively low. "
                "Increase fluids gradually across the day."
            ),
            "priority": "high",
        })

    elif checkin.water_liters < 2.0:
        recommendations.append({
            "category": "hydration",
            "title": "Hydration Goal",
            "message": (
                "Try adding a few extra glasses of water "
                "throughout the day."
            ),
            "priority": "medium",
        })

    # Exercise
    if checkin.exercise_minutes < 15:
        recommendations.append({
            "category": "exercise",
            "title": "Add More Movement",
            "message": (
                "You recorded very little exercise today. "
                "A short walk or another suitable activity "
                "could improve your activity level."
            ),
            "priority": "medium",
        })

    elif checkin.exercise_minutes < 30:
        recommendations.append({
            "category": "exercise",
            "title": "Reach Your Activity Target",
            "message": (
                "You are close to a useful daily activity target. "
                "Consider adding another short movement session."
            ),
            "priority": "low",
        })

    # Nutrition
    if checkin.fruit_vegetable_servings < 3:
        recommendations.append({
            "category": "nutrition",
            "title": "Improve Food Variety",
            "message": (
                "Your fruit and vegetable intake is low today. "
                "Try including more whole fruits and vegetables "
                "in upcoming meals."
            ),
            "priority": "medium",
        })

    # Screen time
    if checkin.screen_time_hours > 8:
        recommendations.append({
            "category": "screen",
            "title": "Reduce Screen Time",
            "message": (
                "Your screen time is high today. "
                "Consider short screen-free breaks and reducing "
                "late-evening device use."
            ),
            "priority": "medium",
        })

    # Stress
    if checkin.stress_level >= 4:
        recommendations.append({
            "category": "mental",
            "title": "Stress Recovery",
            "message": (
                "Your reported stress is elevated. "
                "Try a short relaxation, breathing, stretching, "
                "or quiet recovery period."
            ),
            "priority": "high",
        })

    # Mood
    if checkin.mood <= 2:
        recommendations.append({
            "category": "mental",
            "title": "Support Your Mood",
            "message": (
                "Your mood score is lower today. "
                "Consider restorative activities, social connection, "
                "or taking time away from stressful tasks."
            ),
            "priority": "medium",
        })

    # Energy
    if checkin.energy_level <= 2:
        recommendations.append({
            "category": "general",
            "title": "Low Energy Detected",
            "message": (
                "Your energy score is low today. "
                "Review sleep, hydration, food intake, "
                "and workload before pushing activity harder."
            ),
            "priority": "medium",
        })

    # Mental wellness assessment
    if mental_assessment:

        if mental_assessment.anxiety >= 4:
            recommendations.append({
                "category": "mental",
                "title": "Mental Recovery Break",
                "message": (
                    "You reported elevated anxious feelings. "
                    "Consider a quiet break, structured breathing, "
                    "or another calming activity."
                ),
                "priority": "high",
            })

        if mental_assessment.workload >= 4:
            recommendations.append({
                "category": "mental",
                "title": "Manage Workload",
                "message": (
                    "Your workload score is high. "
                    "Break large tasks into smaller blocks "
                    "and schedule recovery periods."
                ),
                "priority": "medium",
            })

    if not recommendations:
        recommendations.append({
            "category": "general",
            "title": "Keep Up Your Routine",
            "message": (
                "Your recorded lifestyle indicators are "
                "generally balanced today. Focus on consistency."
            ),
            "priority": "low",
        })
        
            # ========================================================
    # ML HEALTH-RISK RECOMMENDATIONS
    # ========================================================

    health_risks = health_risks or {}

    # --------------------------------------------------------
    # DIABETES
    # --------------------------------------------------------

    diabetes = health_risks.get("diabetes", {})

    if (
        diabetes.get("prediction_available")
        and diabetes.get("prediction") == 1
    ):
        recommendations.append({
            "category": "Metabolic Health",
            "icon": "🩸",
            "priority": "high",
            "title": "Pay Attention to Metabolic Health",
            "message": (
                "Your current screening model shows an elevated "
                "diabetes-risk signal. Regular physical activity, "
                "balanced meals, healthy body weight, and routine "
                "health monitoring can support metabolic health."
            ),
        })

    # --------------------------------------------------------
    # HYPERTENSION
    # --------------------------------------------------------

    hypertension = health_risks.get(
        "hypertension",
        {},
    )

    if (
        hypertension.get("prediction_available")
        and hypertension.get("prediction") == 1
    ):
        recommendations.append({
            "category": "Blood Pressure",
            "icon": "❤️",
            "priority": "high",
            "title": "Support Healthy Blood Pressure",
            "message": (
                "Your screening model shows an elevated "
                "hypertension-risk signal. Regular activity, "
                "adequate sleep, balanced nutrition and periodic "
                "blood-pressure monitoring are useful habits."
            ),
        })

    # --------------------------------------------------------
    # CARDIOVASCULAR
    # --------------------------------------------------------

    cardiovascular = health_risks.get(
        "cardiovascular",
        {},
    )

    if (
        cardiovascular.get("prediction_available")
        and cardiovascular.get("prediction") == 1
    ):
        recommendations.append({
            "category": "Heart Health",
            "icon": "🫀",
            "priority": "high",
            "title": "Focus on Cardiovascular Health",
            "message": (
                "Your current screening model indicates an elevated "
                "cardiovascular-risk signal. Staying physically active, "
                "avoiding smoking and monitoring blood pressure and "
                "cholesterol can support cardiovascular health."
            ),
        })

    # --------------------------------------------------------
    # MENTAL WELLNESS
    # --------------------------------------------------------

    mental_risk = health_risks.get(
        "mental_health",
        {},
    )

    if (
        mental_risk.get("prediction_available")
        and mental_risk.get("prediction") == 1
    ):
        recommendations.append({
            "category": "Mental Wellness",
            "icon": "🧠",
            "priority": "high",
            "title": "Give Extra Attention to Mental Wellness",
            "message": (
                "Your current wellness screening shows an elevated "
                "mental-wellness signal. Prioritize consistent sleep, "
                "regular activity, stress-management habits and "
                "supportive social connection. Consider speaking with "
                "a qualified professional if concerns persist or "
                "interfere with daily life."
            ),
        })

    # ========================================================
    # PROFILE-BASED RECOMMENDATIONS
    # ========================================================

    if profile:

        bmi = getattr(
            profile,
            "bmi",
            None,
        )

        if bmi is not None and bmi >= 30:

            recommendations.append({
                "category": "Weight Management",
                "icon": "⚖️",
                "priority": "medium",
                "title": "Work Toward Healthy Weight Habits",
                "message": (
                    "Your BMI is currently in a higher range. "
                    "Gradual improvements in physical activity, "
                    "food quality, sleep and daily routines can "
                    "support long-term weight management."
                ),
            })

        smoking_status = getattr(
            profile,
            "smoking_status",
            None,
        )

        if smoking_status == "current":

            recommendations.append({
                "category": "Smoking",
                "icon": "🚭",
                "priority": "high",
                "title": "Reducing Smoking Can Improve Health",
                "message": (
                    "Smoking is an important modifiable health-risk "
                    "factor. Reducing and ideally stopping smoking "
                    "can improve cardiovascular and overall health."
                ),
            })

    # ========================================================
    # WEEKLY TREND RECOMMENDATIONS (V2)
    # ========================================================

    weekly_trends = weekly_trends or {}

    # Trend input is intentionally optional so existing views continue
    # working unchanged. Expected example:
    #
    # {
    #     "score_trend": "declining",
    #     "score_difference": -6.4,
    #     "avg_sleep": 5.9,
    #     "avg_water": 1.4,
    #     "avg_exercise": 12.0,
    #     "avg_stress": 4.1,
    #     "weakest_component": "exercise",
    # }
    #
    # These rules describe patterns only; they do not diagnose disease.

    score_trend = weekly_trends.get("score_trend")
    score_difference = weekly_trends.get("score_difference")

    if score_trend == "declining":
        difference_text = ""
        if score_difference is not None:
            difference_text = (
                f" by {abs(float(score_difference)):.1f} points"
            )

        recommendations.append({
            "category": "Weekly Trend",
            "icon": "📉",
            "priority": "high",
            "title": "Your Recent Health Trend Needs Attention",
            "message": (
                f"Your AI Health Score has declined{difference_text} "
                "across the recent recorded period. Focus first on the "
                "lowest-scoring lifestyle areas and aim for small, "
                "consistent improvements over the next few days."
            ),
        })

    avg_sleep = weekly_trends.get("avg_sleep")
    if avg_sleep is not None and float(avg_sleep) < 6:
        recommendations.append({
            "category": "Sleep Trend",
            "icon": "🌙",
            "priority": "high",
            "title": "Low Sleep Has Continued Across the Week",
            "message": (
                "Your recent average sleep has remained below six hours. "
                "Prioritize a consistent sleep schedule and gradually "
                "increase sleep duration."
            ),
        })

    avg_water = weekly_trends.get("avg_water")
    if avg_water is not None and float(avg_water) < 1.5:
        recommendations.append({
            "category": "Hydration Trend",
            "icon": "💧",
            "priority": "medium",
            "title": "Hydration Has Been Consistently Low",
            "message": (
                "Your recent average water intake has remained relatively "
                "low. Spread fluid intake across the day rather than trying "
                "to catch up all at once."
            ),
        })

    avg_exercise = weekly_trends.get("avg_exercise")
    if avg_exercise is not None and float(avg_exercise) < 15:
        recommendations.append({
            "category": "Activity Trend",
            "icon": "🏃",
            "priority": "high",
            "title": "Activity Has Remained Low",
            "message": (
                "Your recent exercise pattern has remained low. Start with "
                "short, manageable movement sessions and build consistency "
                "before increasing duration or intensity."
            ),
        })

    avg_stress = weekly_trends.get("avg_stress")
    if avg_stress is not None and float(avg_stress) >= 4:
        recommendations.append({
            "category": "Stress Trend",
            "icon": "🧘",
            "priority": "high",
            "title": "Stress Has Remained Elevated",
            "message": (
                "Your reported stress has stayed elevated across recent "
                "check-ins. Protect recovery time, reduce avoidable workload "
                "where possible, and use calming or restorative routines."
            ),
        })

    weakest_component = weekly_trends.get("weakest_component")
    if weakest_component:
        display_name = str(weakest_component).replace("_", " ").title()

        recommendations.append({
            "category": "Weekly Focus",
            "icon": "🎯",
            "priority": "medium",
            "title": f"Weekly Focus: {display_name}",
            "message": (
                f"{display_name} is currently one of your weakest recent "
                "health-score areas. Make this a primary focus for the next "
                "few days while maintaining the habits that are already "
                "working well."
            ),
        })

    # ========================================================
    # SORT BY PRIORITY
    # ========================================================

    priority_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    recommendations.sort(
        key=lambda item: priority_order.get(
            item.get("priority", "low"),
            3,
        )
    )

    return recommendations