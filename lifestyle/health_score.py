def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(value, maximum))


# ============================================================
# V1 COMPONENT SCORES
# ============================================================

def calculate_sleep_score(checkin):

    hours = float(checkin.sleep_hours or 0)
    quality = checkin.sleep_quality

    if 7 <= hours <= 9:
        duration_score = 100

    elif 6 <= hours < 7 or 9 < hours <= 10:
        duration_score = 80

    elif 5 <= hours < 6 or 10 < hours <= 11:
        duration_score = 60

    elif 4 <= hours < 5:
        duration_score = 40

    else:
        duration_score = 20

    if quality is None:
        return round(
            clamp(duration_score),
            1,
        )

    quality_score = clamp(float(quality) * 10)

    score = (
        duration_score * 0.6
        + quality_score * 0.4
    )

    return round(
        clamp(score),
        1,
    )


def calculate_hydration_score(checkin):

    water = float(
        checkin.water_liters or 0
    )

    target = 2.5

    score = (
        water / target
    ) * 100

    return round(
        clamp(score),
        1,
    )


def calculate_exercise_score(checkin):
    """Score reported exercise without requiring a device-based step count."""
    return round(clamp(float(checkin.exercise_minutes or 0) / 30 * 100), 1)


def calculate_mental_score(checkin):

    mood_score = (
        float(checkin.mood or 0)
        * 20
    )

    stress_score = (
        6
        - float(
            checkin.stress_level or 0
        )
    ) * 20

    energy_score = (
        float(
            checkin.energy_level or 0
        )
        * 20
    )

    score = (
        mood_score * 0.4
        + stress_score * 0.35
        + energy_score * 0.25
    )

    return round(
        clamp(score),
        1,
    )


def calculate_nutrition_score(checkin):

    meal_score = 0

    if checkin.breakfast_taken:
        meal_score += 20

    if checkin.lunch_taken:
        meal_score += 20

    if checkin.dinner_taken:
        meal_score += 20

    fruit_veg_score = min(
        (
            float(
                checkin.fruit_vegetable_servings
                or 0
            )
            / 5
        )
        * 40,
        40,
    )

    score = (
        meal_score
        + fruit_veg_score
    )

    return round(
        clamp(score),
        1,
    )


def calculate_screen_score(checkin):

    hours = float(
        checkin.screen_time_hours or 0
    )

    if hours <= 4:
        score = 100

    elif hours <= 6:
        score = 80

    elif hours <= 8:
        score = 60

    elif hours <= 10:
        score = 40

    else:
        score = 20

    return score


# ============================================================
# EXISTING DAILY HEALTH SCORE
#
# Keep this function unchanged in purpose so older pages and
# views that already call it continue working.
# ============================================================

def calculate_daily_health_score(checkin):

    sleep = calculate_sleep_score(
        checkin
    )

    hydration = (
        calculate_hydration_score(
            checkin
        )
    )

    exercise = (
        calculate_exercise_score(
            checkin
        )
    )

    mental = calculate_mental_score(
        checkin
    )

    nutrition = (
        calculate_nutrition_score(
            checkin
        )
    )

    screen = calculate_screen_score(
        checkin
    )

    overall = (
        sleep * 0.20
        + exercise * 0.20
        + mental * 0.20
        + nutrition * 0.15
        + hydration * 0.15
        + screen * 0.10
    )

    return {
        "overall":
            round(overall, 1),

        "sleep":
            sleep,

        "hydration":
            hydration,

        "exercise":
            exercise,

        "mental":
            mental,

        "nutrition":
            nutrition,

        "screen":
            screen,
    }


# ============================================================
# HEALTH SCORE CATEGORY
# ============================================================

def get_health_score_category(score):

    if score >= 85:
        return "Excellent"

    if score >= 70:
        return "Good"

    if score >= 55:
        return "Fair"

    if score >= 40:
        return "Needs Attention"

    return "Poor"


# ============================================================
# V2 - PHYSICAL LIFESTYLE SCORE
#
# This excludes the DailyCheckIn mental component so that a
# richer MentalWellnessAssessment can be used without
# double-counting mood/stress/energy.
# ============================================================

def calculate_physical_lifestyle_score(
    checkin,
):

    sleep = calculate_sleep_score(
        checkin
    )

    hydration = (
        calculate_hydration_score(
            checkin
        )
    )

    exercise = (
        calculate_exercise_score(
            checkin
        )
    )

    nutrition = (
        calculate_nutrition_score(
            checkin
        )
    )

    screen = calculate_screen_score(
        checkin
    )

    # Original non-mental weights total 0.80.
    # Normalize back to a 0-100 scale.
    weighted_sum = (
        sleep * 0.20
        + exercise * 0.20
        + nutrition * 0.15
        + hydration * 0.15
        + screen * 0.10
    )

    normalized = (
        weighted_sum / 0.80
    )

    return round(
        clamp(normalized),
        1,
    )


# ============================================================
# V2 - DEDICATED MENTAL WELLNESS SCORE
# ============================================================

def calculate_mental_assessment_score(
    assessment,
):

    if not assessment:
        return None

    mood = (
        float(
            assessment.mood or 0
        )
        * 20
    )

    stress = (
        6
        - float(
            assessment.stress or 0
        )
    ) * 20

    anxiety = (
        6
        - float(
            assessment.anxiety or 0
        )
    ) * 20

    energy = (
        float(
            assessment.energy or 0
        )
        * 20
    )

    motivation = (
        float(
            assessment.motivation or 0
        )
        * 20
    )

    concentration = (
        float(
            assessment.concentration or 0
        )
        * 20
    )

    social = (
        float(
            assessment.social_wellbeing
            or 0
        )
        * 20
    )

    workload = (
        6
        - float(
            assessment.workload or 0
        )
    ) * 20

    # Equal weighting keeps the scale transparent.
    values = [
        mood,
        stress,
        anxiety,
        energy,
        motivation,
        concentration,
        social,
        workload,
    ]

    score = (
        sum(values)
        / len(values)
    )

    return round(
        clamp(score),
        1,
    )


# ============================================================
# V2 - ML RISK MODIFIER
#
# IMPORTANT:
# These ML probabilities are not treated as health-quality
# scores. They only apply a bounded penalty when a model's own
# selected decision threshold is crossed.
#
# This is a project-level heuristic, not a clinical score.
# ============================================================

RISK_MODEL_MAX_PENALTIES = {
    "diabetes": 3.0,
    "hypertension": 3.0,
    "cardiovascular": 4.0,
    "mental_health": 5.0,
}


def calculate_single_risk_penalty(
    result,
    max_penalty,
):

    if not result:
        return 0.0

    if (
        result.get(
            "prediction_available"
        )
        is False
    ):
        return 0.0

    # Only penalize when the model's own selected
    # classification threshold is crossed.
    if result.get("prediction") != 1:
        return 0.0

    probability = float(
        result.get(
            "probability",
            0,
        )
        or 0
    )

    threshold = float(
        result.get(
            "threshold",
            0.5,
        )
        or 0.5
    )

    # The farther above the decision threshold,
    # the larger the penalty, capped at max_penalty.
    denominator = max(
        1.0 - threshold,
        0.0001,
    )

    severity = (
        probability
        - threshold
    ) / denominator

    severity = clamp(
        severity,
        0,
        1,
    )

    # Even a threshold-crossing result should have
    # some visible influence, so begin at 40% of
    # the model's allowed maximum.
    scaled = (
        0.40
        + 0.60 * severity
    )

    return round(
        max_penalty * scaled,
        2,
    )


def calculate_ml_risk_modifier(
    health_risks,
):

    health_risks = (
        health_risks
        or {}
    )

    penalties = {}

    total_penalty = 0.0

    for (
        model_name,
        max_penalty,
    ) in (
        RISK_MODEL_MAX_PENALTIES
        .items()
    ):

        result = (
            health_risks.get(
                model_name
            )
        )

        penalty = (
            calculate_single_risk_penalty(
                result,
                max_penalty,
            )
        )

        penalties[
            model_name
        ] = penalty

        total_penalty += penalty

    # Absolute safety cap.
    total_penalty = min(
        total_penalty,
        15.0,
    )

    return {
        "total_penalty":
            round(
                total_penalty,
                2,
            ),

        "penalties":
            penalties,

        "max_possible_penalty":
            15.0,
    }


# ============================================================
# V2 - AI HEALTH SCORE
# ============================================================

def calculate_ai_health_score(
    checkin,
    mental_assessment=None,
    health_risks=None,
):

    if not checkin:

        return {
            "available":
                False,

            "overall":
                None,

            "category":
                "No Data",

            "base_score":
                None,

            "risk_penalty":
                0,

            "components":
                {},

            "risk_details":
                {},

            "mental_source":
                None,
        }

    physical_score = (
        calculate_physical_lifestyle_score(
            checkin
        )
    )

    dedicated_mental = (
        calculate_mental_assessment_score(
            mental_assessment
        )
    )

    if dedicated_mental is not None:

        mental_score = (
            dedicated_mental
        )

        mental_source = (
            "mental_wellness_assessment"
        )

    else:

        mental_score = (
            calculate_mental_score(
                checkin
            )
        )

        mental_source = (
            "daily_checkin"
        )

    # 75% physical lifestyle + 25% mental wellness.
    # ML results are handled separately as a bounded modifier.
    base_score = (
        physical_score * 0.75
        + mental_score * 0.25
    )

    risk_modifier = (
        calculate_ml_risk_modifier(
            health_risks
        )
    )

    risk_penalty = (
        risk_modifier[
            "total_penalty"
        ]
    )

    overall = clamp(
        base_score
        - risk_penalty
    )

    overall = round(
        overall,
        1,
    )

    return {
        "available":
            True,

        "overall":
            overall,

        "category":
            get_health_score_category(
                overall
            ),

        "base_score":
            round(
                base_score,
                1,
            ),

        "risk_penalty":
            risk_penalty,

        "components": {
            "physical_lifestyle":
                physical_score,

            "mental_wellness":
                mental_score,

            "sleep":
                calculate_sleep_score(
                    checkin
                ),

            "hydration":
                calculate_hydration_score(
                    checkin
                ),

            "exercise":
                calculate_exercise_score(
                    checkin
                ),

            "nutrition":
                calculate_nutrition_score(
                    checkin
                ),

            "screen":
                calculate_screen_score(
                    checkin
                ),
        },

        "risk_details":
            risk_modifier[
                "penalties"
            ],

        "mental_source":
            mental_source,

        "explanation": (
            "The AI Health Score combines current lifestyle "
            "and mental-wellness indicators. Machine-learning "
            "screening results are used only as bounded risk "
            "modifiers and are not diagnoses."
        ),
    }
