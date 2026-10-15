# ============================================================
# LIFESYNC AI PRO
# AI HEALTH SCORE ENGINE
# ============================================================


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(
    value,
    minimum=0,
    maximum=100,
):
    """
    Keep a numeric score between 0 and 100.
    """

    try:
        value = float(value)
    except (TypeError, ValueError):
        return minimum

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


def safe_float(
    value,
    default=0.0,
):
    """
    Safely convert values to float.
    """

    if value in [None, ""]:
        return default

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# SLEEP SCORE
# ============================================================

def calculate_sleep_score(
    checkin,
):

    if not checkin:
        return 0.0

    hours = safe_float(
        checkin.sleep_hours
    )

    quality = safe_float(
        checkin.sleep_quality
    )

    # ----------------------------------------
    # Sleep duration
    # ----------------------------------------

    if 7 <= hours <= 9:
        duration_score = 100

    elif (
        6 <= hours < 7
        or
        9 < hours <= 10
    ):
        duration_score = 80

    elif (
        5 <= hours < 6
        or
        10 < hours <= 11
    ):
        duration_score = 60

    elif 4 <= hours < 5:
        duration_score = 40

    else:
        duration_score = 20

    # ----------------------------------------
    # Sleep quality
    # ----------------------------------------

    quality_score = clamp(
        quality * 10
    )

    score = (
        duration_score * 0.60
        +
        quality_score * 0.40
    )

    return round(
        clamp(score),
        1,
    )


# ============================================================
# HYDRATION SCORE
# ============================================================

def calculate_hydration_score(
    checkin,
):

    if not checkin:
        return 0.0

    water = safe_float(
        checkin.water_liters
    )

    target = 2.5

    score = (
        water / target
    ) * 100

    return round(
        clamp(score),
        1,
    )


# ============================================================
# EXERCISE SCORE
# ============================================================

def calculate_exercise_score(
    checkin,
):

    if not checkin:
        return 0.0

    minutes = safe_float(
        checkin.exercise_minutes
    )

    steps = safe_float(
        checkin.steps
    )

    exercise_score = min(
        (
            minutes / 30
        )
        * 100,
        100,
    )

    steps_score = min(
        (
            steps / 8000
        )
        * 100,
        100,
    )

    score = (
        exercise_score * 0.60
        +
        steps_score * 0.40
    )

    return round(
        clamp(score),
        1,
    )


# ============================================================
# DAILY MENTAL SCORE
# ============================================================

def calculate_mental_score(
    checkin,
):

    if not checkin:
        return 0.0

    mood = safe_float(
        checkin.mood
    )

    stress = safe_float(
        checkin.stress_level
    )

    energy = safe_float(
        checkin.energy_level
    )

    mood_score = clamp(
        mood * 20
    )

    stress_score = clamp(
        (
            6 - stress
        )
        * 20
    )

    energy_score = clamp(
        energy * 20
    )

    score = (
        mood_score * 0.40
        +
        stress_score * 0.35
        +
        energy_score * 0.25
    )

    return round(
        clamp(score),
        1,
    )


# ============================================================
# NUTRITION SCORE
# ============================================================

def calculate_nutrition_score(
    checkin,
):

    if not checkin:
        return 0.0

    meal_score = 0

    if checkin.breakfast_taken:
        meal_score += 20

    if checkin.lunch_taken:
        meal_score += 20

    if checkin.dinner_taken:
        meal_score += 20

    fruit_veg = safe_float(
        checkin.fruit_vegetable_servings
    )

    fruit_veg_score = min(
        (
            fruit_veg / 5
        )
        * 40,
        40,
    )

    score = (
        meal_score
        +
        fruit_veg_score
    )

    return round(
        clamp(score),
        1,
    )


# ============================================================
# SCREEN TIME SCORE
# ============================================================

def calculate_screen_score(
    checkin,
):

    if not checkin:
        return 0.0

    hours = safe_float(
        checkin.screen_time_hours
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

    return round(
        clamp(score),
        1,
    )


# ============================================================
# DAILY HEALTH SCORE
# ============================================================

def calculate_daily_health_score(
    checkin,
):

    if not checkin:

        return {
            "overall": 0.0,
            "sleep": 0.0,
            "hydration": 0.0,
            "exercise": 0.0,
            "mental": 0.0,
            "nutrition": 0.0,
            "screen": 0.0,
        }

    sleep = calculate_sleep_score(
        checkin
    )

    hydration = calculate_hydration_score(
        checkin
    )

    exercise = calculate_exercise_score(
        checkin
    )

    mental = calculate_mental_score(
        checkin
    )

    nutrition = calculate_nutrition_score(
        checkin
    )

    screen = calculate_screen_score(
        checkin
    )

    overall = (
        sleep * 0.20
        +
        exercise * 0.20
        +
        mental * 0.20
        +
        nutrition * 0.15
        +
        hydration * 0.15
        +
        screen * 0.10
    )

    return {
        "overall":
            round(
                clamp(overall),
                1,
            ),

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

def get_health_score_category(
    score,
):

    score = safe_float(
        score
    )

    if score >= 85:
        return "Excellent"

    if score >= 70:
        return "Good"

    if score >= 55:
        return "Fair"

    if score >= 40:
        return "Needs Attention"

    if score > 0:
        return "Needs Improvement"

    return "No Data"


# ============================================================
# RECENT LIFESTYLE SCORE
# ============================================================

def calculate_lifestyle_component(
    checkins,
):

    if not checkins:
        return None

    scores = []

    for checkin in checkins:

        daily = calculate_daily_health_score(
            checkin
        )

        scores.append(
            daily["overall"]
        )

    if not scores:
        return None

    return round(
        sum(scores)
        / len(scores),
        1,
    )


# ============================================================
# MENTAL WELLNESS ASSESSMENT SCORE
# ============================================================

def calculate_mental_assessment_score(
    assessment,
):

    if not assessment:
        return None

    # ----------------------------------------
    # Positive indicators
    # Higher = better
    # ----------------------------------------

    mood = clamp(
        safe_float(
            assessment.mood
        )
        * 20
    )

    energy = clamp(
        safe_float(
            assessment.energy
        )
        * 20
    )

    motivation = clamp(
        safe_float(
            assessment.motivation
        )
        * 20
    )

    concentration = clamp(
        safe_float(
            assessment.concentration
        )
        * 20
    )

    social = clamp(
        safe_float(
            assessment.social_wellbeing
        )
        * 20
    )

    # ----------------------------------------
    # Negative indicators
    # Lower = better
    # ----------------------------------------

    stress = clamp(
        (
            6
            -
            safe_float(
                assessment.stress
            )
        )
        * 20
    )

    anxiety = clamp(
        (
            6
            -
            safe_float(
                assessment.anxiety
            )
        )
        * 20
    )

    workload = clamp(
        (
            6
            -
            safe_float(
                assessment.workload
            )
        )
        * 20
    )

    score = (
        mood * 0.15
        +
        stress * 0.15
        +
        anxiety * 0.15
        +
        energy * 0.15
        +
        motivation * 0.10
        +
        concentration * 0.10
        +
        social * 0.10
        +
        workload * 0.10
    )

    return round(
        clamp(score),
        1,
    )


def calculate_mental_component(
    assessments,
    checkins=None,
):

    # ----------------------------------------
    # Prefer dedicated mental assessments
    # ----------------------------------------

    if assessments:

        scores = []

        for assessment in assessments:

            score = (
                calculate_mental_assessment_score(
                    assessment
                )
            )

            if score is not None:

                scores.append(
                    score
                )

        if scores:

            return round(
                sum(scores)
                / len(scores),
                1,
            )

    # ----------------------------------------
    # Fallback to DailyCheckIn
    # ----------------------------------------

    if checkins:

        scores = [
            calculate_mental_score(
                checkin
            )
            for checkin in checkins
        ]

        if scores:

            return round(
                sum(scores)
                / len(scores),
                1,
            )

    return None


# ============================================================
# HABIT ADHERENCE
# ============================================================

def calculate_habit_component(
    habit_results,
):

    if not habit_results:
        return None

    total = len(
        habit_results
    )

    if total == 0:
        return None

    achieved = sum(
        1
        for result in habit_results
        if result.get(
            "achieved",
            False,
        )
    )

    score = (
        achieved
        / total
    ) * 100

    return round(
        clamp(score),
        1,
    )


# ============================================================
# RECOVERY COMPONENT
# ============================================================

def calculate_recovery_component(
    checkins,
):

    if not checkins:
        return None

    recovery_scores = []

    for checkin in checkins:

        sleep = calculate_sleep_score(
            checkin
        )

        energy = clamp(
            safe_float(
                checkin.energy_level
            )
            * 20
        )

        stress = clamp(
            (
                6
                -
                safe_float(
                    checkin.stress_level
                )
            )
            * 20
        )

        recovery = (
            sleep * 0.50
            +
            energy * 0.30
            +
            stress * 0.20
        )

        recovery_scores.append(
            recovery
        )

    if not recovery_scores:
        return None

    return round(
        clamp(
            sum(recovery_scores)
            / len(recovery_scores)
        ),
        1,
    )


# ============================================================
# CONSISTENCY COMPONENT
# ============================================================

def calculate_consistency_component(
    checkins,
    target_days=7,
):

    if not checkins:
        return None

    unique_dates = set()

    for checkin in checkins:

        if getattr(
            checkin,
            "date",
            None,
        ):

            unique_dates.add(
                checkin.date
            )

    recorded_days = len(
        unique_dates
    )

    score = (
        recorded_days
        / target_days
    ) * 100

    return round(
        clamp(score),
        1,
    )


# ============================================================
# ML RISK HELPERS
# ============================================================

def extract_risk_probability(
    result,
):
    """
    Extract a probability from one live ML result.

    Returns None when prediction is unavailable.
    """

    if not result:
        return None

    if (
        result.get(
            "prediction_available"
        )
        is False
    ):
        return None

    probability = result.get(
        "probability"
    )

    if probability is None:
        return None

    probability = safe_float(
        probability,
        default=None,
    )

    if probability is None:
        return None

    return max(
        0.0,
        min(
            probability,
            1.0,
        ),
    )


def probability_to_health_score(
    probability,
):
    """
    Convert risk probability into a health-oriented score.

    Example:
    10% risk -> 90 health-risk score
    70% risk -> 30 health-risk score

    This is only an internal wellness-score component.
    """

    if probability is None:
        return None

    score = (
        1.0
        -
        probability
    ) * 100

    return round(
        clamp(score),
        1,
    )


# ============================================================
# LEGACY PREDICTION SUPPORT
# ============================================================

def extract_legacy_prediction_probability(
    prediction,
):

    if not prediction:
        return None

    probability = getattr(
        prediction,
        "probability",
        None,
    )

    if probability is None:
        return None

    probability = safe_float(
        probability,
        default=None,
    )

    if probability is None:
        return None

    return max(
        0.0,
        min(
            probability,
            1.0,
        ),
    )


# ============================================================
# COMBINED ML RISK COMPONENT
# ============================================================

def calculate_risk_component(
    risk_predictions=None,
    latest_prediction=None,
):

    risk_scores = []

    # ----------------------------------------
    # New multi-model predictions
    # ----------------------------------------

    if risk_predictions:

        for model_name in [
            "diabetes",
            "hypertension",
            "cardiovascular",
        ]:

            result = risk_predictions.get(
                model_name
            )

            probability = (
                extract_risk_probability(
                    result
                )
            )

            health_score = (
                probability_to_health_score(
                    probability
                )
            )

            if health_score is not None:

                risk_scores.append(
                    health_score
                )

    # ----------------------------------------
    # Backward compatibility
    # ----------------------------------------

    if (
        not risk_scores
        and latest_prediction
    ):

        probability = (
            extract_legacy_prediction_probability(
                latest_prediction
            )
        )

        health_score = (
            probability_to_health_score(
                probability
            )
        )

        if health_score is not None:

            risk_scores.append(
                health_score
            )

    if not risk_scores:
        return None

    return round(
        sum(risk_scores)
        / len(risk_scores),
        1,
    )


# ============================================================
# WEIGHTED AI SCORE
# ============================================================

def weighted_available_average(
    components,
    weights,
):
    """
    Only components that actually have data participate.

    Their remaining weights are re-normalized so missing
    optional data does not automatically reduce the score.
    """

    weighted_total = 0.0
    active_weight = 0.0

    for name, weight in weights.items():

        value = components.get(
            name
        )

        if value is None:
            continue

        weighted_total += (
            value
            * weight
        )

        active_weight += weight

    if active_weight <= 0:
        return 0.0

    return round(
        weighted_total
        / active_weight,
        1,
    )


# ============================================================
# MAIN AI HEALTH SCORE
# ============================================================

def calculate_ai_health_score(
    checkins=None,
    assessments=None,
    habit_results=None,
    risk_predictions=None,
    latest_prediction=None,
):
    """
    LifeSync AI Health Intelligence Score.

    This is a wellness/index score and NOT a medical diagnosis.

    Components:
        Lifestyle
        Mental wellness
        Habit adherence
        Recovery
        Consistency
        ML risk context

    Missing components are excluded and remaining weights
    are automatically re-normalized.
    """

    checkins = (
        checkins
        or []
    )

    assessments = (
        assessments
        or []
    )

    habit_results = (
        habit_results
        or []
    )

    risk_predictions = (
        risk_predictions
        or {}
    )

    # ----------------------------------------
    # Components
    # ----------------------------------------

    lifestyle = (
        calculate_lifestyle_component(
            checkins
        )
    )

    mental = (
        calculate_mental_component(
            assessments,
            checkins,
        )
    )

    habits = (
        calculate_habit_component(
            habit_results
        )
    )

    recovery = (
        calculate_recovery_component(
            checkins
        )
    )

    consistency = (
        calculate_consistency_component(
            checkins
        )
    )

    risk = (
        calculate_risk_component(
            risk_predictions=
                risk_predictions,

            latest_prediction=
                latest_prediction,
        )
    )

    components = {
        "lifestyle":
            lifestyle,

        "mental":
            mental,

        "habits":
            habits,

        "recovery":
            recovery,

        "consistency":
            consistency,

        "risk":
            risk,
    }

    # ----------------------------------------
    # Component weights
    # ----------------------------------------

    weights = {
        "lifestyle":
            0.35,

        "mental":
            0.15,

        "habits":
            0.10,

        "recovery":
            0.15,

        "consistency":
            0.10,

        "risk":
            0.15,
    }

    overall = (
        weighted_available_average(
            components,
            weights,
        )
    )

    category = (
        get_health_score_category(
            overall
        )
    )

    available_components = [
        name
        for name, value
        in components.items()
        if value is not None
    ]

    missing_components = [
        name
        for name, value
        in components.items()
        if value is None
    ]

    completeness = (
        (
            len(
                available_components
            )
            /
            len(
                components
            )
        )
        * 100
        if components
        else 0
    )

    return {
        "overall":
            overall,

        "category":
            category,

        "components":
            components,

        "available_components":
            available_components,

        "missing_components":
            missing_components,

        "completeness_percentage":
            round(
                completeness,
                1,
            ),
    }