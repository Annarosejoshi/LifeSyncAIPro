from django.utils import timezone

from lifestyle.models import DailyCheckIn
from mental_health.models import MentalWellnessAssessment

from .models import RecommendationHistory
from .services import (
    calculate_mental_wellness_score,
    find_mental_baseline_for_date,
)


# ============================================================
# DAILY CHECK-IN METRIC CONFIGURATION
# ============================================================

METRIC_CONFIG = {
    "sleep": {
        "field": "sleep_hours",
        "direction": "higher",
        "completion_target": 7.0,
        "improvement_delta": 0.5,
    },

    "water": {
        "field": "water_liters",
        "direction": "higher",
        "completion_target": 2.0,
        "improvement_delta": 0.3,
    },

    "exercise": {
        "field": "exercise_minutes",
        "direction": "higher",
        "completion_target": 30.0,
        "improvement_delta": 5.0,
    },

    "steps": {
        "field": "steps",
        "direction": "higher",
        "completion_target": 8000.0,
        "improvement_delta": 500.0,
    },

    "nutrition": {
        "field": "fruit_vegetable_servings",
        "direction": "higher",
        "completion_target": 5.0,
        "improvement_delta": 1.0,
    },

    "screen_time": {
        "field": "screen_time_hours",
        "direction": "lower",
        "completion_target": 6.0,
        "improvement_delta": 1.0,
    },

    "stress": {
        "field": "stress_level",
        "direction": "lower",
        "completion_target": 3.0,
        "improvement_delta": 1.0,
    },

    "mood": {
        "field": "mood",
        "direction": "higher",
        "completion_target": 4.0,
        "improvement_delta": 1.0,
    },

    "energy": {
        "field": "energy_level",
        "direction": "higher",
        "completion_target": 4.0,
        "improvement_delta": 1.0,
    },
}


# ============================================================
# MENTAL WELLNESS CONFIGURATION
# ============================================================

MENTAL_WELLNESS_CONFIG = {
    "direction": "higher",

    # Improvement of at least 10 wellness-score points
    "improvement_delta": 10.0,

    # Project-level target, not a medical threshold
    "completion_target": 75.0,
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    """
    Convert a value to float safely.
    """

    if value is None:
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def calculate_percentage_change(
    baseline,
    latest,
    direction,
):
    """
    Calculate improvement percentage.

    For higher-is-better metrics:
        larger values = positive improvement

    For lower-is-better metrics:
        smaller values = positive improvement
    """

    baseline = safe_float(baseline)
    latest = safe_float(latest)

    if baseline is None or latest is None:
        return None

    if baseline == 0:

        if latest == 0:
            return 0.0

        return 100.0

    change = (
        (latest - baseline)
        / abs(baseline)
        * 100
    )

    if direction == "lower":
        change *= -1

    return round(change, 1)


def reached_completion_target(
    value,
    config,
):
    """
    Check whether the target value has been reached.
    """

    value = safe_float(value)

    if value is None:
        return False

    target = config["completion_target"]

    if config["direction"] == "higher":
        return value >= target

    return value <= target


def has_meaningful_improvement(
    baseline,
    latest,
    config,
):
    """
    Determine whether change from baseline is large enough
    to count as meaningful improvement.
    """

    baseline = safe_float(baseline)
    latest = safe_float(latest)

    if baseline is None or latest is None:
        return False

    delta = config["improvement_delta"]

    if config["direction"] == "higher":

        return (
            latest
            >= baseline + delta
        )

    return (
        latest
        <= baseline - delta
    )


# ============================================================
# APPLY PROGRESS RESULT
# ============================================================

def apply_progress_result(
    recommendation,
    latest_value,
    config,
    progress_date=None,
):
    """
    Update one RecommendationHistory object.

    progress_date should be the actual date of the check-in
    or mental-wellness assessment responsible for the change.
    """

    baseline = safe_float(
        recommendation.baseline_value
    )

    latest_value = safe_float(
        latest_value
    )

    if latest_value is None:
        return recommendation

    progress_date = (
        progress_date
        or timezone.localdate()
    )

    recommendation.latest_value = (
        latest_value
    )

    recommendation.improvement_percentage = (
        calculate_percentage_change(
            baseline,
            latest_value,
            config["direction"],
        )
    )

    # ========================================================
    # COMPLETED
    # ========================================================

    if reached_completion_target(
        latest_value,
        config,
    ):

        recommendation.status = "completed"

        if recommendation.completed_at is None:

            recommendation.completed_at = (
                progress_date
            )

        if (
            recommendation.first_improved_at is None
            and baseline is not None
            and has_meaningful_improvement(
                baseline,
                latest_value,
                config,
            )
        ):

            recommendation.first_improved_at = (
                progress_date
            )

    # ========================================================
    # IMPROVING
    # ========================================================

    elif has_meaningful_improvement(
        baseline,
        latest_value,
        config,
    ):

        recommendation.status = "improving"

        if recommendation.first_improved_at is None:

            recommendation.first_improved_at = (
                progress_date
            )

    # ========================================================
    # ACTIVE
    # ========================================================

    else:

        recommendation.status = "active"

    recommendation.save(
        update_fields=[
            "baseline_value",
            "latest_value",
            "improvement_percentage",
            "status",
            "first_improved_at",
            "completed_at",
            "updated_at",
        ]
    )

    return recommendation


# ============================================================
# MENTAL WELLNESS PROGRESS
# ============================================================

def evaluate_mental_wellness_progress(
    recommendation,
):
    """
    Evaluate progress for a mental-wellness recommendation
    using MentalWellnessAssessment records.
    """

    baseline = safe_float(
        recommendation.baseline_value
    )

    # ========================================================
    # BACKFILL BASELINE
    # ========================================================

    if baseline is None:

        baseline = find_mental_baseline_for_date(
            user=recommendation.user,
            recommendation_date=(
                recommendation.recommendation_date
            ),
        )

        if baseline is not None:

            recommendation.baseline_value = (
                baseline
            )

            if recommendation.latest_value is None:

                recommendation.latest_value = (
                    baseline
                )

            recommendation.save(
                update_fields=[
                    "baseline_value",
                    "latest_value",
                    "updated_at",
                ]
            )

    # ========================================================
    # FIND LATER ASSESSMENT
    # ========================================================

    latest_assessment = (
        MentalWellnessAssessment.objects
        .filter(
            user=recommendation.user,
            date__gt=(
                recommendation.recommendation_date
            ),
        )
        .order_by("-date")
        .first()
    )

    if not latest_assessment:

        return recommendation

    latest_value = (
        calculate_mental_wellness_score(
            latest_assessment
        )
    )

    if latest_value is None:

        return recommendation

    # ========================================================
    # NO PREVIOUS BASELINE
    #
    # If no valid baseline existed before the recommendation,
    # use this first later assessment only as the baseline.
    # Do not claim improvement yet.
    # ========================================================

    if baseline is None:

        recommendation.baseline_value = (
            latest_value
        )

        recommendation.latest_value = (
            latest_value
        )

        recommendation.status = "active"

        recommendation.save(
            update_fields=[
                "baseline_value",
                "latest_value",
                "status",
                "updated_at",
            ]
        )

        return recommendation

    # ========================================================
    # EVALUATE
    # ========================================================

    return apply_progress_result(
        recommendation=recommendation,
        latest_value=latest_value,
        config=MENTAL_WELLNESS_CONFIG,
        progress_date=latest_assessment.date,
    )


# ============================================================
# DAILY LIFESTYLE PROGRESS
# ============================================================

def evaluate_daily_metric_progress(
    recommendation,
    latest_checkin=None,
):
    """
    Evaluate recommendations mapped directly to DailyCheckIn.
    """

    config = METRIC_CONFIG.get(
        recommendation.metric
    )

    if not config:

        return recommendation

    # ========================================================
    # FIND LATER CHECK-IN
    # ========================================================

    if latest_checkin is None:

        latest_checkin = (
            DailyCheckIn.objects
            .filter(
                user=recommendation.user,
                date__gt=(
                    recommendation.recommendation_date
                ),
            )
            .order_by("-date")
            .first()
        )

    if not latest_checkin:

        return recommendation

    # Never evaluate using baseline-day data
    if (
        latest_checkin.date
        <= recommendation.recommendation_date
    ):

        return recommendation

    field_name = config["field"]

    latest_value = safe_float(
        getattr(
            latest_checkin,
            field_name,
            None,
        )
    )

    if latest_value is None:

        return recommendation

    return apply_progress_result(
        recommendation=recommendation,
        latest_value=latest_value,
        config=config,
        progress_date=latest_checkin.date,
    )


# ============================================================
# SINGLE RECOMMENDATION
# ============================================================

def evaluate_recommendation_progress(
    recommendation,
    latest_checkin=None,
):
    """
    Evaluate one recommendation using the correct data source.
    """

    if recommendation.status in {
        "completed",
        "dismissed",
    }:

        return recommendation

    if (
        recommendation.metric
        == "mental_wellness"
    ):

        return evaluate_mental_wellness_progress(
            recommendation
        )

    return evaluate_daily_metric_progress(
        recommendation,
        latest_checkin=latest_checkin,
    )


# ============================================================
# ALL RECOMMENDATIONS FOR USER
# ============================================================

def evaluate_user_recommendation_progress(
    user,
):
    """
    Evaluate all active/improving recommendations for a user.

    This can be called whenever:
    - a DailyCheckIn is saved
    - a mental-wellness assessment is saved
    """

    latest_checkin = (
        DailyCheckIn.objects
        .filter(
            user=user
        )
        .order_by("-date")
        .first()
    )

    recommendations = (
        RecommendationHistory.objects
        .filter(
            user=user,
            status__in=[
                "active",
                "improving",
            ],
        )
        .order_by(
            "recommendation_date",
            "created_at",
        )
    )

    updated = []

    for recommendation in recommendations:

        # ====================================================
        # MENTAL WELLNESS
        # ====================================================

        if (
            recommendation.metric
            == "mental_wellness"
        ):

            updated.append(
                evaluate_mental_wellness_progress(
                    recommendation
                )
            )

            continue

        # ====================================================
        # DAILY CHECK-IN METRICS
        # ====================================================

        if not latest_checkin:
            continue

        if (
            latest_checkin.date
            <= recommendation.recommendation_date
        ):
            continue

        updated.append(
            evaluate_daily_metric_progress(
                recommendation,
                latest_checkin=latest_checkin,
            )
        )

    return updated