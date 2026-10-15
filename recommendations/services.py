from django.utils import timezone

from mental_health.models import MentalWellnessAssessment

from .models import RecommendationHistory


# ============================================================
# MENTAL WELLNESS SCORE
# ============================================================

def calculate_mental_wellness_score(assessment):
    """
    Convert one MentalWellnessAssessment into a transparent
    0-100 wellness score.

    Higher is better.

    Positive dimensions:
    - mood
    - energy
    - motivation
    - concentration
    - social wellbeing

    Reverse-scored dimensions:
    - stress
    - anxiety
    - workload
    """

    if not assessment:
        return None

    positive_values = [
        assessment.mood,
        assessment.energy,
        assessment.motivation,
        assessment.concentration,
        assessment.social_wellbeing,
    ]

    reverse_values = [
        6 - assessment.stress,
        6 - assessment.anxiety,
        6 - assessment.workload,
    ]

    values = positive_values + reverse_values

    if not values:
        return None

    score_1_to_5 = sum(values) / len(values)

    score_0_to_100 = (
        score_1_to_5 / 5
    ) * 100

    return round(
        max(0, min(score_0_to_100, 100)),
        1,
    )


# ============================================================
# METRIC DETECTION
# ============================================================

def detect_metric(recommendation):
    """
    Map a generated recommendation to the health metric that
    can later be used for progress tracking.
    """

    category = str(
        recommendation.get("category", "")
    ).strip().lower()

    title = str(
        recommendation.get("title", "")
    ).strip().lower()

    if "sleep" in category or "sleep" in title:
        return "sleep"

    if (
        "hydration" in category
        or "water" in category
        or "hydration" in title
        or "water" in title
    ):
        return "water"

    # Keep this BEFORE the generic movement/exercise rule.
    if (
        "step" in category
        or "step" in title
        or "daily movement" in title
    ):
        return "steps"

    if (
        "exercise" in category
        or "activity" in category
        or "movement" in title
        or "exercise" in title
    ):
        return "exercise"

    if (
        "nutrition" in category
        or "food" in title
        or "nutrition" in title
    ):
        return "nutrition"

    if (
        "screen" in category
        or "screen" in title
    ):
        return "screen_time"

    if (
        "stress" in category
        or "stress" in title
    ):
        return "stress"

    if "mood" in title:
        return "mood"

    if "energy" in title:
        return "energy"

    if (
        "mental" in category
        or "wellness" in category
        or "mental" in title
        or "wellness" in title
    ):
        return "mental_wellness"

    if (
        "weekly trend" in category
        or "health trend" in title
    ):
        return "health_score"

    return "general"


# ============================================================
# BASELINE VALUE
# ============================================================

def get_baseline_value(
    recommendation,
    checkin=None,
    mental_assessment=None,
):
    """
    Capture the recommendation's baseline value.

    Daily lifestyle recommendations use DailyCheckIn fields.
    Mental-wellness recommendations use the dedicated
    MentalWellnessAssessment score.
    """

    metric = detect_metric(
        recommendation
    )

    if metric == "mental_wellness":

        if mental_assessment:
            return calculate_mental_wellness_score(
                mental_assessment
            )

        return None

    if not checkin:
        return None

    mapping = {
        "sleep": "sleep_hours",
        "water": "water_liters",
        "exercise": "exercise_minutes",
        "steps": "steps",
        "nutrition": "fruit_vegetable_servings",
        "screen_time": "screen_time_hours",
        "stress": "stress_level",
        "mood": "mood",
        "energy": "energy_level",
    }

    field_name = mapping.get(
        metric
    )

    if not field_name:
        return None

    value = getattr(
        checkin,
        field_name,
        None,
    )

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# BASELINE BACKFILL
# ============================================================

def find_mental_baseline_for_date(
    user,
    recommendation_date,
):
    """
    Prefer an assessment recorded on the recommendation date.
    If none exists, use the latest assessment before that date.
    """

    assessment = (
        MentalWellnessAssessment.objects
        .filter(
            user=user,
            date__lte=recommendation_date,
        )
        .order_by("-date")
        .first()
    )

    if not assessment:
        return None

    return calculate_mental_wellness_score(
        assessment
    )


def backfill_missing_mental_baselines(user):
    """
    Backfill old mental-wellness recommendation records that were
    created before mental baseline tracking was added.
    """

    records = (
        RecommendationHistory.objects
        .filter(
            user=user,
            metric="mental_wellness",
            baseline_value__isnull=True,
        )
    )

    updated = []

    for record in records:

        baseline = find_mental_baseline_for_date(
            user=user,
            recommendation_date=record.recommendation_date,
        )

        if baseline is None:
            continue

        record.baseline_value = baseline

        if record.latest_value is None:
            record.latest_value = baseline

        record.save(
            update_fields=[
                "baseline_value",
                "latest_value",
                "updated_at",
            ]
        )

        updated.append(
            record
        )

    return updated


# ============================================================
# SAVE RECOMMENDATION HISTORY
# ============================================================

def save_recommendation_history(
    user,
    recommendations,
    checkin=None,
    mental_assessment=None,
):
    """
    Save today's generated recommendations without duplicates.
    """

    today = timezone.localdate()

    saved = []

    for recommendation in recommendations:

        metric = detect_metric(
            recommendation
        )

        baseline = get_baseline_value(
            recommendation,
            checkin=checkin,
            mental_assessment=mental_assessment,
        )

        history, created = (
            RecommendationHistory.objects
            .get_or_create(
                user=user,
                recommendation_date=today,
                title=recommendation.get(
                    "title",
                    "Wellness Recommendation",
                ),
                defaults={
                    "category":
                        recommendation.get(
                            "category",
                            "general",
                        ),

                    "message":
                        recommendation.get(
                            "message",
                            "",
                        ),

                    "priority":
                        recommendation.get(
                            "priority",
                            "medium",
                        ),

                    "metric":
                        metric,

                    "baseline_value":
                        baseline,

                    "latest_value":
                        baseline,

                    "status":
                        "active",

                    "source":
                        "recommendation_engine_v2",
                },
            )
        )

        if not created:

            history.category = (
                recommendation.get(
                    "category",
                    history.category,
                )
            )

            history.message = (
                recommendation.get(
                    "message",
                    history.message,
                )
            )

            history.priority = (
                recommendation.get(
                    "priority",
                    history.priority,
                )
            )

            history.metric = metric

            update_fields = [
                "category",
                "message",
                "priority",
                "metric",
                "updated_at",
            ]

            # Backfill a missing baseline if today's recommendation now
            # has a measurable value available.
            if (
                history.baseline_value is None
                and baseline is not None
            ):
                history.baseline_value = baseline

                if history.latest_value is None:
                    history.latest_value = baseline

                update_fields.extend([
                    "baseline_value",
                    "latest_value",
                ])

            history.save(
                update_fields=update_fields
            )

        saved.append(
            history
        )

    return saved
