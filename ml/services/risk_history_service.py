from django.utils import timezone

from predictions.models import PredictionHistory

from ml.services.risk_input_mapper import (
    build_diabetes_input_from_user,
    build_hypertension_input_from_user,
    build_cardiovascular_input_from_user,
    build_mental_health_input_from_user,
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {
    "diabetes": {
        "model_name": "XGBoost",
        "model_version": "v1",
    },

    "hypertension": {
        "model_name": "Random Forest",
        "model_version": "v1",
    },

    "cardiovascular": {
        "model_name": "Random Forest",
        "model_version": "v1",
    },

    "mental_health": {
        "model_name": "Random Forest",
        "model_version": "v1",
    },
}


# ============================================================
# INPUT BUILDERS
# ============================================================

INPUT_BUILDERS = {
    "diabetes":
        build_diabetes_input_from_user,

    "hypertension":
        build_hypertension_input_from_user,

    "cardiovascular":
        build_cardiovascular_input_from_user,

    "mental_health":
        build_mental_health_input_from_user,
}


# ============================================================
# RISK LEVEL NORMALIZER
# ============================================================

def normalize_risk_level(value):

    if not value:
        return "low"

    value = str(value).strip().lower()

    if value in [
        "high",
        "very high",
        "elevated",
    ]:
        return "high"

    if value == "moderate":
        return "moderate"

    return "low"


# ============================================================
# BUILD EXPLANATION
# ============================================================

def build_explanation(result):

    explanation = []

    label = result.get(
        "label"
    )

    if label:
        explanation.append(
            label
        )

    readiness = result.get(
        "readiness"
    )

    if readiness:

        completeness = readiness.get(
            "completeness_percentage"
        )

        if completeness is not None:

            explanation.append(
                f"Data completeness: {completeness}%"
            )

        missing = readiness.get(
            "missing_features",
            [],
        )

        if missing:

            explanation.append(
                "Missing optional information: "
                + ", ".join(missing)
            )

    return explanation


# ============================================================
# SAVE ONE MODEL
# ============================================================

def save_single_risk_history(
    user,
    model_key,
    result,
):
    """
    Save or update today's prediction for one model.
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

    if model_key not in MODEL_INFO:
        return None

    builder = INPUT_BUILDERS.get(
        model_key
    )

    if not builder:
        return None

    # --------------------------------------------------------
    # Rebuild exact current user input
    # --------------------------------------------------------

    input_snapshot = result.get("input_snapshot")
    if input_snapshot is None:
        input_snapshot = builder(user)

    model_info = {
        "model_name": result.get("model_name", MODEL_INFO[model_key]["model_name"]),
        "model_version": result.get("model_version", MODEL_INFO[model_key]["model_version"]),
    }

    probability = float(
        result.get(
            "probability",
            0
        )
    )

    predicted_class = int(
        result.get(
            "prediction",
            0
        )
    )

    threshold = result.get(
        "threshold"
    )

    if threshold is not None:
        threshold = float(
            threshold
        )

    risk_level = normalize_risk_level(
        result.get(
            "risk_level"
        )
    )

    explanation = build_explanation(
        result
    )

    today = timezone.localdate()

    # --------------------------------------------------------
    # Avoid creating duplicates every dashboard refresh.
    #
    # If today's record already exists:
    # update it.
    #
    # Otherwise:
    # create a new record.
    # --------------------------------------------------------

    existing = (
        PredictionHistory.objects
        .filter(
            user=user,
            prediction_type=model_key,
            created_at__date=today,
        )
        .order_by(
            "-created_at"
        )
        .first()
    )

    if existing:

        existing.predicted_class = (
            predicted_class
        )

        existing.probability = (
            probability
        )

        existing.risk_level = (
            risk_level
        )

        existing.model_name = (
            model_info[
                "model_name"
            ]
        )

        existing.model_version = (
            model_info[
                "model_version"
            ]
        )

        existing.threshold = (
            threshold
        )

        existing.input_snapshot = (
            input_snapshot
        )

        existing.explanation = (
            explanation
        )

        existing.save()

        return existing


    # --------------------------------------------------------
    # CREATE NEW HISTORY RECORD
    # --------------------------------------------------------

    return PredictionHistory.objects.create(

        user=user,

        prediction_type=
            model_key,

        predicted_class=
            predicted_class,

        probability=
            probability,

        risk_level=
            risk_level,

        model_name=
            model_info[
                "model_name"
            ],

        model_version=
            model_info[
                "model_version"
            ],

        threshold=
            threshold,

        input_snapshot=
            input_snapshot,

        explanation=
            explanation,
    )


# ============================================================
# SAVE ALL FOUR MODELS
# ============================================================

def save_all_health_risk_history(
    user,
    risk_results,
):
    """
    Save all available LifeSync AI risk predictions.

    Expected risk_results:

    {
        "diabetes": {...},
        "hypertension": {...},
        "cardiovascular": {...},
        "mental_health": {...},
    }
    """

    saved_results = {}

    for model_key in [
        "diabetes",
        "hypertension",
        "cardiovascular",
        "mental_health",
    ]:

        result = risk_results.get(
            model_key
        )

        try:

            saved = save_single_risk_history(
                user,
                model_key,
                result,
            )

            saved_results[
                model_key
            ] = saved

        except Exception as exc:

            print(
                f"Prediction history save error "
                f"({model_key}):",
                exc,
            )

            saved_results[
                model_key
            ] = None

    return saved_results