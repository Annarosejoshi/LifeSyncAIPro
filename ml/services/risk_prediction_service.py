from pathlib import Path
import json

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "ml" / "models"
BRFSS_DIABETES_MODEL_PATH = BASE_DIR / "diabetes_risk_model.joblib"

MODEL_FILES = {
    "cardiovascular": {
        "model": MODEL_DIR / "cardiovascular_lifestyle_v10.joblib",
        "metadata": MODEL_DIR / "cardiovascular_lifestyle_v10_metadata.json",
    },
    "mental_health": {
        "model": MODEL_DIR / "mental_health_lifestyle_v10.joblib",
        "metadata": MODEL_DIR / "mental_health_lifestyle_v10_metadata.json",
    },
    "hypertension": {
        "model": MODEL_DIR / "hypertension_lifestyle_v10.joblib",
        "metadata": MODEL_DIR / "hypertension_lifestyle_v10_metadata.json",
    },
    "diabetes": {
        "model": MODEL_DIR / "diabetes_lifestyle_v10.joblib",
        "metadata": MODEL_DIR / "diabetes_lifestyle_v10_metadata.json",
    },
}


# ============================================================
# CORE READINESS FEATURES
# ============================================================

DIABETES_CORE_FEATURES = ['age', 'sex', 'bmi', 'smoking_status', 'sleep_hours']

HYPERTENSION_CORE_FEATURES = ['age', 'sex', 'bmi', 'smoking_status', 'sleep_hours']

CARDIOVASCULAR_CORE_FEATURES = ['age', 'sex', 'bmi', 'smoking_status']

MENTAL_HEALTH_CORE_FEATURES = ['age', 'sex', 'bmi', 'smoking_status', 'sleep_hours']


# ============================================================
# CACHE
# ============================================================

_MODEL_CACHE = {}
_METADATA_CACHE = {}


# ============================================================
# MODEL / METADATA LOADING
# ============================================================

def load_model(model_name):
    """
    Load and cache a trained model.
    """

    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_path = MODEL_FILES[
        model_name
    ]["model"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    model = joblib.load(
        model_path
    )

    _MODEL_CACHE[
        model_name
    ] = model

    return model


def load_metadata(model_name):
    """
    Load and cache model metadata.
    """

    if model_name in _METADATA_CACHE:
        return _METADATA_CACHE[model_name]

    if model_name not in MODEL_FILES:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    metadata_path = MODEL_FILES[
        model_name
    ]["metadata"]

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(
            file
        )

    _METADATA_CACHE[
        model_name
    ] = metadata

    return metadata


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(probability):
    """
    Convert model probability to a user-facing risk band.

    This is a display band only. Model classification continues
    to use each model's own selected threshold.
    """

    if probability < 0.25:
        return "Low"

    if probability < 0.50:
        return "Moderate"

    if probability < 0.75:
        return "High"

    return "Very High"


# ============================================================
# READINESS
# ============================================================

def check_prediction_readiness(
    input_data,
    required_features,
):
    """
    Measure how many core features are currently available.
    """

    available = []
    missing = []

    for feature in required_features:
        value = input_data.get(
            feature
        )

        if value is None or value == "":
            missing.append(
                feature
            )
        else:
            available.append(
                feature
            )

    total = len(
        required_features
    )

    completeness = (
        len(available)
        / total
        * 100
        if total > 0
        else 0
    )

    return {
        "available_features": available,
        "missing_features": missing,
        "available_count": len(
            available
        ),
        "missing_count": len(
            missing
        ),
        "total_features": total,
        "completeness_percentage": round(
            completeness,
            2,
        ),
    }


# ============================================================
# GENERIC PREDICTOR
# ============================================================

def predict_risk(
    model_name,
    input_data,
):
    """
    Run one trained model using the exact feature order saved
    in its metadata file.
    """

    model = load_model(
        model_name
    )

    metadata = load_metadata(
        model_name
    )

    features = metadata.get(
        "features",
        [],
    )

    if not features:
        raise RuntimeError(
            f"No feature list found for {model_name}"
        )

    # Build the row in EXACT training order.
    row = {}

    for feature in features:
        value = input_data.get(
            feature
        )

        # Missing values are left as None so the fitted pipeline
        # can apply its own preprocessing/imputation.
        if value in [
            "",
            None,
        ]:
            value = None

        row[feature] = value

    input_df = pd.DataFrame(
        [row],
        columns=features,
    )

    probability = float(
        model.predict_proba(
            input_df
        )[0][1]
    )

    threshold = metadata.get(
        "selected_threshold",
        metadata.get(
            "threshold",
            0.50,
        ),
    )

    threshold = float(
        threshold
    )

    prediction = int(
        probability
        >= threshold
    )

    risk_level = get_risk_level(
        probability
    )

    return {
        "model": model_name,
        "model_name": metadata.get("model_name", model_name),
        "model_version": metadata.get("model_version", "unknown"),
        "evaluation": {
            "accuracy": round(metadata.get("test_metrics", {}).get("classification", {}).get("accuracy", 0) * 100, 1),
            "recall": round(metadata.get("test_metrics", {}).get("classification", {}).get("1", {}).get("recall", 0) * 100, 1),
            "precision": round(metadata.get("test_metrics", {}).get("classification", {}).get("1", {}).get("precision", 0) * 100, 1),
            "test_rows": metadata.get("test_rows"),
        },
        "input_snapshot": row,
        "probability": probability,
        "percentage": round(
            probability * 100,
            2,
        ),
        "threshold": threshold,
        "prediction": prediction,
        "risk_level": risk_level,
    }


# ============================================================
# DIABETES — BRFSS THREE-CLASS MODEL
# ============================================================

def brfss_age_group(age):
    """
    Convert age in years to the BRFSS Age category:
    1=18-24, 2=25-29, ..., 12=75-79, 13=80+.
    """
    if age in (None, ""):
        return None

    try:
        age = float(age)
    except (TypeError, ValueError):
        return None

    if age < 18:
        return None
    if age <= 24:
        return 1
    if age <= 29:
        return 2
    if age <= 34:
        return 3
    if age <= 39:
        return 4
    if age <= 44:
        return 5
    if age <= 49:
        return 6
    if age <= 54:
        return 7
    if age <= 59:
        return 8
    if age <= 64:
        return 9
    if age <= 69:
        return 10
    if age <= 74:
        return 11
    if age <= 79:
        return 12
    return 13


def load_brfss_diabetes_bundle():
    """
    Load the bundle saved by train_model.py:
    {"model": fitted_pipeline, "features": [...]}
    """
    cache_key = "_brfss_diabetes_bundle"

    if cache_key not in _MODEL_CACHE:
        if not BRFSS_DIABETES_MODEL_PATH.exists():
            raise FileNotFoundError(
                "BRFSS diabetes model not found: "
                f"{BRFSS_DIABETES_MODEL_PATH}. "
                "Run train_model.py from the project root first."
            )

        bundle = joblib.load(BRFSS_DIABETES_MODEL_PATH)

        if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
            raise RuntimeError(
                "The BRFSS diabetes model file must contain "
                "a dictionary with 'model' and 'features'. "
                "Retrain it using the updated train_model.py."
            )

        _MODEL_CACHE[cache_key] = bundle

    return _MODEL_CACHE[cache_key]


def predict_brfss_diabetes_risk(input_data):
    """
    Run the experimental BRFSS three-class model.

    Class 0 = No diabetes
    Class 1 = Prediabetes
    Class 2 = Diabetes

    The model is a screening experiment, not a medical diagnosis.
    """
    bundle = load_brfss_diabetes_bundle()
    model = bundle["model"]
    features = bundle["features"]

    age = input_data.get("age")
    sex_value = input_data.get("sex")
    bmi = input_data.get("bmi")
    smoking_status = input_data.get("smoking_status")

    # risk_input_mapper.encode_gender() returns male=1, female=2.
    # BRFSS Sex uses male=1, female=0.
    brfss_sex = None
    if sex_value == 1:
        brfss_sex = 1
    elif sex_value == 2:
        brfss_sex = 0

    # BRFSS Smoker means ever smoked at least 100 cigarettes.
    # Never -> 0; former/current -> 1.
    ever_smoked = None
    if smoking_status == 0:
        ever_smoked = 0
    elif smoking_status in (1, 2):
        ever_smoked = 1

    # Only map fields that are actually available in the current
    # mapper. Do not invent medical history or lifestyle answers.
    values = {
        "HighBP": None,
        "HighChol": None,
        "CholCheck": None,
        "BMI": bmi,
        "Smoker": ever_smoked,
        "Stroke": None,
        "HeartDiseaseorAttack": None,
        "PhysActivity": None,
        "Fruits": None,
        "Veggies": None,
        "HvyAlcoholConsump": None,
        "AnyHealthcare": None,
        "NoDocbcCost": None,
        "GenHlth": None,
        "MentHlth": None,
        "PhysHlth": None,
        "DiffWalk": None,
        "Sex": brfss_sex,
        "Age": brfss_age_group(age),
        "Education": None,
        "Income": None,
    }

    for feature in features:
        submitted_value = input_data.get(feature)
        if submitted_value not in (None, ""):
            values[feature] = (
                float(submitted_value)
                if feature == "BMI"
                else int(submitted_value)
            )

    model_readiness = check_prediction_readiness(
        values,
        features,
    )
    row = {feature: values.get(feature) for feature in features}
    input_df = pd.DataFrame([row], columns=features)

    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.classes_

    class_probabilities = {
        int(label): float(probability)
        for label, probability in zip(class_labels, probabilities)
    }

    predicted_class = max(
        class_probabilities,
        key=class_probabilities.get,
    )

    # This is the model's combined probability for classes 1 and 2.
    # Because the training data were oversampled, do not interpret it
    # as a calibrated clinical probability.
    elevated_probability = (
        class_probabilities.get(1, 0.0)
        + class_probabilities.get(2, 0.0)
    )

    labels = {
        0: "No Diabetes Class Predicted",
        1: "Prediabetes Class Predicted",
        2: "Diabetes Class Predicted",
    }

    return {
        "model": "diabetes",
        "prediction_available": True,
        "prediction": int(predicted_class),
        "label": labels.get(int(predicted_class), "Unknown Class"),
        "probability": elevated_probability,
        "percentage": round(elevated_probability * 100, 2),
        "risk_level": get_risk_level(elevated_probability),
        "class_probabilities": {
            labels[class_id]: round(probability * 100, 2)
            for class_id, probability in class_probabilities.items()
            if class_id in labels
        },
        "model_input_readiness": model_readiness,
        "experimental": True,
        "notice": (
            "Experimental BRFSS screening result. The training data used "
            "oversampling, so class probabilities are not clinically "
            "calibrated. This is not a medical diagnosis."
        ),
    }


def predict_diabetes_risk(input_data):
    """Use the model whose training features match the submitted input."""
    if "BMI" in input_data:
        result = predict_brfss_diabetes_risk(input_data)
        result["readiness"] = result["model_input_readiness"]
        return result

    readiness = check_prediction_readiness(input_data, DIABETES_CORE_FEATURES)
    if readiness["missing_count"] > 0 or any(input_data.get(key) in (None, "") for key in ("age", "sex", "bmi")):
        return {
            "model": "diabetes", "prediction_available": False,
            "reason": "Complete your health profile and daily check-in before requesting a lifestyle diabetes estimate.",
            "readiness": readiness,
        }
    result = predict_risk("diabetes", input_data)
    result.update({
        "prediction_available": True,
        "risk_level": "Elevated" if result["prediction"] else "Low",
        "readiness": readiness,
        "label": "Elevated Diabetes Risk" if result["prediction"] else "Lower Diabetes Risk",
        "notice": "Lifestyle screening estimate based on everyday habits and your profile.",
    })
    return result


# ============================================================
# HYPERTENSION
# ============================================================

def predict_hypertension_risk(
    input_data,
):
    readiness = check_prediction_readiness(
        input_data,
        HYPERTENSION_CORE_FEATURES,
    )

    if readiness["missing_count"] > 0 or any(input_data.get(key) in (None, "") for key in ("age", "sex", "bmi")):
        return {
            "model": "hypertension",
            "prediction_available": False,
            "reason": (
                "Not enough personal health data "
                "to generate a reliable hypertension risk estimate."
            ),
            "readiness": readiness,
        }

    result = predict_risk(
        "hypertension",
        input_data,
    )

    result["prediction_available"] = True
    result["risk_level"] = "Elevated" if result["prediction"] else "Low"
    result["readiness"] = readiness

    result["label"] = (
        "Elevated Hypertension Risk"
        if result["prediction"] == 1
        else "Lower Hypertension Risk"
    )

    return result


# ============================================================
# CARDIOVASCULAR
# ============================================================

def predict_cardiovascular_risk(
    input_data,
):
    readiness = check_prediction_readiness(
        input_data,
        CARDIOVASCULAR_CORE_FEATURES,
    )

    if readiness["missing_count"] > 0 or any(input_data.get(key) in (None, "") for key in ("age", "sex", "bmi")):
        return {
            "model": "cardiovascular",
            "prediction_available": False,
            "reason": (
                "Not enough personal health data "
                "to generate a reliable cardiovascular risk estimate."
            ),
            "readiness": readiness,
        }

    result = predict_risk(
        "cardiovascular",
        input_data,
    )

    result["prediction_available"] = True
    result["risk_level"] = "Elevated" if result["prediction"] else "Low"
    result["readiness"] = readiness

    result["label"] = (
        "Elevated Cardiovascular Risk"
        if result["prediction"] == 1
        else "Lower Cardiovascular Risk"
    )

    return result


# ============================================================
# MENTAL WELLNESS
# ============================================================

def predict_mental_health_risk(
    input_data,
):
    readiness = check_prediction_readiness(
        input_data,
        MENTAL_HEALTH_CORE_FEATURES,
    )

    if readiness["missing_count"] > 0 or any(input_data.get(key) in (None, "") for key in ("age", "sex", "bmi")):
        return {
            "model": "mental_health",
            "prediction_available": False,
            "reason": (
                "Not enough personal health data "
                "to generate a mental wellness risk estimate."
            ),
            "readiness": readiness,
        }

    result = predict_risk(
        "mental_health",
        input_data,
    )

    result["prediction_available"] = True
    result["risk_level"] = "Elevated" if result["prediction"] else "Low"
    result["readiness"] = readiness

    result["label"] = (
        "Elevated Mental Wellness Risk"
        if result["prediction"] == 1
        else "Lower Mental Wellness Risk"
    )

    return result


# ============================================================
# OPTIONAL MULTI-MODEL WRAPPER
# ============================================================

def predict_all_risks(
    cardiovascular_data=None,
    hypertension_data=None,
    diabetes_data=None,
):
    """
    Preserve the existing helper API for callers that already provide
    model-specific input dictionaries.
    """

    results = {}

    if cardiovascular_data is not None:
        results[
            "cardiovascular"
        ] = predict_cardiovascular_risk(
            cardiovascular_data
        )

    if hypertension_data is not None:
        results[
            "hypertension"
        ] = predict_hypertension_risk(
            hypertension_data
        )

    if diabetes_data is not None:
        results[
            "diabetes"
        ] = predict_diabetes_risk(
            diabetes_data
        )

    return results


# ============================================================
# USER-LEVEL ALL-HEALTH-RISKS WRAPPER
# ============================================================

def get_all_health_risks(user):
    """
    Build the four model inputs from the current user data and
    return one result dictionary for each screening model.

    A failure in one model does not prevent the other models from
    returning their results.
    """

    from django.utils import timezone
    if not user.daily_checkins.filter(date=timezone.localdate()).exists():
        return {
            name: {
                "model": name,
                "prediction_available": False,
                "reason": "Complete today's daily check-in to see today's prediction.",
                "readiness": {},
            }
            for name in ("diabetes", "hypertension", "cardiovascular", "mental_health")
        }

    from ml.services.risk_input_mapper import (
        build_cardiovascular_input_from_user,
        build_diabetes_input_from_user,
        build_hypertension_input_from_user,
        build_mental_health_input_from_user,
    )

    from ml.services.risk_input_mapper import build_common_risk_input
    common = build_common_risk_input(user)
    if common.get("age") is not None and common["age"] < 18:
        return {
            name: {
                "model": name, "prediction_available": False,
                "reason": "These models were evaluated on adults aged 18 and above. Adult screening estimates are unavailable for this age.",
                "readiness": {},
            }
            for name in ("diabetes", "hypertension", "cardiovascular", "mental_health")
        }
    labels = {
        "age": "date of birth", "sex": "gender in your profile",
        "bmi": "height and weight", "smoking_status": "smoking status",
        "sleep_hours": "sleep today",
    }
    missing = [key for key in DIABETES_CORE_FEATURES if common.get(key) is None]
    if missing:
        return {
            name: {
                "model": name, "prediction_available": False,
                "reason": "Please enter: " + ", ".join(labels[key] for key in missing) + ".",
                "readiness": {"missing_features": missing},
            }
            for name in ("diabetes", "hypertension", "cardiovascular", "mental_health")
        }

    results = {}

    try:
        diabetes_data = (
            build_diabetes_input_from_user(
                user
            )
        )

        results["diabetes"] = (
            predict_diabetes_risk(
                diabetes_data
            )
        )

    except Exception as exc:
        results["diabetes"] = {
            "model": "diabetes",
            "prediction_available": False,
            "reason": str(exc),
        }

    try:
        hypertension_data = (
            build_hypertension_input_from_user(
                user
            )
        )

        results["hypertension"] = (
            predict_hypertension_risk(
                hypertension_data
            )
        )

    except Exception as exc:
        results["hypertension"] = {
            "model": "hypertension",
            "prediction_available": False,
            "reason": str(exc),
        }

    try:
        cardiovascular_data = (
            build_cardiovascular_input_from_user(
                user
            )
        )

        results["cardiovascular"] = (
            predict_cardiovascular_risk(
                cardiovascular_data
            )
        )

    except Exception as exc:
        results["cardiovascular"] = {
            "model": "cardiovascular",
            "prediction_available": False,
            "reason": str(exc),
        }

    try:
        mental_health_data = (
            build_mental_health_input_from_user(
                user
            )
        )

        results["mental_health"] = (
            predict_mental_health_risk(
                mental_health_data
            )
        )

    except Exception as exc:
        results["mental_health"] = {
            "model": "mental_health",
            "prediction_available": False,
            "reason": str(exc),
        }

    return results
