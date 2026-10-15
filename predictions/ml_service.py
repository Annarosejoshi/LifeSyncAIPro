from pathlib import Path
from functools import lru_cache
import json

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "diabetes_risk_final.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "ml"
    / "models"
    / "diabetes_risk_final_metadata.json"
)


@lru_cache(maxsize=1)
def load_diabetes_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Diabetes model not found: {MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


@lru_cache(maxsize=1)
def load_diabetes_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def predict_diabetes_risk(features):

    model = load_diabetes_model()
    metadata = load_diabetes_metadata()

    expected_features = metadata[
        "features"
    ]

    missing = [
        feature
        for feature in expected_features
        if feature not in features
    ]

    if missing:
        raise ValueError(
            "Missing diabetes features: "
            + ", ".join(missing)
        )

    ordered_data = {
        feature: [
            features[feature]
        ]
        for feature in expected_features
    }

    dataframe = pd.DataFrame(
        ordered_data
    )

    probability = float(
        model.predict_proba(
            dataframe
        )[0][1]
    )

    threshold = float(
        metadata["threshold"]
    )

    predicted_class = (
        1
        if probability >= threshold
        else 0
    )

    if probability < 0.35:
        risk_level = "low"

    elif probability < threshold:
        risk_level = "moderate"

    else:
        risk_level = "elevated"

    return {
        "predicted_class":
            predicted_class,

        "probability":
            probability,

        "percentage":
            round(
                probability * 100,
                1
            ),

        "risk_level":
            risk_level,

        "threshold":
            threshold,

        "model_name":
            metadata["model_name"],

        "model_version":
            metadata["version"],
    }