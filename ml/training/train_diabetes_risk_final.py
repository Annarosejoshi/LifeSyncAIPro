from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "ml"
    / "datasets"
    / "diabetes_012_health_indicators_BRFSS2015.csv"
)

MODEL_DIR = BASE_DIR / "ml" / "models"
EVALUATION_DIR = BASE_DIR / "ml" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


ORIGINAL_TARGET = "Diabetes_012"
TARGET = "Diabetes_Risk"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    if ORIGINAL_TARGET not in df.columns:
        raise ValueError(
            f"{ORIGINAL_TARGET} not found."
        )

    df[TARGET] = (
        df[ORIGINAL_TARGET] > 0
    ).astype(int)

    df = df.drop(
        columns=[ORIGINAL_TARGET]
    )

    return df


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    metrics = {
        "threshold": threshold,

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),
    }

    return metrics, matrix


# =========================================================
# MODELS
# =========================================================

def create_random_forest():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=350,
                max_depth=18,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
        ),
    ])


def create_xgboost(
    scale_pos_weight
):

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            XGBClassifier(
                objective="binary:logistic",
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42
            )
        ),
    ])


# =========================================================
# THRESHOLD SEARCH
# =========================================================

def find_best_threshold(
    y_validation,
    probabilities
):

    rows = []

    thresholds = np.arange(
        0.30,
        0.71,
        0.02
    )

    for threshold in thresholds:

        metrics, _ = calculate_metrics(
            y_validation,
            probabilities,
            threshold
        )

        rows.append(metrics)

    table = pd.DataFrame(rows)

    # Screening-oriented selection:
    # require at least 70% recall,
    # then choose highest F1.
    eligible = table[
        table["recall"] >= 0.70
    ]

    if not eligible.empty:

        best = eligible.sort_values(
            by=[
                "f1",
                "specificity",
                "precision"
            ],
            ascending=False
        ).iloc[0]

    else:

        best = table.sort_values(
            by=[
                "f1",
                "recall"
            ],
            ascending=False
        ).iloc[0]

    return (
        float(best["threshold"]),
        table
    )


# =========================================================
# MAIN
# =========================================================

def main():

    df = load_data()

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    feature_names = X.columns.tolist()

    print("=" * 70)
    print("FINAL DIABETES RISK TRAINING")
    print("=" * 70)

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Features: {len(feature_names)}"
    )

    print("\nClass distribution:")

    print(
        y.value_counts().sort_index()
    )


    # -----------------------------------------------------
    # TRAIN / TEMP
    # -----------------------------------------------------

    (
        X_train,
        X_temp,
        y_train,
        y_temp
    ) = train_test_split(
        X,
        y,

        test_size=0.30,

        stratify=y,

        random_state=42
    )


    # -----------------------------------------------------
    # VALIDATION / TEST
    # -----------------------------------------------------

    (
        X_validation,
        X_test,
        y_validation,
        y_test
    ) = train_test_split(
        X_temp,
        y_temp,

        test_size=0.50,

        stratify=y_temp,

        random_state=42
    )


    print("\nSplit sizes:")

    print(
        f"Train:      {len(X_train):,}"
    )

    print(
        f"Validation: {len(X_validation):,}"
    )

    print(
        f"Test:       {len(X_test):,}"
    )


    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count
        / positive_count
    )


    candidates = [
        (
            "Random Forest",
            create_random_forest()
        ),

        (
            "XGBoost",
            create_xgboost(
                scale_pos_weight
            )
        ),
    ]


    model_results = []


    # =====================================================
    # TRAIN + VALIDATE
    # =====================================================

    for name, model in candidates:

        print("\n" + "=" * 70)
        print(
            f"TRAINING: {name}"
        )
        print("=" * 70)

        model.fit(
            X_train,
            y_train
        )

        validation_probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        (
            best_threshold,
            threshold_table
        ) = find_best_threshold(
            y_validation,
            validation_probabilities
        )

        validation_metrics, _ = (
            calculate_metrics(
                y_validation,
                validation_probabilities,
                best_threshold
            )
        )

        print(
            "\nSelected threshold:",
            round(
                best_threshold,
                3
            )
        )

        print("\nValidation metrics:")

        for key, value in (
            validation_metrics.items()
        ):

            print(
                f"{key:12s}: "
                f"{value:.4f}"
            )

        model_results.append(
            {
                "name": name,
                "model": model,
                "threshold":
                    best_threshold,
                "validation_metrics":
                    validation_metrics,
                "threshold_table":
                    threshold_table,
            }
        )


    # =====================================================
    # SELECT MODEL USING VALIDATION ONLY
    # =====================================================

    validation_comparison = pd.DataFrame([
        {
            "model":
                result["name"],

            **result[
                "validation_metrics"
            ]
        }

        for result in model_results
    ])


    validation_comparison = (
        validation_comparison.sort_values(
            by=[
                "f1",
                "recall",
                "roc_auc"
            ],
            ascending=False
        )
    )


    print("\n" + "=" * 70)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)

    print(
        validation_comparison.to_string(
            index=False
        )
    )


    best_name = (
        validation_comparison
        .iloc[0]["model"]
    )


    best_result = next(
        result
        for result in model_results
        if result["name"]
        == best_name
    )


    final_model = best_result["model"]

    final_threshold = (
        best_result["threshold"]
    )


    # =====================================================
    # FINAL TEST
    # =====================================================

    test_probabilities = (
        final_model.predict_proba(
            X_test
        )[:, 1]
    )


    test_metrics, test_matrix = (
        calculate_metrics(
            y_test,
            test_probabilities,
            final_threshold
        )
    )


    test_predictions = (
        test_probabilities
        >= final_threshold
    ).astype(int)


    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nModel: {best_name}"
    )

    print(
        f"Threshold: "
        f"{final_threshold:.3f}"
    )


    for key, value in (
        test_metrics.items()
    ):

        print(
            f"{key:12s}: "
            f"{value:.4f}"
        )


    print(
        "\nConfusion Matrix:"
    )

    print(
        test_matrix
    )


    print(
        "\nClassification Report:"
    )

    report = classification_report(
        y_test,
        test_predictions,

        target_names=[
            "Lower Risk",
            "Elevated Risk"
        ],

        digits=4,

        zero_division=0
    )

    print(
        report
    )


    # =====================================================
    # SAVE FINAL MODEL
    # =====================================================

    model_path = (
        MODEL_DIR
        / "diabetes_risk_final.joblib"
    )

    joblib.dump(
        final_model,
        model_path
    )


    # =====================================================
    # SAVE METADATA
    # =====================================================

    metadata = {

        "model_name":
            best_name,

        "version":
            "4.0-final",

        "problem_type":
            "binary_risk_screening",

        "target":
            TARGET,

        "features":
            feature_names,

        "threshold":
            float(
                final_threshold
            ),

        "labels": {
            "0": "Lower Risk",
            "1": "Elevated Risk",
        },

        "validation_metrics": {
            key: float(value)
            for key, value
            in best_result[
                "validation_metrics"
            ].items()
        },

        "test_metrics": {
            key: float(value)
            for key, value
            in test_metrics.items()
        },

        "disclaimer": (
            "This is a statistical "
            "risk-screening estimate, "
            "not a medical diagnosis."
        ),
    }


    metadata_path = (
        MODEL_DIR
        / "diabetes_risk_final_metadata.json"
    )


    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


    # =====================================================
    # SAVE EVALUATION
    # =====================================================

    validation_comparison.to_csv(
        EVALUATION_DIR
        / "diabetes_final_validation_comparison.csv",
        index=False
    )


    best_result[
        "threshold_table"
    ].to_csv(
        EVALUATION_DIR
        / "diabetes_threshold_analysis.csv",
        index=False
    )


    with open(
        EVALUATION_DIR
        / "diabetes_final_test_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


    with open(
        EVALUATION_DIR
        / "diabetes_final_test_confusion_matrix.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            test_matrix.tolist(),
            file,
            indent=4
        )


    print("\n" + "=" * 70)
    print("FINAL MODEL SAVED")
    print("=" * 70)

    print(
        f"\nModel:\n{model_path}"
    )

    print(
        f"\nMetadata:\n{metadata_path}"
    )


if __name__ == "__main__":
    main()