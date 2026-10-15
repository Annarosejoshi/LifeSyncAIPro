from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
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


# =========================================================
# TARGET CONFIGURATION
# =========================================================

ORIGINAL_TARGET = "Diabetes_012"
TARGET = "Diabetes_Risk"


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if ORIGINAL_TARGET not in df.columns:
        raise ValueError(
            f"Target '{ORIGINAL_TARGET}' not found."
        )

    # Convert the original 3 classes:
    #
    # 0 = No Diabetes
    # 1 = Prediabetes
    # 2 = Diabetes
    #
    # Into binary screening target:
    #
    # 0 = Lower Risk
    # 1 = Elevated Risk

    df[TARGET] = (
        df[ORIGINAL_TARGET] > 0
    ).astype(int)

    df = df.drop(
        columns=[ORIGINAL_TARGET]
    )

    print("\n" + "=" * 70)
    print("DIABETES RISK SCREENING DATASET")
    print("=" * 70)

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Input Features: {df.shape[1] - 1}"
    )

    print("\nTarget Distribution:")

    counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():

        name = (
            "Lower Risk"
            if label == 0
            else "Elevated Risk"
        )

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{label} - {name}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print(
        "\nMissing Values:",
        int(df.isnull().sum().sum())
    )

    return df


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET].astype(int)

    feature_names = X.columns.tolist()

    print(
        "\nFeatures:"
    )

    for feature in feature_names:
        print(
            f" - {feature}"
        )

    return (
        X,
        y,
        feature_names
    )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

def create_logistic_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            ),
        ]
    )


# =========================================================
# RANDOM FOREST
# =========================================================

def create_random_forest_model():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42
                )
            ),
        ]
    )


# =========================================================
# XGBOOST
# =========================================================

def create_xgboost_model(
    scale_pos_weight
):

    return Pipeline(
        steps=[
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

                    n_estimators=350,

                    learning_rate=0.05,

                    max_depth=6,

                    subsample=0.8,

                    colsample_bytree=0.8,

                    scale_pos_weight=scale_pos_weight,

                    eval_metric="logloss",

                    random_state=42,

                    n_jobs=-1
                )
            ),
        ]
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 70)
    print(
        f"TRAINING: {name}"
    )
    print("=" * 70)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    print("\nMetrics")

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")

    print(
        matrix
    )

    report = classification_report(
        y_test,
        predictions,

        target_names=[
            "Lower Risk",
            "Elevated Risk"
        ],

        digits=4,

        zero_division=0
    )

    print(
        "\nClassification Report:"
    )

    print(
        report
    )

    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "report": report,
        "matrix": matrix.tolist(),
    }


# =========================================================
# MAIN
# =========================================================

def main():

    df = load_dataset()

    X, y, feature_names = prepare_data(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,

        test_size=0.20,

        stratify=y,

        random_state=42
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Training Records: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing Records: "
        f"{len(X_test):,}"
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

    print(
        "\nNegative class records:",
        f"{negative_count:,}"
    )

    print(
        "Positive class records:",
        f"{positive_count:,}"
    )

    print(
        "XGBoost scale_pos_weight:",
        round(
            scale_pos_weight,
            3
        )
    )


    # =====================================================
    # MODELS
    # =====================================================

    models = [
        (
            "Logistic Regression",
            create_logistic_model()
        ),

        (
            "Random Forest",
            create_random_forest_model()
        ),

        (
            "XGBoost",
            create_xgboost_model(
                scale_pos_weight
            )
        ),
    ]


    results = []


    for name, model in models:

        result = evaluate_model(
            name,
            model,

            X_train,
            X_test,

            y_train,
            y_test
        )

        results.append(
            result
        )


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    comparison_rows = []

    for result in results:

        comparison_rows.append(
            {
                "model":
                    result["name"],

                **result["metrics"]
            }
        )


    comparison = pd.DataFrame(
        comparison_rows
    )


    # Screening model:
    # prioritize F1 first,
    # then recall,
    # then ROC-AUC.

    comparison = comparison.sort_values(
        by=[
            "f1",
            "recall",
            "roc_auc"
        ],

        ascending=False
    )


    print("\n")
    print("=" * 70)
    print("FINAL DIABETES RISK MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )


    # =====================================================
    # BEST MODEL
    # =====================================================

    best_model_name = (
        comparison.iloc[0]["model"]
    )


    best_result = next(

        result

        for result in results

        if result["name"]
        == best_model_name
    )


    print("\n" + "=" * 70)
    print("BEST DIABETES RISK MODEL")
    print("=" * 70)

    print(
        best_model_name
    )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = (
        MODEL_DIR
        / "diabetes_risk_model.joblib"
    )


    joblib.dump(
        best_result["model"],
        model_path
    )


    # =====================================================
    # SAVE METADATA
    # =====================================================

    metadata = {

        "model_name":
            best_model_name,

        "version":
            "3.0-risk-screening",

        "problem_type":
            "binary_classification",

        "source_target":
            ORIGINAL_TARGET,

        "target":
            TARGET,

        "target_mapping": {
            "0": (
                "Original class 0: "
                "No Diabetes"
            ),

            "1": (
                "Original classes 1 and 2: "
                "Prediabetes or Diabetes"
            ),
        },

        "prediction_labels": {
            "0": "Lower Risk",
            "1": "Elevated Risk",
        },

        "features":
            feature_names,

        "metrics": {

            key: float(value)

            for key, value
            in best_result[
                "metrics"
            ].items()
        },

        "disclaimer": (
            "This model provides a statistical "
            "risk-screening estimate based on "
            "survey health indicators. It is not "
            "a medical diagnosis."
        ),
    }


    metadata_path = (
        MODEL_DIR
        / "diabetes_risk_model_metadata.json"
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
    # SAVE COMPARISON
    # =====================================================

    comparison_path = (
        EVALUATION_DIR
        / "diabetes_risk_model_comparison.csv"
    )


    comparison.to_csv(
        comparison_path,
        index=False
    )


    # =====================================================
    # SAVE REPORT
    # =====================================================

    report_path = (
        EVALUATION_DIR
        / "diabetes_risk_classification_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            best_result["report"]
        )


    # =====================================================
    # SAVE CONFUSION MATRIX
    # =====================================================

    matrix_path = (
        EVALUATION_DIR
        / "diabetes_risk_confusion_matrix.json"
    )


    with open(
        matrix_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_result["matrix"],
            file,
            indent=4
        )


    # =====================================================
    # COMPLETE
    # =====================================================

    print("\nSaved files:")

    print(
        f"\nModel:\n{model_path}"
    )

    print(
        f"\nMetadata:\n{metadata_path}"
    )

    print(
        f"\nComparison:\n{comparison_path}"
    )

    print(
        f"\nClassification Report:\n"
        f"{report_path}"
    )

    print(
        f"\nConfusion Matrix:\n"
        f"{matrix_path}"
    )


    print("\n" + "=" * 70)
    print(
        "DIABETES RISK TRAINING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()