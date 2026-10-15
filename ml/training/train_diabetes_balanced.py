from pathlib import Path
import json

import joblib
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONFIGURATION
# =========================================================

TARGET = "Diabetes_012"

CLASS_NAMES = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 70)
    print("BALANCED DIABETES EXPERIMENT")
    print("=" * 70)

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    print("\nOriginal class distribution:")

    counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    for value, count in counts.items():

        name = CLASS_NAMES[
            int(value)
        ]

        percent = (
            count
            / len(df)
            * 100
        )

        print(
            f"{int(value)} - {name}: "
            f"{count:,} "
            f"({percent:.2f}%)"
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

    return (
        X,
        y,
        feature_names
    )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

def create_logistic_model():

    return ImbPipeline(
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
                "oversampler",
                RandomOverSampler(
                    random_state=42
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42
                )
            ),
        ]
    )


# =========================================================
# RANDOM FOREST
# =========================================================

def create_random_forest_model():

    return ImbPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "oversampler",
                RandomOverSampler(
                    random_state=42
                )
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=18,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )


# =========================================================
# XGBOOST
# =========================================================

def create_xgboost_model():

    return ImbPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "oversampler",
                RandomOverSampler(
                    random_state=42
                )
            ),

            (
                "classifier",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,

                    n_estimators=350,
                    learning_rate=0.05,
                    max_depth=6,

                    subsample=0.8,
                    colsample_bytree=0.8,

                    eval_metric="mlogloss",

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

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Per-class metrics
    per_class_precision = precision_score(
        y_test,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    per_class_recall = recall_score(
        y_test,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    per_class_f1 = f1_score(
        y_test,
        predictions,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "f1_macro":
            f1_macro,

        "precision_weighted":
            precision_weighted,

        "recall_weighted":
            recall_weighted,

        "f1_weighted":
            f1_weighted,

        "no_diabetes_precision":
            per_class_precision[0],

        "no_diabetes_recall":
            per_class_recall[0],

        "no_diabetes_f1":
            per_class_f1[0],

        "prediabetes_precision":
            per_class_precision[1],

        "prediabetes_recall":
            per_class_recall[1],

        "prediabetes_f1":
            per_class_f1[1],

        "diabetes_precision":
            per_class_precision[2],

        "diabetes_recall":
            per_class_recall[2],

        "diabetes_f1":
            per_class_f1[2],
    }

    print("\nOverall Metrics")

    print(
        f"Accuracy:            "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro Precision:     "
        f"{precision_macro:.4f}"
    )

    print(
        f"Macro Recall:        "
        f"{recall_macro:.4f}"
    )

    print(
        f"Macro F1:            "
        f"{f1_macro:.4f}"
    )

    print(
        f"Weighted F1:         "
        f"{f1_weighted:.4f}"
    )

    print("\nPer-Class F1")

    print(
        f"No Diabetes:         "
        f"{per_class_f1[0]:.4f}"
    )

    print(
        f"Prediabetes:         "
        f"{per_class_f1[1]:.4f}"
    )

    print(
        f"Diabetes:            "
        f"{per_class_f1[2]:.4f}"
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    print("\nConfusion Matrix:")

    print(matrix)

    report = classification_report(
        y_test,
        predictions,

        labels=[0, 1, 2],

        target_names=[
            "No Diabetes",
            "Prediabetes",
            "Diabetes"
        ],

        digits=4,

        zero_division=0
    )

    print("\nClassification Report:")

    print(report)

    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix.tolist(),
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

        random_state=42,

        stratify=y
    )

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Training records: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing records: "
        f"{len(X_test):,}"
    )


    print("\nTraining class distribution BEFORE oversampling:")

    before_counts = (
        y_train
        .value_counts()
        .sort_index()
    )

    print(
        before_counts
    )


    # Show what oversampling will produce.
    temporary_sampler = RandomOverSampler(
        random_state=42
    )

    X_resampled, y_resampled = (
        temporary_sampler.fit_resample(
            X_train,
            y_train
        )
    )

    print("\nTraining class distribution AFTER oversampling:")

    after_counts = (
        pd.Series(
            y_resampled
        )
        .value_counts()
        .sort_index()
    )

    print(
        after_counts
    )

    print(
        f"\nBalanced training records: "
        f"{len(y_resampled):,}"
    )


    models = [
        (
            "Balanced Logistic Regression",
            create_logistic_model()
        ),

        (
            "Balanced Random Forest",
            create_random_forest_model()
        ),

        (
            "Balanced XGBoost",
            create_xgboost_model()
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
    # COMPARISON
    # =====================================================

    comparison_rows = []

    for result in results:

        row = {
            "model":
                result["name"]
        }

        row.update(
            result["metrics"]
        )

        comparison_rows.append(
            row
        )


    comparison = pd.DataFrame(
        comparison_rows
    )


    comparison = comparison.sort_values(
        by=[
            "f1_macro",
            "prediabetes_f1",
            "diabetes_f1"
        ],

        ascending=False
    )


    print("\n")
    print("=" * 70)
    print("BALANCED MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison[
            [
                "model",
                "accuracy",
                "f1_macro",
                "prediabetes_precision",
                "prediabetes_recall",
                "prediabetes_f1",
                "diabetes_f1",
                "f1_weighted",
            ]
        ].to_string(
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
    print("BEST BALANCED MODEL")
    print("=" * 70)

    print(
        best_model_name
    )


    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = (
        MODEL_DIR
        / "diabetes_model_balanced.joblib"
    )


    joblib.dump(
        best_result["model"],
        model_path
    )


    # =====================================================
    # METADATA
    # =====================================================

    metadata = {

        "model_name":
            best_model_name,

        "model_version":
            "2.0-balanced",

        "problem_type":
            "multiclass_classification",

        "balancing_method":
            "RandomOverSampler",

        "target":
            TARGET,

        "classes": {
            "0": "No Diabetes",
            "1": "Prediabetes",
            "2": "Diabetes",
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
    }


    metadata_path = (
        MODEL_DIR
        / "diabetes_model_balanced_metadata.json"
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

    comparison_path = (
        EVALUATION_DIR
        / "diabetes_balanced_model_comparison.csv"
    )


    comparison.to_csv(
        comparison_path,
        index=False
    )


    report_path = (
        EVALUATION_DIR
        / "diabetes_balanced_classification_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            best_result["report"]
        )


    matrix_path = (
        EVALUATION_DIR
        / "diabetes_balanced_confusion_matrix.json"
    )


    with open(
        matrix_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_result["confusion_matrix"],
            file,
            indent=4
        )


    print("\nSaved:")

    print(
        model_path
    )

    print(
        metadata_path
    )

    print(
        comparison_path
    )

    print(
        report_path
    )

    print(
        matrix_path
    )


    print("\n" + "=" * 70)
    print("BALANCED TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()