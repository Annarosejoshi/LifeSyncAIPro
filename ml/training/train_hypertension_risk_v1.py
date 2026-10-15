from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "ml"
    / "data"
    / "processed"
    / "hypertension_risk_dataset_v1.csv"
)

MODEL_DIR = BASE_DIR / "ml" / "models"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = (
    MODEL_DIR
    / "hypertension_risk_model_v1.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "hypertension_risk_model_v1_metadata.json"
)

COMPARISON_FILE = (
    EVAL_DIR
    / "hypertension_risk_model_v1_comparison.csv"
)

REPORT_FILE = (
    EVAL_DIR
    / "hypertension_risk_v1_classification_report.txt"
)

CONFUSION_FILE = (
    EVAL_DIR
    / "hypertension_risk_v1_confusion_matrix.json"
)


print("=" * 78)
print("LIFESYNC AI PRO - HYPERTENSION RISK MODEL V1")
print("=" * 78)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded.")
print("\nRows:", len(df))
print("Columns:", len(df.columns))

TARGET = "hypertension"

print("\nTarget distribution:")
print(df[TARGET].value_counts())

print("\nTarget percentages:")
print(
    (
        df[TARGET]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "age",
    "sex",
    "income_ratio",

    "bmi",
    "waist_circumference",
    "obesity_flag",

    "ever_smoked_100",
    "smoking_status",
    "current_smoker",

    "moderate_sessions_week",
    "moderate_minutes_session",

    "vigorous_sessions_week",
    "vigorous_minutes_session",

    "total_activity_minutes_week",
    "sedentary_minutes_day",

    "sleep_hours",
    "short_sleep_flag",

    "total_cholesterol",
    "hdl_cholesterol",

    "alcohol_past_year",
    "alcohol_frequency",
    "average_drinks",

    "mental_symptom_score",
]


# ============================================================
# LEAKAGE SAFETY CHECK
# ============================================================

FORBIDDEN = [
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "BPQ020",
    "BPQ030",
    "BPQ150",
    "diagnosed_hypertension",
    "bp_medication",
    "exam_hypertension",
]


leaks = [
    feature
    for feature in FEATURES
    if feature in FORBIDDEN
]

if leaks:
    raise RuntimeError(
        f"TARGET LEAKAGE DETECTED: {leaks}"
    )


missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise RuntimeError(
        f"Missing required features: {missing_features}"
    )


print("\nInput features:", len(FEATURES))

for feature in FEATURES:
    print("-", feature)


X = df[FEATURES].copy()
y = df[TARGET].astype(int)


# ============================================================
# TRAIN / VALIDATION / TEST
# 70 / 15 / 15
# ============================================================

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )
)


X_val, X_test, y_val, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )
)


print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = FEATURES


linear_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median",
                            add_indicator=True,
                        ),
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                ]
            ),
            numeric_features,
        )
    ]
)


tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
            numeric_features,
        )
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": Pipeline(
        steps=[
            (
                "preprocessor",
                linear_preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),

    "XGBoost": Pipeline(
        steps=[
            (
                "preprocessor",
                tree_preprocessor,
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(
    model,
    X_data,
    y_data,
    threshold=0.50,
):

    probabilities = (
        model.predict_proba(X_data)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_data,
            predictions
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy": accuracy_score(
            y_data,
            predictions
        ),

        "precision": precision_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_data,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_data,
            probabilities
        ),

        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],

        "probabilities": probabilities,

        "predictions": predictions,
    }


# ============================================================
# TRAIN + VALIDATE
# ============================================================

validation_results = []

trained_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 78)
    print("TRAINING:", model_name)
    print("=" * 78)

    model.fit(
        X_train,
        y_train
    )

    trained_models[
        model_name
    ] = model

    metrics = evaluate_model(
        model,
        X_val,
        y_val,
        threshold=0.50,
    )

    print(
        f"\nAccuracy:    "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:   "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:      "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"Specificity: "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1 Score:    "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:     "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:      "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    validation_results.append(
        {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        }
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    validation_results
)

comparison = comparison.sort_values(
    by=[
        "roc_auc",
        "f1",
    ],
    ascending=False,
)


print("\n" + "=" * 78)
print("VALIDATION MODEL COMPARISON")
print("=" * 78)

print(
    comparison.to_string(
        index=False
    )
)


best_model_name = (
    comparison.iloc[0]["model"]
)

best_model = (
    trained_models[
        best_model_name
    ]
)


print(
    "\nBest validation model:",
    best_model_name
)


# ============================================================
# THRESHOLD SELECTION
# ============================================================

val_probabilities = (
    best_model
    .predict_proba(X_val)[:, 1]
)


threshold_results = []


for threshold in np.arange(
    0.20,
    0.81,
    0.01
):

    predictions = (
        val_probabilities
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_val,
            predictions
        ).ravel()
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy":
                balanced_accuracy,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# Select threshold using balanced accuracy
best_threshold_row = (
    threshold_df
    .sort_values(
        by=[
            "balanced_accuracy",
            "f1",
        ],
        ascending=False,
    )
    .iloc[0]
)


best_threshold = float(
    best_threshold_row[
        "threshold"
    ]
)


print("\n" + "=" * 78)
print("SELECTED VALIDATION THRESHOLD")
print("=" * 78)

print(best_threshold_row)


# ============================================================
# FINAL TEST
# ============================================================

test_metrics = evaluate_model(
    best_model,
    X_test,
    y_test,
    threshold=best_threshold,
)


print("\n" + "=" * 78)
print("FINAL TEST RESULTS")
print("=" * 78)

print("Model:", best_model_name)
print(
    f"Threshold:   "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy:    "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Precision:   "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall:      "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Specificity: "
    f"{test_metrics['specificity']:.4f}"
)

print(
    f"F1 Score:    "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC:     "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC:      "
    f"{test_metrics['pr_auc']:.4f}"
)


print("\nConfusion Matrix:")

print(
    np.array(
        test_metrics[
            "confusion_matrix"
        ]
    )
)


report = classification_report(
    y_test,
    test_metrics[
        "predictions"
    ],
    target_names=[
        "No Hypertension",
        "Hypertension",
    ],
    zero_division=0,
)


print("\nClassification Report:")
print(report)


# ============================================================
# BASELINE
# ============================================================

majority_accuracy = (
    y_test
    .value_counts(
        normalize=True
    )
    .max()
)


print("\n" + "=" * 78)
print("BASELINE COMPARISON")
print("=" * 78)

print(
    f"Majority baseline accuracy: "
    f"{majority_accuracy:.4f}"
)

print(
    f"Selected model accuracy:    "
    f"{test_metrics['accuracy']:.4f}"
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    best_model,
    MODEL_FILE
)


metadata = {
    "model_name": best_model_name,
    "target": TARGET,
    "features": FEATURES,
    "threshold": best_threshold,

    "dataset_rows": len(df),

    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test),

    "test_accuracy":
        test_metrics["accuracy"],

    "test_precision":
        test_metrics["precision"],

    "test_recall":
        test_metrics["recall"],

    "test_specificity":
        test_metrics["specificity"],

    "test_f1":
        test_metrics["f1"],

    "test_roc_auc":
        test_metrics["roc_auc"],

    "test_pr_auc":
        test_metrics["pr_auc"],
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(report)


with open(
    CONFUSION_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        test_metrics[
            "confusion_matrix"
        ],
        file,
        indent=4,
    )


print("\n" + "=" * 78)
print("FILES SAVED")
print("=" * 78)

print("\nModel:")
print(MODEL_FILE)

print("\nMetadata:")
print(METADATA_FILE)

print("\nComparison:")
print(COMPARISON_FILE)

print("\nClassification Report:")
print(REPORT_FILE)

print("\nConfusion Matrix:")
print(CONFUSION_FILE)