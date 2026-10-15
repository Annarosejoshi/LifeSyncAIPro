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

DATA_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "processed"
    / "cardiovascular_risk_dataset_v1.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "ml"
    / "models"
)

EVALUATION_DIR = (
    BASE_DIR
    / "ml"
    / "evaluation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODEL_DIR
    / "cardiovascular_risk_model_v1.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "cardiovascular_risk_model_v1_metadata.json"
)

COMPARISON_PATH = (
    EVALUATION_DIR
    / "cardiovascular_risk_model_v1_comparison.csv"
)

REPORT_PATH = (
    EVALUATION_DIR
    / "cardiovascular_risk_v1_classification_report.txt"
)

CONFUSION_PATH = (
    EVALUATION_DIR
    / "cardiovascular_risk_v1_confusion_matrix.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET = "cardiovascular_disease"


FEATURES = [
    "age",
    "sex",
    "income_ratio",

    "bmi",
    "obesity_flag",

    "avg_systolic_bp",
    "avg_diastolic_bp",
    "avg_pulse",

    "high_systolic_flag",
    "high_diastolic_flag",

    "total_cholesterol",
    "hdl_cholesterol",

    "high_total_cholesterol_flag",
    "low_hdl_flag",

    "ever_smoked_100",
    "current_smoking_status",

    "moderate_sessions_week",
    "moderate_minutes_session",

    "vigorous_sessions_week",
    "vigorous_minutes_session",

    "total_activity_minutes_week",
    "sedentary_minutes_day",
]


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    metrics = {
        "threshold": float(threshold),

        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    name,
    metrics,
):

    print(f"\nModel: {name}")

    print(
        f"Threshold:   "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy:    "
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

    print("\nConfusion Matrix:")

    print(
        [
            [
                metrics["tn"],
                metrics["fp"],
            ],
            [
                metrics["fn"],
                metrics["tp"],
            ],
        ]
    )


# ============================================================
# START
# ============================================================

print("=" * 74)
print("LIFESYNC AI PRO - CARDIOVASCULAR RISK MODEL V1")
print("=" * 74)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(
    DATA_PATH
)


print("\nDataset loaded.")

print(
    "\nRows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = (
    FEATURES
    +
    [TARGET]
)


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# TARGET
# ============================================================

df = df[
    df[TARGET].notna()
].copy()


df[TARGET] = (
    df[TARGET]
    .astype(int)
)


print(
    "\nTarget distribution:"
)

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


print(
    "\nTarget percentages:"
)

print(
    (
        df[TARGET]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    ).round(2)
)


# ============================================================
# INPUT FEATURES
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].copy()


print(
    "\nInput features:",
    len(FEATURES)
)


for feature in FEATURES:
    print("-", feature)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
#
# 70% Training
# 15% Validation
# 15% Testing
# ============================================================

X_train, X_temp, y_train, y_temp = (
    train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


X_validation, X_test, y_validation, y_test = (
    train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
)


print(
    "\nDataset split:"
)

print(
    "Training:",
    len(X_train)
)

print(
    "Validation:",
    len(X_validation)
)

print(
    "Testing:",
    len(X_test)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_count
    /
    positive_count
)


print(
    "\nTraining class distribution:"
)

print(
    "No CVD:",
    negative_count
)

print(
    "CVD:",
    positive_count
)

print(
    "XGBoost scale_pos_weight:",
    f"{scale_pos_weight:.3f}"
)


# ============================================================
# PREPROCESSING
#
# Median imputation + missing indicators.
# Scaling is useful for Logistic Regression.
# Tree models are fine with scaled inputs as well.
# ============================================================

preprocessor = Pipeline(
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
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
}


# ============================================================
# TRAIN AND VALIDATE
# ============================================================

validation_results = []

trained_models = {}


for name, estimator in models.items():

    print("\n" + "=" * 74)

    print(
        f"TRAINING: {name}"
    )

    print("=" * 74)


    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


    pipeline.fit(
        X_train,
        y_train,
    )


    probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )


    metrics = calculate_metrics(
        y_validation,
        probabilities,
        threshold=0.50,
    )


    print_metrics(
        name,
        metrics,
    )


    predictions = (
        probabilities >= 0.50
    ).astype(int)


    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_validation,
            predictions,
            target_names=[
                "No CVD",
                "CVD",
            ],
            zero_division=0,
        )
    )


    validation_results.append(
        {
            "model": name,
            **metrics,
        }
    )


    trained_models[
        name
    ] = pipeline


# ============================================================
# VALIDATION COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    validation_results
)


comparison_df = (
    comparison_df
    .sort_values(
    by=[
        "balanced_accuracy",
        "f1",
    ],
    ascending=False,
)
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 74)
print("VALIDATION MODEL COMPARISON")
print("=" * 74)


display_columns = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
]


print(
    comparison_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SELECT MODEL
#
# Primary selection metric = ROC-AUC
# Secondary = PR-AUC
# ============================================================

best_model_name = (
    comparison_df
    .iloc[0]["model"]
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
# THRESHOLD SEARCH
#
# For screening-style risk detection we want useful recall,
# but we should not destroy specificity.
#
# Search for threshold with:
# recall >= 0.70
# and choose highest F1.
# ============================================================

validation_probabilities = (
    best_model.predict_proba(
        X_validation
    )[:, 1]
)


threshold_results = []


for threshold in np.arange(
    0.20,
    0.81,
    0.01,
):

    metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold=float(threshold),
    )

    threshold_results.append(
        metrics
    )


threshold_df = pd.DataFrame(
    threshold_results
)


eligible_thresholds = (
    threshold_df[
        threshold_df["recall"] >= 0.70
    ]
)


if len(eligible_thresholds) > 0:

    selected_threshold_row = (
        eligible_thresholds
        .sort_values(
            by=[
                "f1",
                "specificity",
            ],
            ascending=False,
        )
        .iloc[0]
    )

else:

    selected_threshold_row = (
        threshold_df
        .sort_values(
            by="f1",
            ascending=False,
        )
        .iloc[0]
    )


selected_threshold = float(
    selected_threshold_row[
        "threshold"
    ]
)


print("\n" + "=" * 74)
print("SELECTED VALIDATION THRESHOLD")
print("=" * 74)


print(
    selected_threshold_row[
        [
            "threshold",
            "precision",
            "recall",
            "specificity",
            "f1",
        ]
    ]
)


# ============================================================
# FINAL TEST
#
# Test set has NOT been used for model selection or threshold
# selection.
# ============================================================

test_probabilities = (
    best_model.predict_proba(
        X_test
    )[:, 1]
)


test_metrics = calculate_metrics(
    y_test,
    test_probabilities,
    threshold=selected_threshold,
)


print("\n" + "=" * 74)
print("FINAL TEST RESULTS")
print("=" * 74)


print_metrics(
    best_model_name,
    test_metrics,
)


test_predictions = (
    test_probabilities
    >= selected_threshold
).astype(int)


final_report = classification_report(
    y_test,
    test_predictions,
    target_names=[
        "No CVD",
        "CVD",
    ],
    zero_division=0,
)


print(
    "\nClassification Report:"
)

print(
    final_report
)


# ============================================================
# BASELINE
#
# Majority classifier predicts No CVD for everyone.
# ============================================================

baseline_predictions = np.zeros(
    len(y_test),
    dtype=int,
)


baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)


print("\n" + "=" * 74)
print("BASELINE COMPARISON")
print("=" * 74)


print(
    f"Majority-class baseline accuracy: "
    f"{baseline_accuracy:.4f}"
)


print(
    f"Selected model accuracy:          "
    f"{test_metrics['accuracy']:.4f}"
)


print(
    "\nNOTE:"
)

print(
    "Accuracy is NOT the primary metric because the "
    "cardiovascular target is imbalanced."
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model_name":
        best_model_name,

    "model_version":
        "v1",

    "target":
        TARGET,

    "target_definition":
        (
            "Reported history of at least one of: "
            "congestive heart failure, coronary heart disease, "
            "angina, heart attack, or stroke."
        ),

    "features":
        FEATURES,

    "selected_threshold":
        selected_threshold,

    "training_rows":
        int(
            len(X_train)
        ),

    "validation_rows":
        int(
            len(X_validation)
        ),

    "test_rows":
        int(
            len(X_test)
        ),

    "positive_prevalence":
        float(
            y.mean()
        ),

    "test_metrics": {
        key: float(value)
        if isinstance(
            value,
            (
                np.floating,
                float,
            ),
        )
        else int(value)
        if isinstance(
            value,
            (
                np.integer,
                int,
            ),
        )
        else value

        for key, value
        in test_metrics.items()
    },
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )


# ============================================================
# SAVE EVALUATION
# ============================================================

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        f"Model: {best_model_name}\n"
    )

    file.write(
        f"Threshold: "
        f"{selected_threshold:.2f}\n\n"
    )

    file.write(
        final_report
    )


confusion_data = {
    "tn":
        test_metrics["tn"],

    "fp":
        test_metrics["fp"],

    "fn":
        test_metrics["fn"],

    "tp":
        test_metrics["tp"],
}


with open(
    CONFUSION_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        confusion_data,
        file,
        indent=4,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 74)
print("FILES SAVED")
print("=" * 74)


print(
    "\nModel:\n",
    MODEL_PATH,
    sep=""
)

print(
    "\nMetadata:\n",
    METADATA_PATH,
    sep=""
)

print(
    "\nComparison:\n",
    COMPARISON_PATH,
    sep=""
)

print(
    "\nClassification Report:\n",
    REPORT_PATH,
    sep=""
)

print(
    "\nConfusion Matrix:\n",
    CONFUSION_PATH,
    sep=""
)