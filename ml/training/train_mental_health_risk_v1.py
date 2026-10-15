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
    / "mental_health_risk_dataset_v1.csv"
)

MODEL_DIR = BASE_DIR / "ml" / "models"

EVALUATION_DIR = (
    BASE_DIR
    / "ml"
    / "evaluation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODEL_FILE = (
    MODEL_DIR
    / "mental_health_risk_model_v1.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "mental_health_risk_model_v1_metadata.json"
)

COMPARISON_FILE = (
    EVALUATION_DIR
    / "mental_health_risk_model_v1_comparison.csv"
)

REPORT_FILE = (
    EVALUATION_DIR
    / "mental_health_risk_v1_classification_report.txt"
)

CONFUSION_FILE = (
    EVALUATION_DIR
    / "mental_health_risk_v1_confusion_matrix.json"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "depression_risk"

RANDOM_STATE = 42


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
]


# ============================================================
# START
# ============================================================

print("=" * 78)
print("LIFESYNC AI PRO - MENTAL HEALTH RISK MODEL V1")
print("=" * 78)


# ============================================================
# LOAD DATASET
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )


df = pd.read_csv(
    DATA_FILE
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

    raise RuntimeError(
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
    )
    .round(2)
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

FORBIDDEN_FEATURES = [
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",

    "phq9_score",
    "mental_symptom_score",

    TARGET,
]


leakage_features = [
    feature
    for feature in FEATURES
    if feature in FORBIDDEN_FEATURES
]


if leakage_features:

    raise RuntimeError(
        f"TARGET LEAKAGE DETECTED: "
        f"{leakage_features}"
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

    print(
        "-",
        feature
    )


# ============================================================
# TRAIN / VALIDATION / TEST
#
# 70% train
# 15% validation
# 15% test
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

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)


scale_pos_weight = (
    negative_count
    /
    positive_count
)


print(
    "\nTraining class distribution:"
)

print(
    "Lower symptom burden:",
    negative_count
)

print(
    "Elevated symptom burden:",
    positive_count
)

print(
    "XGBoost scale_pos_weight:",
    round(
        scale_pos_weight,
        3
    )
)


# ============================================================
# PREPROCESSORS
# ============================================================

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
            FEATURES,
        ),
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
            FEATURES,
        ),
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":

        Pipeline(
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),


    "Random Forest":

        Pipeline(
            steps=[
                (
                    "preprocessor",
                    tree_preprocessor,
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=12,
                        min_samples_split=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),


    "XGBoost":

        Pipeline(
            steps=[
                (
                    "preprocessor",
                    tree_preprocessor,
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=4,
                        learning_rate=0.03,

                        subsample=0.85,
                        colsample_bytree=0.85,

                        min_child_weight=3,

                        reg_alpha=0.1,
                        reg_lambda=1.0,

                        scale_pos_weight=
                            scale_pos_weight,

                        objective=
                            "binary:logistic",

                        eval_metric=
                            "logloss",

                        random_state=
                            RANDOM_STATE,

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
        model
        .predict_proba(
            X_data
        )[:, 1]
    )


    predictions = (
        probabilities
        >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_data,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )


    specificity = (
        tn
        /
        (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    return {

        "accuracy":
            accuracy_score(
                y_data,
                predictions,
            ),

        "precision":
            precision_score(
                y_data,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_data,
                predictions,
                zero_division=0,
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_data,
                predictions,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_data,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_data,
                probabilities,
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "probabilities":
            probabilities,

        "predictions":
            predictions,
    }


# ============================================================
# TRAIN MODELS
# ============================================================

trained_models = {}

validation_results = []


for model_name, model in models.items():

    print(
        "\n"
        + "=" * 78
    )

    print(
        "TRAINING:",
        model_name
    )

    print(
        "=" * 78
    )


    model.fit(
        X_train,
        y_train
    )


    trained_models[
        model_name
    ] = model


    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
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
    )


    validation_results.append(
        {
            "model":
                model_name,

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "specificity":
                metrics["specificity"],

            "f1":
                metrics["f1"],

            "roc_auc":
                metrics["roc_auc"],

            "pr_auc":
                metrics["pr_auc"],
        }
    )


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
            "roc_auc",
            "pr_auc",
        ],
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n"
    + "=" * 78
)

print(
    "VALIDATION MODEL COMPARISON"
)

print(
    "=" * 78
)


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# FINAL MODEL SELECTION
#
# Cross-validation showed Random Forest had the strongest
# overall ROC-AUC and F1 performance.
# ============================================================

best_model_name = "Random Forest"

best_model = trained_models[
    best_model_name
]

print(
    "\nFinal selected model:",
    best_model_name
)
# ============================================================
# THRESHOLD SEARCH
# ============================================================

validation_probabilities = (
    best_model
    .predict_proba(
        X_validation
    )[:, 1]
)


threshold_results = []


for threshold in np.arange(
    0.20,
    0.81,
    0.01,
):

    predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)


    tn, fp, fn, tp = (
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )


    recall = (
        tp
        /
        (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    specificity = (
        tn
        /
        (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    precision = (
        precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )
    )


    f1 = (
        f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )
    )


    balanced_accuracy = (
        recall
        +
        specificity
    ) / 2


    threshold_results.append(
        {
            "threshold":
                float(
                    threshold
                ),

            "precision":
                precision,

            "recall":
                recall,

            "specificity":
                specificity,

            "f1":
                f1,

            "balanced_accuracy":
                balanced_accuracy,
        }
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# THRESHOLD SELECTION
#
# For screening, prioritize sensitivity but avoid
# extremely low specificity.
# ============================================================

acceptable = threshold_df[
    (
        threshold_df["recall"]
        >= 0.70
    )
    &
    (
        threshold_df["specificity"]
        >= 0.60
    )
]


if len(acceptable) > 0:

    selected_row = (
        acceptable
        .sort_values(
            by=[
                "f1",
                "balanced_accuracy",
            ],
            ascending=False,
        )
        .iloc[0]
    )

else:

    selected_row = (
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


selected_threshold = float(
    selected_row[
        "threshold"
    ]
)


print(
    "\n"
    + "=" * 78
)

print(
    "SELECTED VALIDATION THRESHOLD"
)

print(
    "=" * 78
)


print(
    selected_row
)


# ============================================================
# FINAL TEST
# ============================================================

test_metrics = evaluate_model(
    best_model,
    X_test,
    y_test,
    threshold=
        selected_threshold,
)


print(
    "\n"
    + "=" * 78
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 78
)


print(
    "Model:",
    best_model_name
)


print(
    f"Threshold:   "
    f"{selected_threshold:.2f}"
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


print(
    "\nConfusion Matrix:"
)


print(
    np.array(
        [
            [
                test_metrics["tn"],
                test_metrics["fp"],
            ],
            [
                test_metrics["fn"],
                test_metrics["tp"],
            ],
        ]
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

classification_text = (
    classification_report(
        y_test,
        test_metrics[
            "predictions"
        ],
        target_names=[
            "Lower Symptom Burden",
            "Elevated Symptom Burden",
        ],
        zero_division=0,
    )
)


print(
    "\nClassification Report:"
)

print(
    classification_text
)


# ============================================================
# BASELINE
# ============================================================

majority_baseline = (
    y_test
    .value_counts(
        normalize=True
    )
    .max()
)


print(
    "\n"
    + "=" * 78
)

print(
    "BASELINE COMPARISON"
)

print(
    "=" * 78
)


print(
    f"Majority baseline accuracy: "
    f"{majority_baseline:.4f}"
)

print(
    f"Selected model accuracy:    "
    f"{test_metrics['accuracy']:.4f}"
)


print(
    "\nNOTE:"
)

print(
    "Because the target is imbalanced, "
    "ROC-AUC, PR-AUC, recall and specificity "
    "are more informative than accuracy alone."
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    MODEL_FILE
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "model_name":
        best_model_name,

    "model_version":
        "v1",

    "prediction_type":
        "mental_health_risk",

    "target":
        TARGET,

    "target_definition":
        (
            "Elevated depressive symptom burden "
            "defined as complete NHANES PHQ-9 score >= 10."
        ),

    "features":
        FEATURES,

    "selected_threshold":
        selected_threshold,

    "dataset_rows":
        int(
            len(df)
        ),

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

        "accuracy":
            float(
                test_metrics[
                    "accuracy"
                ]
            ),

        "precision":
            float(
                test_metrics[
                    "precision"
                ]
            ),

        "recall":
            float(
                test_metrics[
                    "recall"
                ]
            ),

        "specificity":
            float(
                test_metrics[
                    "specificity"
                ]
            ),

        "f1":
            float(
                test_metrics[
                    "f1"
                ]
            ),

        "roc_auc":
            float(
                test_metrics[
                    "roc_auc"
                ]
            ),

        "pr_auc":
            float(
                test_metrics[
                    "pr_auc"
                ]
            ),
    },

    "notes": [
        (
            "DPQ010-DPQ090 are used only "
            "to construct the target."
        ),
        (
            "PHQ-9 score is not included "
            "as a predictor."
        ),
        (
            "mental_symptom_score is excluded "
            "to prevent target leakage."
        ),
        (
            "This model estimates elevated "
            "depressive symptom burden and "
            "is not a clinical diagnosis."
        ),
    ],
}


with open(
    METADATA_FILE,
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
    COMPARISON_FILE,
    index=False,
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        classification_text
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
    CONFUSION_FILE,
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

print(
    "\n"
    + "=" * 78
)

print(
    "FILES SAVED"
)

print(
    "=" * 78
)


print(
    "\nModel:"
)

print(
    MODEL_FILE
)


print(
    "\nMetadata:"
)

print(
    METADATA_FILE
)


print(
    "\nComparison:"
)

print(
    COMPARISON_FILE
)


print(
    "\nClassification Report:"
)

print(
    REPORT_FILE
)


print(
    "\nConfusion Matrix:"
)

print(
    CONFUSION_FILE
)