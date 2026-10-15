from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

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
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
):

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),
    }


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(
    model_name,
    scale_pos_weight,
):

    if model_name == "Logistic Regression":

        return Pipeline(
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

                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )


    if model_name == "Random Forest":

        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
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
        )


    if model_name == "XGBoost":

        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
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
                        scale_pos_weight=scale_pos_weight,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )


    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("LIFESYNC AI PRO - MENTAL HEALTH V1 CROSS VALIDATION")
    print("=" * 78)


    # ========================================================
    # LOAD DATA
    # ========================================================

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )


    df = pd.read_csv(
        DATA_FILE
    )


    print(
        "\nTotal records:",
        len(df)
    )


    # ========================================================
    # VALIDATE COLUMNS
    # ========================================================

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


    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    forbidden = [
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


    leakage = [
        feature
        for feature in FEATURES
        if feature in forbidden
    ]


    if leakage:

        raise RuntimeError(
            f"TARGET LEAKAGE DETECTED: {leakage}"
        )


    # ========================================================
    # X / Y
    # ========================================================

    X = df[
        FEATURES
    ].copy()

    y = df[
        TARGET
    ].astype(int)


    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
    )


    print(
        "\nTarget percentages:"
    )

    print(
        (
            y.value_counts(
                normalize=True
            )
            .sort_index()
            * 100
        ).round(2)
    )


    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


    model_names = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]


    final_results = []


    for model_name in model_names:

        print(
            "\n"
            + "=" * 78
        )

        print(
            model_name.upper()
        )

        print(
            "=" * 78
        )


        fold_results = []


        for fold, (
            train_index,
            test_index,
        ) in enumerate(
            cv.split(
                X,
                y,
            ),
            start=1,
        ):

            X_train = X.iloc[
                train_index
            ]

            X_test = X.iloc[
                test_index
            ]

            y_train = y.iloc[
                train_index
            ]

            y_test = y.iloc[
                test_index
            ]


            negative_count = int(
                (y_train == 0)
                .sum()
            )

            positive_count = int(
                (y_train == 1)
                .sum()
            )


            scale_pos_weight = (
                negative_count
                /
                positive_count
            )


            model = build_model(
                model_name,
                scale_pos_weight,
            )


            # ================================================
            # TRAIN
            # ================================================

            model.fit(
                X_train,
                y_train,
            )


            probabilities = (
                model
                .predict_proba(
                    X_test
                )[:, 1]
            )


            metrics = calculate_metrics(
                y_test,
                probabilities,
                threshold=0.50,
            )


            fold_results.append(
                metrics
            )


            print(
                f"\nFold {fold}: "
                f"Accuracy="
                f"{metrics['accuracy'] * 100:.2f}% | "
                f"ROC-AUC="
                f"{metrics['roc_auc'] * 100:.2f}% | "
                f"PR-AUC="
                f"{metrics['pr_auc'] * 100:.2f}%"
            )


        # ====================================================
        # FOLD AVERAGES
        # ====================================================

        fold_df = pd.DataFrame(
            fold_results
        )


        averages = (
            fold_df.mean()
        )


        roc_std = (
            fold_df[
                "roc_auc"
            ].std()
        )


        pr_std = (
            fold_df[
                "pr_auc"
            ].std()
        )


        print(
            "\nAVERAGE RESULTS"
        )


        print(
            f"ACCURACY   : "
            f"{averages['accuracy'] * 100:.2f}%"
        )

        print(
            f"PRECISION  : "
            f"{averages['precision'] * 100:.2f}%"
        )

        print(
            f"RECALL     : "
            f"{averages['recall'] * 100:.2f}%"
        )

        print(
            f"SPECIFICITY: "
            f"{averages['specificity'] * 100:.2f}%"
        )

        print(
            f"F1         : "
            f"{averages['f1'] * 100:.2f}%"
        )

        print(
            f"ROC_AUC    : "
            f"{averages['roc_auc'] * 100:.2f}%"
        )

        print(
            f"PR_AUC     : "
            f"{averages['pr_auc'] * 100:.2f}%"
        )

        print(
            f"ROC-AUC STD: "
            f"{roc_std * 100:.2f}%"
        )

        print(
            f"PR-AUC STD : "
            f"{pr_std * 100:.2f}%"
        )


        final_results.append(
            {
                "model":
                    model_name,

                "accuracy":
                    averages[
                        "accuracy"
                    ],

                "precision":
                    averages[
                        "precision"
                    ],

                "recall":
                    averages[
                        "recall"
                    ],

                "specificity":
                    averages[
                        "specificity"
                    ],

                "f1":
                    averages[
                        "f1"
                    ],

                "roc_auc":
                    averages[
                        "roc_auc"
                    ],

                "pr_auc":
                    averages[
                        "pr_auc"
                    ],

                "roc_auc_std":
                    roc_std,

                "pr_auc_std":
                    pr_std,
            }
        )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        final_results
    )


    comparison = (
        comparison
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
        "FINAL 5-FOLD MODEL COMPARISON"
    )

    print(
        "=" * 78
    )


    for _, row in comparison.iterrows():

        print(
            f"\n{row['model']}"
        )

        print(
            f"Accuracy:    "
            f"{row['accuracy'] * 100:.2f}%"
        )

        print(
            f"ROC-AUC:     "
            f"{row['roc_auc'] * 100:.2f}% "
            f"± "
            f"{row['roc_auc_std'] * 100:.2f}%"
        )

        print(
            f"PR-AUC:      "
            f"{row['pr_auc'] * 100:.2f}% "
            f"± "
            f"{row['pr_auc_std'] * 100:.2f}%"
        )

        print(
            f"Precision:   "
            f"{row['precision'] * 100:.2f}%"
        )

        print(
            f"Recall:      "
            f"{row['recall'] * 100:.2f}%"
        )

        print(
            f"Specificity: "
            f"{row['specificity'] * 100:.2f}%"
        )

        print(
            f"F1:          "
            f"{row['f1'] * 100:.2f}%"
        )


    print(
        "\nCross-validation complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()