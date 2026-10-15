from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "processed"
    / "sleep_risk_dataset_v2.csv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "age",
    "sex",
    "bmi",
    "income_ratio",
    "obesity_flag",

    "moderate_sessions_week",
    "moderate_minutes_session",
    "vigorous_sessions_week",
    "vigorous_minutes_session",
    "total_activity_minutes_week",
    "sedentary_minutes_day",
    "high_sedentary_flag",

    "little_interest",
    "depressed_mood",
    "low_energy",
    "appetite_problem",
    "negative_self_view",
    "concentration_problem",
    "movement_problem",
    "mental_symptom_score",

    "ever_smoked_100",
    "current_smoking_status",

    "alcohol_past_year",
    "alcohol_frequency",
    "average_drinks",
    "heavy_drinking_frequency",

    "work_hours_week",
    "long_work_week",
]

TARGET = "short_sleep"


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].astype(int)


print("=" * 70)
print("LIFESYNC AI PRO - SLEEP MODEL V2 CROSS VALIDATION")
print("=" * 70)

print("\nTotal records:", len(df))

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nTarget percentages:")
print(
    (
        y.value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
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
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]),

    "Random Forest": Pipeline([
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
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]),

    "XGBoost": Pipeline([
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
                scale_pos_weight=3.78,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]),
}


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


all_results = []


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name.upper())
    print("=" * 70)

    fold_results = []

    for fold, (train_index, test_index) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        result = {
            "accuracy": accuracy_score(
                y_test,
                predictions,
            ),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(
                y_test,
                probabilities,
            ),
            "pr_auc": average_precision_score(
                y_test,
                probabilities,
            ),
        }

        fold_results.append(result)

        print(
            f"\nFold {fold}: "
            f"Accuracy={result['accuracy'] * 100:.2f}% | "
            f"ROC-AUC={result['roc_auc'] * 100:.2f}% | "
            f"PR-AUC={result['pr_auc'] * 100:.2f}%"
        )


    # ========================================================
    # AVERAGES
    # ========================================================

    print("\nAVERAGE RESULTS")

    summary = {
        metric: np.mean([
            result[metric]
            for result in fold_results
        ])
        for metric in fold_results[0]
    }

    roc_std = np.std([
        result["roc_auc"]
        for result in fold_results
    ])

    for metric, value in summary.items():
        print(
            f"{metric.upper():10s}: "
            f"{value * 100:.2f}%"
        )

    print(
        f"ROC-AUC STD: {roc_std * 100:.2f}%"
    )

    all_results.append({
        "model": model_name,
        **summary,
        "roc_auc_std": roc_std,
    })


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    all_results
)

results_df = results_df.sort_values(
    "roc_auc",
    ascending=False,
)


print("\n" + "=" * 70)
print("FINAL 5-FOLD MODEL COMPARISON")
print("=" * 70)

for _, row in results_df.iterrows():

    print(f"\n{row['model']}")

    print(
        f"Accuracy:  "
        f"{row['accuracy'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC:   "
        f"{row['roc_auc'] * 100:.2f}% "
        f"± {row['roc_auc_std'] * 100:.2f}%"
    )

    print(
        f"PR-AUC:    "
        f"{row['pr_auc'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{row['precision'] * 100:.2f}%"
    )

    print(
        f"Recall:    "
        f"{row['recall'] * 100:.2f}%"
    )

    print(
        f"F1:        "
        f"{row['f1'] * 100:.2f}%"
    )


print("\nCross-validation complete.")