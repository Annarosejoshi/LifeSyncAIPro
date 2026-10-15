from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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

DATA_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "processed"
    / "cardiovascular_risk_dataset_v1.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

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
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.50,
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

    return {
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
    }


# ============================================================
# START
# ============================================================

print("=" * 74)
print("LIFESYNC AI PRO - CARDIOVASCULAR V1 CROSS VALIDATION")
print("=" * 74)


# ============================================================
# LOAD DATA
# ============================================================

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)


print(
    f"\nTotal records: {len(df)}"
)


# ============================================================
# VALIDATE
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing_columns)
    )


df = df[
    df[TARGET].notna()
].copy()


df[TARGET] = (
    df[TARGET]
    .astype(int)
)


X = df[FEATURES].copy()
y = df[TARGET].copy()


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


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


model_names = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]


all_results = []


# ============================================================
# MODEL LOOP
# ============================================================

for model_name in model_names:

    print("\n" + "=" * 74)
    print(model_name.upper())
    print("=" * 74)

    fold_results = []


    # ========================================================
    # FOLD LOOP
    # ========================================================

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        X_train = X.iloc[
            train_index
        ]

        X_validation = X.iloc[
            validation_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_validation = y.iloc[
            validation_index
        ]


        # ----------------------------------------------------
        # Calculate class imbalance using ONLY training fold
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        if model_name == "Logistic Regression":

            estimator = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )


        elif model_name == "Random Forest":

            estimator = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )


        else:

            estimator = XGBClassifier(
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
            )


        # ----------------------------------------------------
        # Preprocessing
        #
        # IMPORTANT:
        # preprocessing is fitted separately inside each fold.
        # This prevents data leakage.
        # ----------------------------------------------------

        pipeline = Pipeline(
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
                    estimator,
                ),
            ]
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )


        # ----------------------------------------------------
        # Predict probability
        # ----------------------------------------------------

        probabilities = (
            pipeline.predict_proba(
                X_validation
            )[:, 1]
        )


        # ----------------------------------------------------
        # Evaluate at 0.50
        #
        # We use a fixed threshold during model comparison.
        # Threshold tuning remains a separate step.
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_validation,
            probabilities,
            threshold=0.50,
        )


        fold_results.append(
            metrics
        )


        print(
            f"\nFold {fold}: "
            f"Accuracy={metrics['accuracy'] * 100:.2f}% | "
            f"ROC-AUC={metrics['roc_auc'] * 100:.2f}% | "
            f"PR-AUC={metrics['pr_auc'] * 100:.2f}%"
        )


    # ========================================================
    # AVERAGE
    # ========================================================

    results_df = pd.DataFrame(
        fold_results
    )


    averages = (
        results_df.mean()
    )

    stds = (
        results_df.std()
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
        f"{stds['roc_auc'] * 100:.2f}%"
    )


    all_results.append(
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
                stds[
                    "roc_auc"
                ],
        }
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    all_results
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


print("\n" + "=" * 74)
print("FINAL 5-FOLD MODEL COMPARISON")
print("=" * 74)


for _, row in comparison_df.iterrows():

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
        f"{row['pr_auc'] * 100:.2f}%"
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


print("\nCross-validation complete.")