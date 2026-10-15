from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


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
    / "sleep_risk_dataset_v2.csv"
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
    / "sleep_duration_regression_model.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "sleep_duration_regression_metadata.json"
)

COMPARISON_PATH = (
    EVALUATION_DIR
    / "sleep_duration_regression_comparison.csv"
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


# IMPORTANT:
# We now predict the actual reported sleep duration,
# NOT the short_sleep binary variable.

TARGET = "SLD012"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("LIFESYNC AI PRO - SLEEP DURATION REGRESSION")
print("=" * 72)

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )


df = pd.read_csv(
    DATA_PATH
)


print("\nDataset loaded.")

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features: "
        + ", ".join(missing_features)
    )

if TARGET not in df.columns:

    raise ValueError(
        f"Target {TARGET} not found."
    )


# ============================================================
# CLEAN TARGET
# ============================================================

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


df = df[
    df[TARGET].between(
        2,
        14
    )
].copy()


print(
    "\nUsable records:",
    len(df)
)


print(
    "\nSleep duration statistics:"
)

print(
    df[TARGET].describe()
)


# ============================================================
# X / Y
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].copy()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
)


print("\nDataset split:")

print(
    "Training:",
    len(X_train)
)

print(
    "Testing:",
    len(X_test)
)


# ============================================================
# PREPROCESSING
# ============================================================

scaled_preprocessing = Pipeline(
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


tree_preprocessing = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        ),
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Ridge Regression": Pipeline(
        steps=[
            (
                "preprocessing",
                scaled_preprocessing,
            ),
            (
                "model",
                Ridge(
                    alpha=1.0
                ),
            ),
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessing",
                tree_preprocessing,
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),

    "Gradient Boosting": Pipeline(
        steps=[
            (
                "preprocessing",
                tree_preprocessing,
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=3,
                    min_samples_leaf=5,
                    random_state=42,
                ),
            ),
        ]
    ),

    "XGBoost": Pipeline(
        steps=[
            (
                "preprocessing",
                tree_preprocessing,
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=3,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


results = []


print("\n" + "=" * 72)
print("5-FOLD CROSS VALIDATION")
print("=" * 72)


for name, model in models.items():

    print(
        f"\nEvaluating: {name}"
    )

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae":
                "neg_mean_absolute_error",

            "mse":
                "neg_mean_squared_error",

            "r2":
                "r2",
        },
        n_jobs=-1,
    )


    mae_values = (
        -scores[
            "test_mae"
        ]
    )

    mse_values = (
        -scores[
            "test_mse"
        ]
    )

    rmse_values = np.sqrt(
        mse_values
    )

    r2_values = (
        scores[
            "test_r2"
        ]
    )


    mean_mae = np.mean(
        mae_values
    )

    mean_rmse = np.mean(
        rmse_values
    )

    mean_r2 = np.mean(
        r2_values
    )

    std_r2 = np.std(
        r2_values
    )


    print(
        f"Mean MAE:  "
        f"{mean_mae:.3f} hours"
    )

    print(
        f"Mean RMSE: "
        f"{mean_rmse:.3f} hours"
    )

    print(
        f"Mean R2:   "
        f"{mean_r2:.4f}"
    )

    print(
        f"R2 STD:    "
        f"{std_r2:.4f}"
    )


    results.append({
        "model":
            name,

        "cv_mae":
            mean_mae,

        "cv_rmse":
            mean_rmse,

        "cv_r2":
            mean_r2,

        "cv_r2_std":
            std_r2,
    })


# ============================================================
# COMPARE
# ============================================================

comparison_df = pd.DataFrame(
    results
)


comparison_df = (
    comparison_df
    .sort_values(
        [
            "cv_mae",
            "cv_rmse",
        ],
        ascending=True
    )
)


print("\n" + "=" * 72)
print("CROSS-VALIDATION COMPARISON")
print("=" * 72)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = (
    comparison_df
    .iloc[0]["model"]
)

best_model = (
    models[
        best_model_name
    ]
)


print(
    "\nSelected model:",
    best_model_name
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

best_model.fit(
    X_train,
    y_train
)


# ============================================================
# UNTOUCHED TEST SET
# ============================================================

predictions = (
    best_model.predict(
        X_test
    )
)


# Sleep duration should remain
# within plausible range.

predictions = np.clip(
    predictions,
    2,
    14
)


test_mae = mean_absolute_error(
    y_test,
    predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

test_r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 72)
print("FINAL TEST RESULTS")
print("=" * 72)


print(
    "Model:",
    best_model_name
)

print(
    f"MAE:  "
    f"{test_mae:.3f} hours"
)

print(
    f"RMSE: "
    f"{test_rmse:.3f} hours"
)

print(
    f"R2:   "
    f"{test_r2:.4f}"
)


# ============================================================
# INTERPRETABLE ERROR
# ============================================================

absolute_errors = np.abs(
    y_test.to_numpy()
    -
    predictions
)


within_30_minutes = np.mean(
    absolute_errors <= 0.5
)

within_1_hour = np.mean(
    absolute_errors <= 1.0
)

within_2_hours = np.mean(
    absolute_errors <= 2.0
)


print(
    "\nPrediction error:"
)

print(
    f"Within 30 minutes: "
    f"{within_30_minutes * 100:.2f}%"
)

print(
    f"Within 1 hour:     "
    f"{within_1_hour * 100:.2f}%"
)

print(
    f"Within 2 hours:    "
    f"{within_2_hours * 100:.2f}%"
)


# ============================================================
# BASELINE
# ============================================================
#
# Important:
# compare ML against simply predicting
# the training-set mean for everybody.
# ============================================================

baseline_value = (
    y_train.mean()
)

baseline_predictions = np.full(
    len(y_test),
    baseline_value
)


baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


print("\n" + "=" * 72)
print("BASELINE COMPARISON")
print("=" * 72)

print(
    f"Baseline prediction: "
    f"{baseline_value:.3f} hours"
)

print(
    f"Baseline MAE:  "
    f"{baseline_mae:.3f}"
)

print(
    f"Model MAE:     "
    f"{test_mae:.3f}"
)

print(
    f"Baseline RMSE: "
    f"{baseline_rmse:.3f}"
)

print(
    f"Model RMSE:    "
    f"{test_rmse:.3f}"
)

print(
    f"Baseline R2:   "
    f"{baseline_r2:.4f}"
)

print(
    f"Model R2:      "
    f"{test_r2:.4f}"
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    best_model,
    MODEL_PATH
)


metadata = {

    "model_name":
        best_model_name,

    "model_version":
        "1.0",

    "prediction_type":
        "sleep_duration_regression",

    "target":
        TARGET,

    "target_description":
        "Usual weekday/workday sleep duration in hours",

    "features":
        FEATURES,

    "records":
        len(df),

    "train_records":
        len(X_train),

    "test_records":
        len(X_test),

    "cross_validation":
        {
            "folds": 5,
        },

    "test_metrics":
        {
            "mae_hours":
                float(test_mae),

            "rmse_hours":
                float(test_rmse),

            "r2":
                float(test_r2),

            "within_30_minutes":
                float(
                    within_30_minutes
                ),

            "within_1_hour":
                float(
                    within_1_hour
                ),

            "within_2_hours":
                float(
                    within_2_hours
                ),
        },

    "baseline":
        {
            "mean_sleep_hours":
                float(
                    baseline_value
                ),

            "mae":
                float(
                    baseline_mae
                ),

            "rmse":
                float(
                    baseline_rmse
                ),

            "r2":
                float(
                    baseline_r2
                ),
        },

    "data_source":
        "NHANES 2021-2023",
}


with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print("\n" + "=" * 72)
print("FILES SAVED")
print("=" * 72)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nMetadata:"
)

print(
    METADATA_PATH
)

print(
    "\nComparison:"
)

print(
    COMPARISON_PATH
)