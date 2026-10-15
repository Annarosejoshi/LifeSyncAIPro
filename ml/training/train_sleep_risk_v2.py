from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


FINAL_MODEL_PATH = (
    MODEL_DIR
    / "sleep_risk_model.joblib"
)

METADATA_PATH = (
    MODEL_DIR
    / "sleep_risk_model_metadata.json"
)

COMPARISON_PATH = (
    EVALUATION_DIR
    / "sleep_risk_model_comparison.csv"
)

REPORT_PATH = (
    EVALUATION_DIR
    / "sleep_risk_classification_report.txt"
)

CONFUSION_PATH = (
    EVALUATION_DIR
    / "sleep_risk_confusion_matrix.json"
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
# LOAD DATA
# ============================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print("=" * 72)
    print("LIFESYNC AI PRO - SLEEP RISK MODEL TRAINING")
    print("=" * 72)

    print("\nDataset loaded.")

    print(
        "\nRows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found."
        )

    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    name,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

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

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Adequate Sleep",
            "Short Sleep",
        ],
        zero_division=0,
    )

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

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

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    print(
        "\nClassification Report:"
    )

    print(report)

    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "probabilities": probabilities,
        "predictions": predictions,
        "matrix": matrix,
        "report": report,
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    results = []

    thresholds = np.arange(
        0.20,
        0.81,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1]
            )
            .ravel()
        )

        specificity = (
            tn
            / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        results.append({
            "threshold":
                round(
                    float(threshold),
                    2
                ),

            "precision":
                precision,

            "recall":
                recall,

            "specificity":
                specificity,

            "f1":
                f1,
        })

    threshold_df = pd.DataFrame(
        results
    )

    # For health-risk screening,
    # don't select a model that gets high F1
    # by ignoring most positive cases.
    #
    # Prefer recall >= 0.65, then choose
    # the best F1 within that region.

    acceptable = threshold_df[
        threshold_df["recall"] >= 0.65
    ]

    if len(acceptable) > 0:

        best_row = (
            acceptable
            .sort_values(
                [
                    "f1",
                    "recall",
                    "specificity",
                ],
                ascending=False
            )
            .iloc[0]
        )

    else:

        best_row = (
            threshold_df
            .sort_values(
                "f1",
                ascending=False
            )
            .iloc[0]
        )

    return (
        float(
            best_row["threshold"]
        ),
        threshold_df,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()


    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df = df[
        df[TARGET].notna()
    ].copy()

    df[TARGET] = (
        df[TARGET]
        .astype(int)
    )


    print("\nTarget distribution:")

    target_counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    print(target_counts)


    print("\nTarget percentages:")

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


    # --------------------------------------------------------
    # INPUT / TARGET
    # --------------------------------------------------------

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
            " -",
            feature
        )


    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST
    # --------------------------------------------------------
    #
    # 70% training
    # 15% validation
    # 15% final test
    #
    # Threshold selection happens ONLY on validation.
    # Final test remains untouched until the end.
    # --------------------------------------------------------

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp,
        )
    )


    print("\nDataset split:")

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


    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count
        / positive_count
    )


    print("\nTraining class distribution:")

    print(
        "Adequate sleep:",
        negative_count
    )

    print(
        "Short sleep:",
        positive_count
    )

    print(
        "XGBoost scale_pos_weight:",
        round(
            scale_pos_weight,
            3
        )
    )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    preprocessing_scaled = Pipeline(
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


    preprocessing_tree = Pipeline(
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


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing_scaled,
            ),

            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing_tree,
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
        ]
    )


    # ========================================================
    # XGBOOST
    # ========================================================

    xgboost_model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing_tree,
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

                    objective="binary:logistic",

                    eval_metric="logloss",

                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


    models = {
        "Logistic Regression":
            logistic_model,

        "Random Forest":
            random_forest_model,

        "XGBoost":
            xgboost_model,
    }


    # ========================================================
    # TRAIN MODELS
    # ========================================================

    validation_results = []

    trained_models = {}


    for name, model in models.items():

        print(
            "\nTraining:",
            name
        )

        model.fit(
            X_train,
            y_train
        )

        trained_models[
            name
        ] = model


        result = evaluate_model(
            name,
            model,
            X_validation,
            y_validation,
        )

        validation_results.append(
            result
        )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_rows = []

    for result in validation_results:

        comparison_rows.append({
            "model":
                result["model"],

            "accuracy":
                result["accuracy"],

            "precision":
                result["precision"],

            "recall":
                result["recall"],

            "f1":
                result["f1"],

            "roc_auc":
                result["roc_auc"],
        })


    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df = (
        comparison_df
        .sort_values(
            [
                "roc_auc",
                "f1",
            ],
            ascending=False
        )
    )


    print("\n" + "=" * 72)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 72)

    print(
        comparison_df
        .to_string(
            index=False
        )
    )


    # ========================================================
    # CHOOSE MODEL
    # ========================================================
    #
    # ROC-AUC is our primary comparison here
    # because we have class imbalance.
    #
    # Threshold is tuned separately.
    # ========================================================

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


    # ========================================================
    # THRESHOLD TUNING ON VALIDATION SET
    # ========================================================

    validation_probabilities = (
        best_model
        .predict_proba(
            X_validation
        )[:, 1]
    )


    best_threshold, threshold_df = (
        find_best_threshold(
            y_validation,
            validation_probabilities,
        )
    )


    print(
        "\nSelected threshold:",
        best_threshold
    )


    selected_threshold_row = (
        threshold_df[
            threshold_df[
                "threshold"
            ]
            == best_threshold
        ]
        .iloc[0]
    )


    print("\nValidation threshold metrics:")

    print(
        selected_threshold_row
    )


    # ========================================================
    # FINAL TEST
    # ========================================================

    test_probabilities = (
        best_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)


    final_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    final_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    final_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    final_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    final_roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            test_predictions,
            labels=[0, 1]
        )
        .ravel()
    )

    final_specificity = (
        tn
        / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    final_matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )


    final_report = classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Adequate Sleep",
            "Short Sleep",
        ],
        zero_division=0,
    )


    print("\n" + "=" * 72)
    print("FINAL TEST RESULTS")
    print("=" * 72)

    print(
        "Model:",
        best_model_name
    )

    print(
        "Threshold:",
        best_threshold
    )

    print(
        f"Accuracy:    "
        f"{final_accuracy:.4f}"
    )

    print(
        f"Precision:   "
        f"{final_precision:.4f}"
    )

    print(
        f"Recall:      "
        f"{final_recall:.4f}"
    )

    print(
        f"Specificity: "
        f"{final_specificity:.4f}"
    )

    print(
        f"F1 Score:    "
        f"{final_f1:.4f}"
    )

    print(
        f"ROC-AUC:     "
        f"{final_roc_auc:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        final_matrix
    )


    print(
        "\nClassification Report:"
    )

    print(
        final_report
    )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    joblib.dump(
        best_model,
        FINAL_MODEL_PATH
    )


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "model_name":
            best_model_name,

        "model_version":
            "1.0",

        "prediction_type":
            "sleep_risk",

        "target":
            TARGET,

        "target_definition":
            "1 = usual weekday/workday sleep "
            "duration below 7 hours; "
            "0 = 7 hours or more",

        "threshold":
            best_threshold,

        "features":
            FEATURES,

        "training_records":
            len(X_train),

        "validation_records":
            len(X_validation),

        "test_records":
            len(X_test),

        "final_metrics": {
            "accuracy":
                final_accuracy,

            "precision":
                final_precision,

            "recall":
                final_recall,

            "specificity":
                final_specificity,

            "f1":
                final_f1,

            "roc_auc":
                final_roc_auc,
        },

        "data_source":
            "NHANES 2021-2023",

        "notes": [
            "SLD012 is not used as an input feature.",
            "DPQ030 is excluded to reduce target leakage.",
            "Missing predictors are handled inside the model pipeline.",
        ],
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


    # ========================================================
    # SAVE EVALUATION
    # ========================================================

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False
    )


    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"Model: "
            f"{best_model_name}\n"
        )

        file.write(
            f"Threshold: "
            f"{best_threshold}\n\n"
        )

        file.write(
            final_report
        )


    confusion_data = {
        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "matrix":
            final_matrix.tolist(),
    }


    with open(
        CONFUSION_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            confusion_data,
            file,
            indent=4
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 72)
    print("SAVED FILES")
    print("=" * 72)

    print(
        "\nModel:\n",
        FINAL_MODEL_PATH
    )

    print(
        "\nMetadata:\n",
        METADATA_PATH
    )

    print(
        "\nComparison:\n",
        COMPARISON_PATH
    )

    print(
        "\nClassification Report:\n",
        REPORT_PATH
    )

    print(
        "\nConfusion Matrix:\n",
        CONFUSION_PATH
    )


if __name__ == "__main__":
    main()