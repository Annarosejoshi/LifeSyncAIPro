from pathlib import Path
import json

import joblib
import pandas as pd

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
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# TARGET
# ---------------------------------------------------------

TARGET = "Diabetes_012"


CLASS_NAMES = {
    0: "No Diabetes",
    1: "Prediabetes",
    2: "Diabetes",
}


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

def load_dataset():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 70)
    print("DATASET LOADED")
    print("=" * 70)

    print(
        f"\nRows: {df.shape[0]:,}"
    )

    print(
        f"Columns: {df.shape[1]}"
    )

    print("\nColumn names:")

    for column in df.columns:
        print(
            f" - {column}"
        )

    if TARGET not in df.columns:

        raise ValueError(
            f"\nTarget column '{TARGET}' "
            f"was not found in the dataset."
        )

    print("\nTarget distribution:")

    target_counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    for class_value, count in target_counts.items():

        name = CLASS_NAMES.get(
            int(class_value),
            str(class_value)
        )

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"{class_value} - {name}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print(
        "\nMissing values:",
        int(df.isnull().sum().sum())
    )

    return df


# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

def prepare_data(df):

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET].astype(int)

    feature_names = X.columns.tolist()

    print("\nNumber of input features:")
    print(
        len(feature_names)
    )

    return (
        X,
        y,
        feature_names
    )


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------

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
                    n_estimators=250,
                    max_depth=18,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            ),
        ]
    )


# ---------------------------------------------------------
# XGBOOST
# ---------------------------------------------------------

def create_xgboost_model():

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
                    objective="multi:softprob",
                    num_class=3,

                    n_estimators=300,
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


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

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

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": weighted_f1,
        "f1_macro": macro_f1,
    }

    print("\nMetrics")

    print(
        f"Accuracy:           "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision Weighted: "
        f"{precision:.4f}"
    )

    print(
        f"Recall Weighted:    "
        f"{recall:.4f}"
    )

    print(
        f"F1 Weighted:        "
        f"{weighted_f1:.4f}"
    )

    print(
        f"F1 Macro:           "
        f"{macro_f1:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(
        matrix
    )

    print(
        "\nClassification Report:"
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Diabetes",
            "Prediabetes",
            "Diabetes"
        ],
        digits=4,
        zero_division=0
    )

    print(
        report
    )

    return {
        "name": name,
        "model": model,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": matrix.tolist(),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    df = load_dataset()

    X, y, feature_names = prepare_data(
        df
    )

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # COMPARISON TABLE
    # -----------------------------------------------------

    comparison_data = []

    for result in results:

        comparison_data.append(
            {
                "model":
                    result["name"],

                "accuracy":
                    result[
                        "metrics"
                    ][
                        "accuracy"
                    ],

                "precision_weighted":
                    result[
                        "metrics"
                    ][
                        "precision_weighted"
                    ],

                "recall_weighted":
                    result[
                        "metrics"
                    ][
                        "recall_weighted"
                    ],

                "f1_weighted":
                    result[
                        "metrics"
                    ][
                        "f1_weighted"
                    ],

                "f1_macro":
                    result[
                        "metrics"
                    ][
                        "f1_macro"
                    ],
            }
        )


    comparison = pd.DataFrame(
        comparison_data
    )


    # Macro F1 is important because
    # the prediabetes class is much smaller.
    comparison = comparison.sort_values(
        by=[
            "f1_macro",
            "f1_weighted"
        ],
        ascending=False
    )


    print("\n")
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )


    # -----------------------------------------------------
    # SELECT BEST MODEL
    # -----------------------------------------------------

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

    print(
        "BEST MODEL:"
    )

    print(
        best_model_name
    )

    print("=" * 70)


    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path = (
        MODEL_DIR
        / "diabetes_model.joblib"
    )


    joblib.dump(
        best_result["model"],
        model_path
    )


    # -----------------------------------------------------
    # SAVE METADATA
    # -----------------------------------------------------

    metadata = {

        "model_name":
            best_model_name,

        "model_version":
            "1.0",

        "problem_type":
            "multiclass_classification",

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
        / "diabetes_model_metadata.json"
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


    # -----------------------------------------------------
    # SAVE MODEL COMPARISON
    # -----------------------------------------------------

    comparison_path = (
        EVALUATION_DIR
        / "diabetes_model_comparison.csv"
    )


    comparison.to_csv(
        comparison_path,
        index=False
    )


    # -----------------------------------------------------
    # SAVE CLASSIFICATION REPORT
    # -----------------------------------------------------

    report_path = (
        EVALUATION_DIR
        / "diabetes_classification_report.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            best_result["report"]
        )


    # -----------------------------------------------------
    # SAVE CONFUSION MATRIX
    # -----------------------------------------------------

    confusion_path = (
        EVALUATION_DIR
        / "diabetes_confusion_matrix.json"
    )


    with open(
        confusion_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_result[
                "confusion_matrix"
            ],
            file,
            indent=4
        )


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
        f"\nClassification report:\n"
        f"{report_path}"
    )

    print(
        f"\nConfusion matrix:\n"
        f"{confusion_path}"
    )


    print("\n")
    print("=" * 70)
    print("DIABETES MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()