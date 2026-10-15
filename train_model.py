from pathlib import Path

import joblib
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "ml"
    / "datasets"
    / "diabetes_012_health_indicators_BRFSS2015.csv"
)

OUTPUT_PATH = BASE_DIR / "diabetes_risk_model.joblib"
TARGET = "Diabetes_012"


# ==========================================
# 2. INPUT FEATURES
# ==========================================

FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


def main():
    # ==========================================
    # 3. LOAD DATASET
    # ==========================================

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print("Loading diabetes dataset...")
    df = pd.read_csv(DATASET_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1, 2])]

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("Dataset rows:", len(df))
    print("\nOriginal target distribution:")
    print(y.value_counts().sort_index())

    # ==========================================
    # 4. SPLIT BEFORE OVERSAMPLING
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ==========================================
    # 5. OVERSAMPLE THE PREDIABETES CLASS
    # ==========================================

    print("\nBalancing training data...")
    sampler = RandomOverSampler(
        sampling_strategy={1: 15000},
        random_state=42,
    )

    X_train_balanced, y_train_balanced = sampler.fit_resample(
        X_train,
        y_train,
    )

    print("\nTraining distribution after oversampling:")
    print(y_train_balanced.value_counts().sort_index())

    # ==========================================
    # 6. TRAIN PIPELINE
    # ==========================================

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining model...")
    model.fit(X_train_balanced, y_train_balanced)

    # ==========================================
    # 7. EVALUATE
    # ==========================================

    print("\nEvaluating on untouched test data...")
    predictions = model.predict(X_test)

    print("\nConfusion Matrix (classes 0, 1, 2):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

    print(
        "\nBalanced Accuracy:",
        round(balanced_accuracy_score(y_test, predictions) * 100, 2),
        "%",
    )
    print(
        "\nTest Accuracy:",
        round(accuracy_score(y_test, predictions) * 100, 2),
        "%",
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=["No Diabetes", "Prediabetes", "Diabetes"],
            zero_division=0,
        )
    )

    # ==========================================
    # 8. SAVE MODEL BUNDLE
    # ==========================================

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        OUTPUT_PATH,
    )

    print("\nModel saved successfully:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
