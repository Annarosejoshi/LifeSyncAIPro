"""Train reproducible screens from inputs people can reasonably report.

Keep source datasets intact; write reduced CSVs and separate model artifacts.
Select on validation ROC AUC and report performance on untouched test rows.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
COMMON = ["age", "sex", "bmi", "smoking_status", "total_activity_minutes_week", "sedentary_minutes_day"]
SPECS = {
    "diabetes": ("diabetes_risk_dataset_v1.csv", "diabetes"),
    "hypertension": ("hypertension_risk_dataset_v1.csv", "hypertension"),
    "cardiovascular": ("cardiovascular_risk_dataset_v1.csv", "cardiovascular_disease"),
    "mental_health": ("mental_health_risk_dataset_v1.csv", "depression_risk"),
}

def train():
    for name, (filename, target) in SPECS.items():
        data = pd.read_csv(ROOT / "ml/data/processed" / filename)
        if name == "cardiovascular":
            # Current-smoking frequency differs from never/former/current.
            data["smoking_status"] = float("nan")
            data.loc[data["ever_smoked_100"] == 0, "smoking_status"] = 0
            data.loc[(data["ever_smoked_100"] == 1) & (data["current_smoking_status"] == 0), "smoking_status"] = 1
            data.loc[data["current_smoking_status"].isin([1, 2]), "smoking_status"] = 2
        features = COMMON + ([] if name == "cardiovascular" else ["sleep_hours"])
        reduced = data[["SEQN", *features, target]].dropna(subset=[target]).drop_duplicates("SEQN")
        reduced.to_csv(ROOT / f"ml/data/processed/{name}_lifestyle_v2.csv", index=False)
        x, y = reduced[features], reduced[target].astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=42)
        x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=.25, stratify=y_train, random_state=42)
        candidates = {
            "LogisticRegression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)),
            "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=180, min_samples_leaf=15, max_depth=10, random_state=42, n_jobs=2)),
        }
        comparison = []
        for label, pipeline in candidates.items():
            pipeline.fit(x_fit, y_fit)
            probabilities = pipeline.predict_proba(x_val)[:, 1]
            comparison.append({"model": label, "validation_roc_auc": roc_auc_score(y_val, probabilities)})
        winner = max(comparison, key=lambda row: row["validation_roc_auc"])["model"]
        model = candidates[winner]
        validation_probabilities = model.predict_proba(x_val)[:, 1]
        false_positive, true_positive, thresholds = roc_curve(y_val, validation_probabilities)
        # Select a balanced screening threshold using validation rows only.
        # A blanket 0.5 threshold misses most minority-class cases.
        threshold_index = int((true_positive - false_positive).argmax())
        threshold = float(thresholds[threshold_index])
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics = {"roc_auc": roc_auc_score(y_test, probabilities), "average_precision": average_precision_score(y_test, probabilities), "positive_prevalence": float(y_test.mean()), "threshold": threshold, "classification": classification_report(y_test, probabilities >= threshold, output_dict=True, zero_division=0)}
        metadata = {"model_name": winner, "model_version": "lifestyle_v2", "features": features, "target": target, "selected_threshold": threshold, "threshold_selection": "Maximum validation sensitivity minus false-positive rate (Youden J); test set untouched", "dataset_rows": len(reduced), "random_state": 42, "test_rows": len(y_test), "comparison": comparison, "test_metrics": metrics, "limitations": "Population screening, not diagnosis. Activity and sitting time must use matching units. Self-reported targets and a single dataset limit generalization."}
        joblib.dump(model, ROOT / f"ml/models/{name}_lifestyle_v2.joblib")
        (ROOT / f"ml/models/{name}_lifestyle_v2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (ROOT / f"ml/evaluation/{name}_lifestyle_v2.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"{name}: {winner}, test AUC={metrics['roc_auc']:.3f}, average precision={metrics['average_precision']:.3f}", flush=True)

if __name__ == "__main__":
    train()
