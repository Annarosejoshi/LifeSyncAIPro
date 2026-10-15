"""Verify deployed artifacts against their reproducible held-out test split.

Does not retrain, alter the dataset, or select thresholds from test data.
Run: venv/Scripts/python.exe ml/evaluation/audit_saved_models.py
"""
import json
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def audit(version="lifestyle_v2", model_names=None):
    results = {}
    for name in model_names or ("diabetes", "hypertension", "cardiovascular", "mental_health"):
        metadata = json.loads((ROOT / f"ml/models/{name}_{version}_metadata.json").read_text())
        data = pd.read_csv(ROOT / f"ml/data/processed/{name}_lifestyle_v2.csv")
        features, target = metadata["features"], metadata["target"]
        x = data[features]
        y = data[target].astype(int)
        _, x_test, _, y_test = train_test_split(
            x, y, test_size=.2, stratify=y, random_state=metadata["random_state"]
        )
        model = joblib.load(ROOT / f"ml/models/{name}_{version}.joblib")
        assert list(model.feature_names_in_) == features, f"{name}: feature mismatch"
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(x_test)[:, positive_index]
        classification = classification_report(
            y_test, probabilities >= metadata["selected_threshold"],
            output_dict=True, zero_division=0,
        )
        auc = roc_auc_score(y_test, probabilities)
        assert abs(auc - metadata["test_metrics"]["roc_auc"]) < 1e-10
        assert abs(classification["accuracy"] - metadata["test_metrics"]["classification"]["accuracy"]) < 1e-10
        results[name] = {
            "test_rows": len(y_test), "accuracy": classification["accuracy"],
            "roc_auc": auc, "positive_precision": classification["1"]["precision"],
            "positive_recall": classification["1"]["recall"],
            "confusion_matrix": confusion_matrix(y_test, probabilities >= metadata["selected_threshold"]).tolist(),
            "artifact_matches_recorded_metrics": True,
            "age_range": [float(data.age.min()), float(data.age.max())],
        }
        print(f"{name}: accuracy {classification['accuracy']:.1%}, AUC {auc:.3f}, "
              f"positive precision {classification['1']['precision']:.1%}, "
              f"recall {classification['1']['recall']:.1%}", flush=True)
    output_name = "verified_model_metrics.json" if version == "lifestyle_v2" else f"verified_{version}_metrics.json"
    (ROOT / "ml/evaluation" / output_name).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["lifestyle_v2", "lifestyle_v3", "lifestyle_v4", "lifestyle_v5", "lifestyle_v6", "lifestyle_v7", "lifestyle_v8", "lifestyle_v9", "lifestyle_v9_screening", "lifestyle_v10"], default="lifestyle_v2")
    parser.add_argument("--model", action="append", choices=["diabetes", "hypertension", "cardiovascular", "mental_health"])
    args = parser.parse_args()
    audit(args.version, args.model)
