"""Compare models on validation rows; evaluate the selected model once on test.

Uses the original fixed stratified split, real lifestyle inputs, and no synthetic
test rows. Keeps v2 artifacts intact. Thresholds are selected before seeing test
predictions. Validation accuracy targets cannot guarantee test accuracy.
"""
from pathlib import Path
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]


def choose_threshold(y, probability, minimum_recall=.60, selection_metric="balanced_accuracy"):
    rows = []
    for threshold in np.linspace(.03, .65, 125):
        prediction = probability >= threshold
        tp = int(((y == 1) & prediction).sum())
        tn = int(((y == 0) & ~prediction).sum())
        recall = tp / int((y == 1).sum())
        specificity = tn / int((y == 0).sum())
        rows.append({"threshold": float(threshold), "accuracy": (tp + tn) / len(y),
                     "recall": recall, "specificity": specificity,
                     "balanced_accuracy": (recall + specificity) / 2})
    # Never attain accuracy by returning the majority class for everybody.
    eligible = [row for row in rows if row["recall"] >= minimum_recall and row["specificity"] >= .60]
    target = [row for row in eligible if row["accuracy"] >= .76]
    pool = target or eligible or rows
    return max(pool, key=lambda row: (row[selection_metric], row["balanced_accuracy"]))


def candidates():
    return {
        "LogisticRegression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1, max_iter=3000)),
        "PolynomialLogisticRegression": make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LogisticRegression(C=.1, max_iter=3000)),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_depth=12, random_state=42, n_jobs=2)),
        "ExtraTrees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10, max_depth=14, random_state=42, n_jobs=2)),
        "HistGradientBoosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=.05, l2_regularization=5, random_state=42)),
        "XGBoostDepth2": make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=250, max_depth=2, learning_rate=.035, min_child_weight=15, reg_lambda=10, subsample=.85, colsample_bytree=.9, random_state=42, n_jobs=2, eval_metric="logloss")),
        "XGBoostDepth3": make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.035, min_child_weight=20, reg_lambda=15, subsample=.85, colsample_bytree=.9, random_state=42, n_jobs=2, eval_metric="logloss")),
    }


def train(version="lifestyle_v3", minimum_recall=.60, selection_metric="balanced_accuracy", excluded_features=()):
    summary = {}
    for name in ("diabetes", "hypertension", "cardiovascular", "mental_health"):
        previous = json.loads((ROOT / f"ml/models/{name}_lifestyle_v2_metadata.json").read_text())
        data = pd.read_csv(ROOT / f"ml/data/processed/{name}_lifestyle_v2.csv")
        features, target = previous["features"], previous["target"]
        features = [feature for feature in features if feature not in excluded_features]
        x, y = data[features], data[target].astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=42)
        x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=.25, stratify=y_train, random_state=42)
        models = candidates()
        comparison = []
        for label, model in models.items():
            model.fit(x_fit, y_fit)
            probability = model.predict_proba(x_val)[:, 1]
            operating = choose_threshold(y_val.to_numpy(), probability, minimum_recall, selection_metric)
            comparison.append({"model": label, **operating,
                               "validation_roc_auc": roc_auc_score(y_val, probability)})
            print(f"{name} / {label}: validation accuracy {operating['accuracy']:.1%}, "
                  f"recall {operating['recall']:.1%}, AUC {comparison[-1]['validation_roc_auc']:.3f}", flush=True)
        eligible = [row for row in comparison if row["accuracy"] >= .76 and row["recall"] >= minimum_recall and row["specificity"] >= .60]
        winner = max(eligible or comparison, key=lambda row: (row[selection_metric], row["validation_roc_auc"]))
        model, threshold = models[winner["model"]], winner["threshold"]
        # Keep the exact fitted estimator used to select the threshold. Refit
        # could change the probability scale and invalidate that threshold.
        probability = model.predict_proba(x_test)[:, 1]
        metrics = {"roc_auc": roc_auc_score(y_test, probability),
                   "average_precision": average_precision_score(y_test, probability),
                   "positive_prevalence": float(y_test.mean()), "threshold": threshold,
                   "majority_baseline_accuracy": float(max(y_test.mean(), 1-y_test.mean())),
                   "classification": classification_report(y_test, probability >= threshold, output_dict=True, zero_division=0),
                   "confusion_matrix": confusion_matrix(y_test, probability >= threshold).tolist()}
        metadata = {"model_name": winner["model"], "model_version": version,
                    "features": features, "target": target, "selected_threshold": threshold,
                    "threshold_selection": f"Validation only: prefer accuracy >=76%, sensitivity >={minimum_recall:.0%} and specificity >=60%; maximize {selection_metric}. Fallback maximizes {selection_metric}.",
                    "dataset_rows": len(data), "random_state": 42, "test_rows": len(y_test),
                    "fit_rows": len(y_fit), "validation_rows": len(y_val),
                    "comparison": comparison, "selected_validation_metrics": winner,
                    "test_metrics": metrics,
                    "limitations": "Internal evaluation on the same source dataset and original split; not external validation. Percentages are not clinically calibrated. Accuracy and recall trade off; majority baseline reported explicitly."}
        joblib.dump(model, ROOT / f"ml/models/{name}_{version}.joblib")
        (ROOT / f"ml/models/{name}_{version}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (ROOT / f"ml/evaluation/{name}_{version}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary[name] = {"model": winner["model"], "accuracy": metrics["classification"]["accuracy"],
                         "positive_recall": metrics["classification"]["1"]["recall"], "roc_auc": metrics["roc_auc"],
                         "majority_baseline_accuracy": metrics["majority_baseline_accuracy"]}
        print(f"SELECTED {name}: {winner['model']}, test accuracy {summary[name]['accuracy']:.1%}, "
              f"recall {summary[name]['positive_recall']:.1%}, AUC {metrics['roc_auc']:.3f}", flush=True)
    summary["average_accuracy"] = float(np.mean([row["accuracy"] for row in summary.values()]))
    (ROOT / f"ml/evaluation/{version}_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"AVERAGE TEST ACCURACY: {summary['average_accuracy']:.1%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["lifestyle_v3", "lifestyle_v4", "lifestyle_v5", "lifestyle_v6"], default="lifestyle_v3")
    parser.add_argument("--minimum-recall", type=float, choices=[.5, .6], default=.6)
    parser.add_argument("--selection-metric", choices=["accuracy", "balanced_accuracy"], default="balanced_accuracy")
    parser.add_argument("--exclude-feature", action="append", default=[])
    args = parser.parse_args()
    train(args.version, args.minimum_recall, args.selection_metric, args.exclude_feature)
