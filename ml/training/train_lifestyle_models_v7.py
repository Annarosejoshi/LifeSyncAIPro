"""Accuracy-focused comparison using only the currently available user inputs.

Selection uses validation data only, with a minimum positive recall. The fixed
internal test split has been evaluated in earlier experiments; it is not new
external validation. Keep active artifacts intact and write separate candidates.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
VERSION = "lifestyle_v7"
TARGET_ACCURACY = .80
MINIMUM_RECALL = .50
MINIMUM_SPECIFICITY = .60


def operating_point(y, probability, minimum_recall=MINIMUM_RECALL, minimum_specificity=MINIMUM_SPECIFICITY):
    false_positive, recall, thresholds = roc_curve(y, probability, drop_intermediate=False)
    specificity = 1 - false_positive
    prevalence = float(np.mean(y))
    accuracy = recall * prevalence + specificity * (1 - prevalence)
    eligible = np.isfinite(thresholds) & (recall >= minimum_recall) & (specificity >= minimum_specificity)
    indices = np.flatnonzero(eligible)
    meets_constraint = bool(len(indices))
    if not meets_constraint:
        indices = np.flatnonzero(np.isfinite(thresholds))
    index = max(indices, key=lambda i: (accuracy[i], (recall[i] + specificity[i]) / 2))
    return dict(threshold=float(thresholds[index]), accuracy=float(accuracy[index]),
                recall=float(recall[index]), specificity=float(specificity[index]),
                balanced_accuracy=float((recall[index] + specificity[index]) / 2),
                eligible=meets_constraint)


def candidates():
    models = {}
    for c in (.03, .3, 3):
        models[f"Logistic_C{c}"] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    for c in (.01, .1, 1):
        models[f"Polynomial_C{c}"] = make_pipeline(SimpleImputer(strategy="median"), PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    for knots in (3, 5):
        for c in (1, 10):
            models[f"Spline_K{knots}_C{c}"] = make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=knots, degree=3), StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    for leaves in (7, 15, 31):
        models[f"HistBoost_Leaves{leaves}"] = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=leaves, min_samples_leaf=30, learning_rate=.04, l2_regularization=10, random_state=42))
    for leaf in (5, 20):
        models[f"RandomForest_Leaf{leaf}"] = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, max_depth=12, random_state=42, n_jobs=2))
        models[f"ExtraTrees_Leaf{leaf}"] = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=300, min_samples_leaf=leaf, max_depth=14, random_state=42, n_jobs=2))
    for depth in (1, 2, 3, 4):
        models[f"XGBoost_Depth{depth}"] = make_pipeline(SimpleImputer(strategy="median"), XGBClassifier(n_estimators=350, max_depth=depth, learning_rate=.025, min_child_weight=15, reg_lambda=10, subsample=.85, colsample_bytree=.9, random_state=42, n_jobs=2, eval_metric="logloss"))
    for c in (.3, 3, 30):
        models[f"RBF_SVM_C{c}"] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVC(C=c, gamma="scale", probability=True, random_state=42))
    return models


def train():
    summary = {}
    for name in ("diabetes", "hypertension", "cardiovascular", "mental_health"):
        current = json.loads((ROOT / f"ml/models/{name}_lifestyle_v6_metadata.json").read_text())
        data = pd.read_csv(ROOT / f"ml/data/processed/{name}_lifestyle_v2.csv")
        features, target = current["features"], current["target"]
        assert data.SEQN.is_unique, "Participant IDs must be unique before splitting."
        x, y = data[features], data[target].astype(int)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, stratify=y, random_state=42)
        x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=.25, stratify=y_train, random_state=42)
        models, comparison = candidates(), []
        print(f"\n{name}: comparing {len(models)} candidates, using {features}", flush=True)
        for label, model in models.items():
            model.fit(x_fit, y_fit)
            positive = list(model.classes_).index(1)
            probability = model.predict_proba(x_val)[:, positive]
            point = operating_point(y_val.to_numpy(), probability)
            comparison.append(dict(model=label, **point, validation_roc_auc=float(roc_auc_score(y_val, probability))))
            print(f"{label}: validation accuracy {point['accuracy']:.1%}, recall {point['recall']:.1%}, specificity {point['specificity']:.1%}", flush=True)
        pool = [row for row in comparison if row["eligible"]] or comparison
        winner = max(pool, key=lambda row: (row["accuracy"], row["balanced_accuracy"], row["validation_roc_auc"]))
        model = models[winner["model"]]
        # Preserve the fitted probability scale used to select the cutoff.
        probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        threshold = winner["threshold"]
        classification = classification_report(y_test, probability >= threshold, output_dict=True, zero_division=0)
        metrics = dict(roc_auc=float(roc_auc_score(y_test, probability)),
                       average_precision=float(average_precision_score(y_test, probability)),
                       positive_prevalence=float(y_test.mean()), threshold=threshold,
                       majority_baseline_accuracy=float(max(y_test.mean(), 1-y_test.mean())),
                       classification=classification,
                       confusion_matrix=confusion_matrix(y_test, probability >= threshold).tolist())
        meets_target = bool(winner["eligible"] and classification["accuracy"] > TARGET_ACCURACY
                            and classification["1"]["recall"] >= MINIMUM_RECALL
                            and classification["0"]["recall"] >= MINIMUM_SPECIFICITY)
        metadata = dict(model_name=winner["model"], model_version=VERSION, features=features,
                        target=target, selected_threshold=threshold, random_state=42,
                        dataset_rows=len(data), test_rows=len(y_test), fit_rows=len(y_fit), validation_rows=len(y_val),
                        comparison=comparison, selected_validation_metrics=winner, test_metrics=metrics,
                        threshold_selection="Validation accuracy maximized with positive recall >=50% and specificity >=60%; no test tuning.",
                        excluded_inputs=["sedentary_minutes_day", "total_activity_minutes_week", "blood_pressure", "waist", "laboratory_measurements"],
                        deployment_target_met=meets_target,
                        limitations="Previously evaluated internal test split, not independent external validation. Accuracy is not clinical reliability. Report recall, precision and majority baseline; probabilities are not clinically calibrated.")
        joblib.dump(model, ROOT / f"ml/models/{name}_{VERSION}.joblib")
        (ROOT / f"ml/models/{name}_{VERSION}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (ROOT / f"ml/evaluation/{name}_{VERSION}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary[name] = dict(model=winner["model"], accuracy=classification["accuracy"],
                             positive_recall=classification["1"]["recall"], positive_precision=classification["1"]["precision"],
                             specificity=classification["0"]["recall"], roc_auc=metrics["roc_auc"],
                             majority_baseline_accuracy=metrics["majority_baseline_accuracy"],
                             target_met=meets_target)
        print(f"SELECTED {name}: test accuracy {classification['accuracy']:.1%}, positive recall {classification['1']['recall']:.1%}, target met: {meets_target}", flush=True)
        (ROOT / f"ml/evaluation/{VERSION}_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["average_accuracy"] = float(np.mean([row["accuracy"] for row in summary.values()]))
    summary["all_models_above_80_percent"] = all(row["target_met"] for row in summary.values() if isinstance(row, dict))
    summary["deployed"] = False
    (ROOT / f"ml/evaluation/{VERSION}_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"MEAN ACCURACY: {summary['average_accuracy']:.1%}; ALL FOUR TARGETS MET: {summary['all_models_above_80_percent']}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        train()
