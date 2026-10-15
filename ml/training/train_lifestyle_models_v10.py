"""Additional boosting-budget search with fold-contained preprocessing.

Run: python -m ml.training.train_lifestyle_models_v10
Preserves deployed artifacts and the existing internal test partition. Repeated
use of this partition means it is not independent external validation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from ml.training.train_lifestyle_models_v7 import operating_point
from ml.training.train_lifestyle_models_v8 import out_of_fold, preprocessing

ROOT = Path(__file__).resolve().parents[2]
VERSION = "lifestyle_v10"


def candidates(features):
    models = {}
    for budget in (150, 450, 900):
        for regularization in (1, 30, 100):
            label = f"BoostBudget{budget}_L2{regularization}"
            models[label] = make_pipeline(preprocessing(features),
                HistGradientBoostingClassifier(
                    max_iter=budget, learning_rate=.025, max_leaf_nodes=7,
                    min_samples_leaf=40, l2_regularization=regularization,
                    early_stopping=True, validation_fraction=.15,
                    n_iter_no_change=35, random_state=2026))
    for c in (.03, .3, 3):
        models[f"SplineElasticNet_C{c}"] = make_pipeline(
            preprocessing(features, knots=5),
            LogisticRegression(C=c, solver="saga", l1_ratio=.5,
                               max_iter=4000, random_state=2026))
    return models


def select_candidate(comparison):
    eligible = [row for row in comparison if row["eligible"]]
    return max(eligible, key=lambda row: (row["accuracy"],
        row["balanced_accuracy"], row["validation_roc_auc"])) if eligible else None


def train():
    summary = {}
    for name in ("diabetes", "hypertension", "cardiovascular", "mental_health"):
        active = json.loads((ROOT / f"ml/models/{name}_lifestyle_v6_metadata.json").read_text())
        data = pd.read_csv(ROOT / f"ml/data/processed/{name}_lifestyle_v2.csv")
        assert data.SEQN.is_unique
        features, target = active["features"], active["target"]
        x_train, x_test, y_train, y_test = train_test_split(
            data[features], data[target].astype(int), test_size=.2,
            stratify=data[target].astype(int), random_state=42)
        x_train, y_train = x_train.reset_index(drop=True), y_train.reset_index(drop=True)
        folds = list(StratifiedKFold(n_splits=3, shuffle=True,
                                   random_state=127).split(x_train, y_train))
        models, comparison = candidates(features), []
        for label, model in models.items():
            probability = out_of_fold(model, x_train, y_train, folds)
            point = operating_point(y_train.to_numpy(), probability,
                                    minimum_recall=.5, minimum_specificity=.6)
            comparison.append(dict(model=label, **point,
                validation_roc_auc=float(roc_auc_score(y_train, probability))))
            print(f"{name} {label}: OOF accuracy {point['accuracy']:.3f}, "
                  f"recall {point['recall']:.3f}, eligible {point['eligible']}", flush=True)
        winner = select_candidate(comparison)
        if winner is None:
            summary[name] = dict(eligible=False, target_met=False, deployed=False,
                                comparison=comparison)
        else:
            model = models[winner["model"]].fit(x_train, y_train)
            threshold = winner["threshold"]
            # No changes to estimator or threshold after test evaluation.
            probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
            report = classification_report(y_test, probability >= threshold,
                                           output_dict=True, zero_division=0)
            metrics = dict(classification=report,
                roc_auc=float(roc_auc_score(y_test, probability)),
                confusion_matrix=confusion_matrix(y_test, probability >= threshold).tolist(),
                majority_baseline_accuracy=float(max(y_test.mean(), 1-y_test.mean())))
            metadata = dict(model_name=winner["model"], model_version=VERSION,
                features=features, target=target, random_state=42,
                selected_threshold=threshold, fit_rows=len(y_train), test_rows=len(y_test),
                validation_method="3-fold training OOF, seed 127; preprocessing inside folds",
                selected_validation_metrics=winner, comparison=comparison,
                test_metrics=metrics,
                limitations="Previously evaluated internal test split; selection scores are optimistic after model/threshold search. Not external validation. Probabilities are not clinically calibrated.")
            model_path = ROOT / f"ml/models/{name}_{VERSION}.joblib"
            joblib.dump(model, model_path)
            # Verify persistence before accepting the experiment output.
            np.testing.assert_allclose(joblib.load(model_path).predict_proba(x_test)[:, 1], probability)
            (ROOT / f"ml/models/{name}_{VERSION}_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8")
            summary[name] = dict(model=winner["model"], eligible=True,
                accuracy=report["accuracy"], positive_recall=report["1"]["recall"],
                positive_precision=report["1"]["precision"], specificity=report["0"]["recall"],
                roc_auc=metrics["roc_auc"], majority_baseline_accuracy=metrics["majority_baseline_accuracy"],
                target_met=bool(report["accuracy"] > .8 and report["1"]["recall"] >= .5
                                and report["0"]["recall"] >= .6), deployed=False)
            print(f"TEST {name}: {summary[name]}", flush=True)
        (ROOT / f"ml/evaluation/{VERSION}_training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    print("ALL FOUR TARGETS MET: " + str(all(row["target_met"] for row in summary.values())), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        train()
