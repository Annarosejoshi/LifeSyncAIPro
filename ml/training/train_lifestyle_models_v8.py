"""Broader cross-validated search with categorical preprocessing and ensembles.

Run from project root: python -m ml.training.train_lifestyle_models_v8
The original test split is reserved from selection, but has been reported in
prior experiments. This is internal evaluation, not new external validation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from ml.training.train_lifestyle_models_v7 import operating_point

ROOT = Path(__file__).resolve().parents[2]
VERSION = "lifestyle_v8"


def preprocessing(features, knots=None):
    numeric = [feature for feature in features if feature not in ("sex", "smoking_status")]
    categorical = [feature for feature in features if feature in ("sex", "smoking_status")]
    steps = [SimpleImputer(strategy="median")]
    if knots:
        steps.append(SplineTransformer(n_knots=knots, degree=3))
    steps.append(StandardScaler())
    return ColumnTransformer([
        ("numeric", make_pipeline(*steps), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                     OneHotEncoder(handle_unknown="ignore", sparse_output=False)), categorical),
    ])


def candidates(features, prevalence):
    models = {}
    for c in (.03, .3, 3, 30):
        models[f"CategoricalLogistic_C{c}"] = make_pipeline(preprocessing(features), LogisticRegression(C=c, max_iter=3000))
    for degree in (2, 3):
        for c in (.01, .1, 1):
            models[f"Interaction_D{degree}_C{c}"] = make_pipeline(preprocessing(features), PolynomialFeatures(degree=degree, include_bias=False), StandardScaler(), LogisticRegression(C=c, max_iter=4000))
    for knots in (3, 5, 7):
        for c in (.1, 1):
            models[f"CategoricalSpline_K{knots}_C{c}"] = make_pipeline(preprocessing(features, knots), LogisticRegression(C=c, max_iter=3000))
    for leaves in (7, 15, 31):
        for balanced in (False, True):
            models[f"HistBoost_L{leaves}_Balanced{balanced}"] = make_pipeline(preprocessing(features), HistGradientBoostingClassifier(max_iter=300, learning_rate=.035, max_leaf_nodes=leaves, min_samples_leaf=30, l2_regularization=10, class_weight="balanced" if balanced else None, random_state=42))
    for leaf in (5, 20):
        for balanced in (False, True):
            models[f"RandomForest_Leaf{leaf}_Balanced{balanced}"] = make_pipeline(preprocessing(features), RandomForestClassifier(n_estimators=240, max_depth=12, min_samples_leaf=leaf, class_weight="balanced_subsample" if balanced else None, random_state=42, n_jobs=2))
        models[f"BalancedExtraTrees_Leaf{leaf}"] = make_pipeline(preprocessing(features), ExtraTreesClassifier(n_estimators=240, max_depth=14, min_samples_leaf=leaf, class_weight="balanced", random_state=42, n_jobs=2))
    for depth in (2, 3):
        models[f"WeightedXGBoost_D{depth}"] = make_pipeline(preprocessing(features), XGBClassifier(n_estimators=400, max_depth=depth, learning_rate=.025, min_child_weight=15, reg_lambda=15, scale_pos_weight=float(np.sqrt((1-prevalence)/prevalence)), subsample=.85, colsample_bytree=.9, random_state=42, n_jobs=2, eval_metric="logloss"))
    for layers in ((16,), (32, 16)):
        models[f"NeuralNetwork_{'_'.join(map(str, layers))}"] = make_pipeline(preprocessing(features), MLPClassifier(hidden_layer_sizes=layers, alpha=.1, max_iter=500, early_stopping=True, validation_fraction=.2, n_iter_no_change=20, random_state=42))
    return models


def out_of_fold(model, x, y, folds):
    probabilities = np.full(len(y), np.nan)
    assigned = np.zeros(len(y), dtype=int)
    for fitting, validation in folds:
        assert not np.intersect1d(fitting, validation).size
        fitted = clone(model).fit(x.iloc[fitting], y.iloc[fitting])
        probabilities[validation] = fitted.predict_proba(x.iloc[validation])[:, list(fitted.classes_).index(1)]
        assigned[validation] += 1
    assert np.all(assigned == 1) and np.isfinite(probabilities).all()
    return probabilities


def train():
    summary = {}
    for name in ("diabetes", "hypertension", "cardiovascular", "mental_health"):
        active = json.loads((ROOT / f"ml/models/{name}_lifestyle_v6_metadata.json").read_text())
        data = pd.read_csv(ROOT / f"ml/data/processed/{name}_lifestyle_v2.csv")
        assert data.SEQN.is_unique
        features, target = active["features"], active["target"]
        x_train, x_test, y_train, y_test = train_test_split(data[features], data[target].astype(int), test_size=.2, stratify=data[target].astype(int), random_state=42)
        x_train, y_train = x_train.reset_index(drop=True), y_train.reset_index(drop=True)
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=127).split(x_train, y_train))
        models = candidates(features, float(y_train.mean()))
        comparison, predictions = [], {}
        print(f"\n{name}: {len(models)} candidates with 3-fold validation", flush=True)
        for label, model in models.items():
            probability = out_of_fold(model, x_train, y_train, folds)
            point = operating_point(y_train.to_numpy(), probability)
            predictions[label] = probability
            comparison.append(dict(model=label, **point, validation_roc_auc=float(roc_auc_score(y_train, probability))))
            print(f"{label}: OOF accuracy {point['accuracy']:.1%}, recall {point['recall']:.1%}, eligible {point['eligible']}", flush=True)
        eligible = [row for row in comparison if row["eligible"]] or comparison
        rankings = {
            "AccuracyTop3": sorted(eligible, key=lambda row: row["accuracy"], reverse=True)[:3],
            "AccuracyTop5": sorted(eligible, key=lambda row: row["accuracy"], reverse=True)[:5],
            "AUCTop3": sorted(eligible, key=lambda row: row["validation_roc_auc"], reverse=True)[:3],
        }
        for label, members in rankings.items():
            probability = np.mean([predictions[row["model"]] for row in members], axis=0)
            point = operating_point(y_train.to_numpy(), probability)
            models[label] = VotingClassifier(estimators=[(f"m{i}", models[row["model"]]) for i, row in enumerate(members)], voting="soft")
            comparison.append(dict(model=label, members=[row["model"] for row in members], **point,
                                   validation_roc_auc=float(roc_auc_score(y_train, probability))))
            print(f"Ensemble {label}: OOF accuracy {point['accuracy']:.1%}, recall {point['recall']:.1%}", flush=True)
        pool = [row for row in comparison if row["eligible"]] or comparison
        winner = max(pool, key=lambda row: (row["accuracy"], row["balanced_accuracy"], row["validation_roc_auc"]))
        model = models[winner["model"]].fit(x_train, y_train)
        threshold = winner["threshold"]
        probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        classification = classification_report(y_test, probability >= threshold, output_dict=True, zero_division=0)
        metrics = dict(roc_auc=float(roc_auc_score(y_test, probability)), average_precision=float(average_precision_score(y_test, probability)),
                       positive_prevalence=float(y_test.mean()), threshold=threshold,
                       majority_baseline_accuracy=float(max(y_test.mean(), 1-y_test.mean())),
                       classification=classification, confusion_matrix=confusion_matrix(y_test, probability >= threshold).tolist())
        meets_target = bool(winner["eligible"] and classification["accuracy"] > .80
                            and classification["1"]["recall"] >= .50 and classification["0"]["recall"] >= .60)
        metadata = dict(model_name=winner["model"], model_version=VERSION, features=features, target=target,
                        selected_threshold=threshold, random_state=42, dataset_rows=len(data), test_rows=len(y_test),
                        fit_rows=len(y_train), validation_method="3-fold out-of-fold on training rows, seed 127", comparison=comparison,
                        selected_validation_metrics=winner, test_metrics=metrics, deployment_target_met=meets_target,
                        threshold_selection="OOF accuracy maximum subject to positive recall >=50% and specificity >=60%, then refit on all training rows.",
                        limitations="Repeatedly evaluated internal test split, not external validation. OOF cutoff may transfer imperfectly after final refit. Probabilities are not clinically calibrated; report minority recall and precision.")
        joblib.dump(model, ROOT / f"ml/models/{name}_{VERSION}.joblib")
        (ROOT / f"ml/models/{name}_{VERSION}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (ROOT / f"ml/evaluation/{name}_{VERSION}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary[name] = dict(model=winner["model"], accuracy=classification["accuracy"], positive_recall=classification["1"]["recall"],
                            positive_precision=classification["1"]["precision"], specificity=classification["0"]["recall"],
                            roc_auc=metrics["roc_auc"], majority_baseline_accuracy=metrics["majority_baseline_accuracy"], target_met=meets_target,
                            base_candidates=len(predictions), ensemble_candidates=3)
        print(f"SELECTED {name}: test accuracy {classification['accuracy']:.1%}, positive recall {classification['1']['recall']:.1%}, target met {meets_target}", flush=True)
        (ROOT / f"ml/evaluation/{VERSION}_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["average_accuracy"] = float(np.mean([row["accuracy"] for row in summary.values()]))
    summary["all_models_above_80_percent"] = all(row["target_met"] for row in summary.values() if isinstance(row, dict))
    summary["deployed"] = False
    (ROOT / f"ml/evaluation/{VERSION}_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"MEAN ACCURACY {summary['average_accuracy']:.1%}; ALL FOUR TARGETS MET {summary['all_models_above_80_percent']}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        train()
