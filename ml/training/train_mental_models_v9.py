"""Mental-only search with calibration and training-fold minority balancing.

Run: python -m ml.training.train_mental_models_v9
Select separate accuracy and higher-recall candidates before test evaluation.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ResamplingPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from ml.training.train_lifestyle_models_v7 import operating_point
from ml.training.train_lifestyle_models_v8 import out_of_fold, preprocessing

ROOT = Path(__file__).resolve().parents[2]


def select_candidate(comparison, selection):
    pool = [row for row in comparison if row[selection]["eligible"]]
    if not pool:
        return None
    return max(pool, key=lambda row: (row[selection]["accuracy"], row[selection]["balanced_accuracy"], row["validation_roc_auc"]))


def resampling_model(features, ratio, classifier):
    numeric = [feature for feature in features if feature not in ("sex", "smoking_status")]
    categorical = [feature for feature in features if feature in ("sex", "smoking_status")]
    numeric_indices = list(range(len(numeric)))
    categorical_indices = list(range(len(numeric), len(features)))
    before = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", SimpleImputer(strategy="most_frequent"), categorical),
    ])
    after = ColumnTransformer([
        ("numeric", "passthrough", numeric_indices),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_indices),
    ])
    return ResamplingPipeline([
        ("prepare", before),
        ("balance", SMOTENC(categorical_features=categorical_indices, sampling_strategy=ratio, random_state=42)),
        ("encode", after), ("classifier", classifier),
    ])


def candidates(features):
    models = {}
    for c in (.1, 1, 10, 100):
        for gamma in (.1, 1):
            base = make_pipeline(preprocessing(features), SVC(C=c, gamma=gamma, class_weight="balanced"))
            models[f"CalibratedSVM_C{c}_G{gamma}"] = CalibratedClassifierCV(base, cv=3, method="sigmoid", ensemble=True)
    for neighbours in (25, 75, 150, 300):
        for weights in ("uniform", "distance"):
            models[f"KNN_K{neighbours}_{weights}"] = make_pipeline(preprocessing(features), KNeighborsClassifier(n_neighbors=neighbours, weights=weights, n_jobs=2))
    for smooth in (1e-9, 1e-4, .01):
        models[f"GaussianNB_{smooth}"] = make_pipeline(preprocessing(features), GaussianNB(var_smoothing=smooth))
    for regularization in (.1, .5, .9):
        numeric = [feature for feature in features if feature not in ("sex", "smoking_status")]
        categories = [feature for feature in features if feature in ("sex", "smoking_status")]
        transform = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)), categories),
        ])
        models[f"RegularizedQDA_{regularization}"] = make_pipeline(transform, QuadraticDiscriminantAnalysis(reg_param=regularization))
    for ratio in (.4, .7, 1):
        models[f"SMOTENC_{ratio}_Logistic"] = resampling_model(features, ratio, LogisticRegression(C=.3, max_iter=3000))
        for leaves in (7, 15):
            models[f"SMOTENC_{ratio}_Boost_L{leaves}"] = resampling_model(features, ratio, HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=leaves, learning_rate=.04, l2_regularization=10, min_samples_leaf=30, random_state=42))
    for leaf in (5, 15, 30):
        models[f"BalancedForest_Leaf{leaf}"] = make_pipeline(preprocessing(features), BalancedRandomForestClassifier(n_estimators=250, min_samples_leaf=leaf, max_depth=12, sampling_strategy="all", replacement=True, bootstrap=False, random_state=42, n_jobs=2))
    for leaves in (7, 15):
        base = make_pipeline(preprocessing(features), HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=leaves, learning_rate=.04, min_samples_leaf=30, l2_regularization=10, random_state=42))
        models[f"CalibratedBoost_L{leaves}"] = CalibratedClassifierCV(base, cv=3, method="sigmoid", ensemble=True)
    for activation in ("relu", "tanh"):
        for alpha in (.1, 1):
            models[f"Neural_{activation}_A{alpha}"] = make_pipeline(preprocessing(features), MLPClassifier(hidden_layer_sizes=(32, 16), activation=activation, alpha=alpha, max_iter=600, early_stopping=True, validation_fraction=.15, n_iter_no_change=30, random_state=42))
    return models


def train():
    active = json.loads((ROOT / "ml/models/mental_health_lifestyle_v6_metadata.json").read_text())
    features, target = active["features"], active["target"]
    data = pd.read_csv(ROOT / "ml/data/processed/mental_health_lifestyle_v2.csv")
    assert data.SEQN.is_unique
    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target].astype(int), test_size=.2, stratify=data[target].astype(int), random_state=42)
    x_train, y_train = x_train.reset_index(drop=True), y_train.reset_index(drop=True)
    folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=127).split(x_train, y_train))
    models, comparison, predictions = candidates(features), [], {}

    def record(label, probability, members=None):
        accuracy = operating_point(y_train.to_numpy(), probability, minimum_recall=.50)
        screening = operating_point(y_train.to_numpy(), probability, minimum_recall=.70)
        comparison.append(dict(model=label, accuracy_selection=accuracy, screening_selection=screening,
                               validation_roc_auc=float(roc_auc_score(y_train, probability)), members=members))
        accuracy_text = f"{accuracy['accuracy']:.1%} at {accuracy['recall']:.1%} recall" if accuracy["eligible"] else "no eligible threshold"
        screening_text = f"{screening['accuracy']:.1%} at {screening['recall']:.1%} recall" if screening["eligible"] else "no eligible threshold"
        print(f"{label}: accuracy selection {accuracy_text}; higher-detection selection {screening_text}", flush=True)

    print(f"Mental wellness: {len(models)} configurations, 3-fold validation; inputs {features}", flush=True)
    for label, model in models.items():
        probability = out_of_fold(model, x_train, y_train, folds)
        predictions[label] = probability
        record(label, probability)
    accuracy_pool = [row for row in comparison if row["accuracy_selection"]["eligible"]] or comparison
    screening_pool = [row for row in comparison if row["screening_selection"]["eligible"]] or comparison
    rankings = {
        "AccuracyTop3": sorted(accuracy_pool, key=lambda row: row["accuracy_selection"]["accuracy"], reverse=True)[:3],
        "ScreeningTop3": sorted(screening_pool, key=lambda row: row["screening_selection"]["accuracy"], reverse=True)[:3],
        "AUCTop5": sorted(comparison, key=lambda row: row["validation_roc_auc"], reverse=True)[:5],
    }
    for label, members in rankings.items():
        probability = np.mean([predictions[row["model"]] for row in members], axis=0)
        models[label] = VotingClassifier(estimators=[(f"m{i}", models[row["model"]]) for i, row in enumerate(members)], voting="soft")
        record(label, probability, members=[row["model"] for row in members])
    winners, summary = {}, {}
    # Choose both estimators and cutoffs before inspecting either test result.
    for suffix, selection in (("lifestyle_v9", "accuracy_selection"), ("lifestyle_v9_screening", "screening_selection")):
        winner = select_candidate(comparison, selection)
        if winner is None:
            summary[suffix] = dict(eligible=False, deployed=False, reason="No candidate met the required validation recall and specificity.")
            print(f"{suffix}: no eligible candidate; no artifact will be saved", flush=True)
            for stale in (ROOT / f"ml/models/mental_health_{suffix}.joblib",
                          ROOT / f"ml/models/mental_health_{suffix}_metadata.json",
                          ROOT / f"ml/evaluation/mental_health_{suffix}.json"):
                stale.unlink(missing_ok=True)
            continue
        winners[suffix] = (winner, selection)
    fitted = {}
    for version, (winner, selection) in winners.items():
        label = winner["model"]
        if label not in fitted:
            fitted[label] = clone(models[label]).fit(x_train, y_train)
        model = fitted[label]
        threshold = winner[selection]["threshold"]
        probability = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        classification = classification_report(y_test, probability >= threshold, output_dict=True, zero_division=0)
        metrics = dict(roc_auc=float(roc_auc_score(y_test, probability)), average_precision=float(average_precision_score(y_test, probability)),
                       positive_prevalence=float(y_test.mean()), threshold=threshold, majority_baseline_accuracy=float(max(y_test.mean(), 1-y_test.mean())),
                       classification=classification, confusion_matrix=confusion_matrix(y_test, probability >= threshold).tolist())
        metadata = dict(model_name=label, model_version=version, features=features, target=target, selected_threshold=threshold,
                        random_state=42, dataset_rows=len(data), test_rows=len(y_test), fit_rows=len(y_train),
                        comparison=comparison, selected_validation_metrics=winner, test_metrics=metrics,
                        validation_method="3-fold OOF on training data, seed 127; resampling only inside fitting folds",
                        threshold_selection=f"OOF accuracy maximum with positive recall >={'70%' if selection == 'screening_selection' else '50%'} and specificity >=60%; both winners locked before test evaluation.",
                        limitations="Repeatedly evaluated internal test split, not independent clinical validation. SMOTENC samples only training-fold data; source and test rows are unchanged. OOF cutoffs may transfer imperfectly after refit. Probabilities are not clinically calibrated.")
        joblib.dump(model, ROOT / f"ml/models/mental_health_{version}.joblib")
        (ROOT / f"ml/models/mental_health_{version}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (ROOT / f"ml/evaluation/mental_health_{version}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary[version] = dict(model=label, accuracy=classification["accuracy"], positive_recall=classification["1"]["recall"],
                                positive_precision=classification["1"]["precision"], specificity=classification["0"]["recall"], roc_auc=metrics["roc_auc"],
                                majority_baseline_accuracy=metrics["majority_baseline_accuracy"],
                                eligible=True, above_80_percent=classification["accuracy"] > .80, deployed=False)
        print(f"{version}: TEST accuracy {classification['accuracy']:.1%}, positive recall {classification['1']['recall']:.1%}, precision {classification['1']['precision']:.1%}", flush=True)
    summary["base_configurations"] = len(predictions)
    summary["ensemble_comparisons"] = 3
    (ROOT / "ml/evaluation/mental_health_v9_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        train()
