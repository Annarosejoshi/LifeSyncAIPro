"""Research comparison on local CDC BRFSS 2023; never deploy automatically.

Run: python -m ml.training.train_brfss_current_inputs
Only existing profile fields are used. Sleep is unavailable in this source;
its historical diagnosis targets are not daily changes or future disease risk.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from ml.training.train_lifestyle_models_v7 import operating_point
from ml.training.train_lifestyle_models_v8 import preprocessing

ROOT = Path(__file__).resolve().parents[2]
VERSION = "brfss2023_profile_v1"
FEATURES = ["age", "sex", "bmi", "smoking_status"]
SOURCE_COLUMNS = ["_STATE", "SEQNO", "_AGE80", "SEXVAR", "_BMI5", "_SMOKER3",
                  "DIABETE4", "BPHIGH6", "CVDINFR4", "CVDCRHD4", "ADDEPEV3"]
TARGETS = {
    "diabetes": "Self-reported previously diagnosed diabetes, excluding pregnancy-only; prediabetes is negative for diabetes",
    "hypertension": "Self-reported previously diagnosed hypertension; pregnancy-only and borderline are negative",
    "cardiovascular": "Self-reported previous heart attack or coronary heart disease",
    "mental_health": "Self-reported previously diagnosed depressive disorder; not PHQ screening or today's mood",
}


def clean_chunk(raw):
    """Decode documented CDC categories; unknown outcomes remain missing."""
    data = pd.DataFrame(index=raw.index)
    sequence = raw["SEQNO"].map(lambda value: value.decode("ascii").strip() if isinstance(value, bytes) else str(value))
    data["respondent_id"] = raw["_STATE"].astype("Int64").astype(str) + ":" + sequence
    data["age"] = raw["_AGE80"].where(raw["_AGE80"].between(18, 80))
    data["sex"] = raw["SEXVAR"].where(raw["SEXVAR"].isin([1, 2]))
    data["bmi"] = (raw["_BMI5"] / 100).where(raw["_BMI5"].between(1200, 9999))
    data["smoking_status"] = raw["_SMOKER3"].map({1: 2, 2: 2, 3: 1, 4: 0})
    data["diabetes"] = raw["DIABETE4"].map({1: 1, 2: 0, 3: 0, 4: 0})
    data["hypertension"] = raw["BPHIGH6"].map({1: 1, 2: 0, 3: 0, 4: 0})
    heart_attack, coronary = raw["CVDINFR4"], raw["CVDCRHD4"]
    data["cardiovascular"] = np.nan
    data.loc[(heart_attack == 2) & (coronary == 2), "cardiovascular"] = 0
    data.loc[(heart_attack == 1) | (coronary == 1), "cardiovascular"] = 1
    data["mental_health"] = raw["ADDEPEV3"].map({1: 1, 2: 0})
    return data.dropna(subset=FEATURES)


def read_source():
    destination = ROOT / "ml/data/processed/brfss2023_current_profile.csv"
    if destination.exists():
        return pd.read_csv(destination, dtype={"respondent_id": str})
    chunks = []
    with pd.read_sas(ROOT / "ml/data/sleep/LLCP2023.XPT", format="xport", chunksize=10000, iterator=True) as reader:
        for number, raw in enumerate(reader, 1):
            chunks.append(clean_chunk(raw[SOURCE_COLUMNS]))
            if number % 5 == 0:
                print(f"Read {number * 10000:,} source records", flush=True)
    data = pd.concat(chunks, ignore_index=True)
    assert data.respondent_id.is_unique, "Duplicate respondent IDs require investigation before splitting"
    data.to_csv(destination, index=False)
    return data


def candidates():
    models = {}
    for knots in (None, 4, 6):
        models[f"Logistic_spline{knots}"] = make_pipeline(preprocessing(FEATURES, knots), LogisticRegression(C=1, max_iter=1500))
    for leaves in (7, 15, 31):
        models[f"HistBoost_leaves{leaves}"] = make_pipeline(preprocessing(FEATURES), HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=leaves, min_samples_leaf=100, learning_rate=.04, l2_regularization=15, random_state=2026))
    for depth in (2, 3, 4):
        models[f"XGBoost_depth{depth}"] = make_pipeline(preprocessing(FEATURES), XGBClassifier(n_estimators=260, max_depth=depth, learning_rate=.035, min_child_weight=30, reg_lambda=20, subsample=.85, random_state=2026, n_jobs=2, eval_metric="logloss"))
    return models


def select_candidate(comparison):
    eligible = [row for row in comparison if row["eligible"]]
    return max(eligible, key=lambda row: (row["accuracy"], row["balanced_accuracy"])) if eligible else None


def split_indices(y):
    fitting, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=2026)
    fitting, validation = train_test_split(fitting, test_size=.25, stratify=y.iloc[fitting], random_state=2027)
    assert not (set(fitting) & set(test) or set(validation) & set(test) or set(fitting) & set(validation))
    return fitting, validation, test


def metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(accuracy=float(accuracy_score(y, predicted)), recall=float(recall_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)), specificity=float(tn / (tn + fp)),
                roc_auc=float(roc_auc_score(y, probability)), average_precision=float(average_precision_score(y, probability)),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                negative_only_baseline_accuracy=float(1 - y.mean()))


def train():
    data = read_source()
    summary = {}
    for name, description in TARGETS.items():
        sample = data.dropna(subset=[name]).reset_index(drop=True)
        x, y = sample[FEATURES], sample[name].astype(int)
        fitting, validation, test = split_indices(y)
        models, comparison = candidates(), []
        print(f"\n{name}: {len(sample):,} records, {len(models)} configurations", flush=True)
        for label, model in models.items():
            model.fit(x.iloc[fitting], y.iloc[fitting])
            probability = model.predict_proba(x.iloc[validation])[:, list(model.classes_).index(1)]
            point = operating_point(y.iloc[validation].to_numpy(), probability)
            comparison.append(dict(model=label, **point))
            print(f"{label}: validation accuracy {point['accuracy']:.1%}, recall {point['recall']:.1%}, eligible={point['eligible']}", flush=True)
        winner = select_candidate(comparison)
        if winner is None:
            summary[name] = dict(eligible=False, deployed=False, comparison=comparison)
            continue
        # Keep the exact validation-selected estimator/cutoff; no refit-induced cutoff drift.
        model = models[winner["model"]]
        probability = model.predict_proba(x.iloc[test])[:, list(model.classes_).index(1)]
        result = metrics(y.iloc[test], probability, winner["threshold"])
        artifact = ROOT / f"ml/models/{name}_{VERSION}.joblib"
        joblib.dump(model, artifact)
        # Reload saved model and reproduce held-out probabilities before reporting success.
        reloaded = joblib.load(artifact)
        np.testing.assert_allclose(probability, reloaded.predict_proba(x.iloc[test])[:, list(reloaded.classes_).index(1)], rtol=0, atol=0)
        metadata = dict(model_version=VERSION, model_name=winner["model"], features=FEATURES, target=name,
                        target_description=description, source="CDC BRFSS 2023 local LLCP2023.XPT",
                        dataset_file="ml/data/processed/brfss2023_current_profile.csv", deployed=False,
                        missing_daily_predictors="No sleep or current-day lifestyle observations in this comparison",
                        source_coding="_AGE80 imputed age capped at 80; SEXVAR 1 male/2 female; _BMI5 /100; _SMOKER3 1/2 current=2,3 former=1,4 never=0",
                        evaluation="Unweighted internal respondent holdout; not clinical/external validation; targets differ from NHANES",
                        source_documentation="https://www.cdc.gov/brfss/annual_data/annual_2023.html",
                        dataset_rows=len(sample), fit_rows=len(fitting), validation_rows=len(validation), test_rows=len(test),
                        split_seeds=dict(test=2026, validation=2027), selected_threshold=winner["threshold"],
                        selection="Validation accuracy with recall >=50% and specificity >=60%; test never used for tuning",
                        test_respondent_ids_sha256=hashlib.sha256("\n".join(sample.iloc[test].respondent_id).encode()).hexdigest(),
                        test_metrics=result, comparison=comparison)
        (ROOT / f"ml/models/{name}_{VERSION}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        summary[name] = dict(eligible=True, deployed=False, **result, meets_accuracy_target=result["accuracy"] >= .8,
                             meets_test_detection_constraints=result["recall"] >= .5 and result["specificity"] >= .6)
        print(f"TEST {name}: accuracy {result['accuracy']:.1%}, recall {result['recall']:.1%}, precision {result['precision']:.1%}", flush=True)
    (ROOT / "ml/evaluation/brfss2023_current_inputs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        train()
