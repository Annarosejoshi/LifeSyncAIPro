"""Independently reload and reproduce the new source's saved test results."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ml.training.train_brfss_current_inputs import ROOT, TARGETS, VERSION, metrics, split_indices


def audit():
    results = {}
    for name in TARGETS:
        metadata = json.loads((ROOT / f"ml/models/{name}_{VERSION}_metadata.json").read_text(encoding="utf-8"))
        data = pd.read_csv(ROOT / metadata["dataset_file"], dtype={"respondent_id": str}).dropna(subset=[name]).reset_index(drop=True)
        assert data.respondent_id.is_unique
        y = data[name].astype(int)
        _, _, test = split_indices(y)
        assert hashlib.sha256("\n".join(data.iloc[test].respondent_id).encode()).hexdigest() == metadata["test_respondent_ids_sha256"]
        model = joblib.load(ROOT / f"ml/models/{name}_{VERSION}.joblib")
        assert list(model.feature_names_in_) == metadata["features"]
        probability = model.predict_proba(data.iloc[test][metadata["features"]])[:, list(model.classes_).index(1)]
        result = metrics(y.iloc[test], probability, metadata["selected_threshold"])
        for field in ("accuracy", "recall", "precision", "specificity", "roc_auc", "average_precision"):
            np.testing.assert_allclose(result[field], metadata["test_metrics"][field], rtol=0, atol=1e-12)
        assert result["confusion_matrix"] == metadata["test_metrics"]["confusion_matrix"]
        results[name] = dict(reproduced=True, deployed=False, **result)
        print(f"{name}: reproduced accuracy={result['accuracy']:.1%}, positive recall={result['recall']:.1%}", flush=True)
    (ROOT / "ml/evaluation/verified_brfss2023_current_inputs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        audit()
