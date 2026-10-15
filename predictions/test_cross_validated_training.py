import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

from ml.training.train_lifestyle_models_v8 import out_of_fold, preprocessing


class HoldoutCheckingEstimator(ClassifierMixin, BaseEstimator):
    def fit(self, x, y):
        self.seen = set(x["row_id"])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x):
        if self.seen.intersection(x["row_id"]):
            raise AssertionError("Validation row was used for fitting")
        positive = x["row_id"].to_numpy() / 10
        return np.column_stack([1 - positive, positive])


class CrossValidatedTrainingTests(SimpleTestCase):
    def test_each_validation_prediction_is_held_out_and_restored_to_row_order(self):
        x = pd.DataFrame({"row_id": [5, 0, 3, 2, 4, 1]})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=127).split(x, y))
        prediction = out_of_fold(HoldoutCheckingEstimator(), x, y, folds)
        np.testing.assert_allclose(prediction, x.row_id.to_numpy() / 10)

    def test_overlapping_fold_is_rejected(self):
        x = pd.DataFrame({"row_id": [0, 1, 2, 3]})
        with self.assertRaises(AssertionError):
            out_of_fold(HoldoutCheckingEstimator(), x, pd.Series([0, 1, 0, 1]),
                        [(np.array([0, 1]), np.array([1, 2]))])

    def test_gender_and_smoking_use_separate_categories(self):
        features = ["age", "sex", "bmi", "smoking_status", "sleep_hours"]
        x = pd.DataFrame([[30, 1, 24, 0, 8], [40, 2, 28, 1, 6], [50, 1, 30, 2, 7]], columns=features)
        transformer = preprocessing(features).fit(x)
        names = transformer.get_feature_names_out()
        self.assertEqual(len([name for name in names if "categorical__sex_" in name]), 2)
        self.assertEqual(len([name for name in names if "categorical__smoking_status_" in name]), 3)
        self.assertTrue(np.isfinite(transformer.transform(x)).all())
