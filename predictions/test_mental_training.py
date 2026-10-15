import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from sklearn.linear_model import LogisticRegression

from ml.training.train_lifestyle_models_v7 import operating_point
from ml.training.train_mental_models_v9 import resampling_model, select_candidate


class MentalTrainingChecks(SimpleTestCase):
    def test_high_accuracy_cannot_select_a_candidate_that_failed_detection_constraints(self):
        comparison = [dict(model="majority_like", screening_selection=dict(eligible=False, accuracy=.87,
                           balanced_accuracy=.51), validation_roc_auc=.70)]
        self.assertIsNone(select_candidate(comparison, "screening_selection"))
        comparison.append(dict(model="eligible", screening_selection=dict(eligible=True, accuracy=.68,
                               balanced_accuracy=.66), validation_roc_auc=.71))
        self.assertEqual(select_candidate(comparison, "screening_selection")["model"], "eligible")

    def test_higher_detection_requirement_changes_cutoff_instead_of_faking_accuracy(self):
        target = np.array([0] * 8 + [1] * 4)
        probability = np.array([.05, .1, .15, .16, .2, .22, .4, .5, .18, .28, .6, .8])
        accuracy = operating_point(target, probability, minimum_recall=.5)
        screening = operating_point(target, probability, minimum_recall=.7)
        self.assertTrue(screening["eligible"])
        self.assertGreaterEqual(screening["recall"], .7)
        self.assertGreaterEqual(screening["specificity"], .6)
        self.assertLess(screening["threshold"], accuracy["threshold"])
        self.assertLess(screening["accuracy"], accuracy["accuracy"])

    def test_training_balancing_preserves_categories_and_does_not_expand_inference(self):
        features = ["age", "sex", "bmi", "smoking_status", "sleep_hours"]
        x = pd.DataFrame([
            [25 + i, 1 + i % 2, 20 + i % 12, i % 3, 5 + i % 4]
            for i in range(40)
        ], columns=features)
        y = np.array([0] * 32 + [1] * 8)
        model = resampling_model(features, .7, LogisticRegression(max_iter=1000)).fit(x, y)
        probability = model.predict_proba(x.iloc[:3])
        self.assertEqual(probability.shape, (3, 2))
        prepared = model.named_steps["prepare"].transform(x)
        balanced, labels = model.named_steps["balance"].fit_resample(prepared, y)
        self.assertGreater(len(labels), len(y))
        self.assertTrue(set(balanced[:, 3]).issubset({1, 2}))
        self.assertTrue(set(balanced[:, 4]).issubset({0, 1, 2}))
