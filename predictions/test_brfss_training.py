import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from ml.training.train_brfss_current_inputs import clean_chunk, select_candidate, split_indices


class BrfssTrainingChecks(SimpleTestCase):
    def source(self):
        return pd.DataFrame({
            "_STATE": [1] * 5, "SEQNO": [b"001", b"002", b"003", b"004", b"005"],
            "_AGE80": [20, 40, 60, 80, 35], "SEXVAR": [1, 2, 1, 2, 1],
            "_BMI5": [2100, 2550, 3000, 4000, 2800], "_SMOKER3": [1, 2, 3, 4, 9],
            "DIABETE4": [1, 2, 3, 4, 9], "BPHIGH6": [4, 3, 1, 7, 9],
            "CVDINFR4": [1, 2, 2, 7, 9], "CVDCRHD4": [7, 2, 9, 1, 9],
            "ADDEPEV3": [1, 2, 7, 9, 9],
        })

    def test_cdc_categories_are_decoded_without_guessing_unknown_answers(self):
        result = clean_chunk(self.source())
        self.assertEqual(result.respondent_id.tolist(), ["1:001", "1:002", "1:003", "1:004"])
        self.assertEqual(result.smoking_status.tolist(), [2, 2, 1, 0])
        self.assertEqual(result.bmi.tolist(), [21, 25.5, 30, 40])
        self.assertEqual(result.diabetes.tolist(), [1, 0, 0, 0])
        self.assertTrue(pd.isna(result.loc[3, "hypertension"]))
        self.assertTrue(pd.isna(result.loc[2, "mental_health"]))

    def test_unknown_heart_answers_do_not_become_negative_cases(self):
        result = clean_chunk(self.source())
        self.assertEqual(result.loc[0, "cardiovascular"], 1)
        self.assertEqual(result.loc[1, "cardiovascular"], 0)
        self.assertTrue(pd.isna(result.loc[2, "cardiovascular"]))
        self.assertEqual(result.loc[3, "cardiovascular"], 1)

    def test_test_rows_are_disjoint_and_keep_natural_positive_prevalence(self):
        target = pd.Series([0] * 900 + [1] * 100)
        fitting, validation, test = split_indices(target)
        self.assertEqual((len(fitting), len(validation), len(test)), (600, 200, 200))
        self.assertEqual(len(set(fitting) | set(validation) | set(test)), len(target))
        self.assertAlmostEqual(target.iloc[test].mean(), .1)
        self.assertTrue(np.array_equal(test, split_indices(target)[2]))

    def test_high_accuracy_failed_detection_is_ineligible(self):
        failed = dict(model="always_negative", eligible=False, accuracy=.9, balanced_accuracy=.5)
        self.assertIsNone(select_candidate([failed]))
        valid = dict(model="detects_positive", eligible=True, accuracy=.76, balanced_accuracy=.65)
        self.assertEqual(select_candidate([failed, valid]), valid)
