import numpy as np
from django.test import SimpleTestCase
from ml.training.train_lifestyle_models_v7 import operating_point


class AccuracyTrainingSelectionTests(SimpleTestCase):
    def test_majority_class_accuracy_cannot_bypass_positive_recall(self):
        y = np.array([0] * 8 + [1] * 2)
        probabilities = np.array([.1] * 8 + [.09, .8])
        point = operating_point(y, probabilities)
        prediction = probabilities >= point["threshold"]
        self.assertTrue(point["eligible"])
        self.assertGreaterEqual(point["recall"], .5)
        self.assertGreaterEqual(point["specificity"], .6)
        self.assertAlmostEqual(point["accuracy"], float(np.mean(prediction == y)))
        self.assertTrue(prediction[y == 1].any())

    def test_unattainable_recall_specificity_constraints_are_explicit(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        probabilities = np.array([.1, .2, .7, .8, .3, .4])
        point = operating_point(y, probabilities)
        self.assertFalse(point["eligible"])

    def test_uninformative_probabilities_are_not_eligible(self):
        point = operating_point(np.array([0, 0, 0, 1]), np.array([.2] * 4))
        self.assertFalse(point["eligible"])
