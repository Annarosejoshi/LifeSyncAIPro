from django.test import SimpleTestCase

from ml.training.train_lifestyle_models_v10 import select_candidate


class AdditionalTrainingSelectionTests(SimpleTestCase):
    def test_high_accuracy_ineligible_candidate_cannot_win(self):
        majority_only = dict(model="majority", eligible=False, accuracy=.9,
                             balanced_accuracy=.5, validation_roc_auc=.5)
        detector = dict(model="detector", eligible=True, accuracy=.75,
                        balanced_accuracy=.7, validation_roc_auc=.78)
        self.assertEqual(select_candidate([majority_only, detector]), detector)

    def test_no_eligible_model_does_not_fall_back_to_majority_prediction(self):
        self.assertIsNone(select_candidate([dict(eligible=False, accuracy=.95)]))
