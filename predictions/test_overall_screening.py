from django.test import SimpleTestCase

from predictions.presentation import overall_screening_summary


class OverallScreeningTests(SimpleTestCase):
    def cards(self, classes):
        return [(f"Screening {index}", dict(prediction_available=value is not None,
                 prediction=value)) for index, value in enumerate(classes)]

    def test_all_four_negative_results_are_required_for_lower_summary(self):
        result = overall_screening_summary(self.cards([0, 0, 0, 0]))
        self.assertEqual(result["status"], "lower")
        self.assertTrue(result["complete"])

    def test_elevated_uses_model_class_not_raw_probability(self):
        cards = self.cards([0, 0, 0, 1])
        cards[-1][1]["probability"] = .15
        result = overall_screening_summary(cards)
        self.assertEqual(result["status"], "attention")
        self.assertEqual(result["elevated"], ["Screening 3"])

    def test_missing_or_invalid_class_is_not_treated_as_lower(self):
        for classes in ([0, 0, 0, None], [None] * 4, [0, 0, 0, 2]):
            result = overall_screening_summary(self.cards(classes))
            self.assertEqual(result["status"], "incomplete")
            self.assertFalse(result["complete"])

    def test_partial_summary_preserves_known_elevated_result(self):
        result = overall_screening_summary(self.cards([1, 0, None, None]))
        self.assertEqual(result["status"], "attention")
        self.assertEqual(result["available_count"], 2)
        self.assertEqual(result["elevated_count"], 1)
        self.assertFalse(result["complete"])

    def test_mental_flag_names_the_concern_without_claiming_a_diagnosis(self):
        cards = self.cards([0, 0, 0, 1])
        cards[-1][1]["model"] = "mental_health"
        result = overall_screening_summary(cards)
        self.assertEqual(result["title"], "Mental wellness needs attention")
        self.assertIn("depression-related screening risk", result["concerns"][0]["explanation"])
        self.assertIn("does not identify", result["concerns"][0]["explanation"])
