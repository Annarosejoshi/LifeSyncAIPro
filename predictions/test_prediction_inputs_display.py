from datetime import date
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import TestCase, SimpleTestCase
from django.urls import reverse
from django.utils import timezone

from accounts.models import UserProfile
from lifestyle.models import DailyCheckIn
from ml.services.risk_prediction_service import load_metadata, predict_mental_health_risk


class PredictionInputDisplayTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="input_display_user")
        UserProfile.objects.filter(user=self.user).update(
            date_of_birth=date(1990, 1, 1), gender="female", height_cm=165,
            weight_kg=60, smoking_status="never",
        )
        self.record = DailyCheckIn.objects.create(
            user=self.user, date=timezone.localdate(), sleep_hours=6.5,
            water_liters=2, exercise_minutes=30, screen_time_hours=4,
            mood=4, stress_level=2, energy_level=4,
        )
        self.client.force_login(self.user)

    def test_each_input_note_matches_that_results_exact_model_inputs(self):
        response = self.client.get(reverse("dashboard:risks"))
        self.assertContains(response, "Based on your inputs", count=4)
        self.assertContains(response, "The lifestyle risk estimate uses your profile and sleep.")
        for _, result in response.context["risk_cards"]:
            self.assertTrue(result["prediction_available"], result)
            details = {item["label"]: item["value"] for item in result["display_inputs"]}
            self.assertEqual(details["Gender"], "Female")
            self.assertEqual(details["Smoking"], "Never")
            self.assertEqual(details["BMI"], f'{result["input_snapshot"]["bmi"]:.2f}')
            if result["model"] == "cardiovascular":
                self.assertNotIn("Sleep today", details)
            else:
                self.assertEqual(details["Sleep today"], "6.5 hours")
            self.assertNotIn("Mood", details)
            self.assertNotIn("Stress", details)

    def test_editing_today_refreshes_displayed_values(self):
        self.client.get(reverse("dashboard:risks"))
        self.record.sleep_hours = 8
        self.record.save()
        response = self.client.get(reverse("dashboard:risks"))
        self.assertContains(response, "8 hours", count=3)
        self.assertNotContains(response, "6.5 hours")


class MentalScreeningThresholdTests(SimpleTestCase):
    @patch("ml.services.risk_prediction_service.load_model")
    def test_mental_result_uses_saved_threshold_and_exact_input_order(self, loader):
        metadata = load_metadata("mental_health")
        model = MagicMock()
        loader.return_value = model
        values = dict(age=35, sex=2, bmi=24, smoking_status=0, sleep_hours=6,
                      mood=1, stress_level=5, water_liters=2)
        for probability, expected in ((metadata["selected_threshold"] - .01, 0),
                                      (metadata["selected_threshold"] + .01, 1)):
            model.predict_proba.return_value = [[1 - probability, probability]]
            result = predict_mental_health_risk(values)
            self.assertEqual(result["prediction"], expected)
            self.assertEqual(result["risk_level"], "Elevated" if expected else "Low")
            self.assertEqual(result["percentage"], round(probability * 100, 2))
            frame = model.predict_proba.call_args.args[0]
            self.assertEqual(list(frame.columns), metadata["features"])
            self.assertEqual(frame.loc[0, "sleep_hours"], 6)
            self.assertNotIn("mood", frame.columns)
            self.assertNotIn("stress_level", frame.columns)
