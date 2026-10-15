"""Compare the real dashboard with independent calls to deployed artifacts."""
from datetime import date
from pathlib import Path
import json

import joblib
import pandas as pd
from django.conf import settings
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from accounts.models import UserProfile
from lifestyle.models import DailyCheckIn
from mental_health.models import MentalWellnessAssessment
from predictions.models import PredictionHistory


class ActualPredictionVerificationTests(TestCase):
    def test_assessment_edit_updates_daily_score_without_fabricating_model_probability(self):
        self.profile()
        assessment = MentalWellnessAssessment.objects.create(
            user=self.user, date=timezone.localdate(), mood=1, stress=5,
            anxiety=5, energy=1, motivation=1, concentration=1,
            social_wellbeing=1, workload=5,
        )
        before = self.client.get(reverse("dashboard:risks"))
        self.assertEqual(before.context["mental_assessment_result"]["score"], 20)
        probability = before.context["risk_cards"][3][1]["probability"]
        response = self.client.post(reverse("mental_health:assessment_edit", args=[assessment.pk]), {
            "mood": 5, "stress": 1, "anxiety": 1, "energy": 5,
            "motivation": 5, "concentration": 5, "social_wellbeing": 5,
            "workload": 1, "notes": "Updated assessment",
        })
        self.assertEqual(response.status_code, 302)
        after = self.client.get(reverse("dashboard:risks"))
        self.assertEqual(after.context["mental_assessment_result"]["score"], 100)
        self.assertContains(after, "100.0 / 100")
        self.assertContains(after, "Mental wellness today:")
        self.assertEqual(after.context["risk_cards"][3][1]["probability"], probability)

    def test_previous_day_assessment_is_not_presented_as_todays_wellness(self):
        from datetime import timedelta
        MentalWellnessAssessment.objects.create(
            user=self.user, date=timezone.localdate()-timedelta(days=1),
            mood=5, stress=1, anxiety=1, energy=5, motivation=5,
            concentration=5, social_wellbeing=5, workload=1,
        )
        response = self.client.get(reverse("dashboard:risks"))
        self.assertIsNone(response.context["mental_assessment_result"])
        self.assertContains(response, "No mental wellness assessment saved today.")

    def test_post_to_create_updates_existing_assessment_and_returns_fresh_results(self):
        assessment = MentalWellnessAssessment.objects.create(
            user=self.user, date=timezone.localdate(), mood=1, stress=5,
            anxiety=5, energy=1, motivation=1, concentration=1,
            social_wellbeing=1, workload=5,
        )
        response = self.client.post(reverse("mental_health:assessment_create"), {
            "mood": 5, "stress": 1, "anxiety": 1, "energy": 5,
            "motivation": 5, "concentration": 5, "social_wellbeing": 5,
            "workload": 1,
        }, follow=True)
        self.assertEqual(response.redirect_chain[-1][0], reverse("dashboard:risks"))
        self.assertEqual(MentalWellnessAssessment.objects.filter(user=self.user).count(), 1)
        assessment.refresh_from_db()
        self.assertEqual(assessment.mood, 5)
        self.assertEqual(response.context["mental_assessment_result"]["score"], 100)
        self.assertContains(response, "Mental wellness saved: 100.0/100.")
        self.assertIn("no-store", response.headers["Cache-Control"])

    def setUp(self):
        self.user = User.objects.create_user(username="verify_actual_prediction")
        self.client.force_login(self.user)
        self.record = DailyCheckIn.objects.create(
            user=self.user, date=timezone.localdate(), sleep_hours=8,
            water_liters=2, exercise_minutes=30, screen_time_hours=4,
            mood=4, stress_level=2, energy_level=4,
        )
        self.artifacts = {}
        for name in ("diabetes", "hypertension", "cardiovascular", "mental_health"):
            root = Path(settings.BASE_DIR) / "ml/models"
            metadata = json.loads((root / f"{name}_lifestyle_v10_metadata.json").read_text())
            self.artifacts[name] = (joblib.load(root / f"{name}_lifestyle_v10.joblib"), metadata)

    def profile(self, age=35, gender="female", weight=64, smoking="never"):
        today = timezone.localdate()
        UserProfile.objects.filter(user=self.user).update(
            date_of_birth=date(today.year-age, today.month, today.day),
            gender=gender, height_cm=160, weight_kg=weight, smoking_status=smoking,
        )

    def test_dashboard_matches_independent_model_calls_for_different_inputs(self):
        scenarios = [
            (24, "female", 51.2, "never", 0, 8),
            (48, "male", 76.8, "former", 1, 6),
            (70, "female", 89.6, "current", 2, 4),
        ]
        for age, gender, weight, smoking, smoking_code, sleep in scenarios:
            with self.subTest(age=age, smoking=smoking, sleep=sleep):
                self.profile(age, gender, weight, smoking)
                self.record.sleep_hours = sleep
                self.record.save()
                response = self.client.get(reverse("dashboard:risks"))
                expected = dict(age=age, sex=1 if gender == "male" else 2,
                                bmi=round(weight / 1.6**2, 2), smoking_status=smoking_code,
                                sleep_hours=sleep)
                for _, result in response.context["risk_cards"]:
                    name = result["model"]
                    model, metadata = self.artifacts[name]
                    snapshot = {feature: expected[feature] for feature in metadata["features"]}
                    self.assertEqual(result["input_snapshot"], snapshot)
                    frame = pd.DataFrame([snapshot], columns=metadata["features"])
                    positive_column = list(model.classes_).index(1)
                    probability = float(model.predict_proba(frame)[0][positive_column])
                    prediction = int(probability >= metadata["selected_threshold"])
                    self.assertAlmostEqual(result["probability"], probability, places=12)
                    self.assertEqual(result["percentage"], round(probability * 100, 2))
                    self.assertEqual(result["prediction"], prediction)
                    self.assertEqual(result["risk_level"], "Elevated" if prediction else "Low")
                    self.assertTrue(result["label"].startswith("Elevated" if prediction else "Lower"))
                    saved = PredictionHistory.objects.get(user=self.user, prediction_type=name)
                    self.assertEqual(saved.input_snapshot, snapshot)
                    self.assertEqual(saved.predicted_class, prediction)
                    self.assertAlmostEqual(saved.probability, probability, places=12)
                self.assertEqual(PredictionHistory.objects.filter(user=self.user).count(), 4)

    def test_daily_edit_refreshes_model_inputs_and_unrelated_fields_are_not_substituted(self):
        self.profile()
        before = self.client.get(reverse("dashboard:risks")).context["risk_cards"]
        response = self.client.post(reverse("dashboard:checkin"), dict(
            action="daily_checkin", sleep_hours="5", water_liters="1",
            exercise_hours="0", exercise_extra_minutes="10", screen_time_hours="8",
            mood="1", stress_level="5", energy_level="2",
        ), follow=True)
        after = response.context["risk_cards"]
        for (_, old), (_, new) in zip(before, after):
            if new["model"] == "cardiovascular":
                self.assertEqual(old["input_snapshot"], new["input_snapshot"])
                self.assertEqual(old["probability"], new["probability"])
            else:
                self.assertEqual(new["input_snapshot"]["sleep_hours"], 5)
            for field in ("mood", "stress_level", "water_liters", "exercise_minutes",
                          "sedentary_minutes_day", "total_activity_minutes_week"):
                self.assertNotIn(field, new["input_snapshot"])
        self.assertTrue(any(old["probability"] != new["probability"]
                            for (_, old), (_, new) in zip(before, after)))
        UserProfile.objects.filter(user=self.user).update(gender="")
        missing = self.client.get(reverse("dashboard:risks"))
        self.assertTrue(all(not result["prediction_available"] for _, result in missing.context["risk_cards"]))
        self.assertNotContains(missing, "Based on your inputs")

    def test_invalid_checkin_does_not_replace_previous_prediction_inputs(self):
        self.profile()
        initial = self.client.get(reverse("dashboard:risks")).context["risk_cards"]
        response = self.client.post(reverse("dashboard:checkin"), dict(
            action="daily_checkin", sleep_hours="25", water_liters="2",
            exercise_hours="0", exercise_extra_minutes="30", screen_time_hours="4",
            mood="4", stress_level="2", energy_level="4",
        ))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["checkin_form"].errors)
        self.record.refresh_from_db()
        self.assertEqual(self.record.sleep_hours, 8)
        final = self.client.get(reverse("dashboard:risks")).context["risk_cards"]
        for (_, before), (_, after) in zip(initial, final):
            self.assertEqual(before["input_snapshot"], after["input_snapshot"])
            self.assertEqual(before["probability"], after["probability"])
