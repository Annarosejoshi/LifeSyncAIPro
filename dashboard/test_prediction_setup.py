from datetime import date

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from accounts.models import UserProfile
from lifestyle.models import DailyCheckIn


class PredictionSetupTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="missing_gender")
        UserProfile.objects.filter(user=self.user).update(
            date_of_birth=date(1990, 1, 1), gender="", height_cm=165,
            weight_kg=60, smoking_status="never",
        )
        DailyCheckIn.objects.create(
            user=self.user, date=timezone.localdate(), sleep_hours=8,
            water_liters=2, exercise_minutes=30, weekly_activity_minutes=150,
            sitting_hours=6, screen_time_hours=3, mood=4, stress_level=2,
            energy_level=4,
        )
        self.client.force_login(self.user)

    def test_missing_gender_can_be_completed_on_results_page(self):
        response = self.client.get(reverse("dashboard:risks"))
        self.assertEqual(list(response.context["prediction_profile_form"].fields), ["gender"])
        self.assertEqual(response.context["risk_ready_count"], 0)
        self.assertContains(response, "Save details and show predictions")
        response = self.client.post(reverse("dashboard:risks"), {
            "action": "prediction_profile", "gender": "female",
        }, follow=True)
        self.assertEqual(response.context["risk_ready_count"], 4)
        self.assertContains(response, "Lifestyle screening estimate", count=4)
        self.assertNotContains(response, 'name="action" value="prediction_profile"')
        profile = UserProfile.objects.get(user=self.user)
        self.assertEqual(profile.gender, "female")
        self.assertEqual(profile.height_cm, 165)
        self.assertEqual(profile.smoking_status, "never")

    def test_invalid_gender_does_not_fabricate_predictions(self):
        response = self.client.post(reverse("dashboard:risks"), {
            "action": "prediction_profile", "gender": "invalid",
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["prediction_profile_form"].errors)
        self.assertEqual(response.context["risk_ready_count"], 0)

    def test_dashboard_shows_one_clear_next_action_for_missing_details(self):
        response = self.client.get(reverse("dashboard:home"))
        self.assertContains(response, "Finish your prediction details")
        self.assertNotContains(response, "Lifestyle screening estimate")
        self.assertNotContains(response, 'id="trendChart"')
        self.assertContains(response, 'Total Health Score')
