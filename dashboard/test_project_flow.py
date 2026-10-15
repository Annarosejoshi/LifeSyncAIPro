"""Exercise the real user journey with CSRF checks and deployed models."""
from html.parser import HTMLParser
from urllib.parse import urlsplit

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from lifestyle.models import DailyCheckIn
from mental_health.models import MentalWellnessAssessment
from notifications.models import MedicationRecord
from predictions.models import PredictionHistory


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.startswith("/") and not href.startswith("//"):
                self.paths.add(urlsplit(href).path)


class FullProjectFlowTests(TestCase):
    def setUp(self):
        self.client = Client(enforce_csrf_checks=True)

    def post(self, name, data):
        if name != "accounts:logout":
            self.client.get(reverse(name), follow=True)
        return self.client.post(reverse(name), {
            **data, "csrfmiddlewaretoken": self.client.cookies["csrftoken"].value,
        })

    def test_registration_to_daily_predictions_reports_and_logout(self):
        self.assertEqual(self.client.get(reverse("dashboard:landing")).status_code, 200)
        response = self.post("accounts:register", {
            "first_name": "Flow", "last_name": "Tester", "username": "flowtester",
            "email": "flow@example.com", "password1": "DailyHabitSecure482!",
            "password2": "DailyHabitSecure482!", "date_of_birth": "1990-01-01",
            "gender": "female", "height_cm": "165", "weight_kg": "60",
            "smoking_status": "never", "diabetes_history": "no",
            "family_diabetes_history": "no", "heart_disease_history": "no",
        })
        self.assertRedirects(response, reverse("accounts:login"))
        response = self.post("accounts:login", {
            "username": "flowtester", "password": "DailyHabitSecure482!",
        })
        self.assertRedirects(response, reverse("dashboard:home"))
        user = User.objects.get(username="flowtester")
        empty = self.client.get(reverse("dashboard:risks"))
        self.assertTrue(all(not result["prediction_available"] for _, result in empty.context["risk_cards"]))

        inputs = {
            "action": "daily_checkin", "sleep_hours": "8", "water_liters": "2",
            "exercise_hours": "0", "exercise_extra_minutes": "30",
            "screen_time_hours": "3", "mood": "4", "stress_level": "2",
            "energy_level": "4", "breakfast_taken": "on", "lunch_taken": "on",
            "dinner_taken": "on",
        }
        response = self.post("dashboard:checkin", inputs)
        self.assertRedirects(response, reverse("dashboard:risks"))
        self.assertEqual(DailyCheckIn.objects.filter(user=user).count(), 1)
        self.assertEqual(PredictionHistory.objects.filter(user=user).count(), 4)

        response = self.post("dashboard:bmi", {
            "action": "bmi_calculation", "height_cm": "170", "weight_kg": "70",
        })
        self.assertRedirects(response, reverse("dashboard:bmi"))
        results = self.client.get(reverse("dashboard:risks"))
        self.assertEqual(results.context["risk_cards"][0][1]["input_snapshot"]["bmi"], round(70 / 1.7 ** 2, 2))
        overview = self.client.get(reverse("dashboard:home"))
        self.assertContains(overview, "Total Health Score")
        snapshot = overview.context["risk_cards"][0][1]["input_snapshot"]
        self.assertEqual(snapshot["sleep_hours"], 8)
        self.assertNotIn("total_activity_minutes_week", snapshot)
        self.assertNotIn("sedentary_minutes_day", snapshot)
        self.assertEqual(overview.context["dashboard_data"]["scoreTrend"][-1], overview.context["ai_health_score"])
        predictions_before = {title: result["probability"] for title, result in overview.context["risk_cards"]}

        inputs.update(sleep_hours="4")
        self.post("dashboard:checkin", inputs)
        overview = self.client.get(reverse("dashboard:home"))
        predictions_after = {title: result["probability"] for title, result in overview.context["risk_cards"]}
        self.assertNotEqual(predictions_before, predictions_after)
        self.assertEqual(DailyCheckIn.objects.filter(user=user).count(), 1)
        self.assertEqual(PredictionHistory.objects.filter(user=user).count(), 4)

        self.post("mental_health:assessment_create", {
            key: "3" for key in ("mood", "stress", "anxiety", "energy", "motivation",
                                 "concentration", "social_wellbeing", "workload")
        })
        self.assertEqual(MentalWellnessAssessment.objects.filter(user=user).count(), 1)
        self.post("notifications:medication_create", {
            "medication_name": "Prescribed medicine", "dosage": "As prescribed",
            "scheduled_date": timezone.localdate().isoformat(), "scheduled_time": "20:00",
        })
        self.assertEqual(MedicationRecord.objects.filter(user=user).count(), 1)

        links = LinkParser()
        for page in ("dashboard:home", "dashboard:checkin", "dashboard:bmi",
                     "dashboard:risks", "dashboard:reports", "accounts:profile",
                     "accounts:profile_edit", "mental_health:assessment_create",
                     "mental_health:history", "notifications:medication_list",
                     "recommendations:home", "recommendations:history"):
            response = self.client.get(reverse(page), follow=True)
            self.assertEqual(response.status_code, 200, page)
            links.feed(response.content.decode())
        for path in links.paths:
            if path != reverse("accounts:logout"):
                with self.subTest(link=path):
                    response = self.client.get(path, follow=True)
                    self.assertEqual(response.status_code, 200)
        response = self.post("accounts:logout", {})
        self.assertRedirects(response, reverse("dashboard:landing"))
        self.assertNotIn("_auth_user_id", self.client.session)
