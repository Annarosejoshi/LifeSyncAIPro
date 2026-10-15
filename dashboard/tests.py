from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from datetime import date
from unittest.mock import patch

from django.utils import timezone

from accounts.models import UserProfile
from lifestyle.models import BMIRecord, DailyCheckIn
from recommendations.models import RecommendationHistory


class UserDashboardTests(TestCase):
	def test_dashboard_shows_each_risk_screening_state(self):
		user = User.objects.create_user(
			username="risk_cards_user",
			password="StrongPass123",
		)
		self.client.force_login(user)

		response = self.client.get(reverse("dashboard:risks"))

		self.assertEqual(response.status_code, 200)
		self.assertEqual(len(response.context["risk_cards"]), 4)
		for title in (
			"Diabetes screening",
			"Hypertension screening",
			"Cardiovascular screening",
			"Mental wellness screening",
		):
			self.assertContains(response, title)

	@patch(
		"dashboard.views.answer_with_optional_local_model",
		return_value=("Your check-in records 7.5 hours of sleep.", "Reviewed local guidance"),
	)
	def test_dashboard_answers_coach_question_on_the_same_page(self, mock_coach):
		user = User.objects.create_user(
			username="dashboard_coach_user",
			password="StrongPass123",
		)
		self.client.force_login(user)

		response = self.client.post(
			reverse("dashboard:home"),
			{"action": "coach_question", "coach_question": "How did I sleep?"},
		)

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, "Your check-in records 7.5 hours of sleep.")
		mock_coach.assert_called_once()

	def test_public_project_landing_page_has_account_actions(self):
		response = self.client.get(reverse("dashboard:landing"))

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, "Life Sync AI")
		self.assertContains(response, reverse("accounts:register"))
		self.assertContains(response, reverse("accounts:login"))

	def test_new_user_sees_their_own_empty_dashboard(self):
		previous_user = User.objects.create_user(
			username="previoususer",
			first_name="Anna",
			last_name="Rose Joshi",
			password="StrongPass123",
		)
		DailyCheckIn.objects.create(
			user=previous_user,
			date="2026-10-10",
			sleep_hours=7.5,
			sleep_quality=8,
			water_liters=2,
			exercise_minutes=30,
			mood=4,
			stress_level=2,
			energy_level=4,
		)
		new_user = User.objects.create_user(
			username="newuser",
			first_name="New",
			last_name="Member",
			password="StrongPass123",
		)
		self.client.force_login(new_user)

		response = self.client.get(reverse("dashboard:home"))

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, "New Member")
		self.assertNotContains(response, "Anna Rose Joshi")
		self.assertContains(response, '"hasCheckin": false')
		self.assertContains(response, '"sleep": null')

	def test_dashboard_loads_signed_in_users_checkin_and_recommendations(self):
		user = User.objects.create_user(
			username="wellness_user",
			first_name="Wellness",
			last_name="Member",
			password="StrongPass123",
		)
		DailyCheckIn.objects.create(
			user=user,
			date=timezone.localdate(),
			sleep_hours=7.5,
			sleep_quality=8,
			water_liters=2.5,
			exercise_minutes=35,
			fruit_vegetable_servings=4,
			mood=4,
			stress_level=2,
			energy_level=4,
		)
		RecommendationHistory.objects.create(
			user=user,
			recommendation_date=timezone.localdate(),
			category="Hydration",
			title="Keep water nearby",
			message="Continue recording your water intake.",
			metric="water",
		)
		self.client.force_login(user)

		response = self.client.get(reverse("dashboard:home"))

		self.assertEqual(response.status_code, 200)
		self.assertTrue(response.context["dashboard_data"]["hasCheckin"])
		self.assertEqual(response.context["dashboard_data"]["sleep"], 7.5)
		self.assertEqual(response.context["dashboard_data"]["nutrition"], 4)
		self.assertEqual(response.context["top_recommendations"][0].title, "Keep water nearby")

	def test_dashboard_saves_inline_checkin_for_logged_in_user(self):
		user = User.objects.create_user(
			username="inline_checkin_user",
			password="StrongPass123",
		)
		self.client.force_login(user)

		response = self.client.post(
			reverse("dashboard:home"),
			{
				"action": "daily_checkin",
				"sleep_hours": "7.5",
				"water_liters": "2.5",
				"exercise_hours": "0",
				"exercise_extra_minutes": "35",
				"steps": "8000",
				"screen_time_hours": "4",
				"fruit_vegetable_items": ["apple", "banana", "carrot", "tomato"],
				"breakfast_taken": "on",
				"lunch_taken": "on",
				"dinner_taken": "on",
				"mood": "4",
				"stress_level": "2",
				"energy_level": "4",
				"notes": "Dashboard check-in test.",
			},
		)

		self.assertRedirects(
			response,
			reverse("dashboard:risks"),
			fetch_redirect_response=False,
		)
		checkin = DailyCheckIn.objects.get(
			user=user,
			date=timezone.localdate(),
		)
		self.assertEqual(checkin.sleep_hours, 7.5)
		self.assertEqual(checkin.fruit_vegetable_servings, 4)
		self.assertEqual(
			checkin.fruit_vegetable_items,
			["apple", "banana", "carrot", "tomato"],
		)
		self.assertTrue(
			RecommendationHistory.objects.filter(
				user=user,
				recommendation_date=timezone.localdate(),
			).exists()
		)

	def test_dashboard_saves_dated_bmi_result_for_user(self):
		user = User.objects.create_user(
			username="bmi_user",
			password="StrongPass123",
		)
		UserProfile.objects.filter(user=user).update(
			date_of_birth=date(1990, 1, 1),
		)
		self.client.force_login(user)

		response = self.client.post(
			reverse("dashboard:home"),
			{
				"action": "bmi_calculation",
				"height_cm": "170",
				"weight_kg": "70",
			},
		)

		self.assertEqual(response.status_code, 302)
		self.assertEqual(
			response["Location"],
			reverse("dashboard:bmi"),
		)
		record = BMIRecord.objects.get(user=user)
		self.assertEqual(record.date, timezone.localdate())
		self.assertAlmostEqual(record.bmi, 24.22, places=2)
		self.assertEqual(record.category, "Usual adult range")

	def test_dashboard_does_not_apply_adult_bmi_category_to_minors(self):
		user = User.objects.create_user(
			username="minor_bmi_user",
			password="StrongPass123",
		)
		UserProfile.objects.filter(user=user).update(
			date_of_birth=date(2012, 1, 1),
		)
		self.client.force_login(user)

		self.client.post(
			reverse("dashboard:home"),
			{
				"action": "bmi_calculation",
				"height_cm": "150",
				"weight_kg": "45",
			},
		)

		record = BMIRecord.objects.get(user=user)
		self.assertEqual(record.category, "Age-specific chart required")

	def test_dashboard_does_not_apply_adult_bmi_category_without_age(self):
		user = User.objects.create_user(
			username="unknown_age_bmi_user",
			password="StrongPass123",
		)
		self.client.force_login(user)

		self.client.post(
			reverse("dashboard:home"),
			{
				"action": "bmi_calculation",
				"height_cm": "170",
				"weight_kg": "70",
			},
		)

		record = BMIRecord.objects.get(user=user)
		self.assertEqual(record.category, "Age not available")


class DashboardPageTests(TestCase):
    def test_landing_focuses_on_predictions_without_submission_details(self):
        response=self.client.get(reverse('dashboard:landing'))
        self.assertContains(response,'Daily health risk predictions')
        for text in ('Submitted by','Project guide','About the project','Asst. Prof.'):
            self.assertNotContains(response,text)

    def test_predictions_can_be_refreshed_and_filtered(self):
        user=User.objects.create_user(username='refreshpredictions')
        self.client.force_login(user)
        response=self.client.post(reverse('dashboard:risks'),{'action':'refresh_predictions'})
        self.assertEqual(response.status_code,200)
        self.assertContains(response,'data-condition-filter="all"')
        self.assertContains(response,"Update today's check-in")
        self.assertContains(response, 'id="overall-screening"')
        self.assertContains(response, 'Your health overview')
        self.assertNotContains(response, 'Results calculated:')
        self.assertNotContains(response, 'needs attention')
        self.assertNotContains(response,'class="checkin-module"')
        self.assertNotContains(response,'decision threshold')

    def test_pages_require_login(self):
        for name in ("checkin", "bmi", "risks", "reports"):
            self.assertEqual(self.client.get(reverse("dashboard:" + name)).status_code, 302)

    def test_features_have_separate_pages_without_demo_forms(self):
        user = User.objects.create_user(username="page_user", password="StrongPass123")
        self.client.force_login(user)
        for name, marker in (("home", 'id="overview"'), ("checkin", 'id="check-in-section"'), ("bmi", 'id="bmi-calculator"'), ("risks", 'id="risk-section"'), ("reports", 'id="trends"')):
            response = self.client.get(reverse("dashboard:" + name))
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, marker)
            self.assertNotContains(response, 'id="lifestyleForm"')
            self.assertNotContains(response, 'id="profileForm"')
        response = self.client.get(reverse("dashboard:home"))
        self.assertNotContains(response, 'id="check-in-section"')
        self.assertNotContains(response, 'id="risk-section"')
