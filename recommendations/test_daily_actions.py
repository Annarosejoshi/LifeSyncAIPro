from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from lifestyle.models import DailyCheckIn


class DailyActionPagesTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="daily_action_user")
        self.client.force_login(self.user)
        self.record = DailyCheckIn.objects.create(
            user=self.user, date=timezone.localdate(), sleep_hours=5,
            water_liters=1, exercise_minutes=10, screen_time_hours=4,
            mood=4, stress_level=2, energy_level=4,
            breakfast_taken=True, lunch_taken=False, dinner_taken=False,
        )

    def test_progress_and_coach_show_current_habits_and_simple_actions(self):
        for page in ("dashboard:reports", "recommendations:home"):
            with self.subTest(page=page):
                response = self.client.get(reverse(page))
                self.assertContains(response, "Plan an earlier bedtime tonight.")
                self.assertContains(response, "Recorded: 10 minutes")
                self.assertContains(response, "1 of 3 meals recorded")
                self.assertNotContains(response, "7-day")
                self.assertNotContains(response, "Coach Is Using")
                self.assertNotContains(response, "Reviewed local guidance")
        self.record.sleep_hours = 8
        self.record.exercise_minutes = 30
        self.record.lunch_taken = self.record.dinner_taken = True
        self.record.save()
        for page in ("dashboard:reports", "recommendations:home"):
            response = self.client.get(reverse(page))
            self.assertNotContains(response, "Plan an earlier bedtime tonight.")
            self.assertContains(response, "Recorded: 30 minutes")
            self.assertContains(response, "3 of 3 meals recorded")

    def test_meals_render_as_three_separate_named_controls(self):
        response = self.client.get(reverse("dashboard:checkin"))
        for name in ("breakfast_taken", "lunch_taken", "dinner_taken"):
            self.assertContains(response, f'name="{name}"', count=1)
        self.assertNotContains(response, 'name="sitting_hours"')
        self.assertNotContains(response, 'name="weekly_activity_minutes"')

    def test_coach_answer_uses_today_without_displaying_implementation(self):
        response = self.client.post(reverse("recommendations:home"), {
            "coach_question": "How much sleep did I get?",
        })
        self.assertContains(response, "5 hours")
        self.assertNotContains(response, "not configured")

    def test_long_coach_question_is_handled(self):
        response = self.client.post(reverse("recommendations:home"), {
            "coach_question": "a" * 501,
        })
        self.assertContains(response, "500 characters or fewer")
