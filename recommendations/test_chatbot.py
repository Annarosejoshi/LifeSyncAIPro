from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from lifestyle.models import DailyCheckIn


class FloatingChatbotTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="chat_user")
        self.client.force_login(self.user)

    def record(self, user, day, sleep):
        return DailyCheckIn.objects.create(
            user=user, date=day, sleep_hours=sleep, water_liters=2,
            exercise_minutes=30, screen_time_hours=4, mood=4,
            stress_level=2, energy_level=4,
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_reply_uses_only_current_users_today_checkin(self):
        today = timezone.localdate()
        self.record(self.user, today, 6.5)
        self.record(self.user, today - timedelta(days=1), 9)
        other = User.objects.create_user(username="another_chat_user")
        self.record(other, today, 3)
        response = self.client.post(reverse("recommendations:chat"), {
            "question": "How much sleep did I get?",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("6.5 hours", response.json()["answer"])
        self.assertEqual(set(response.json()), {"answer"})

    @patch.dict("os.environ", {}, clear=True)
    def test_yesterday_is_not_used_for_todays_reply(self):
        self.record(self.user, timezone.localdate() - timedelta(days=1), 9)
        response = self.client.post(reverse("recommendations:chat"), {"question": "How much sleep did I get?"})
        self.assertIn("not recorded", response.json()["answer"])
        self.assertNotIn("9 hours", response.json()["answer"])

    @patch("recommendations.views.answer_with_optional_local_model")
    def test_invalid_questions_are_rejected_before_generating_reply(self, answer):
        for question in ("", "  ", "x" * 501):
            response = self.client.post(reverse("recommendations:chat"), {"question": question})
            self.assertEqual(response.status_code, 400)
        answer.assert_not_called()
        self.assertEqual(self.client.get(reverse("recommendations:chat")).status_code, 405)

    @patch.dict("os.environ", {}, clear=True)
    def test_login_and_csrf_required_and_valid_submission_works(self):
        guest = Client()
        self.assertEqual(guest.post(reverse("recommendations:chat"), {"question": "Sleep?"}).status_code, 302)
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.user)
        self.assertEqual(client.post(reverse("recommendations:chat"), {"question": "Sleep?"}).status_code, 403)
        client.get(reverse("dashboard:home"))
        response = client.post(reverse("recommendations:chat"), {
            "question": "Sleep?", "csrfmiddlewaretoken": client.cookies["csrftoken"].value,
        })
        self.assertEqual(response.status_code, 200)

    def test_chat_available_across_both_layouts_without_coach_sidebar_link(self):
        for page in ("dashboard:home", "dashboard:checkin", "dashboard:reports", "accounts:profile"):
            response = self.client.get(reverse(page))
            self.assertContains(response, 'id="wellnessChatForm"', count=1)
            self.assertContains(response, 'id="wellnessChatPanel"', count=1)
            self.assertNotContains(response, 'href="' + reverse("recommendations:home") + '"')
        guest = Client().get(reverse("dashboard:landing"))
        self.assertNotContains(guest, 'id="wellnessChatForm"')
