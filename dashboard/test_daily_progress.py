from datetime import timedelta

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from lifestyle.models import DailyCheckIn


class DailyProgressTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='daily_progress')
        self.client.force_login(self.user)

    def record(self, day):
        return DailyCheckIn.objects.create(user=self.user, date=day, sleep_hours=7.5,
            water_liters=2.5, exercise_minutes=45, screen_time_hours=4,
            mood=4, stress_level=2, energy_level=4)

    def test_chart_is_available_on_the_first_saved_day(self):
        self.record(timezone.localdate())
        response = self.client.get(reverse('dashboard:reports'))
        self.assertContains(response, "Today's daily progress")
        self.assertContains(response, 'id="trendChart"')
        self.assertEqual(response.context['dashboard_data']['sleep'], 7.5)
        self.assertEqual(response.context['dashboard_data']['water'], 2.5)
        self.assertEqual(self.user.daily_checkins.count(), 1)

    def test_yesterdays_entry_does_not_fill_todays_chart(self):
        self.record(timezone.localdate()-timedelta(days=1))
        response = self.client.get(reverse('dashboard:reports'))
        self.assertNotContains(response, 'id="trendChart"')
        self.assertContains(response, "Save today's check-in")
