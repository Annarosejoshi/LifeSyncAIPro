from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from lifestyle.forms import DailyCheckInForm
from lifestyle.models import DailyCheckIn


User = get_user_model()


class DailyCheckInViewIntegrationTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="lifestyle_user",
            email="lifestyle@example.com",
            password="TestPass123!",
        )

        self.other_user = User.objects.create_user(
            username="other_lifestyle_user",
            email="other_lifestyle@example.com",
            password="TestPass123!",
        )

        self.valid_data = {
            "sleep_hours": "7.5",
            "sleep_quality": "8",
            "water_liters": "2.2",
            "exercise_hours": "0",
            "exercise_extra_minutes": "30",
            "steps": "7000",
            "mood": "4",
            "stress_level": "2",
            "energy_level": "4",
            "screen_time_hours": "5.0",
            "fruit_vegetable_servings": "4",
            "breakfast_taken": "on",
            "lunch_taken": "on",
            "dinner_taken": "on",
            "notes": "Integration test check-in.",
        }

    def create_checkin(
        self,
        user=None,
        date=None,
        **overrides,
    ):
        values = {
            "user": user or self.user,
            "date": date or timezone.localdate(),
            "sleep_hours": 7.0,
            "sleep_quality": 7,
            "water_liters": 2.0,
            "exercise_minutes": 20,
            "steps": 5000,
            "mood": 4,
            "stress_level": 2,
            "energy_level": 4,
            "screen_time_hours": 5.0,
            "fruit_vegetable_servings": 3,
            "breakfast_taken": True,
            "lunch_taken": True,
            "dinner_taken": True,
            "notes": "",
        }
        values.update(overrides)

        return DailyCheckIn.objects.create(
            **values
        )

    def test_checkin_form_does_not_expose_date(self):
        form = DailyCheckInForm()

        self.assertNotIn(
            "date",
            form.fields,
        )

    def test_sleep_quality_is_not_requested(self):
        form = DailyCheckInForm()
        self.assertNotIn("sleep_quality", form.fields)

    def test_selected_produce_items_are_saved_and_counted(self):
        data = dict(self.valid_data)
        data.pop("sleep_quality")
        data.pop("fruit_vegetable_servings")
        data["fruit_vegetable_items"] = ["apple", "leafy_greens", "carrot"]

        form = DailyCheckInForm(data=data)

        self.assertTrue(form.is_valid(), form.errors)
        checkin = form.save(commit=False)

        self.assertEqual(
            checkin.fruit_vegetable_items,
            ["apple", "leafy_greens", "carrot"],
        )
        self.assertEqual(checkin.fruit_vegetable_servings, 3)

    def test_selected_produce_items_are_restored_when_editing(self):
        checkin = self.create_checkin()
        checkin.fruit_vegetable_items = ["apple", "leafy_greens"]
        checkin.fruit_vegetable_servings = 2
        checkin.save()

        form = DailyCheckInForm(instance=checkin)

        self.assertEqual(
            form["fruit_vegetable_items"].value(),
            ["apple", "leafy_greens"],
        )

    def test_exercise_hours_and_extra_minutes_are_combined(self):
        data = dict(self.valid_data)
        data["exercise_hours"] = "1"
        data["exercise_extra_minutes"] = "25"
        form = DailyCheckInForm(data=data)

        self.assertTrue(form.is_valid(), form.errors)
        checkin = form.save(commit=False)
        self.assertEqual(checkin.exercise_minutes, 85)

    def test_exercise_hours_and_minutes_save_as_total_minutes(self):
        data = dict(self.valid_data)
        data["exercise_hours"] = "1"
        data["exercise_extra_minutes"] = "25"
        form = DailyCheckInForm(data=data)

        self.assertTrue(form.is_valid(), form.errors)
        checkin = form.save(commit=False)

        self.assertEqual(checkin.exercise_minutes, 85)

    @patch(
        "lifestyle.views."
        "evaluate_user_recommendation_progress"
    )
    def test_checkin_saves_when_sleep_quality_is_unknown(self, mock_progress):
        self.client.force_login(self.user)
        data = dict(self.valid_data)
        data.pop("sleep_quality")

        response = self.client.post(
            reverse("lifestyle:checkin_create"),
            data=data,
        )

        self.assertEqual(response.status_code, 302)
        checkin = DailyCheckIn.objects.get(user=self.user)
        self.assertIsNone(checkin.sleep_quality)
        mock_progress.assert_called_once_with(self.user)

    def test_checkin_create_requires_login(self):
        response = self.client.get(
            reverse(
                "lifestyle:checkin_create"
            )
        )

        self.assertEqual(
            response.status_code,
            302,
        )

        self.assertIn(
            "/login",
            response.url,
        )

    @patch(
        "lifestyle.views."
        "evaluate_user_recommendation_progress"
    )
    def test_valid_checkin_is_saved_with_server_date_and_triggers_progress(
        self,
        mock_progress,
    ):
        self.client.force_login(
            self.user
        )

        data = dict(
            self.valid_data
        )

        # The current form intentionally has no date field.
        # A malicious/old client-supplied date must therefore
        # not control the stored date.
        data["date"] = str(
            timezone.localdate()
            - timedelta(days=5)
        )

        response = self.client.post(
            reverse(
                "lifestyle:checkin_create"
            ),
            data=data,
        )

        self.assertEqual(
            response.status_code,
            302,
        )

        checkin = DailyCheckIn.objects.get(
            user=self.user
        )

        self.assertEqual(
            checkin.date,
            timezone.localdate(),
        )

        self.assertAlmostEqual(
            float(checkin.sleep_hours),
            7.5,
        )

        mock_progress.assert_called_once_with(
            self.user
        )

    def test_existing_same_day_checkin_redirects_to_edit(self):
        self.client.force_login(
            self.user
        )

        checkin = self.create_checkin()

        response = self.client.get(
            reverse(
                "lifestyle:checkin_create"
            )
        )

        self.assertRedirects(
            response,
            reverse(
                "lifestyle:checkin_edit",
                kwargs={
                    "pk": checkin.pk,
                },
            ),
            fetch_redirect_response=False,
        )

    def test_invalid_checkin_stays_on_form_and_is_not_saved(self):
        self.client.force_login(
            self.user
        )

        data = dict(
            self.valid_data
        )

        data["stress_level"] = "0"

        response = self.client.post(
            reverse(
                "lifestyle:checkin_create"
            ),
            data=data,
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertFalse(
            DailyCheckIn.objects.filter(
                user=self.user
            ).exists()
        )

        self.assertTrue(
            response.context[
                "form"
            ].errors
        )

    @patch(
        "lifestyle.views."
        "evaluate_user_recommendation_progress"
    )
    def test_checkin_edit_preserves_date_and_triggers_progress(
        self,
        mock_progress,
    ):
        original_date = (
            timezone.localdate()
            - timedelta(days=1)
        )

        checkin = self.create_checkin(
            date=original_date
        )

        self.client.force_login(
            self.user
        )

        data = dict(
            self.valid_data
        )

        data["water_liters"] = "3.0"
        data["date"] = str(
            timezone.localdate()
        )

        response = self.client.post(
            reverse(
                "lifestyle:checkin_edit",
                kwargs={
                    "pk": checkin.pk,
                },
            ),
            data=data,
        )

        self.assertEqual(
            response.status_code,
            302,
        )

        checkin.refresh_from_db()

        self.assertEqual(
            checkin.date,
            original_date,
        )

        self.assertAlmostEqual(
            float(
                checkin.water_liters
            ),
            3.0,
        )

        mock_progress.assert_called_once_with(
            self.user
        )

    def test_user_cannot_edit_another_users_checkin(self):
        other_checkin = (
            self.create_checkin(
                user=self.other_user
            )
        )

        self.client.force_login(
            self.user
        )

        response = self.client.get(
            reverse(
                "lifestyle:checkin_edit",
                kwargs={
                    "pk": other_checkin.pk,
                },
            )
        )

        self.assertEqual(
            response.status_code,
            404,
        )

    def test_owner_can_delete_checkin_but_other_user_cannot(self):
        checkin = self.create_checkin()
        other_checkin = self.create_checkin(user=self.other_user)
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("lifestyle:checkin_delete", args=[checkin.pk])
        )
        other_response = self.client.post(
            reverse("lifestyle:checkin_delete", args=[other_checkin.pk])
        )

        self.assertRedirects(response, reverse("lifestyle:checkin_history"))
        self.assertFalse(DailyCheckIn.objects.filter(pk=checkin.pk).exists())
        self.assertEqual(other_response.status_code, 404)
        self.assertTrue(DailyCheckIn.objects.filter(pk=other_checkin.pk).exists())
