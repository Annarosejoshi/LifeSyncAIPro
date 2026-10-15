from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from mental_health.forms import (
    MentalWellnessAssessmentForm,
)
from mental_health.models import (
    MentalWellnessAssessment,
)


User = get_user_model()


class MentalWellnessViewIntegrationTests(
    TestCase
):

    def setUp(self):
        self.user = User.objects.create_user(
            username="mental_user",
            email="mental@example.com",
            password="TestPass123!",
        )

        self.other_user = (
            User.objects.create_user(
                username="other_mental_user",
                email="other_mental@example.com",
                password="TestPass123!",
            )
        )

        self.valid_data = {
            "mood": "4",
            "stress": "2",
            "anxiety": "2",
            "energy": "4",
            "motivation": "4",
            "concentration": "4",
            "social_wellbeing": "4",
            "workload": "2",
            "notes": "Integration test assessment.",
        }

    def create_assessment(
        self,
        user=None,
        date=None,
        **overrides,
    ):
        values = {
            "user": user or self.user,
            "date": date or timezone.localdate(),
            "mood": 4,
            "stress": 2,
            "anxiety": 2,
            "energy": 4,
            "motivation": 4,
            "concentration": 4,
            "social_wellbeing": 4,
            "workload": 2,
            "notes": "",
        }

        values.update(
            overrides
        )

        return (
            MentalWellnessAssessment.objects
            .create(
                **values
            )
        )

    def test_assessment_form_does_not_expose_date(self):
        form = (
            MentalWellnessAssessmentForm()
        )

        self.assertNotIn(
            "date",
            form.fields,
        )

    def test_assessment_create_requires_login(self):
        response = self.client.get(
            reverse(
                "mental_health:"
                "assessment_create"
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
        "mental_health.views."
        "evaluate_user_recommendation_progress"
    )
    def test_valid_assessment_is_saved_with_server_date_and_triggers_progress(
        self,
        mock_progress,
    ):
        self.client.force_login(
            self.user
        )

        data = dict(
            self.valid_data
        )

        data["date"] = str(
            timezone.localdate()
            - timedelta(days=10)
        )

        response = self.client.post(
            reverse(
                "mental_health:"
                "assessment_create"
            ),
            data=data,
        )

        self.assertEqual(
            response.status_code,
            302,
        )

        assessment = (
            MentalWellnessAssessment.objects
            .get(
                user=self.user
            )
        )

        self.assertEqual(
            assessment.date,
            timezone.localdate(),
        )

        self.assertEqual(
            assessment.anxiety,
            2,
        )

        mock_progress.assert_called_once_with(
            self.user
        )

    def test_anxiety_zero_is_rejected_and_nothing_is_saved(self):
        self.client.force_login(
            self.user
        )

        data = dict(
            self.valid_data
        )

        data["anxiety"] = "0"

        response = self.client.post(
            reverse(
                "mental_health:"
                "assessment_create"
            ),
            data=data,
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertFalse(
            MentalWellnessAssessment.objects
            .filter(
                user=self.user
            )
            .exists()
        )

        self.assertIn(
            "anxiety",
            response.context[
                "form"
            ].errors,
        )

    def test_existing_same_day_assessment_redirects_to_edit(self):
        assessment = (
            self.create_assessment()
        )

        self.client.force_login(
            self.user
        )

        response = self.client.get(
            reverse(
                "mental_health:"
                "assessment_create"
            )
        )

        self.assertRedirects(
            response,
            reverse(
                "mental_health:"
                "assessment_edit",
                kwargs={
                    "pk": assessment.pk,
                },
            ),
            fetch_redirect_response=False,
        )

    @patch(
        "mental_health.views."
        "evaluate_user_recommendation_progress"
    )
    def test_assessment_edit_preserves_date_and_triggers_progress(
        self,
        mock_progress,
    ):
        original_date = (
            timezone.localdate()
            - timedelta(days=1)
        )

        assessment = (
            self.create_assessment(
                date=original_date
            )
        )

        self.client.force_login(
            self.user
        )

        data = dict(
            self.valid_data
        )

        data["mood"] = "5"
        data["date"] = str(
            timezone.localdate()
        )

        response = self.client.post(
            reverse(
                "mental_health:"
                "assessment_edit",
                kwargs={
                    "pk": assessment.pk,
                },
            ),
            data=data,
        )

        self.assertEqual(
            response.status_code,
            302,
        )

        assessment.refresh_from_db()

        self.assertEqual(
            assessment.date,
            original_date,
        )

        self.assertEqual(
            assessment.mood,
            5,
        )

        mock_progress.assert_called_once_with(
            self.user
        )

    def test_user_cannot_edit_another_users_assessment(self):
        other_assessment = (
            self.create_assessment(
                user=self.other_user
            )
        )

        self.client.force_login(
            self.user
        )

        response = self.client.get(
            reverse(
                "mental_health:"
                "assessment_edit",
                kwargs={
                    "pk":
                        other_assessment.pk,
                },
            )
        )

        self.assertEqual(
            response.status_code,
            404,
        )
