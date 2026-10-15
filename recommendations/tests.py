from datetime import timedelta
import json
import os
from io import BytesIO
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.test import SimpleTestCase, TestCase
from django.utils import timezone
from django.urls import reverse

from lifestyle.models import DailyCheckIn
from mental_health.models import MentalWellnessAssessment

from .models import RecommendationHistory
from .progress_service import (
    MENTAL_WELLNESS_CONFIG,
    apply_progress_result,
    calculate_percentage_change,
    evaluate_mental_wellness_progress,
)
from .services import (
    calculate_mental_wellness_score,
    detect_metric,
    get_baseline_value,
    save_recommendation_history,
)
from .coach import answer_wellness_question, answer_with_optional_local_model


class LocalCoachModelTests(SimpleTestCase):
    def test_without_configured_model_uses_reviewed_local_answer(self):
        with patch.dict(os.environ, {}, clear=True):
            answer, mode = answer_with_optional_local_model(
                "How can I improve my sleep?"
            )

        self.assertIn("regular sleep", answer)
        self.assertIn("not configured", mode)

    def test_sensitive_question_is_not_sent_to_model(self):
        with patch.dict(os.environ, {"LIFESYNC_OLLAMA_MODEL": "local-model"}), \
                patch("recommendations.coach.urlopen") as open_model:
            answer, mode = answer_with_optional_local_model(
                "Should I stop my medication?"
            )

        open_model.assert_not_called()
        self.assertIn("cannot recommend", answer)
        self.assertIn("not sent", mode)

    def test_configured_local_model_answers_supported_question(self):
        model_response = BytesIO(
            json.dumps({"response": "Try a consistent bedtime."}).encode()
        )
        with patch.dict(os.environ, {"LIFESYNC_OLLAMA_MODEL": "test-model"}), \
                patch("recommendations.coach.urlopen", return_value=model_response) as open_model:
            answer, mode = answer_with_optional_local_model(
                "How can I improve my sleep?"
            )

        self.assertEqual(answer, "Try a consistent bedtime.")
        self.assertIn("Local Ollama", mode)
        request = open_model.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")


class CoachQuestionViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="coach_question_user",
            password="StrongPass123",
        )

    def test_question_returns_local_guidance_when_no_model_is_configured(self):
        self.client.force_login(self.user)
        with patch.dict(os.environ, {}, clear=True), patch(
            "recommendations.views.get_all_health_risks",
            return_value={},
        ):
            response = self.client.post(
                reverse("recommendations:home"),
                {"coach_question": "How can I improve my sleep?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("not recorded", response.context["coach_answer"])
        self.assertIn("not configured", response.context["coach_mode"])


class WellnessCoachAnswerTests(SimpleTestCase):
    def test_sleep_answer_uses_logged_value(self):
        checkin = type("CheckIn", (), {"sleep_hours": 6.5})()
        answer = answer_wellness_question("How much sleep did I get?", checkin)
        self.assertIn("6.5 hours", answer)

    def test_medication_question_does_not_change_dose(self):
        answer = answer_wellness_question("Should I stop my medication?")
        self.assertIn("cannot recommend starting, stopping, or changing", answer)

    def test_emergency_question_directs_to_emergency_services(self):
        answer = answer_wellness_question("I have chest pain")
        self.assertIn("emergency service now", answer)

    def test_unknown_question_does_not_invent_medical_answer(self):
        answer = answer_wellness_question("What illness do I have?")
        self.assertIn("cannot diagnose", answer)


class RecommendationServiceTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser",
            password="TestPass123!",
        )
        self.today = timezone.localdate()

        self.checkin = DailyCheckIn.objects.create(
            user=self.user,
            date=self.today,
            sleep_hours=5.0,
            sleep_quality=5,
            water_liters=1.0,
            exercise_minutes=10,
            steps=3000,
            mood=3,
            stress_level=4,
            energy_level=3,
            screen_time_hours=8.0,
            fruit_vegetable_servings=2,
            breakfast_taken=True,
            lunch_taken=True,
            dinner_taken=True,
        )

    def test_detect_metric(self):
        cases = [
            ({"category": "Sleep", "title": "Prioritize Sleep"}, "sleep"),
            ({"category": "Hydration", "title": "Increase Hydration"}, "water"),
            ({"category": "Activity", "title": "Increase Daily Movement"}, "steps"),
            ({"category": "Activity", "title": "Add More Movement"}, "exercise"),
            ({"category": "Nutrition", "title": "Improve Food Variety"}, "nutrition"),
            ({"category": "Mental Wellness", "title": "Give Extra Attention to Mental Wellness"}, "mental_wellness"),
        ]

        for recommendation, expected in cases:
            with self.subTest(recommendation=recommendation):
                self.assertEqual(detect_metric(recommendation), expected)

    def test_daily_baseline_value(self):
        recommendation = {
            "category": "Sleep",
            "title": "Prioritize Sleep",
        }

        self.assertEqual(
            get_baseline_value(
                recommendation,
                checkin=self.checkin,
            ),
            5.0,
        )

    def test_save_history_does_not_duplicate_same_title_same_day(self):
        recommendation = {
            "category": "Sleep",
            "title": "Prioritize Sleep",
            "message": "Aim for more consistent sleep.",
            "priority": "high",
        }

        save_recommendation_history(
            self.user,
            [recommendation],
            checkin=self.checkin,
        )

        save_recommendation_history(
            self.user,
            [recommendation],
            checkin=self.checkin,
        )

        self.assertEqual(
            RecommendationHistory.objects.filter(
                user=self.user,
                recommendation_date=self.today,
                title="Prioritize Sleep",
            ).count(),
            1,
        )


class MentalWellnessScoreTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="mentaluser",
            password="TestPass123!",
        )

    def test_score_calculation(self):
        assessment = MentalWellnessAssessment(
            user=self.user,
            date=timezone.localdate(),
            mood=4,
            stress=2,
            anxiety=2,
            energy=4,
            motivation=4,
            concentration=4,
            social_wellbeing=4,
            workload=2,
        )

        # Values become:
        # 4,4,4,4,4 and reverse-scored 4,4,4 -> 4/5 * 100 = 80.
        self.assertEqual(
            calculate_mental_wellness_score(assessment),
            80.0,
        )

    def test_assessment_validation_rejects_zero_anxiety(self):
        assessment = MentalWellnessAssessment(
            user=self.user,
            date=timezone.localdate(),
            mood=4,
            stress=2,
            anxiety=0,
            energy=4,
            motivation=4,
            concentration=4,
            social_wellbeing=4,
            workload=2,
        )

        with self.assertRaises(ValidationError):
            assessment.full_clean()


class ProgressHelperTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="progressuser",
            password="TestPass123!",
        )
        self.today = timezone.localdate()

    def make_recommendation(self, **overrides):
        values = {
            "user": self.user,
            "recommendation_date": self.today,
            "category": "Sleep",
            "title": "Prioritize Sleep",
            "message": "Improve sleep.",
            "priority": "high",
            "metric": "sleep",
            "baseline_value": 5.0,
            "latest_value": 5.0,
            "status": "active",
        }
        values.update(overrides)
        return RecommendationHistory.objects.create(**values)

    def test_percentage_change_higher_is_better(self):
        self.assertEqual(
            calculate_percentage_change(5, 6, "higher"),
            20.0,
        )

    def test_percentage_change_lower_is_better(self):
        self.assertEqual(
            calculate_percentage_change(8, 6, "lower"),
            25.0,
        )

    def test_progress_becomes_improving(self):
        recommendation = self.make_recommendation()
        progress_date = self.today + timedelta(days=1)

        apply_progress_result(
            recommendation,
            latest_value=6.0,
            config={
                "direction": "higher",
                "completion_target": 7.0,
                "improvement_delta": 0.5,
            },
            progress_date=progress_date,
        )

        recommendation.refresh_from_db()

        self.assertEqual(recommendation.status, "improving")
        self.assertEqual(recommendation.latest_value, 6.0)
        self.assertEqual(recommendation.improvement_percentage, 20.0)
        self.assertEqual(recommendation.first_improved_at, progress_date)
        self.assertIsNone(recommendation.completed_at)

    def test_progress_becomes_completed(self):
        recommendation = self.make_recommendation()
        progress_date = self.today + timedelta(days=1)

        apply_progress_result(
            recommendation,
            latest_value=7.5,
            config={
                "direction": "higher",
                "completion_target": 7.0,
                "improvement_delta": 0.5,
            },
            progress_date=progress_date,
        )

        recommendation.refresh_from_db()

        self.assertEqual(recommendation.status, "completed")
        self.assertEqual(recommendation.completed_at, progress_date)
        self.assertEqual(recommendation.first_improved_at, progress_date)


class MentalWellnessProgressTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="mentalprogress",
            password="TestPass123!",
        )
        self.today = timezone.localdate()

        MentalWellnessAssessment.objects.create(
            user=self.user,
            date=self.today,
            mood=3,
            stress=3,
            anxiety=3,
            energy=3,
            motivation=3,
            concentration=3,
            social_wellbeing=3,
            workload=3,
        )

        self.recommendation = RecommendationHistory.objects.create(
            user=self.user,
            recommendation_date=self.today,
            category="Mental Wellness",
            title="Give Extra Attention to Mental Wellness",
            message="Support mental wellness.",
            priority="high",
            metric="mental_wellness",
            baseline_value=60.0,
            latest_value=60.0,
            status="active",
        )

    def test_later_assessment_completes_recommendation_and_uses_assessment_date(self):
        future_date = self.today + timedelta(days=1)

        MentalWellnessAssessment.objects.create(
            user=self.user,
            date=future_date,
            mood=5,
            stress=1,
            anxiety=1,
            energy=4,
            motivation=4,
            concentration=4,
            social_wellbeing=4,
            workload=1,
        )

        evaluate_mental_wellness_progress(
            self.recommendation
        )

        self.recommendation.refresh_from_db()

        self.assertEqual(
            self.recommendation.latest_value,
            90.0,
        )
        self.assertEqual(
            self.recommendation.status,
            "completed",
        )
        self.assertEqual(
            self.recommendation.completed_at,
            future_date,
        )
        self.assertEqual(
            self.recommendation.first_improved_at,
            future_date,
        )


class DailyCheckInModelTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="checkinuser",
            password="TestPass123!",
        )

    def checkin_values(self):
        return {
            "user": self.user,
            "date": timezone.localdate(),
            "sleep_hours": 7.0,
            "sleep_quality": 8,
            "water_liters": 2.0,
            "exercise_minutes": 30,
            "steps": 8000,
            "mood": 4,
            "stress_level": 2,
            "energy_level": 4,
            "screen_time_hours": 5.0,
            "fruit_vegetable_servings": 5,
            "breakfast_taken": True,
            "lunch_taken": True,
            "dinner_taken": True,
        }

    def test_one_checkin_per_user_per_day(self):
        values = self.checkin_values()
        DailyCheckIn.objects.create(**values)

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                DailyCheckIn.objects.create(**values)


class MentalAssessmentModelTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="assessmentuser",
            password="TestPass123!",
        )

    def assessment_values(self):
        return {
            "user": self.user,
            "date": timezone.localdate(),
            "mood": 4,
            "stress": 2,
            "anxiety": 2,
            "energy": 4,
            "motivation": 4,
            "concentration": 4,
            "social_wellbeing": 4,
            "workload": 2,
        }

    def test_one_assessment_per_user_per_day(self):
        values = self.assessment_values()
        MentalWellnessAssessment.objects.create(**values)

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                MentalWellnessAssessment.objects.create(**values)
