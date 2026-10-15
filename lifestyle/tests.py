from types import SimpleNamespace

from django.test import SimpleTestCase

from lifestyle.health_score import (
    RISK_MODEL_MAX_PENALTIES,
    calculate_ai_health_score,
    calculate_mental_assessment_score,
    calculate_ml_risk_modifier,
    calculate_single_risk_penalty,
)


class AIHealthScoreV2Tests(SimpleTestCase):

    def make_checkin(self, **overrides):
        values = {
            "sleep_hours": 8.0,
            "sleep_quality": 9,
            "water_liters": 2.5,
            "exercise_minutes": 35,
            "steps": 9000,
            "mood": 4,
            "stress_level": 2,
            "energy_level": 4,
            "screen_time_hours": 4.0,
            "fruit_vegetable_servings": 5,
            "breakfast_taken": True,
            "lunch_taken": True,
            "dinner_taken": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def make_mental_assessment(self, **overrides):
        values = {
            "mood": 4,
            "stress": 2,
            "anxiety": 2,
            "energy": 4,
            "motivation": 4,
            "concentration": 4,
            "social_wellbeing": 4,
            "workload": 2,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def make_risk(
        self,
        probability,
        prediction,
        threshold=0.50,
        prediction_available=True,
    ):
        return {
            "prediction_available": prediction_available,
            "probability": probability,
            "prediction": prediction,
            "threshold": threshold,
        }

    def test_no_checkin_returns_no_score(self):
        result = calculate_ai_health_score(
            checkin=None
        )

        self.assertFalse(
            result["available"]
        )
        self.assertIsNone(
            result["overall"]
        )
        self.assertEqual(
            result["category"],
            "No Data",
        )
        self.assertEqual(
            result["risk_penalty"],
            0,
        )

    def test_unknown_sleep_quality_does_not_reduce_duration_score(self):
        checkin = self.make_checkin(
            sleep_hours=8,
            sleep_quality=None,
        )

        result = calculate_ai_health_score(
            checkin=checkin,
            health_risks={},
        )

        self.assertEqual(result["components"]["sleep"], 100)

    def test_strong_lifestyle_scores_higher_than_poor_lifestyle(self):
        strong = self.make_checkin()

        poor = self.make_checkin(
            sleep_hours=4.0,
            sleep_quality=3,
            water_liters=0.5,
            exercise_minutes=0,
            steps=500,
            mood=2,
            stress_level=5,
            energy_level=2,
            screen_time_hours=12.0,
            fruit_vegetable_servings=0,
            breakfast_taken=False,
            lunch_taken=False,
            dinner_taken=False,
        )

        strong_result = calculate_ai_health_score(
            checkin=strong,
            health_risks={},
        )

        poor_result = calculate_ai_health_score(
            checkin=poor,
            health_risks={},
        )

        self.assertGreater(
            strong_result["overall"],
            poor_result["overall"],
        )

    def test_dedicated_mental_assessment_is_used_when_available(self):
        checkin = self.make_checkin(
            mood=2,
            stress_level=5,
            energy_level=2,
        )

        assessment = self.make_mental_assessment()

        result = calculate_ai_health_score(
            checkin=checkin,
            mental_assessment=assessment,
            health_risks={},
        )

        self.assertEqual(
            result["mental_source"],
            "mental_wellness_assessment",
        )

        self.assertEqual(
            result["components"]["mental_wellness"],
            calculate_mental_assessment_score(
                assessment
            ),
        )

    def test_daily_checkin_mental_values_are_fallback(self):
        checkin = self.make_checkin()

        result = calculate_ai_health_score(
            checkin=checkin,
            mental_assessment=None,
            health_risks={},
        )

        self.assertEqual(
            result["mental_source"],
            "daily_checkin",
        )

    def test_non_elevated_ml_results_apply_no_penalty(self):
        risks = {
            "diabetes": self.make_risk(
                probability=0.49,
                prediction=0,
                threshold=0.45,
            ),
            "hypertension": self.make_risk(
                probability=0.40,
                prediction=0,
                threshold=0.55,
            ),
            "cardiovascular": self.make_risk(
                probability=0.37,
                prediction=0,
                threshold=0.38,
            ),
            "mental_health": self.make_risk(
                probability=0.37,
                prediction=0,
                threshold=0.38,
            ),
        }

        modifier = calculate_ml_risk_modifier(
            risks
        )

        self.assertEqual(
            modifier["total_penalty"],
            0.0,
        )

        for penalty in modifier["penalties"].values():
            self.assertEqual(
                penalty,
                0.0,
            )

    def test_elevated_result_applies_bounded_penalty(self):
        max_penalty = (
            RISK_MODEL_MAX_PENALTIES[
                "mental_health"
            ]
        )

        penalty = calculate_single_risk_penalty(
            self.make_risk(
                probability=0.80,
                prediction=1,
                threshold=0.38,
            ),
            max_penalty=max_penalty,
        )

        self.assertGreater(
            penalty,
            0.0,
        )

        self.assertLessEqual(
            penalty,
            max_penalty,
        )

    def test_unavailable_prediction_applies_no_penalty(self):
        penalty = calculate_single_risk_penalty(
            self.make_risk(
                probability=0.99,
                prediction=1,
                threshold=0.50,
                prediction_available=False,
            ),
            max_penalty=5.0,
        )

        self.assertEqual(
            penalty,
            0.0,
        )

    def test_total_ml_penalty_never_exceeds_15(self):
        risks = {
            model_name: self.make_risk(
                probability=1.0,
                prediction=1,
                threshold=0.50,
            )
            for model_name
            in RISK_MODEL_MAX_PENALTIES
        }

        modifier = calculate_ml_risk_modifier(
            risks
        )

        self.assertLessEqual(
            modifier["total_penalty"],
            15.0,
        )

        self.assertEqual(
            modifier["max_possible_penalty"],
            15.0,
        )

    def test_final_score_stays_between_zero_and_100(self):
        excellent_checkin = self.make_checkin()

        maximum_risks = {
            model_name: self.make_risk(
                probability=1.0,
                prediction=1,
                threshold=0.50,
            )
            for model_name
            in RISK_MODEL_MAX_PENALTIES
        }

        result = calculate_ai_health_score(
            checkin=excellent_checkin,
            mental_assessment=self.make_mental_assessment(),
            health_risks=maximum_risks,
        )

        self.assertGreaterEqual(
            result["overall"],
            0.0,
        )

        self.assertLessEqual(
            result["overall"],
            100.0,
        )

    def test_historical_score_with_empty_risks_has_zero_penalty(self):
        result = calculate_ai_health_score(
            checkin=self.make_checkin(),
            mental_assessment=self.make_mental_assessment(),
            health_risks={},
        )

        self.assertTrue(
            result["available"]
        )

        self.assertEqual(
            result["risk_penalty"],
            0.0,
        )

        self.assertEqual(
            result["base_score"],
            result["overall"],
        )

    def test_elevated_ml_risk_reduces_overall_score(self):
        checkin = self.make_checkin()
        assessment = self.make_mental_assessment()

        no_risk = calculate_ai_health_score(
            checkin=checkin,
            mental_assessment=assessment,
            health_risks={},
        )

        elevated = calculate_ai_health_score(
            checkin=checkin,
            mental_assessment=assessment,
            health_risks={
                "mental_health": self.make_risk(
                    probability=0.90,
                    prediction=1,
                    threshold=0.38,
                ),
            },
        )

        self.assertGreater(
            elevated["risk_penalty"],
            0.0,
        )

        self.assertLess(
            elevated["overall"],
            no_risk["overall"],
        )
