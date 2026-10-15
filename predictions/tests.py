from datetime import date
from unittest.mock import MagicMock, patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from accounts.models import UserProfile
from lifestyle.models import DailyCheckIn
from predictions.models import PredictionHistory

from ml.services.risk_history_service import (
    build_explanation,
    normalize_risk_level,
    save_all_health_risk_history,
    save_single_risk_history,
)
from ml.services.risk_input_mapper import (
    build_cardiovascular_input_from_user,
    build_common_risk_input,
    build_diabetes_input_from_user,
    build_hypertension_input_from_user,
    build_mental_health_input_from_user,
)
from ml.services.risk_prediction_service import (
    CARDIOVASCULAR_CORE_FEATURES,
    DIABETES_CORE_FEATURES,
    HYPERTENSION_CORE_FEATURES,
    MENTAL_HEALTH_CORE_FEATURES,
    check_prediction_readiness,
    get_all_health_risks,
    get_risk_level,
    predict_cardiovascular_risk,
    predict_brfss_diabetes_risk,
    predict_diabetes_risk,
    predict_hypertension_risk,
    predict_mental_health_risk,
    predict_risk,
)


class CompleteUserMixin:
    """
    Creates a deterministic user with enough profile/check-in data
    for all four LifeSync risk-input builders.
    """

    def create_complete_user(self, username="mluser"):
        user = User.objects.create_user(
            username=username,
            password="TestPass123!",
        )

        profile, _ = UserProfile.objects.get_or_create(
            user=user
        )

        profile.date_of_birth = date(2000, 1, 1)
        profile.gender = "male"
        profile.height_cm = 175
        profile.weight_kg = 80
        profile.waist_circumference = 90
        profile.systolic_bp = 120
        profile.diastolic_bp = 78
        profile.total_cholesterol = 180
        profile.hdl_cholesterol = 50
        profile.smoking_status = "never"
        profile.alcohol_past_year = "no"
        profile.save()

        DailyCheckIn.objects.create(
            user=user,
            date=timezone.localdate(),
            sleep_hours=6.5,
            sleep_quality=7,
            water_liters=2.0,
            exercise_minutes=30,
            weekly_activity_minutes=210,
            sitting_hours=5,
            steps=7000,
            mood=4,
            stress_level=2,
            energy_level=4,
            screen_time_hours=5.0,
            fruit_vegetable_servings=4,
            breakfast_taken=True,
            lunch_taken=True,
            dinner_taken=True,
        )

        return user


class RiskLevelTests(TestCase):

    def test_probability_risk_level_boundaries(self):
        cases = [
            (0.00, "Low"),
            (0.24, "Low"),
            (0.25, "Moderate"),
            (0.49, "Moderate"),
            (0.50, "High"),
            (0.74, "High"),
            (0.75, "Very High"),
            (1.00, "Very High"),
        ]

        for probability, expected in cases:
            with self.subTest(probability=probability):
                self.assertEqual(
                    get_risk_level(probability),
                    expected,
                )

    def test_history_risk_level_normalizer(self):
        self.assertEqual(normalize_risk_level("Low"), "low")
        self.assertEqual(normalize_risk_level("Moderate"), "moderate")
        self.assertEqual(normalize_risk_level("High"), "high")
        self.assertEqual(normalize_risk_level("Very High"), "high")
        self.assertEqual(normalize_risk_level("Elevated"), "high")
        self.assertEqual(normalize_risk_level(None), "low")


class PredictionReadinessTests(TestCase):

    def test_complete_diabetes_input_is_100_percent_ready(self):
        data = {
            feature: 1
            for feature in DIABETES_CORE_FEATURES
        }

        result = check_prediction_readiness(
            data,
            DIABETES_CORE_FEATURES,
        )

        self.assertEqual(result["available_count"], len(DIABETES_CORE_FEATURES))
        self.assertEqual(result["missing_count"], 0)
        self.assertEqual(
            result["completeness_percentage"],
            100.0,
        )

    def test_missing_values_reduce_readiness(self):
        data = {
            "age": 26,
            "sex": 1,
            "bmi": None,
            "waist_circumference": "",
        }

        result = check_prediction_readiness(
            data,
            DIABETES_CORE_FEATURES,
        )

        self.assertEqual(result["available_count"], 2)
        self.assertIn("bmi", result["missing_features"])
        self.assertIn(
            "sleep_hours",
            result["missing_features"],
        )


class GenericPredictionTests(TestCase):

    @patch("ml.services.risk_prediction_service.load_metadata")
    @patch("ml.services.risk_prediction_service.load_model")
    def test_predict_risk_uses_probability_threshold_and_feature_order(
        self,
        mock_load_model,
        mock_load_metadata,
    ):
        fake_model = MagicMock()
        fake_model.predict_proba.return_value = [
            [0.18, 0.82]
        ]

        mock_load_model.return_value = fake_model
        mock_load_metadata.return_value = {
            "features": [
                "age",
                "bmi",
                "sleep_hours",
            ],
            "selected_threshold": 0.45,
        }

        result = predict_risk(
            "diabetes",
            {
                "sleep_hours": 7.0,
                "bmi": 26.0,
                "age": 26,
                "ignored_extra_feature": 999,
            },
        )

        self.assertAlmostEqual(
            result["probability"],
            0.82,
        )
        self.assertEqual(
            result["percentage"],
            82.0,
        )
        self.assertEqual(
            result["threshold"],
            0.45,
        )
        self.assertEqual(
            result["prediction"],
            1,
        )
        self.assertEqual(
            result["risk_level"],
            "Very High",
        )

        input_df = (
            fake_model.predict_proba
            .call_args.args[0]
        )

        self.assertEqual(
            list(input_df.columns),
            [
                "age",
                "bmi",
                "sleep_hours",
            ],
        )


class BRFSSPredictionInputTests(TestCase):
    @patch("ml.services.risk_prediction_service.load_brfss_diabetes_bundle")
    def test_explicit_brfss_answers_fill_all_model_features(self, mock_bundle):
        features = [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
            "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
            "Education", "Income",
        ]
        model = MagicMock()
        model.classes_ = [0, 1, 2]
        model.predict_proba.return_value = [[0.2, 0.3, 0.5]]
        mock_bundle.return_value = {"model": model, "features": features}
        input_data = {feature: 1 for feature in features}
        input_data.update({
            "age": 55,
            "sex": 1,
            "bmi": 27.5,
            "smoking_status": 1,
            "BMI": 27.5,
        })

        result = predict_brfss_diabetes_risk(input_data)

        self.assertEqual(result["model_input_readiness"]["available_count"], 21)
        self.assertEqual(result["model_input_readiness"]["missing_count"], 0)
        input_frame = model.predict_proba.call_args.args[0]
        self.assertEqual(list(input_frame.columns), features)
        self.assertEqual(input_frame.loc[0, "BMI"], 27.5)


class FourModelWrapperTests(TestCase):

    def complete_data_for(self, features):
        return {
            feature: 1
            for feature in features
        }

    @patch("ml.services.risk_prediction_service.predict_risk")
    def test_all_four_wrappers_return_available_predictions(
        self,
        mock_predict_risk,
    ):
        mock_predict_risk.return_value = {
            "model": "placeholder",
            "probability": 0.80,
            "percentage": 80.0,
            "threshold": 0.50,
            "prediction": 1,
            "risk_level": "Very High",
        }

        cases = [
            (
                "diabetes",
                predict_diabetes_risk,
                DIABETES_CORE_FEATURES,
                "Elevated Diabetes Risk",
            ),
            (
                "hypertension",
                predict_hypertension_risk,
                HYPERTENSION_CORE_FEATURES,
                "Elevated Hypertension Risk",
            ),
            (
                "cardiovascular",
                predict_cardiovascular_risk,
                CARDIOVASCULAR_CORE_FEATURES,
                "Elevated Cardiovascular Risk",
            ),
            (
                "mental_health",
                predict_mental_health_risk,
                MENTAL_HEALTH_CORE_FEATURES,
                "Elevated Mental Wellness Risk",
            ),
        ]

        for model_name, predictor, features, expected_label in cases:
            with self.subTest(model=model_name):
                result = predictor(
                    self.complete_data_for(features)
                )

                self.assertTrue(
                    result["prediction_available"]
                )
                if expected_label is None:
                    self.assertIn(
                        result["label"],
                        {
                            "No Diabetes Class Predicted",
                            "Prediabetes Class Predicted",
                            "Diabetes Class Predicted",
                        },
                    )
                    self.assertTrue(result["class_probabilities"])
                else:
                    self.assertEqual(
                        result["label"],
                        expected_label,
                    )
                self.assertEqual(
                    result["readiness"][
                        "completeness_percentage"
                    ],
                    100.0,
                )

    @patch("ml.services.risk_prediction_service.predict_risk")
    def test_all_four_wrappers_refuse_insufficient_data(
        self,
        mock_predict_risk,
    ):
        cases = [
            predict_diabetes_risk,
            predict_hypertension_risk,
            predict_cardiovascular_risk,
            predict_mental_health_risk,
        ]

        for predictor in cases:
            with self.subTest(
                predictor=predictor.__name__
            ):
                result = predictor({})

                self.assertFalse(
                    result["prediction_available"]
                )
                self.assertIn(
                    "readiness",
                    result,
                )

        mock_predict_risk.assert_not_called()


class RiskInputMapperTests(
    CompleteUserMixin,
    TestCase,
):

    def setUp(self):
        self.user = self.create_complete_user()

    def test_common_mapper_builds_expected_lifestyle_features(self):
        data = build_common_risk_input(
            self.user
        )

        expected_age = (
            date.today().year
            - 2000
            - (
                (date.today().month, date.today().day)
                < (1, 1)
            )
        )

        self.assertEqual(
            data["age"],
            expected_age,
        )
        self.assertEqual(data["sex"], 1)
        self.assertAlmostEqual(
            data["bmi"],
            26.12,
            places=2,
        )
        self.assertEqual(
            data["waist_circumference"],
            90.0,
        )
        self.assertEqual(
            data["sleep_hours"],
            6.5,
        )
        self.assertEqual(
            data["short_sleep_flag"],
            1,
        )
        self.assertEqual(
            data["total_activity_minutes_week"],
            210.0,
        )
        self.assertEqual(
            data["sedentary_minutes_day"],
            300.0,
        )
        self.assertEqual(
            data["smoking_status"],
            0,
        )

    def test_four_input_builders_return_expected_core_features(self):
        diabetes = build_diabetes_input_from_user(
            self.user
        )
        hypertension = build_hypertension_input_from_user(
            self.user
        )
        cardiovascular = build_cardiovascular_input_from_user(
            self.user
        )
        mental = build_mental_health_input_from_user(
            self.user
        )

        for feature in DIABETES_CORE_FEATURES:
            self.assertIn(feature, diabetes)
            self.assertIsNotNone(
                diabetes[feature]
            )

        for feature in HYPERTENSION_CORE_FEATURES:
            self.assertIn(feature, hypertension)
            self.assertIsNotNone(
                hypertension[feature]
            )

        for feature in CARDIOVASCULAR_CORE_FEATURES:
            self.assertIn(feature, cardiovascular)
            self.assertIsNotNone(
                cardiovascular[feature]
            )

        for feature in MENTAL_HEALTH_CORE_FEATURES:
            self.assertIn(feature, mental)
            self.assertIsNotNone(
                mental[feature]
            )

        self.assertEqual(
            cardiovascular[
                "avg_systolic_bp"
            ],
            120.0,
        )
        self.assertEqual(
            cardiovascular[
                "avg_diastolic_bp"
            ],
            78.0,
        )


class GetAllHealthRisksTests(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="allriskuser",
            password="TestPass123!",
        )

    @patch(
        "ml.services.risk_prediction_service."
        "predict_mental_health_risk"
    )
    @patch(
        "ml.services.risk_prediction_service."
        "predict_cardiovascular_risk"
    )
    @patch(
        "ml.services.risk_prediction_service."
        "predict_hypertension_risk"
    )
    @patch(
        "ml.services.risk_prediction_service."
        "predict_diabetes_risk"
    )
    @patch(
        "ml.services.risk_input_mapper."
        "build_mental_health_input_from_user"
    )
    @patch(
        "ml.services.risk_input_mapper."
        "build_cardiovascular_input_from_user"
    )
    @patch(
        "ml.services.risk_input_mapper."
        "build_hypertension_input_from_user"
    )
    @patch(
        "ml.services.risk_input_mapper."
        "build_diabetes_input_from_user"
    )
    def test_get_all_health_risks_returns_all_four_keys(
        self,
        diabetes_builder,
        hypertension_builder,
        cardiovascular_builder,
        mental_builder,
        diabetes_predictor,
        hypertension_predictor,
        cardiovascular_predictor,
        mental_predictor,
    ):
        diabetes_builder.return_value = {"age": 26}
        hypertension_builder.return_value = {"age": 26}
        cardiovascular_builder.return_value = {"age": 26}
        mental_builder.return_value = {"age": 26}

        diabetes_predictor.return_value = {
            "prediction_available": True,
            "model": "diabetes",
        }
        hypertension_predictor.return_value = {
            "prediction_available": True,
            "model": "hypertension",
        }
        cardiovascular_predictor.return_value = {
            "prediction_available": True,
            "model": "cardiovascular",
        }
        mental_predictor.return_value = {
            "prediction_available": True,
            "model": "mental_health",
        }

        result = get_all_health_risks(
            self.user
        )

        self.assertEqual(
            set(result.keys()),
            {
                "diabetes",
                "hypertension",
                "cardiovascular",
                "mental_health",
            },
        )


class PredictionHistoryServiceTests(
    CompleteUserMixin,
    TestCase,
):

    def setUp(self):
        self.user = self.create_complete_user(
            username="historyuser"
        )

    def available_result(
        self,
        probability=0.30,
        prediction=0,
        risk_level="Moderate",
    ):
        return {
            "prediction_available": True,
            "probability": probability,
            "percentage": round(
                probability * 100,
                2,
            ),
            "prediction": prediction,
            "risk_level": risk_level,
            "threshold": 0.45,
            "label": "Screening result",
            "readiness": {
                "completeness_percentage": 100.0,
                "missing_features": [],
            },
        }

    def test_unavailable_result_is_not_saved(self):
        saved = save_single_risk_history(
            self.user,
            "diabetes",
            {
                "prediction_available": False,
                "reason": "Not enough data",
            },
        )

        self.assertIsNone(saved)
        self.assertEqual(
            PredictionHistory.objects.count(),
            0,
        )

    def test_same_model_same_day_updates_instead_of_duplicating(self):
        first = save_single_risk_history(
            self.user,
            "diabetes",
            self.available_result(
                probability=0.20,
                prediction=0,
                risk_level="Low",
            ),
        )

        second = save_single_risk_history(
            self.user,
            "diabetes",
            self.available_result(
                probability=0.80,
                prediction=1,
                risk_level="Very High",
            ),
        )

        self.assertEqual(first.pk, second.pk)
        self.assertEqual(
            PredictionHistory.objects.filter(
                user=self.user,
                prediction_type="diabetes",
            ).count(),
            1,
        )


class DiabetesAssessmentViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="prediction_form_user",
            password="TestPass123!",
        )
        self.valid_data = {
            "high_bp": "0",
            "high_chol": "0",
            "chol_check": "1",
            "bmi": "26.5",
            "smoker": "0",
            "stroke": "0",
            "heart_disease": "0",
            "physical_activity": "1",
            "fruits": "1",
            "veggies": "1",
            "heavy_alcohol": "0",
            "healthcare": "1",
            "no_doctor_cost": "0",
            "general_health": "2",
            "mental_health_days": "0",
            "physical_health_days": "0",
            "difficulty_walking": "0",
            "sex": "1",
            "age_category": "8",
            "education": "6",
            "income": "7",
        }

    @staticmethod
    def available_result():
        return {
            "prediction_available": True,
            "probability": 0.80,
            "percentage": 80.0,
            "prediction": 1,
            "risk_level": "High",
            "threshold": 0.45,
            "label": "Screening result",
            "readiness": {
                "completeness_percentage": 100.0,
                "missing_features": [],
            },
        }

    @patch("ml.services.risk_prediction_service.load_brfss_diabetes_bundle")
    def test_valid_assessment_runs_model_and_saves_user_snapshot(self, mock_bundle):
        model = MagicMock()
        model.classes_ = [0, 1, 2]
        model.predict_proba.return_value = [[.2, .3, .5]]
        mock_bundle.return_value = {"model": model, "features": [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
        ]}
        self.client.force_login(self.user)

        response = self.client.post(
            reverse("predictions:diabetes_risk"),
            self.valid_data,
        )

        self.assertEqual(response.status_code, 200)
        result = response.context["assessment_result"]
        self.assertTrue(result["prediction_available"])
        self.assertEqual(len(result["class_probabilities"]), 3)
        self.assertEqual(
            result["model_input_readiness"]["completeness_percentage"],
            100.0,
        )

        history = PredictionHistory.objects.get(user=self.user)
        self.assertEqual(history.prediction_type, "diabetes")
        self.assertEqual(history.input_snapshot["HighBP"], 0)
        self.assertEqual(history.input_snapshot["Age"], 8)
        self.assertEqual(history.input_snapshot["BMI"], 26.5)

    @patch("ml.services.risk_prediction_service.load_brfss_diabetes_bundle", side_effect=FileNotFoundError)
    def test_optional_missing_model_gives_current_screening_link(self, mock_bundle):
        self.client.force_login(self.user)
        response = self.client.post(reverse("predictions:diabetes_risk"), self.valid_data)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context["assessment_result"]["prediction_available"])
        self.assertContains(response, "Use Health Predictions for the current diabetes screening.")
        self.assertFalse(PredictionHistory.objects.filter(user=self.user).exists())

    def test_invalid_assessment_does_not_create_prediction(self):
        self.client.force_login(self.user)
        data = dict(self.valid_data)
        data["age_category"] = "14"

        response = self.client.post(
            reverse("predictions:diabetes_risk"),
            data,
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["form"].errors)
        self.assertFalse(PredictionHistory.objects.filter(user=self.user).exists())

    def test_save_all_health_risk_history_saves_four_models(self):
        result = self.available_result()

        saved = save_all_health_risk_history(
            self.user,
            {
                "diabetes": result.copy(),
                "hypertension": result.copy(),
                "cardiovascular": result.copy(),
                "mental_health": result.copy(),
            },
        )

        self.assertEqual(
            set(saved.keys()),
            {
                "diabetes",
                "hypertension",
                "cardiovascular",
                "mental_health",
            },
        )

        self.assertEqual(
            PredictionHistory.objects.filter(
                user=self.user
            ).count(),
            4,
        )

    def test_explanation_contains_label_and_completeness(self):
        explanation = build_explanation(
            self.available_result()
        )

        self.assertIn(
            "Screening result",
            explanation,
        )
        self.assertIn(
            "Data completeness: 100.0%",
            explanation,
        )
