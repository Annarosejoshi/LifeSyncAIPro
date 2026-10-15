from io import StringIO
from unittest.mock import patch

from django.core.management import call_command, CommandError
from django.test import SimpleTestCase


class ModelReadinessTests(SimpleTestCase):
    def test_all_active_models_load_without_accessing_user_database(self):
        output = StringIO()
        call_command("check_models", stdout=output)
        self.assertEqual(output.getvalue().count("OK:"), 4)
        self.assertIn("All four active models are ready.", output.getvalue())

    @patch("ml.services.risk_prediction_service.predict_risk", side_effect=ModuleNotFoundError("No module named 'xgboost'"))
    def test_missing_dependency_reports_same_interpreter_install_command(self, predictor):
        with self.assertRaisesMessage(CommandError, "-m pip install -r requirements.txt"):
            call_command("check_models", stdout=StringIO())
