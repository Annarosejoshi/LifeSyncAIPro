import sys

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Load and exercise all four active model pipelines without reading or saving user data."
    requires_system_checks = []

    def handle(self, *args, **options):
        try:
            from ml.services.risk_prediction_service import MODEL_FILES, predict_risk
            inputs = dict(age=35, sex=2, bmi=24, smoking_status=0, sleep_hours=7)
            for name in MODEL_FILES:
                result = predict_risk(name, inputs)
                if not 0 <= result["probability"] <= 1:
                    raise ValueError(f"Invalid model output: {name}")
                self.stdout.write(f"OK: {name} ({result['model_version']})")
        except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as exc:
            raise CommandError(
                f"Model readiness check failed: {exc}\n"
                f'Python: {sys.executable}\n'
                f'Install dependencies with this same interpreter: "{sys.executable}" -m pip install -r requirements.txt\n'
                "Also confirm the four active model artifacts and their metadata are present."
            ) from exc
        self.stdout.write(self.style.SUCCESS("All four active models are ready."))
