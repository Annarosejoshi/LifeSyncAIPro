from django.db import models
from django.contrib.auth.models import User


class PredictionHistory(models.Model):

    PREDICTION_TYPES = [
    ("diabetes", "Diabetes Risk"),
    ("hypertension", "Hypertension Risk"),
    ("cardiovascular", "Cardiovascular Risk"),
    ("mental_health", "Mental Wellness Risk"),
]

    RISK_LEVELS = [
    ("low", "Low Risk"),
    ("moderate", "Moderate Risk"),
    ("high", "High Risk"),
]
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="prediction_history"
    )

    prediction_type = models.CharField(
        max_length=30,
        choices=PREDICTION_TYPES
    )

    predicted_class = models.PositiveSmallIntegerField()

    probability = models.FloatField()

    risk_level = models.CharField(
        max_length=20,
        choices=RISK_LEVELS
    )

    model_name = models.CharField(
        max_length=150
    )

    model_version = models.CharField(
        max_length=50
    )

    threshold = models.FloatField(
        null=True,
        blank=True
    )

    input_snapshot = models.JSONField(
        default=dict
    )

    explanation = models.JSONField(
        default=list,
        blank=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"{self.user.username} - "
            f"{self.get_prediction_type_display()} - "
            f"{self.risk_level}"
        )