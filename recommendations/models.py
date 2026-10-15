from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models


class RecommendationHistory(models.Model):

    PRIORITY_CHOICES = [
        ("high", "High"),
        ("medium", "Medium"),
        ("low", "Low"),
    ]

    STATUS_CHOICES = [
        ("active", "Active"),
        ("improving", "Improving"),
        ("completed", "Completed"),
        ("dismissed", "Dismissed"),
    ]

    METRIC_CHOICES = [
        ("sleep", "Sleep"),
        ("water", "Water Intake"),
        ("exercise", "Exercise"),
        ("steps", "Daily Steps"),
        ("nutrition", "Fruit & Vegetable Intake"),
        ("screen_time", "Screen Time"),
        ("stress", "Stress"),
        ("mood", "Mood"),
        ("energy", "Energy"),
        ("mental_wellness", "Mental Wellness"),
        ("health_score", "AI Health Score"),
        ("general", "General Wellness"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="recommendation_history",
    )

    recommendation_date = models.DateField()

    category = models.CharField(
        max_length=60,
    )

    title = models.CharField(
        max_length=200,
    )

    message = models.TextField()

    priority = models.CharField(
        max_length=10,
        choices=PRIORITY_CHOICES,
        default="medium",
    )

    metric = models.CharField(
        max_length=30,
        choices=METRIC_CHOICES,
        default="general",
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="active",
    )

    baseline_value = models.FloatField(
        null=True,
        blank=True,
    )

    latest_value = models.FloatField(
        null=True,
        blank=True,
    )

    improvement_percentage = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(-10000),
            MaxValueValidator(10000),
        ],
    )

    first_improved_at = models.DateField(
        null=True,
        blank=True,
    )

    completed_at = models.DateField(
        null=True,
        blank=True,
    )

    source = models.CharField(
        max_length=30,
        default="recommendation_engine_v2",
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    updated_at = models.DateTimeField(
        auto_now=True,
    )

    class Meta:
        ordering = [
            "-recommendation_date",
            "-created_at",
        ]

        indexes = [
            models.Index(
                fields=[
                    "user",
                    "recommendation_date",
                ],
            ),
            models.Index(
                fields=[
                    "user",
                    "status",
                ],
            ),
            models.Index(
                fields=[
                    "user",
                    "metric",
                ],
            ),
        ]

        constraints = [
            models.UniqueConstraint(
                fields=[
                    "user",
                    "recommendation_date",
                    "title",
                ],
                name="unique_daily_recommendation_title",
            ),
        ]

    def __str__(self):
        return (
            f"{self.user.username} - "
            f"{self.title} - "
            f"{self.status}"
        )
