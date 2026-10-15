from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator


class DailyCheckIn(models.Model):

    MOOD_CHOICES = [
        (1, "Very Low"),
        (2, "Low"),
        (3, "Neutral"),
        (4, "Good"),
        (5, "Excellent"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="daily_checkins"
    )

    date = models.DateField()

    # Sleep
    sleep_hours = models.FloatField(
        validators=[
            MinValueValidator(0),
            MaxValueValidator(24)
        ]
    )

    sleep_quality = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(1),
            MaxValueValidator(10)
        ]
    )

    # Hydration
    water_liters = models.FloatField(
        validators=[
            MinValueValidator(0),
            MaxValueValidator(15)
        ]
    )

    # Exercise
    exercise_minutes = models.PositiveIntegerField(
        default=0,
        validators=[
            MaxValueValidator(600)
        ]
    )

    sitting_hours = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(24)])
    weekly_activity_minutes = models.PositiveIntegerField(null=True, blank=True, validators=[MaxValueValidator(4200)])

    steps = models.PositiveIntegerField(
        default=0
    )

    # Mental Wellness
    mood = models.PositiveSmallIntegerField(
        choices=MOOD_CHOICES
    )

    stress_level = models.PositiveSmallIntegerField(
        validators=[
            MinValueValidator(1),
            MaxValueValidator(5)
        ]
    )

    energy_level = models.PositiveSmallIntegerField(
        validators=[
            MinValueValidator(1),
            MaxValueValidator(5)
        ]
    )

    # Lifestyle
    screen_time_hours = models.FloatField(
        default=0,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(24)
        ]
    )

    fruit_vegetable_servings = models.PositiveSmallIntegerField(
        default=0,
        validators=[
            MaxValueValidator(20)
        ]
    )

    fruit_vegetable_items = models.JSONField(
        default=list,
        blank=True,
    )

    breakfast_taken = models.BooleanField(default=False)
    lunch_taken = models.BooleanField(default=False)
    dinner_taken = models.BooleanField(default=False)

    notes = models.TextField(
        blank=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    class Meta:
        ordering = ["-date"]

        constraints = [
            models.UniqueConstraint(
                fields=["user", "date"],
                name="unique_daily_checkin_per_user"
            )
        ]

    def __str__(self):
        return f"{self.user.username} - {self.date}"
    
class HabitGoal(models.Model):

    HABIT_TYPES = [
        ("water", "Water Intake"),
        ("sleep", "Sleep"),
        ("exercise", "Exercise"),
        ("steps", "Daily Steps"),
        ("nutrition", "Fruit & Vegetable Intake"),
        ("screen_time", "Screen Time"),
        ("stress", "Stress Management"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="habit_goals"
    )

    habit_type = models.CharField(
        max_length=30,
        choices=HABIT_TYPES
    )

    target_value = models.FloatField(
        validators=[MinValueValidator(0)]
    )

    active = models.BooleanField(
        default=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return (
            f"{self.user.username} - "
            f"{self.get_habit_type_display()}"
        )


class BMIRecord(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="bmi_records",
    )
    date = models.DateField()
    height_cm = models.FloatField(
        validators=[MinValueValidator(50), MaxValueValidator(250)],
    )
    weight_kg = models.FloatField(
        validators=[MinValueValidator(20), MaxValueValidator(500)],
    )
    bmi = models.FloatField(editable=False)
    category = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date", "-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "date"],
                name="unique_bmi_record_per_user_date",
            ),
        ]

    def save(self, *args, **kwargs):
        height_m = self.height_cm / 100
        self.bmi = round(self.weight_kg / (height_m * height_m), 2)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.username} BMI {self.bmi} on {self.date}"