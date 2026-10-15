
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import (
    MinValueValidator,
    MaxValueValidator,
)


class UserProfile(models.Model):

    # =========================================================
    # CHOICES
    # =========================================================

    GENDER_CHOICES = [
        ("male", "Male"),
        ("female", "Female"),
        ("other", "Other"),
        ("prefer_not_to_say", "Prefer not to say"),
    ]

    YES_NO_UNKNOWN = [
        ("yes", "Yes"),
        ("no", "No"),
        ("unknown", "Unknown"),
    ]

    SMOKING_STATUS_CHOICES = [
        ("never", "Never Smoked"),
        ("former", "Former Smoker"),
        ("current", "Current Smoker"),
        ("unknown", "Prefer Not to Say / Unknown"),
    ]

    ALCOHOL_STATUS_CHOICES = [
        ("yes", "Yes"),
        ("no", "No"),
        ("unknown", "Prefer Not to Say / Unknown"),
    ]

    # =========================================================
    # USER
    # =========================================================

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name="profile",
    )

    # =========================================================
    # PERSONAL INFORMATION
    # =========================================================

    phone = models.CharField(
        max_length=15,
        blank=True,
    )

    date_of_birth = models.DateField(
        null=True,
        blank=True,
    )

    gender = models.CharField(
        max_length=20,
        choices=GENDER_CHOICES,
        blank=True,
    )

    occupation = models.CharField(
        max_length=150,
        blank=True,
    )

    # =========================================================
    # BODY MEASUREMENTS
    # =========================================================

    height_cm = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(50),
            MaxValueValidator(250),
        ],
    )

    weight_kg = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(20),
            MaxValueValidator(500),
        ],
    )

    waist_circumference = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(30),
            MaxValueValidator(250),
        ],
        help_text="Waist circumference in centimetres.",
    )

    # =========================================================
    # BLOOD PRESSURE
    # =========================================================

    systolic_bp = models.PositiveIntegerField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(50),
            MaxValueValidator(300),
        ],
    )

    diastolic_bp = models.PositiveIntegerField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(30),
            MaxValueValidator(200),
        ],
    )

    # =========================================================
    # CHOLESTEROL
    # =========================================================

    total_cholesterol = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(50),
            MaxValueValidator(500),
        ],
        help_text="Total cholesterol in mg/dL.",
    )

    hdl_cholesterol = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(10),
            MaxValueValidator(200),
        ],
        help_text="HDL cholesterol in mg/dL.",
    )

    # =========================================================
    # HEALTH HISTORY
    # =========================================================

    cholesterol_history = models.CharField(
        max_length=10,
        choices=YES_NO_UNKNOWN,
        default="unknown",
    )

    diabetes_history = models.CharField(
        max_length=10,
        choices=YES_NO_UNKNOWN,
        default="unknown",
    )

    family_diabetes_history = models.CharField(
        max_length=10,
        choices=YES_NO_UNKNOWN,
        default="unknown",
    )

    heart_disease_history = models.CharField(
        max_length=10,
        choices=YES_NO_UNKNOWN,
        default="unknown",
    )

    # =========================================================
    # SMOKING
    # =========================================================

    smoker = models.BooleanField(
        default=False,
    )

    smoking_status = models.CharField(
        max_length=20,
        choices=SMOKING_STATUS_CHOICES,
        default="unknown",
    )

    # =========================================================
    # ALCOHOL
    # =========================================================

    alcohol_past_year = models.CharField(
        max_length=10,
        choices=ALCOHOL_STATUS_CHOICES,
        default="unknown",
    )

    alcohol_frequency = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(365),
        ],
        help_text="Approximate number of drinking days per year.",
    )

    average_drinks = models.FloatField(
        null=True,
        blank=True,
        validators=[
            MinValueValidator(0),
            MaxValueValidator(50),
        ],
        help_text="Average number of drinks on a drinking day.",
    )

    # =========================================================
    # TIMESTAMPS
    # =========================================================

    created_at = models.DateTimeField(
        auto_now_add=True,
    )

    updated_at = models.DateTimeField(
        auto_now=True,
    )

    # =========================================================
    # BMI
    # =========================================================

    @property
    def bmi(self):
        """Calculate BMI using weight in kg and height in metres."""

        if self.height_cm is None or self.weight_kg is None:
            return None

        height_m = self.height_cm / 100

        if height_m <= 0:
            return None

        return round(
            self.weight_kg / (height_m ** 2),
            2,
        )

    @property
    def bmi_category(self):

        bmi = self.bmi

        if bmi is None:
            return "Not Available"

        if bmi < 18.5:
            return "Underweight"

        if bmi < 25:
            return "Normal"

        if bmi < 30:
            return "Overweight"

        return "Obesity"

    # =========================================================
    # PROFILE COMPLETENESS
    # =========================================================

    @property
    def health_profile_completeness(self):
        """Percentage of important health-profile fields available."""

        fields = [
            self.date_of_birth,
            self.gender,
            self.height_cm,
            self.weight_kg,
            self.waist_circumference,
            self.systolic_bp,
            self.diastolic_bp,
            self.smoking_status,
            self.total_cholesterol,
            self.hdl_cholesterol,
        ]

        available = sum(
            1
            for value in fields
            if value not in [None, "", "unknown"]
        )

        return round(
            (available / len(fields)) * 100,
            1,
        )

    # =========================================================
    # STRING REPRESENTATION
    # =========================================================

    def __str__(self):
        return f"{self.user.username} Profile"
