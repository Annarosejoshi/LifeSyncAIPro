from django import forms
from config.form_inputs import PlainNumericInputsMixin

from .models import (
    DailyCheckIn,
    HabitGoal,
)


PRODUCE_CHOICES = [
    ("apple", "Apple"),
    ("banana", "Banana"),
    ("berries", "Berries"),
    ("citrus", "Orange or other citrus"),
    ("other_fruit", "Other fruit"),
    ("leafy_greens", "Leafy greens"),
    ("broccoli", "Broccoli or cauliflower"),
    ("carrot", "Carrot"),
    ("tomato", "Tomato"),
    ("other_vegetable", "Other vegetable"),
]


class DailyCheckInForm(PlainNumericInputsMixin, forms.ModelForm):
    exercise_hours = forms.DecimalField(
        required=False,
        min_value=0,
        max_value=10,
        decimal_places=2,
        label="Exercise hours",
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "min": "0",
            "max": "10",
            "placeholder": "Example: 0, 1 or 1.5",
        }),
    )
    exercise_extra_minutes = forms.IntegerField(
        required=False,
        min_value=0,
        max_value=59,
        label="Extra minutes",
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "min": "0",
            "max": "59",
            "placeholder": "0 to 59",
        }),
    )

    fruit_vegetable_items = forms.MultipleChoiceField(
        choices=PRODUCE_CHOICES,
        required=False,
        label="Fruits and vegetables eaten today",
        help_text="Select each type you ate. Each selected type counts as one estimated serving.",
        widget=forms.CheckboxSelectMultiple(attrs={
            "class": "produce-options",
        }),
    )

    stress_level = forms.TypedChoiceField(
        choices=[
            (1, "Very relaxed"),
            (2, "A little tense"),
            (3, "Moderately stressed"),
            (4, "Very stressed"),
            (5, "Extremely stressed"),
        ],
        coerce=int,
        label="How stressed did you feel today?",
        help_text="Choose the closest description; there is no right or wrong answer.",
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    energy_level = forms.TypedChoiceField(
        choices=[
            (1, "Very low: usual tasks felt difficult"),
            (2, "Low: less energy than usual"),
            (3, "Moderate: about your usual energy"),
            (4, "High: energetic most of the day"),
            (5, "Very high: much more energy than usual"),
        ],
        coerce=int,
        label="How was your energy today?",
        help_text="Choose the closest description of how you felt.",
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    mood = forms.TypedChoiceField(
        choices=DailyCheckIn.MOOD_CHOICES,
        coerce=int,
        label="How was your mood today?",
        help_text="Choose the description that best matches most of your day.",
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    class Meta:
        model = DailyCheckIn

        # Date is deliberately excluded.
        # The view assigns today's date server-side.
        fields = [
            "sleep_hours",

            "water_liters",


            "screen_time_hours",

            "fruit_vegetable_items",
            "breakfast_taken",
            "lunch_taken",
            "dinner_taken",

            "mood",
            "stress_level",
            "energy_level",

            "notes",
        ]

        widgets = {

            # ==================================================
            # AI PREDICTION INPUTS
            # ==================================================

            "sleep_hours": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "min": "0",
                    "max": "24",
                    "step": "0.1",
                    "placeholder": "Example: 7.5",
                }
            ),

            "screen_time_hours": forms.TextInput(
                attrs={
                    "class": "form-control",
                    "inputmode": "decimal",
                    "placeholder": "Example: 5.0",
                }
            ),

            # ==================================================
            # HEALTH SCORE / COACH
            # ==================================================

            "water_liters": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "min": "0",
                    "max": "15",
                    "step": "0.1",
                    "placeholder": "Example: 2.0",
                }
            ),

            "steps": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "min": "0",
                    "step": "1",
                    "placeholder": "Example: 8000",
                }
            ),

            "breakfast_taken": forms.CheckboxInput(
                attrs={
                    "class": "form-check-input",
                }
            ),

            "lunch_taken": forms.CheckboxInput(
                attrs={
                    "class": "form-check-input",
                }
            ),

            "dinner_taken": forms.CheckboxInput(
                attrs={
                    "class": "form-check-input",
                }
            ),

            "notes": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": "3",
                    "maxlength": "500",
                    "placeholder": (
                        "Optional note about today's "
                        "health or lifestyle..."
                    ),
                }
            ),
        }


    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        if self.instance and self.instance.pk:
            self.initial["fruit_vegetable_items"] = (
                self.instance.fruit_vegetable_items or []
            )

        self.fields["sleep_hours"].label = "😴 Sleep last night (hours)"
        for name in ("breakfast", "lunch", "dinner"):
            self.fields[f"{name}_taken"].label = name.title()

        if self.instance and self.instance.pk:
            exercise_hours, exercise_minutes = divmod(
                self.instance.exercise_minutes,
                60,
            )
            self.initial.setdefault("exercise_hours", exercise_hours)
            self.initial.setdefault(
                "exercise_extra_minutes",
                exercise_minutes,
            )

        self.fields[
            "screen_time_hours"
        ].label = "📱 Screen time (hours)"

        self.fields[
            "water_liters"
        ].label = "💧 Water intake (litres)"


    def clean_sleep_hours(self):

        value = self.cleaned_data[
            "sleep_hours"
        ]

        if value < 0 or value > 24:
            raise forms.ValidationError(
                "Sleep must be between 0 and 24 hours."
            )

        return value


    def clean_screen_time_hours(self):

        value = self.cleaned_data[
            "screen_time_hours"
        ]

        if value < 0 or value > 24:
            raise forms.ValidationError(
                "Screen time must be between 0 and 24 hours."
            )

        return value

    def clean(self):
        cleaned_data = super().clean()
        hours = cleaned_data.get("exercise_hours") or 0
        minutes = cleaned_data.get("exercise_extra_minutes") or 0
        total_minutes = hours * 60 + minutes
        if total_minutes != int(total_minutes):
            raise forms.ValidationError("Exercise duration must total whole minutes; for example, 0.5 hours is 30 minutes.")
        total_minutes = int(total_minutes)

        if total_minutes > 600:
            raise forms.ValidationError(
                "Exercise duration cannot exceed 10 hours per day."
            )

        sleep = cleaned_data.get("sleep_hours")
        if sleep is not None and sleep + total_minutes / 60 > 24:
            raise forms.ValidationError("Sleep and exercise together cannot exceed 24 hours.")
        cleaned_data["exercise_minutes"] = total_minutes
        return cleaned_data


    def save(self, commit=True):
        checkin = super().save(commit=False)
        selected_items = self.cleaned_data.get(
            "fruit_vegetable_items",
            [],
        )
        checkin.fruit_vegetable_items = selected_items
        checkin.fruit_vegetable_servings = len(selected_items)
        checkin.exercise_minutes = self.cleaned_data.get(
            "exercise_minutes",
            0,
        )

        if commit:
            checkin.save()
            self.save_m2m()

        return checkin


    def clean_stress_level(self):

        value = self.cleaned_data[
            "stress_level"
        ]

        if value < 1 or value > 5:
            raise forms.ValidationError(
                "Stress level must be between 1 and 5."
            )

        return value


    def clean_energy_level(self):

        value = self.cleaned_data[
            "energy_level"
        ]

        if value < 1 or value > 5:
            raise forms.ValidationError(
                "Energy level must be between 1 and 5."
            )

        return value


class HabitGoalForm(PlainNumericInputsMixin, forms.ModelForm):

    class Meta:

        model = HabitGoal

        fields = [
            "habit_type",
            "target_value",
        ]

        widgets = {

            "habit_type": forms.Select(
                attrs={
                    "class": "form-select",
                }
            ),

            "target_value": forms.NumberInput(
                attrs={
                    "class": "form-control",
                    "min": "0",
                    "step": "0.1",
                    "placeholder": "Enter your target",
                }
            ),
        }


class BMIForm(PlainNumericInputsMixin, forms.Form):
    height_cm = forms.FloatField(
        min_value=50,
        max_value=250,
        label="Height (cm)",
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "id": "dashboardBmiHeight",
            "min": "50",
            "max": "250",
            "step": "0.1",
        }),
    )
    weight_kg = forms.FloatField(
        min_value=20,
        max_value=500,
        label="Weight (kg)",
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "id": "dashboardBmiWeight",
            "min": "20",
            "max": "500",
            "step": "0.1",
        }),
    )
