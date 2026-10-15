from django import forms

from .models import MentalWellnessAssessment


RATING_CHOICES = [
    (1, "1"),
    (2, "2"),
    (3, "3"),
    (4, "4"),
    (5, "5"),
]


class MentalWellnessAssessmentForm(
    forms.ModelForm
):

    class Meta:

        model = MentalWellnessAssessment

        # Date is intentionally excluded.
        # The view controls the assessment date.
        fields = [
            "mood",
            "stress",
            "anxiety",
            "energy",
            "motivation",
            "concentration",
            "social_wellbeing",
            "workload",
            "notes",
        ]

        widgets = {

            "mood": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "stress": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "anxiety": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "energy": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "motivation": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "concentration": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "social_wellbeing": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "workload": forms.RadioSelect(
                choices=RATING_CHOICES
            ),

            "notes": forms.Textarea(
                attrs={
                    "class": "form-control",
                    "rows": "4",
                    "maxlength": "500",
                    "placeholder": (
                        "Optional notes about how "
                        "you are feeling today..."
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

        self.fields[
            "mood"
        ].label = "Mood"

        self.fields[
            "stress"
        ].label = "Stress"

        self.fields[
            "anxiety"
        ].label = "Anxiety"

        self.fields[
            "energy"
        ].label = "Energy"

        self.fields[
            "motivation"
        ].label = "Motivation"

        self.fields[
            "concentration"
        ].label = "Concentration"

        self.fields[
            "social_wellbeing"
        ].label = "Social Wellbeing"

        self.fields[
            "workload"
        ].label = "Workload"


    def clean(self):

        cleaned = super().clean()

        rating_fields = [
            "mood",
            "stress",
            "anxiety",
            "energy",
            "motivation",
            "concentration",
            "social_wellbeing",
            "workload",
        ]

        for field_name in rating_fields:

            value = cleaned.get(
                field_name
            )

            if value is None:
                continue

            if value < 1 or value > 5:

                self.add_error(
                    field_name,
                    (
                        "Please select a value "
                        "between 1 and 5."
                    ),
                )

        return cleaned