from django import forms


YES_NO = [
    (0, "No"),
    (1, "Yes"),
]


class DiabetesRiskForm(forms.Form):

    high_bp = forms.ChoiceField(
        label="Have you been told you have high blood pressure?",
        choices=YES_NO
    )

    high_chol = forms.ChoiceField(
        label="Have you been told you have high cholesterol?",
        choices=YES_NO
    )

    chol_check = forms.ChoiceField(
        label="Have you had your cholesterol checked in the past 5 years?",
        choices=YES_NO
    )

    bmi = forms.FloatField(
        label="BMI",
        min_value=10,
        max_value=80,
        widget=forms.NumberInput(attrs={
            "class": "form-control",
            "min": "10",
            "max": "80",
            "step": "0.1",
        }),
    )

    smoker = forms.ChoiceField(
        label="Have you smoked at least 100 cigarettes in your lifetime?",
        choices=YES_NO
    )

    stroke = forms.ChoiceField(
        label="Have you ever been told you had a stroke?",
        choices=YES_NO
    )

    heart_disease = forms.ChoiceField(
        label="History of coronary heart disease or heart attack?",
        choices=YES_NO
    )

    physical_activity = forms.ChoiceField(
        label="Physical activity during the past 30 days?",
        choices=YES_NO
    )

    fruits = forms.ChoiceField(
        label="Do you consume fruit one or more times per day?",
        choices=YES_NO
    )

    veggies = forms.ChoiceField(
        label="Do you consume vegetables one or more times per day?",
        choices=YES_NO
    )

    heavy_alcohol = forms.ChoiceField(
        label="Heavy alcohol consumption?",
        choices=YES_NO
    )

    healthcare = forms.ChoiceField(
        label="Do you have healthcare coverage?",
        choices=YES_NO
    )

    no_doctor_cost = forms.ChoiceField(
        label="Was there a time you could not see a doctor because of cost?",
        choices=YES_NO
    )

    general_health = forms.IntegerField(
        label="General Health (1 Excellent - 5 Poor)",
        min_value=1,
        max_value=5
    )

    mental_health_days = forms.IntegerField(
        label="Poor mental health days in the past 30 days",
        min_value=0,
        max_value=30
    )

    physical_health_days = forms.IntegerField(
        label="Poor physical health days in the past 30 days",
        min_value=0,
        max_value=30
    )

    difficulty_walking = forms.ChoiceField(
        label="Serious difficulty walking or climbing stairs?",
        choices=YES_NO
    )

    sex = forms.ChoiceField(
        label="Sex used by the BRFSS source data",
        choices=[
            (0, "Female"),
            (1, "Male"),
        ],
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    age_category = forms.ChoiceField(
        label="Age group",
        choices=[
            (1, "18-24"), (2, "25-29"), (3, "30-34"),
            (4, "35-39"), (5, "40-44"), (6, "45-49"),
            (7, "50-54"), (8, "55-59"), (9, "60-64"),
            (10, "65-69"), (11, "70-74"), (12, "75-79"),
            (13, "80 or older"),
        ],
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    education = forms.ChoiceField(
        label="Highest education completed",
        choices=[
            (1, "No formal education or primary school"),
            (2, "Some primary school"),
            (3, "Some high school"),
            (4, "High school graduate"),
            (5, "Some college or technical school"),
            (6, "College graduate"),
        ],
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    income = forms.ChoiceField(
        label="Annual household income range",
        choices=[
            (1, "Under $10,000"), (2, "$10,000-$15,000"),
            (3, "$15,000-$20,000"), (4, "$20,000-$25,000"),
            (5, "$25,000-$35,000"), (6, "$35,000-$50,000"),
            (7, "$50,000-$75,000"), (8, "$75,000 or more"),
        ],
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            if isinstance(field.widget, forms.Select):
                field.widget.attrs.setdefault("class", "form-select")
            else:
                field.widget.attrs.setdefault("class", "form-control")