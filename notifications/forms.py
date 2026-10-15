from django import forms
from django.utils import timezone

from .models import MedicationRecord


class MedicationRecordForm(forms.ModelForm):
    class Meta:
        model = MedicationRecord
        fields = [
            "medication_name",
            "dosage",
            "scheduled_date",
            "scheduled_time",
            "taken",
            "notes",
        ]
        widgets = {
            "medication_name": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Medication name",
            }),
            "dosage": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "As prescribed",
            }),
            "scheduled_date": forms.DateInput(attrs={
                "class": "form-control",
                "type": "date",
            }),
            "scheduled_time": forms.TimeInput(attrs={
                "class": "form-control",
                "type": "time",
            }),
            "taken": forms.CheckboxInput(attrs={
                "class": "form-check-input",
            }),
            "notes": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Optional note",
                "maxlength": "300",
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.instance.pk:
            self.fields["scheduled_date"].initial = timezone.localdate()
