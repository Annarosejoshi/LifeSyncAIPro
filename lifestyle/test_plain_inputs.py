from django import forms
from django.test import SimpleTestCase

from accounts.forms import RegisterForm, UserProfileForm
from lifestyle.forms import BMIForm, DailyCheckInForm, HabitGoalForm
from predictions.forms import DiabetesRiskForm


class PlainInputTests(SimpleTestCase):
    def test_numeric_form_controls_are_plain_text_with_suitable_keyboards(self):
        for form_type in (RegisterForm, UserProfileForm, BMIForm, DailyCheckInForm, HabitGoalForm, DiabetesRiskForm):
            for name, field in form_type().fields.items():
                if isinstance(field, forms.IntegerField) and not isinstance(field.widget, forms.Select):
                    with self.subTest(form=form_type.__name__, field=name):
                        self.assertEqual(field.widget.input_type, 'text')
                        decimal = isinstance(field, (forms.FloatField, forms.DecimalField))
                        self.assertEqual(field.widget.attrs['inputmode'], 'decimal' if decimal else 'numeric')

    def test_bmi_accepts_whole_and_decimal_measurements_but_rejects_invalid_data(self):
        for height, weight in (('170', '65'), ('170.5', '65.75')):
            form = BMIForm(dict(height_cm=height, weight_kg=weight))
            self.assertTrue(form.is_valid(), form.errors)
        self.assertFalse(BMIForm(dict(height_cm='abc', weight_kg='65')).is_valid())
        self.assertFalse(BMIForm(dict(height_cm='10', weight_kg='65')).is_valid())

    def test_fractional_exercise_hours_and_decimal_daily_values_are_saved_in_minutes(self):
        form = DailyCheckInForm(dict(sleep_hours='7.5', water_liters='2.5',
            exercise_hours='1.5', exercise_extra_minutes='15', screen_time_hours='4.25',
            mood='4', stress_level='2', energy_level='4'))
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.save(commit=False).exercise_minutes, 105)
        self.assertEqual(form.cleaned_data['screen_time_hours'], 4.25)
