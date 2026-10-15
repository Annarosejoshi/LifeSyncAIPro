from django.test import TestCase
from django.contrib.auth.models import User
from django.utils import timezone
from lifestyle.forms import DailyCheckInForm
from lifestyle.health_score import calculate_exercise_score
from lifestyle.models import DailyCheckIn
from ml.services.risk_input_mapper import build_common_risk_input

class EverydayInputTests(TestCase):
    def data(self, **changes):
        data = dict(sleep_hours='8', water_liters='2', exercise_hours='0', exercise_extra_minutes='30', screen_time_hours='4', mood='4', stress_level='2', energy_level='4')
        return {**data, **changes}

    def test_no_device_needed_and_direct_activity_inputs_are_used(self):
        form=DailyCheckInForm(self.data())
        self.assertNotIn('steps',form.fields)
        self.assertNotIn('sitting_hours',form.fields)
        self.assertNotIn('weekly_activity_minutes',form.fields)
        self.assertTrue(form.is_valid(),form.errors)
        user=User.objects.create_user(username='simpleinputs')
        record=form.save(commit=False);record.user=user;record.date=timezone.localdate();record.save()
        self.assertEqual(calculate_exercise_score(record),100)
        values=build_common_risk_input(user)
        self.assertIsNone(values['total_activity_minutes_week'])
        self.assertIsNone(values['sedentary_minutes_day'])

    def test_impossible_time_budget_is_rejected(self):
        form=DailyCheckInForm(self.data(sleep_hours='20',exercise_hours='5'))
        self.assertFalse(form.is_valid())
        self.assertIn('cannot exceed 24 hours',str(form.errors))

    def test_zero_exercise_remains_a_valid_answer(self):
        form=DailyCheckInForm(self.data(exercise_extra_minutes='0'))
        self.assertTrue(form.is_valid(),form.errors)

    def test_invalid_exercise_minutes_are_rejected(self):
        form=DailyCheckInForm(self.data(exercise_extra_minutes='60'))
        self.assertFalse(form.is_valid())
        self.assertIn('exercise_extra_minutes',form.errors)

