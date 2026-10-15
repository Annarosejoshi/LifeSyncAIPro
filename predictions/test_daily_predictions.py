from datetime import date, timedelta
from unittest.mock import patch
from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from django.urls import reverse
from accounts.models import UserProfile
from predictions.models import PredictionHistory
from lifestyle.models import DailyCheckIn
from ml.services.risk_input_mapper import build_common_risk_input
from ml.services.risk_prediction_service import get_all_health_risks

class DailyPredictionTests(TestCase):
    @patch('ml.services.risk_prediction_service.load_model')
    def test_adult_models_do_not_generate_predictions_for_children(self, model):
        today = timezone.localdate()
        UserProfile.objects.filter(user=self.user).update(
            date_of_birth=date(today.year-15, today.month, today.day),
            gender='female', height_cm=165, weight_kg=60, smoking_status='never')
        self.record(today, 8)
        results = get_all_health_risks(self.user)
        self.assertTrue(all(not result['prediction_available'] for result in results.values()))
        self.assertTrue(all('adults' in result['reason'] for result in results.values()))
        model.assert_not_called()
    def setUp(self):
        self.user=User.objects.create_user(username='daily_predictions')

    def record(self,day,sleep):
        return DailyCheckIn.objects.create(user=self.user,date=day,sleep_hours=sleep,water_liters=2,exercise_minutes=30,screen_time_hours=4,mood=4,stress_level=2,energy_level=4)

    @patch('ml.services.risk_prediction_service.load_model')
    def test_yesterday_is_never_used_for_today(self,model):
        self.record(timezone.localdate()-timedelta(days=1),8)
        results=get_all_health_risks(self.user)
        self.assertEqual(len(results),4)
        self.assertTrue(all(not result['prediction_available'] for result in results.values()))
        self.assertIsNone(build_common_risk_input(self.user)['sleep_hours'])
        model.assert_not_called()

    def test_today_wins_over_future_and_previous_records(self):
        today=timezone.localdate()
        self.record(today-timedelta(days=1),9)
        self.record(today,7)
        self.record(today+timedelta(days=1),3)
        self.assertEqual(build_common_risk_input(self.user)['sleep_hours'],7)

    def test_editing_today_changes_prediction_inputs(self):
        record=self.record(timezone.localdate(),8)
        self.assertEqual(build_common_risk_input(self.user)['sleep_hours'],8)
        record.sleep_hours=5;record.save()
        self.assertEqual(build_common_risk_input(self.user)['sleep_hours'],5)

    def test_missing_activity_and_sitting_do_not_block_simplified_models(self):
        UserProfile.objects.filter(user=self.user).update(date_of_birth=date(1990,1,1),gender='female',height_cm=165,weight_kg=60,smoking_status='never')
        self.record(timezone.localdate(),8)
        inputs = build_common_risk_input(self.user)
        self.assertIsNone(inputs['total_activity_minutes_week'])
        self.assertIsNone(inputs['sedentary_minutes_day'])
        results = get_all_health_risks(self.user)
        for result in results.values():
            self.assertTrue(result['prediction_available'], result)
            self.assertNotIn('total_activity_minutes_week', result['input_snapshot'])
            self.assertNotIn('sedentary_minutes_day', result['input_snapshot'])

    def test_saving_today_refreshes_four_daily_results_without_duplicates(self):
        UserProfile.objects.filter(user=self.user).update(date_of_birth=date(1990,1,1),gender='female',height_cm=165,weight_kg=60,smoking_status='never')
        self.client.force_login(self.user)
        data=dict(action='daily_checkin',sleep_hours='8',water_liters='2',exercise_hours='0',exercise_extra_minutes='30',weekly_activity_minutes='150',sitting_hours='6',screen_time_hours='3',mood='4',stress_level='2',energy_level='4')
        response=self.client.post(reverse('dashboard:risks'),data)
        self.assertEqual(response.status_code,302)
        rows=PredictionHistory.objects.filter(user=self.user,created_at__date=timezone.localdate())
        self.assertEqual(rows.count(),4)
        original_ids=set(rows.values_list('pk',flat=True))
        data['sleep_hours']='5'
        response=self.client.post(reverse('dashboard:risks'),data)
        self.assertEqual(response.status_code,302)
        self.assertEqual(set(rows.values_list('pk',flat=True)),original_ids)
        self.assertEqual(rows.get(prediction_type='diabetes').input_snapshot['sleep_hours'],5)
        response = self.client.get(reverse('dashboard:risks'))
        self.assertNotContains(response, 'Values used for this result')
        self.assertContains(response, 'Lifestyle screening estimate', count=4)
        self.assertNotContains(response, 'class="checkin-module"')
        overview = self.client.get(reverse('dashboard:home'))
        self.assertContains(overview, 'Total Health Score')
        self.assertNotContains(overview, 'Values used for this result')
        self.assertEqual(
            overview.context['risk_cards'][0][1]['input_snapshot']['sleep_hours'], 5
        )
