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

    def complete_profile(self):
        UserProfile.objects.filter(user=self.user).update(
            date_of_birth=date(1990, 1, 1), gender='female', height_cm=165,
            weight_kg=60, smoking_status='never')

    def test_first_day_generates_all_results_without_week_of_records(self):
        self.complete_profile()
        checkin = self.record(timezone.localdate(), 6)
        results = get_all_health_risks(self.user)
        self.assertEqual(self.user.daily_checkins.count(), 1)
        for name, result in results.items():
            self.assertTrue(result['prediction_available'], result)
            self.assertEqual(result['input_date'], checkin.date.isoformat())
            self.assertEqual(result['checkin_id'], checkin.pk)
            if name != 'cardiovascular':
                self.assertEqual(result['input_snapshot']['sleep_hours'], 6)

    def test_changing_previous_six_days_cannot_change_todays_estimates(self):
        self.complete_profile()
        today = timezone.localdate()
        for days in range(1, 7):
            self.record(today-timedelta(days=days), 10)
        self.record(today, 6)
        before = get_all_health_risks(self.user)
        self.user.daily_checkins.exclude(date=today).update(sleep_hours=2, water_liters=.1, exercise_minutes=0)
        after = get_all_health_risks(self.user)
        for name in before:
            self.assertEqual(before[name]['input_snapshot'], after[name]['input_snapshot'])
            self.assertEqual(before[name]['probability'], after[name]['probability'])

    def test_new_day_requires_new_entry_and_never_reuses_yesterday(self):
        self.complete_profile()
        today = timezone.localdate()
        self.record(today, 6)
        self.assertTrue(all(row['prediction_available'] for row in get_all_health_risks(self.user).values()))
        tomorrow = today + timedelta(days=1)
        with patch('django.utils.timezone.localdate', return_value=tomorrow):
            missing = get_all_health_risks(self.user)
            self.assertTrue(all(not row['prediction_available'] for row in missing.values()))
            self.record(tomorrow, 8)
            updated = get_all_health_risks(self.user)
            for name, row in updated.items():
                self.assertEqual(row['input_date'], tomorrow.isoformat())
                if name != 'cardiovascular':
                    self.assertEqual(row['input_snapshot']['sleep_hours'], 8)

    def test_prior_day_result_cannot_overwrite_todays_history(self):
        from ml.services.risk_history_service import save_single_risk_history
        self.complete_profile()
        self.record(timezone.localdate(), 6)
        result = get_all_health_risks(self.user)['diabetes']
        saved = save_single_risk_history(self.user, 'diabetes', result)
        stale = dict(result, input_date=(timezone.localdate()-timedelta(days=1)).isoformat(), probability=.99)
        self.assertIsNone(save_single_risk_history(self.user, 'diabetes', stale))
        saved.refresh_from_db()
        self.assertEqual(saved.probability, result['probability'])
        self.assertEqual(PredictionHistory.objects.filter(user=self.user).count(), 1)

    def test_request_day_stays_consistent_if_clock_crosses_midnight(self):
        self.complete_profile()
        request_day = timezone.localdate()
        source = self.record(request_day, 6)
        next_day = request_day + timedelta(days=1)
        self.record(next_day, 9)
        with patch('django.utils.timezone.localdate', return_value=next_day):
            results = get_all_health_risks(self.user, prediction_date=request_day)
        for name, row in results.items():
            self.assertEqual(row['checkin_id'], source.pk)
            self.assertEqual(row['input_date'], request_day.isoformat())
            if name != 'cardiovascular':
                self.assertEqual(row['input_snapshot']['sleep_hours'], 6)

    def test_daily_habits_update_wellness_score_without_inventing_model_changes(self):
        self.complete_profile()
        self.record(timezone.localdate(), 8)
        self.client.force_login(self.user)
        before = self.client.get(reverse('dashboard:risks'))
        after = self.client.post(reverse('dashboard:checkin'), dict(
            action='daily_checkin', sleep_hours='8', water_liters='.1',
            exercise_hours='0', exercise_extra_minutes='0', screen_time_hours='15',
            mood='1', stress_level='5', energy_level='1'), follow=True)
        self.assertEqual(after.status_code, 200)
        self.assertLess(after.context['ai_health_score'], before.context['ai_health_score'])
        self.assertContains(after, "No week of history is required.")
        self.assertContains(after, "Verify today's saved habits")
        for (_, old), (_, new) in zip(before.context['risk_cards'], after.context['risk_cards']):
            self.assertEqual(old['probability'], new['probability'])

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
