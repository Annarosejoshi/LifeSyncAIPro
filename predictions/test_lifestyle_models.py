from unittest.mock import patch, MagicMock
from django.test import SimpleTestCase
from ml.services.risk_prediction_service import load_metadata, predict_diabetes_risk, predict_risk, predict_hypertension_risk, predict_cardiovascular_risk, predict_mental_health_risk

class EverydayModelTests(SimpleTestCase):
    @patch('ml.services.risk_prediction_service.predict_risk')
    def test_missing_essential_inputs_never_produce_a_result(self, predictor):
        values=dict(age=35,sex=2,bmi=24,smoking_status=0,total_activity_minutes_week=150,sedentary_minutes_day=420,sleep_hours=8)
        for wrapper in (predict_diabetes_risk,predict_hypertension_risk,predict_cardiovascular_risk,predict_mental_health_risk):
            for field in ('age','sex','bmi'):
                with self.subTest(wrapper=wrapper.__name__,field=field):
                    result=wrapper({**values,field:None})
                    self.assertFalse(result['prediction_available'])
        predictor.assert_not_called()

    def test_all_deployed_models_exclude_medical_measurements(self):
        forbidden={'waist_circumference','avg_systolic_bp','avg_diastolic_bp','total_cholesterol','hdl_cholesterol'}
        for name in ('diabetes','hypertension','cardiovascular','mental_health'):
            metadata=load_metadata(name)
            self.assertEqual(metadata['model_version'],'lifestyle_v10')
            self.assertFalse(forbidden.intersection(metadata['features']))
            self.assertNotIn('total_activity_minutes_week',metadata['features'])
            self.assertNotIn('sedentary_minutes_day',metadata['features'])
            self.assertGreater(metadata['test_rows'],0)
            self.assertGreater(metadata['test_metrics']['classification']['accuracy'], .70)

    @patch('ml.services.risk_prediction_service.load_model')
    def test_diabetes_passes_entered_values_in_training_order(self,load_model):
        model=MagicMock();model.predict_proba.return_value=[[.7,.3]];load_model.return_value=model
        values=dict(age=35,sex=2,bmi=24,smoking_status=0,total_activity_minutes_week=150,sedentary_minutes_day=420,sleep_hours=8,waist_circumference=999)
        result=predict_diabetes_risk(values)
        frame=model.predict_proba.call_args.args[0]
        self.assertEqual(list(frame.columns),load_metadata('diabetes')['features'])
        self.assertEqual(frame.loc[0,'sleep_hours'],8)
        self.assertEqual(frame.loc[0,'bmi'],24)
        self.assertNotIn('total_activity_minutes_week',frame.columns)
        self.assertNotIn('sedentary_minutes_day',frame.columns)
        self.assertNotIn('waist_circumference',frame.columns)
        self.assertEqual(result['prediction'],int(.3>=result['threshold']))
        self.assertEqual(result['model_version'],'lifestyle_v10')
