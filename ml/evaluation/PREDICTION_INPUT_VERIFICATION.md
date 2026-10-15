# Prediction verification: 10 October 2026

All 48 prediction/dashboard tests passed. The artifact audit independently reproduced all four saved models' held-out metrics. No discrepancy was found between supplied model inputs, direct model output, displayed results and stored prediction snapshots in the tested flows.

## What was verified

- Three entered profiles (ages 24, 48 and 70; BMI 20, 30 and 35; never/former/current smoking; male/female; sleep 8, 6 and 4 hours) were submitted to the real dashboard using a separate test database.
- For all twelve results, independent joblib loads and direct model calls matched the dashboard probabilities to twelve decimal places. Input snapshots matched independently calculated age, BMI, gender encoding, smoking encoding and sleep values.
- Percentage equals probability multiplied by 100 and rounded to two decimal places. The predicted class and displayed risk label use each saved model's selected cutoff.
- Saved prediction inputs, classes and probabilities match displayed results. Editing today's check-in refreshes values without adding duplicate daily records.
- Missing gender prevents displaying a new result; old results are not substituted. Invalid sleep input is rejected without replacing saved values. Earlier tests verify yesterday's check-in cannot supply today's results.
- Side notes list each model's actual inputs, not every field the user entered. Cardiovascular screening does not include sleep. Mental screening does not include mood/stress or an invented symptom scale.

These tests use fixtures, not changes to real user profiles. They establish software consistency, not medical correctness for any individual.

## Actual predictive performance

| Screening | Held-out accuracy | Positive precision | Positive recall |
| --- | --- | --- | --- |
| Diabetes | 64.2% | 24.4% | 77.2% |
| Hypertension | 72.1% | 76.8% | 72.2% |
| Cardiovascular | 75.5% | 29.9% | 67.7% |
| Mental wellness | 61.9% | 21.0% | 67.6% |

The unweighted mean accuracy is 68.4%; there is no combined project accuracy. Low positive precision means many elevated results are false positives. Probabilities are not clinically calibrated. These models are internally evaluated, not externally or clinically validated, and cannot guarantee a correct health prediction.

Age, sex, BMI and smoking are used by all models; today's sleep is also used by diabetes, hypertension and mental screening. Daily exercise, water, diet, mood and stress affect lifestyle scoring/guidance, not the trained disease models. Cardiovascular output stays the same if only those daily values change. Mental screening targets depression-related risk, not today's mood. Improving the interface or passing software tests does not increase these measured accuracies.

Reproduce verification:

```powershell
venv\Scripts\python.exe manage.py test predictions dashboard
venv\Scripts\python.exe ml/evaluation/audit_saved_models.py --version lifestyle_v6
```

See `verified_lifestyle_v6_metrics.json` for confusion matrices and `LIFESTYLE_MODELS.md` for model features and limitations.
