# Models with everyday inputs

## Current deployment: lifestyle_v10 (15 October 2026)

The dashboard now uses all four lifestyle_v10 artifacts at the user's request
to use models with internal test accuracy above 70% for every outcome. Inputs
and validation-selected thresholds are unchanged. Verified test accuracies are
77.0% diabetes, 73.1% hypertension, 81.5% cardiovascular, and 74.6% mental
wellness; the unweighted mean is 76.5% (76.546%). This is not a combined accuracy
for the application or a guarantee for an individual prediction.

Positive recall is respectively 48.8%, 82.4%, 47.1%, and 50.3%. This change
prioritizes the requested accuracy criterion and reduces positive-case detection
for three outcomes compared with v6. The models do not meet the earlier
combined above-80% accuracy and minimum detection target. Results use a
repeatedly evaluated internal test split; no independent external validation
or probability calibration has been established. See `RETRAINING_V10_RESULTS.md`.

```powershell
python ml/evaluation/audit_saved_models.py --version lifestyle_v10
python manage.py test predictions dashboard --noinput
```

## Previous deployment: lifestyle_v6 (historical)

Latest source comparison: the user kept the existing inputs only. Thirty-six configurations were trained on the larger local BRFSS 2023 source using age, sex, BMI and smoking status. Independently reproduced test accuracies are 76.5%, 70.1%, 81.1% and 69.4%; positive recall is 50.4%, 64.7%, 49.8% and 49.5%. All-four above 80% was not achieved. This source omits sleep and targets previously diagnosed conditions, including a different mental outcome, so these are research candidates only and cannot replace the dashboard task's metrics. No inputs or deployment changed. See `CURRENT_INPUTS_LARGER_DATA_RESULTS.md`.

Mental-only follow-up: v9 compared 40 configurations plus three ensembles. Its candidate achieved 74.9% accuracy and 50.3% positive recall; no higher-detection candidate met both 70% recall and 60% specificity in validation. The 80% target remains unmet, and the active mental model is still v6. Details: `MENTAL_WELLNESS_TRAINING_V9.md`.

Previous NHANES training attempt: v8 compared 128 configurations across three folds plus twelve ensembles. Selected test accuracies are 76.8%, 72.4%, 82.0% and 74.7%. All-four above 80% was not achieved, and positive recall remains lower than the active models for three outcomes. v6 remains active. See `RETRAINING_V8_RESULTS.md` for that comparison and `RETRAINING_80_PERCENT_RESULTS.md` for the preceding v7 attempt.

The active models now exclude sitting time and seven-day exercise totals. Diabetes, hypertension and mental-health screening use age, sex, BMI, smoking status and today's sleep hours. Cardiovascular screening uses age, sex, BMI and smoking status, so its output does not change with daily sleep or exercise. Daily exercise still contributes to the health score and lifestyle actions. No weekly totals are guessed.

Reproduce and independently audit the current models:

```powershell
venv\Scripts\python.exe ml/training/train_lifestyle_models_v3.py --version lifestyle_v6 --exclude-feature total_activity_minutes_week --exclude-feature sedentary_minutes_day
venv\Scripts\python.exe ml/evaluation/audit_saved_models.py --version lifestyle_v6
```

| Screening | Held-out accuracy | Positive recall | ROC AUC |
| --- | --- | --- | --- |
| Diabetes | 64.2% | 77.2% | 0.756 |
| Hypertension | 72.1% | 72.2% | 0.780 |
| Cardiovascular | 75.5% | 67.7% | 0.809 |
| Mental health | 61.9% | 67.6% | 0.699 |

The unweighted mean is 68.4%, down from v2's 69.7%. It is not a combined project accuracy and does not meet a 75% mean-accuracy target. Diabetes, cardiovascular and mental-health accuracy is below the majority-class baseline. Their positive precision is limited (24.4%, 29.9%, and 21.0%), so many positive screenings will be false positives. These remain screening estimates, not diagnoses or guaranteed predictions. Models are not externally or clinically validated. The independent audit confirmed the saved artifacts reproduce their held-out metrics; see `verified_lifestyle_v6_metrics.json`.

Original datasets and prior artifacts are retained for provenance. v6 trains on the subset of columns in the existing reduced CSVs. All model selection and threshold tuning use validation data, separate from the test group.

## Previous deployment: lifestyle_v2 (historical)

Run `venv\Scripts\python.exe ml\training\train_lifestyle_models_v2.py` to reproduce all four models. Reduced CSVs contain only participant ID, the target, and age, sex, BMI, smoking status, exercise minutes/week, sitting minutes/day, and sleep hours where available in the source dataset. No waist, BP, or laboratory fields are used. Source data are retained to preserve provenance.

Participants are split into training, validation, and test groups with seed 42. Logistic regression and random forest are compared using validation ROC AUC. The threshold maximizes validation sensitivity minus false-positive rate, then the selected estimator is fitted on training plus validation data. The test group stays separate. Each JSON report includes ROC AUC, average precision, prevalence, precision, recall, and classification metrics.

Do not present AUC as percentage accuracy or screening probabilities as guaranteed outcomes. Targets reflect self-reported diagnoses or questionnaire-defined risk, not future disease occurrence. Models are not externally validated or clinically approved. Less input information can limit accuracy. Mental screening uses lifestyle factors, not a substituted PHQ score.

Check-ins accept sitting hours and exercise minutes over the last seven days. Daily dashboard predictions require all selected inputs. Missing entries are shown to the user; screen time is never substituted for sitting time, and daily exercise is never multiplied to invent a weekly total. Legacy records must be updated with direct values before showing a prediction.
