# Above-80% accuracy training attempt — 10 October 2026

Follow-up: a further three-fold v8 search is complete. See `RETRAINING_V8_RESULTS.md` for the latest results; the v7 comparison below is retained as experiment history.

**The requested target was not achieved for all four predictions.** Training compared 24 configurations per outcome (96 fits): logistic regression, quadratic logistic regression, spline logistic regression, histogram gradient boosting, random forest, extra trees, XGBoost and RBF support vector machines. Candidates are saved as `lifestyle_v7`. The dashboard remains on `lifestyle_v6`.

## Selected candidates: internal test results

| Screening | Selected estimator | Accuracy | Positive precision | Positive recall | Majority-only accuracy |
| --- | --- | --- | --- | --- | --- |
| Diabetes | XGBoost depth 4 | 76.6% | 27.6% | 43.7% | 86.4% |
| Hypertension | XGBoost depth 4 | 72.0% | 73.5% | 78.3% | 56.2% |
| Cardiovascular | Random forest, leaf size 20 | 81.3% | 33.3% | 45.8% | 87.2% |
| Mental screening | Spline logistic regression, 5 knots | 75.8% | 26.7% | 46.9% | 86.7% |

The unweighted mean accuracy is **76.4%**, not a combined project accuracy. Only cardiovascular accuracy is above 80%; its positive recall means it misses 54.2% of positive test cases. Diabetes misses 56.3%, and mental screening misses 53.1%. These three candidates have substantially lower positive recall than the active v6 models. They were not deployed.

## Method

- The source datasets and target labels were preserved. No rows were invented, targets changed, test data balanced, or majority-only predictions presented as useful screening.
- Inputs match the current forms: age, sex, BMI, smoking status; sleep where the source contains it. No sitting, seven-day activity, blood pressure, waist or laboratory fields were added.
- Participant IDs are unique and excluded from model inputs. The original stratified train/validation/test split uses seed 42. Imputation and transformations are fitted inside pipelines on training rows only.
- Model and cutoff selection use validation data only. The selection seeks maximum accuracy while requiring at least 50% positive recall and 60% specificity. The exact fitted estimator is retained so refitting cannot invalidate its cutoff.
- Each winner is evaluated on test data only after selection. Internal test data have been evaluated in earlier experiments, so these results do not constitute new independent or external validation.
- Deployment checks require all four accuracies above 80%, positive recall at least 50%, and specificity at least 60%. The target was not met. Passing those checks alone would still not establish clinical reliability.

The dataset is imbalanced for diabetes, heart and mental screening. Classifying everyone as negative can exceed 80% accuracy while detecting no positive cases, as shown by the majority-only baselines. Accuracy must be accompanied by recall, precision and confusion matrices.

These experiments do not show that 80% is impossible; they show it was not achieved with this dataset, available inputs and tested models. Further credible improvement needs more informative, accurately labelled data matching inputs users can supply. Known diagnosis answers cannot be used as inputs to predict the same diagnosis, and symptom scores must not be invented from unrelated mood ratings.

## Reproduce

```powershell
venv\Scripts\python.exe ml/training/train_lifestyle_models_v7.py
venv\Scripts\python.exe ml/evaluation/audit_saved_models.py --version lifestyle_v7
venv\Scripts\python.exe manage.py test predictions.test_training_selection
```

All three model-selection checks passed. The independent artifact audit reproduced the four recorded test accuracies and ROC AUCs. See `lifestyle_v7_training_summary.json`, `verified_lifestyle_v7_metrics.json`, and individual model metadata for all validation comparisons and confusion matrices.
