# Larger local dataset comparison — 10 October 2026

The user requested accuracy above 80% for all predictions and explicitly chose **Keep current inputs only**. No questions, input mappings, dashboard behavior, original datasets or active models were changed. This experiment tested whether a much larger existing local source could support stronger predictions using the available profile inputs.

**The all-four above 80% objective was not achieved.** Four research artifacts were saved separately as `*_brfss2023_profile_v1.joblib`; none is deployed. The dashboard still uses `lifestyle_v6` with the performance documented in `LIFESTYLE_MODELS.md`.

## New internal test results

| Research outcome | Usable records | Test records | Accuracy | Positive recall | Positive precision | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Diabetes | 382,936 | 76,588 | 76.5% | 50.4% | 29.8% | 0.753 |
| Hypertension | 382,294 | 76,459 | 70.1% | 64.7% | 63.5% | 0.767 |
| Cardiovascular | 379,943 | 75,989 | 81.1% | 49.8% | 22.9% | 0.780 |
| Previously diagnosed depressive disorder | 381,627 | 76,326 | 69.4% | 49.5% | 34.2% | 0.675 |

Cardiovascular alone exceeds 80% accuracy, but misses about half the positive cases and falls slightly below the 50% detection constraint on test data. Mental prediction also falls below that test constraint. Those outcomes are explicitly marked in the summary. Simply predicting every record negative would score 86.1% for diabetes, 91.3% for cardiovascular and 79.1% for depression while detecting no positive cases; these baselines are not successful prediction models.

## Source and compatibility

The existing local file `ml/data/sleep/LLCP2023.XPT` is the CDC BRFSS 2023 source. Coding was checked against the [CDC questionnaire](https://www.cdc.gov/brfss/questionnaires/pdf-ques/2023-BRFSS-Questionnaire-508.pdf) and [calculated-variable documentation](https://www.cdc.gov/brfss/annual_data/2023/pdf/2023-calculated-variables-version4-508.pdf). The [CDC source page](https://www.cdc.gov/brfss/annual_data/annual_2023.html) documents this survey and its sampling/weighting limitations.

Predictors are age, recorded sex category, BMI and smoking status. CDC's imputed age is capped at 80; BMI is decoded from two implied decimal places. Smoking categories are mapped to the existing never/former/current encoding. Missing predictor rows are excluded; unknown/refused outcomes are not treated as negative. A positive heart-attack or coronary-heart-disease answer establishes a positive cardiovascular outcome; a negative requires both answers to be negative. Pregnancy-only diabetes or hypertension and borderline categories are not positive for the corresponding diagnosed condition. Prediabetes is not relabelled as diabetes.

**This source has no sleep field or current-day lifestyle observations.** Its outcomes are self-reported past diagnoses, including previously diagnosed depressive disorder. That mental outcome differs from the existing NHANES questionnaire-defined depression risk. These figures cannot be presented as improved accuracy on the existing dashboard task, today's mental state, future disease occurrence, or clinically validated probabilities. No daily sleep, activity, mood or diet answers were converted into invented survey measurements. Blood pressure, waist, sitting, diagnosis-related medication and target-derived symptom scores were not used as predictors.

## Training and verification

Nine configurations per outcome (36 total) compare logistic models with categorical preprocessing and spline features, histogram gradient boosting, and XGBoost. Respondents are split into 60% fitting, 20% validation and 20% test groups, with fixed seeds 2026/2027. Selection maximizes validation accuracy subject to at least 50% positive recall and 60% specificity. Models failing validation eligibility cannot be selected. The validation-selected fitted estimator and cutoff are preserved without refitting or tuning against test results.

Test groups retain their natural prevalence; no test balancing, synthetic labels or target leakage is used. Test rows were excluded from fitting and selection. These are newly evaluated internal holdouts for this source, not independent external or clinical validation. Metrics are unweighted respondent-level estimates, not population prevalence estimates.

All four saved artifacts were independently reloaded and reproduced their accuracy, recall, precision, specificity, AUC, average precision and confusion matrices. Test respondent identifier fingerprints and feature order were checked. Thirteen training checks passed, covering source decoding, unknown outcomes, disjoint splits, natural test prevalence, category-preserving training resampling and rejection of ineligible candidates.

```powershell
venv\Scripts\python.exe -m ml.training.train_brfss_current_inputs
venv\Scripts\python.exe -m ml.evaluation.audit_brfss_current_inputs
venv\Scripts\python.exe manage.py test predictions.test_brfss_training predictions.test_training_selection predictions.test_cross_validated_training predictions.test_mental_training
```

Detailed outputs: `brfss2023_current_inputs_summary.json`, `verified_brfss2023_current_inputs.json`, and the four model metadata JSON files. The derived CSV `ml/data/processed/brfss2023_current_profile.csv` preserves respondent identifiers for reproducible splitting. The original source is intact.

The larger source did not establish that all four predictions can exceed 80% using these inputs. Earlier NHANES tuning results remain in `RETRAINING_V8_RESULTS.md` and `MENTAL_WELLNESS_TRAINING_V9.md`. Repeated tuning or a different cutoff cannot justify claiming the requested accuracy without adequate positive detection and compatible validation.
