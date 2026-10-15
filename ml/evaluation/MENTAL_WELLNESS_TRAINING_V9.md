# Mental wellness focused training — 10 October 2026

Forty additional mental-only configurations were evaluated with three-fold validation, followed by three soft-voting ensembles. Inputs remain age, sex, BMI, smoking status and sleep. The target is depression-related screening risk, not today's mood. The dataset, labels, original test split and active dashboard model were preserved.

## Result

The selected accuracy candidate is a three-model soft-voting ensemble:

| Measure | Internal test result |
| --- | --- |
| Accuracy | **74.9%** |
| Positive-case detection (recall) | **50.3%** |
| Positive precision | 26.5% |
| Specificity | 78.6% |
| ROC AUC | 0.713 |
| Test records | 1,091 |

An independent reload reproduced the saved artifact's test accuracy and ROC AUC. The candidate was saved as `mental_health_lifestyle_v9.joblib` and **not deployed**. Above 80% accuracy was not achieved. It misses 49.7% of positive test cases, and only about one in four positive flags corresponds to a positive dataset label.

The higher-detection search required at least 70% positive recall and 60% specificity during validation. **No candidate qualified.** Its summary explicitly records no eligible model. The earlier fallback could save an ineligible candidate; that behavior was corrected and regression-tested. The rejected fallback had 86.3% test accuracy but only 0.7% positive recall. Its executable artifact was removed, while `rejected_mental_health_v9_screening.json` retains the rejection reason and observed metrics. This result must not be advertised as successful mental prediction accuracy.

| Mental model | Accuracy | Positive recall | Deployment |
| --- | --- | --- | --- |
| v6 | 61.9% | 67.6% | Active |
| v7 | 75.8% | 46.9% | Candidate only |
| v8 | 74.7% | 51.7% | Candidate only |
| v9 | 74.9% | 50.3% | Candidate only |

The focused search did not improve on the v7 candidate's accuracy or preserve the active v6 model's positive detection. High accuracy can be misleading here: predicting everyone negative scores 86.7% while identifying no positive cases. Continued tuning with the same weak input signal does not establish clinical reliability.

## Method and checks

The new search includes calibrated balanced SVMs, nearest-neighbour classifiers, naive Bayes, regularized discriminant analysis, calibrated boosting, balanced random forests, neural networks, and SMOTENC training-fold balancing. Synthetic balancing rows are created only inside fitting folds; sex and smoking retain valid categorical values. Test rows are never resampled or balanced. No symptom-score or target-derived fields are used to fabricate better accuracy.

Both accuracy and higher-detection winners are chosen from out-of-fold predictions before test evaluation. The final estimator is refitted on all training rows. The original internal test split has already been reported in previous runs; these experiments are not new independent external validation. Cutoffs can transfer imperfectly after refitting, and probabilities are not clinically calibrated. Better accurately labelled data with relevant user-supplied predictors and genuinely independent validation remain necessary for stronger evidence.

Nine training checks passed, including held-out fold separation, category-preserving balancing, unchanged inference row counts, cutoff tradeoffs, and rejection of high-accuracy candidates that fail detection constraints.

```powershell
venv\Scripts\python.exe -m ml.training.train_mental_models_v9
venv\Scripts\python.exe ml/evaluation/audit_saved_models.py --version lifestyle_v9 --model mental_health
venv\Scripts\python.exe manage.py test predictions.test_mental_training predictions.test_training_selection predictions.test_cross_validated_training
```

See `mental_health_v9_training_summary.json`, `verified_lifestyle_v9_metrics.json`, and `mental_health_lifestyle_v9_metadata.json` for results and all validation comparisons. The other three active prediction models were not retrained in this round.
