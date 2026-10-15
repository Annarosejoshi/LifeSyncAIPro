# Retraining results — 10 October 2026

Current update: the dashboard now uses v6, retrained after removing sitting and seven-day activity inputs at the user's request. Its mean held-out accuracy is 68.4%. The v5 comparison below is historical, and those candidates remain undeployed. See `LIFESTYLE_MODELS.md` and `verified_lifestyle_v6_metrics.json` for current performance and limitations.

Seven model methods were compared: logistic regression, quadratic logistic
regression, random forest, extra trees, histogram gradient boosting, and two
regularized XGBoost configurations. Only existing lifestyle inputs were used.
No BP, waist or laboratory inputs were added. Source CSVs and deployed v2
artifacts remain intact.

The reproducible split is 60% fitting, 20% validation and 20% test, stratified
with seed 42. Candidate selection and threshold selection use validation rows.
Selected estimators are not refitted after cutoff selection, so their probability
scale stays the same. Test rows are neither fitted nor oversampled. These are
exploratory internal comparisons using the same previously evaluated dataset
and split; a new external dataset is needed for independent confirmation.

## Selected accuracy-focused candidates (lifestyle_v5)

| Model | Test accuracy | Positive recall | ROC AUC |
|---|---:|---:|---:|
| Diabetes / XGBoost | 79.6% | 45.1% | 0.771 |
| Hypertension / Extra Trees | 72.4% | 81.3% | 0.769 |
| Cardiovascular / XGBoost | 81.0% | 41.9% | 0.807 |
| Mental wellness / XGBoost | 75.7% | 49.0% | 0.720 |
| Unweighted mean accuracy | **77.2%** | | |

The 75% goal was reached for the average, not for every model. Hypertension
remains below 75%. This is not one combined accuracy measured on one common
test cohort.

Higher accuracy does not establish better screening: diabetes now misses
54.9% of positive test cases, cardiovascular misses 58.1%, and mental screening
misses 51.0%. These datasets are imbalanced; the majority-only accuracy baseline
is reported in each evaluation JSON and can exceed the model's accuracy while
detecting no positive cases. The previous diabetes model had 78.6% recall.

**At the time of this v5 evaluation, the dashboard used lifestyle_v2.** v5 models are saved for review, not
silently promoted as more reliable health predictions. v3 (minimum 60%
validation recall, balanced-accuracy ranking) achieved 73.2% average accuracy;
v4 (minimum 50%, balanced-accuracy ranking) achieved 74.5%; v5 ranks by validation
accuracy while retaining at least 50% validation recall and 60% specificity.
Validation constraints do not guarantee equivalent test sensitivity.

Run the accuracy-focused training:

```powershell
venv\Scripts\python.exe ml/training/train_lifestyle_models_v3.py --version lifestyle_v5 --minimum-recall 0.5 --selection-metric accuracy
```

Independently reload and verify its saved artifacts and reported metrics:

```powershell
venv\Scripts\python.exe ml/evaluation/audit_saved_models.py --version lifestyle_v5
```

Models are stored in `ml/models/*_lifestyle_v5.joblib`; selection metadata and
per-model evaluation JSONs document exact cutoffs, confusion matrices, accuracy,
precision and recall. The summary is `lifestyle_v5_training_summary.json`.
