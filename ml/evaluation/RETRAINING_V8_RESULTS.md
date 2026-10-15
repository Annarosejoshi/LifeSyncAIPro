# Further training results — 10 October 2026

The additional v8 search completed **128 base configurations with three-fold validation (384 validation fits), plus twelve ensemble comparisons and final refits**. It uses the current inputs and preserves original datasets, target labels and active v6 artifacts. The all-four above-80% target remains unmet. v8 models are saved separately and are not deployed.

## Independently verified results

| Prediction | Selected candidate | v7 accuracy | v8 accuracy | v8 positive precision | v8 positive recall |
| --- | --- | --- | --- | --- | --- |
| Diabetes | Categorical spline logistic regression | 76.6% | 76.8% | 28.7% | 47.0% |
| Hypertension | Categorical logistic regression | 72.0% | 72.4% | 73.1% | 80.3% |
| Cardiovascular | Categorical logistic regression | 81.3% | 82.0% | 35.2% | 48.4% |
| Mental screening | Soft-voting ensemble of three candidates | 75.8% | 74.7% | 26.7% | 51.7% |

Mean accuracy moved from 76.4% to **76.5%**; it is an unweighted average of separate outcomes, not a combined project accuracy. Mental accuracy decreased, although positive recall improved over v7. Only cardiovascular accuracy exceeds 80%, and it still misses 51.6% of positive test cases. Diabetes misses 53.0%, and mental screening misses 48.3%. These three candidates detect fewer positive cases than the currently deployed v6 models.

## Changes in this training round

- Gender and smoking status are encoded as separate categories, rather than treating them as continuous numerical measurements.
- Numeric preprocessing uses training-fold imputation, standardization, spline transforms and higher-order interactions. Preprocessing is fitted only within each fold.
- The search includes balanced and unbalanced tree models, weighted XGBoost, neural networks, and probability-voting ensembles. It adds no blood pressure, waist, sitting, weekly exercise or laboratory inputs.
- Instead of selecting on one small validation split, each training row receives a prediction from a model fitted on other rows in three-fold cross-validation. Tests verify held-out separation and restoration of row order.
- Models and thresholds are selected using these out-of-fold predictions. The final estimator is fitted on the full 80% training partition. Thresholds can transfer imperfectly after refitting; this is included in metadata and evaluated on the test partition without retuning.
- Validation selection requires at least 50% positive recall and 60% specificity. A neural network with 86.4% diabetes validation accuracy but zero positive recall was explicitly ineligible.
- Ensemble membership is selected using training cross-validation only. No test labels are used to choose ensemble weights, model settings or thresholds.

The original internal test split uses seed 42 and was evaluated in prior training runs. These results are not independent external or clinical validation. Probabilities are not clinically calibrated. Diabetes, heart and mental datasets are imbalanced: a majority-only classifier scores 86.4%, 87.2% and 86.7% respectively while detecting no positive cases. Therefore an accuracy claim above 80% alone does not show helpful predictions.

The broader search yielded little mean improvement. That suggests a limit in the predictive information available to the tested methods with these inputs; it does not prove that better models cannot exist. A credible next improvement requires better labelled data with informative inputs users can actually provide, followed by evaluation on genuinely independent data. Target diagnosis and target-derived questionnaire scores must not be reintroduced as predictors to manufacture accuracy.

## Validation and reproduction

Six training-selection and cross-validation checks passed. An independent reload of all four artifacts reproduced the recorded test accuracy and ROC AUC. `verified_lifestyle_v8_metrics.json` includes precision, recall and confusion matrices; `lifestyle_v8_training_summary.json` contains the summary and deployment status.

```powershell
venv\Scripts\python.exe -m ml.training.train_lifestyle_models_v8
venv\Scripts\python.exe ml/evaluation/audit_saved_models.py --version lifestyle_v8
venv\Scripts\python.exe manage.py test predictions.test_training_selection predictions.test_cross_validated_training
```
