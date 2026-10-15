# Additional training results — 15 October 2026

The requested above-80% accuracy across all four outcomes was not achieved.
Forty-eight additional configurations were fitted with three-fold stratified
validation (144 fold fits), followed by four final refits. Current inputs,
original labels, datasets, and deployed lifestyle_v6 artifacts are unchanged.
At training time these lifestyle_v10 artifacts were research candidates only.
They were subsequently activated at the user's explicit request on 15 October
2026 using the revised requirement of >70% internal test accuracy for each
outcome. The original >80% plus detection requirement remains unmet. This
activation does not establish independent validation or improve all recalls.

| Outcome | Test accuracy | Positive recall | Positive precision | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Diabetes | 77.0% | 48.8% | 29.4% | 0.765 |
| Hypertension | 73.1% | 82.4% | 73.1% | 0.780 |
| Cardiovascular | 81.5% | 47.1% | 34.0% | 0.814 |
| Mental wellness | 74.6% | 50.3% | 26.3% | 0.707 |

## Method

Each outcome compares nine histogram boosting configurations with budgets of
150, 450, and 900 iterations and L2 regularization of 1, 30, and 100. Early
stopping can terminate before the iteration budget. Three spline elastic-net
logistic configurations compare C=0.03, 0.3, and 3. All four validation-selected
winners use C=0.3. Some unselected C=3 folds issued convergence warnings; the
selected configurations and final refits issued none in the recorded log.

The original 20% stratified test partition uses seed 42. Selection uses only
three-fold out-of-fold predictions from the remaining 80%, with shuffled folds
and seed 127. Imputation, encoding, and scaling fit inside each training fold.
Model and threshold selection maximize OOF accuracy while requiring at least
50% positive recall and 60% specificity. Ineligible candidates cannot win;
there is no fallback to a majority-only classifier. Final models are refitted
on all training rows before test evaluation; test results do not retune cutoffs.

## Interpretation

The larger boosting budgets did not establish that more iterations yield more
accurate predictions. Diabetes and mental accuracy are similar to previous
research candidates. Heart accuracy exceeds 80% but misses 52.9% of positive
test cases; diabetes misses 51.2%. Majority-negative baselines are approximately
86.4%, 56.2%, 87.2%, and 86.7%, respectively. Higher accuracy alone is not
evidence of better positive-case detection. No candidate met the combined
test requirement of >80% accuracy, >=50% recall, and >=60% specificity.

The internal test partition has been reported repeatedly across experiments,
so it is not fresh independent validation. Selection scores can be optimistic
after comparing models and thresholds on the same OOF predictions. These
outputs are not externally validated or clinically calibrated probabilities.
Individual correctness cannot be established from a user's inputs without
independent outcome labels. The mental target remains depression-related
screening risk, not a prediction of today's mood.

Further improvement should focus on representative, accurately labelled data
with informative predictors available at inference, followed by a locked
evaluation on genuinely independent participants. Additional questions or
outcome definitions require explicit design decisions; target-derived answers
must not be introduced as predictors to inflate scores.

## Verification and reproduction

Eight selection and fold-separation tests passed. Saved artifacts were reloaded
during training and independently audited; their held-out accuracy and ROC AUC
match metadata. Detailed results are in `lifestyle_v10_training_summary.json`,
`verified_lifestyle_v10_metrics.json`, and `lifestyle_v10_training.log`.

```powershell
python -m ml.training.train_lifestyle_models_v10
python ml/evaluation/audit_saved_models.py --version lifestyle_v10
python manage.py test predictions.test_training_v10 predictions.test_training_selection predictions.test_cross_validated_training --noinput
```
