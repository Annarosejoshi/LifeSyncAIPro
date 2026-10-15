# Project demonstration and verification

Use a separate demo account; keep real user records intact.

On Windows, start with `run_windows.bat --check`, then `run_windows.bat`.
The launcher checks dependencies and all four model pipelines. Frontend styles,
icons and chart assets are bundled locally, so the demonstration can run without
CDN access once Python dependencies have been installed.

1. Register, sign in, and complete the profile with date of birth, gender,
   height, weight, and smoking status. Adult estimates require age >=18.
2. Save today's check-in. Reopen it to verify saved values. Screen time accepts
   direct decimal input without a spinner.
3. Open Health Predictions. Compare each result's displayed inputs with saved
   data. Expand “How these results work” and “Measured model performance”.
4. Change sleep and save. Confirm the displayed sleep input changes. The heart
   model should remain unchanged because it does not use sleep. A risk category
   may remain unchanged even when its percentage changes.
5. Save a mental wellness assessment. It returns to predictions with a saved
   score confirmation. Change one answer and save again. Today's score should
   update; the separate profile/sleep estimate should remain unchanged. Higher
   stress, anxiety, or workload lowers the score; higher values for the other
   five dimensions raise it. Offsetting edits can preserve the same score.
6. Use the BMI calculator and confirm saved measurements reach prediction
   inputs. Review BMI history, medication tracking, profile editing, and Daily
   Wellness. Change the progress-chart metric using its selector.
7. Open the floating chat and submit a daily-habit question. Verify a response
   grounded in the signed-in user's data. Sign out and confirm protected pages
   require authentication.

## Evidence and limits

- Deployed models: lifestyle_v10 for all four tasks.
- Test accuracies: 77.0%, 73.1%, 81.5%, 74.6%; unweighted mean 76.5%.
- Positive recall: 48.8%, 82.4%, 47.1%, 50.3%. Report alongside accuracy.
- Selection: three-fold stratified OOF predictions; preprocessing inside folds;
  test rows excluded from model/threshold selection in that run.
- The test partition has been evaluated repeatedly in earlier experiments.
  It is not independent external validation.
- Daily wellness scores are rule-based summaries, not model probabilities.
- Application tests verify implementation, not clinical correctness.

Do not claim 100% correctness, 80% across all tasks, guaranteed diagnoses, or
that every daily habit is used by every trained model. Correctness for an
individual cannot be established merely from their inputs.

## Reproduce checks

```powershell
python manage.py check
python manage.py check_models
python manage.py makemigrations --check --dry-run
python manage.py test --noinput
python ml/evaluation/audit_saved_models.py --version lifestyle_v10
```

Use `ABSTRACT.md` for submission wording. Earlier evaluation reports remain
as historical experiment records.
