# Latest runtime reliability review

- All 157 application tests passed after the startup and offline-asset changes.
  The independent audit reproduced all four deployed model metrics unchanged.
- Bootstrap 5.3.8, Bootstrap Icons 1.13.1 and Chart.js 4.5.1 are served locally
  with licenses and an integrity manifest. No templates load CDN assets or Google Fonts.
- Eight synthetic page previews passed browser checks with external HTTP/HTTPS
  requests blocked. Local charts loaded, the progress metric selector updated
  the chart, and filtering, chat controls and mobile navigation worked without
  JavaScript errors. No horizontal overflow occurred at a 390-pixel viewport.
- A temporary live Django server served the login page and local CSS, icon font
  and chart script with HTTP 200 responses. It was stopped after verification.
- `run_windows.bat --check` passed. The launcher consistently uses the project
  environment, runs readiness checks and starts the server. Setup repairs an
  invalid copied environment in place and installs missing dependencies into
  that same environment. Initial dependency installation requires internet.
- `manage.py check_models` exercises all four pipelines without user database
  access and reports actionable missing-dependency or missing-artifact errors.
- Mental assessment create/edit flows remain covered by integration tests;
  questionnaire scores and profile/sleep model estimates remain separate.
- Django found no system issues or model migration changes. No schema changes
  or new prediction targets were introduced.

The earlier clean-package review and historical model evaluations follow.

# Earlier review — 15 October 2026

The active prediction models are lifestyle_v10. Independently reloaded artifacts
reproduced test accuracies of 77.0% diabetes, 73.1% hypertension, 81.5%
cardiovascular, and 74.6% depression-related screening. The unweighted mean is
76.5%; positive recall is 48.8%, 82.4%, 47.1%, and 50.3%. These are results on a
repeatedly evaluated internal holdout, not external or clinical validation.

## Changes and evidence

- Added ABSTRACT.md and EVALUATION_GUIDE.md with supported claims, input mappings,
  reproducible checks, limitations, and an evaluator demonstration sequence.
- Corrected README claims about deployed versions, exercise totals, and screen
  time. No weekly activity or sitting-time values are invented.
- Added collapsible prediction explanations and measured performance from the
  active artifacts' metadata, including recall and precision alongside accuracy.
- Improved result-summary spacing, mobile layout and keyboard focus indicators.
  Detailed explanations remain collapsed by default.
- Adult risk models now refuse to generate estimates for profiles under 18.
- Mental assessment answers update a separate daily wellness score. Today's
  existing entry is updated on POST; successful saves return to predictions.
  Changing assessment answers does not fabricate a different model probability.
- The complete automated suite passed all 152 tests. Django found no system
  issues; the dry-run migration check reported no changes.
- Eight rendered synthetic-user pages returned HTTP 200. All synthetic database
  writes were rolled back. Chrome screenshots were inspected at desktop and
  phone sizes; all eight pages had scroll width equal to the 390-pixel mobile
  viewport. Browser checks found no JavaScript errors and verified sidebar,
  chat open/close, and prediction filtering interactions.
- Browser visual checks used rendered HTML previews with local CSS/JavaScript
  embedded. They did not submit live forms; real authenticated form submissions,
  CSRF, account isolation, model persistence, and input changes are covered by
  the Django test suite. Browser chat submission and chart rendering were not
  independently validated by this preview check.

Screenshots and synthetic previews: artifacts/ui-review/.
Reproduction helpers: tools/review_ui.py and tools/review_mobile.ps1.
Full automated test output: review_test_results.log.

This review does not guarantee every prediction or examination marks.
No database schema changes were made.

---

# Historical review records (superseded where they describe active deployment)

# Project workflow and prediction check

Checked on 10 October 2026.

## Latest simple-input update

Larger-source training follow-up: with the user's explicit choice to keep current inputs only, 36 configurations were trained on the existing local BRFSS 2023 file. Four independently reloaded research artifacts reproduce test accuracies of 76.5%, 70.1%, 81.1% and 69.4%, with positive recall 50.4%, 64.7%, 49.8% and 49.5%. All-four above 80% was not achieved. The source lacks sleep/current-day observations and has different diagnosis targets, so these figures are not dashboard accuracy improvements; none was deployed. Thirteen training checks passed. See `ml/evaluation/CURRENT_INPUTS_LARGER_DATA_RESULTS.md`. Existing inputs, original data, user records and active v6 models remain unchanged.

Mental-only training follow-up: forty additional v9 configurations plus three ensembles were evaluated. The independently verified candidate achieved 74.9% test accuracy and 50.3% positive recall; it remains undeployed. No candidate met both 70% recall and 60% specificity in validation. An ineligible fallback was rejected, its executable artifact removed, and training now refuses to save candidates when no valid threshold exists. Nine training checks passed. See `ml/evaluation/MENTAL_WELLNESS_TRAINING_V9.md`.

Latest training: v8 added 128 configurations across three folds (384 validation fits) plus twelve ensemble comparisons. Independently verified test accuracies are diabetes 76.8%, hypertension 72.4%, cardiovascular 82.0% and mental screening 74.7%, with positive recall 47.0%, 80.3%, 48.4% and 51.7%. The all-four above 80% target remains unmet. Six training-method checks passed. Candidates were saved separately; v6 remains active because three v8 candidates detect fewer positive cases. Full details: `ml/evaluation/RETRAINING_V8_RESULTS.md`. The earlier v7 run is recorded in `ml/evaluation/RETRAINING_80_PERCENT_RESULTS.md`.

Follow-up: Wellness Coach is now a floating chatbot shared by the authenticated layouts. The sidebar and top navigation Coach links were removed, the main dashboard's third step opens Progress, and the Progress help button opens the chatbot. Replies use only the signed-in user's check-in for today through a POST endpoint protected by login and CSRF. Empty/overlong questions, expired sign-in, failed requests and reply timeouts are handled. The affected recommendations/dashboard suite passed all 46 tests. Live chatbot JavaScript returned HTTP 200. Browser interaction and JavaScript syntax were not independently executed (Node is unavailable).

Check-in now collects daily exercise without sitting time or seven-day exercise totals. Breakfast, lunch and dinner have separate labelled checkbox cards. Progress and Wellness Coach show today's habit values and simple actions, with habits needing attention first. Coach source explanations and seven-day pattern panels were removed; interactive question buttons remain.

The full 115-test run passed 114 tests. The remaining test had an incorrect expectation of `5.0 hours` when the coach correctly displays `5 hours`; after correcting that expectation, all four tests in its daily-action page module passed. The full run included the real registration, login, check-in, four predictions, progress, coach and logout flow. Django reported no issues. Visual browser appearance was not audited.

Active prediction models are now **lifestyle_v6**, trained without the removed inputs. Their independently verified held-out accuracies are diabetes 64.2%, hypertension 72.1%, cardiovascular 75.5%, and mental health 61.9%. The unweighted mean is 68.4%, not a combined project accuracy and not above 75%. Limited positive precision and lack of clinical validation remain significant limitations. See [current model details](ml/evaluation/LIFESTYLE_MODELS.md).

## Previous workflow and evaluation (historical v2)

Previous complete regression run: **108 tests passed**. The model metrics and feature list below describe the earlier v2 deployment, not the current models.

## Workflow

The automated end-to-end check uses real Django requests, CSRF protection,
and the deployed prediction models. It covers:

- Public home page, registration, sign-in and logout.
- Empty dashboard and missing-input prediction states.
- Saving and editing today's check-in without duplicate daily records.
- Four predictions on the main dashboard and Health Predictions page.
- Exact input units: sleep hours, exercise minutes over seven days,
  sitting hours converted to minutes, and BMI from saved height/weight.
- Changed inputs changing the real model outputs.
- BMI updates reaching the prediction inputs.
- Mental wellness and medication saves.
- Progress, profile, coach and history pages, and their internal links.
- Agreement between today's health-score card and graph.

Fixed the health-score graph to include each day's own saved screening results,
and to use refreshed results for today. Logout now returns to the public home
page instead of redirecting through a protected dashboard to sign-in.

The live public page and dashboard stylesheet both returned HTTP 200.
Django configuration and database migration checks passed.

## Independently reproduced prediction performance

`ml/evaluation/audit_saved_models.py` loads the deployed artifacts and repeats
the original held-out test split. It does not retrain or tune on test records.
All four artifacts reproduced their recorded metrics and feature order.

| Screening | Test records | Accuracy | ROC AUC | Positive precision | Positive recall |
|---|---:|---:|---:|---:|---:|
| Diabetes | 1,576 | 64.5% | 0.773 | 24.8% | 78.6% |
| Hypertension | 1,241 | 71.6% | 0.776 | 75.3% | 73.5% |
| Cardiovascular | 1,208 | 72.4% | 0.796 | 28.0% | 73.5% |
| Mental wellness | 1,091 | 70.2% | 0.717 | 24.0% | 57.2% |

These are measured performance figures, not guarantees for an individual.
Diabetes, cardiovascular and mental screening have substantial false positives.
Displayed percentages are model estimates, not independently validated clinical
probabilities. The targets represent diagnoses/questionnaire outcomes in the
source dataset; they do not establish future disease onset or daily disease
development. Daily updates mean recalculating the estimate from current inputs.

The deployed models use age, recorded sex category, BMI, smoking status,
seven-day activity, sitting time, and sleep where present in the source data.
Water, diet and mood contribute to lifestyle tracking, scores and guidance;
they are not additional inputs to these trained risk models. Mental wellness
ratings remain a separate assessment rather than a fabricated clinical scale.

Application tests confirm functionality; they do not prove clinical accuracy.
Browser appearance and accessibility have not been comprehensively audited.
