# LifeSync AI: Lifestyle Tracking and Health Risk Screening

LifeSync AI is a web-based application that combines daily lifestyle tracking,
machine-learning risk screening, and self-reported mental wellness assessment
in a single dashboard. Developed using Python, Django, SQLite, and
scikit-learn, the system provides authenticated user accounts, health profiles,
BMI calculation, daily check-ins, medication tracking, interactive progress
charts, and personalized lifestyle guidance.

Four deployed models estimate diabetes, hypertension, cardiovascular, and
depression-related screening risk from profile measurements and sleep
information where supported. Preprocessing and model selection use three-fold
stratified cross-validation within the training partition, with thresholds
selected before evaluating a separate internal test partition. The models
achieved test accuracies of 77.0%, 73.1%, 81.5%, and 74.6%, respectively,
giving an unweighted mean accuracy of 76.5% across the four tasks. Positive-case
detection was 48.8%, 82.4%, 47.1%, and 50.3%, respectively, and is reported
alongside accuracy to describe performance on imbalanced data.

A separate eight-item mental wellness assessment produces a daily score that
updates when users revise their answers; this score is not a model probability
or a measure of prediction accuracy. The interface presents saved inputs,
screening results, and progress interactively, with validation for missing or
invalid data. Application tests verify account access, data persistence,
prediction-input mapping, and agreement with saved model artifacts. The
system supports lifestyle awareness and screening rather than diagnosis.
Results are based on a repeatedly evaluated internal test partition and do
not establish independent external or clinical validation.

**Keywords:** Django, lifestyle tracking, machine learning, health screening,
mental wellness, cross-validation.
