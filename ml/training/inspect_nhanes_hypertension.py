from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 76)
print("LIFESYNC AI PRO - NHANES HYPERTENSION TARGET INSPECTION")
print("=" * 76)

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "ml"
    / "data"
    / "sleep_nhanes"
)

BP_PATH = DATA_DIR / "BPXO_L.xpt"
MCQ_PATH = DATA_DIR / "MCQ_L.xpt"


# ============================================================
# LOAD BLOOD PRESSURE
# ============================================================

print("\nLoading Blood Pressure:")
print(BP_PATH)

bp = pd.read_sas(
    BP_PATH,
    format="xport"
)

print("\nLoaded successfully.")
print("Rows:", len(bp))
print("Columns:", len(bp.columns))


# ============================================================
# LOAD MEDICAL QUESTIONNAIRE
# ============================================================

print("\nLoading Medical Conditions:")
print(MCQ_PATH)

mcq = pd.read_sas(
    MCQ_PATH,
    format="xport"
)

print("\nLoaded successfully.")
print("Rows:", len(mcq))
print("Columns:", len(mcq.columns))


# ============================================================
# BLOOD PRESSURE VARIABLES
# ============================================================

print("\n" + "=" * 76)
print("BLOOD PRESSURE VARIABLES")
print("=" * 76)

bp_variables = [
    "BPXOSY1",
    "BPXODI1",
    "BPXOSY2",
    "BPXODI2",
    "BPXOSY3",
    "BPXODI3",
]


for variable in bp_variables:

    if variable in bp.columns:

        print(f"\n{variable}")

        print(
            bp[variable]
            .describe()
        )

        print(
            "Missing:",
            bp[variable].isna().sum()
        )


# ============================================================
# CALCULATE AVERAGE BP
# ============================================================

systolic_columns = [
    "BPXOSY1",
    "BPXOSY2",
    "BPXOSY3",
]

diastolic_columns = [
    "BPXODI1",
    "BPXODI2",
    "BPXODI3",
]


bp["avg_systolic_bp"] = (
    bp[systolic_columns]
    .mean(
        axis=1,
        skipna=True
    )
)


bp["avg_diastolic_bp"] = (
    bp[diastolic_columns]
    .mean(
        axis=1,
        skipna=True
    )
)


print("\n" + "=" * 76)
print("AVERAGE BLOOD PRESSURE")
print("=" * 76)

print("\nAverage systolic:")
print(
    bp["avg_systolic_bp"]
    .describe()
)

print("\nAverage diastolic:")
print(
    bp["avg_diastolic_bp"]
    .describe()
)


# ============================================================
# NUMBER OF AVAILABLE READINGS
# ============================================================

bp["systolic_readings"] = (
    bp[systolic_columns]
    .notna()
    .sum(axis=1)
)

bp["diastolic_readings"] = (
    bp[diastolic_columns]
    .notna()
    .sum(axis=1)
)


print("\n" + "=" * 76)
print("NUMBER OF BP READINGS")
print("=" * 76)

print("\nSystolic readings:")
print(
    bp["systolic_readings"]
    .value_counts()
    .sort_index()
)

print("\nDiastolic readings:")
print(
    bp["diastolic_readings"]
    .value_counts()
    .sort_index()
)


# ============================================================
# EXAM-BASED HYPERTENSION PREVIEW
#
# This is only a TARGET PREVIEW.
# We are NOT training yet.
# ============================================================

valid_bp = (
    bp["avg_systolic_bp"].notna()
    &
    bp["avg_diastolic_bp"].notna()
)


bp.loc[
    valid_bp,
    "exam_hypertension"
] = (
    (
        bp.loc[
            valid_bp,
            "avg_systolic_bp"
        ] >= 130
    )
    |
    (
        bp.loc[
            valid_bp,
            "avg_diastolic_bp"
        ] >= 80
    )
).astype(int)


print("\n" + "=" * 76)
print("EXAM-BASED HYPERTENSION PREVIEW")
print("=" * 76)

print(
    bp["exam_hypertension"]
    .value_counts(
        dropna=False
    )
)


valid_target = (
    bp["exam_hypertension"]
    .dropna()
)


print("\nPercentages:")

print(
    (
        valid_target
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# SEARCH MEDICAL QUESTIONNAIRE
# ============================================================

print("\n" + "=" * 76)
print("SEARCHING MCQ FOR BP / HYPERTENSION VARIABLES")
print("=" * 76)

keywords = [
    "BP",
    "HYPER",
    "PRESS",
]


possible_mcq = []

for column in mcq.columns:

    upper_column = column.upper()

    if any(
        keyword in upper_column
        for keyword in keywords
    ):

        possible_mcq.append(
            column
        )


print(
    "\nPossible variables found:",
    len(possible_mcq)
)


for column in possible_mcq:

    print(
        "\nVARIABLE:",
        column
    )

    print(
        mcq[column]
        .value_counts(
            dropna=False
        )
        .head(20)
    )


# ============================================================
# SEARCH ALL MCQ COLUMN NAMES
# ============================================================

print("\n" + "=" * 76)
print("ALL MCQ COLUMNS")
print("=" * 76)

for column in mcq.columns:

    print(column)


# ============================================================
# OVERLAP
# ============================================================

print("\n" + "=" * 76)
print("PARTICIPANT CHECK")
print("=" * 76)

print(
    "BP participants:",
    len(bp)
)

print(
    "Unique BP SEQN:",
    bp["SEQN"].nunique()
)

print(
    "Duplicate BP SEQN:",
    bp["SEQN"].duplicated().sum()
)


print("\n" + "=" * 76)
print("INSPECTION COMPLETE")
print("=" * 76)