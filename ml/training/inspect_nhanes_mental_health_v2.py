from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "ml"
    / "data"
    / "sleep_nhanes"
)

DPQ_PATH = DATA_DIR / "DPQ_L.xpt"


# ============================================================
# START
# ============================================================

print("=" * 78)
print("LIFESYNC AI PRO - NHANES MENTAL HEALTH TARGET INSPECTION")
print("=" * 78)


# ============================================================
# LOAD DPQ
# ============================================================

print("\nLoading:")
print(DPQ_PATH)

if not DPQ_PATH.exists():

    raise FileNotFoundError(
        f"File not found:\n{DPQ_PATH}"
    )


dpq = pd.read_sas(
    DPQ_PATH,
    format="xport",
)


print("\nDataset loaded successfully.")

print(
    "Rows:",
    len(dpq)
)

print(
    "Columns:",
    len(dpq.columns)
)


# ============================================================
# COLUMNS
# ============================================================

print("\n" + "=" * 78)
print("DPQ COLUMNS")
print("=" * 78)

for column in dpq.columns:

    print(column)


# ============================================================
# PHQ-9 ITEMS
# ============================================================

PHQ9_ITEMS = [
    "DPQ010",
    "DPQ020",
    "DPQ030",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
    "DPQ090",
]


# ============================================================
# DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 78)
print("PHQ-9 ITEM DISTRIBUTIONS")
print("=" * 78)


for column in PHQ9_ITEMS:

    print(
        f"\nVARIABLE: {column}"
    )

    if column not in dpq.columns:

        print("NOT FOUND")
        continue

    print(
        dpq[column]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "Missing:",
        dpq[column]
        .isna()
        .sum()
    )


# ============================================================
# CLEAN SPECIAL CODES
#
# Valid DPQ responses:
# 0 = Not at all
# 1 = Several days
# 2 = More than half the days
# 3 = Nearly every day
#
# 7 / 9 are non-response codes.
# ============================================================

mental = dpq[
    ["SEQN"] + PHQ9_ITEMS
].copy()


for column in PHQ9_ITEMS:

    mental.loc[
        mental[column].isin(
            [7, 9]
        ),
        column,
    ] = np.nan


# ============================================================
# WEIRD SAS ZERO CHECK
#
# Sometimes XPT zeros appear as extremely tiny values such as
# 5.397605e-79. Convert those to true 0.
# ============================================================

for column in PHQ9_ITEMS:

    mental.loc[
        mental[column].abs()
        < 1e-10,
        column,
    ] = 0


# ============================================================
# COMPLETE PHQ-9 RECORDS
# ============================================================

mental[
    "complete_phq9"
] = (
    mental[
        PHQ9_ITEMS
    ]
    .notna()
    .all(
        axis=1
    )
)


print("\n" + "=" * 78)
print("PHQ-9 COMPLETENESS")
print("=" * 78)

print(
    "Total participants:",
    len(mental)
)

print(
    "Complete PHQ-9:",
    mental[
        "complete_phq9"
    ].sum()
)

print(
    "Incomplete PHQ-9:",
    (
        ~mental[
            "complete_phq9"
        ]
    ).sum()
)


# ============================================================
# PHQ-9 SCORE
# ============================================================

mental[
    "phq9_score"
] = (
    mental[
        PHQ9_ITEMS
    ]
    .sum(
        axis=1,
        min_count=len(
            PHQ9_ITEMS
        ),
    )
)


print("\n" + "=" * 78)
print("PHQ-9 SCORE STATISTICS")
print("=" * 78)

print(
    mental[
        "phq9_score"
    ].describe()
)


# ============================================================
# SEVERITY CATEGORY
# ============================================================

def phq9_category(
    score
):

    if pd.isna(score):
        return np.nan

    if score <= 4:
        return "Minimal"

    if score <= 9:
        return "Mild"

    if score <= 14:
        return "Moderate"

    if score <= 19:
        return "Moderately Severe"

    return "Severe"


mental[
    "phq9_category"
] = (
    mental[
        "phq9_score"
    ]
    .apply(
        phq9_category
    )
)


print("\n" + "=" * 78)
print("PHQ-9 SEVERITY DISTRIBUTION")
print("=" * 78)

print(
    mental[
        "phq9_category"
    ]
    .value_counts(
        dropna=False
    )
)


print("\nPercentages:")

valid_categories = (
    mental[
        "phq9_category"
    ]
    .dropna()
)

print(
    (
        valid_categories
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# BINARY TARGET OPTION 1
#
# Clinically relevant depressive symptom burden:
# PHQ-9 >= 10
# ============================================================

mental[
    "depression_risk"
] = np.where(
    mental[
        "phq9_score"
    ].notna(),
    (
        mental[
            "phq9_score"
        ]
        >= 10
    ).astype(int),
    np.nan,
)


print("\n" + "=" * 78)
print("BINARY TARGET: PHQ-9 >= 10")
print("=" * 78)

print(
    mental[
        "depression_risk"
    ]
    .value_counts(
        dropna=False
    )
)


valid_target = (
    mental[
        "depression_risk"
    ]
    .dropna()
)


print("\nValid target records:")
print(
    len(
        valid_target
    )
)


print("\nTarget percentages:")

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
# ALTERNATIVE TARGET
#
# Any meaningful symptoms:
# PHQ-9 >= 5
#
# We inspect this only. Do NOT choose it yet.
# ============================================================

mental[
    "any_depressive_symptoms"
] = np.where(
    mental[
        "phq9_score"
    ].notna(),
    (
        mental[
            "phq9_score"
        ]
        >= 5
    ).astype(int),
    np.nan,
)


print("\n" + "=" * 78)
print("ALTERNATIVE TARGET: PHQ-9 >= 5")
print("=" * 78)

print(
    mental[
        "any_depressive_symptoms"
    ]
    .value_counts(
        dropna=False
    )
)


valid_alt = (
    mental[
        "any_depressive_symptoms"
    ]
    .dropna()
)


print("\nPercentages:")

print(
    (
        valid_alt
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ============================================================
# SELF-HARM ITEM CHECK
#
# DPQ090 is part of PHQ-9, but we inspect it separately.
# We will NOT expose this as a standalone diagnosis.
# ============================================================

print("\n" + "=" * 78)
print("DPQ090 DISTRIBUTION CHECK")
print("=" * 78)

print(
    mental[
        "DPQ090"
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 78)
print("PARTICIPANT CHECK")
print("=" * 78)

print(
    "Participants:",
    len(mental)
)

print(
    "Unique SEQN:",
    mental[
        "SEQN"
    ].nunique()
)

print(
    "Duplicate SEQN:",
    mental[
        "SEQN"
    ]
    .duplicated()
    .sum()
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("INSPECTION COMPLETE")
print("=" * 78)

print(
    "\nRecommended target candidate:"
)

print(
    "depression_risk = 1 when PHQ-9 >= 10"
)

print(
    "\nDo NOT use DPQ010-DPQ090 as predictors "
    "when this score is the target."
)