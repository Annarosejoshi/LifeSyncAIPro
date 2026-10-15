from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "ml"
    / "data"
    / "sleep_nhanes"
)

BPQ_PATH = DATA_DIR / "BPQ_L.xpt"
BP_PATH = DATA_DIR / "BPXO_L.xpt"


print("=" * 76)
print("LIFESYNC AI PRO - NHANES BPQ INSPECTION")
print("=" * 76)


# ============================================================
# LOAD
# ============================================================

bpq = pd.read_sas(
    BPQ_PATH,
    format="xport"
)

bp = pd.read_sas(
    BP_PATH,
    format="xport"
)


print("\nBPQ loaded successfully.")
print("BPQ rows:", len(bpq))
print("BPQ columns:", len(bpq.columns))

print("\nBP examination rows:", len(bp))


# ============================================================
# ALL BPQ COLUMNS
# ============================================================

print("\n" + "=" * 76)
print("BPQ COLUMNS")
print("=" * 76)

for column in bpq.columns:
    print(column)


# ============================================================
# IMPORTANT VARIABLES
# ============================================================

important = [
    "SEQN",
    "BPQ020",
    "BPQ030",
    "BPQ040A",
    "BPQ050A",
    "BPQ080",
    "BPQ090D",
    "BPQ100D",
    "BPQ150",
]


print("\n" + "=" * 76)
print("IMPORTANT BPQ VARIABLES")
print("=" * 76)

for column in important:

    if column not in bpq.columns:

        print(f"\nNOT FOUND: {column}")
        continue

    print(f"\nVARIABLE: {column}")

    print(
        bpq[column]
        .value_counts(
            dropna=False
        )
        .sort_index()
        .head(40)
    )

    print(
        "Missing:",
        bpq[column].isna().sum()
    )


# ============================================================
# BP AVERAGES
# ============================================================

systolic = [
    "BPXOSY1",
    "BPXOSY2",
    "BPXOSY3",
]

diastolic = [
    "BPXODI1",
    "BPXODI2",
    "BPXODI3",
]


bp["avg_systolic_bp"] = (
    bp[systolic]
    .mean(
        axis=1,
        skipna=True
    )
)

bp["avg_diastolic_bp"] = (
    bp[diastolic]
    .mean(
        axis=1,
        skipna=True
    )
)


# ============================================================
# MERGE
# ============================================================

merged = bpq.merge(
    bp[
        [
            "SEQN",
            "avg_systolic_bp",
            "avg_diastolic_bp",
        ]
    ],
    on="SEQN",
    how="inner",
)


print("\n" + "=" * 76)
print("BPQ + BP MERGE")
print("=" * 76)

print("BPQ rows:", len(bpq))
print("Rows after BP merge:", len(merged))

print(
    "Retention:",
    f"{len(merged) / len(bpq) * 100:.2f}%"
)


# ============================================================
# TARGET PREVIEW
# ============================================================

def yes_no(series):

    result = pd.Series(
        pd.NA,
        index=series.index,
        dtype="Int64"
    )

    result.loc[
        series == 1
    ] = 1

    result.loc[
        series == 2
    ] = 0

    return result


if "BPQ020" in merged.columns:

    merged["diagnosed_hypertension"] = (
        yes_no(
            merged["BPQ020"]
        )
    )

else:

    merged["diagnosed_hypertension"] = pd.NA


if "BPQ150" in merged.columns:

    merged["bp_medication"] = (
        yes_no(
            merged["BPQ150"]
        )
    )

else:

    merged["bp_medication"] = pd.NA


merged["exam_hypertension"] = pd.NA


valid_bp = (
    merged["avg_systolic_bp"].notna()
    &
    merged["avg_diastolic_bp"].notna()
)


merged.loc[
    valid_bp,
    "exam_hypertension"
] = (
    (
        merged.loc[
            valid_bp,
            "avg_systolic_bp"
        ] >= 130
    )
    |
    (
        merged.loc[
            valid_bp,
            "avg_diastolic_bp"
        ] >= 80
    )
).astype(int)


# Combined target preview:
#
# Positive if:
# diagnosed hypertension
# OR current BP medication
# OR examination BP meets threshold.

merged["hypertension_combined"] = pd.NA


positive = (
    (merged["diagnosed_hypertension"] == 1)
    |
    (merged["bp_medication"] == 1)
    |
    (merged["exam_hypertension"] == 1)
)


known_negative = (
    (merged["diagnosed_hypertension"] == 0)
    &
    (
        (
            merged["bp_medication"] == 0
        )
        |
        merged["bp_medication"].isna()
    )
    &
    (merged["exam_hypertension"] == 0)
)


merged.loc[
    positive,
    "hypertension_combined"
] = 1


merged.loc[
    known_negative,
    "hypertension_combined"
] = 0


print("\n" + "=" * 76)
print("HYPERTENSION TARGET PREVIEW")
print("=" * 76)


print("\nDiagnosed hypertension:")

print(
    merged[
        "diagnosed_hypertension"
    ]
    .value_counts(
        dropna=False
    )
)


print("\nBP medication:")

print(
    merged[
        "bp_medication"
    ]
    .value_counts(
        dropna=False
    )
)


print("\nExam hypertension:")

print(
    merged[
        "exam_hypertension"
    ]
    .value_counts(
        dropna=False
    )
)


print("\nCombined hypertension target:")

print(
    merged[
        "hypertension_combined"
    ]
    .value_counts(
        dropna=False
    )
)


valid_target = (
    merged[
        "hypertension_combined"
    ]
    .dropna()
)


print(
    "\nValid target records:",
    len(valid_target)
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


print("\n" + "=" * 76)
print("INSPECTION COMPLETE")
print("=" * 76)