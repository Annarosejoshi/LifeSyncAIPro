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

OUTPUT_DIR = (
    BASE_DIR
    / "ml"
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SLEEP_PATH = DATA_DIR / "SLQ_L.xpt"
DEMO_PATH = DATA_DIR / "DEMO_L.xpt"
BODY_PATH = DATA_DIR / "BMX_L.xpt"
ACTIVITY_PATH = DATA_DIR / "PAQ_L.xpt"
MENTAL_PATH = DATA_DIR / "DPQ_L.xpt"

OUTPUT_PATH = (
    OUTPUT_DIR
    / "sleep_risk_dataset.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_nhanes_zero(series):
    """
    XPT files sometimes represent numeric zero
    as a tiny positive floating-point number such as
    5.397605e-79.

    Convert values extremely close to zero back to 0.
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    return series.mask(
        series.abs() < 1e-10,
        0
    )


def decode_unit(value):
    """
    Decode byte/string PAQ frequency units.
    """

    if pd.isna(value):
        return None

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="ignore"
        )

    value = str(value).strip().upper()

    if value in {
        "D",
        "W",
        "M",
        "Y",
    }:
        return value

    return None


def frequency_to_weekly(
    quantity,
    unit
):
    """
    Convert physical-activity frequency
    into approximate sessions per week.

    D = per day
    W = per week
    M = per month
    Y = per year
    """

    if pd.isna(quantity):
        return np.nan

    if unit is None:
        return np.nan

    quantity = float(quantity)

    # NHANES refused / don't know codes
    if quantity in {
        7777,
        9999,
    }:
        return np.nan

    if quantity < 0:
        return np.nan

    if unit == "D":
        return quantity * 7

    if unit == "W":
        return quantity

    if unit == "M":
        return (
            quantity
            * 12
            / 52
        )

    if unit == "Y":
        return (
            quantity
            / 52
        )

    return np.nan


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 72)
print("LIFESYNC AI PRO - BUILD SLEEP RISK DATASET")
print("=" * 72)

print("\nLoading NHANES files...")

sleep_df = pd.read_sas(
    SLEEP_PATH,
    format="xport"
)

demo_df = pd.read_sas(
    DEMO_PATH,
    format="xport"
)

body_df = pd.read_sas(
    BODY_PATH,
    format="xport"
)

activity_df = pd.read_sas(
    ACTIVITY_PATH,
    format="xport"
)

mental_df = pd.read_sas(
    MENTAL_PATH,
    format="xport"
)

print("Files loaded successfully.")


# ============================================================
# MERGE
# ============================================================

df = (
    sleep_df
    .merge(
        demo_df,
        on="SEQN",
        how="inner"
    )
    .merge(
        body_df,
        on="SEQN",
        how="inner"
    )
    .merge(
        activity_df,
        on="SEQN",
        how="inner"
    )
    .merge(
        mental_df,
        on="SEQN",
        how="inner"
    )
)

print("\nRows after 5-way merge:")
print(len(df))


# ============================================================
# NORMALIZE NUMERIC ZERO VALUES
# ============================================================

numeric_candidates = [
    "RIDAGEYR",
    "RIAGENDR",
    "INDFMPIR",
    "BMXBMI",

    "SLD012",
    "SLD013",

    "PAD790Q",
    "PAD800",

    "PAD810Q",
    "PAD820",

    "PAD680",

    "DPQ010",
    "DPQ020",
    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
]

for column in numeric_candidates:

    if column in df.columns:

        df[column] = (
            normalize_nhanes_zero(
                df[column]
            )
        )


# ============================================================
# ADULT SAMPLE
# ============================================================

df = df[
    df["RIDAGEYR"] >= 18
].copy()

print("\nAdult records:")
print(len(df))


# ============================================================
# CLEAN SLEEP TARGET VARIABLE
# ============================================================

df["SLD012"] = pd.to_numeric(
    df["SLD012"],
    errors="coerce"
)

# Keep plausible reported sleep duration
df.loc[
    ~df["SLD012"].between(
        2,
        14
    ),
    "SLD012"
] = np.nan


# ============================================================
# TARGET
# ============================================================

# NHANES SLD012 is weekday/workday sleep duration.
# Short sleep target:
# 1 = less than 7 hours
# 0 = 7 hours or more

df["short_sleep"] = np.where(
    df["SLD012"].notna(),
    (
        df["SLD012"] < 7
    ).astype(int),
    np.nan
)


# ============================================================
# BASIC DEMOGRAPHIC FEATURES
# ============================================================

df["age"] = df[
    "RIDAGEYR"
]

df["sex"] = df[
    "RIAGENDR"
]

df["bmi"] = df[
    "BMXBMI"
]

df["income_ratio"] = df[
    "INDFMPIR"
]


# ============================================================
# ACTIVITY FEATURES
# ============================================================

moderate_units = (
    df["PAD790U"]
    .apply(
        decode_unit
    )
)

vigorous_units = (
    df["PAD810U"]
    .apply(
        decode_unit
    )
)


df["moderate_sessions_week"] = [
    frequency_to_weekly(
        quantity,
        unit
    )
    for quantity, unit
    in zip(
        df["PAD790Q"],
        moderate_units
    )
]


df["vigorous_sessions_week"] = [
    frequency_to_weekly(
        quantity,
        unit
    )
    for quantity, unit
    in zip(
        df["PAD810Q"],
        vigorous_units
    )
]


# Clean session duration
for column in [
    "PAD800",
    "PAD820",
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # Reject unrealistic/special values
    df.loc[
        (
            df[column] <= 0
        )
        |
        (
            df[column] > 1440
        ),
        column
    ] = np.nan


df["moderate_minutes_session"] = (
    df["PAD800"]
)

df["vigorous_minutes_session"] = (
    df["PAD820"]
)


# Weekly exercise minutes
df["moderate_minutes_week"] = (
    df[
        "moderate_sessions_week"
    ]
    *
    df[
        "moderate_minutes_session"
    ]
)

df["vigorous_minutes_week"] = (
    df[
        "vigorous_sessions_week"
    ]
    *
    df[
        "vigorous_minutes_session"
    ]
)


df["total_activity_minutes_week"] = (
    df[
        [
            "moderate_minutes_week",
            "vigorous_minutes_week",
        ]
    ]
    .fillna(0)
    .sum(axis=1)
)


# Sedentary time
df["sedentary_minutes_day"] = (
    pd.to_numeric(
        df["PAD680"],
        errors="coerce"
    )
)

df.loc[
    (
        df["sedentary_minutes_day"]
        < 0
    )
    |
    (
        df["sedentary_minutes_day"]
        > 1440
    ),
    "sedentary_minutes_day"
] = np.nan


# ============================================================
# MENTAL WELLNESS FEATURES
# ============================================================

mental_columns = [
    "DPQ010",
    "DPQ020",

    # DPQ030 intentionally excluded
    # because it directly asks about sleep.

    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
]


for column in mental_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    # Valid PHQ response range = 0 to 3.
    # 7/9 etc. become missing.

    df.loc[
        ~df[column].between(
            0,
            3
        ),
        column
    ] = np.nan


# Rename mental variables
rename_map = {
    "DPQ010":
        "little_interest",

    "DPQ020":
        "depressed_mood",

    "DPQ040":
        "low_energy",

    "DPQ050":
        "appetite_problem",

    "DPQ060":
        "negative_self_view",

    "DPQ070":
        "concentration_problem",

    "DPQ080":
        "movement_problem",
}

df = df.rename(
    columns=rename_map
)


# ============================================================
# SELECT FINAL FEATURES
# ============================================================

feature_columns = [
    "SEQN",

    "age",
    "sex",
    "bmi",
    "income_ratio",

    "moderate_sessions_week",
    "moderate_minutes_session",

    "vigorous_sessions_week",
    "vigorous_minutes_session",

    "total_activity_minutes_week",

    "sedentary_minutes_day",

    "little_interest",
    "depressed_mood",
    "low_energy",
    "appetite_problem",
    "negative_self_view",
    "concentration_problem",
    "movement_problem",

    "SLD012",
    "short_sleep",
]


final_df = df[
    feature_columns
].copy()


# ============================================================
# REMOVE TARGET-MISSING RECORDS
# ============================================================

final_df = final_df[
    final_df[
        "short_sleep"
    ].notna()
].copy()

final_df[
    "short_sleep"
] = (
    final_df[
        "short_sleep"
    ]
    .astype(int)
)


# ============================================================
# DUPLICATES
# ============================================================

duplicate_seqn = (
    final_df[
        "SEQN"
    ]
    .duplicated()
    .sum()
)

print("\nDuplicate participant IDs:")
print(duplicate_seqn)

if duplicate_seqn > 0:

    final_df = (
        final_df
        .drop_duplicates(
            subset="SEQN",
            keep="first"
        )
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 72)
print("FINAL DATASET")
print("=" * 72)

print(
    "Rows:",
    len(final_df)
)

print(
    "Columns:",
    len(final_df.columns)
)


print("\nTarget distribution:")

target_counts = (
    final_df[
        "short_sleep"
    ]
    .value_counts()
    .sort_index()
)

print(target_counts)


print("\nTarget percentages:")

print(
    (
        final_df[
            "short_sleep"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )
    .round(2)
)


print("\nMissing values:")

missing = (
    final_df
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)

print(
    missing[
        missing > 0
    ]
)


print("\nDataset preview:")

print(
    final_df.head()
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 72)
print("SAVED")
print("=" * 72)

print(
    OUTPUT_PATH
)