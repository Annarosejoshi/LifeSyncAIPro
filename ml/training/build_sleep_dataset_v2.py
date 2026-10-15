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

SMOKING_PATH = DATA_DIR / "SMQ_L.xpt"
ALCOHOL_PATH = DATA_DIR / "ALQ_L.xpt"
OCCUPATION_PATH = DATA_DIR / "OCQ_L.xpt"


OUTPUT_PATH = (
    OUTPUT_DIR
    / "sleep_risk_dataset_v2.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_nhanes_zero(series):
    """
    NHANES XPT files can represent numeric zero
    as extremely tiny floating-point values.

    Convert values very close to zero back to 0.
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
    Decode PAQ frequency units.
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
    unit,
):
    """
    Convert activity frequency to
    approximate sessions per week.
    """

    if pd.isna(quantity):
        return np.nan

    if unit is None:
        return np.nan

    quantity = float(quantity)

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


def clean_binary_yes_no(series):
    """
    NHANES yes/no:
    1 = Yes
    2 = No

    Return:
    1 = Yes
    0 = No
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    result.loc[
        series == 1
    ] = 1

    result.loc[
        series == 2
    ] = 0

    return result


def clean_smoking_status(series):
    """
    SMQ040:
    1 = Every day
    2 = Some days
    3 = Not at all

    Convert to:
    2 = daily smoker
    1 = some-day smoker
    0 = not currently smoking
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    result.loc[
        series == 1
    ] = 2

    result.loc[
        series == 2
    ] = 1

    result.loc[
        series == 3
    ] = 0

    return result


def clean_alcohol_frequency(series):
    """
    ALQ121 / ALQ142 are ordinal
    frequency categories.

    Keep valid category codes 0-10.
    Convert 77/99 to missing.
    """

    series = normalize_nhanes_zero(
        series
    )

    series.loc[
        ~series.between(
            0,
            10
        )
    ] = np.nan

    return series


def clean_drinks_per_day(series):
    """
    ALQ130 average drinks on drinking days.

    Remove refused/don't-know codes and
    extreme special values.
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    series.loc[
        series.isin(
            [
                777,
                999,
            ]
        )
    ] = np.nan

    series.loc[
        (
            series < 0
        )
        |
        (
            series > 30
        )
    ] = np.nan

    return series


def clean_work_hours(series):
    """
    OCQ180 = hours worked last week.

    Remove NHANES special values.
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    series.loc[
        series.isin(
            [
                77777,
                99999,
            ]
        )
    ] = np.nan

    series.loc[
        (
            series < 0
        )
        |
        (
            series > 168
        )
    ] = np.nan

    return series


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 74)
print("LIFESYNC AI PRO - BUILD SLEEP RISK DATASET V2")
print("=" * 74)

print("\nLoading NHANES datasets...")

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

smoking_df = pd.read_sas(
    SMOKING_PATH,
    format="xport"
)

alcohol_df = pd.read_sas(
    ALCOHOL_PATH,
    format="xport"
)

occupation_df = pd.read_sas(
    OCCUPATION_PATH,
    format="xport"
)

print("All files loaded.")


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
    .merge(
        smoking_df,
        on="SEQN",
        how="inner"
    )
    .merge(
        alcohol_df,
        on="SEQN",
        how="inner"
    )
    .merge(
        occupation_df,
        on="SEQN",
        how="inner"
    )
)

print(
    "\nRows after 8-way merge:",
    len(df)
)


# ============================================================
# NORMALIZE ZERO VALUES
# ============================================================

numeric_columns = [
    "RIDAGEYR",
    "RIAGENDR",
    "INDFMPIR",
    "BMXBMI",

    "SLD012",

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

    "SMQ020",
    "SMQ040",

    "ALQ111",
    "ALQ121",
    "ALQ130",
    "ALQ142",

    "OCQ180",
]

for column in numeric_columns:

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

print(
    "Adult records:",
    len(df)
)


# ============================================================
# TARGET
# ============================================================

df["SLD012"] = pd.to_numeric(
    df["SLD012"],
    errors="coerce"
)

df.loc[
    ~df["SLD012"].between(
        2,
        14
    ),
    "SLD012"
] = np.nan


df["short_sleep"] = np.where(
    df["SLD012"].notna(),
    (
        df["SLD012"] < 7
    ).astype(int),
    np.nan
)


# ============================================================
# DEMOGRAPHICS / BODY
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
# PHYSICAL ACTIVITY
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
    for quantity, unit in zip(
        df["PAD790Q"],
        moderate_units
    )
]


df["vigorous_sessions_week"] = [
    frequency_to_weekly(
        quantity,
        unit
    )
    for quantity, unit in zip(
        df["PAD810Q"],
        vigorous_units
    )
]


for column in [
    "PAD800",
    "PAD820",
]:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

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


df["sedentary_minutes_day"] = (
    pd.to_numeric(
        df["PAD680"],
        errors="coerce"
    )
)

df.loc[
    (
        df[
            "sedentary_minutes_day"
        ] < 0
    )
    |
    (
        df[
            "sedentary_minutes_day"
        ] > 1440
    ),
    "sedentary_minutes_day"
] = np.nan


# ============================================================
# MENTAL WELLNESS
# ============================================================

mental_columns = [
    "DPQ010",
    "DPQ020",

    # DPQ030 excluded intentionally

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

    df.loc[
        ~df[column].between(
            0,
            3
        ),
        column
    ] = np.nan


df = df.rename(
    columns={
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
)


# ============================================================
# SMOKING
# ============================================================

df["ever_smoked_100"] = (
    clean_binary_yes_no(
        df["SMQ020"]
    )
)


df["current_smoking_status"] = (
    clean_smoking_status(
        df["SMQ040"]
    )
)


# Important:
# People who never smoked 100 cigarettes
# legitimately have no current-smoking follow-up.
# Treat them as current-smoking status = 0.

df.loc[
    (
        df[
            "ever_smoked_100"
        ] == 0
    )
    &
    (
        df[
            "current_smoking_status"
        ].isna()
    ),
    "current_smoking_status"
] = 0


# ============================================================
# ALCOHOL
# ============================================================

df["alcohol_past_year"] = (
    clean_binary_yes_no(
        df["ALQ111"]
    )
)


df["alcohol_frequency"] = (
    clean_alcohol_frequency(
        df["ALQ121"]
    )
)


df["average_drinks"] = (
    clean_drinks_per_day(
        df["ALQ130"]
    )
)


df["heavy_drinking_frequency"] = (
    clean_alcohol_frequency(
        df["ALQ142"]
    )
)


# Respondents reporting no alcohol use
# should logically receive zero exposure
# where follow-up variables are skipped.

no_alcohol_mask = (
    df[
        "alcohol_past_year"
    ] == 0
)


df.loc[
    no_alcohol_mask
    &
    df[
        "alcohol_frequency"
    ].isna(),
    "alcohol_frequency"
] = 0


df.loc[
    no_alcohol_mask
    &
    df[
        "average_drinks"
    ].isna(),
    "average_drinks"
] = 0


df.loc[
    no_alcohol_mask
    &
    df[
        "heavy_drinking_frequency"
    ].isna(),
    "heavy_drinking_frequency"
] = 0


# ============================================================
# OCCUPATION
# ============================================================

df["work_hours_week"] = (
    clean_work_hours(
        df["OCQ180"]
    )
)


# ============================================================
# DERIVED FEATURES
# ============================================================

# BMI categories
df["obesity_flag"] = np.where(
    df["bmi"].notna(),
    (
        df["bmi"] >= 30
    ).astype(int),
    np.nan
)


# High sedentary behavior
df["high_sedentary_flag"] = np.where(
    df[
        "sedentary_minutes_day"
    ].notna(),
    (
        df[
            "sedentary_minutes_day"
        ] >= 480
    ).astype(int),
    np.nan
)


# Long work week
df["long_work_week"] = np.where(
    df[
        "work_hours_week"
    ].notna(),
    (
        df[
            "work_hours_week"
        ] > 40
    ).astype(int),
    np.nan
)


# Mental symptom summary
mental_feature_names = [
    "little_interest",
    "depressed_mood",
    "low_energy",
    "appetite_problem",
    "negative_self_view",
    "concentration_problem",
    "movement_problem",
]


df["mental_symptom_score"] = (
    df[
        mental_feature_names
    ]
    .mean(
        axis=1,
        skipna=True
    )
)


# ============================================================
# FINAL FEATURES
# ============================================================

feature_columns = [
    "SEQN",

    # Demographics/body
    "age",
    "sex",
    "bmi",
    "income_ratio",
    "obesity_flag",

    # Activity
    "moderate_sessions_week",
    "moderate_minutes_session",

    "vigorous_sessions_week",
    "vigorous_minutes_session",

    "total_activity_minutes_week",
    "sedentary_minutes_day",
    "high_sedentary_flag",

    # Mental
    "little_interest",
    "depressed_mood",
    "low_energy",
    "appetite_problem",
    "negative_self_view",
    "concentration_problem",
    "movement_problem",
    "mental_symptom_score",

    # Smoking
    "ever_smoked_100",
    "current_smoking_status",

    # Alcohol
    "alcohol_past_year",
    "alcohol_frequency",
    "average_drinks",
    "heavy_drinking_frequency",

    # Occupation
    "work_hours_week",
    "long_work_week",

    # Audit target source
    "SLD012",

    # Target
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

duplicate_count = (
    final_df[
        "SEQN"
    ]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate participant IDs:",
    duplicate_count
)

if duplicate_count > 0:

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

print("\n" + "=" * 74)
print("FINAL SLEEP RISK DATASET V2")
print("=" * 74)


print(
    "\nRows:",
    len(final_df)
)

print(
    "Columns:",
    len(final_df.columns)
)


print(
    "\nTarget distribution:"
)

print(
    final_df[
        "short_sleep"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nTarget percentages:"
)

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


print(
    "\nMissing values:"
)

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


print(
    "\nSelected feature count:"
)

# Minus SEQN, SLD012 and target
print(
    len(feature_columns) - 3
)


print(
    "\nPreview:"
)

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


print("\n" + "=" * 74)
print("DATASET SAVED")
print("=" * 74)

print(
    OUTPUT_PATH
)