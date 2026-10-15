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


DEMO_PATH = DATA_DIR / "DEMO_L.xpt"
BODY_PATH = DATA_DIR / "BMX_L.xpt"
BP_PATH = DATA_DIR / "BPXO_L.xpt"
TCHOL_PATH = DATA_DIR / "TCHOL_L.xpt"
HDL_PATH = DATA_DIR / "HDL_L.xpt"
SMOKING_PATH = DATA_DIR / "SMQ_L.xpt"
ACTIVITY_PATH = DATA_DIR / "PAQ_L.xpt"
MCQ_PATH = DATA_DIR / "MCQ_L.xpt"


OUTPUT_PATH = (
    OUTPUT_DIR
    / "cardiovascular_risk_dataset_v1.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_zero(series):
    """
    NHANES XPT sometimes stores numeric zero
    as extremely tiny positive values.
    """

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    return series.mask(
        series.abs() < 1e-10,
        0
    )


def clean_yes_no(series):
    """
    NHANES:
    1 = Yes
    2 = No
    7 = Refused
    9 = Don't know

    Return:
    1 = Yes
    0 = No
    NaN = invalid/missing
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
    2 = daily
    1 = some days
    0 = not currently
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


def decode_unit(value):

    if pd.isna(value):
        return None

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="ignore"
        )

    value = str(
        value
    ).strip().upper()

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
        return quantity * 12 / 52

    if unit == "Y":
        return quantity / 52

    return np.nan


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 74)
print("LIFESYNC AI PRO - BUILD CARDIOVASCULAR DATASET V1")
print("=" * 74)

print("\nLoading NHANES datasets...")


demo_df = pd.read_sas(
    DEMO_PATH,
    format="xport"
)

body_df = pd.read_sas(
    BODY_PATH,
    format="xport"
)

bp_df = pd.read_sas(
    BP_PATH,
    format="xport"
)

tchol_df = pd.read_sas(
    TCHOL_PATH,
    format="xport"
)

hdl_df = pd.read_sas(
    HDL_PATH,
    format="xport"
)

smoking_df = pd.read_sas(
    SMOKING_PATH,
    format="xport"
)

activity_df = pd.read_sas(
    ACTIVITY_PATH,
    format="xport"
)

mcq_df = pd.read_sas(
    MCQ_PATH,
    format="xport"
)


print("All files loaded successfully.")


# ============================================================
# BUILD TARGET
# ============================================================

cardio_columns = [
    "MCQ160B",  # Congestive heart failure
    "MCQ160C",  # Coronary heart disease
    "MCQ160D",  # Angina
    "MCQ160E",  # Heart attack
    "MCQ160F",  # Stroke
]


target_df = mcq_df[
    ["SEQN"] + cardio_columns
].copy()


for column in cardio_columns:

    target_df[column] = (
        pd.to_numeric(
            target_df[column],
            errors="coerce"
        )
    )

    target_df[column] = (
        target_df[column]
        .where(
            target_df[column]
            .isin([1, 2])
        )
    )


any_yes = (
    target_df[
        cardio_columns
    ] == 1
).any(
    axis=1
)

all_no = (
    target_df[
        cardio_columns
    ] == 2
).all(
    axis=1
)


target_df[
    "cardiovascular_disease"
] = np.nan


target_df.loc[
    any_yes,
    "cardiovascular_disease"
] = 1


target_df.loc[
    all_no,
    "cardiovascular_disease"
] = 0


target_df = target_df[
    target_df[
        "cardiovascular_disease"
    ].notna()
].copy()


target_df[
    "cardiovascular_disease"
] = (
    target_df[
        "cardiovascular_disease"
    ]
    .astype(int)
)


print(
    "\nValid cardiovascular target records:",
    len(target_df)
)


print(
    "\nOriginal target distribution:"
)

print(
    target_df[
        "cardiovascular_disease"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# BLOOD PRESSURE FEATURES
# ============================================================

bp_features = bp_df[
    [
        "SEQN",
        "BPXOSY1",
        "BPXODI1",
        "BPXOSY2",
        "BPXODI2",
        "BPXOSY3",
        "BPXODI3",
        "BPXOPLS1",
        "BPXOPLS2",
        "BPXOPLS3",
    ]
].copy()


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

pulse_columns = [
    "BPXOPLS1",
    "BPXOPLS2",
    "BPXOPLS3",
]


bp_features[
    "avg_systolic_bp"
] = (
    bp_features[
        systolic_columns
    ]
    .mean(
        axis=1,
        skipna=True
    )
)


bp_features[
    "avg_diastolic_bp"
] = (
    bp_features[
        diastolic_columns
    ]
    .mean(
        axis=1,
        skipna=True
    )
)


bp_features[
    "avg_pulse"
] = (
    bp_features[
        pulse_columns
    ]
    .mean(
        axis=1,
        skipna=True
    )
)


bp_features = bp_features[
    [
        "SEQN",
        "avg_systolic_bp",
        "avg_diastolic_bp",
        "avg_pulse",
    ]
]


# ============================================================
# DEMOGRAPHICS
# ============================================================

demo_features = demo_df[
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "INDFMPIR",
    ]
].copy()


demo_features = demo_features.rename(
    columns={
        "RIDAGEYR":
            "age",

        "RIAGENDR":
            "sex",

        "INDFMPIR":
            "income_ratio",
    }
)


for column in [
    "age",
    "sex",
    "income_ratio",
]:

    demo_features[column] = (
        normalize_zero(
            demo_features[column]
        )
    )


# ============================================================
# BODY
# ============================================================

body_features = body_df[
    [
        "SEQN",
        "BMXBMI",
    ]
].copy()


body_features = (
    body_features.rename(
        columns={
            "BMXBMI":
                "bmi"
        }
    )
)


body_features[
    "bmi"
] = normalize_zero(
    body_features["bmi"]
)


# ============================================================
# TOTAL CHOLESTEROL
# ============================================================

tchol_features = tchol_df[
    [
        "SEQN",
        "LBXTC",
    ]
].copy()


tchol_features = (
    tchol_features.rename(
        columns={
            "LBXTC":
                "total_cholesterol"
        }
    )
)


# ============================================================
# HDL
# ============================================================

hdl_features = hdl_df[
    [
        "SEQN",
        "LBDHDD",
    ]
].copy()


hdl_features = (
    hdl_features.rename(
        columns={
            "LBDHDD":
                "hdl_cholesterol"
        }
    )
)


# ============================================================
# SMOKING
# ============================================================

smoking_features = smoking_df[
    [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ]
].copy()


smoking_features[
    "ever_smoked_100"
] = (
    clean_yes_no(
        smoking_features[
            "SMQ020"
        ]
    )
)


smoking_features[
    "current_smoking_status"
] = (
    clean_smoking_status(
        smoking_features[
            "SMQ040"
        ]
    )
)


# If person never smoked 100 cigarettes,
# current-smoking follow-up is often skipped.

smoking_features.loc[
    (
        smoking_features[
            "ever_smoked_100"
        ] == 0
    )
    &
    (
        smoking_features[
            "current_smoking_status"
        ].isna()
    ),
    "current_smoking_status"
] = 0


smoking_features = smoking_features[
    [
        "SEQN",
        "ever_smoked_100",
        "current_smoking_status",
    ]
]


# ============================================================
# PHYSICAL ACTIVITY
# ============================================================

activity_features = activity_df[
    [
        "SEQN",
        "PAD790Q",
        "PAD790U",
        "PAD800",
        "PAD810Q",
        "PAD810U",
        "PAD820",
        "PAD680",
    ]
].copy()


for column in [
    "PAD790Q",
    "PAD800",
    "PAD810Q",
    "PAD820",
    "PAD680",
]:

    activity_features[column] = (
        normalize_zero(
            activity_features[column]
        )
    )


moderate_units = (
    activity_features[
        "PAD790U"
    ]
    .apply(
        decode_unit
    )
)


vigorous_units = (
    activity_features[
        "PAD810U"
    ]
    .apply(
        decode_unit
    )
)


activity_features[
    "moderate_sessions_week"
] = [
    frequency_to_weekly(
        quantity,
        unit,
    )
    for quantity, unit
    in zip(
        activity_features[
            "PAD790Q"
        ],
        moderate_units,
    )
]


activity_features[
    "vigorous_sessions_week"
] = [
    frequency_to_weekly(
        quantity,
        unit,
    )
    for quantity, unit
    in zip(
        activity_features[
            "PAD810Q"
        ],
        vigorous_units,
    )
]


for column in [
    "PAD800",
    "PAD820",
]:

    activity_features.loc[
        (
            activity_features[column] <= 0
        )
        |
        (
            activity_features[column] > 1440
        ),
        column
    ] = np.nan


activity_features[
    "moderate_minutes_session"
] = (
    activity_features[
        "PAD800"
    ]
)


activity_features[
    "vigorous_minutes_session"
] = (
    activity_features[
        "PAD820"
    ]
)


activity_features[
    "moderate_minutes_week"
] = (
    activity_features[
        "moderate_sessions_week"
    ]
    *
    activity_features[
        "moderate_minutes_session"
    ]
)


activity_features[
    "vigorous_minutes_week"
] = (
    activity_features[
        "vigorous_sessions_week"
    ]
    *
    activity_features[
        "vigorous_minutes_session"
    ]
)


activity_features[
    "total_activity_minutes_week"
] = (
    activity_features[
        [
            "moderate_minutes_week",
            "vigorous_minutes_week",
        ]
    ]
    .fillna(0)
    .sum(
        axis=1
    )
)


activity_features[
    "sedentary_minutes_day"
] = (
    activity_features[
        "PAD680"
    ]
)


activity_features.loc[
    (
        activity_features[
            "sedentary_minutes_day"
        ] < 0
    )
    |
    (
        activity_features[
            "sedentary_minutes_day"
        ] > 1440
    ),
    "sedentary_minutes_day"
] = np.nan


activity_features = (
    activity_features[
        [
            "SEQN",
            "moderate_sessions_week",
            "moderate_minutes_session",
            "vigorous_sessions_week",
            "vigorous_minutes_session",
            "total_activity_minutes_week",
            "sedentary_minutes_day",
        ]
    ]
)


# ============================================================
# MERGE EVERYTHING
# ============================================================

print("\n" + "=" * 74)
print("MERGE PROCESS")
print("=" * 74)


df = target_df[
    [
        "SEQN",
        "cardiovascular_disease",
    ]
].copy()


print(
    "\nStart target rows:",
    len(df)
)


merge_steps = [
    (
        "Demographics",
        demo_features,
    ),
    (
        "Body measures",
        body_features,
    ),
    (
        "Blood pressure",
        bp_features,
    ),
    (
        "Total cholesterol",
        tchol_features,
    ),
    (
        "HDL cholesterol",
        hdl_features,
    ),
    (
        "Smoking",
        smoking_features,
    ),
    (
        "Physical activity",
        activity_features,
    ),
]


for name, merge_df in merge_steps:

    before = len(df)

    df = df.merge(
        merge_df,
        on="SEQN",
        how="inner"
    )

    after = len(df)

    print(
        f"\n{name}:"
    )

    print(
        "Before:",
        before
    )

    print(
        "After:",
        after
    )

    print(
        "Lost:",
        before - after
    )


# ============================================================
# ADULTS ONLY
# ============================================================

df = df[
    df["age"] >= 18
].copy()


print(
    "\nAdult records after merge:",
    len(df)
)


# ============================================================
# DERIVED FEATURES
# ============================================================

df[
    "obesity_flag"
] = np.where(
    df["bmi"].notna(),
    (
        df["bmi"] >= 30
    ).astype(int),
    np.nan
)


df[
    "high_systolic_flag"
] = np.where(
    df[
        "avg_systolic_bp"
    ].notna(),
    (
        df[
            "avg_systolic_bp"
        ] >= 130
    ).astype(int),
    np.nan
)


df[
    "high_diastolic_flag"
] = np.where(
    df[
        "avg_diastolic_bp"
    ].notna(),
    (
        df[
            "avg_diastolic_bp"
        ] >= 80
    ).astype(int),
    np.nan
)


df[
    "high_total_cholesterol_flag"
] = np.where(
    df[
        "total_cholesterol"
    ].notna(),
    (
        df[
            "total_cholesterol"
        ] >= 200
    ).astype(int),
    np.nan
)


df[
    "low_hdl_flag"
] = np.where(
    (
        df[
            "hdl_cholesterol"
        ].notna()
    )
    &
    (
        df[
            "sex"
        ].notna()
    ),
    np.where(
        df[
            "sex"
        ] == 1,
        (
            df[
                "hdl_cholesterol"
            ] < 40
        ).astype(int),
        (
            df[
                "hdl_cholesterol"
            ] < 50
        ).astype(int),
    ),
    np.nan
)


# ============================================================
# FINAL COLUMN SET
# ============================================================

feature_columns = [
    "SEQN",

    "age",
    "sex",
    "income_ratio",

    "bmi",
    "obesity_flag",

    "avg_systolic_bp",
    "avg_diastolic_bp",
    "avg_pulse",

    "high_systolic_flag",
    "high_diastolic_flag",

    "total_cholesterol",
    "hdl_cholesterol",

    "high_total_cholesterol_flag",
    "low_hdl_flag",

    "ever_smoked_100",
    "current_smoking_status",

    "moderate_sessions_week",
    "moderate_minutes_session",

    "vigorous_sessions_week",
    "vigorous_minutes_session",

    "total_activity_minutes_week",
    "sedentary_minutes_day",

    "cardiovascular_disease",
]


final_df = df[
    feature_columns
].copy()


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_count = (
    final_df[
        "SEQN"
    ]
    .duplicated()
    .sum()
)


print(
    "\nDuplicate SEQN:",
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
# TARGET REPORT
# ============================================================

print("\n" + "=" * 74)
print("FINAL CARDIOVASCULAR DATASET V1")
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
        "cardiovascular_disease"
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
            "cardiovascular_disease"
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )
    .round(2)
)


# ============================================================
# MISSING VALUES
# ============================================================

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


# ============================================================
# POSITIVE CASE COUNT
# ============================================================

positive_count = (
    final_df[
        "cardiovascular_disease"
    ] == 1
).sum()


print(
    "\nPositive cardiovascular cases:",
    positive_count
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