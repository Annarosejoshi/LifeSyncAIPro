from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "ml" / "data" / "sleep_nhanes"
OUTPUT_DIR = BASE_DIR / "ml" / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "diabetes_risk_dataset_v1.csv"

print("=" * 78)
print("LIFESYNC AI PRO - BUILD DIABETES DATASET V1")
print("=" * 78)


# ============================================================
# LOAD FUNCTION
# ============================================================

def load_xpt(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_sas(path, format="xport")


print("\nLoading NHANES datasets...")

diq = load_xpt("DIQ_L.xpt")
demo = load_xpt("DEMO_L.xpt")
body = load_xpt("BMX_L.xpt")
activity = load_xpt("PAQ_L.xpt")
smoking = load_xpt("SMQ_L.xpt")
sleep = load_xpt("SLQ_L.xpt")
tchol = load_xpt("TCHOL_L.xpt")
hdl = load_xpt("HDL_L.xpt")
alcohol = load_xpt("ALQ_L.xpt")
mental = load_xpt("DPQ_L.xpt")

print("All files loaded successfully.")


# ============================================================
# DIABETES TARGET
# ============================================================

target = diq[
    [
        "SEQN",
        "DIQ010",
    ]
].copy()

target["diabetes"] = np.nan

# Doctor told participant they have diabetes
target.loc[
    target["DIQ010"] == 1,
    "diabetes"
] = 1

# Doctor told participant they do NOT have diabetes
target.loc[
    target["DIQ010"] == 2,
    "diabetes"
] = 0

# DIQ010 = 3 (borderline) is intentionally excluded.
# Refused / don't know / missing are also excluded.

print("\nInitial diabetes target:")
print(target["diabetes"].value_counts(dropna=False))

target = target.dropna(
    subset=["diabetes"]
).copy()

target["diabetes"] = target["diabetes"].astype(int)

print("\nValid diabetes target records:")
print(len(target))

print("\nTarget distribution:")
print(target["diabetes"].value_counts())

print("\nTarget percentages:")
print(
    (
        target["diabetes"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# DEMOGRAPHICS
# ============================================================

demo_features = demo[
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "INDFMPIR",
    ]
].copy()

demo_features = demo_features.rename(
    columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "sex",
        "INDFMPIR": "income_ratio",
    }
)


# ============================================================
# BODY MEASURES
# ============================================================

body_features = body[
    [
        "SEQN",
        "BMXBMI",
        "BMXWAIST",
    ]
].copy()

body_features = body_features.rename(
    columns={
        "BMXBMI": "bmi",
        "BMXWAIST": "waist_circumference",
    }
)

body_features["obesity_flag"] = np.where(
    body_features["bmi"].notna(),
    (body_features["bmi"] >= 30).astype(float),
    np.nan,
)


# ============================================================
# SMOKING
# ============================================================

smoking_features = smoking[
    [
        "SEQN",
        "SMQ020",
        "SMQ040",
    ]
].copy()

# Ever smoked at least 100 cigarettes
smoking_features["ever_smoked_100"] = np.select(
    [
        smoking_features["SMQ020"] == 1,
        smoking_features["SMQ020"] == 2,
    ],
    [
        1.0,
        0.0,
    ],
    default=np.nan,
)

# Smoking status:
# 0 = never smoker
# 1 = former smoker
# 2 = current smoker

smoking_features["smoking_status"] = np.nan

smoking_features.loc[
    smoking_features["SMQ020"] == 2,
    "smoking_status"
] = 0

smoking_features.loc[
    (smoking_features["SMQ020"] == 1)
    & (smoking_features["SMQ040"] == 3),
    "smoking_status"
] = 1

smoking_features.loc[
    (smoking_features["SMQ020"] == 1)
    & (smoking_features["SMQ040"].isin([1, 2])),
    "smoking_status"
] = 2

smoking_features["current_smoker"] = np.where(
    smoking_features["smoking_status"].notna(),
    (smoking_features["smoking_status"] == 2).astype(float),
    np.nan,
)

smoking_features = smoking_features[
    [
        "SEQN",
        "ever_smoked_100",
        "smoking_status",
        "current_smoker",
    ]
]


# ============================================================
# PHYSICAL ACTIVITY
# ============================================================

activity_features = activity[
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


def decode_unit(value):

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="ignore"
        ).strip()

    if pd.isna(value):
        return ""

    return str(value).strip()


def sessions_per_week(quantity, unit):

    if pd.isna(quantity):
        return np.nan

    if quantity in [7777, 9999]:
        return np.nan

    unit = decode_unit(unit)

    if unit == "D":
        return quantity * 7

    if unit == "W":
        return quantity

    if unit == "M":
        return quantity / 4.345

    if unit == "Y":
        return quantity / 52.0

    # blank unit commonly corresponds to zero reported activity
    if unit == "" and quantity < 1:
        return 0.0

    return np.nan


activity_features["moderate_sessions_week"] = (
    activity_features.apply(
        lambda row: sessions_per_week(
            row["PAD790Q"],
            row["PAD790U"],
        ),
        axis=1,
    )
)

activity_features["vigorous_sessions_week"] = (
    activity_features.apply(
        lambda row: sessions_per_week(
            row["PAD810Q"],
            row["PAD810U"],
        ),
        axis=1,
    )
)

activity_features[
    "moderate_minutes_session"
] = activity_features["PAD800"]

activity_features[
    "vigorous_minutes_session"
] = activity_features["PAD820"]

# Remove obvious questionnaire special codes / implausible values
for column in [
    "moderate_minutes_session",
    "vigorous_minutes_session",
]:

    activity_features.loc[
        (
            activity_features[column] <= 0
        )
        | (
            activity_features[column] > 1440
        ),
        column,
    ] = np.nan


activity_features["total_activity_minutes_week"] = (
    activity_features[
        "moderate_sessions_week"
    ].fillna(0)
    *
    activity_features[
        "moderate_minutes_session"
    ].fillna(0)
    +
    activity_features[
        "vigorous_sessions_week"
    ].fillna(0)
    *
    activity_features[
        "vigorous_minutes_session"
    ].fillna(0)
)

activity_features[
    "sedentary_minutes_day"
] = activity_features["PAD680"]

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
    "sedentary_minutes_day",
] = np.nan


activity_features = activity_features[
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


# ============================================================
# SLEEP
# ============================================================

sleep_features = sleep[
    [
        "SEQN",
        "SLD012",
    ]
].copy()

sleep_features = sleep_features.rename(
    columns={
        "SLD012": "sleep_hours",
    }
)

sleep_features.loc[
    (
        sleep_features["sleep_hours"] < 2
    )
    |
    (
        sleep_features["sleep_hours"] > 14
    ),
    "sleep_hours",
] = np.nan

sleep_features["short_sleep_flag"] = np.where(
    sleep_features["sleep_hours"].notna(),
    (sleep_features["sleep_hours"] < 7).astype(float),
    np.nan,
)


# ============================================================
# TOTAL CHOLESTEROL
# ============================================================

chol_features = tchol[
    [
        "SEQN",
        "LBXTC",
    ]
].copy()

chol_features = chol_features.rename(
    columns={
        "LBXTC": "total_cholesterol",
    }
)


# ============================================================
# HDL
# ============================================================

hdl_features = hdl[
    [
        "SEQN",
        "LBDHDD",
    ]
].copy()

hdl_features = hdl_features.rename(
    columns={
        "LBDHDD": "hdl_cholesterol",
    }
)


# ============================================================
# ALCOHOL
# ============================================================

alcohol_features = alcohol[
    [
        "SEQN",
        "ALQ111",
        "ALQ121",
        "ALQ130",
    ]
].copy()


alcohol_features["alcohol_past_year"] = np.select(
    [
        alcohol_features["ALQ111"] == 1,
        alcohol_features["ALQ111"] == 2,
    ],
    [
        1.0,
        0.0,
    ],
    default=np.nan,
)

alcohol_features["alcohol_frequency"] = (
    alcohol_features["ALQ121"]
)

alcohol_features.loc[
    alcohol_features[
        "alcohol_frequency"
    ].isin([77, 99]),
    "alcohol_frequency",
] = np.nan


alcohol_features["average_drinks"] = (
    alcohol_features["ALQ130"]
)

alcohol_features.loc[
    alcohol_features[
        "average_drinks"
    ].isin([777, 999]),
    "average_drinks",
] = np.nan


alcohol_features = alcohol_features[
    [
        "SEQN",
        "alcohol_past_year",
        "alcohol_frequency",
        "average_drinks",
    ]
]


# ============================================================
# MENTAL HEALTH
# ============================================================

dpq_columns = [
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

mental_features = mental[
    ["SEQN"] + dpq_columns
].copy()

for column in dpq_columns:

    mental_features.loc[
        mental_features[column].isin([7, 9]),
        column,
    ] = np.nan


mental_features[
    "mental_symptom_score"
] = mental_features[
    dpq_columns
].sum(
    axis=1,
    min_count=len(dpq_columns),
)


mental_features = mental_features[
    [
        "SEQN",
        "mental_symptom_score",
    ]
]


# ============================================================
# MERGE
# ============================================================

df = target[
    [
        "SEQN",
        "diabetes",
    ]
].copy()


print("\n" + "=" * 78)
print("MERGE PROCESS")
print("=" * 78)


datasets = [
    ("Demographics", demo_features),
    ("Body measures", body_features),
    ("Smoking", smoking_features),
    ("Physical activity", activity_features),
    ("Sleep", sleep_features),
    ("Total cholesterol", chol_features),
    ("HDL cholesterol", hdl_features),
    ("Alcohol", alcohol_features),
    ("Mental health", mental_features),
]


for name, dataset in datasets:

    before = len(df)

    df = df.merge(
        dataset,
        on="SEQN",
        how="left",
        validate="one_to_one",
    )

    after = len(df)

    print(f"\n{name}:")
    print("Before:", before)
    print("After:", after)
    print("Lost:", before - after)


# ============================================================
# ADULTS ONLY
# ============================================================

before_adult = len(df)

df = df[
    df["age"] >= 18
].copy()

print("\nAdults only:")
print("Before:", before_adult)
print("After:", len(df))
print(
    "Removed participants under 18:",
    before_adult - len(df)
)


# ============================================================
# REQUIRED CORE DATA
# ============================================================

# Age and sex are essential model variables.
df = df.dropna(
    subset=[
        "age",
        "sex",
    ]
).copy()


# ============================================================
# DUPLICATES
# ============================================================

duplicates = df[
    "SEQN"
].duplicated().sum()


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "age",
    "sex",
    "income_ratio",
    "bmi",
    "waist_circumference",
    "obesity_flag",
    "ever_smoked_100",
    "smoking_status",
    "current_smoker",
    "moderate_sessions_week",
    "moderate_minutes_session",
    "vigorous_sessions_week",
    "vigorous_minutes_session",
    "total_activity_minutes_week",
    "sedentary_minutes_day",
    "sleep_hours",
    "short_sleep_flag",
    "total_cholesterol",
    "hdl_cholesterol",
    "alcohol_past_year",
    "alcohol_frequency",
    "average_drinks",
    "mental_symptom_score",
]


# ============================================================
# IMPORTANT LEAKAGE CHECK
# ============================================================

for forbidden_feature in [
    "LBXGH",
    "LBXGLU",
    "DIQ050",
    "DIQ070",
]:

    if forbidden_feature in df.columns:
        raise RuntimeError(
            f"Potential target leakage detected: "
            f"{forbidden_feature}"
        )


# ============================================================
# FINAL DATASET
# ============================================================

final_columns = (
    ["SEQN"]
    + FEATURES
    + ["diabetes"]
)

df = df[final_columns].copy()


print("\n" + "=" * 78)
print("FINAL DIABETES DATASET V1")
print("=" * 78)

print("\nRows:", len(df))
print("Columns:", len(df))

print("\nDuplicate participant IDs:")
print(duplicates)

print("\nTarget distribution:")
print(
    df["diabetes"]
    .value_counts()
    .sort_index()
)

print("\nTarget percentages:")
print(
    (
        df["diabetes"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

print("\nMissing values:")

missing = (
    df.isna()
    .sum()
)

missing = missing[
    missing > 0
].sort_values(
    ascending=False
)

print(missing)


print("\nFeature count:", len(FEATURES))

for feature in FEATURES:
    print("-", feature)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 78)
print("DATASET SAVED")
print("=" * 78)

print(OUTPUT_PATH)