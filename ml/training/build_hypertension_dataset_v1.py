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

OUTPUT_FILE = OUTPUT_DIR / "hypertension_risk_dataset_v1.csv"


# ============================================================
# FILES
# ============================================================

BPQ_FILE = DATA_DIR / "BPQ_L.xpt"
BP_FILE = DATA_DIR / "BPXO_L.xpt"
DEMO_FILE = DATA_DIR / "DEMO_L.xpt"
BODY_FILE = DATA_DIR / "BMX_L.xpt"
SMOKING_FILE = DATA_DIR / "SMQ_L.xpt"
ACTIVITY_FILE = DATA_DIR / "PAQ_L.xpt"
SLEEP_FILE = DATA_DIR / "SLQ_L.xpt"
TCHOL_FILE = DATA_DIR / "TCHOL_L.xpt"
HDL_FILE = DATA_DIR / "HDL_L.xpt"
ALCOHOL_FILE = DATA_DIR / "ALQ_L.xpt"
MENTAL_FILE = DATA_DIR / "DPQ_L.xpt"


print("=" * 78)
print("LIFESYNC AI PRO - BUILD HYPERTENSION DATASET V1")
print("=" * 78)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading NHANES datasets...")


def load_xpt(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_sas(path, format="xport")


bpq = load_xpt(BPQ_FILE)
bp = load_xpt(BP_FILE)
demo = load_xpt(DEMO_FILE)
body = load_xpt(BODY_FILE)
smoking = load_xpt(SMOKING_FILE)
activity = load_xpt(ACTIVITY_FILE)
sleep = load_xpt(SLEEP_FILE)
tchol = load_xpt(TCHOL_FILE)
hdl = load_xpt(HDL_FILE)
alcohol = load_xpt(ALCOHOL_FILE)
mental = load_xpt(MENTAL_FILE)

print("All files loaded successfully.")


# ============================================================
# BLOOD PRESSURE AVERAGES
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

bp["avg_systolic_bp"] = bp[systolic_columns].mean(axis=1)

bp["avg_diastolic_bp"] = bp[diastolic_columns].mean(axis=1)


# ============================================================
# BP EXAM TARGET
# ============================================================

bp["exam_hypertension"] = np.nan

valid_bp = (
    bp["avg_systolic_bp"].notna()
    & bp["avg_diastolic_bp"].notna()
)

bp.loc[
    valid_bp,
    "exam_hypertension"
] = (
    (bp.loc[valid_bp, "avg_systolic_bp"] >= 130)
    |
    (bp.loc[valid_bp, "avg_diastolic_bp"] >= 80)
).astype(int)


# ============================================================
# BPQ DIAGNOSIS
# ============================================================

def convert_yes_no(series):

    output = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    output.loc[series == 1] = 1
    output.loc[series == 2] = 0

    return output


bpq["diagnosed_hypertension"] = convert_yes_no(
    bpq["BPQ020"]
)

bpq["bp_medication"] = convert_yes_no(
    bpq["BPQ150"]
)


# ============================================================
# CREATE TARGET BASE
# ============================================================

target = bpq[
    [
        "SEQN",
        "diagnosed_hypertension",
        "bp_medication",
    ]
].merge(
    bp[
        [
            "SEQN",
            "exam_hypertension",
        ]
    ],
    on="SEQN",
    how="inner",
)


target["hypertension"] = np.nan


positive = (
    (target["diagnosed_hypertension"] == 1)
    |
    (target["bp_medication"] == 1)
    |
    (target["exam_hypertension"] == 1)
)


negative = (
    (target["diagnosed_hypertension"] == 0)
    &
    (
        (target["bp_medication"] == 0)
        |
        target["bp_medication"].isna()
    )
    &
    (target["exam_hypertension"] == 0)
)


target.loc[positive, "hypertension"] = 1
target.loc[negative, "hypertension"] = 0


print("\nInitial target rows:", len(target))

print("\nInitial target distribution:")
print(target["hypertension"].value_counts(dropna=False))


# Remove unresolved target
target = target[
    target["hypertension"].notna()
].copy()

target["hypertension"] = (
    target["hypertension"].astype(int)
)


print("\nValid target records:", len(target))


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
    (body_features["bmi"] >= 30).astype(int),
    np.nan
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


# ------------------------------------------------------------
# Ever smoked at least 100 cigarettes
#
# SMQ020:
# 1 = Yes
# 2 = No
# 7/9 = Refused / Don't know
# ------------------------------------------------------------

smoking_features["ever_smoked_100"] = np.nan

smoking_features.loc[
    smoking_features["SMQ020"] == 1,
    "ever_smoked_100"
] = 1

smoking_features.loc[
    smoking_features["SMQ020"] == 2,
    "ever_smoked_100"
] = 0


# ------------------------------------------------------------
# SMOKING STATUS
#
# 0 = Never smoker
# 1 = Former smoker
# 2 = Current smoker
#
# Logic:
#
# SMQ020 = 2
#     Never smoked 100 cigarettes -> Never smoker
#
# SMQ020 = 1 and SMQ040 = 3
#     Smoked >=100 cigarettes but currently not smoking
#     -> Former smoker
#
# SMQ020 = 1 and SMQ040 = 1 or 2
#     Currently smoking every day / some days
#     -> Current smoker
# ------------------------------------------------------------

smoking_features["smoking_status"] = np.nan


# Never smoker
smoking_features.loc[
    smoking_features["SMQ020"] == 2,
    "smoking_status"
] = 0


# Former smoker
smoking_features.loc[
    (
        (smoking_features["SMQ020"] == 1)
        &
        (smoking_features["SMQ040"] == 3)
    ),
    "smoking_status"
] = 1


# Current smoker
smoking_features.loc[
    (
        (smoking_features["SMQ020"] == 1)
        &
        (smoking_features["SMQ040"].isin([1, 2]))
    ),
    "smoking_status"
] = 2


# Binary current-smoking indicator
smoking_features["current_smoker"] = np.nan

smoking_features.loc[
    smoking_features["smoking_status"].isin([0, 1]),
    "current_smoker"
] = 0

smoking_features.loc[
    smoking_features["smoking_status"] == 2,
    "current_smoker"
] = 1


# Keep final smoking features only
smoking_features = smoking_features[
    [
        "SEQN",
        "ever_smoked_100",
        "smoking_status",
        "current_smoker",
    ]
]


print("\nSmoking status distribution:")

print(
    smoking_features[
        "smoking_status"
    ].value_counts(
        dropna=False
    ).sort_index()
)

print(
    "\nSmoking status missing:",
    smoking_features[
        "smoking_status"
    ].isna().sum()
)
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
        )

    return value


activity_features["PAD790U"] = (
    activity_features["PAD790U"]
    .apply(decode_unit)
)

activity_features["PAD810U"] = (
    activity_features["PAD810U"]
    .apply(decode_unit)
)


# Invalid questionnaire codes
for column in ["PAD790Q", "PAD810Q"]:

    activity_features.loc[
        activity_features[column].isin(
            [7777, 9999]
        ),
        column
    ] = np.nan


# Convert frequency to sessions/week
def sessions_per_week(quantity, unit):

    if pd.isna(quantity):
        return np.nan

    if unit == "D":
        return quantity * 7

    if unit == "W":
        return quantity

    if unit == "M":
        return quantity / 4.345

    if unit == "Y":
        return quantity / 52.1429

    return np.nan


activity_features[
    "moderate_sessions_week"
] = activity_features.apply(
    lambda row: sessions_per_week(
        row["PAD790Q"],
        row["PAD790U"]
    ),
    axis=1
)


activity_features[
    "vigorous_sessions_week"
] = activity_features.apply(
    lambda row: sessions_per_week(
        row["PAD810Q"],
        row["PAD810U"]
    ),
    axis=1
)


activity_features[
    "moderate_minutes_session"
] = activity_features["PAD800"]


activity_features[
    "vigorous_minutes_session"
] = activity_features["PAD820"]


activity_features[
    "total_activity_minutes_week"
] = (
    activity_features[
        "moderate_sessions_week"
    ]
    *
    activity_features[
        "moderate_minutes_session"
    ]
).fillna(0) + (
    activity_features[
        "vigorous_sessions_week"
    ]
    *
    activity_features[
        "vigorous_minutes_session"
    ]
).fillna(0)


activity_features[
    "sedentary_minutes_day"
] = activity_features["PAD680"]


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
        "SLD012": "sleep_hours"
    }
)

sleep_features["short_sleep_flag"] = np.where(
    sleep_features["sleep_hours"].notna(),
    (sleep_features["sleep_hours"] < 7).astype(int),
    np.nan
)


# ============================================================
# TOTAL CHOLESTEROL
# ============================================================

tchol_features = tchol[
    [
        "SEQN",
        "LBXTC",
    ]
].copy()

tchol_features = tchol_features.rename(
    columns={
        "LBXTC": "total_cholesterol"
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
        "LBDHDD": "hdl_cholesterol"
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


alcohol_features["alcohol_past_year"] = (
    convert_yes_no(
        alcohol_features["ALQ111"]
    )
)


alcohol_features["alcohol_frequency"] = (
    alcohol_features["ALQ121"]
)

alcohol_features.loc[
    alcohol_features[
        "alcohol_frequency"
    ].isin([77, 99]),
    "alcohol_frequency"
] = np.nan


alcohol_features["average_drinks"] = (
    alcohol_features["ALQ130"]
)

alcohol_features.loc[
    alcohol_features[
        "average_drinks"
    ].isin([777, 999]),
    "average_drinks"
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
        mental_features[column].isin(
            [7, 9]
        ),
        column
    ] = np.nan


mental_features[
    "mental_symptom_score"
] = mental_features[
    dpq_columns
].sum(
    axis=1,
    min_count=len(dpq_columns)
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

dataset = target[
    [
        "SEQN",
        "hypertension",
    ]
].copy()


datasets = [
    ("Demographics", demo_features),
    ("Body measures", body_features),
    ("Smoking", smoking_features),
    ("Physical activity", activity_features),
    ("Sleep", sleep_features),
    ("Total cholesterol", tchol_features),
    ("HDL cholesterol", hdl_features),
    ("Alcohol", alcohol_features),
    ("Mental health", mental_features),
]


print("\n" + "=" * 78)
print("MERGE PROCESS")
print("=" * 78)


for name, frame in datasets:

    before = len(dataset)

    dataset = dataset.merge(
        frame,
        on="SEQN",
        how="left"
    )

    after = len(dataset)

    print(f"\n{name}:")
    print("Before:", before)
    print("After:", after)
    print("Lost:", before - after)


# ============================================================
# ADULTS ONLY
# ============================================================

dataset = dataset[
    dataset["age"] >= 18
].copy()


# ============================================================
# SAFETY CHECK: TARGET LEAKAGE
# ============================================================

forbidden_features = [
    "avg_systolic_bp",
    "avg_diastolic_bp",
    "BPQ020",
    "BPQ030",
    "BPQ150",
    "diagnosed_hypertension",
    "bp_medication",
    "exam_hypertension",
]


leaks = [
    column
    for column in forbidden_features
    if column in dataset.columns
]


if leaks:
    raise RuntimeError(
        f"TARGET LEAKAGE DETECTED: {leaks}"
    )


# ============================================================
# DUPLICATES
# ============================================================

duplicates = dataset[
    "SEQN"
].duplicated().sum()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 78)
print("FINAL HYPERTENSION DATASET V1")
print("=" * 78)

print("\nRows:", len(dataset))
print("Columns:", len(dataset.columns))

print("\nDuplicate participant IDs:")
print(duplicates)


print("\nTarget distribution:")

print(
    dataset[
        "hypertension"
    ].value_counts()
)


print("\nTarget percentages:")

print(
    (
        dataset[
            "hypertension"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


print("\nMissing values:")

missing = (
    dataset
    .isna()
    .sum()
)

print(
    missing[
        missing > 0
    ].sort_values(
        ascending=False
    )
)


# ============================================================
# FEATURE LIST
# ============================================================

feature_columns = [
    column
    for column in dataset.columns
    if column not in [
        "SEQN",
        "hypertension",
    ]
]


print("\nFeature count:", len(feature_columns))

for feature in feature_columns:
    print("-", feature)


# ============================================================
# SAVE
# ============================================================

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 78)
print("DATASET SAVED")
print("=" * 78)

print(OUTPUT_FILE)