from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data" / "sleep_nhanes"

SLEEP_PATH = DATA_DIR / "SLQ_L.xpt"
DEMO_PATH = DATA_DIR / "DEMO_L.xpt"
BODY_PATH = DATA_DIR / "BMX_L.xpt"
ACTIVITY_PATH = DATA_DIR / "PAQ_L.xpt"
MENTAL_PATH = DATA_DIR / "DPQ_L.xpt"


print("=" * 70)
print("LIFESYNC AI PRO - NHANES MENTAL HEALTH INSPECTION")
print("=" * 70)


# =========================================================
# LOAD DATASETS
# =========================================================

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


print("\nFiles loaded successfully.")

print("\nSleep rows:", len(sleep_df))
print("Demographics rows:", len(demo_df))
print("Body rows:", len(body_df))
print("Activity rows:", len(activity_df))
print("Mental-health rows:", len(mental_df))


# =========================================================
# DPQ COLUMNS
# =========================================================

print("\n" + "=" * 70)
print("MENTAL HEALTH COLUMNS")
print("=" * 70)

for column in mental_df.columns:
    print(column)


# =========================================================
# DPQ DISTRIBUTIONS
# =========================================================

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
    "DPQ100",
]

print("\n" + "=" * 70)
print("DPQ VARIABLE DISTRIBUTIONS")
print("=" * 70)

for column in dpq_columns:

    if column in mental_df.columns:

        print(f"\nVARIABLE: {column}")

        print(
            mental_df[column]
            .value_counts(dropna=False)
            .sort_index()
        )


# =========================================================
# BUILD MERGED DATASET
# =========================================================

merged = (
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
)


print("\n" + "=" * 70)
print("BEFORE MENTAL HEALTH MERGE")
print("=" * 70)

print("Rows:", len(merged))


merged_mental = merged.merge(
    mental_df,
    on="SEQN",
    how="inner"
)


print("\n" + "=" * 70)
print("5-WAY MERGE")
print("=" * 70)

print(
    "Rows before mental-health merge:",
    len(merged)
)

print(
    "Rows after mental-health merge:",
    len(merged_mental)
)

lost = (
    len(merged)
    - len(merged_mental)
)

print(
    "Records lost when adding DPQ:",
    lost
)

if len(merged) > 0:

    retention = (
        len(merged_mental)
        / len(merged)
        * 100
    )

    print(
        f"Mental-health merge retention: "
        f"{retention:.2f}%"
    )


# =========================================================
# CLEAN DPQ SPECIAL CODES
# =========================================================

mental_features = [
    "DPQ010",
    "DPQ020",

    # DPQ030 intentionally excluded because
    # it directly contains sleep information.

    "DPQ040",
    "DPQ050",
    "DPQ060",
    "DPQ070",
    "DPQ080",
]

available_mental_features = [
    column
    for column in mental_features
    if column in merged_mental.columns
]


clean = merged_mental.copy()

for column in available_mental_features:

    clean[column] = clean[column].replace(
        [7, 9],
        pd.NA
    )


# =========================================================
# CURRENT MODEL FEATURES
# =========================================================

current_features = [
    "SLD012",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
]

current_features += available_mental_features


# Only keep columns that exist
current_features = [
    column
    for column in current_features
    if column in clean.columns
]


usable = (
    clean[current_features]
    .dropna()
)


usable_adults = usable[
    usable["RIDAGEYR"] >= 18
]


print("\n" + "=" * 70)
print("CURRENT USABLE SAMPLE WITH MENTAL HEALTH")
print("=" * 70)

print(
    "Features checked:",
    current_features
)

print(
    "Rows before removing missing values:",
    len(clean)
)

print(
    "Rows with complete current features:",
    len(usable)
)

print(
    "Adult usable records:",
    len(usable_adults)
)


# =========================================================
# TARGET DISTRIBUTION PREVIEW
# =========================================================

print("\n" + "=" * 70)
print("SLEEP TARGET PREVIEW")
print("=" * 70)

target_df = usable_adults.copy()

target_df["short_sleep"] = (
    target_df["SLD012"] < 7
).astype(int)


print(
    target_df["short_sleep"]
    .value_counts()
    .sort_index()
)

print("\nPercentages:")

print(
    target_df["short_sleep"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)