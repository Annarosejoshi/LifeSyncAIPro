from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data" / "sleep_nhanes"

SLEEP_PATH = DATA_DIR / "SLQ_L.xpt"
DEMO_PATH = DATA_DIR / "DEMO_L.xpt"
BODY_PATH = DATA_DIR / "BMX_L.xpt"
ACTIVITY_PATH = DATA_DIR / "PAQ_L.xpt"


print("=" * 70)
print("LIFESYNC AI PRO - NHANES PHYSICAL ACTIVITY INSPECTION")
print("=" * 70)


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


print("\nFiles loaded successfully.")

print("\nSleep rows:", len(sleep_df))
print("Demographics rows:", len(demo_df))
print("Body rows:", len(body_df))
print("Physical Activity rows:", len(activity_df))


# =========================================================
# ACTIVITY COLUMNS
# =========================================================

print("\n" + "=" * 70)
print("PHYSICAL ACTIVITY COLUMNS")
print("=" * 70)

for column in activity_df.columns:
    print(column)


# =========================================================
# INSPECT ALL PAQ VARIABLES
# =========================================================

print("\n" + "=" * 70)
print("PHYSICAL ACTIVITY VARIABLE DISTRIBUTIONS")
print("=" * 70)

for column in activity_df.columns:

    if column == "SEQN":
        continue

    print(f"\nVARIABLE: {column}")

    print(
        activity_df[column]
        .value_counts(dropna=False)
        .sort_index()
        .head(30)
    )


# =========================================================
# MERGE
# =========================================================

sleep_demo = sleep_df.merge(
    demo_df,
    on="SEQN",
    how="inner"
)

sleep_demo_body = sleep_demo.merge(
    body_df,
    on="SEQN",
    how="inner"
)

merged = sleep_demo_body.merge(
    activity_df,
    on="SEQN",
    how="inner"
)


print("\n" + "=" * 70)
print("4-WAY MERGE TEST")
print("=" * 70)

print(
    "Sleep rows:",
    len(sleep_df)
)

print(
    "After demographics:",
    len(sleep_demo)
)

print(
    "After body measures:",
    len(sleep_demo_body)
)

print(
    "After physical activity:",
    len(merged)
)


lost_activity = (
    len(sleep_demo_body)
    - len(merged)
)

print(
    "Records lost when adding activity:",
    lost_activity
)

if len(sleep_demo_body) > 0:

    retention = (
        len(merged)
        / len(sleep_demo_body)
        * 100
    )

    print(
        f"Activity merge retention: "
        f"{retention:.2f}%"
    )


# =========================================================
# BASIC USABLE RECORD COUNT
# =========================================================

required_basic = [
    "SLD012",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
]

usable = (
    merged[
        required_basic
    ]
    .dropna()
)

adults = usable[
    usable["RIDAGEYR"] >= 18
]


print("\n" + "=" * 70)
print("CURRENT USABLE SAMPLE")
print("=" * 70)

print(
    "Records with sleep + age + sex + BMI:",
    len(usable)
)

print(
    "Adult records (18+):",
    len(adults)
)


# =========================================================
# IDENTIFY POSSIBLE ACTIVITY FEATURES
# =========================================================

print("\n" + "=" * 70)
print("POSSIBLE ACTIVITY FEATURES")
print("=" * 70)

keywords = [
    "VIG",
    "MOD",
    "WALK",
    "ACT",
    "PAQ",
    "PAD",
]

possible_activity = []

for column in activity_df.columns:

    upper = column.upper()

    if any(
        keyword in upper
        for keyword in keywords
    ):
        possible_activity.append(column)


print(
    "Possible activity columns:",
    possible_activity
)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)