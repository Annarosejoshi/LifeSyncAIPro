from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data" / "sleep_nhanes"

SLEEP_PATH = DATA_DIR / "SLQ_L.xpt"
DEMO_PATH = DATA_DIR / "DEMO_L.xpt"
BODY_PATH = DATA_DIR / "BMX_L.xpt"


print("=" * 70)
print("LIFESYNC AI PRO - NHANES BODY MEASURES INSPECTION")
print("=" * 70)


# =========================================================
# LOAD FILES
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

print("\nFiles loaded successfully.")

print("\nSleep rows:", len(sleep_df))
print("Demographics rows:", len(demo_df))
print("Body Measures rows:", len(body_df))


# =========================================================
# BODY MEASURE COLUMNS
# =========================================================

print("\n" + "=" * 70)
print("BODY MEASURE COLUMNS")
print("=" * 70)

for column in body_df.columns:
    print(column)


# =========================================================
# IMPORTANT BODY VARIABLES
# =========================================================

important_variables = [
    "SEQN",
    "BMXWT",
    "BMXHT",
    "BMXBMI",
    "BMXWAIST",
]

print("\n" + "=" * 70)
print("IMPORTANT VARIABLES")
print("=" * 70)

for column in important_variables:

    if column in body_df.columns:

        print(f"\nFOUND: {column}")

        print("\nStatistics:")
        print(
            body_df[column]
            .describe()
        )

        print("\nMissing values:")
        print(
            body_df[column]
            .isna()
            .sum()
        )

    else:

        print(
            f"\nNOT FOUND: {column}"
        )


# =========================================================
# BMI INSPECTION
# =========================================================

if "BMXBMI" in body_df.columns:

    print("\n" + "=" * 70)
    print("BMI DISTRIBUTION")
    print("=" * 70)

    print(
        body_df["BMXBMI"]
        .describe()
    )

    print("\nBMI missing:")
    print(
        body_df["BMXBMI"]
        .isna()
        .sum()
    )

    print("\nSample BMI values:")

    print(
        body_df[
            [
                "SEQN",
                "BMXBMI"
            ]
        ]
        .dropna()
        .head(20)
    )


# =========================================================
# FIRST MERGE
# SLEEP + DEMOGRAPHICS
# =========================================================

sleep_demo = sleep_df.merge(
    demo_df,
    on="SEQN",
    how="inner"
)

print("\n" + "=" * 70)
print("SLEEP + DEMOGRAPHICS")
print("=" * 70)

print(
    "Rows after merge:",
    len(sleep_demo)
)


# =========================================================
# SECOND MERGE
# + BODY MEASURES
# =========================================================

merged = sleep_demo.merge(
    body_df,
    on="SEQN",
    how="inner"
)

print("\n" + "=" * 70)
print("SLEEP + DEMOGRAPHICS + BODY MEASURES")
print("=" * 70)

print(
    "Rows before body merge:",
    len(sleep_demo)
)

print(
    "Rows after body merge:",
    len(merged)
)

lost_rows = (
    len(sleep_demo)
    - len(merged)
)

print(
    "Records lost:",
    lost_rows
)

if len(sleep_demo) > 0:

    retention = (
        len(merged)
        / len(sleep_demo)
        * 100
    )

    print(
        f"Record retention: "
        f"{retention:.2f}%"
    )


# =========================================================
# USABLE MODEL RECORDS
# =========================================================

required_columns = [
    "SLD012",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
]

available_required = [
    column
    for column in required_columns
    if column in merged.columns
]

usable = (
    merged[
        available_required
    ]
    .dropna()
)

print("\n" + "=" * 70)
print("CURRENT USABLE MODEL RECORDS")
print("=" * 70)

print(
    "Required variables:",
    available_required
)

print(
    "Rows before dropping missing:",
    len(merged)
)

print(
    "Rows with all current "
    "required values:",
    len(usable)
)


# =========================================================
# ADULT-ONLY CHECK
# =========================================================

if "RIDAGEYR" in usable.columns:

    adults = usable[
        usable["RIDAGEYR"] >= 18
    ]

    print("\nAdult records (18+):")
    print(
        len(adults)
    )


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)