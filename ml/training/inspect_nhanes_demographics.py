from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data" / "sleep_nhanes"

SLEEP_PATH = DATA_DIR / "SLQ_L.xpt"
DEMO_PATH = DATA_DIR / "DEMO_L.xpt"


print("=" * 70)
print("LIFESYNC AI PRO - NHANES DEMOGRAPHICS INSPECTION")
print("=" * 70)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

sleep_df = pd.read_sas(SLEEP_PATH, format="xport")
demo_df = pd.read_sas(DEMO_PATH, format="xport")

print("\nFiles loaded successfully.")

print("\nSleep dataset:")
print("Rows:", len(sleep_df))
print("Columns:", len(sleep_df.columns))

print("\nDemographics dataset:")
print("Rows:", len(demo_df))
print("Columns:", len(demo_df.columns))


# ---------------------------------------------------------
# DEMOGRAPHIC COLUMNS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DEMOGRAPHICS COLUMNS")
print("=" * 70)

for column in demo_df.columns:
    print(column)


# ---------------------------------------------------------
# IMPORTANT VARIABLES
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTANT VARIABLES")
print("=" * 70)

important_variables = [
    "SEQN",
    "RIAGENDR",
    "RIDAGEYR",
    "RIDRETH3",
    "DMDEDUC2",
    "DMDMARTZ",
    "INDFMPIR",
]

for column in important_variables:

    if column in demo_df.columns:

        print(f"\nFOUND: {column}")

        print(
            demo_df[column]
            .value_counts(dropna=False)
            .sort_index()
            .head(30)
        )

        print("\nStatistics:")
        print(demo_df[column].describe())

    else:
        print(f"\nNOT FOUND: {column}")


# ---------------------------------------------------------
# MERGE TEST
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SLEEP + DEMOGRAPHICS MERGE TEST")
print("=" * 70)

merged = sleep_df.merge(
    demo_df,
    on="SEQN",
    how="inner"
)

print("Sleep rows before merge:", len(sleep_df))
print("Demographics rows:", len(demo_df))
print("Rows after inner merge:", len(merged))

lost_rows = len(sleep_df) - len(merged)

print("Sleep records lost:", lost_rows)

if len(sleep_df) > 0:
    retention = (len(merged) / len(sleep_df)) * 100
    print(f"Sleep record retention: {retention:.2f}%")


# ---------------------------------------------------------
# TARGET AVAILABILITY
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SLEEP TARGET AVAILABILITY AFTER MERGE")
print("=" * 70)

for column in ["SLD012", "SLD013"]:

    if column in merged.columns:

        usable = merged[column].notna().sum()
        missing = merged[column].isna().sum()

        print(f"\n{column}")
        print("Usable:", usable)
        print("Missing:", missing)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)