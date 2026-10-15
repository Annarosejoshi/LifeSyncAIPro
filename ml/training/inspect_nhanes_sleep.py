from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data" / "sleep_nhanes"

files = list(DATA_DIR.glob("*.xpt")) + list(DATA_DIR.glob("*.XPT"))

if not files:
    raise FileNotFoundError(
        f"No XPT file found in:\n{DATA_DIR}"
    )

DATA_PATH = files[0]

print("=" * 70)
print("LIFESYNC AI PRO - NHANES SLEEP DATA INSPECTION")
print("=" * 70)

print("\nLoading:")
print(DATA_PATH)

df = pd.read_sas(DATA_PATH, format="xport")

print("\nDataset loaded successfully.")

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n" + "=" * 70)
print("ALL COLUMNS")
print("=" * 70)

for column in df.columns:
    print(column)

print("\n" + "=" * 70)
print("IMPORTANT SLEEP VARIABLES")
print("=" * 70)

variables = [
    "SEQN",
    "SLD012",
    "SLD013",
    "SLQ300",
    "SLQ310",
]

for column in variables:

    if column in df.columns:

        print(f"\nFOUND: {column}")

        print(
            df[column]
            .value_counts(dropna=False)
            .sort_index()
            .head(40)
        )

        print("\nStatistics:")
        print(df[column].describe())

    else:
        print(f"\nNOT FOUND: {column}")


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

for column in ["SLD012", "SLD013"]:

    if column in df.columns:

        print(
            column,
            ":",
            df[column].isna().sum()
        )


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)