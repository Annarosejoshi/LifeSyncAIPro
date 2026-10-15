from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "ml" / "data" / "sleep"

# Automatically find the XPT file
xpt_files = list(DATA_DIR.glob("*.XPT")) + list(DATA_DIR.glob("*.xpt"))

if not xpt_files:
    raise FileNotFoundError(
        f"No .XPT file found inside:\n{DATA_DIR}"
    )

DATA_PATH = xpt_files[0]

print("=" * 70)
print("LIFESYNC AI PRO - SLEEP DATASET INSPECTION")
print("=" * 70)

print("\nDataset:")
print(DATA_PATH)

print("\nLoading BRFSS dataset...")

df = pd.read_sas(
    DATA_PATH,
    format="xport",
    encoding="latin1"
)

print("\nDataset loaded successfully.")

print("\n" + "=" * 70)
print("DATASET SIZE")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n" + "=" * 70)
print("SEARCHING FOR SLEEP VARIABLES")
print("=" * 70)

sleep_columns = [
    column
    for column in df.columns
    if "SLEEP" in column.upper()
]

print("Sleep-related columns:")

for column in sleep_columns:
    print(" -", column)

print("\n" + "=" * 70)
print("IMPORTANT LIFESYNC CANDIDATE FEATURES")
print("=" * 70)

candidate_features = [
    "SLEPTIM1",
    "_AGEG5YR",
    "_SEX",
    "_BMI5",
    "_RFSMOK3",
    "_TOTINDA",
    "MENTHLTH",
    "PHYSHLTH",
    "GENHLTH",
    "ALCDAY4",
    "_RFDRHV8",
]

for column in candidate_features:

    if column in df.columns:

        print(f"\nFOUND: {column}")

        print(
            df[column]
            .value_counts(dropna=False)
            .head(20)
        )

    else:
        print(f"\nNOT FOUND: {column}")


print("\n" + "=" * 70)
print("SLEEP DURATION VARIABLE")
print("=" * 70)

if "SLEPTIM1" in df.columns:

    sleep = df["SLEPTIM1"]

    print("\nRaw SLEPTIM1 distribution:")
    print(
        sleep
        .value_counts(dropna=False)
        .sort_index()
        .head(40)
    )

    print("\nMissing values:")
    print(sleep.isna().sum())

    print("\nBasic statistics:")
    print(sleep.describe())

else:

    print("SLEPTIM1 was not found.")
    
    
print("\n" + "=" * 70)
print("SEARCHING ALL POSSIBLE SLEEP / REST VARIABLES")
print("=" * 70)

keywords = [
    "SLEEP",
    "SLEP",
    "REST",
    "HOUR",
    "INSOM",
]

matching_columns = []

for column in df.columns:
    upper = column.upper()

    if any(keyword in upper for keyword in keywords):
        matching_columns.append(column)

print(f"\nFound {len(matching_columns)} possible columns:\n")

for column in matching_columns:
    print(column)

    try:
        print(df[column].value_counts(dropna=False).head(15))
    except Exception as exc:
        print("Could not inspect:", exc)

    print("-" * 50)


print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)