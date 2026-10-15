from pathlib import Path
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "ml"
    / "data"
    / "sleep_nhanes"
    / "BPXO_L.xpt"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 72)
print("LIFESYNC AI PRO - NHANES BLOOD PRESSURE INSPECTION")
print("=" * 72)

print("\nLoading:")
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"BPXO_L.xpt not found:\n{DATA_PATH}"
    )

df = pd.read_sas(
    DATA_PATH,
    format="xport"
)

print("\nDataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# COLUMNS
# ============================================================

print("\n" + "=" * 72)
print("ALL COLUMNS")
print("=" * 72)

for column in df.columns:
    print(column)


# ============================================================
# SEARCH IMPORTANT VARIABLES
# ============================================================

print("\n" + "=" * 72)
print("POSSIBLE BLOOD PRESSURE VARIABLES")
print("=" * 72)

keywords = [
    "SYS",
    "DIA",
    "PULSE",
    "BPX",
]

possible_columns = []

for column in df.columns:

    upper = column.upper()

    if any(
        keyword in upper
        for keyword in keywords
    ):
        possible_columns.append(column)


print(
    "Found:",
    len(possible_columns)
)

for column in possible_columns:
    print(column)


# ============================================================
# INSPECT EACH POSSIBLE VARIABLE
# ============================================================

print("\n" + "=" * 72)
print("VARIABLE STATISTICS")
print("=" * 72)

for column in possible_columns:

    print(
        f"\nVARIABLE: {column}"
    )

    print(
        "Missing:",
        df[column].isna().sum()
    )

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        print(
            df[column].describe()
        )

    else:

        print(
            df[column]
            .value_counts(
                dropna=False
            )
            .head(30)
        )


# ============================================================
# PARTICIPANT CHECK
# ============================================================

print("\n" + "=" * 72)
print("PARTICIPANT CHECK")
print("=" * 72)

if "SEQN" in df.columns:

    print(
        "Participants:",
        df["SEQN"].nunique()
    )

    print(
        "Duplicate SEQN:",
        df["SEQN"].duplicated().sum()
    )

else:

    print("SEQN NOT FOUND")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 72)
print("INSPECTION COMPLETE")
print("=" * 72)