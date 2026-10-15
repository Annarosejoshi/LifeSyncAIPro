from pathlib import Path
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

TCHOL_PATH = DATA_DIR / "TCHOL_L.xpt"
TRIGLY_PATH = DATA_DIR / "TRIGLY_L.xpt"


# ============================================================
# HELPER
# ============================================================

def load_xpt(path, name):

    print("\n" + "=" * 72)
    print(f"LOADING {name}")
    print("=" * 72)

    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    df = pd.read_sas(
        path,
        format="xport"
    )

    print("\nLoaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


def inspect_dataset(df, name):

    print("\n" + "=" * 72)
    print(f"{name} - ALL COLUMNS")
    print("=" * 72)

    for column in df.columns:
        print(column)

    print("\n" + "=" * 72)
    print(f"{name} - VARIABLE STATISTICS")
    print("=" * 72)

    for column in df.columns:

        if column == "SEQN":
            continue

        print(f"\nVARIABLE: {column}")

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
# LOAD
# ============================================================

print("=" * 72)
print("LIFESYNC AI PRO - NHANES CHOLESTEROL INSPECTION")
print("=" * 72)

tchol = load_xpt(
    TCHOL_PATH,
    "TOTAL CHOLESTEROL"
)

trigly = load_xpt(
    TRIGLY_PATH,
    "TRIGLYCERIDES / LDL"
)


# ============================================================
# INSPECT
# ============================================================

inspect_dataset(
    tchol,
    "TOTAL CHOLESTEROL"
)

inspect_dataset(
    trigly,
    "TRIGLYCERIDES / LDL"
)


# ============================================================
# PARTICIPANT CHECK
# ============================================================

print("\n" + "=" * 72)
print("PARTICIPANT CHECK")
print("=" * 72)

for name, df in [
    ("TCHOL", tchol),
    ("TRIGLY", trigly),
]:

    print(f"\n{name}")

    print(
        "Rows:",
        len(df)
    )

    print(
        "Unique SEQN:",
        df["SEQN"].nunique()
    )

    print(
        "Duplicate SEQN:",
        df["SEQN"].duplicated().sum()
    )


# ============================================================
# OVERLAP
# ============================================================

tchol_ids = set(
    tchol["SEQN"]
)

trigly_ids = set(
    trigly["SEQN"]
)

overlap = (
    tchol_ids
    &
    trigly_ids
)


print("\n" + "=" * 72)
print("CHOLESTEROL DATASET OVERLAP")
print("=" * 72)

print(
    "TCHOL participants:",
    len(tchol_ids)
)

print(
    "TRIGLY participants:",
    len(trigly_ids)
)

print(
    "Participants in both:",
    len(overlap)
)

if len(tchol_ids) > 0:

    print(
        "TRIGLY retention relative to TCHOL:",
        f"{len(overlap) / len(tchol_ids) * 100:.2f}%"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 72)
print("INSPECTION COMPLETE")
print("=" * 72)