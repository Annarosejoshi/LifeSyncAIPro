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

FILES = {
    "Smoking": DATA_DIR / "SMQ_L.xpt",
    "Alcohol": DATA_DIR / "ALQ_L.xpt",
    "Occupation": DATA_DIR / "OCQ_L.xpt",
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("LIFESYNC AI PRO - LIFESTYLE V2 INSPECTION")
print("=" * 72)

datasets = {}

for name, path in FILES.items():

    print(f"\nLoading {name}:")
    print(path)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_sas(
        path,
        format="xport"
    )

    datasets[name] = df

    print("Loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


# ============================================================
# SHOW ALL COLUMNS
# ============================================================

for name, df in datasets.items():

    print("\n" + "=" * 72)
    print(name.upper(), "COLUMNS")
    print("=" * 72)

    for column in df.columns:
        print(column)


# ============================================================
# VARIABLES WE ARE INTERESTED IN
# ============================================================

candidate_variables = {

    "Smoking": [
        "SMQ020",
        "SMQ040",
        "SMD030",
        "SMQ050Q",
        "SMQ050U",
    ],

    "Alcohol": [
        "ALQ111",
        "ALQ121",
        "ALQ130",
        "ALQ142",
        "ALQ270",
        "ALQ280",
    ],

    "Occupation": [
        "OCQ180",
        "OCQ260",
        "OCD270",
        "OCQ670",
    ],
}


# ============================================================
# INSPECT CANDIDATE VARIABLES
# ============================================================

for dataset_name, variables in candidate_variables.items():

    if dataset_name not in datasets:
        continue

    df = datasets[dataset_name]

    print("\n" + "=" * 72)
    print(dataset_name.upper(), "CANDIDATE VARIABLES")
    print("=" * 72)

    for column in variables:

        if column not in df.columns:

            print(
                f"\nNOT FOUND: {column}"
            )

            continue

        print(
            f"\nVARIABLE: {column}"
        )

        print(
            df[column]
            .value_counts(
                dropna=False
            )
            .sort_index()
            .head(60)
        )

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            print("\nStatistics:")

            print(
                df[column]
                .describe()
            )

        print(
            "\nMissing:",
            df[column]
            .isna()
            .sum()
        )


# ============================================================
# LOAD EXISTING CORE FILES
# ============================================================

print("\n" + "=" * 72)
print("TESTING MERGE RETENTION")
print("=" * 72)

sleep_df = pd.read_sas(
    DATA_DIR / "SLQ_L.xpt",
    format="xport"
)

demo_df = pd.read_sas(
    DATA_DIR / "DEMO_L.xpt",
    format="xport"
)

body_df = pd.read_sas(
    DATA_DIR / "BMX_L.xpt",
    format="xport"
)

activity_df = pd.read_sas(
    DATA_DIR / "PAQ_L.xpt",
    format="xport"
)

mental_df = pd.read_sas(
    DATA_DIR / "DPQ_L.xpt",
    format="xport"
)


core = (
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
    .merge(
        mental_df,
        on="SEQN",
        how="inner"
    )
)


# Adults only
core = core[
    core["RIDAGEYR"] >= 18
].copy()


print(
    "\nCore adult records:",
    len(core)
)


# ============================================================
# MERGE EACH NEW DATASET SEPARATELY
# ============================================================

for name, new_df in datasets.items():

    before = len(core)

    merged = core.merge(
        new_df,
        on="SEQN",
        how="inner"
    )

    after = len(merged)

    lost = before - after

    retention = (
        after
        / before
        * 100
        if before > 0
        else 0
    )

    print(
        f"\n{name}:"
    )

    print(
        "Before:",
        before
    )

    print(
        "After:",
        after
    )

    print(
        "Lost:",
        lost
    )

    print(
        f"Retention: {retention:.2f}%"
    )


# ============================================================
# MERGE ALL THREE
# ============================================================

all_merged = core.copy()

for name in [
    "Smoking",
    "Alcohol",
    "Occupation",
]:

    if name in datasets:

        all_merged = (
            all_merged
            .merge(
                datasets[name],
                on="SEQN",
                how="inner"
            )
        )


print("\n" + "=" * 72)
print("ALL DATASETS MERGED")
print("=" * 72)

print(
    "Core records:",
    len(core)
)

print(
    "After smoking + alcohol + occupation:",
    len(all_merged)
)

lost = (
    len(core)
    - len(all_merged)
)

print(
    "Records lost:",
    lost
)

if len(core) > 0:

    retention = (
        len(all_merged)
        / len(core)
        * 100
    )

    print(
        f"Overall retention: {retention:.2f}%"
    )


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicates = (
    all_merged["SEQN"]
    .duplicated()
    .sum()
)

print(
    "\nDuplicate SEQN:",
    duplicates
)


print("\n" + "=" * 72)
print("INSPECTION COMPLETE")
print("=" * 72)