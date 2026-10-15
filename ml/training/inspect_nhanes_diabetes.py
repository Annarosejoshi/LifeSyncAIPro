from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "ml" / "data" / "sleep_nhanes"

DIQ_PATH = DATA_DIR / "DIQ_L.xpt"
GHB_PATH = DATA_DIR / "GHB_L.xpt"
GLU_PATH = DATA_DIR / "GLU_L.xpt"

print("=" * 76)
print("LIFESYNC AI PRO - NHANES DIABETES INSPECTION")
print("=" * 76)


def load_xpt(name, path):
    print(f"\nLoading {name}:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_sas(path, format="xport")

    print("Loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


diq = load_xpt("Diabetes Questionnaire", DIQ_PATH)
ghb = load_xpt("Glycohemoglobin / HbA1c", GHB_PATH)
glu = load_xpt("Fasting Glucose", GLU_PATH)


# ---------------------------------------------------------
# COLUMNS
# ---------------------------------------------------------

print("\n" + "=" * 76)
print("DIABETES QUESTIONNAIRE COLUMNS")
print("=" * 76)

for column in diq.columns:
    print(column)


print("\n" + "=" * 76)
print("HbA1c COLUMNS")
print("=" * 76)

for column in ghb.columns:
    print(column)


print("\n" + "=" * 76)
print("FASTING GLUCOSE COLUMNS")
print("=" * 76)

for column in glu.columns:
    print(column)


# ---------------------------------------------------------
# IMPORTANT QUESTIONNAIRE VARIABLES
# ---------------------------------------------------------

print("\n" + "=" * 76)
print("IMPORTANT DIABETES QUESTIONNAIRE VARIABLES")
print("=" * 76)

diabetes_variables = [
    "DIQ010",   # Doctor told you have diabetes
    "DIQ160",   # Told you have prediabetes
    "DIQ050",   # Taking insulin
    "DIQ070",   # Taking diabetes pills
]

for variable in diabetes_variables:

    print(f"\nVARIABLE: {variable}")

    if variable in diq.columns:

        print(
            diq[variable]
            .value_counts(dropna=False)
            .sort_index()
        )

        print("Missing:", diq[variable].isna().sum())

    else:
        print("NOT FOUND")


# ---------------------------------------------------------
# HbA1c
# ---------------------------------------------------------

print("\n" + "=" * 76)
print("HbA1c / GLYCOHEMOGLOBIN")
print("=" * 76)

if "LBXGH" in ghb.columns:

    print("\nLBXGH statistics:")
    print(ghb["LBXGH"].describe())

    print(
        "\nMissing:",
        ghb["LBXGH"].isna().sum()
    )

    print("\nHbA1c category preview:")

    hba1c_category = pd.cut(
        ghb["LBXGH"],
        bins=[-np.inf, 5.7, 6.5, np.inf],
        right=False,
        labels=[
            "Normal",
            "Prediabetes range",
            "Diabetes range"
        ],
    )

    print(
        hba1c_category
        .value_counts(dropna=False)
    )


# ---------------------------------------------------------
# FASTING GLUCOSE
# ---------------------------------------------------------

print("\n" + "=" * 76)
print("FASTING PLASMA GLUCOSE")
print("=" * 76)

if "LBXGLU" in glu.columns:

    print("\nLBXGLU statistics:")
    print(glu["LBXGLU"].describe())

    print(
        "\nMissing:",
        glu["LBXGLU"].isna().sum()
    )

    glucose_category = pd.cut(
        glu["LBXGLU"],
        bins=[-np.inf, 100, 126, np.inf],
        right=False,
        labels=[
            "Normal",
            "Prediabetes range",
            "Diabetes range"
        ],
    )

    print("\nFasting glucose category preview:")

    print(
        glucose_category
        .value_counts(dropna=False)
    )


# ---------------------------------------------------------
# DIAGNOSED DIABETES TARGET
# ---------------------------------------------------------

print("\n" + "=" * 76)
print("DIAGNOSED DIABETES TARGET PREVIEW")
print("=" * 76)

if "DIQ010" in diq.columns:

    target = pd.Series(
        pd.NA,
        index=diq.index,
        dtype="Int64"
    )

    target.loc[diq["DIQ010"] == 1] = 1
    target.loc[diq["DIQ010"] == 2] = 0

    print(target.value_counts(dropna=False))

    valid = target.dropna()

    print("\nValid target records:", len(valid))

    print("\nTarget percentages:")

    print(
        (
            valid.value_counts(normalize=True)
            * 100
        ).round(2)
    )


# ---------------------------------------------------------
# PARTICIPANT OVERLAP
# ---------------------------------------------------------

print("\n" + "=" * 76)
print("PARTICIPANT OVERLAP")
print("=" * 76)

diq_ids = set(diq["SEQN"].dropna())
ghb_ids = set(ghb["SEQN"].dropna())
glu_ids = set(glu["SEQN"].dropna())

print("DIQ participants:", len(diq_ids))
print("HbA1c participants:", len(ghb_ids))
print("Glucose participants:", len(glu_ids))

diq_ghb = diq_ids & ghb_ids
all_three = diq_ids & ghb_ids & glu_ids

print("\nDIQ + HbA1c:", len(diq_ghb))
print("DIQ + HbA1c + Glucose:", len(all_three))

if len(diq_ids) > 0:

    print(
        "All-three retention relative to DIQ:",
        f"{len(all_three) / len(diq_ids) * 100:.2f}%"
    )


# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

print("\n" + "=" * 76)
print("DUPLICATE PARTICIPANT CHECK")
print("=" * 76)

print(
    "DIQ duplicate SEQN:",
    diq["SEQN"].duplicated().sum()
)

print(
    "HbA1c duplicate SEQN:",
    ghb["SEQN"].duplicated().sum()
)

print(
    "Glucose duplicate SEQN:",
    glu["SEQN"].duplicated().sum()
)


print("\n" + "=" * 76)
print("INSPECTION COMPLETE")
print("=" * 76)