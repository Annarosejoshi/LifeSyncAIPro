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

HDL_PATH = DATA_DIR / "HDL_L.xpt"
MCQ_PATH = DATA_DIR / "MCQ_L.xpt"
TCHOL_PATH = DATA_DIR / "TCHOL_L.xpt"
BP_PATH = DATA_DIR / "BPXO_L.xpt"


# ============================================================
# LOAD FUNCTION
# ============================================================

def load_xpt(path, name):

    print("\n" + "=" * 72)
    print(f"LOADING {name}")
    print("=" * 72)

    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    df = pd.read_sas(
        path,
        format="xport"
    )

    print("Loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    return df


# ============================================================
# START
# ============================================================

print("=" * 72)
print("LIFESYNC AI PRO - NHANES HDL INSPECTION")
print("=" * 72)


hdl = load_xpt(
    HDL_PATH,
    "HDL CHOLESTEROL"
)

mcq = load_xpt(
    MCQ_PATH,
    "MEDICAL CONDITIONS"
)

tchol = load_xpt(
    TCHOL_PATH,
    "TOTAL CHOLESTEROL"
)

bp = load_xpt(
    BP_PATH,
    "BLOOD PRESSURE"
)


# ============================================================
# HDL COLUMNS
# ============================================================

print("\n" + "=" * 72)
print("HDL COLUMNS")
print("=" * 72)

for column in hdl.columns:
    print(column)


# ============================================================
# HDL VARIABLE STATISTICS
# ============================================================

print("\n" + "=" * 72)
print("HDL VARIABLE STATISTICS")
print("=" * 72)

for column in hdl.columns:

    if column == "SEQN":
        continue

    print(f"\nVARIABLE: {column}")
    print("Missing:", hdl[column].isna().sum())

    if pd.api.types.is_numeric_dtype(hdl[column]):

        print(
            hdl[column].describe()
        )

    else:

        print(
            hdl[column]
            .value_counts(dropna=False)
            .head(30)
        )


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 72)
print("PARTICIPANT CHECK")
print("=" * 72)

print(
    "HDL rows:",
    len(hdl)
)

print(
    "HDL unique SEQN:",
    hdl["SEQN"].nunique()
)

print(
    "HDL duplicate SEQN:",
    hdl["SEQN"].duplicated().sum()
)


# ============================================================
# TARGET CREATION
# ============================================================

cardio_columns = [
    "MCQ160B",   # Congestive heart failure
    "MCQ160C",   # Coronary heart disease
    "MCQ160D",   # Angina
    "MCQ160E",   # Heart attack
    "MCQ160F",   # Stroke
]


target = mcq[
    ["SEQN"] + cardio_columns
].copy()


for column in cardio_columns:

    target[column] = target[
        column
    ].where(
        target[column].isin([1, 2])
    )


any_yes = (
    target[cardio_columns] == 1
).any(axis=1)

all_no = (
    target[cardio_columns] == 2
).all(axis=1)


target["cardiovascular_disease"] = pd.NA

target.loc[
    any_yes,
    "cardiovascular_disease"
] = 1

target.loc[
    all_no,
    "cardiovascular_disease"
] = 0


target = target[
    target["cardiovascular_disease"].notna()
].copy()


# ============================================================
# OVERLAP ANALYSIS
# ============================================================

target_ids = set(
    target["SEQN"]
)

hdl_ids = set(
    hdl["SEQN"]
)

tchol_ids = set(
    tchol["SEQN"]
)

bp_ids = set(
    bp["SEQN"]
)


print("\n" + "=" * 72)
print("CARDIOVASCULAR DATA OVERLAP")
print("=" * 72)

print(
    "Valid cardiovascular target:",
    len(target_ids)
)

print(
    "HDL participants:",
    len(hdl_ids)
)

print(
    "TCHOL participants:",
    len(tchol_ids)
)

print(
    "Blood-pressure participants:",
    len(bp_ids)
)


target_hdl = (
    target_ids
    & hdl_ids
)

target_hdl_tchol = (
    target_ids
    & hdl_ids
    & tchol_ids
)

target_hdl_tchol_bp = (
    target_ids
    & hdl_ids
    & tchol_ids
    & bp_ids
)


print("\nTarget + HDL:")
print(
    len(target_hdl)
)

print(
    f"Retention: "
    f"{len(target_hdl) / len(target_ids) * 100:.2f}%"
)


print("\nTarget + HDL + Total Cholesterol:")
print(
    len(target_hdl_tchol)
)

print(
    f"Retention: "
    f"{len(target_hdl_tchol) / len(target_ids) * 100:.2f}%"
)


print("\nTarget + HDL + Total Cholesterol + Blood Pressure:")
print(
    len(target_hdl_tchol_bp)
)

print(
    f"Retention: "
    f"{len(target_hdl_tchol_bp) / len(target_ids) * 100:.2f}%"
)


# ============================================================
# POSITIVE TARGET RETENTION
# ============================================================

positive_ids = set(
    target.loc[
        target["cardiovascular_disease"] == 1,
        "SEQN"
    ]
)


positive_final = (
    positive_ids
    & hdl_ids
    & tchol_ids
    & bp_ids
)


print("\n" + "=" * 72)
print("POSITIVE CASE RETENTION")
print("=" * 72)

print(
    "Original positive cardiovascular cases:",
    len(positive_ids)
)

print(
    "Positive cases with HDL + TCHOL + BP:",
    len(positive_final)
)

if len(positive_ids) > 0:

    print(
        "Positive retention:",
        f"{len(positive_final) / len(positive_ids) * 100:.2f}%"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 72)
print("INSPECTION COMPLETE")
print("=" * 72)