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
    / "MCQ_L.xpt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("LIFESYNC AI PRO - NHANES CARDIOVASCULAR INSPECTION")
print("=" * 72)

print("\nLoading:")
print(DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nMCQ_L.xpt was not found:\n{DATA_PATH}"
    )

df = pd.read_sas(
    DATA_PATH,
    format="xport"
)

print("\nDataset loaded successfully.")

print("\nRows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# SHOW ALL COLUMNS
# ============================================================

print("\n" + "=" * 72)
print("ALL COLUMNS")
print("=" * 72)

for column in df.columns:
    print(column)


# ============================================================
# CARDIOVASCULAR VARIABLES
# ============================================================

cardio_variables = [
    "SEQN",
    "MCQ160B",   # Congestive heart failure
    "MCQ160C",   # Coronary heart disease
    "MCQ160D",   # Angina
    "MCQ160E",   # Heart attack
    "MCQ160F",   # Stroke
]


print("\n" + "=" * 72)
print("CARDIOVASCULAR VARIABLES")
print("=" * 72)


for variable in cardio_variables:

    if variable not in df.columns:

        print(
            f"\nNOT FOUND: {variable}"
        )

        continue

    print(
        f"\nVARIABLE: {variable}"
    )

    print(
        df[variable]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    print(
        "\nMissing:",
        df[variable].isna().sum()
    )


# ============================================================
# CLEAN YES / NO VALUES
# ============================================================

available_conditions = [
    variable
    for variable in cardio_variables[1:]
    if variable in df.columns
]


clean = df[
    ["SEQN"] + available_conditions
].copy()


# NHANES:
# 1 = Yes
# 2 = No
# 7 = Refused
# 9 = Don't know
#
# 7 and 9 should not be treated as disease status.

for variable in available_conditions:

    clean[variable] = clean[
        variable
    ].where(
        clean[variable].isin([1, 2])
    )


# ============================================================
# INDIVIDUAL CONDITION PREVALENCE
# ============================================================

print("\n" + "=" * 72)
print("VALID YES/NO DISTRIBUTIONS")
print("=" * 72)


for variable in available_conditions:

    valid = clean[
        variable
    ].dropna()

    yes = (
        valid == 1
    ).sum()

    no = (
        valid == 2
    ).sum()

    total = len(valid)

    prevalence = (
        yes / total * 100
        if total
        else 0
    )

    print(
        f"\n{variable}"
    )

    print(
        f"Valid records: {total}"
    )

    print(
        f"Yes: {yes}"
    )

    print(
        f"No: {no}"
    )

    print(
        f"Positive percentage: "
        f"{prevalence:.2f}%"
    )


# ============================================================
# COMBINED CARDIOVASCULAR TARGET PREVIEW
# ============================================================
#
# For inspection only:
#
# 1 = participant reports at least one
#     cardiovascular condition
#
# 0 = participant answered NO to all
#     available cardiovascular conditions
#
# Missing/unknown responses are handled carefully.
# ============================================================

if available_conditions:

    yes_matrix = (
        clean[
            available_conditions
        ] == 1
    )

    no_matrix = (
        clean[
            available_conditions
        ] == 2
    )

    any_yes = yes_matrix.any(
        axis=1
    )

    all_no = no_matrix.all(
        axis=1
    )

    clean[
        "cardiovascular_disease"
    ] = pd.NA

    clean.loc[
        any_yes,
        "cardiovascular_disease"
    ] = 1

    clean.loc[
        all_no,
        "cardiovascular_disease"
    ] = 0


    print("\n" + "=" * 72)
    print("COMBINED CARDIOVASCULAR TARGET PREVIEW")
    print("=" * 72)


    print(
        clean[
            "cardiovascular_disease"
        ].value_counts(
            dropna=False
        )
    )


    valid_target = clean[
        "cardiovascular_disease"
    ].dropna()


    print(
        "\nValid target records:",
        len(valid_target)
    )


    if len(valid_target) > 0:

        print(
            "\nTarget percentages:"
        )

        print(
            (
                valid_target
                .value_counts(
                    normalize=True
                )
                * 100
            ).round(2)
        )


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 72)
print("PARTICIPANT CHECK")
print("=" * 72)

print(
    "Duplicate SEQN:",
    df["SEQN"].duplicated().sum()
)


print("\n" + "=" * 72)
print("INSPECTION COMPLETE")
print("=" * 72)