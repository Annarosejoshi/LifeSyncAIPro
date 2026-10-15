# Evaluation datasets

The four included reduced CSV files under `processed/` are derived survey
datasets used to reproduce the current lifestyle model experiment:

- `diabetes_lifestyle_v2.csv`
- `hypertension_lifestyle_v2.csv`
- `cardiovascular_lifestyle_v2.csv`
- `mental_health_lifestyle_v2.csv`

They originate from the project's NHANES processing scripts, not from application
users. `SEQN` is the source survey participant identifier. Column definitions,
outcome construction, and preprocessing are recorded in `ml/training/` and
model metadata. Current models use only their metadata-listed feature columns;
the reduced CSVs retain some columns used in earlier experiments.

Training uses a stratified 80/20 split with random state 42. Selection uses
three-fold OOF predictions on training rows, with fold seed 127. The test split
has been evaluated repeatedly across experiments and is not independent
external validation.

Raw NHANES XPT files, the larger BRFSS source, and optional historical processed
datasets are excluded from Git. Rebuilding those datasets requires obtaining
the appropriate original sources and following the corresponding build scripts.
Review the source provider's documentation and terms before redistribution or
research use; selecting a code license does not change data-source terms.

Never place application user exports or identifiable health records here.
