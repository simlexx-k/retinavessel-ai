# Kaggle V6 official-test run

V6 is the single frozen evaluation of all nine checkpoints on the 200-image
official FIVES test partition. Running it opens the test pixels and therefore
changes the research record permanently. Do not use its results to change the
model, threshold, preprocessing or analysis policy.

## Prepare the Kaggle inputs

Create one **private** Kaggle dataset containing these three files without
renaming, recompressing or extracting them:

- `retinavessel_v2_seed42.zip`
- `retinavessel_v4_seed7.zip`
- `retinavessel_v5_seed123.zip`

The notebook checks the exact byte count, archive SHA-256, ZIP integrity and
all nine checkpoint SHA-256 values against
`fives_official_test_plan.json`. A modified or incomplete input stops before
test pixels are opened.

## Run

1. Import `retinavessel-ai-v6-official-test.ipynb` into Kaggle.
2. Attach the existing private `fives-retinal-dataset` dataset.
3. Attach the private dataset containing the three run ZIPs.
4. Enable a GPU and Internet access, then choose **Run all**.
5. Do not edit cells or the frozen JSON plan.

The runner writes a checkpoint-complete partial table after each of the nine
models. If a cell is interrupted while the same Kaggle session is still
available, rerunning the evaluation cell resumes only completed checkpoints
from the unchanged plan. Do not start a differently configured replacement
run.

## Download before closing the session

Download both:

- the saved, executed V6 notebook version;
- `/kaggle/working/retinavessel_v6_official_test.zip`.

The result archive contains 1,800 per-image rows, per-checkpoint and
across-seed configuration summaries, disease-group summaries, the full JSON
report, run state, environment snapshot and a checksum manifest. These two
downloads must be ingested into `provenance/` before HRF or CHASE_DB1 external
validation begins.
