# RetinaVessel AI

**Research title:** Robust and Computationally Efficient Retinal Blood Vessel Segmentation Using Deep Learning Under Variable Fundus Image Quality

A reproducible, **research-only** BSc Data Science project starter. It performs binary retinal-vessel segmentation on fundus photography, using publicly released images with expert vessel masks. It is **not a diagnostic or medical device**.

## Verified datasets and rights

| Dataset | Images | What it provides | Access / rights |
|---|---:|---|---|
| [FIVES (Figshare original)](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169) | 800 | 2048x2048 RGB images, manual vessel masks, image-quality ratings, official train/test | Direct public download, CC BY 4.0 |
| [HRF](https://www5.informatik.uni-erlangen.de/research/data/fundus-images/) | 45 | Expert vessel masks and official FOV masks | Public download; check the provider's reuse rules |
| [CHASE_DB1](https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/) | 28 | Fundus images, two vessel ground truths per image (both eyes from 14 children) | Public 2.4 MB ZIP, CC BY 4.0 |

The original FIVES file is a **1,764,974,308-byte RAR** with MD5 `789c80dd5376a82063e27fa49192bac9`. The `download_fives.py` script verifies the archived bytes. FIVES consists of 600 nominal training images and 200 nominal test images, with images drawn from 573 participants. **Patient identifiers are not provided for enforcing patient-independent evaluation.** A curated copy of FIVES has reported suspicious masks and duplicate images; audit the original data independently and resolve any flags before training. Do not automatically discard data based on a heuristic.

## Quick start

Python 3.10+ is recommended. Install a supported PyTorch build for your GPU/Apple Silicon as appropriate.

```bash
cd retinavessel_ai
python -m venv .venv
# Activate your venv on macOS/Linux: source .venv/bin/activate
# Activate your venv on Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
export PYTHONPATH=src       # macOS/Linux
# $env:PYTHONPATH="src"    # Windows PowerShell

# Downloads original Figshare RAR (approx. 1.76 GB), verifies MD5;
# --extract requires 7-Zip installed and on PATH.
python scripts/download_fives.py --extract

# Verify actual folder layout, labels, duplicate images, mask anomalies.
python -m retinavessel.audit --data data/fives --out outputs/data_audit.json

# After reviewing audit flags, freeze the quality-stratified development split.
python scripts/prepare_fives_split.py --data data/fives \
  --audit outputs/data_audit.json --out splits/fives_seed42.json

# After the audit is reviewed and accepted, train with a maximum epoch budget
# and validation-based early stopping:
python -m retinavessel.train --data data/fives --model unet --epochs 30 \
  --patience 5 --out outputs/unet

# Resume an interrupted run exactly; --epochs is the new total target:
python -m retinavessel.train --data data/fives --model unet --epochs 30 \
  --patience 5 --out outputs/unet --resume outputs/unet/last.pt

# Train the compact variant with training-only image-quality augmentation:
python -m retinavessel.train --data data/fives --model lite_unet --degradation \
  --epochs 30 --patience 5 --out outputs/lite_robust

# Validation-only v3 robustness/topology run; repeat --model for checkpoints.
# This command resolves only the frozen development-validation IDs.
python -m retinavessel.validation --data data/fives \
  --split-manifest splits/fives_seed42.json \
  --model lite_robust=outputs/lite_robust/best.pt \
  --topology --device cuda --out outputs/validation_v3

# Official FIVES test, only after freezing all model-selection decisions:
python -m retinavessel.evaluate --data data/fives --checkpoint outputs/unet/best.pt --topology --out outputs/test_unet

# Research visualization for any fundus image:
python -m retinavessel.predict --checkpoint outputs/unet/best.pt --image PATH/TO/FUNDUS.png --out outputs/prediction
```

### Inference backend

The working-application phase now has a separately packaged FastAPI backend in
[`backend/`](backend/README.md). It loads one or more trusted checkpoints at
startup and exposes health, model-registry and mask/overlay endpoints without
coupling FastAPI to the training package. See the backend README for setup,
configuration and `curl` examples.

If automatic RAR extraction fails, use 7-Zip to unpack the manually downloaded official archive, then pass the directory containing `train/Original`, `train/Ground Truth`, `test/Original`, and `test/Ground Truth` via `--data`.

Tests (no medical dataset needed):

```bash
PYTHONPATH=src pytest -q
```

The commit-pinned `retinavessel-ai-v3.ipynb` consumes the verified v2
checkpoint archive and runs the complete validation-only robustness, topology
and efficiency matrix on Kaggle. It intentionally contains no official-test
evaluation cell.

### Current capabilities

- Verified download of the **original** FIVES release using MD5.
- Integrity audit: paired images/masks, pixel-duplicate groups, cross-split duplicates, duplicate images with conflicting masks, empty/full masks, suspicious foreground fractions.
- Dependency-free parser for the official FIVES quality workbook.
- Frozen disease/quality-stratified, duplicate-aware training/validation split; official test remains untouched.
- Patch-based, native-resolution training with no downsampling of full image to 224/512.
- Standard U-Net vs depthwise-separable lightweight U-Net with BCE + soft Dice loss.
- Optional synthetic blur/low contrast/brightness/noise **on training images only**.
- Full-resolution overlapping tiled inference.
- Per-image and micro-averaged field-of-view restricted Dice, IoU, sensitivity, specificity, precision; optional clDice.
- Deterministic validation-only quality stress tests, thin-centerline recall,
  skeleton fragmentation, paired degradation drops and image-level bootstrap intervals.
- Model output with overlay visualization and JSON/CSV experiment records.
- Separate multi-model FastAPI inference backend with bounded image uploads.

### Not implemented yet (planned research phases)

- CHASE_DB1 and HRF importers and external-domain evaluation; full dataset-specific ground-truth/FOV reconciliation.
- True thin-vessel centerline recall, topology-aware training loss and carefully pre-registered severe-degradation tests.
- Statistical bootstrap CIs across images/patients where identifiers permit, multiple training seeds, reproducibility report, dashboard.

## Research cautions

1. **Data quality is a gate.** Do not train on unreviewed suspect masks; if train/test duplicate leakage is confirmed, document and redesign the split before any claimed independent testing.
2. The original FIVES split does not provide a verified patient-disjoint guarantee. Do not claim otherwise.
3. Masking metrics with an automatically estimated retinal field of view introduces measurement uncertainty; test it against official HRF FOV masks and audit manually.
4. Because vascular pixels are much less common than background, report Dice and vessel sensitivity as well as specificity (overall pixel accuracy is misleading).
5. Image-quality degradations are simulated, not necessarily representative of all clinical cameras.
6. The training script selects the best checkpoint by **validation Dice only**; do not tune on the official test.
7. Use the research prototype for investigation only, **not for individual diagnosis, treatment selection, triage, or patient-facing claims**.

## Source publication and licence

Jin K, Huang X, Zhou J et al. (2022). *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation*. Scientific Data 9, 475. https://doi.org/10.1038/s41597-022-01564-3

FIVES original dataset: https://doi.org/10.6084/m9.figshare.19688169.v1, CC BY 4.0. Preserve attribution and the dataset creators' citation requirements.

More detail: [research protocol](docs/research_protocol.md), [dataset register](docs/data_register.md), [experiment log](docs/experiment_plan.md).
