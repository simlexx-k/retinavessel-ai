# Experiment notebook / decisions log

Status updated 2026-11-15: **official FIVES archive verified and audited; data
policy and development split frozen; three preliminary 10-epoch Kaggle
validation runs completed; official test remains untouched.**

## Data audit gate

- [x] Obtain official FIVES archive and confirm original MD5.
- [x] Audit 800 image/mask pairs and scan for duplicates across train/test.
- [x] Manually examine empty/all-vessel and conflicting duplicate masks.
- [x] Parse original quality workbook; verify coding and orientation of all quality score columns.
- [x] Examine image-level split limitations due to missing subject IDs.
- [x] Preserve the official test partition and declare a duplicate sensitivity analysis.
- [x] Record source version, preprocessing and explicit exclusions in `docs/data_adjudication.md`.

## Baseline gate

- [x] Complete an explicitly limited end-to-end U-Net smoke run (not a scientific result).
- [x] Train standard U-Net (BCE+Dice), log hyperparameters and seeds.
- [x] Train lite U-Net under matched settings.
- [x] Train lite U-Net with quality augmentation under matched settings.
- [x] Compare preliminary fixed-budget validation trends.
- [ ] Repeat under the declared 30-epoch maximum / patience-5 policy.
- [ ] Run full untouched primary test once, with declared threshold and FOV method.
- [ ] Include clDice and parameter count in summary tables.
- [ ] Save examples of genuine successes and failures.

## Generalization & extension gate

- [ ] Independently verify HRF and CHASE_DB1 structure, rights and mask/FOV conventions.
- [ ] Implement dataset-specific external importers, independent observer ground-truth policy.
- [ ] Evaluate at native resolution without external-set fine-tuning.
- [ ] Add a rigorous thin-vessel width metric and predeclared severity tests.
- [ ] Add confidence intervals and multi-seed experimental replication.
- [x] Build a separate multi-model FastAPI inference backend with bounded uploads.
- [ ] Connect the frozen winning checkpoint and build the upload/overlay frontend.

## Results templates

| Model | Seed | Test dataset | Mean Dice | Mean clDice | Vessel sensitivity | Parameters | Seconds/image |
|---|---:|---|---:|---:|---:|---:|---:|
| Standard U-Net | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |
| Lite U-Net | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |
| Lite U-Net + degradation | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |

**Never enter invented benchmark scores or classify the pipeline as clinically validated.**

## Preliminary validation results

These are seed-42 development-set results transcribed from the Kaggle training
logs. They are not official-test results. All runs used the frozen manifest,
384-pixel patches, batch size 2, two patches per image and a 10-epoch budget.
All three achieved their best validation Dice at epoch 10, motivating the
predeclared 30-epoch maximum with patience 5 for the next controlled runs.

| Model | Parameters | Best epoch | Mean validation Dice |
|---|---:|---:|---:|
| Standard U-Net | 1,964,097 | 10 | 0.793534 |
| Lite U-Net | 236,156 | 10 | 0.833296 |
| Lite U-Net + degradation | 236,156 | 10 | 0.839926 |

The degradation-trained model's 0.006630 advantage over the clean lite model
is too small to interpret without repeated seeds and degradation-stratified
validation. Checkpoint artifacts remain in Kaggle outputs and are deliberately
excluded from Git.

## Compute readiness

The current host exposes CPU-only PyTorch. A real-size benchmark with the
standard U-Net, batch size 2 and 384-pixel patches indicates that a 10-epoch run
with full 89-image native-resolution validation would take many hours on this
host. The debug run in `outputs/smoke_unet` validates the pipeline but must not
be reported as E1. Run the initial baseline on a CUDA host with limits omitted:

```bash
PYTHONPATH=src python -m retinavessel.train \
  --data data/fives --model unet --epochs 30 --patience 5 \
  --split-manifest splits/fives_seed42.json \
  --device cuda --out outputs/unet_seed42
```

Every epoch writes an atomic `last.pt` containing model, optimizer, history and
random-number-generator state. Resume with `--resume outputs/unet_seed42/last.pt`;
`--epochs` always means the total target epoch, not additional epochs. The
inference-ready `best.pt` tracks the highest validation Dice. Apply the same
maximum epoch and patience policy to every model in a controlled comparison.
