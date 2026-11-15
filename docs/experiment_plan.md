# Experiment notebook / decisions log

Status updated 2026-11-15: **official FIVES archive verified and audited; data policy and development split frozen; no model trained or evaluated.**

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
- [ ] Train standard U-Net (BCE+Dice), log hyperparameters and seeds.
- [ ] Train lite U-Net under matched settings.
- [ ] Train lite U-Net with quality augmentation under matched settings.
- [ ] Compare validation trends before freezing model choice.
- [ ] Run full untouched primary test once, with declared threshold and FOV method.
- [ ] Include clDice and parameter count in summary tables.
- [ ] Save examples of genuine successes and failures.

## Generalization & extension gate

- [ ] Independently verify HRF and CHASE_DB1 structure, rights and mask/FOV conventions.
- [ ] Implement dataset-specific external importers, independent observer ground-truth policy.
- [ ] Evaluate at native resolution without external-set fine-tuning.
- [ ] Add a rigorous thin-vessel width metric and predeclared severity tests.
- [ ] Add confidence intervals and multi-seed experimental replication.
- [ ] Prototype a research visualization interface and document its medical limitations.

## Results templates

| Model | Seed | Test dataset | Mean Dice | Mean clDice | Vessel sensitivity | Parameters | Seconds/image |
|---|---:|---|---:|---:|---:|---:|---:|
| Standard U-Net | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |
| Lite U-Net | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |
| Lite U-Net + degradation | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |

**Never enter invented benchmark scores or classify the pipeline as clinically validated.**

## Compute readiness

The current host exposes CPU-only PyTorch. A real-size benchmark with the
standard U-Net, batch size 2 and 384-pixel patches indicates that a 10-epoch run
with full 89-image native-resolution validation would take many hours on this
host. The debug run in `outputs/smoke_unet` validates the pipeline but must not
be reported as E1. Run the initial baseline on a CUDA host with limits omitted:

```bash
PYTHONPATH=src python -m retinavessel.train \
  --data data/fives --model unet --epochs 10 \
  --split-manifest splits/fives_seed42.json \
  --device cuda --out outputs/unet_seed42
```
