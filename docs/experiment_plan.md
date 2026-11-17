# Experiment notebook / decisions log

Status updated 2026-11-17: **the three predeclared seeds and complete
validation-only matrices are verified and aggregated. The degradation-trained
Lite U-Net configuration and final analysis policy are frozen. The official
FIVES test remains untouched; the commit-pinned V6 one-time evaluation is the
active stage.**

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
- [x] Repeat under the declared 30-epoch maximum / patience-5 policy.
- [ ] Run full untouched primary test once, with declared threshold and FOV method.
- [x] Include clDice and parameter count in summary tables.
- [ ] Save examples of genuine successes and failures.

## Generalization & extension gate

- [ ] Independently verify HRF and CHASE_DB1 structure, rights and mask/FOV conventions.
- [ ] Implement dataset-specific external importers, independent observer ground-truth policy.
- [ ] Evaluate at native resolution without external-set fine-tuning.
- [x] Implement and predeclare v3 thin-vessel, topology, severity and efficiency measures.
- [x] Execute and independently verify the full three-model v3 validation matrix.
- [x] Add image-level bootstrap confidence intervals.
- [x] Complete predeclared training seeds 7 and 123 and aggregate across seeds.
- [x] Build a separate multi-model FastAPI inference backend with bounded uploads.
- [x] Build a separate Next.js upload/overlay frontend with a server-side API proxy.
- [x] Connect the leading development candidate to the backend with verified provenance.
- [ ] Connect the frozen winning checkpoint and build the upload/overlay frontend.

## Results templates

| Model | Seed | Test dataset | Mean Dice | Mean clDice | Vessel sensitivity | Parameters | Seconds/image |
|---|---:|---|---:|---:|---:|---:|---:|
| Standard U-Net | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |
| Lite U-Net | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |
| Lite U-Net + degradation | 42 | FIVES | Pending | Pending | Pending | Pending | Pending |

**Never enter invented benchmark scores or classify the pipeline as clinically validated.**

## Controlled validation results (v2)

These are seed-42 development-set results verified from the downloaded Kaggle
checkpoints, histories, split manifests and run manifest. They are not
official-test results. Every run used the same frozen 89-image validation set,
384-pixel patches, batch size 2, two patches per training image, a 30-epoch
maximum and patience-5 early stopping. The frozen split SHA-256 was
`aee8809176974d6f880013d85a3ab4a405a43121d0cf31363fcc47dd75256c87`.

| Model | Parameters | Best epoch | Stopped epoch | Mean validation Dice |
|---|---:|---:|---:|---:|
| Standard U-Net | 1,964,097 | 15 | 20 | 0.823469 |
| Lite U-Net | 236,156 | 8 | 13 | 0.826155 |
| Lite U-Net + degradation | 236,156 | 10 | 15 | **0.845217** |

The degradation-trained lite model leads the clean lite model by 0.019062 Dice
and the standard U-Net by 0.021749 on this one seed. It is therefore the
current development candidate, not a frozen research winner: multi-seed,
degradation-stratified, topology and efficiency evaluation remain pending.
Its SHA-addressed checkpoint was promoted to the local inference registry with
the Kaggle run provenance attached. Model binaries and the local registry are
deliberately excluded from Git.

## Robustness, topology and efficiency results (v3)

The complete v3 artifact contains 2,403 records: three seed-42 checkpoints,
nine conditions and all 89 frozen validation images. Its archive SHA-256 is
`6f60f9eb3718e42ade78af49b4af633ecebff292afb10a617e5b6b56c3232f5d`.
All checkpoint/split hashes and saved summaries were independently verified;
`official_test_evaluated` is false.

| Model | Clean Dice (95% image-bootstrap CI) | Clean clDice | Thin recall | Median T4 latency | Peak allocated CUDA memory |
|---|---:|---:|---:|---:|---:|
| Standard U-Net | 0.823469 (0.795601–0.848908) | 0.832439 | 0.215944 | 815 ms | 81.1 MiB |
| Lite U-Net | 0.826155 (0.800501–0.848157) | 0.837423 | 0.309949 | 446 ms | 92.9 MiB |
| Lite U-Net + degradation | **0.845217 (0.818727–0.865821)** | **0.857174** | **0.348004** | 449 ms | 92.9 MiB |

The robust lite model exceeded the clean lite model by a paired mean Dice of
0.019062 on clean images (95% image-bootstrap CI 0.013615–0.024418). Its
advantage increased to 0.097333 under severe noise. Severe-condition Dice was
0.758819 for contrast, 0.767207 for illumination and 0.768065 for noise.

The lightweight architecture used 87.98% fewer trainable parameters and was
about 1.8 times faster than the standard U-Net on this T4 run. It did not use
less peak allocated CUDA memory, so no memory-efficiency advantage is claimed.
Absolute thin-centerline recall remained low for every model, and the mild
effect of the predeclared blur radii is retained without post-result severity
changes. These are single-seed findings until v4 replication is complete.

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
