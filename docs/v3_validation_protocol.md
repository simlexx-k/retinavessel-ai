# V3 validation-only robustness and efficiency protocol

Status: **predeclared before the full v3 run.** This stage uses only the 89
images in the frozen FIVES development-validation partition. It must not read,
display or score any image in the official 200-image test partition.

## Fixed inputs

- Split manifest: `splits/fives_seed42.json`.
- Required split SHA-256:
  `aee8809176974d6f880013d85a3ab4a405a43121d0cf31363fcc47dd75256c87`.
- Seed-42 checkpoints: standard U-Net, clean lightweight U-Net and
  degradation-trained lightweight U-Net from the controlled v2 run.
- Each checkpoint's stored threshold remains fixed at 0.5. No threshold or
  model selection may be performed from v3 results.
- Native 2048-square images are evaluated with each checkpoint's stored
  384-pixel tile size and 64-pixel overlap.

## Deterministic stress conditions

Every condition leaves the ground-truth mask unchanged. The FOV is estimated
once from the clean image and reused for every version of that image.

| Condition | Mild | Severe |
|---|---:|---:|
| Gaussian blur radius | 1.0 px | 2.0 px |
| Contrast multiplier | 0.75 | 0.55 |
| Illumination/brightness multiplier | 0.75 | 0.55 |
| Independent RGB Gaussian-noise sigma | 6/255 | 12/255 |

Noise is deterministic per image and condition from base seed 2026. Results
include the clean image plus all eight stress conditions.

## Endpoints

The primary development endpoint remains per-image Dice. Secondary endpoints
are IoU, sensitivity, specificity, precision and clDice.

Thin-vessel recall is operationally defined as follows:

1. Skeletonize the FOV-restricted ground-truth vessel mask.
2. Compute the Euclidean distance to the ground-truth vessel boundary.
3. Select skeleton pixels whose local radius is at most 2 pixels.
4. Count a selected centerline pixel as detected when the prediction lies
   within a one-pixel, 8-connected tolerance.

This is a deliberately strict centerline measure at FIVES native resolution.
It is not interchangeable with a published metric that uses a different
width or tolerance definition. Connectivity reporting also includes clDice
and the ratio of predicted to reference skeleton components of at least five
pixels; interpret component ratios descriptively because small spurs can
increase the count.

## Efficiency and uncertainty

- Run two tile-sized warm-ups before timing.
- Synchronize CUDA immediately before and after each native-resolution
  inference.
- Report clean-condition mean, median and 95th-percentile latency, trainable
  parameters, parameter bytes and peak allocated CUDA memory.
- Calculate 95% bootstrap intervals by resampling images 2,000 times with
  seed 2026. Do not bootstrap individual pixels.
- Report paired clean-minus-degraded changes for Dice, clDice and thin-vessel
  recall.
- FIVES public subject identifiers are unavailable, so these are image-level,
  not participant-level, intervals.

## Execution and decision boundary

Run `retinavessel.validation` for all three models in one Kaggle environment so
latency and memory comparisons share hardware and software. Preserve the
per-image CSV, summary CSV, paired-drop CSV and JSON provenance report.

The full v3 results may identify failure patterns but do not authorize opening
the official test set. First complete the separately versioned additional-seed
training runs, assess whether the architecture/augmentation conclusion is
stable, and freeze the final model and analysis policy.
