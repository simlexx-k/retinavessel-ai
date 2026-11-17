# Frozen model and final evaluation policy

Status: **frozen on 2026-11-17 before opening official FIVES test pixels.**

This decision follows the predeclared three-seed protocol in
`v4_multiseed_protocol.md`. Seeds 7, 42 and 123 completed the same frozen
development split, training budget and 3 × 9 × 89 validation matrix. All
executed notebooks, archive identities, checkpoint hashes and summary tables
are recorded under `provenance/`.

## Frozen configuration decision

The selected research configuration is the **degradation-trained lightweight
U-Net**:

- architecture: depthwise-separable `lite_unet`, base width 16;
- parameters: 236,156;
- patch size: 384 pixels, two patches per training image;
- inference overlap: 64 pixels;
- AdamW learning rate: 0.001;
- loss: binary cross-entropy plus soft Dice;
- maximum epochs: 30, early-stopping patience: 5;
- output threshold: 0.5;
- training-only blur, brightness, contrast and noise degradation policy;
- frozen split SHA-256:
  `aee8809176974d6f880013d85a3ab4a405a43121d0cf31363fcc47dd75256c87`.

No individual training seed is selected as the research result. Results are
reported per seed and as the across-seed mean with sample standard deviation.
The existing seed-42 checkpoint may remain the application demonstration
checkpoint because it was deployed before aggregation; it is not described as
the best or representative statistical result.

## Evidence used for the decision

Clean frozen-validation results:

| Configuration | Dice | clDice | Thin recall | T4 latency (ms) |
|---|---:|---:|---:|---:|
| Standard U-Net | 0.8124 ± 0.0147 | 0.8199 ± 0.0173 | 0.2021 ± 0.0212 | 838.84 ± 21.85 |
| Clean Lite U-Net | 0.8235 ± 0.0069 | 0.8337 ± 0.0094 | 0.2579 ± 0.0513 | 453.49 ± 4.15 |
| Robust Lite U-Net | **0.8466 ± 0.0023** | **0.8572 ± 0.0040** | **0.3546 ± 0.0073** | **453.43 ± 3.82** |

The robust configuration leads all three accuracy/topology metrics in every
seed. Its across-seed mean robust-minus-clean-lite effect is positive for Dice,
clDice and thin-centerline recall under all eight degradation conditions.

Relative to standard U-Net, robust Lite U-Net has 87.98% fewer parameters and
45.95% lower mean T4 latency. The measured peak accelerator allocation was
97,457,664 bytes for both lightweight configurations versus 85,031,936 bytes
for standard U-Net. Therefore this project claims parameter-storage and latency
improvements, but **does not claim lower measured GPU peak memory**.

The complete numeric evidence and aggregation method are in
`provenance/aggregation/`. The difference of the two model means equals the
mean within-image paired difference because both models were evaluated on the
same 89 images; the across-seed summaries use the three within-seed effects.

## One-time official FIVES test policy

The official 200-image FIVES test partition remains sealed at this freeze
point. It will be evaluated once in a separately versioned, commit-pinned
script or notebook under the following immutable policy:

1. Evaluate all nine frozen best checkpoints: three configurations × seeds 7,
   42 and 123. Their exact SHA-256 values are in `provenance/manifest.json`.
2. Use native-resolution tiled inference with tile 384, overlap 64 and the
   checkpoint threshold 0.5. Do not tune thresholds or post-processing.
3. Evaluate only clean official-test images. Synthetic stress conclusions
   remain based on the predeclared frozen-validation matrix.
4. Report Dice, IoU, sensitivity, specificity, precision, clDice,
   thin-centerline recall and skeleton-component ratio for each checkpoint.
5. Report each seed and the across-seed mean and sample standard deviation for
   every configuration. Do not select a seed from test performance.
6. Use 2,000 image-level bootstrap repetitions with seed 2026 for confidence
   intervals, noting that public patient identifiers are unavailable.
7. Preserve per-image results, summaries, executed notebook, environment,
   checkpoint hashes and the `official_test_evaluated=true` record under
   `provenance/`.
8. Do not revise architecture, augmentation, threshold, stopping policy or the
   selected configuration after observing official-test results.

External HRF and CHASE_DB1 validation will use the same frozen checkpoints and
inference policy. External results may characterize generalization but cannot
change the frozen model decision.
