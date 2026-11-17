# V4 multi-seed replication protocol

Status: **predeclared after the seed-42 v3 results and before training either
additional seed.** The official FIVES test partition remains sealed.

Completion note (2026-11-17): all three predeclared seeds and validation
matrices completed successfully. The aggregation and configuration decision
are recorded in `provenance/aggregation/` and `docs/frozen_model_policy.md`.
The official test remained sealed throughout aggregation.

## Fixed replication design

- Training-randomness seeds: **7, 42 and 123**. Seed 42 is the completed v2
  run; seeds 7 and 123 are additional independent fits selected before their
  results are known.
- The audited image split remains `splits/fives_seed42.json` for every run.
  Its seed controls the split only; it does not prevent varying training RNG.
- Split SHA-256:
  `aee8809176974d6f880013d85a3ab4a405a43121d0cf31363fcc47dd75256c87`.
- Models per seed: standard U-Net, clean lightweight U-Net, and lightweight
  U-Net with the existing training-only degradation policy.
- Matched settings: maximum 30 epochs, patience 5, 384-pixel patches, two
  patches per image, batch size 2, 64-pixel inference overlap, AdamW defaults,
  BCE plus soft Dice loss, and checkpoint threshold 0.5.
- Each new seed is run in a separate Kaggle notebook version to stay within
  accelerator limits and preserve complete artifacts.

No architecture, augmentation, severity, threshold or stopping-policy changes
are permitted between seeds. A failed/interrupted run may resume only from its
own exact `last.pt` state.

## Per-seed evaluation

After all three models for a seed finish, run the complete v3 validation-only
matrix: 89 frozen development images under one clean and eight deterministic
stress conditions. Use the metric definitions and bootstrap settings in
`v3_validation_protocol.md`. Do not add a validation limit.

Each seed archive must contain training histories, best and final-resume
checkpoints, copied split manifests, a seed run manifest, and the four v3
validation result files. Checkpoint hashes and the `official_test_evaluated`
flag must be verified before aggregation.

## Aggregation and decision rule

- Report each seed separately plus across-seed mean and sample standard
  deviation for clean Dice, clDice and thin-centerline recall.
- For augmentation robustness, calculate within-seed robust-minus-clean-lite
  paired differences for every degradation condition, then summarize those
  effects across the three seeds.
- Do not select the single highest-scoring seed as the research result.
- Freeze the architecture/augmentation conclusion only after all three seeds
  and their validation matrices are complete.
- The later official-test run will evaluate the already frozen checkpoints in
  one scripted pass. No decision may be revised from its test results.

The seed-42 FastAPI checkpoint remains a development demonstration until this
replication and final model-freezing process are complete.
