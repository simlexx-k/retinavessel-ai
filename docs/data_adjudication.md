# FIVES data adjudication and frozen split

Decision date: 2026-11-15. These decisions were made before model training or
inspection of any model result on the official test partition.

## Provenance and audit outcome

- Original archive size: 1,764,974,308 bytes.
- Verified MD5: `789c80dd5376a82063e27fa49192bac9`.
- Paired records: 600 official training and 200 official test images.
- Shape mismatches: none.
- Exact train/test pixel duplicates: none.
- Exact within-split duplicate groups: five in training and one in test.
- Empty masks: training records `447_G` and `448_G`.

The official `Quality Assessment.xlsx` contains one row per image. `IC`,
`Blur`, and `LC` are binary scores for illumination/color, blur, and low
contrast respectively; 1 is good and 0 is poor. Workbook IDs exactly match all
800 image filenames.

## Manual review and decisions

`447_G` and `448_G` are extremely dark, opaque fundus captures, both scored
`IC=0, Blur=0, LC=0`. Their masks are entirely background. They are retained
unchanged in the immutable raw dataset and may be used in failure/quality
analysis, but are excluded from supervised model fitting and validation because
they provide no positive vessel reference.

The duplicate image pairs have differing vessel masks, principally in thin
branches. Mask Dice values range from 0.8777 to 0.9932. To avoid double
weighting and post-review selection of whichever label appears preferable, the
training policy is label-blind: retain the lowest numeric filename and exclude
the later exact-pixel copy.

| Split | Exact duplicate pair | Mask Dice | Decision |
|---|---|---:|---|
| Train | `34_A`, `118_A` | 0.9822 | retain `34_A` |
| Train | `61_A`, `121_A` | 0.9779 | retain `61_A` |
| Train | `39_A`, `132_A` | 0.8777 | retain `39_A` |
| Train | `55_A`, `134_A` | 0.9697 | retain `55_A` |
| Train | `53_A`, `97_A` | 0.9601 | retain `53_A` |
| Test | `32_A`, `48_A` | 0.9932 | retain both in official primary test |

The primary test remains the complete official 200-image partition. The
`32_A`/`48_A` pair will also be handled in a declared secondary sensitivity
analysis using `32_A` as the deterministic lowest-ID representative. Test
labels and eventual test scores must not influence model selection.

## Frozen model-development split

The machine-readable manifest is `splits/fives_seed42.json`. Of 600 official
training records, seven are excluded (two empty masks and five later exact
duplicates), leaving 593 eligible records. A fixed seed of 42 and requested
validation fraction of 0.15 produce:

- model fitting: 504 images;
- validation: 89 images;
- official test: 200 untouched images.

The split is stratified by disease code and a coarse official quality class:
all three quality fields good versus one or more fields poor. This keeps every
disease/quality stratum in both fitting and validation without creating
unstable one-record strata. FIVES does not publish image-to-participant IDs, so
this is not claimed to be participant-disjoint.

The raw dataset is never modified. Exclusions are manifest-level decisions and
can therefore be audited or reversed in a separately declared sensitivity
analysis.
