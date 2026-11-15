# Research protocol — RetinaVessel AI

**Working title**: Robust and Computationally Efficient Retinal Blood Vessel Segmentation Using Deep Learning Under Variable Fundus Image Quality

**Field**: Medical computer vision / binary semantic segmentation.  
**Target**: Pixel-level retinal vessels (not diagnosis of retinal or systemic disease).  
**Research design**: Reproducible machine-learning benchmark with ablations, error analysis and independent cross-dataset testing, contingent on dataset audit.

## Problem statement

Automated segmentation of retinal vessels can support research on vascular morphology, but standard segmentation metrics may obscure thin-vessel failures, breaks in vessel connectivity and sensitivity to image degradations. Heavy segmentation models can also limit deployment on modest computers. This project examines the accuracy/robustness/efficiency trade-off when segmenting high-resolution fundus photographs with compact architectures, plus how well FIVES-trained models generalize to separately collected public datasets.

## Aim

To design, benchmark and evaluate a resource-efficient retinal vessel segmentation pipeline that preserves vessel continuity and sensitivity under variation in fundus image quality, using public image data and transparent quality assurance.

## Specific objectives

1. Audit, document and quality-control the public FIVES retinal image and segmentation-mask archive.
2. Establish the performance of a standard U-Net on FIVES with FOV-restricted, full-resolution evaluation.
3. Compare a depthwise-separable lightweight U-Net under matched data splits, training budgets, thresholding and preprocessing.
4. Test whether training-time image-quality degradation augmentation improves performance on genuinely lower-quality images and held-out simulated degradations.
5. Evaluate fine-vessel preservation and connectivity (clDice and a predeclared vessel-width-stratified measure), computational cost, uncertainty and external-domain generalization on HRF/CHASE_DB1.
6. Build a documented research prototype displaying a segmented vessel mask and overlay, with clear limits on clinical use.

## Research questions

- RQ1: How much does a lightweight U-Net reduce model size and inference cost, and what is its effect on full-resolution vessel segmentation quality?
- RQ2: Does image-quality-aware training improve robustness to blur, illumination variation, contrast reduction and noise compared with standard augmentation?
- RQ3: Are thinner vessel structures and the continuity of predicted vessel networks preserved when image quality is reduced?
- RQ4: How much accuracy is lost when models trained on FIVES are evaluated without fine-tuning on HRF and CHASE_DB1?

## Hypotheses (testable; not established facts)

- H1: A compact U-Net can reduce trainable parameters and inference cost relative to a conventional U-Net, with an acceptable change in Dice.
- H2: Quality-aware augmentation improves segmentation on predefined image degradation conditions compared with the same architecture trained without such augmentation.
- H3: Models with similar pixel Dice may still show meaningfully different vessel-centerline continuity and thin-vessel recall.
- H4: Independent external datasets will reveal performance changes not evident from the within-FIVES held-out test.

## Data and split design

**Primary source:** FIVES (800 images, 2048², official 600/200 train/test); 200 normal, 200 diabetic retinopathy, 200 glaucoma and 200 age-related macular degeneration images. Original archive records image-quality criteria (illumination/color distortion, blur, contrast).  
**Secondary sources:** HRF (45 fundus images with expert vessel and FOV masks); CHASE_DB1 (28 images, 14 children, two manual masks per image).

1. Parse and preserve the official test partition. Check image/mask shape, pairing, empty/full labels, class balance, exact pixel duplicates within and across splits and conflicting labels.
2. Independently review flagged records and the author's quality-scoring workbook. Keep an audit trail; **do not silently alter ground truths**.
3. Exclude only audited unusable empty labels and later exact-pixel training duplicates according to the pre-model policy in `data_adjudication.md`. Split the remaining official training portion by image, disease and coarse official quality class (fixed seed), with explicit warning that FIVES lacks public patient identifiers for enforcing subject-disjoint splits. A more defensible participant-grouped split is preferable if metadata become available.
4. Patch sampling uses native-resolution crops for training to protect thin vessels. Full-image reconstruction uses fixed tiled inference with overlap. Avoid resizing original 2048² images to 224² for reported primary results.
5. Fit augmentation and tune models on the training partition; select hyperparameters and decision threshold on validation only. Evaluate the official test **once after freezing decisions**, unless data audit requires leakage correction, which must be documented as a protocol amendment.
6. Evaluate external sets without tuning or fine-tuning. CHASE_DB1 has correlated images from two eyes of each child; use child-grouped uncertainty estimates where appropriate.

## Experiment matrix (complete in this order)

| Exp | Architecture | Loss | Train-time degradations | Goal |
|---|---|---|---|---|
| E0 | Traditional matched-filter/Frangi baseline (future) | N/A | N/A | Non-neural reference |
| E1 | U-Net | BCE + soft Dice | None | Core deep-learning benchmark |
| E2 | Lightweight depthwise-separable U-Net | BCE + soft Dice | None | Parameter/compute comparison |
| E3 | Lightweight U-Net | BCE + soft Dice | Mild blur, contrast, brightness, noise | Robustness ablation |
| E4 | Lightweight U-Net + pre-specified topology-aware loss (future) | BCE + Dice + clDice | Same as E3 | Vessel continuity ablation |

Use **multiple seeds** for final comparisons; keep the image split frozen across architecture variants, record package versions and hardware, and use identical evaluation code.

## Evaluation

Primary endpoint: **per-image mean Dice (FOV-restricted)** at native resolution on the held-out set.

Secondary: per-image IoU, sensitivity, specificity, precision, centerline Dice (clDice), thin-vessel recall (an explicit skeleton-and-width operational definition to be preregistered), inference seconds/image, trainable parameters, peak memory where measurable, performance stratified by disease and by original FIVES image quality. Compare 95% bootstrap confidence intervals across images, and report category-wise n. External tests: HRF and CHASE_DB1, with the proper ground truth and FOV policy for each dataset.

**FOV evaluation:** FIVES does not contain a standard dedicated FOV mask in the two image/mask directories, so the starter code derives an approximate FOV from fundus image intensity. This requires sensitivity analysis and visual confirmation. HRF supplies official FOV masks. Do not compare cross-study scores without harmonizing FOV and thresholding policies.

**Synthetic stress tests:** For fixed model checkpoints, evaluate the exact
blur, illumination, contrast and noise perturbations predeclared in
`v3_validation_protocol.md`. Never alter the ground-truth vessel mask.
Distinguish these from real low-quality photos and be mindful that high
perturbation may physically erase vessels, making ground truth ambiguous.

## Statistical analysis and validity threats

- Pixel observations within the same fundus image are highly dependent; do not bootstrap pixels as if independent samples.
- Where participant identities are available (CHASE_DB1), account for two eyes per participant.
- FIVES includes 800 photographs from 573 participants but public patient identifiers have not been established; any patient-level leakage cannot be ruled out solely by image-file separation.
- Image-derived FOV may bias specificity/precision; label provenance and duplicates are potential leakage sources.
- Public data come from distinct cameras, populations, ages and ophthalmic conditions; external generalisation may be limited.
- Avoid claims of novelty based only on using U-Net, clDice or augmentation: these are well-established. The contribution is a rigorous **auditable comparison of segmentation robustness, topology, efficiency and external transfer**, unless a genuinely new method is later developed.

## Deliverables

- Version-controlled repo, immutable dataset provenance, machine-readable audit report.
- Code, fixed split manifest, training scripts, test metrics, failure case analysis.
- Comparison tables and plots; uncertainty and limitations statement.
- Research-only visual application showing original image, vessel mask and overlay.
- Final thesis with reproducible experiments and citations.

## Governance

All listed primary/secondary sources are openly published. No application to a hospital or DHS microdata program is required to begin; comply with licences and the university's research/ethics review procedures. Do not collect patient identifiers, upload private clinical data, claim diagnostic accuracy or deploy for clinical decisions without appropriate independent validation and authorization.
