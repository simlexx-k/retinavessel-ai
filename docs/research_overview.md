# Research overview: RetinaVessel AI

**Title:** Robust and Computationally Efficient Retinal Blood Vessel Segmentation Using Deep Learning Under Variable Fundus Image Quality

**Problem:** High segmentation scores may conceal missed fine vessels and interrupted retinal vessel networks. Image quality varies and many large networks are demanding. A carefully audited evaluation of the trade-offs between reliability, robustness and efficiency can provide useful evidence even without inventing a new architecture.

**Research gap to evaluate:** How well do matched U-Net variants preserve thin retinal vessels, topological continuity and generalization when image quality degrades or imaging dataset changes? This is a specific reproducibility and robustness contribution, not a claim that the general question is unstudied.

**Sources:** Official FIVES (800 labelled 2048-square photographs; 600 train / 200 held-out), with no data-access gate and CC BY 4.0 licence. Public HRF (45) and CHASE_DB1 (28) for testing transfer. This gives a data-access-approval-free route; institution-specific research supervision and ethics requirements may still apply.

**Methods:** Image/mask QA, binary segmentation, U-Net, lightweight depthwise separable U-Net, patch-based training, BCE+Dice, quality-aware training augmentations, full-resolution tiled inference, external-domain evaluation, centerline metrics, runtime and model-size comparisons.

**Measured outputs:** Dice, IoU, sensitivity, specificity, clDice, fine-vessel recall (after definition), model parameters, inference time and uncertainty intervals.

**Prototype:** Upload an image and view predicted vessel pixels as a mask or transparent overlay. The tool is only a visualization of a research model, not a source of medical diagnosis or clinical decision support.

**Prerequisite:** Audit exact duplicates and faulty labels before modeling; a curated FIVES derivative reports possible mask errors. Source patient-level grouping is unavailable in published accessible metadata, so do not mislabel image-level splits as patient-separated.

**First reproducible milestone:** Run the provided dataset download+checksum script and `retinavessel.audit`, examine red flags, and freeze an explicit data inclusion/split protocol. Then run the U-Net baseline and compare with lite model.
