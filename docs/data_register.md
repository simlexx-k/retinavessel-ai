# Data source register (verified 2026-11-15)

| ID | Dataset | Original source | Number | Ground truth | Access/licence | Intended use | Mandatory caveat |
|---|---|---|---:|---|---|---|---|
| D1 | FIVES | https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169 | 800 | Pixel vessel masks, image quality scores | Public RAR, CC BY 4.0, original md5 `789c80dd5376a82063e27fa49192bac9` | Training / validation / internal test | 573 participants but no established public patient ID per image; possible suspect/duplicate masks flagged in curated dataset |
| D2 | HRF | https://www5.informatik.uni-erlangen.de/research/data/fundus-images/ | 45 | Vessel masks + FOV masks | Public download; check provider reuse terms | External test | Different capture conditions; no tuning on test |
| D3 | CHASE_DB1 | https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/ | 28 | Two manual vessel masks per image | Public 2.4 MB ZIP; CC BY 4.0 | External test | 14 children, both eyes per child: correlated observations |

FIVES metadata API: https://api.figshare.com/v2/articles/19688169

The original FIVES paper: https://doi.org/10.1038/s41597-022-01564-3

Independent curated dataset report noting suspicious masks: https://huggingface.co/datasets/Kaphathy/OphGen/blob/main/cards/FIVES.md. Treat it as an anomaly report, not proof that the entire official dataset is invalid.

### Source acceptance criteria

1. Source files are directly downloadable and rights documented.
2. Full archive checksums verified where provided.
3. Image/mask pairs and independent data-quality flags checked.
4. Split IDs frozen before training. Exact duplicates flagged; near-duplicates and possible paired-eye leakage separately investigated.
5. Test labels never consulted for model selection.
6. No pretense that open ophthalmic images represent Kenya or routine clinical deployment conditions.
