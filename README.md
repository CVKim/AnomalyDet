# AnomalyDet

Memory-bank-based unsupervised anomaly detection for rotation-capture
inspection of cylindrical automotive parts. Built on PatchCore
(Roth et al., CVPR 2022). Trains on a normal-only image set and produces,
for every input image, a defect heatmap, a binary mask, and a
LabelMe-compatible JSON of polygon annotations.

## Why this exists

Rule-based pre-filtering misses defect categories the rules were not
written for, and labelling effort is dominated by humans eyeballing
candidates against goldens. This pipeline runs a recall-first second
stage: large normal set + zero-to-few defects, output a sortable score
plus pixel-level localization that can be relabelled and fed back into a
supervised loop later.

## Method

- Backbone: WideResNet-50 (ImageNet), `layer2 + layer3` mid-block features,
  3x3 local neighbourhood aggregation (avg-pool s=1).
- Memory bank: every training patch embedding stacked, then k-center
  greedy coreset subsampling (default 10%) so inference is a single
  nearest-neighbour pass against ~tens of thousands of vectors.
- Anomaly map: per-patch nearest-neighbour distance, bilinearly
  upsampled to input resolution, smoothed with an 11x11 Gaussian.
- Threshold calibration: pixel-score percentile of the training set
  (all-normal). Anything above is classified as anomalous, which is what
  makes recall easy to dial in.
- Postprocess: morphological open+close, area filter, contour
  extraction, polygon simplification, write LabelMe JSON.

## Repo layout

```
configs/default.yaml          backbone, layers, coreset ratio, threshold percentile
src/data/                     MVTec-style + generic folder datasets
src/models/feature_extractor  hooked frozen ImageNet backbone
src/models/patchcore          memory bank build / score
src/utils/coreset             k-center greedy with random projection
src/utils/postprocess         heatmap -> mask -> LabelMe JSON
src/utils/visualize           heatmap + mask overlays
src/train.py                  build memory bank
src/inference.py              produce mask + JSON for a folder or MVTec test split
scripts/smoke_check.py        synthetic-data sanity test (no MVTec needed)
scripts/run_demo.ps1          end-to-end demo on MVTec bottle
scripts/download_mvtec.py     download a single MVTec category
tests/test_smoke.py           pytest covering pipeline + postprocess
```

## Setup

```powershell
conda env create -f environment.yml
conda activate anomalydet
python scripts/smoke_check.py
```

The smoke check runs train + inference on synthetic images and
confirms torch + CUDA + the full pipeline are wired up.

## Train + inference

```powershell
# train: build the memory bank from train/good
python -m src.train `
    --config configs/default.yaml `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category bottle

# inference: MVTec test split
python -m src.inference `
    --config configs/default.yaml `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category bottle `
    --memory-bank outputs/bottle/memory_bank.pt

# inference: arbitrary folder of new images
python -m src.inference `
    --memory-bank outputs/bottle/memory_bank.pt `
    --input-dir path\to\new\images `
    --output outputs\custom_run
```

Per-image artifacts under `outputs/<category>/predictions/`:

```
<defect>_<stem>_heatmap.png        normalized anomaly map
<defect>_<stem>_mask.png           binary defect mask
<defect>_<stem>_overlay_heatmap.png  heatmap blended over original
<defect>_<stem>_overlay_mask.png     mask blended over original
<defect>_<stem>.json               LabelMe shapes + image score + threshold
```

## Validated baseline (MVTec bottle)

209 train / 83 test, 22 s to build the bank on RTX 3080.

| Category      | n  | min   | max   | mean  |
|---------------|----|-------|-------|-------|
| good          | 20 |  9.78 | 13.71 | 11.29 |
| broken_large  | 20 | 26.27 | 31.17 | 28.51 |
| broken_small  | 22 | 19.55 | 30.55 | 26.69 |
| contamination | 21 | 17.85 | 35.68 | 26.22 |

Image-level separation is clean (good max 13.71 vs defect min 17.85).

## Branching

`main` holds the validated baseline. Each experiment lives on a branch
off `dev` (e.g. `feat/threshold-calibration`, `feat/efficientad`,
`exp/dinov2-backbone`) and only merges back to `main` after validation.

## License

Internal project. MVTec AD images are subject to MVTec's own license.
