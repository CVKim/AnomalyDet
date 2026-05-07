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

- Backbone (config-selectable, see [Backbones](#backbones)):
  - **WideResNet-50** (ImageNet), `layer2 + layer3` mid-block features.
  - **DINOv2 ViT-S/14** (Meta), transformer blocks 5 + 11 reshaped to
    spatial maps.
- Local neighbourhood aggregation: 3x3 average pooling, stride 1.
- Memory bank: every training patch embedding stacked, then k-center
  greedy coreset subsampling (default 10%) so inference is a single
  nearest-neighbour pass against tens of thousands of vectors.
- Anomaly map: per-patch nearest-neighbour distance, bilinearly
  upsampled to input resolution, smoothed with an 11x11 Gaussian.
- Threshold calibration: pixel-score percentile of the training set
  (all-normal) recorded in the memory bank file. Inference picks one of
  five strategies (see [Threshold strategies](#threshold-strategies)).
- Postprocess: morphological open+close, area filter, contour
  extraction, polygon simplification, write LabelMe JSON.

## Backbones

A single feature-extractor factory ([src/models/feature_extractor.py](src/models/feature_extractor.py))
dispatches on backbone name. To switch backbones, swap the config:

```yaml
# configs/default.yaml — ResNet family
backbone: wide_resnet50_2     # or resnet50, resnet18
layers: [layer2, layer3]      # named ResNet stages
```

```yaml
# configs/dinov2.yaml — DINOv2 family
backbone: dinov2_vits14       # or dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
layers: [5, 11]               # transformer block indices
input_size: 224               # must be divisible by patch_size (14)
```

The PatchCore code itself is backbone-agnostic — the factory returns a
module whose `forward()` yields `{layer_name: (B, D, H, W)}` regardless
of architecture.

## Threshold strategies

Set with `threshold_mode` in the YAML or `--threshold-mode` on the CLI.

| Mode | What it does | When to use |
|---|---|---|
| `adaptive` (default) | Per-image gate (skip if `image_score < train_image_max * image_gate_factor`) + Otsu/severity/floor max | Recall-first with tight masks; works when good vs defect image-score gap is wide |
| `train_max` | Global: any pixel above the worst training pixel | Catches every anomaly; bleeds into normal regions |
| `train_p999` / `train_p99` | Global: percentile of training-set pixel scores | Stricter than `train_max` but still calibration-driven |
| `test_percentile` | Legacy: percentile across all test pixels | Backwards compat only |
| `--threshold <float>` | Hard-coded value | When you have a validated number |

Adaptive knobs (CLI flags or YAML):
- `image_gate_factor` (default 1.3) — multiplier on `train_image_max` for the gate
- `severity_fraction` (default 0.5) — threshold floor at `image_score * fraction`
- `pixel_floor_factor` (default 1.1) — hard floor at `train_pixel_max * factor`

## Repo layout

```
configs/
  default.yaml                 ResNet/WideResNet config
  dinov2.yaml                  DINOv2 ViT-S/14 config
src/data/                      MVTec-style + generic folder datasets
src/models/feature_extractor   factory: ResNet hooks vs DINOv2 intermediate layers
src/models/patchcore           memory bank build / score / calibration
src/utils/coreset              k-center greedy with random projection
src/utils/postprocess          heatmap -> mask -> LabelMe JSON; adaptive threshold
src/utils/visualize            heatmap + mask overlays
src/train.py                   build memory bank + calibration
src/inference.py               produce mask + JSON for a folder or MVTec test split
scripts/smoke_check.py         synthetic-data sanity test (no MVTec needed)
scripts/run_demo.ps1           end-to-end demo on MVTec bottle
scripts/download_mvtec.py      download a single MVTec category
scripts/sweep_thresholds.py    run inference under 8 threshold configs and summarize
scripts/compare_runs.py        side-by-side markdown comparison of multiple sweeps
tests/test_smoke.py            pytest covering pipeline + postprocess
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
# train: build the memory bank from train/good (WideResNet)
python -m src.train `
    --config configs/default.yaml `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category bottle

# train with DINOv2 instead — just a different config file
python -m src.train `
    --config configs/dinov2.yaml `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category bottle `
    --output outputs/bottle_dinov2

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

# inference: override adaptive knobs without editing YAML
python -m src.inference `
    --config configs/dinov2.yaml `
    --memory-bank outputs/bottle_dinov2/memory_bank.pt `
    --data-root "E:\dataset\mvtec_anomaly_detection_" --category bottle `
    --image-gate-factor 1.5 --severity-fraction 0.7
```

Per-image artifacts under `outputs/<category>/predictions/`:

```
<defect>_<stem>_heatmap.png        normalized anomaly map
<defect>_<stem>_mask.png           binary defect mask
<defect>_<stem>_overlay_heatmap.png  heatmap blended over original
<defect>_<stem>_overlay_mask.png     mask blended over original
<defect>_<stem>.json               LabelMe shapes + image score + threshold
```

## Threshold sweep

Eight common threshold configs run in one shot, each saved to its own
output directory plus a `summary.csv`:

```powershell
python scripts/sweep_thresholds.py `
    --config configs/default.yaml `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category bottle `
    --memory-bank outputs/bottle/memory_bank.pt `
    --output-root outputs/bottle/sweep
```

Combine sweeps from multiple `(backbone, category)` runs into one
markdown table:

```powershell
python scripts/compare_runs.py `
    "outputs/bottle/sweep/summary.csv:wrn50_bottle" `
    "outputs/bottle_dinov2/sweep/summary.csv:dinov2_bottle" `
    "outputs/hazelnut/sweep/summary.csv:wrn50_hazelnut" `
    "outputs/hazelnut_dinov2/sweep/summary.csv:dinov2_hazelnut" `
    --out outputs/comparison.md
```

## Validated results

MVTec AD, RTX 3080. Mask coverage as % of image, `gated/n` is how many
test images were short-circuited to an empty mask by the image-level
gate.

### bottle (`adaptive` baseline, `gate=1.3 sev=0.5`)

| Backbone | good gated | broken_large mean | broken_small mean | contamination mean | bank size |
|---|---|---|---|---|---|
| WideResNet-50 | 14/20 | 22.46% | 8.54% | 16.81% | 16385x1536 |
| **DINOv2 ViT-S/14** | **20/20** | **20.71%** | **6.22%** | **11.87%** | **5350x768** |

### hazelnut

WideResNet adaptive gating fails on hazelnut because train_image_max
(12.69) is too close to good test image scores (min 16.85). Fall back
to `--threshold 22.0` for WideResNet on this category.

| Setup | good min mask | crack mean | cut mean | hole mean | print mean |
|---|---|---|---|---|---|
| WideResNet + `fixed_22` | **0.00%** | 10.36% | 3.60% | 4.25% | 5.16% |
| **DINOv2 + `adaptive`** | **0.00% (40/40 gated)** | **17.38%** | **4.01%** | **4.84%** | **8.18%** |

DINOv2 wins on both categories: every good frame is gated to an empty
mask, every defect is flagged, and the memory bank is ~3-6x smaller.

## Branching

`main` holds the validated baseline. Each experiment lives on a branch
off `dev` (e.g. `feat/threshold-calibration`, `exp/dinov2-backbone`)
and only merges back to `main` after validation.

```powershell
git checkout dev
git pull
git checkout -b feat/your-experiment
# ... iterate, commit ...
git push -u origin feat/your-experiment
# open PR: dev <- feat/your-experiment, then dev -> main
```

## License

Internal project. MVTec AD images are subject to MVTec's own license.
DINOv2 weights are released by Meta under their own terms; see
https://github.com/facebookresearch/dinov2.
