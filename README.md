# AnomalyDet

Memory-bank-based unsupervised anomaly detection for rotation-capture
inspection of cylindrical automotive parts. Paper-faithful PatchCore
(Roth et al., CVPR 2022) on a frozen DINOv2 ViT-S/14 backbone.

Train on a normal-only image set → for every input image, emit a
defect heatmap, a binary mask, a 6-panel composite, and a
LabelMe-compatible polygon JSON.

## Result

**MVTec hazelnut, pixel-level metrics vs ground truth** (single MVTec
test split, 110 images, the F1-optimal threshold from a 200-step
sweep over the full positive pixel pool + 50M sampled negatives):

| Backbone | Config | F1 | Precision | Recall | IoU |
|---|---|---|---|---|---|
| WRN-50 @ 224         | paper default                  | 0.683 | 0.567 | 0.858 | — |
| DINOv2 ViT-B/14 @ 224 | layers 5+11                   | 0.749 | 0.688 | 0.821 | — |
| DINOv2 ViT-S/14 @ 518 | layers 5+11 (K=1, baseline)   | 0.818 | 0.780 | 0.858 | 0.692 |
| DINOv2 ViT-S/14 @ 518 | **+ K=3 NN + Guided filter**  | **0.824** | **0.789** | **0.861** | **0.700** |

Recommended config:
[configs/patchcore_official_dinov2_518.yaml](configs/patchcore_official_dinov2_518.yaml)
+ `--num-nn 3 --guided-filter`.

### Hazelnut 6-panel composites

`image | heatmap | mask (pred) | gt | pred conf fg | pred conf bg`
— heatmap is the raw score, anchored to `train_pixel_max`; the binary
mask is the same one used for overlays, JSON, and evaluation.

| Input     | Panel |
|---|---|
| good 000  | ![](docs/samples/patchcore_k3_gf/panel/good_000_panel.png) |
| crack 000 | ![](docs/samples/patchcore_k3_gf/panel/crack_000_panel.png) |
| crack 005 | ![](docs/samples/patchcore_k3_gf/panel/crack_005_panel.png) |
| cut 001   | ![](docs/samples/patchcore_k3_gf/panel/cut_001_panel.png) |
| hole 005  | ![](docs/samples/patchcore_k3_gf/panel/hole_005_panel.png) |
| print 005 | ![](docs/samples/patchcore_k3_gf/panel/print_005_panel.png) |

## Setup

```powershell
conda env create -f environment.yml
conda activate anomalydet
python scripts/smoke_check.py    # synthetic-data sanity check
```

The smoke check trains + infers on synthetic data and confirms
torch + CUDA + the full pipeline are wired up.

## Run

One runner covers fit + inference + threshold sweep + panels +
LabelMe JSON in one shot:
[scripts/run_patchcore_official.py](scripts/run_patchcore_official.py).

```powershell
# build memory bank + run inference + emit panels + dump summary
python scripts/run_patchcore_official.py `
    --config configs/patchcore_official_dinov2_518.yaml `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category hazelnut `
    --output outputs/patchcore_k3_gf_hazelnut `
    --num-nn 3 --guided-filter `
    --threshold-target iou

# subsequent runs: reuse the saved memory bank, re-run inference only
python scripts/run_patchcore_official.py `
    --config configs/patchcore_official_dinov2_518.yaml `
    --memory-bank outputs/patchcore_k3_gf_hazelnut/memory_bank.pt `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category hazelnut `
    --output outputs/hazelnut_recall90 `
    --num-nn 3 --guided-filter `
    --threshold-target target_recall --min-recall 0.90
```

Each output directory is self-describing:

```
outputs/<run>/
  config_used.yaml             snapshot of the YAML used
  run_command.txt              full CLI + timestamp + flag values
  train_manifest.json          every train image fed into the bank
  memory_bank.pt               coreset features + backbone metadata
  summary.json                 per-image image_score + mask_pct + chosen threshold
  threshold_sweep.json         full F1 / P / R / IoU sweep
  predictions/
    <defect>_<stem>_scores.npy        raw float score map (full image res)
    <defect>_<stem>_mask.png          final binary mask
    <defect>_<stem>_overlay_mask.png  mask blended over original
    <defect>_<stem>_overlay_heatmap.png  calibrated heatmap blended over original
    <defect>_<stem>_real_gt.png       dataset GT mask (defective images only)
  panel/
    <defect>_<stem>_panel.png         6-panel composite
```

## Threshold modes

Set with `--threshold-target`. Same binary mask is produced once at
the chosen threshold and reused for every artifact — panel, overlay,
JSON, evaluation all agree.

| Mode | Picks the threshold that… | When to use |
|---|---|---|
| `f1` (default)                       | maximises pixel-F1 vs GT                              | Balanced P/R during development |
| `iou`                                | maximises pixel-IoU vs GT                             | Mask outline closer to GT |
| `target_recall` + `--min-recall r`   | is most precise while keeping pixel recall ≥ r        | DM team verifies each flagged defect; pick r=0.90 / 0.95 |
| `manual` + `--threshold v`           | uses `v` directly                                     | The right number is already known |
| **`train_p999`**                     | **= 99.9th percentile of train-pixel scores**         | **Production (no GT available)** |
| `train_pixel_max`                    | = max train pixel score                               | Strictest GT-free option |

### Production (no GT) — `train_p999`

In deployment there are no MVTec masks. `train_p999` derives the
threshold from the training pixel-score distribution alone:

```powershell
python scripts/run_patchcore_official.py `
    --config configs/patchcore_official_dinov2_518.yaml `
    --memory-bank outputs/patchcore_k3_gf_hazelnut/memory_bank.pt `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category hazelnut `
    --output outputs/hazelnut_prod_p999 `
    --num-nn 3 --guided-filter `
    --threshold-target train_p999
```

On hazelnut K=3+GF this gives `threshold ≈ 32.6` vs the GT-optimum
of `~50.9` — production threshold is **more permissive** than the
GT-optimum, catches everything the GT-tuned mask catches plus some
edge-of-normal false positives. For recall-first inspection where a
human verifies flagged regions, this is the correct side to err on.
The chosen threshold, plus `train_pixel_max` and `train_p999`, are
stored in `summary.json` for every run.

## Mechanism notes

Three orthogonal inference-time mechanisms can stack on top of the
single-scale 518 baseline. All measured on the same MVTec hazelnut
test split with `--threshold-target iou`.

- **`--num-nn 3`** (+0.4 pp F1, +0.6 pp IoU). Averages the 3 nearest
  coreset neighbours' L2 distances per patch instead of K=1 — smooths
  isolated coreset outliers. Reuses the K=1 memory bank; no retraining.
- **`--guided-filter`** (+0.2 pp F1, +0.3 pp IoU on top of K=3).
  He et al. 2010 guided filter using the grayscale of the original
  RGB as guide. Snaps the score-map gradient to image edges so the
  mask outline tracks the defect boundary. `--gf-radius 8 --gf-eps 1e-3`
  works for hazelnut; ~50 ms / image extra.
- **`--tta flips`** is a no-op on its own and *hurts* stacked with
  K=3+GF (0.824 → 0.820 F1). Hazelnut defects are directional —
  averaging the score map with horizontally / vertically flipped
  versions diffuses the response across the symmetry axis. DINOv2
  patches are already close to flip-invariant. Keep TTA off for
  hazelnut.

[scripts/run_multiscale_ensemble.py](scripts/run_multiscale_ensemble.py)
also ships an ensemble variant (averages score maps across 224 / 392 /
518 input scales). On hazelnut it scores F1=0.804 — **worse** than
the single-scale 518 K=1 baseline (0.818), because the lower-resolution
members drag the well-tuned 518 boundary back toward their noisier
maps. Kept for categories where no single scale dominates.

## Reference baseline (anomalib)

The Intel anomalib v1.2 PatchCore reference runs through
[scripts/run_anomalib.py](scripts/run_anomalib.py) and provides the
first two rows of the F1 table at the top. The implementation is the
canonical one; the F1 gap to our row at the same backbone+resolution
(0.804 → 0.818) is the K=3+GF + paper-faithful FAISS-K1 + same
threshold sweep applied across runs.

```powershell
python scripts/run_anomalib.py --preset wrn50 `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category hazelnut `
    --output outputs/anomalib_wrn50_hazelnut

python scripts/run_anomalib.py --preset dinov2 `
    --data-root "E:\dataset\mvtec_anomaly_detection_" `
    --category hazelnut --image-size 518 `
    --output outputs/anomalib_dinov2_hazelnut
```

## Repo layout

```
configs/
  patchcore_official_dinov2_518.yaml   recommended: DINOv2 ViT-S/14 @ 518
  patchcore_official_dinov2.yaml       DINOv2 ViT-B/14 @ 224
  patchcore_official_wrn50.yaml        WideResNet-50 @ 224 (paper default)
  default.yaml, dinov2.yaml,
  hazelnut*.yaml                       legacy configs (kept for archaeology)

scripts/
  run_patchcore_official.py    primary runner — train + inference + sweep
                               + panels + LabelMe JSON in one pass
  run_multiscale_ensemble.py   ensemble across input scales
  run_anomalib.py              anomalib v1.2 reference baseline
  evaluate_against_gt.py       standalone GT F1 / IoU sweep on a predictions dir
  smoke_check.py               synthetic-data sanity test (no MVTec needed)
  download_mvtec.py            fetch a single MVTec category
  compare_runs.py              merge sweep summaries into one markdown table
  sweep_thresholds.py          legacy: 8-config threshold sweep
  run_demo.ps1                 legacy end-to-end demo

src/
  models/patchcore_official.py  paper-faithful PatchCore (FAISS + k-center)
  models/feature_extractor.py   backbone factory: DINOv2 blocks vs ResNet hooks
  data/dataset.py, transforms.py
  utils/coreset.py              k-center greedy with random projection
  utils/postprocess.py          heatmap → mask → LabelMe JSON
  utils/visualize.py            calibrated heatmap normalize + overlays
  train.py, inference.py        legacy entry points (use run_patchcore_official.py)

docs/samples/patchcore_k3_gf/   panels shown above
outputs/<run>/                  self-describing artifact dir per run
```

## Branching

```powershell
git checkout dev
git pull
git checkout -b feat/your-experiment
# iterate, commit
git push -u origin feat/your-experiment
# PR: dev <- feat/your-experiment, then dev -> main
```

`main` holds the validated baseline. Experiments live on branches off
`dev` and only merge to `main` after the numbers above are verified.

## License

Internal project. MVTec AD images are subject to MVTec's own license.
DINOv2 weights are released by Meta under their own terms; see
https://github.com/facebookresearch/dinov2.
