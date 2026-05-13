"""Visualise each pipeline stage on one test image, end to end.

Outputs PNGs into <out-dir>/ to be embedded into the report:
    01_original.png            full 4096x2851 frame with the ROI bbox drawn
    02_roi_cropped.png         post-ROI-crop image (active pixel area only)
    03_tile_grid.png           cropped image with the 392/256 tile grid overlay
    04_sample_tile.png         one representative tile (~around a defect)
    05_heatmap_raw.png         stitched score map (cosine-window weighted) before post-process
    06_heatmap_guided.png      score map after guided filter
    07_mask_overlay.png        binarised mask overlaid on the original image
    08_final_panel.png         the 5-column composite from foosung_make_panels

This is single-image. Pass --image-stem to choose which test image to use.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import roi_bbox_from_image
from scripts.run_patchcore_tiled import (
    extract_tiles_from_image, predict_image_tiled, guided_filter_gray, _to_tensor,
)
from src.models.patchcore_official import PatchCoreOfficial
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--memory-bank', required=True)
    p.add_argument('--image', required=True,
                   help='Path to the test image to walk through.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--tile-size', type=int, default=392)
    p.add_argument('--tile-stride', type=int, default=256)
    p.add_argument('--threshold', type=float, default=47.0,
                   help='Score threshold for mask (use GT-tuned value).')
    p.add_argument('--bg-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--gf-radius', type=int, default=8)
    p.add_argument('--gf-eps', type=float, default=1e-3)
    p.add_argument('--num-nn', type=int, default=3)
    return p.parse_args()


def save(img_rgb, path):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f'image: {args.image}')
    orig = np.asarray(Image.open(args.image).convert('RGB'))
    H_full, W_full = orig.shape[:2]
    print(f'  full size: {W_full}x{H_full}')

    # ----- 01: original + ROI bbox drawn ---------------------------------
    x0, y0, x1, y1 = roi_bbox_from_image(orig, args.bg_threshold, args.roi_margin)
    print(f'  ROI bbox: ({x0},{y0}) -> ({x1},{y1})   size: {x1-x0}x{y1-y0}')
    stage1 = orig.copy()
    cv2.rectangle(stage1, (x0, y0), (x1, y1), (255, 0, 0), 16)
    cv2.putText(stage1, f'ROI bbox: ({x0},{y0})-({x1},{y1})',
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 4, cv2.LINE_AA)
    save(stage1, out / '01_original_with_roi_bbox.png')

    # ----- 02: ROI cropped image ------------------------------------------
    cropped = orig[y0:y1, x0:x1].copy()
    H, W = cropped.shape[:2]
    save(cropped, out / '02_roi_cropped.png')
    print(f'  cropped: {W}x{H}')

    # ----- 03: tile grid overlay -----------------------------------------
    stage3 = cropped.copy()
    # Replicate extract_tiles_from_image origin logic without filtering
    from scripts.run_patchcore_tiled import gen_tile_origins
    origins = gen_tile_origins(H, W, args.tile_size, args.tile_stride)
    print(f'  tile origins: {len(origins)} (tile={args.tile_size}, stride={args.tile_stride}, overlap={args.tile_size-args.tile_stride}px)')
    # Draw all tiles
    for i, (oy, ox) in enumerate(origins):
        cv2.rectangle(stage3, (ox, oy),
                       (ox + args.tile_size, oy + args.tile_size),
                       (0, 255, 255), 4)
    # Highlight one tile in the middle of the defect region in red
    mid_idx = len(origins) // 2
    oy_mid, ox_mid = origins[mid_idx]
    cv2.rectangle(stage3, (ox_mid, oy_mid),
                   (ox_mid + args.tile_size, oy_mid + args.tile_size),
                   (255, 0, 0), 14)
    label = (f'tile {args.tile_size}x{args.tile_size}  '
             f'stride {args.tile_stride}  overlap {args.tile_size-args.tile_stride}px  '
             f'total {len(origins)} tiles')
    cv2.putText(stage3, label, (40, 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (255, 255, 0), 4, cv2.LINE_AA)
    save(stage3, out / '03_tile_grid_overlay.png')

    # ----- 04: sample tile + its individual score map --------------------
    sample_tile = cropped[oy_mid:oy_mid+args.tile_size,
                           ox_mid:ox_mid+args.tile_size].copy()
    save(sample_tile, out / '04_sample_tile.png')

    # ----- Build the model + bank -----------------------------------------
    model = PatchCoreOfficial(
        backbone=cfg['backbone'], layers=tuple(cfg['layers']),
        input_size=args.tile_size,
        coreset_ratio=float(cfg.get('coreset_ratio', 0.1)),
        coreset_projection_dim=int(cfg.get('coreset_projection_dim', 128)),
        anomaly_score_num_nn=args.num_nn,
        device=device, fp16=False,
    )
    model.load(args.memory_bank)
    print('memory bank loaded')

    # ----- 04b: sample tile score map -------------------------------------
    with torch.no_grad():
        x = _to_tensor(sample_tile).unsqueeze(0).to(device)
        sm_low, _ = model.predict(x, target_size=(args.tile_size, args.tile_size))
    tile_score = sm_low[0]
    # colour the score map
    tile_norm = np.clip(tile_score / max(float(tile_score.max()), 1e-6), 0, 1)
    tile_heat = cv2.cvtColor(
        cv2.applyColorMap((tile_norm * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    sample_tile_overlay = cv2.addWeighted(sample_tile, 0.45, tile_heat, 0.55, 0)
    save(sample_tile_overlay, out / '04b_sample_tile_score_overlay.png')

    # ----- 05: stitched raw score map -------------------------------------
    print('running full-image tile-stitch inference...')
    score_raw = predict_image_tiled(
        model, cropped, args.tile_size, args.tile_stride,
        args.bg_threshold, min_fg_pixels=100,
        device=device, batch_size=cfg.get('batch_size', 4),
        per_tile_fg=True,
    )
    anchor = float(score_raw.max()) if score_raw.size else 1.0
    raw_norm = np.clip(score_raw / max(anchor, 1e-6), 0, 1)
    raw_heat = cv2.cvtColor(
        cv2.applyColorMap((raw_norm * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    raw_overlay = cv2.addWeighted(cropped, 0.4, raw_heat, 0.6, 0)
    save(raw_overlay, out / '05_heatmap_raw.png')

    # ----- 06: after guided filter ----------------------------------------
    score_gf = guided_filter_gray(cropped, score_raw,
                                   radius=args.gf_radius, eps=args.gf_eps)
    gf_norm = np.clip(score_gf / max(anchor, 1e-6), 0, 1)
    gf_heat = cv2.cvtColor(
        cv2.applyColorMap((gf_norm * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB)
    gf_overlay = cv2.addWeighted(cropped, 0.4, gf_heat, 0.6, 0)
    save(gf_overlay, out / '06_heatmap_guided.png')

    # ----- 07: binary mask + overlay --------------------------------------
    mask = (score_gf >= args.threshold).astype(np.uint8) * 255
    red = np.zeros_like(cropped); red[..., 0] = 255
    mask_overlay = np.where(mask[..., None] > 0,
                             (cropped * 0.45 + red * 0.55).astype(np.uint8),
                             cropped)
    save(mask_overlay, out / '07_mask_overlay.png')

    # ----- Print stats -----------------------------------------------------
    print(f'\n  raw score: min={score_raw.min():.3f}, max={score_raw.max():.3f}, '
          f'mean={score_raw.mean():.3f}')
    print(f'  after GF:  min={score_gf.min():.3f}, max={score_gf.max():.3f}')
    print(f'  threshold {args.threshold}: mask% = {100.0*(mask>0).sum()/mask.size:.2f}%')
    print(f'\nwrote {len(list(out.glob("*.png")))} stage images to {out}')


if __name__ == '__main__':
    main()
