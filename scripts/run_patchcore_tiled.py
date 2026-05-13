"""Tile-based PatchCore — train + infer at NATIVE image resolution.

The default PatchCore runner resizes 4096x2851 BMPs to 518x518 (after
optional ROI-crop). That's ~8x downsampling — fine for hazelnut-scale
defects but bleeds out thin cracks on Foosung-scale inputs.

This runner avoids that. Build the memory bank from 518x518 tiles
extracted at native resolution from training images, and at inference
tile the test image the same way, score each tile through PatchCore,
and stitch the score maps back at full resolution. Each on-part patch
ends up evaluated at its actual pixel scale.

Same output layout as scripts/run_patchcore_official.py so the
existing evaluators (`evaluate_foosung.py`, `_eval_prob_vs_gt.py`,
`foosung_pred_vs_gt_viz.py`) just work. Memory bank is saved as
memory_bank.pt and is interchangeable with the official PatchCore
format.

Unsupervised. Uses no defect labels at any stage.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, roi_bbox_from_image
from src.models.patchcore_official import PatchCoreOfficial


# ---------- args ------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--train-dir', nargs='+', required=True,
                   help='One or more folders of training (normal) images.')
    p.add_argument('--test-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--memory-bank', default=None,
                   help='Reuse an existing tiled memory_bank.pt; skip fit().')
    # tile parameters
    p.add_argument('--tile-size', type=int, default=518,
                   help='Side length of each tile (px, in the image\'s native '
                        'resolution after ROI crop).')
    p.add_argument('--tile-stride', type=int, default=384,
                   help='Stride between tile origins. tile_stride < tile_size '
                        'gives overlap; tile_stride = tile_size gives non-'
                        'overlapping tiles.')
    p.add_argument('--max-train-tiles', type=int, default=2000,
                   help='Cap on the total number of training tiles. With '
                        '32 train images x ~50 tiles each ~= 1600 tiles, '
                        'so 2000 is generous.')
    # preprocessing flags (same semantics as run_patchcore_official.py)
    p.add_argument('--roi-crop', action='store_true', default=False)
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--bg-mask', action='store_true', default=False,
                   help='Zero score where grayscale < --bg-threshold (after stitch).')
    p.add_argument('--bg-threshold', type=int, default=8)
    p.add_argument('--guided-filter', action='store_true', default=False)
    p.add_argument('--gf-radius', type=int, default=8)
    p.add_argument('--gf-eps', type=float, default=1e-3)
    # PatchCore knobs
    p.add_argument('--num-nn', type=int, default=None)
    p.add_argument('--coreset-ratio', type=float, default=None)
    # threshold
    p.add_argument('--threshold-target',
                   choices=['train_p999', 'train_pixel_max', 'manual'],
                   default='train_p999')
    p.add_argument('--threshold', type=float, default=None)
    # tile foreground filter
    p.add_argument('--min-fg-pixels', type=int, default=100,
                   help='Skip tiles whose foreground (>bg-threshold) pixel '
                        'count is below this. Avoids polluting the bank with '
                        'all-black border tiles.')
    p.add_argument('--fp16', action='store_true', default=False,
                   help='Run the DINOv2 / ResNet backbone in half precision. '
                        'Memory bank + downstream FAISS stay FP32.')
    p.add_argument('--per-tile-fg', action='store_true', default=False,
                   help='Apply per-tile foreground mask to each tile score '
                        'map BEFORE cosine-window stitching. Sharpens part '
                        'edges in the stitched output and removes '
                        'cross-tile bleeding from background regions.')
    return p.parse_args()


# ---------- tiling primitives ----------------------------------------------

def gen_tile_origins(H: int, W: int, tile: int, stride: int) -> List[Tuple[int, int]]:
    """Tile origin coordinates (y, x) covering [0, H) x [0, W) with overlap.
    The final row/column is anchored so the tile ends at the image edge."""
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if not ys or ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    if not xs or xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    ys = sorted(set(ys))
    xs = sorted(set(xs))
    return [(y, x) for y in ys for x in xs]


def extract_tiles_from_image(arr_rgb: np.ndarray, tile: int, stride: int,
                              bg_threshold: int = 8,
                              min_fg_pixels: int = 100) -> List[dict]:
    """Returns list of {'tile': (H, W, 3) uint8, 'y0', 'x0', 'fg_pixels'}."""
    H, W = arr_rgb.shape[:2]
    gray = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2GRAY)
    out = []
    for (y, x) in gen_tile_origins(H, W, tile, stride):
        sub_gray = gray[y:y + tile, x:x + tile]
        fg = int((sub_gray > bg_threshold).sum())
        if fg < min_fg_pixels:
            continue
        out.append({
            'tile': arr_rgb[y:y + tile, x:x + tile].copy(),
            'y0': int(y), 'x0': int(x), 'fg_pixels': fg,
        })
    return out


def _to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Match the standard image_transform (Resize -> ToTensor -> Normalize)
    but without resize (already tile_size). Returns (3, H, W) float."""
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


# ---------- dataset for memory-bank fit ------------------------------------

class TileBankDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], tile: int, stride: int,
                 roi_crop: bool, bg_threshold: int, min_fg_pixels: int,
                 max_tiles: int):
        self.tile = int(tile)
        self.stride = int(stride)
        self.tiles: List[dict] = []
        for p in image_paths:
            rgb = np.asarray(Image.open(p).convert('RGB'))
            if roi_crop:
                x0, y0, x1, y1 = roi_bbox_from_image(rgb, bg_threshold, 16)
                rgb = rgb[y0:y1, x0:x1]
            tiles = extract_tiles_from_image(rgb, tile, stride,
                                              bg_threshold=bg_threshold,
                                              min_fg_pixels=min_fg_pixels)
            for t in tiles:
                t['src'] = str(p)
                self.tiles.append(t)
        # cap
        if len(self.tiles) > max_tiles:
            rng = np.random.default_rng(0)
            keep = rng.choice(len(self.tiles), max_tiles, replace=False)
            self.tiles = [self.tiles[i] for i in keep]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        t = self.tiles[idx]
        x = _to_tensor(t['tile'])
        return {'image': x}


# ---------- inference: tile + stitch ---------------------------------------

@torch.no_grad()
def predict_image_tiled(model, image_rgb: np.ndarray, tile: int, stride: int,
                        bg_threshold: int, min_fg_pixels: int,
                        device: str, batch_size: int = 4,
                        per_tile_fg: bool = False) -> np.ndarray:
    """Returns score map (H, W) at the input image's resolution.

    When `per_tile_fg=True`, each tile's score map is multiplied by a
    binary foreground mask (grayscale > bg_threshold) BEFORE the cosine
    window weighting and stitching. Removes cross-tile bleeding from
    background regions at part edges; small but real F1 / IoU win.
    """
    H, W = image_rgb.shape[:2]
    tiles = extract_tiles_from_image(image_rgb, tile, stride,
                                      bg_threshold=bg_threshold,
                                      min_fg_pixels=min_fg_pixels)
    if not tiles:
        return np.zeros((H, W), dtype=np.float32)

    score_sum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    win1d = np.hanning(tile).astype(np.float32) + 1e-3
    win2d = win1d[:, None] * win1d[None, :]

    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i + batch_size]
        x = torch.stack([_to_tensor(t['tile']) for t in batch_tiles]).to(device)
        sm_low, _ = model.predict(x, target_size=(tile, tile))
        for j, t in enumerate(batch_tiles):
            y0, x0 = t['y0'], t['x0']
            sm = sm_low[j].astype(np.float32)
            if per_tile_fg:
                tile_gray = cv2.cvtColor(t['tile'], cv2.COLOR_RGB2GRAY)
                fg_mask = (tile_gray > bg_threshold).astype(np.float32)
                sm = sm * fg_mask
                score_sum[y0:y0 + tile, x0:x0 + tile] += sm * win2d
                weight[y0:y0 + tile, x0:x0 + tile] += win2d * fg_mask
            else:
                score_sum[y0:y0 + tile, x0:x0 + tile] += sm * win2d
                weight[y0:y0 + tile, x0:x0 + tile] += win2d
    score_map = score_sum / np.maximum(weight, 1e-6)
    return score_map


def guided_filter_gray(guide_rgb: np.ndarray, src: np.ndarray,
                       radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """He et al. 2010 guided filter, grayscale guide. Copied from
    run_patchcore_official.py so this script is self-contained."""
    I = cv2.cvtColor(guide_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = src.astype(np.float32)
    r = int(radius)
    ksize = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.boxFilter(I, -1, ksize)
    mean_p = cv2.boxFilter(p, -1, ksize)
    corr_Ip = cv2.boxFilter(I * p, -1, ksize)
    corr_II = cv2.boxFilter(I * I, -1, ksize)
    var_I = corr_II - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    return mean_a * I + mean_b


def list_images(directories) -> List[Path]:
    if isinstance(directories, (str, Path)):
        directories = [directories]
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    out = []
    for d in directories:
        for p in Path(d).iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                out.append(p)
    return sorted(out)


# ---------- main ------------------------------------------------------------

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}; backbone: {cfg["backbone"]}; layers: {cfg["layers"]}')
    print(f'tile_size={args.tile_size}  tile_stride={args.tile_stride}  '
          f'roi_crop={args.roi_crop}  bg_mask={args.bg_mask}')

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / 'config_used.yaml')
    with open(out_root / 'run_command.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write(f'run_at: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'tile_size: {args.tile_size}\n')
        f.write(f'tile_stride: {args.tile_stride}\n')
        f.write(f'roi_crop: {args.roi_crop}\n')
        f.write(f'bg_mask: {args.bg_mask}\n')
        f.write(f'guided_filter: {args.guided_filter}\n')
        f.write(f'num_nn: {args.num_nn or "(from config)"}\n')

    # ---- training tiles --------------------------------------------------
    train_paths = list_images(args.train_dir)
    print(f'training images: {len(train_paths)} from {len(args.train_dir)} folder(s)')
    fit_ds = TileBankDataset(
        train_paths, tile=args.tile_size, stride=args.tile_stride,
        roi_crop=args.roi_crop,
        bg_threshold=args.bg_threshold, min_fg_pixels=args.min_fg_pixels,
        max_tiles=args.max_train_tiles,
    )
    print(f'training tiles: {len(fit_ds)} (foreground-filtered, capped at '
          f'{args.max_train_tiles})')
    fit_loader = DataLoader(fit_ds, batch_size=cfg.get('batch_size', 4),
                            shuffle=False, num_workers=0,
                            pin_memory=(device == 'cuda'))

    # train manifest
    train_files = sorted(set(t['src'] for t in fit_ds.tiles))
    with open(out_root / 'train_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'tile_size': args.tile_size, 'tile_stride': args.tile_stride,
            'n_train_images': len(train_files),
            'n_train_tiles': len(fit_ds),
            'train_images': train_files,
        }, f, indent=2)

    # ---- model -----------------------------------------------------------
    if args.coreset_ratio is not None:
        cfg['coreset_ratio'] = float(args.coreset_ratio)
    num_nn = int(args.num_nn) if args.num_nn else int(cfg.get('anomaly_score_num_nn', 1))
    model = PatchCoreOfficial(
        backbone=cfg['backbone'],
        layers=tuple(cfg['layers']),
        input_size=args.tile_size,
        coreset_ratio=float(cfg.get('coreset_ratio', 0.1)),
        coreset_projection_dim=int(cfg.get('coreset_projection_dim', 128)),
        anomaly_score_num_nn=num_nn,
        device=device, fp16=args.fp16,
    )
    bank_path = out_root / 'memory_bank.pt'
    if args.memory_bank:
        print(f'loading memory bank: {args.memory_bank}')
        model.load(args.memory_bank)
    else:
        model.fit(fit_loader)
        model.save(str(bank_path))
        print(f'saved memory bank: {bank_path}')

    # ---- training-set score for threshold derivation ---------------------
    train_pix = []
    for p in train_paths:
        rgb = np.asarray(Image.open(p).convert('RGB'))
        if args.roi_crop:
            x0, y0, x1, y1 = roi_bbox_from_image(rgb, args.bg_threshold, args.roi_margin)
            rgb = rgb[y0:y1, x0:x1]
        sm = predict_image_tiled(model, rgb, args.tile_size, args.tile_stride,
                                  args.bg_threshold, args.min_fg_pixels,
                                  device=device, batch_size=cfg.get('batch_size', 4),
                                  per_tile_fg=args.per_tile_fg)
        train_pix.append(sm.flatten())
    train_pix = np.concatenate(train_pix)
    train_pixel_max = float(train_pix.max())
    train_p999 = float(np.percentile(train_pix, 99.9))
    print(f'train_pixel_max={train_pixel_max:.4f}  train_p99.9={train_p999:.4f}')

    if args.threshold_target == 'manual':
        if args.threshold is None:
            raise SystemExit('--threshold-target manual needs --threshold')
        chosen_thr = float(args.threshold)
    elif args.threshold_target == 'train_pixel_max':
        chosen_thr = train_pixel_max
    else:
        chosen_thr = train_p999
    print(f'chosen threshold ({args.threshold_target}): {chosen_thr:.4f}')

    # ---- test inference --------------------------------------------------
    test_paths = list_images([args.test_dir])
    pred_dir = out_root / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    print(f'\ntest images: {len(test_paths)}')
    for p in tqdm(test_paths, desc='tile-stitch inference'):
        rgb = np.asarray(Image.open(p).convert('RGB'))
        if args.roi_crop:
            x0, y0, x1, y1 = roi_bbox_from_image(rgb, args.bg_threshold, args.roi_margin)
            rgb = rgb[y0:y1, x0:x1]
        sm = predict_image_tiled(model, rgb, args.tile_size, args.tile_stride,
                                  args.bg_threshold, args.min_fg_pixels,
                                  device=device, batch_size=cfg.get('batch_size', 4),
                                  per_tile_fg=args.per_tile_fg)
        if args.bg_mask:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            sm = np.where(gray < args.bg_threshold, 0.0, sm).astype(np.float32)
        if args.guided_filter:
            sm = guided_filter_gray(rgb, sm, radius=args.gf_radius, eps=args.gf_eps)
        mask = (sm >= chosen_thr).astype(np.uint8) * 255
        stem = f'test_{p.stem}'
        np.save(str(pred_dir / f'{stem}_scores.npy'), sm.astype(np.float32))
        cv2.imwrite(str(pred_dir / f'{stem}_mask.png'), mask)
        # overlay heatmap (colour) + mask (red)
        sm_norm = np.clip(sm / max(train_pixel_max, 1e-6), 0, 1)
        u8 = (sm_norm * 255).astype(np.uint8)
        heat = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay_h = cv2.addWeighted(rgb, 0.5, heat, 0.5, 0)
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_heatmap.png'),
                    cv2.cvtColor(overlay_h, cv2.COLOR_RGB2BGR))
        red = np.zeros_like(rgb); red[..., 0] = 255
        overlay_m = np.where(mask[..., None] > 0,
                              (rgb * 0.5 + red * 0.5).astype(np.uint8), rgb)
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_mask.png'),
                    cv2.cvtColor(overlay_m, cv2.COLOR_RGB2BGR))
        summary.append({
            'image_path': str(p),
            'defect_type': 'test',
            'image_score': float(sm.max()),
            'mask_pct': float(100.0 * (mask > 0).sum() / mask.size),
        })

    # Per-defect summary
    print(f"\n{'defect':14s} {'n':>3s} {'mask% mean':>11s} {'mask% max':>10s}")
    by = {'test': summary}
    for d in sorted(by):
        rows = by[d]
        mp = np.array([r['mask_pct'] for r in rows])
        sc = np.array([r['image_score'] for r in rows])
        print(f"  {d:12s} {len(rows):>3d} {mp.mean():>10.2f}% {mp.max():>9.2f}%  "
              f"img_score mean={sc.mean():>6.2f}")

    with open(out_root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'backbone': cfg['backbone'], 'layers': list(cfg['layers']),
            'tile_size': args.tile_size, 'tile_stride': args.tile_stride,
            'n_train_tiles': len(fit_ds),
            'train_pixel_max': train_pixel_max,
            'train_p999': train_p999,
            'threshold_meta': {'mode': args.threshold_target,
                               'threshold': chosen_thr},
            'predictions': summary,
        }, f, indent=2)


if __name__ == '__main__':
    main()
