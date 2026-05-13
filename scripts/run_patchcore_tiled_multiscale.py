"""Multi-scale tile-based PatchCore ensemble.

Trains one tile-based memory bank per `--scales` entry (e.g. 392 / 518 /
770). At inference, each test image is scored by every bank, the score
maps are normalised by their respective `train_pixel_max`, and then
averaged at the full image resolution. The final mask is thresholded
from that averaged map.

Same input/output layout as scripts/run_patchcore_tiled.py so the
existing GT evaluators work on the ensemble output too.

Unsupervised. No defect labels at any stage.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.transforms import roi_bbox_from_image
from src.models.patchcore_official import PatchCoreOfficial
from scripts.run_patchcore_tiled import (
    TileBankDataset,
    predict_image_tiled,
    guided_filter_gray,
    list_images,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--train-dir', nargs='+', required=True)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scales', type=int, nargs='+', default=[392, 518, 770],
                   help='Tile sizes for the ensemble. All must be divisible '
                        'by the backbone patch size (14 for DINOv2).')
    p.add_argument('--stride-ratio', type=float, default=0.75,
                   help='Tile stride = tile_size * stride_ratio.')
    p.add_argument('--max-train-tiles', type=int, default=1500,
                   help='Per-scale cap on training tiles.')
    p.add_argument('--bank-cache-root', default=None,
                   help='Optional folder of pre-built memory_bank_<scale>.pt '
                        'to reuse across runs.')
    p.add_argument('--roi-crop', action='store_true', default=False)
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--bg-mask', action='store_true', default=False)
    p.add_argument('--bg-threshold', type=int, default=8)
    p.add_argument('--guided-filter', action='store_true', default=False)
    p.add_argument('--gf-radius', type=int, default=8)
    p.add_argument('--gf-eps', type=float, default=1e-3)
    p.add_argument('--num-nn', type=int, default=None)
    p.add_argument('--coreset-ratio', type=float, default=0.02)
    p.add_argument('--min-fg-pixels', type=int, default=100)
    p.add_argument('--normalize', choices=['tp_max', 'p999', 'none'],
                   default='tp_max',
                   help='Per-scale normalisation before averaging. tp_max '
                        '= divide by per-scale train_pixel_max (default).')
    p.add_argument('--threshold-target',
                   choices=['train_p999', 'train_pixel_max', 'manual'],
                   default='train_p999')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--fp16', action='store_true', default=False,
                   help='Run backbone in half precision (~1.7x faster).')
    return p.parse_args()


def build_scale_bank(cfg, scale, train_paths, args, device, out_root):
    bank_name = f'memory_bank_{scale}.pt'
    bank_path = out_root / bank_name
    cached = (Path(args.bank_cache_root) / bank_name
              if args.bank_cache_root else None)

    stride = max(int(scale * args.stride_ratio), scale // 2)
    ds = TileBankDataset(
        train_paths, tile=scale, stride=stride,
        roi_crop=args.roi_crop,
        bg_threshold=args.bg_threshold,
        min_fg_pixels=args.min_fg_pixels,
        max_tiles=args.max_train_tiles,
    )
    print(f'[scale {scale}] {len(ds)} training tiles (stride={stride})')
    loader = DataLoader(ds, batch_size=cfg.get('batch_size', 4),
                        shuffle=False, num_workers=0,
                        pin_memory=(device == 'cuda'))

    num_nn = int(args.num_nn) if args.num_nn else int(cfg.get('anomaly_score_num_nn', 1))
    model = PatchCoreOfficial(
        backbone=cfg['backbone'], layers=tuple(cfg['layers']),
        input_size=scale,
        coreset_ratio=float(args.coreset_ratio),
        coreset_projection_dim=int(cfg.get('coreset_projection_dim', 128)),
        anomaly_score_num_nn=num_nn,
        device=device, fp16=getattr(args, 'fp16', False),
    )
    if cached is not None and cached.exists():
        print(f'[scale {scale}] loading cached bank: {cached}')
        model.load(str(cached))
        bank_path.write_bytes(cached.read_bytes())
    elif bank_path.exists():
        print(f'[scale {scale}] loading bank: {bank_path}')
        model.load(str(bank_path))
    else:
        model.fit(loader)
        model.save(str(bank_path))
        print(f'[scale {scale}] saved {bank_path}')
    return model, stride


def per_scale_train_calibration(model, scale, stride, train_paths, args, device, cfg):
    """Run train-set tile-stitch inference to derive train_pixel_max etc."""
    pix = []
    for p in train_paths:
        rgb = np.asarray(Image.open(p).convert('RGB'))
        if args.roi_crop:
            x0, y0, x1, y1 = roi_bbox_from_image(rgb, args.bg_threshold, args.roi_margin)
            rgb = rgb[y0:y1, x0:x1]
        sm = predict_image_tiled(model, rgb, scale, stride,
                                  args.bg_threshold, args.min_fg_pixels,
                                  device=device,
                                  batch_size=cfg.get('batch_size', 4))
        pix.append(sm.flatten())
    pix = np.concatenate(pix)
    return float(pix.max()), float(np.percentile(pix, 99.9)), \
           float(pix.mean()), float(pix.std())


def normalise(sm, info, how):
    if how == 'tp_max':
        return sm / max(info['train_pixel_max'], 1e-6)
    if how == 'p999':
        return sm / max(info['train_p999'], 1e-6)
    return sm


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_root / 'config_used.yaml')
    with open(out_root / 'run_command.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write(f'run_at: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'scales: {args.scales}\n')
        f.write(f'stride_ratio: {args.stride_ratio}\n')
        f.write(f'normalize: {args.normalize}\n')

    train_paths = list_images(args.train_dir)
    test_paths = list_images([args.test_dir])
    print(f'train images: {len(train_paths)}; test images: {len(test_paths)}')
    print(f'scales: {args.scales}')

    # ---- per-scale banks ------------------------------------------------
    scale_state = {}
    for scale in args.scales:
        print(f'\n=== scale {scale} ===')
        model, stride = build_scale_bank(cfg, scale, train_paths, args, device, out_root)
        tp_max, tp999, mean_, std_ = per_scale_train_calibration(
            model, scale, stride, train_paths, args, device, cfg)
        print(f'  train_pixel_max={tp_max:.4f}  p99.9={tp999:.4f}')
        scale_state[scale] = dict(
            model=model, stride=stride,
            train_pixel_max=tp_max, train_p999=tp999,
            mean=mean_, std=std_,
        )

    # ---- ensemble train calibration (avg of normalised train maps) ------
    ensemble_train_pix = []
    print('\n=== ensemble train calibration ===')
    for p in train_paths:
        rgb = np.asarray(Image.open(p).convert('RGB'))
        if args.roi_crop:
            x0, y0, x1, y1 = roi_bbox_from_image(rgb, args.roi_threshold, args.roi_margin)
            rgb = rgb[y0:y1, x0:x1]
        normed_maps = []
        for scale, info in scale_state.items():
            sm = predict_image_tiled(info['model'], rgb, scale, info['stride'],
                                      args.bg_threshold, args.min_fg_pixels,
                                      device=device,
                                      batch_size=cfg.get('batch_size', 4))
            normed_maps.append(normalise(sm, info, args.normalize))
        avg = np.mean(np.stack(normed_maps, axis=0), axis=0)
        ensemble_train_pix.append(avg.flatten())
    ensemble_train_pix = np.concatenate(ensemble_train_pix)
    ensemble_train_max = float(ensemble_train_pix.max())
    ensemble_train_p999 = float(np.percentile(ensemble_train_pix, 99.9))
    print(f'ensemble train_pixel_max={ensemble_train_max:.4f}  '
          f'p99.9={ensemble_train_p999:.4f}')

    if args.threshold_target == 'manual':
        chosen_thr = float(args.threshold)
    elif args.threshold_target == 'train_pixel_max':
        chosen_thr = ensemble_train_max
    else:
        chosen_thr = ensemble_train_p999
    print(f'chosen threshold ({args.threshold_target}): {chosen_thr:.4f}')

    # ---- test inference --------------------------------------------------
    pred_dir = out_root / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    print(f'\n=== ensemble test inference ===')
    for p in tqdm(test_paths, desc='ensemble test'):
        rgb = np.asarray(Image.open(p).convert('RGB'))
        if args.roi_crop:
            x0, y0, x1, y1 = roi_bbox_from_image(rgb, args.roi_threshold, args.roi_margin)
            rgb = rgb[y0:y1, x0:x1]
        normed = []
        for scale, info in scale_state.items():
            sm = predict_image_tiled(info['model'], rgb, scale, info['stride'],
                                      args.bg_threshold, args.min_fg_pixels,
                                      device=device,
                                      batch_size=cfg.get('batch_size', 4))
            normed.append(normalise(sm, info, args.normalize))
        avg = np.mean(np.stack(normed, axis=0), axis=0)
        if args.bg_mask:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            avg = np.where(gray < args.bg_threshold, 0.0, avg).astype(np.float32)
        if args.guided_filter:
            avg = guided_filter_gray(rgb, avg, radius=args.gf_radius,
                                      eps=args.gf_eps)
        mask = (avg >= chosen_thr).astype(np.uint8) * 255
        stem = f'test_{p.stem}'
        np.save(str(pred_dir / f'{stem}_scores.npy'), avg.astype(np.float32))
        cv2.imwrite(str(pred_dir / f'{stem}_mask.png'), mask)
        sm_norm = np.clip(avg / max(ensemble_train_max, 1e-6), 0, 1)
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
            'image_score': float(avg.max()),
            'mask_pct': float(100.0 * (mask > 0).sum() / mask.size),
        })

    by = {'test': summary}
    for d in sorted(by):
        rows = by[d]
        mp = np.array([r['mask_pct'] for r in rows])
        sc = np.array([r['image_score'] for r in rows])
        print(f"  {d:12s} {len(rows):>3d} mask% mean={mp.mean():>5.2f}% "
              f"max={mp.max():>5.2f}%  img_score mean={sc.mean():>6.2f}")

    with open(out_root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'backbone': cfg['backbone'], 'layers': list(cfg['layers']),
            'scales': args.scales, 'stride_ratio': args.stride_ratio,
            'normalize': args.normalize,
            'per_scale': {str(s): dict(stride=v['stride'],
                                        train_pixel_max=v['train_pixel_max'],
                                        train_p999=v['train_p999'])
                          for s, v in scale_state.items()},
            'train_pixel_max': ensemble_train_max,
            'train_p999': ensemble_train_p999,
            'threshold_meta': {'mode': args.threshold_target,
                               'threshold': chosen_thr},
            'predictions': summary,
        }, f, indent=2)


if __name__ == '__main__':
    main()
