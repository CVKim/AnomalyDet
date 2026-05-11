"""Multi-scale PatchCore ensemble.

Builds one PatchCore memory bank per input scale (e.g., 224 / 392 / 518)
and averages the per-scale score maps at the original image resolution.
At each scale, the algorithm is exactly the paper-faithful PatchCore
pipeline from src.models.patchcore_official (frozen DINOv2 backbone,
3x3 aggregation, k-center coreset, FAISS K=1, bilinear upsample).

Usage:
    python scripts/run_multiscale_ensemble.py `
        --base-config configs/patchcore_official_dinov2_518.yaml `
        --scales 224 392 518 `
        --data-root "E:\\dataset\\mvtec_anomaly_detection_" `
        --category hazelnut `
        --output outputs/patchcore_multiscale_hazelnut `
        --threshold-target iou
"""
import argparse
import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import MVTecDataset
from src.data.transforms import build_image_transform, build_train_transform
from src.models.patchcore_official import PatchCoreOfficial
# Reuse find_threshold, make_6panel, overlay helpers from the single-scale runner.
from scripts.run_patchcore_official import (
    find_threshold, make_6panel, overlay_heatmap_rgb, overlay_mask_rgb,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base-config', required=True,
                   help='Template YAML; input_size is overridden per scale.')
    p.add_argument('--scales', type=int, nargs='+', required=True,
                   help='Input sizes for the ensemble members, e.g. 224 392 518')
    p.add_argument('--data-root', required=True)
    p.add_argument('--category', default='hazelnut')
    p.add_argument('--output', required=True)
    p.add_argument('--threshold-target',
                   choices=['f1', 'iou', 'target_recall', 'manual', 'train_p999'],
                   default='iou')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--min-recall', type=float, default=0.90)
    p.add_argument('--bank-cache-root', type=str, default=None,
                   help='Optional folder of existing memory_bank_<scale>.pt '
                        'to reuse across runs.')
    p.add_argument('--normalize', choices=['none', 'zscore', 'tp_max'],
                   default='tp_max',
                   help='How to bring different scales onto a common scale '
                        'before averaging. tp_max divides by per-scale '
                        'train_pixel_max, zscore subtracts mean and divides '
                        'by std.')
    return p.parse_args()


def build_or_load_scale(cfg, scale, args, device, out_root):
    """Build or load the PatchCore memory bank for one input scale.
    Returns the model + train_pixel_max + train_p999."""
    cfg = copy.deepcopy(cfg)
    cfg['input_size'] = int(scale)

    bank_name = f'memory_bank_{scale}.pt'
    bank_path = out_root / bank_name
    cached = (Path(args.bank_cache_root) / bank_name
              if args.bank_cache_root else None)

    fit_transform = build_train_transform(
        cfg['input_size'],
        augment=bool(cfg.get('train_augment', False)),
    )
    fit_ds = MVTecDataset(args.data_root, args.category,
                          split='train', transform=fit_transform,
                          repeat=int(cfg.get('train_repeat', 1)))
    bs = max(1, int(cfg.get('batch_size', 8)))
    if scale >= 392:
        bs = min(bs, 4)
    if scale >= 518:
        bs = min(bs, 2)
    fit_loader = DataLoader(fit_ds, batch_size=bs, shuffle=False,
                            num_workers=cfg.get('num_workers', 4),
                            pin_memory=(device == 'cuda'))

    model = PatchCoreOfficial(
        backbone=cfg['backbone'],
        layers=tuple(cfg['layers']),
        input_size=cfg['input_size'],
        coreset_ratio=cfg['coreset_ratio'],
        coreset_projection_dim=cfg.get('coreset_projection_dim', 128),
        anomaly_score_num_nn=cfg.get('anomaly_score_num_nn', 1),
        device=device,
    )

    if cached is not None and cached.exists():
        print(f'[scale {scale}] loading cached bank: {cached}')
        model.load(str(cached))
        bank_path.write_bytes(cached.read_bytes())
    elif bank_path.exists():
        print(f'[scale {scale}] loading bank: {bank_path}')
        model.load(str(bank_path))
    else:
        print(f'[scale {scale}] fitting fresh bank')
        model.fit(fit_loader)
        model.save(str(bank_path))

    # Calibrate train_pixel_max for this scale (used for normalisation).
    train_eval_ds = MVTecDataset(args.data_root, args.category,
                                  split='train',
                                  transform=build_image_transform(cfg['input_size']))
    train_eval_loader = DataLoader(train_eval_ds, batch_size=bs,
                                   num_workers=cfg.get('num_workers', 4),
                                   shuffle=False, pin_memory=(device == 'cuda'))
    pix_vals = []
    for batch in train_eval_loader:
        sm_low, _ = model.predict(batch['image'].to(device, non_blocking=True),
                                  target_size=(cfg['input_size'], cfg['input_size']))
        pix_vals.append(sm_low.flatten())
    pix_vals = np.concatenate(pix_vals)
    train_pixel_max = float(pix_vals.max())
    train_p999 = float(np.percentile(pix_vals, 99.9))
    train_mean = float(pix_vals.mean())
    train_std = float(pix_vals.std())
    print(f'[scale {scale}] train_pixel_max={train_pixel_max:.4f}  '
          f'train_p99.9={train_p999:.4f}')
    return model, dict(input_size=cfg['input_size'],
                       train_pixel_max=train_pixel_max,
                       train_p999=train_p999,
                       train_mean=train_mean, train_std=train_std,
                       batch_size=bs)


def predict_at_scale(model, args, scale_info, device, cfg):
    """Run inference at one scale; return list of (path, defect, sm_orig_res, image_score, gt)."""
    cfg = copy.deepcopy(cfg)
    cfg['input_size'] = scale_info['input_size']
    transform = build_image_transform(cfg['input_size'])
    test_ds = MVTecDataset(args.data_root, args.category, split='test',
                           transform=transform)
    bs = scale_info['batch_size']
    loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                        num_workers=cfg.get('num_workers', 4),
                        pin_memory=(device == 'cuda'))
    out = []
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        sm_low, img_scores = model.predict(
            images, target_size=(cfg['input_size'], cfg['input_size']))
        for i in range(images.shape[0]):
            img_path = batch['image_path'][i]
            defect = batch['defect_type'][i]
            stem = f'{defect}_{Path(img_path).stem}'
            orig = np.array(Image.open(img_path).convert('RGB'))
            H, W = orig.shape[:2]
            sm = cv2.resize(sm_low[i].astype(np.float32), (W, H),
                            interpolation=cv2.INTER_LINEAR)
            gt = None
            mp = batch.get('mask_path', [''])[i]
            if mp:
                gt_arr = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if gt_arr is not None:
                    if gt_arr.shape != (H, W):
                        gt_arr = cv2.resize(gt_arr, (W, H),
                                            interpolation=cv2.INTER_NEAREST)
                    gt = gt_arr
            out.append(dict(image_path=img_path, defect_type=defect, stem=stem,
                            orig=orig, sm=sm, gt=gt,
                            image_score=float(img_scores[i])))
    return out


def normalize_map(sm, info, how):
    if how == 'none':
        return sm
    if how == 'tp_max':
        anchor = max(info['train_pixel_max'], 1e-6)
        return sm / anchor
    if how == 'zscore':
        return (sm - info['train_mean']) / max(info['train_std'], 1e-6)
    return sm


def main():
    args = parse_args()
    with open(args.base_config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.base_config, out_root / 'base_config_used.yaml')
    with open(out_root / 'run_command.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write(f'run_at: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'scales: {args.scales}\n')
        f.write(f'normalize: {args.normalize}\n')
        f.write(f'threshold_target: {args.threshold_target}\n')

    # ---- Per-scale memory banks + train calibration -----------------------
    print(f'\n=== building/loading memory banks for scales {args.scales} ===')
    scale_models = {}
    scale_infos = {}
    for scale in args.scales:
        m, info = build_or_load_scale(cfg, scale, args, device, out_root)
        scale_models[scale] = m
        scale_infos[scale] = info

    # ---- Predict at each scale and ensemble -------------------------------
    print(f'\n=== running ensemble inference (normalize={args.normalize}) ===')
    per_scale_records = {scale: predict_at_scale(scale_models[scale], args,
                                                  scale_infos[scale], device, cfg)
                         for scale in args.scales}
    n_test = len(per_scale_records[args.scales[0]])
    print(f'test images: {n_test}')

    # Re-order: per-image list of (scale -> record). Assumes loader order matches.
    ensemble_records = []
    for i in range(n_test):
        per_scale = {scale: per_scale_records[scale][i] for scale in args.scales}
        normed_maps = [normalize_map(per_scale[scale]['sm'], scale_infos[scale],
                                     args.normalize)
                       for scale in args.scales]
        avg_map = np.mean(np.stack(normed_maps, axis=0), axis=0)
        ref = per_scale[args.scales[0]]
        ensemble_records.append(dict(
            image_path=ref['image_path'],
            defect_type=ref['defect_type'],
            stem=ref['stem'],
            orig=ref['orig'],
            sm=avg_map,
            gt=ref['gt'],
            image_score=float(avg_map.max()),
        ))

    # train_pixel_max for the ensemble = max ensemble-score on train data
    print('\n=== ensemble train calibration (for viz anchor) ===')
    train_eval_ds = {scale: MVTecDataset(
        args.data_root, args.category, split='train',
        transform=build_image_transform(scale)) for scale in args.scales}
    loaders = {scale: DataLoader(train_eval_ds[scale],
                                 batch_size=scale_infos[scale]['batch_size'],
                                 num_workers=cfg.get('num_workers', 4),
                                 shuffle=False, pin_memory=(device == 'cuda'))
               for scale in args.scales}
    # We need every train image's score at every scale, in matched order.
    # MVTecDataset is deterministic so iterating per-scale in lockstep works.
    train_scores_per_scale = {scale: [] for scale in args.scales}
    for scale in args.scales:
        for batch in loaders[scale]:
            sm_low, _ = scale_models[scale].predict(
                batch['image'].to(device, non_blocking=True),
                target_size=(scale, scale))
            for i in range(sm_low.shape[0]):
                train_scores_per_scale[scale].append(sm_low[i].copy())
    ensemble_train_max = 0.0
    sample_size = min(len(train_scores_per_scale[args.scales[0]]), 60)
    ensemble_train_vals = []
    for i in range(sample_size):
        maps = []
        for scale in args.scales:
            sm = train_scores_per_scale[scale][i]
            # bring all scales to a common 256x256 grid then normalise
            sm_r = cv2.resize(sm.astype(np.float32), (256, 256),
                              interpolation=cv2.INTER_LINEAR)
            maps.append(normalize_map(sm_r, scale_infos[scale], args.normalize))
        avg = np.mean(np.stack(maps, axis=0), axis=0)
        ensemble_train_vals.append(avg.flatten())
        ensemble_train_max = max(ensemble_train_max, float(avg.max()))
    ensemble_train_vals = np.concatenate(ensemble_train_vals)
    ensemble_train_p999 = float(np.percentile(ensemble_train_vals, 99.9))
    print(f'ensemble train_pixel_max={ensemble_train_max:.4f}  '
          f'train_p99.9={ensemble_train_p999:.4f}')

    # ---- Threshold tuning --------------------------------------------------
    score_maps_full = [r['sm'] for r in ensemble_records]
    gt_masks_full = [r['gt'] for r in ensemble_records]
    sweep = None
    if args.threshold_target == 'manual':
        chosen_thr = float(args.threshold)
        thr_meta = dict(mode='manual', threshold=chosen_thr)
    elif args.threshold_target == 'train_p999':
        chosen_thr = ensemble_train_p999
        thr_meta = dict(mode='train_p999', threshold=chosen_thr)
    else:
        best, sweep = find_threshold(score_maps_full, gt_masks_full,
                                     mode=args.threshold_target,
                                     min_recall=args.min_recall)
        if best is None:
            chosen_thr = ensemble_train_p999
            thr_meta = dict(mode='train_p99.9 (fallback)', threshold=chosen_thr)
        else:
            chosen_thr = best['threshold']
            thr_meta = dict(mode=args.threshold_target, **best)
            print(f"GT-tuned ensemble threshold: {chosen_thr:.4f}  "
                  f"F1={best['f1']:.4f}  P={best['precision']:.4f}  "
                  f"R={best['recall']:.4f}  IoU={best.get('iou', 0.0):.4f}")
    if sweep is not None:
        with open(out_root / 'threshold_sweep.json', 'w', encoding='utf-8') as f:
            json.dump(dict(best=thr_meta, sweep=sweep,
                           ensemble_train_pixel_max=ensemble_train_max,
                           ensemble_train_p999=ensemble_train_p999), f, indent=2)

    # ---- Write artifacts ---------------------------------------------------
    pred_dir = out_root / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_root / 'panel'
    panel_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for r in ensemble_records:
        sm = r['sm']
        orig = r['orig']
        H, W = orig.shape[:2]
        mask = (sm >= chosen_thr).astype(np.uint8) * 255
        stem = r['stem']

        cv2.imwrite(str(pred_dir / f'{stem}_mask.png'), mask)
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_mask.png'),
                    cv2.cvtColor(overlay_mask_rgb(orig, mask), cv2.COLOR_RGB2BGR))
        ov_hm = overlay_heatmap_rgb(orig, sm, ensemble_train_max)
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_heatmap.png'),
                    cv2.cvtColor(ov_hm, cv2.COLOR_RGB2BGR))
        np.save(str(pred_dir / f'{stem}_scores.npy'), sm.astype(np.float32))
        if r['gt'] is not None and (r['gt'] > 0).any():
            cv2.imwrite(str(pred_dir / f'{stem}_real_gt.png'), r['gt'])

        panel = make_6panel(orig, mask, r['gt'], sm, ensemble_train_max,
                            panel_size=cfg.get('panel_size', 320))
        cv2.imwrite(str(panel_dir / f'{stem}_panel.png'),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        summary.append(dict(image_path=r['image_path'],
                            defect_type=r['defect_type'],
                            image_score=r['image_score'],
                            mask_pct=float(100.0 * (mask > 0).sum() / mask.size)))

    by = {}
    for s in summary:
        by.setdefault(s['defect_type'], []).append(s)
    print()
    print(f"{'defect':14s} {'n':>3s} {'mask% mean':>11s} {'mask% max':>10s}  "
          f"{'img_score mean':>14s}")
    for d in sorted(by):
        rows = by[d]
        mp = np.array([r['mask_pct'] for r in rows])
        sc = np.array([r['image_score'] for r in rows])
        print(f"  {d:12s} {len(rows):>3d} {mp.mean():>10.2f}% "
              f"{mp.max():>9.2f}% {sc.mean():>13.2f}")

    with open(out_root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(dict(
            backbone=cfg['backbone'], layers=list(cfg['layers']),
            scales=list(args.scales),
            normalize=args.normalize,
            coreset_ratio=cfg['coreset_ratio'],
            anomaly_score_num_nn=cfg.get('anomaly_score_num_nn', 1),
            per_scale=scale_infos,
            ensemble_train_pixel_max=ensemble_train_max,
            ensemble_train_p999=ensemble_train_p999,
            threshold_meta=thr_meta,
            predictions=summary,
        ), f, indent=2, default=str)


if __name__ == '__main__':
    main()
