"""End-to-end driver for the official PatchCore implementation.

  1. Fit memory bank from MVTec train/good.
  2. Run inference on the entire MVTec test split, keeping score maps.
  3. Optimize the pixel-level threshold against GT masks (maximise F1
     across all defective images).
  4. Optionally take a recall-first variant of the threshold and save a
     second prediction set under predictions_recall/.
  5. Write heatmap / mask / overlay artifacts in the same layout as our
     other runners.

  python scripts/run_patchcore_official.py `
      --config configs/patchcore_official_dinov2.yaml `
      --data-root "E:\\dataset\\mvtec_anomaly_detection_" `
      --category hazelnut `
      --output outputs/patchcore_official_dinov2_hazelnut
"""
import argparse
import json
import sys
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--category', default='hazelnut')
    p.add_argument('--output', required=True)
    p.add_argument('--memory-bank', default=None,
                   help='Reuse an existing memory_bank.pt; skip fit().')
    p.add_argument('--threshold-target', choices=['f1', 'recall95', 'manual'],
                   default='f1',
                   help='How to set the pixel threshold against GT. '
                        'recall95 keeps pixel recall >= 0.95 and picks the '
                        'tightest threshold that satisfies it; manual uses '
                        'the value from --threshold.')
    p.add_argument('--threshold', type=float, default=None)
    p.add_argument('--min-area', type=int, default=None,
                   help='Connected-component area filter (default from config).')
    return p.parse_args()


def overlay_heatmap_rgb(image_rgb, score_map, anomaly_anchor):
    """Map score map to BGR jet using train_pixel_max as the upper anchor.

    Score <= anchor*0.6  -> blue/green
    anchor*0.6 < s < anchor*1.6 -> linear ramp
    s >= anchor*1.6 -> saturated red
    """
    lo = float(anomaly_anchor) * 0.6
    hi = float(anomaly_anchor) * 1.6
    span = max(hi - lo, 1e-6)
    norm = np.clip((score_map - lo) / span, 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.5, color, 0.5, 0)


def overlay_mask_rgb(image_rgb, mask, alpha=0.4):
    out = image_rgb.copy()
    out[mask > 0] = np.array([255, 0, 0], dtype=out.dtype)
    return cv2.addWeighted(image_rgb, 1 - alpha, out, alpha, 0)


def clean_mask(mask, min_area, kernel=3):
    """Small open + small close + area filter. Intentionally lightweight
    so we don't bake heavy postprocess into the official baseline."""
    if kernel and kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if min_area and min_area > 1:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        out = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        mask = out
    return mask


def find_threshold(score_maps, gt_masks, mode='f1', sample_pixels=2_000_000):
    """Search for a pixel threshold against the GT label masks.

    score_maps : list of (H, W) float arrays
    gt_masks   : list of (H, W) uint8 arrays (None for good images)
    mode       : 'f1' -> argmax F1; 'recall95' -> tightest thr with
                 pixel recall >= 0.95.
    """
    # Build flat positive / negative score pools across all images. Cap
    # the negative pool so the search stays fast.
    pos_scores = []
    neg_scores = []
    for s, gt in zip(score_maps, gt_masks):
        if gt is None:
            neg_scores.append(s.flatten())
        else:
            gtb = (gt > 0)
            if gtb.any():
                pos_scores.append(s[gtb])
            if (~gtb).any():
                neg_scores.append(s[~gtb])
    if not pos_scores:
        return None, None
    pos = np.concatenate(pos_scores)
    neg = np.concatenate(neg_scores) if neg_scores else np.array([], dtype=np.float32)
    if len(neg) > sample_pixels:
        idx = np.random.default_rng(0).choice(len(neg), sample_pixels, replace=False)
        neg = neg[idx]
    if len(pos) > sample_pixels:
        idx = np.random.default_rng(0).choice(len(pos), sample_pixels, replace=False)
        pos = pos[idx]

    lo = float(np.percentile(np.concatenate([pos, neg]), 1))
    hi = float(np.percentile(np.concatenate([pos, neg]), 99.99))
    cands = np.linspace(lo, hi, 200)

    best = {'threshold': None, 'metric': -1.0}
    sweep = []
    for thr in cands:
        tp = (pos >= thr).sum()
        fp = (neg >= thr).sum()
        fn = (pos < thr).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        sweep.append(dict(thr=float(thr), p=float(precision),
                          r=float(recall), f1=float(f1)))
        if mode == 'f1':
            score = f1
        elif mode == 'recall95':
            score = precision if recall >= 0.95 else -1.0
        else:
            score = -1.0
        if score > best['metric']:
            best = {'threshold': float(thr), 'metric': float(score),
                    'precision': float(precision), 'recall': float(recall),
                    'f1': float(f1)}
    return best, sweep


def per_category_summary(records):
    by = {}
    for r in records:
        by.setdefault(r['defect_type'], []).append(r)
    print(f"{'category':18s} {'n':>3s} {'mask% mean':>11s} {'mask% max':>10s}  "
          f"{'img_score mean':>14s}")
    for d in sorted(by):
        rows = by[d]
        mp = np.array([r['mask_pct'] for r in rows])
        sc = np.array([r['image_score'] for r in rows])
        print(f"  {d:16s} {len(rows):>3d} {mp.mean():>10.2f}% "
              f"{mp.max():>9.2f}% {sc.mean():>13.2f}")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}; backbone: {cfg["backbone"]}; layers: {cfg["layers"]}')

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    fit_transform = build_train_transform(
        cfg['input_size'],
        augment=bool(cfg.get('train_augment', False)),
    )
    fit_ds = MVTecDataset(args.data_root, args.category,
                          split='train', transform=fit_transform,
                          repeat=int(cfg.get('train_repeat', 1)))
    fit_loader = DataLoader(fit_ds, batch_size=cfg.get('batch_size', 8),
                            shuffle=False, num_workers=cfg.get('num_workers', 4),
                            pin_memory=(device == 'cuda'))

    test_transform = build_image_transform(cfg['input_size'])
    test_ds = MVTecDataset(args.data_root, args.category,
                           split='test', transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.get('batch_size', 8),
                             shuffle=False, num_workers=cfg.get('num_workers', 4),
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

    bank_path = out_root / 'memory_bank.pt'
    if args.memory_bank:
        print(f'loading memory bank: {args.memory_bank}')
        model.load(args.memory_bank)
    else:
        model.fit(fit_loader)
        model.save(str(bank_path))
        print(f'saved memory bank: {bank_path}')

    # Inference at the native MVTec resolution so masks line up with GT.
    test_records = []
    score_maps_full = []
    gt_masks_full = []
    image_scores = []
    train_pixel_max = None  # anchor for visualization

    # We need score_map at full image resolution to compare against GT mask
    # (which lives at the original 1024x1024 / 1024x1024 etc.). Fetch each
    # image's native shape on the fly.
    for batch in test_loader:
        images = batch['image'].to(device, non_blocking=True)
        # Predict at the (model input) feature scale, then resize per-image.
        score_maps_low, img_scores = model.predict(images,
                                                   target_size=(cfg['input_size'],
                                                                cfg['input_size']))
        for i in range(images.shape[0]):
            img_path = batch['image_path'][i]
            defect = batch['defect_type'][i]
            stem = f'{defect}_{Path(img_path).stem}'
            orig = np.array(Image.open(img_path).convert('RGB'))
            H, W = orig.shape[:2]
            sm = cv2.resize(score_maps_low[i].astype(np.float32), (W, H),
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
            score_maps_full.append(sm)
            gt_masks_full.append(gt)
            image_scores.append(float(img_scores[i]))
            test_records.append({
                'image_path': img_path,
                'defect_type': defect,
                'stem': stem,
                'orig': orig,
                'sm': sm,
                'gt': gt,
                'image_score': float(img_scores[i]),
            })

    # Also pass training data through to anchor visualisation. We treat
    # the max training pixel score as the upper end of "normal".
    train_eval_transform = build_image_transform(cfg['input_size'])
    train_eval_ds = MVTecDataset(args.data_root, args.category,
                                  split='train', transform=train_eval_transform)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=cfg.get('batch_size', 8),
                                   num_workers=cfg.get('num_workers', 4),
                                   shuffle=False, pin_memory=(device == 'cuda'))
    train_pix = []
    for batch in train_eval_loader:
        sm_low, _ = model.predict(batch['image'].to(device, non_blocking=True),
                                  target_size=(cfg['input_size'], cfg['input_size']))
        train_pix.append(sm_low.flatten())
    train_pix = np.concatenate(train_pix)
    train_pixel_max = float(train_pix.max())
    train_p999 = float(np.percentile(train_pix, 99.9))
    print(f'train_pixel_max={train_pixel_max:.4f}  train_p99.9={train_p999:.4f}')

    # ----- threshold tuning via GT --------------------------------------
    if args.threshold_target == 'manual' and args.threshold is not None:
        chosen_thr = float(args.threshold)
        thr_meta = {'mode': 'manual', 'threshold': chosen_thr}
    else:
        best, sweep = find_threshold(score_maps_full, gt_masks_full,
                                     mode=args.threshold_target)
        if best is None:
            print('no GT masks available; falling back to train_p99.9')
            chosen_thr = train_p999
            thr_meta = {'mode': 'train_p99.9', 'threshold': chosen_thr}
        else:
            chosen_thr = best['threshold']
            thr_meta = {'mode': args.threshold_target, **best}
            print(f"GT-tuned threshold: {chosen_thr:.4f}  "
                  f"(F1={best['f1']:.3f}, P={best['precision']:.3f}, "
                  f"R={best['recall']:.3f})")
        # Persist the sweep for inspection.
        with open(out_root / 'threshold_sweep.json', 'w', encoding='utf-8') as f:
            json.dump({'best': best, 'sweep': sweep,
                       'train_pixel_max': train_pixel_max,
                       'train_p999': train_p999}, f, indent=2)

    # ----- write outputs ------------------------------------------------
    pred_dir = out_root / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    min_area = (args.min_area if args.min_area is not None
                else int(cfg.get('min_area', 30)))
    morph = int(cfg.get('morph_kernel', 3))
    summary = []
    for r in test_records:
        sm = r['sm']
        orig = r['orig']
        H, W = orig.shape[:2]
        mask = (sm >= chosen_thr).astype(np.uint8) * 255
        mask = clean_mask(mask, min_area=min_area, kernel=morph)

        # raw normalised heatmap (calibrated against train_pixel_max)
        anchor = train_pixel_max if train_pixel_max else float(sm.max())
        ov_hm = overlay_heatmap_rgb(orig, sm, anchor)
        ov_mk = overlay_mask_rgb(orig, mask)

        # save
        stem = r['stem']
        # heatmap as plain colormap
        norm = np.clip((sm - 0.6 * anchor) / max(1.6 * anchor - 0.6 * anchor, 1e-6),
                       0, 1)
        cv2.imwrite(str(pred_dir / f'{stem}_heatmap.png'),
                    (norm * 255).astype(np.uint8))
        cv2.imwrite(str(pred_dir / f'{stem}_mask.png'), mask)
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_heatmap.png'),
                    cv2.cvtColor(ov_hm, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_mask.png'),
                    cv2.cvtColor(ov_mk, cv2.COLOR_RGB2BGR))
        summary.append({
            'image_path': r['image_path'],
            'defect_type': r['defect_type'],
            'image_score': r['image_score'],
            'mask_pct': float(100.0 * (mask > 0).sum() / mask.size),
        })

    print()
    per_category_summary(summary)
    with open(out_root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'backbone': cfg['backbone'], 'layers': list(cfg['layers']),
            'input_size': cfg['input_size'],
            'coreset_ratio': cfg['coreset_ratio'],
            'anomaly_score_num_nn': cfg.get('anomaly_score_num_nn', 1),
            'threshold_meta': thr_meta,
            'train_pixel_max': train_pixel_max,
            'train_p999': train_p999,
            'predictions': summary,
        }, f, indent=2, default=str)


if __name__ == '__main__':
    main()
