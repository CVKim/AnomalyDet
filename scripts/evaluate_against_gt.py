"""Apples-to-apples evaluation against MVTec ground-truth masks.

Walks a predictions directory expected to contain `<defect>_<stem>_scores.npy`
(or, as fallback, `<defect>_<stem>_heatmap.png` -- but that's per-image
normalised so the metric is approximate). Sweeps a global pixel threshold
and reports the F1-optimal one.

Usage:
    python scripts/evaluate_against_gt.py `
        --pred-dir outputs/anomalib_dinov2_hazelnut `
        --data-root "E:\\dataset\\mvtec_anomaly_detection_" `
        --category hazelnut
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', required=True,
                   help='Folder with <stem>_scores.npy (preferred) or '
                        '_heatmap.png files.')
    p.add_argument('--data-root', required=True)
    p.add_argument('--category', default='hazelnut')
    p.add_argument('--label', default=None,
                   help='Label used in the summary print. Default: derived '
                        'from pred-dir name.')
    p.add_argument('--write-masks', action='store_true', default=True,
                   help='Write per-image PRED-thresholded masks at the '
                        'GT-tuned threshold, plus a copy of the real GT '
                        'mask for direct comparison.')
    p.add_argument('--threshold-target', choices=['f1', 'recall95'],
                   default='f1')
    return p.parse_args()


def load_predictions(pred_dir: Path):
    """Yield dict(stem, defect, scores, score_source) per image."""
    pred_dir = Path(pred_dir)
    npy_files = sorted(pred_dir.glob('*_scores.npy'))
    if npy_files:
        for f in npy_files:
            stem_full = f.stem.replace('_scores', '')  # e.g. crack_000
            yield dict(stem_full=stem_full, scores=np.load(f), source='npy')
        return
    # Fallback: read heatmap PNGs. These are per-image normalised so
    # global-threshold sweeps will be sub-optimal but still informative.
    for f in sorted(pred_dir.glob('*_heatmap.png')):
        stem_full = f.stem.replace('_heatmap', '')
        arr = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        yield dict(stem_full=stem_full, scores=arr.astype(np.float32), source='png')


def load_gt(data_root: str, category: str, defect: str, stem: str, target_hw):
    """Return GT mask resized to target_hw, or None if not defective."""
    if defect == 'good':
        return np.zeros(target_hw, dtype=np.uint8)
    gt_path = Path(data_root) / category / 'ground_truth' / defect / f'{stem}_mask.png'
    if not gt_path.exists():
        return None
    m = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]),
                       interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8) * 255


def sweep_thresholds(records, mode='f1', n_steps=200, sample_per_image=4_000_000):
    rng = np.random.default_rng(0)
    pos_vals = []
    neg_vals = []
    for r in records:
        sm = r['scores']
        gt = r['gt']
        if gt is None:
            continue
        gtb = gt > 0
        pos_pix = sm[gtb]
        neg_pix = sm[~gtb]
        if len(pos_pix):
            pos_vals.append(pos_pix)
        if len(neg_pix) > sample_per_image:
            idx = rng.choice(len(neg_pix), sample_per_image, replace=False)
            neg_pix = neg_pix[idx]
        if len(neg_pix):
            neg_vals.append(neg_pix)
    if not pos_vals:
        return None, None
    pos = np.concatenate(pos_vals)
    neg = np.concatenate(neg_vals)

    all_vals = np.concatenate([pos, neg])
    lo = float(np.percentile(all_vals, 0.1))
    hi = float(np.percentile(all_vals, 99.99))
    cands = np.linspace(lo, hi, n_steps)

    best = None
    sweep = []
    for thr in cands:
        tp = int((pos >= thr).sum())
        fp = int((neg >= thr).sum())
        fn = int((pos < thr).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        sweep.append(dict(thr=float(thr), p=float(precision), r=float(recall), f1=float(f1)))
        if mode == 'f1':
            metric = f1
        elif mode == 'recall95':
            metric = precision if recall >= 0.95 else -1.0
        else:
            metric = -1.0
        if best is None or metric > best['metric']:
            best = dict(threshold=float(thr), metric=float(metric),
                        precision=float(precision), recall=float(recall),
                        f1=float(f1), tp=tp, fp=fp, fn=fn)
    return best, sweep


def main():
    args = parse_args()
    label = args.label or Path(args.pred_dir).name
    pred_dir = Path(args.pred_dir)

    records = []
    for p in load_predictions(pred_dir):
        # filenames look like <defect>_<stem> -- defect can contain '_'
        # (e.g. broken_large). Split off the trailing numeric stem.
        parts = p['stem_full'].rsplit('_', 1)
        defect, stem = parts[0], parts[1] if len(parts) == 2 else ('', p['stem_full'])
        sm = p['scores']
        gt = load_gt(args.data_root, args.category, defect, stem, sm.shape)
        records.append(dict(defect=defect, stem=stem, scores=sm, gt=gt,
                            source=p['source']))

    n_total = len(records)
    n_defect = sum(1 for r in records if r['defect'] != 'good' and r['gt'] is not None)
    print(f'{label}: n_total={n_total}, n_defect_with_gt={n_defect}, '
          f'score_source={records[0]["source"]}')

    best, sweep = sweep_thresholds(records, mode=args.threshold_target)
    if best is None:
        print('no GT to evaluate against; aborting.')
        return
    print(f'  F1-best: thr={best["threshold"]:.4f}  '
          f'F1={best["f1"]:.4f}  P={best["precision"]:.4f}  R={best["recall"]:.4f}')

    # Optional: write the F1-optimal masks back into pred-dir.
    # Naming is explicit so nothing here gets mistaken for the GT itself:
    #   <defect>_<stem>_pred_at_gt_thr.png  -- our prediction, thresholded
    #                                          at the GT-tuned threshold
    #   <defect>_<stem>_real_gt.png         -- copy of the dataset GT mask
    #                                          (only present when GT exists)
    by_defect = {}
    for r in records:
        m = (r['scores'] >= best['threshold']).astype(np.uint8) * 255
        mask_pct = float(100.0 * (m > 0).sum() / m.size)
        by_defect.setdefault(r['defect'], []).append(mask_pct)
        if args.write_masks:
            cv2.imwrite(str(pred_dir / f"{r['defect']}_{r['stem']}_pred_at_gt_thr.png"), m)
            if r['gt'] is not None and (r['gt'] > 0).any():
                cv2.imwrite(str(pred_dir / f"{r['defect']}_{r['stem']}_real_gt.png"),
                            r['gt'])

    print(f"  {'defect':14s} {'n':>3s} {'mask% mean':>11s} {'mask% max':>10s}")
    for d in sorted(by_defect):
        v = by_defect[d]
        print(f"  {d:14s} {len(v):>3d} {np.mean(v):>10.2f}% {np.max(v):>9.2f}%")

    summary_path = pred_dir / 'gt_eval.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(dict(label=label, best=best,
                       by_defect={d: dict(n=len(v), mean=float(np.mean(v)),
                                          max=float(np.max(v)))
                                  for d, v in by_defect.items()},
                       sweep=sweep), f, indent=2)
    print(f'  wrote {summary_path}')


if __name__ == '__main__':
    main()
