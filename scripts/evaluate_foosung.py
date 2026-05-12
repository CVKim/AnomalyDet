"""Sweep pixel thresholds against the Foosung CRACK bbox GT.

Reads <pred-dir>/predictions/test_<stem>_scores.npy + matching GT mask
from <gt-dir>/<sub>/<stem>_mask.png, finds the bbox of GT pixels, builds
the same F1 / IoU / target_recall sweep as scripts/evaluate_against_gt.py
but for the flat custom-dir naming convention.

Outputs:
    <pred-dir>/gt_eval.json           summary
    <pred-dir>/predictions/<stem>_pred_at_gt_thr.png
    <pred-dir>/predictions/<stem>_real_gt.png
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.transforms import roi_bbox_from_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', required=True,
                   help='Run output dir, must contain predictions/test_*_scores.npy')
    p.add_argument('--gt-dir', required=True,
                   help='Folder of <stem>_mask.png (any nesting). Stems must '
                        'match the test image stems (e.g. "#5_mask.png").')
    p.add_argument('--orig-dir', default=None,
                   help='Folder with the original test images. Required when '
                        'predictions were generated with --roi-crop, so the '
                        'evaluator can re-detect the same crop bbox and align '
                        'the GT mask to the score map.')
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--target', choices=['f1', 'iou', 'target_recall'],
                   default='iou')
    p.add_argument('--min-recall', type=float, default=0.50)
    p.add_argument('--n-steps', type=int, default=200)
    return p.parse_args()


def _fuzzy_gt(gt_lookup, stem):
    if stem in gt_lookup:
        return gt_lookup[stem]
    for suf in ('_Normal', '_normal', '_Defect', '_defect', '_NG', '_OK', '_ng', '_ok'):
        if stem.endswith(suf):
            short = stem[: -len(suf)]
            if short in gt_lookup:
                return gt_lookup[short]
    return None


def load_pairs(pred_dir: Path, gt_dir: Path, orig_dir: Path = None,
               roi_threshold: int = 8, roi_margin: int = 16):
    out = []
    gt_lookup = {p.stem.replace('_mask', ''): p
                 for p in Path(gt_dir).rglob('*_mask.png')}
    for f in sorted(Path(pred_dir / 'predictions').glob('*_scores.npy')):
        stem_full = f.stem.replace('_scores', '')        # "test_#5"
        stem = stem_full.split('_', 1)[-1] if '_' in stem_full else stem_full
        gt_path = _fuzzy_gt(gt_lookup, stem)
        scores = np.load(f)
        gt = None
        if gt_path is not None:
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt.shape != scores.shape:
                # Score map is cropped (ROI mode); apply the same crop to GT.
                if orig_dir is not None:
                    orig_path = None
                    for ext in ('.bmp', '.png', '.jpg', '.jpeg', '.tif'):
                        cand = Path(orig_dir) / f'{stem}{ext}'
                        if cand.exists():
                            orig_path = cand
                            break
                    if orig_path is not None:
                        # PIL handles Unicode paths; cv2.imread doesn't on Windows.
                        orig_rgb = np.asarray(Image.open(orig_path).convert('RGB'))
                        x0, y0, x1, y1 = roi_bbox_from_image(
                            orig_rgb, threshold=roi_threshold, margin=roi_margin)
                        gt = gt[y0:y1, x0:x1]
                if gt.shape != scores.shape:
                    gt = cv2.resize(gt, (scores.shape[1], scores.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        out.append(dict(stem=stem, scores=scores, gt=gt, gt_path=gt_path))
    return out


def sweep(records, mode='iou', n_steps=200, min_recall=0.5,
          neg_cap=50_000_000):
    rng = np.random.default_rng(0)
    pos_vals, neg_vals = [], []
    for r in records:
        if r['gt'] is None:
            continue
        gtb = r['gt'] > 0
        pos = r['scores'][gtb]
        neg = r['scores'][~gtb]
        if pos.size:
            pos_vals.append(pos)
        if neg.size:
            neg_vals.append(neg)
    if not pos_vals:
        return None, None
    pos = np.concatenate(pos_vals)
    neg = np.concatenate(neg_vals)
    if neg.size > neg_cap:
        idx = rng.choice(neg.size, neg_cap, replace=False)
        neg = neg[idx]
    all_vals = np.concatenate([pos, neg])
    lo = float(np.percentile(all_vals, 0.1))
    hi = float(np.percentile(all_vals, 99.99))
    cands = np.linspace(lo, hi, n_steps)
    best, sweep_rows = None, []
    for thr in cands:
        tp = int((pos >= thr).sum())
        fp = int((neg >= thr).sum())
        fn = int((pos < thr).sum())
        p_ = tp / (tp + fp + 1e-9)
        r_ = tp / (tp + fn + 1e-9)
        f1 = 2 * p_ * r_ / (p_ + r_ + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        row = dict(thr=float(thr), p=float(p_), r=float(r_),
                   f1=float(f1), iou=float(iou))
        sweep_rows.append(row)
        if mode == 'f1':
            metric = f1
        elif mode == 'iou':
            metric = iou
        elif mode == 'target_recall':
            metric = p_ if r_ >= min_recall else -1.0
        else:
            metric = -1.0
        if best is None or metric > best['metric']:
            best = dict(threshold=float(thr), metric=float(metric),
                        precision=float(p_), recall=float(r_),
                        f1=float(f1), iou=float(iou),
                        tp=tp, fp=fp, fn=fn)
    return best, sweep_rows


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    records = load_pairs(pred_dir, Path(args.gt_dir),
                          orig_dir=Path(args.orig_dir) if args.orig_dir else None,
                          roi_threshold=args.roi_threshold,
                          roi_margin=args.roi_margin)
    n_defect = sum(1 for r in records if r['gt'] is not None and (r['gt'] > 0).any())
    n_good = sum(1 for r in records if r['gt'] is not None and not (r['gt'] > 0).any())
    n_missing = sum(1 for r in records if r['gt'] is None)
    print(f'{pred_dir.name}: n_defect={n_defect}, n_good={n_good}, n_missing_gt={n_missing}')

    best, sweep_rows = sweep(records, mode=args.target,
                              min_recall=args.min_recall, n_steps=args.n_steps)
    if best is None:
        print('no GT pixels found; abort')
        return
    print(f"best ({args.target}): thr={best['threshold']:.3f}  "
          f"F1={best['f1']:.4f}  P={best['precision']:.4f}  "
          f"R={best['recall']:.4f}  IoU={best['iou']:.4f}")

    # write pred-at-gt-thr + real_gt copy
    for r in records:
        if r['gt'] is None:
            continue
        m = (r['scores'] >= best['threshold']).astype(np.uint8) * 255
        out_pred = pred_dir / 'predictions' / f"{r['stem']}_pred_at_gt_thr.png"
        cv2.imwrite(str(out_pred), m)
        if (r['gt'] > 0).any():
            cv2.imwrite(str(pred_dir / 'predictions' / f"{r['stem']}_real_gt.png"),
                        r['gt'])

    with open(pred_dir / 'gt_eval.json', 'w', encoding='utf-8') as f:
        json.dump(dict(target=args.target, best=best, sweep=sweep_rows,
                       n_defect=n_defect, n_good=n_good,
                       n_missing_gt=n_missing), f, indent=2)
    print(f'wrote {pred_dir / "gt_eval.json"}')


if __name__ == '__main__':
    main()
