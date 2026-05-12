"""Threshold sweep on Level 3 prob.npy outputs against polygon/rect GT."""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.transforms import roi_bbox_from_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', required=True)
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--orig-dir', required=True)
    p.add_argument('--target', default='iou')
    p.add_argument('--n-steps', type=int, default=99)
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


def main():
    args = parse_args()
    gt_lookup = {p.stem.replace('_mask', ''): p
                 for p in Path(args.gt_dir).rglob('*_mask.png')}
    all_pos, all_neg = [], []
    n_def = 0
    for f in sorted(Path(args.pred_dir).glob('*_prob.npy')):
        stem = f.stem.replace('_prob', '')
        gt_path = _fuzzy_gt(gt_lookup, stem)
        if gt_path is None:
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        prob = np.load(f)
        # find orig + crop to ROI
        orig_path = None
        for ext in ('.bmp', '.png', '.jpg', '.jpeg', '.tif'):
            cand = Path(args.orig_dir) / f'{stem}{ext}'
            if cand.exists():
                orig_path = cand; break
        if orig_path is None: continue
        orig = np.asarray(Image.open(orig_path).convert('RGB'))
        x0, y0, x1, y1 = roi_bbox_from_image(orig, 8, 16)
        gt = gt[y0:y1, x0:x1]
        if gt.shape != prob.shape:
            gt = cv2.resize(gt, (prob.shape[1], prob.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        gtb = gt > 0
        if gtb.any():
            n_def += 1
            all_pos.append(prob[gtb])
            all_neg.append(prob[~gtb])
        else:
            all_neg.append(prob.flatten())
    pos = np.concatenate(all_pos)
    neg = np.concatenate(all_neg)
    if neg.size > 50_000_000:
        neg = np.random.default_rng(0).choice(neg, 50_000_000, replace=False)
    cands = np.linspace(0.01, 0.99, args.n_steps)
    best = None
    for thr in cands:
        tp = int((pos >= thr).sum()); fp = int((neg >= thr).sum())
        fn = int((pos < thr).sum())
        p_ = tp / (tp + fp + 1e-9); r_ = tp / (tp + fn + 1e-9)
        f1 = 2 * p_ * r_ / (p_ + r_ + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        m = iou if args.target == 'iou' else f1
        if best is None or m > best['_m']:
            best = dict(thr=float(thr), p=float(p_), r=float(r_),
                        f1=float(f1), iou=float(iou), _m=float(m))
    print(f"pred_dir={args.pred_dir}")
    print(f"gt_dir={args.gt_dir}")
    print(f"n_defective={n_def}")
    print(f"best: thr={best['thr']:.3f} F1={best['f1']:.4f} "
          f"P={best['p']:.4f} R={best['r']:.4f} IoU={best['iou']:.4f}")


if __name__ == '__main__':
    main()
