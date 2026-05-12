"""Level 3 UNet refinement with on-the-fly synthetic crack augmentation.

Loads the same real samples as foosung_level3_refinement.py, but during
training, for each batch additionally creates SYNTHETIC defective samples
by pasting real GT crack patches from defective images into normal
images. This bypasses the "only 6 defective" bottleneck.

LOO eval is done on the 6 real defective images only (held-out one at
a time); synthetics + remaining real defectives + normals form the
training set.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.transforms import roi_bbox_from_image
from src.models.refinement_unet import RefinementUNet, dice_bce_loss


def _fuzzy_gt(gt_lookup, stem):
    if stem in gt_lookup:
        return gt_lookup[stem]
    for suf in ('_Normal', '_normal', '_Defect', '_defect', '_NG', '_OK', '_ng', '_ok'):
        if stem.endswith(suf):
            short = stem[: -len(suf)]
            if short in gt_lookup:
                return gt_lookup[short]
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', required=True)
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--orig-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--train-pixel-max', type=float, default=None)
    p.add_argument('--input-size', type=int, default=384)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--pos-weight', type=float, default=100.0)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--base', type=int, default=16)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--n-synth', type=int, default=24,
                   help='Synthetic defective samples PER FOLD per epoch '
                        '(regenerated each epoch).')
    p.add_argument('--paste-per-image', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_samples(pred_dir, gt_dir, orig_dir, roi_threshold, roi_margin):
    gt_lookup = {p.stem.replace('_mask', ''): p
                 for p in Path(gt_dir).rglob('*_mask.png')}
    samples = []
    for f in sorted(Path(pred_dir / 'predictions').glob('*_scores.npy')):
        stem_full = f.stem.replace('_scores', '')
        stem = stem_full.split('_', 1)[-1] if '_' in stem_full else stem_full
        scores = np.load(f)
        orig_path = None
        for ext in ('.bmp', '.png', '.jpg', '.jpeg', '.tif'):
            cand = Path(orig_dir) / f'{stem}{ext}'
            if cand.exists():
                orig_path = cand; break
        if orig_path is None:
            continue
        orig = np.asarray(Image.open(orig_path).convert('RGB'))
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        x0, y0, x1, y1 = roi_bbox_from_image(orig, roi_threshold, roi_margin)
        gray = gray[y0:y1, x0:x1]
        if scores.shape != gray.shape:
            scores = cv2.resize(scores.astype(np.float32),
                                 (gray.shape[1], gray.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        gt = np.zeros_like(gray)
        gt_path = _fuzzy_gt(gt_lookup, stem)
        if gt_path is not None:
            gt_full = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt = gt_full[y0:y1, x0:x1]
            if gt.shape != gray.shape:
                gt = cv2.resize(gt, (gray.shape[1], gray.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        defect = bool((gt > 0).any())
        samples.append(dict(stem=stem, scores=scores, gray=gray, gt=gt,
                            defect=defect))
    return samples


def extract_crack_patches(defective_samples):
    """For each defective sample, return one tuple per connected crack
    region: (score_patch, gray_patch, mask_patch, src_stem)."""
    patches = []
    for s in defective_samples:
        n_lab, labels = cv2.connectedComponents((s['gt'] > 0).astype(np.uint8))
        for lbl in range(1, n_lab):
            yy, xx = np.where(labels == lbl)
            if xx.size < 20:
                continue
            x0, x1 = int(xx.min()), int(xx.max()) + 1
            y0, y1 = int(yy.min()), int(yy.max()) + 1
            if (x1 - x0) < 5 or (y1 - y0) < 5:
                continue
            patches.append((
                s['scores'][y0:y1, x0:x1].copy(),
                s['gray'][y0:y1, x0:x1].copy(),
                ((labels == lbl)[y0:y1, x0:x1]).astype(np.uint8) * 255,
                s['stem'],
            ))
    return patches


def make_synth(rng, normal_samples, patches, n_synth, paste_per_image):
    """Generate n_synth synthetic defective samples by pasting crack patches
    from `patches` into random normal images. Returns list of dicts in the
    same schema as real samples."""
    synth = []
    for i in range(n_synth):
        base = normal_samples[rng.integers(len(normal_samples))]
        sm = base['scores'].astype(np.float32).copy()
        gr = base['gray'].copy()
        gt = np.zeros_like(base['gt'])
        fg = gr > 20
        ys_fg, xs_fg = np.where(fg)
        if xs_fg.size == 0:
            continue
        H, W = gr.shape
        n_paste = max(1, int(rng.integers(1, paste_per_image + 1)))
        for _ in range(n_paste):
            patch_sm, patch_gr, patch_gt, _ = patches[rng.integers(len(patches))]
            ph, pw = patch_gt.shape
            if ph >= H or pw >= W:
                continue
            pick = rng.integers(xs_fg.size)
            cx = int(xs_fg[pick]); cy = int(ys_fg[pick])
            cx = int(np.clip(cx, pw // 2, W - pw // 2 - 1))
            cy = int(np.clip(cy, ph // 2, H - ph // 2 - 1))
            x0_, x1_ = cx - pw // 2, cx - pw // 2 + pw
            y0_, y1_ = cy - ph // 2, cy - ph // 2 + ph
            m = (patch_gt > 0).astype(np.float32)
            sm[y0_:y1_, x0_:x1_] = np.maximum(sm[y0_:y1_, x0_:x1_],
                                               patch_sm * m)
            gr_roi = gr[y0_:y1_, x0_:x1_].astype(np.float32)
            blended = gr_roi * (1 - m) + patch_gr.astype(np.float32) * m
            gr[y0_:y1_, x0_:x1_] = blended.astype(np.uint8)
            gt[y0_:y1_, x0_:x1_] = np.maximum(gt[y0_:y1_, x0_:x1_],
                                               (patch_gt > 0).astype(np.uint8) * 255)
        synth.append(dict(stem=f'synth{i:03d}', scores=sm, gray=gr, gt=gt,
                          defect=True))
    return synth


class RefDataset(Dataset):
    def __init__(self, samples, input_size, tp_max, train: bool):
        self.samples = samples
        self.input_size = int(input_size)
        self.tp_max = float(tp_max)
        self.train = bool(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        sm = cv2.resize(s['scores'], (self.input_size, self.input_size),
                        interpolation=cv2.INTER_LINEAR)
        gr = cv2.resize(s['gray'], (self.input_size, self.input_size),
                        interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(s['gt'], (self.input_size, self.input_size),
                        interpolation=cv2.INTER_NEAREST)
        sm = (sm / max(self.tp_max, 1e-6)).astype(np.float32)
        gr = (gr.astype(np.float32) / 255.0)
        gt = (gt > 0).astype(np.float32)
        if self.train:
            if np.random.rand() < 0.5:
                sm = sm[:, ::-1].copy(); gr = gr[:, ::-1].copy(); gt = gt[:, ::-1].copy()
            if np.random.rand() < 0.5:
                sm = sm[::-1, :].copy(); gr = gr[::-1, :].copy(); gt = gt[::-1, :].copy()
            gr = np.clip(gr * (1 + (np.random.rand() - 0.5) * 0.2), 0, 1)
        x = np.stack([sm, gr], axis=0)
        y = gt[None, ...]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), s['stem']


def train_fold(real_train, normal_samples, patches, val_sample, args, device, tp_max, rng):
    """Train on (real_train + synthetic samples regenerated each epoch),
    predict held-out val_sample, return prob map."""
    model = RefinementUNet(in_ch=2, base=args.base, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    model.train()
    for ep in range(args.epochs):
        synth = make_synth(rng, normal_samples, patches, args.n_synth,
                           args.paste_per_image)
        ds = RefDataset(real_train + synth, args.input_size, tp_max, train=True)
        dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
        for x, y, _ in dl:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = dice_bce_loss(logits, y, pos_weight=args.pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    val_ds = RefDataset([val_sample], args.input_size, tp_max, train=False)
    with torch.no_grad():
        x, _, _ = val_ds[0]
        x = x.unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')
    rng = np.random.default_rng(args.seed)

    pred_dir = Path(args.pred_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.train_pixel_max is None:
        try:
            summary = json.loads((pred_dir / 'summary.json').read_text(encoding='utf-8'))
            args.train_pixel_max = float(summary['train_pixel_max'])
        except Exception:
            args.train_pixel_max = 20.0
    tp_max = args.train_pixel_max
    print(f'train_pixel_max = {tp_max:.4f}')

    samples = load_samples(pred_dir, args.gt_dir, args.orig_dir,
                           args.roi_threshold, args.roi_margin)
    defective = [s for s in samples if s['defect']]
    normal = [s for s in samples if not s['defect']]
    print(f'real: {len(defective)} defective, {len(normal)} normal')

    patches = extract_crack_patches(defective)
    print(f'extracted {len(patches)} crack patches')
    print(f'synth per epoch: {args.n_synth} (regenerated)')

    fold_probs = {}
    per_image = {}
    t0 = time.time()
    for idx, hold in enumerate(defective):
        train_set = [s for s in defective if s['stem'] != hold['stem']] + normal
        # build patches list excluding the held-out image (avoid leakage)
        held_patches = extract_crack_patches(
            [s for s in defective if s['stem'] != hold['stem']]
        )
        print(f'\n[fold {idx+1}/{len(defective)}] hold-out = {hold["stem"]}  '
              f'train={len(train_set)}+{args.n_synth}/epoch synth')
        prob = train_fold(train_set, normal, held_patches, hold, args, device,
                          tp_max, rng)
        prob_full = cv2.resize(prob, (hold['gray'].shape[1], hold['gray'].shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        fold_probs[hold['stem']] = prob_full
        # quick metric
        pred = (prob_full >= args.threshold).astype(np.uint8)
        gt_b = (hold['gt'] > 0).astype(np.uint8)
        tp = int(((pred == 1) & (gt_b == 1)).sum())
        fp = int(((pred == 1) & (gt_b == 0)).sum())
        fn = int(((pred == 0) & (gt_b == 1)).sum())
        p_ = tp / (tp + fp + 1e-9); r_ = tp / (tp + fn + 1e-9)
        f1 = 2 * p_ * r_ / (p_ + r_ + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        per_image[hold['stem']] = dict(f1=float(f1), p=float(p_), r=float(r_),
                                        iou=float(iou), tp=tp, fp=fp, fn=fn)
        print(f"  F1={f1:.3f}  P={p_:.3f}  R={r_:.3f}  IoU={iou:.3f}")

    # FP check on normals (train on everything)
    print(f'\n[normals] FP check (train on all real)')
    all_patches = extract_crack_patches(defective)
    for n in normal:
        train_set = defective + [x for x in normal if x['stem'] != n['stem']]
        prob = train_fold(train_set, normal, all_patches, n, args, device, tp_max, rng)
        prob_full = cv2.resize(prob, (n['gray'].shape[1], n['gray'].shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        fold_probs[n['stem']] = prob_full
        fp = int((prob_full >= args.threshold).sum())
        per_image[n['stem']] = dict(fp=fp)
        print(f"  {n['stem']:14s}  FP_pixels={fp}")

    elapsed = time.time() - t0
    print(f'\ntotal training time: {elapsed:.1f}s')

    # Save probability maps + threshold-binarised masks at the best sweep threshold
    all_pos, all_neg = [], []
    for s in defective:
        prob = fold_probs[s['stem']]
        gt = (s['gt'] > 0)
        all_pos.append(prob[gt]); all_neg.append(prob[~gt])
    for n in normal:
        all_neg.append(fold_probs[n['stem']].flatten())
    pos = np.concatenate(all_pos); neg = np.concatenate(all_neg)
    if neg.size > 50_000_000:
        neg = rng.choice(neg, 50_000_000, replace=False)
    cands = np.linspace(0.01, 0.99, 99)
    best = None
    for thr in cands:
        tp = int((pos >= thr).sum()); fp = int((neg >= thr).sum())
        fn = int((pos < thr).sum())
        p_ = tp / (tp + fp + 1e-9); r_ = tp / (tp + fn + 1e-9)
        f1 = 2 * p_ * r_ / (p_ + r_ + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        if best is None or iou > best['iou']:
            best = dict(threshold=float(thr), f1=float(f1), precision=float(p_),
                        recall=float(r_), iou=float(iou), tp=tp, fp=fp, fn=fn)
    print(f"\n*** LOO-aggregated best (IoU-target): "
          f"thr={best['threshold']:.3f} F1={best['f1']:.4f} "
          f"P={best['precision']:.4f} R={best['recall']:.4f} IoU={best['iou']:.4f}")

    preds_out = out / 'predictions'
    preds_out.mkdir(parents=True, exist_ok=True)
    for stem, prob in fold_probs.items():
        np.save(str(preds_out / f"{stem}_prob.npy"), prob.astype(np.float32))
        mask = (prob >= best['threshold']).astype(np.uint8) * 255
        cv2.imwrite(str(preds_out / f"{stem}_mask.png"), mask)
    with open(out / 'level3_summary.json', 'w', encoding='utf-8') as f:
        json.dump(dict(
            pred_dir=str(pred_dir),
            train_pixel_max=tp_max,
            input_size=args.input_size,
            epochs=args.epochs, lr=args.lr, pos_weight=args.pos_weight,
            n_synth=args.n_synth, paste_per_image=args.paste_per_image,
            best_thr_meta=best,
            per_image_metrics=per_image,
        ), f, indent=2)
    print(f'wrote {out / "level3_summary.json"}')


if __name__ == '__main__':
    main()
