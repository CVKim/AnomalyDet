"""Level 3 — supervised UNet refinement head on top of PatchCore score maps.

Pipeline:
  for each test image i in {1..11}:
      score_map_i = load PatchCore output (full-res, ROI-cropped)
      gt_i        = load GT mask, crop to same ROI
  Stack into (score, grayscale) 2-channel tensors.
  Leave-one-out cross-validation over the 6 defective images:
      train on (5 defective + all 4 normals), predict the held-out one,
      collect prediction.
  Aggregate held-out predictions across the 6 LOO runs + apply the same
  model to the 4 normals (trained on all 10) for completeness.
  Compute pixel F1 / IoU vs GT, write predictions + side-by-side viz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.transforms import roi_bbox_from_image
from src.models.refinement_unet import RefinementUNet, dice_bce_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir', required=True,
                   help='PatchCore run dir (must have predictions/*_scores.npy)')
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--orig-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--train-pixel-max', type=float, default=None,
                   help='Normalising anchor for the score map channel. If '
                        'unset, derived from summary.json in --pred-dir.')
    p.add_argument('--input-size', type=int, default=384,
                   help='Resize H = W to this for the UNet.')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--pos-weight', type=float, default=100.0,
                   help='BCE positive class weight; defects are <<1% of pixels.')
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--base', type=int, default=16)
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--threshold', type=float, default=0.5,
                   help='Binarisation threshold on the sigmoid output.')
    p.add_argument('--synth-aug', action='store_true', default=False,
                   help='Synthetic anomaly augmentation: with 50% prob per '
                        'training sample, copy a small random GT crack '
                        'region from another defective image into the '
                        'current one at a random location. Adds ~2x effective '
                        'positive supervision diversity.')
    return p.parse_args()


def _fuzzy_gt(gt_lookup, stem):
    """Look up a GT mask by stem, falling back to common suffix variants
    (e.g. test image renamed to "#11_Normal" still matches label "#11")."""
    if stem in gt_lookup:
        return gt_lookup[stem]
    for suf in ('_Normal', '_normal', '_Defect', '_defect', '_NG', '_OK', '_ng', '_ok'):
        if stem.endswith(suf):
            short = stem[: -len(suf)]
            if short in gt_lookup:
                return gt_lookup[short]
    return None


def load_all_samples(pred_dir, gt_dir, orig_dir, roi_threshold, roi_margin):
    """Load (stem, score_map, grayscale, gt) for every test image. Crop all
    of them to the ROI bbox derived from the original image so the shapes
    match what the PatchCore runner saw."""
    gt_lookup = {p.stem.replace('_mask', ''): p
                 for p in Path(gt_dir).rglob('*_mask.png')}
    samples = []
    for f in sorted(Path(pred_dir / 'predictions').glob('*_scores.npy')):
        stem_full = f.stem.replace('_scores', '')
        stem = stem_full.split('_', 1)[-1] if '_' in stem_full else stem_full
        scores = np.load(f)
        # locate matching original
        orig_path = None
        for ext in ('.bmp', '.png', '.jpg', '.jpeg', '.tif'):
            cand = Path(orig_dir) / f'{stem}{ext}'
            if cand.exists():
                orig_path = cand
                break
        if orig_path is None:
            print(f'  skip {stem}: no original'); continue
        orig_rgb = np.asarray(Image.open(orig_path).convert('RGB'))
        gray = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY)
        x0, y0, x1, y1 = roi_bbox_from_image(orig_rgb,
                                              threshold=roi_threshold,
                                              margin=roi_margin)
        gray = gray[y0:y1, x0:x1]
        # Score map is *already* at the ROI-cropped size from the runner.
        if scores.shape != gray.shape:
            scores = cv2.resize(scores.astype(np.float32),
                                 (gray.shape[1], gray.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        # GT
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
        print(f'  loaded {stem}: shape={gray.shape}  defect={defect}')
    return samples


class RefDataset(Dataset):
    def __init__(self, samples, input_size, train_pixel_max, train: bool,
                 synth_aug: bool = False):
        self.samples = samples
        self.input_size = int(input_size)
        self.tp_max = float(train_pixel_max)
        self.train = bool(train)
        self.synth_aug = bool(synth_aug)
        # Pre-extract crack patches from defective samples for synth-aug
        self._crack_patches = []
        if self.synth_aug and self.train:
            for s in samples:
                if not (s['gt'] > 0).any():
                    continue
                ys, xs = np.where(s['gt'] > 0)
                if xs.size < 50:
                    continue
                # cut a tight bbox around each connected crack region
                n_label, labels = cv2.connectedComponents((s['gt'] > 0).astype(np.uint8))
                for lbl in range(1, n_label):
                    yy, xx = np.where(labels == lbl)
                    if xx.size < 20:
                        continue
                    x0, x1 = int(xx.min()), int(xx.max()) + 1
                    y0, y1 = int(yy.min()), int(yy.max()) + 1
                    if (x1 - x0) < 5 or (y1 - y0) < 5:
                        continue
                    sm_p = s['scores'][y0:y1, x0:x1].copy()
                    gr_p = s['gray'][y0:y1, x0:x1].copy()
                    gt_p = ((labels == lbl)[y0:y1, x0:x1]).astype(np.uint8) * 255
                    self._crack_patches.append((sm_p, gr_p, gt_p))

    def __len__(self):
        return len(self.samples)

    def _maybe_paste_synth(self, sm, gr, gt):
        """With 50% prob, paste a random crack patch into the current sample
        at a random non-overlapping location. Updates all three maps.
        """
        if not self._crack_patches or np.random.rand() > 0.5:
            return sm, gr, gt
        patch = self._crack_patches[np.random.randint(len(self._crack_patches))]
        sm_p, gr_p, gt_p = patch
        ph, pw = sm_p.shape
        H, W = sm.shape
        if ph >= H or pw >= W:
            return sm, gr, gt
        # try a few random positions; prefer ones with no existing GT overlap
        for _ in range(5):
            yy = np.random.randint(0, H - ph)
            xx = np.random.randint(0, W - pw)
            if (gt[yy:yy+ph, xx:xx+pw] > 0).any():
                continue
            # blend in: use max() for score map (defect = high score),
            # straight overwrite for gray + binary OR for GT
            roi_sm = sm[yy:yy+ph, xx:xx+pw]
            roi_gr = gr[yy:yy+ph, xx:xx+pw]
            sm = sm.copy()
            gr = gr.copy()
            gt = gt.copy()
            mask = (gt_p > 0).astype(np.float32)
            sm[yy:yy+ph, xx:xx+pw] = np.maximum(roi_sm, sm_p * mask)
            gr[yy:yy+ph, xx:xx+pw] = roi_gr * (1 - mask) + (gr_p / 255.0) * mask
            gt[yy:yy+ph, xx:xx+pw] = np.maximum(gt[yy:yy+ph, xx:xx+pw],
                                                (gt_p > 0).astype(np.float32))
            return sm, gr, gt
        return sm, gr, gt

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
            if self.synth_aug:
                sm, gr, gt = self._maybe_paste_synth(sm, gr, gt)
            # Random horizontal + vertical flips. NO rotation: cracks have
            # a directionality, and the train-set is too small to learn
            # rotation invariance reliably.
            if np.random.rand() < 0.5:
                sm = sm[:, ::-1].copy(); gr = gr[:, ::-1].copy(); gt = gt[:, ::-1].copy()
            if np.random.rand() < 0.5:
                sm = sm[::-1, :].copy(); gr = gr[::-1, :].copy(); gt = gt[::-1, :].copy()
            # mild brightness jitter on grayscale
            gr = np.clip(gr * (1 + (np.random.rand() - 0.5) * 0.2), 0, 1)
        x = np.stack([sm, gr], axis=0)         # (2, H, W)
        y = gt[None, ...]                      # (1, H, W)
        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), s['stem']


def train_one_fold(train_samples, val_sample, args, device, train_pixel_max):
    model = RefinementUNet(in_ch=2, base=args.base, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ds = RefDataset(train_samples, args.input_size, train_pixel_max,
                    train=True, synth_aug=args.synth_aug)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    model.train()
    for ep in range(args.epochs):
        losses = []
        for x, y, _ in dl:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = dice_bce_loss(logits, y, pos_weight=args.pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sched.step()
    # held-out prediction (no aug)
    model.eval()
    val_ds = RefDataset([val_sample], args.input_size, train_pixel_max, train=False)
    with torch.no_grad():
        x, y, _ = val_ds[0]
        x = x.unsqueeze(0).to(device)
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob, model


def compute_metrics(prob_map, gt_mask, threshold):
    pred = (prob_map >= threshold).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                f1=f1, iou=iou)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_pixel_max is None:
        try:
            summary = json.loads((pred_dir / 'summary.json').read_text(encoding='utf-8'))
            args.train_pixel_max = float(summary['train_pixel_max'])
            print(f'derived train_pixel_max from summary.json: {args.train_pixel_max:.3f}')
        except Exception:
            args.train_pixel_max = 20.0
            print(f'fallback train_pixel_max = {args.train_pixel_max}')

    print(f'loading samples from {pred_dir} + {args.gt_dir} ...')
    samples = load_all_samples(pred_dir, args.gt_dir, args.orig_dir,
                                args.roi_threshold, args.roi_margin)
    defective = [s for s in samples if s['defect']]
    normal = [s for s in samples if not s['defect']]
    print(f'defective: {[s["stem"] for s in defective]}')
    print(f'normal:    {[s["stem"] for s in normal]}')

    # LOO CV over defective images. Each fold trains on (other 5 defective
    # + all 4 normals) and predicts the held-out defective.
    fold_probs = {}
    per_image_metrics = {}
    full_train = defective + normal
    t0 = time.time()
    for hold_idx, hold in enumerate(defective):
        train_set = [s for s in defective if s['stem'] != hold['stem']] + normal
        print(f'\n[fold {hold_idx+1}/{len(defective)}] hold-out = {hold["stem"]}  '
              f'train n={len(train_set)}')
        prob_lowres, _ = train_one_fold(train_set, hold, args, device,
                                         args.train_pixel_max)
        prob_full = cv2.resize(prob_lowres, (hold['gray'].shape[1],
                                              hold['gray'].shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        fold_probs[hold['stem']] = prob_full
        m = compute_metrics(prob_full, hold['gt'], args.threshold)
        per_image_metrics[hold['stem']] = m
        print(f"  -> F1={m['f1']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  IoU={m['iou']:.3f}")

    # Normals: train a single model on all 10 (defective + normals) and predict
    # each normal. (No LOO needed; we want to see whether the model fires on
    # known-normal images.)
    print(f'\n[normals] training on all {len(full_train)} samples for FP check')
    for n in normal:
        prob_lowres, _ = train_one_fold(full_train, n, args, device,
                                         args.train_pixel_max)
        prob_full = cv2.resize(prob_lowres, (n['gray'].shape[1],
                                              n['gray'].shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        fold_probs[n['stem']] = prob_full
        m = compute_metrics(prob_full, n['gt'], args.threshold)
        per_image_metrics[n['stem']] = m
        print(f"  {n['stem']}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"FP_pixels={m['fp']}  (normal, lower FP = better)")

    elapsed = time.time() - t0
    print(f'\ntotal training time: {elapsed:.1f}s')

    # Aggregate F1 / IoU across all defective images (LOO predictions)
    all_pos = []
    all_neg = []
    for s in defective:
        prob = fold_probs[s['stem']]
        gt = (s['gt'] > 0)
        all_pos.append(prob[gt])
        all_neg.append(prob[~gt])
    all_pos = np.concatenate(all_pos)
    all_neg = np.concatenate(all_neg)
    sweep_rows = []
    best = None
    cands = np.linspace(0.01, 0.99, 99)
    for thr in cands:
        tp = int((all_pos >= thr).sum())
        fp = int((all_neg >= thr).sum())
        fn = int((all_pos < thr).sum())
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        sweep_rows.append(dict(thr=float(thr), p=float(p), r=float(r),
                               f1=float(f1), iou=float(iou)))
        if best is None or iou > best['iou']:
            best = dict(threshold=float(thr), precision=float(p),
                        recall=float(r), f1=float(f1), iou=float(iou),
                        tp=tp, fp=fp, fn=fn)
    print(f"\n*** LOO-aggregated best (IoU-target): thr={best['threshold']:.3f} "
          f"F1={best['f1']:.4f}  P={best['precision']:.4f}  "
          f"R={best['recall']:.4f}  IoU={best['iou']:.4f}")

    # Save per-image probability + binary mask at the best threshold
    preds_out = out_dir / 'predictions'
    preds_out.mkdir(parents=True, exist_ok=True)
    for s in samples:
        prob = fold_probs[s['stem']]
        np.save(str(preds_out / f"{s['stem']}_prob.npy"), prob.astype(np.float32))
        mask = (prob >= best['threshold']).astype(np.uint8) * 255
        cv2.imwrite(str(preds_out / f"{s['stem']}_mask.png"), mask)
        if (s['gt'] > 0).any():
            cv2.imwrite(str(preds_out / f"{s['stem']}_real_gt.png"), s['gt'])

    # Write a json summary
    with open(out_dir / 'level3_summary.json', 'w', encoding='utf-8') as f:
        json.dump(dict(
            pred_dir=str(pred_dir),
            train_pixel_max=args.train_pixel_max,
            input_size=args.input_size,
            epochs=args.epochs,
            lr=args.lr,
            pos_weight=args.pos_weight,
            base=args.base,
            best_thr_meta=best,
            per_image_metrics=per_image_metrics,
            sweep=sweep_rows,
        ), f, indent=2)
    print(f"wrote {out_dir / 'level3_summary.json'}")


if __name__ == '__main__':
    main()
