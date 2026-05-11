"""End-to-end driver for the paper-faithful PatchCore implementation.

Single source of truth for mask generation:
    score_map -> threshold (GT-tuned or manual) -> optional clean_mask
                                                -> FINAL_MASK
    -> used for:
       <stem>_mask.png             binary mask PNG
       <stem>_overlay_mask.png     mask blended over original
       <stem>_overlay_heatmap.png  anomaly heatmap blended over original
       <stem>_panel5.png           5-panel comparison: image | mask | GT | FG | BG

Every output directory also gets:
    config_used.yaml      exact YAML for this run
    run_command.txt       CLI + timestamp + memory bank source
    summary.json          backbone / layers / coreset / K / threshold / F1
    threshold_sweep.json  full threshold candidate sweep
    train_manifest.json   ordered list of training images used to build the bank
    *_scores.npy          raw float anomaly map per image (re-tune offline)
    *_real_gt.png         straight copy of the dataset GT mask for defect images

Usage:
    python scripts/run_patchcore_official.py `
        --config configs/patchcore_official_dinov2_518.yaml `
        --data-root "E:\\dataset\\mvtec_anomaly_detection_" `
        --category hazelnut `
        --output outputs/patchcore_official_dinov2_518_hazelnut `
        --threshold-target f1
"""
import argparse
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

from src.data.dataset import MVTecDataset, FolderDataset
from src.data.transforms import build_image_transform, build_train_transform
from src.models.patchcore_official import PatchCoreOfficial


# ---------- args -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', default=None,
                   help='MVTec dataset root. Required unless --train-dir and '
                        '--test-dir are given.')
    p.add_argument('--category', default='hazelnut',
                   help='MVTec category (used with --data-root).')
    p.add_argument('--train-dir', default=None,
                   help='Custom folder of training (known-good) images. '
                        'When set, --data-root / --category are ignored for '
                        'training. Pair with --test-dir.')
    p.add_argument('--test-dir', default=None,
                   help='Custom folder of test images (any defect status). '
                        'No GT masks expected; threshold defaults to '
                        'train_p999. Pair with --train-dir.')
    p.add_argument('--output', required=True)
    p.add_argument('--memory-bank', default=None,
                   help='Reuse an existing memory_bank.pt; skip fit().')
    p.add_argument('--threshold-target',
                   choices=['f1', 'iou', 'recall95',
                            'precision_recall70', 'target_recall',
                            'manual', 'train_p999', 'train_pixel_max'],
                   default='f1',
                   help='f1: argmax F1. iou: argmax IoU vs GT (mask outline '
                        'closest to GT). recall95 / precision_recall70: '
                        'legacy. target_recall: highest precision threshold '
                        'subject to pixel recall >= --min-recall. manual: '
                        '--threshold. train_p999 / train_pixel_max: GT-free, '
                        'derived from the training pixel-score distribution '
                        '(production mode).')
    p.add_argument('--threshold', type=float, default=None,
                   help='Used when --threshold-target manual.')
    p.add_argument('--min-recall', type=float, default=0.70,
                   help='Recall floor for precision_recall70 / target_recall.')
    p.add_argument('--apply-clean-mask', action='store_true', default=False,
                   help='Apply morphological open+close + min_area filter '
                        'to the final mask. Off by default so masks match '
                        'the raw thresholded heatmap.')
    p.add_argument('--num-nn', type=int, default=None,
                   help='Override anomaly_score_num_nn from the config. '
                        'K=1 = paper default; K=3 averages 3 nearest '
                        'coreset neighbours, sharpens response.')
    p.add_argument('--tta', choices=['none', 'flips'], default='none',
                   help='Test-time augmentation: flips averages 4 score '
                        'maps (identity + hflip + vflip + both) per image.')
    p.add_argument('--guided-filter', action='store_true', default=False,
                   help='Apply guided-filter post-process on the full-res '
                        'score map using the original RGB as guide. Snaps '
                        'the mask outline to image edges before threshold.')
    p.add_argument('--gf-radius', type=int, default=8,
                   help='Guided-filter window radius (pixels).')
    p.add_argument('--gf-eps', type=float, default=1e-3,
                   help='Guided-filter regularisation epsilon.')
    p.add_argument('--letterbox', action='store_true', default=False,
                   help='Preserve aspect ratio: resize so the long side '
                        '== input_size and pad the short side with black. '
                        'Use for non-square sources (e.g. 4096x2851).')
    p.add_argument('--bg-mask', action='store_true', default=False,
                   help='Zero out the score map where the original image '
                        'is darker than --bg-threshold (8-bit). Suppresses '
                        'anomaly response on letterbox padding and dark '
                        'background.')
    p.add_argument('--bg-threshold', type=int, default=8,
                   help='8-bit grayscale value below which a pixel counts '
                        'as background for --bg-mask.')
    return p.parse_args()


# ---------- helpers ---------------------------------------------------------

def clean_mask_strict(mask, kernel=3, min_area=30):
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


def find_threshold(score_maps, gt_masks, mode='f1', neg_cap=50_000_000,
                   min_recall=0.70):
    """Sweep a pixel threshold against the GT masks and return the
    threshold that maximises the requested metric. Operates on the full
    pixel pool (no aggressive sub-sampling)."""
    pos_scores = []
    neg_scores = []
    for s, gt in zip(score_maps, gt_masks):
        if gt is None:
            continue
        gtb = (gt > 0)
        if gtb.any():
            pos_scores.append(s[gtb])
        if (~gtb).any():
            neg_scores.append(s[~gtb])
    if not pos_scores:
        return None, None
    pos = np.concatenate(pos_scores)
    neg = np.concatenate(neg_scores) if neg_scores else np.array([], np.float32)
    if len(neg) > neg_cap:
        idx = np.random.default_rng(0).choice(len(neg), neg_cap, replace=False)
        neg = neg[idx]
    print(f'    sweep pool: pos={len(pos):,}  neg={len(neg):,}')
    lo = float(np.percentile(np.concatenate([pos, neg]), 0.1))
    hi = float(np.percentile(np.concatenate([pos, neg]), 99.99))
    cands = np.linspace(lo, hi, 200)

    best = None
    sweep = []
    for thr in cands:
        tp = int((pos >= thr).sum())
        fp = int((neg >= thr).sum())
        fn = int((pos < thr).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        # IoU = TP / (TP + FP + FN) -- the most direct "mask matches GT
        # outline" objective. Penalises both over- and under-segmentation.
        iou = tp / (tp + fp + fn + 1e-9)
        sweep.append(dict(thr=float(thr), p=float(prec), r=float(rec),
                          f1=float(f1), iou=float(iou)))
        if mode == 'f1':
            score = f1
        elif mode == 'iou':
            score = iou
        elif mode == 'recall95':
            score = prec if rec >= 0.95 else -1.0
        elif mode in ('precision_recall70', 'target_recall'):
            score = prec if rec >= min_recall else -1.0
        else:
            score = -1.0
        if best is None or score > best['metric']:
            best = dict(threshold=float(thr), metric=float(score),
                        precision=float(prec), recall=float(rec),
                        f1=float(f1), iou=float(iou), tp=tp, fp=fp, fn=fn)
    return best, sweep


# ---------- visualisation ---------------------------------------------------

def _put_centered_label(canvas, text, x0, x1, label_h):
    font = cv2.FONT_HERSHEY_SIMPLEX
    size, _ = cv2.getTextSize(text, font, 0.7, 2)
    x = x0 + ((x1 - x0) - size[0]) // 2
    y = (label_h + size[1]) // 2
    cv2.putText(canvas, text, (x, y), font, 0.7, (255, 255, 255), 2)


def make_6panel(image_rgb, mask, gt_mask, score_map, train_pixel_max,
                panel_size=320):
    """Build a 6-panel comparison image:
       image | heatmap | mask pred | gt | pred conf fg | pred conf bg.

    heatmap      : image with anomaly heatmap blended over the whole frame
                   -- shows the raw anomaly response BEFORE thresholding.
    pred_conf_fg : black background, anomaly-score colormap inside the
                   predicted mask only -- shows the anomaly response
                   AFTER thresholding (where the model says 'defect').
    pred_conf_bg : full blue field with the predicted mask cut out as
                   black -- the inverse view: what the model considers
                   normal.
    """
    H, W = image_rgb.shape[:2]
    H2 = panel_size
    W2 = int(W * panel_size / H)

    def _resize(a):
        return cv2.resize(a, (W2, H2), interpolation=cv2.INTER_AREA)

    # Normalise score with train-pixel-max anchor (same scale as the
    # standalone overlay_heatmap.png) so both panels match.
    anchor = float(train_pixel_max) if train_pixel_max else float(score_map.max())
    lo = anchor * 0.5
    hi = anchor * 1.6
    norm = np.clip((score_map - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    norm_u8 = (norm * 255).astype(np.uint8)
    jet_rgb = cv2.cvtColor(cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)

    # Panel 1: original image
    p1 = _resize(image_rgb)

    # Panel 2: heatmap blended over the original image
    heat_overlay = cv2.addWeighted(image_rgb, 0.5, jet_rgb, 0.5, 0)
    p2 = _resize(heat_overlay)

    # Panel 3: predicted mask overlay (cyan @ 50%)
    overlay = image_rgb.copy()
    overlay[mask > 0] = (0, 220, 255)
    p3 = _resize(cv2.addWeighted(image_rgb, 0.5, overlay, 0.5, 0))

    # Panel 4: GT mask overlay (cyan @ 50%) or just the raw image
    p4 = image_rgb.copy()
    if gt_mask is not None and (gt_mask > 0).any():
        gov = image_rgb.copy()
        gov[gt_mask > 0] = (0, 220, 255)
        p4 = cv2.addWeighted(image_rgb, 0.5, gov, 0.5, 0)
    p4 = _resize(p4)

    # Panel 5: pred conf FG -- jet inside the mask, black outside
    fg = jet_rgb.copy()
    fg[mask == 0] = (0, 0, 0)
    p5 = _resize(fg)

    # Panel 6: pred conf BG -- blue field with mask region cut to black
    inv = (1.0 - norm)
    bg_color = np.zeros_like(image_rgb)
    bg_color[..., 2] = (60 + inv * 195).astype(np.uint8)  # B (RGB idx 2)
    bg_color[..., 1] = (inv * 90).astype(np.uint8)
    bg_color[..., 0] = (inv * 80).astype(np.uint8)
    bg_color[mask > 0] = (0, 0, 0)
    p6 = _resize(bg_color)

    label_h = 32
    panels = [p1, p2, p3, p4, p5, p6]
    labels = ['image', 'heatmap', 'mask pred', 'gt',
              'pred conf fg', 'pred conf bg']
    canvas = np.zeros((H2 + label_h, W2 * 6, 3), dtype=np.uint8)
    for i, (lbl, panel) in enumerate(zip(labels, panels)):
        canvas[label_h:label_h + H2, i * W2:(i + 1) * W2] = panel
        _put_centered_label(canvas, lbl, i * W2, (i + 1) * W2, label_h)
    return canvas


# Back-compat alias (older calls).
make_5panel = make_6panel


def overlay_heatmap_rgb(image_rgb, score_map, train_pixel_max):
    anchor = float(train_pixel_max) if train_pixel_max else float(score_map.max())
    lo = anchor * 0.5
    hi = anchor * 1.6
    norm = np.clip((score_map - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    u8 = (norm * 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.5, color, 0.5, 0)


def overlay_mask_rgb(image_rgb, mask):
    ov = image_rgb.copy()
    ov[mask > 0] = (255, 0, 0)
    return cv2.addWeighted(image_rgb, 0.6, ov, 0.4, 0)


def guided_filter_gray(guide_rgb, src, radius=8, eps=1e-3):
    """He et al. 2010 guided filter, grayscale guide.

    guide_rgb : HxWx3 uint8 (original image)
    src       : HxW float32 — the score map to refine (any range)
    Returns   : HxW float32, same range as src but snapped to guide edges.
    """
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


def predict_tta(model, images, target_size, mode='none'):
    """Test-time augmentation wrapper around model.predict.

    images : (B, 3, H, W) torch tensor on device.
    mode   : 'none' (single pass) or 'flips' (identity + hflip + vflip + both).
    Returns score_maps (B, H, W) numpy, image_scores (B,) numpy.
    """
    if mode == 'none':
        return model.predict(images, target_size=target_size)
    if mode != 'flips':
        raise ValueError(f'unknown tta mode: {mode}')
    sm_acc = None
    score_acc = None
    transforms = [
        ('id',    lambda x: x,                lambda y: y),
        ('hflip', lambda x: torch.flip(x, dims=[3]), lambda y: y[..., :, ::-1].copy()),
        ('vflip', lambda x: torch.flip(x, dims=[2]), lambda y: y[..., ::-1, :].copy()),
        ('both',  lambda x: torch.flip(x, dims=[2, 3]),
                  lambda y: y[..., ::-1, ::-1].copy()),
    ]
    for _, fwd, inv in transforms:
        sm, sc = model.predict(fwd(images), target_size=target_size)
        sm = inv(sm)
        if sm_acc is None:
            sm_acc = sm
            score_acc = sc
        else:
            sm_acc = sm_acc + sm
            score_acc = score_acc + sc
    return sm_acc / len(transforms), score_acc / len(transforms)


# ---------- main ------------------------------------------------------------

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}; backbone: {cfg["backbone"]}; layers: {cfg["layers"]}')

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # Snapshot config / command.
    shutil.copy(args.config, out_root / 'config_used.yaml')
    with open(out_root / 'run_command.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write(f'run_at: {datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'category: {args.category}\n')
        f.write(f'data_root: {args.data_root}\n')
        f.write(f'memory_bank: {args.memory_bank or "(built fresh)"}\n')
        f.write(f'threshold_target: {args.threshold_target}\n')
        f.write(f'num_nn: {args.num_nn or "(from config)"}\n')
        f.write(f'tta: {args.tta}\n')
        f.write(f'guided_filter: {args.guided_filter}'
                f' (r={args.gf_radius}, eps={args.gf_eps})\n')
        f.write(f'letterbox: {args.letterbox}\n')
        f.write(f'bg_mask: {args.bg_mask} (threshold={args.bg_threshold})\n')

    # ---- Choose dataset mode -----------------------------------------------
    use_custom = bool(args.train_dir or args.test_dir)
    if use_custom:
        if not (args.train_dir and args.test_dir):
            raise SystemExit('--train-dir and --test-dir must be set together.')

    fit_transform = build_train_transform(
        cfg['input_size'],
        augment=bool(cfg.get('train_augment', False)),
        letterbox=bool(args.letterbox),
    )
    test_transform = build_image_transform(cfg['input_size'],
                                            letterbox=bool(args.letterbox))

    if use_custom:
        fit_ds = FolderDataset.from_dir(
            args.train_dir, transform=fit_transform,
            defect_type='good', label=0,
            repeat=int(cfg.get('train_repeat', 1)),
        )
        if len(fit_ds.samples) == 0:
            raise SystemExit(f'no images found in --train-dir {args.train_dir}')
        test_ds = FolderDataset.from_dir(
            args.test_dir, transform=test_transform,
            defect_type='test', label=1,
        )
        if len(test_ds.samples) == 0:
            raise SystemExit(f'no images found in --test-dir {args.test_dir}')
        train_src = f'--train-dir {args.train_dir}'
        category_label = '(custom)'
    else:
        if not args.data_root:
            raise SystemExit('--data-root required when --train-dir/--test-dir '
                             'are not set.')
        fit_ds = MVTecDataset(args.data_root, args.category,
                              split='train', transform=fit_transform,
                              repeat=int(cfg.get('train_repeat', 1)))
        test_ds = MVTecDataset(args.data_root, args.category,
                               split='test', transform=test_transform)
        train_src = f'{args.data_root}/{args.category}/train/good'
        category_label = args.category

    fit_loader = DataLoader(fit_ds, batch_size=cfg.get('batch_size', 8),
                            shuffle=False, num_workers=cfg.get('num_workers', 4),
                            pin_memory=(device == 'cuda'))
    test_loader = DataLoader(test_ds, batch_size=cfg.get('batch_size', 8),
                             shuffle=False, num_workers=cfg.get('num_workers', 4),
                             pin_memory=(device == 'cuda'))

    # Train manifest: the actual image files that produced the bank.
    train_files = sorted(str(p['image']) for p in fit_ds.samples)
    with open(out_root / 'train_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({
            'category': category_label,
            'split': train_src,
            'n_train_images': len(train_files),
            'augment': bool(cfg.get('train_augment', False)),
            'repeat': int(cfg.get('train_repeat', 1)),
            'effective_passes': len(train_files) * int(cfg.get('train_repeat', 1)),
            'train_images': train_files,
        }, f, indent=2)
    print(f'training files: {len(train_files)} (aug={cfg.get("train_augment", False)}, '
          f'repeat={cfg.get("train_repeat", 1)})')

    # Auto-flip threshold target to train_p999 in custom-dir mode when the
    # user left it at the default `f1` -- there's no GT to sweep against.
    if use_custom and args.threshold_target == 'f1':
        print('custom-dir mode: no GT available; switching '
              '--threshold-target f1 -> train_p999')
        args.threshold_target = 'train_p999'

    num_nn = int(args.num_nn) if args.num_nn else int(cfg.get('anomaly_score_num_nn', 1))
    model = PatchCoreOfficial(
        backbone=cfg['backbone'],
        layers=tuple(cfg['layers']),
        input_size=cfg['input_size'],
        coreset_ratio=cfg['coreset_ratio'],
        coreset_projection_dim=cfg.get('coreset_projection_dim', 128),
        anomaly_score_num_nn=num_nn,
        device=device,
    )
    print(f'num_nn={num_nn}  tta={args.tta}  guided_filter={args.guided_filter}')

    bank_path = out_root / 'memory_bank.pt'
    if args.memory_bank:
        print(f'loading memory bank: {args.memory_bank}')
        model.load(args.memory_bank)
    else:
        model.fit(fit_loader)
        model.save(str(bank_path))
        print(f'saved memory bank: {bank_path}')

    # ---- Inference on test split -------------------------------------------
    test_records = []
    for batch in test_loader:
        images = batch['image'].to(device, non_blocking=True)
        score_maps_low, img_scores = predict_tta(
            model, images,
            target_size=(cfg['input_size'], cfg['input_size']),
            mode=args.tta,
        )
        for i in range(images.shape[0]):
            img_path = batch['image_path'][i]
            defect = batch['defect_type'][i]
            stem = f'{defect}_{Path(img_path).stem}'
            orig = np.array(Image.open(img_path).convert('RGB'))
            H, W = orig.shape[:2]
            sm = cv2.resize(score_maps_low[i].astype(np.float32), (W, H),
                            interpolation=cv2.INTER_LINEAR)
            if args.bg_mask:
                gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
                sm = np.where(gray < args.bg_threshold, 0.0, sm).astype(np.float32)
            if args.guided_filter:
                sm = guided_filter_gray(orig, sm,
                                        radius=args.gf_radius, eps=args.gf_eps)
            mp = batch.get('mask_path', [''])[i]
            gt = None
            if mp:
                gt_arr = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if gt_arr is not None:
                    if gt_arr.shape != (H, W):
                        gt_arr = cv2.resize(gt_arr, (W, H), interpolation=cv2.INTER_NEAREST)
                    gt = gt_arr
            test_records.append(dict(
                image_path=img_path, defect_type=defect, stem=stem,
                orig=orig, sm=sm, gt=gt, image_score=float(img_scores[i]),
            ))

    # ---- Pass through training data to anchor heatmap viz ------------------
    if use_custom:
        train_eval_ds = FolderDataset.from_dir(
            args.train_dir,
            transform=build_image_transform(cfg['input_size'],
                                            letterbox=bool(args.letterbox)),
            defect_type='good', label=0,
        )
    else:
        train_eval_ds = MVTecDataset(
            args.data_root, args.category, split='train',
            transform=build_image_transform(cfg['input_size'],
                                            letterbox=bool(args.letterbox)))
    train_eval_loader = DataLoader(train_eval_ds, batch_size=cfg.get('batch_size', 8),
                                   num_workers=cfg.get('num_workers', 4),
                                   shuffle=False, pin_memory=(device == 'cuda'))
    train_pix_vals = []
    for batch in train_eval_loader:
        sm_low, _ = predict_tta(
            model,
            batch['image'].to(device, non_blocking=True),
            target_size=(cfg['input_size'], cfg['input_size']),
            mode=args.tta,
        )
        train_pix_vals.append(sm_low.flatten())
    train_pix_vals = np.concatenate(train_pix_vals)
    train_pixel_max = float(train_pix_vals.max())
    train_p999 = float(np.percentile(train_pix_vals, 99.9))
    print(f'train_pixel_max={train_pixel_max:.4f}  train_p99.9={train_p999:.4f}')

    # ---- Pick the pixel threshold ------------------------------------------
    score_maps_full = [r['sm'] for r in test_records]
    gt_masks_full = [r['gt'] for r in test_records]
    sweep = None
    if args.threshold_target == 'manual':
        if args.threshold is None:
            raise SystemExit('--threshold-target manual requires --threshold')
        chosen_thr = float(args.threshold)
        thr_meta = dict(mode='manual', threshold=chosen_thr)
    elif args.threshold_target == 'train_p999':
        chosen_thr = train_p999
        thr_meta = dict(mode='train_p999', threshold=chosen_thr,
                        note='GT-free; threshold = 99.9th percentile of '
                             'train pixel scores. Use for production.')
        print(f'GT-free threshold (train_p99.9): {chosen_thr:.4f}')
    elif args.threshold_target == 'train_pixel_max':
        chosen_thr = train_pixel_max
        thr_meta = dict(mode='train_pixel_max', threshold=chosen_thr,
                        note='GT-free; threshold = max train pixel score.')
        print(f'GT-free threshold (train_pixel_max): {chosen_thr:.4f}')
    else:
        best, sweep = find_threshold(score_maps_full, gt_masks_full,
                                     mode=args.threshold_target,
                                     min_recall=args.min_recall)
        if best is None:
            print('no GT to tune on; falling back to train_p99.9')
            chosen_thr = train_p999
            thr_meta = dict(mode='train_p99.9', threshold=chosen_thr)
        else:
            chosen_thr = best['threshold']
            thr_meta = dict(mode=args.threshold_target, **best)
            print(f"GT-tuned threshold: {chosen_thr:.4f}  "
                  f"F1={best['f1']:.4f}  P={best['precision']:.4f}  "
                  f"R={best['recall']:.4f}  IoU={best.get('iou', 0.0):.4f}")
    if sweep is not None:
        with open(out_root / 'threshold_sweep.json', 'w', encoding='utf-8') as f:
            json.dump(dict(best=thr_meta, sweep=sweep,
                           train_pixel_max=train_pixel_max,
                           train_p999=train_p999), f, indent=2)

    # ---- Generate FINAL mask once and reuse everywhere ---------------------
    pred_dir = out_root / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_root / 'panel'
    panel_dir.mkdir(parents=True, exist_ok=True)

    cleanup_kernel = int(cfg.get('morph_kernel', 3)) if args.apply_clean_mask else 0
    cleanup_min_area = int(cfg.get('min_area', 30)) if args.apply_clean_mask else 0

    summary = []
    for r in test_records:
        sm = r['sm']
        orig = r['orig']
        H, W = orig.shape[:2]
        # The one and only mask.
        mask = (sm >= chosen_thr).astype(np.uint8) * 255
        if args.apply_clean_mask:
            mask = clean_mask_strict(mask, kernel=cleanup_kernel, min_area=cleanup_min_area)
        stem = r['stem']

        # raw + overlay + 5-panel use the SAME mask
        cv2.imwrite(str(pred_dir / f'{stem}_mask.png'), mask)
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_mask.png'),
                    cv2.cvtColor(overlay_mask_rgb(orig, mask), cv2.COLOR_RGB2BGR))
        ov_hm = overlay_heatmap_rgb(orig, sm, train_pixel_max)
        cv2.imwrite(str(pred_dir / f'{stem}_overlay_heatmap.png'),
                    cv2.cvtColor(ov_hm, cv2.COLOR_RGB2BGR))
        # normalised heatmap PNG for documentation
        anchor = train_pixel_max
        norm_u8 = (np.clip((sm - anchor * 0.5) / max(anchor * 1.1, 1e-6), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(pred_dir / f'{stem}_heatmap.png'), norm_u8)
        np.save(str(pred_dir / f'{stem}_scores.npy'), sm.astype(np.float32))
        if r['gt'] is not None and (r['gt'] > 0).any():
            cv2.imwrite(str(pred_dir / f'{stem}_real_gt.png'), r['gt'])

        # 6-panel composite: image | heatmap | mask | gt | conf_fg | conf_bg
        panel = make_6panel(orig, mask, r['gt'], sm, train_pixel_max,
                            panel_size=cfg.get('panel_size', 320))
        cv2.imwrite(str(panel_dir / f'{stem}_panel.png'),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        summary.append(dict(
            image_path=r['image_path'],
            defect_type=r['defect_type'],
            image_score=r['image_score'],
            mask_pct=float(100.0 * (mask > 0).sum() / mask.size),
        ))

    # ---- per-defect summary printout --------------------------------------
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
            input_size=cfg['input_size'],
            coreset_ratio=cfg['coreset_ratio'],
            anomaly_score_num_nn=num_nn,
            train_augment=bool(cfg.get('train_augment', False)),
            train_repeat=int(cfg.get('train_repeat', 1)),
            n_train_images=len(train_files),
            threshold_meta=thr_meta,
            train_pixel_max=train_pixel_max,
            train_p999=train_p999,
            apply_clean_mask=bool(args.apply_clean_mask),
            tta=args.tta,
            guided_filter=dict(enabled=bool(args.guided_filter),
                               radius=args.gf_radius, eps=args.gf_eps),
            predictions=summary,
        ), f, indent=2, default=str)


if __name__ == '__main__':
    main()
