"""6-panel composites from an arbitrary tile-based run output.

Reads <pred-dir>/predictions/test_<stem>_scores.npy + original image
+ both GT masks (rect / poly), produces per-image panels:
    image | heatmap | mask pred | gt | pred conf fg | pred conf bg

Threshold defaults to the run's chosen threshold (read from summary.json)
but can be overridden. Same look as the panel/<stem>_panel.png that the
default PatchCore runner emits.
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
    p.add_argument('--pred-dir', required=True)
    p.add_argument('--orig-dir', required=True)
    p.add_argument('--gt-dir', default=None,
                   help='Folder with <stem>_mask.png. Optional; if omitted, '
                        'the GT column is blank.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--threshold', type=float, default=None,
                   help='Mask threshold (defaults to summary.json threshold).')
    p.add_argument('--roi-crop', action='store_true', default=False)
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--panel-size', type=int, default=320)
    return p.parse_args()


def _label(panel, txt, panel_size):
    bar = np.zeros((30, panel_size, 3), dtype=np.uint8)
    cv2.putText(bar, txt, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, panel], axis=0)


def make_6panel(orig_rgb, mask, gt, score_map, anchor, panel_size):
    H, W = orig_rgb.shape[:2]
    # heatmap colourised
    sm_norm = np.clip(score_map / max(anchor, 1e-6), 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((sm_norm * 255).astype(np.uint8),
                                            cv2.COLORMAP_JET),
                         cv2.COLOR_BGR2RGB)
    # mask pred overlay — solid cyan for max visibility on the small panels
    pred_overlay = orig_rgb.copy()
    pred_overlay[mask > 0] = (0, 255, 255)
    # gt overlay — same solid cyan + a contour border for thin defects
    if gt is not None and (gt > 0).any():
        gt_overlay = orig_rgb.copy()
        gt_overlay[gt > 0] = (0, 255, 255)
        # Dilate GT so thin cracks survive the downscale to panel_size.
        gt_dil = cv2.dilate((gt > 0).astype(np.uint8) * 255,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        contours, _ = cv2.findContours(gt_dil, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_overlay, contours, -1, (0, 255, 0), 6)
    else:
        gt_overlay = orig_rgb.copy()
    # pred conf fg (heatmap masked to predicted defect)
    fg = np.zeros_like(orig_rgb); fg[mask > 0] = heat[mask > 0]
    # pred conf bg (heatmap masked to predicted non-defect)
    bg = np.zeros_like(orig_rgb); bg[mask == 0] = heat[mask == 0]

    def resize_pad(img):
        scale = panel_size / max(H, W)
        nw, nh = int(W * scale), int(H * scale)
        r = cv2.resize(img, (nw, nh))
        pad_h, pad_w = panel_size - nh, panel_size - nw
        return cv2.copyMakeBorder(r, pad_h // 2, pad_h - pad_h // 2,
                                   pad_w // 2, pad_w - pad_w // 2,
                                   cv2.BORDER_CONSTANT, value=0)

    tiles = [
        _label(resize_pad(orig_rgb),    'image',         panel_size),
        _label(resize_pad(heat),         'heatmap',       panel_size),
        _label(resize_pad(pred_overlay), 'mask pred',     panel_size),
        _label(resize_pad(gt_overlay),   'gt',            panel_size),
        _label(resize_pad(fg),           'pred conf fg',  panel_size),
        _label(resize_pad(bg),           'pred conf bg',  panel_size),
    ]
    return np.concatenate(tiles, axis=1)


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # threshold: prefer the GT-best from gt_eval.json (set by
    # evaluate_foosung.py) so the mask shown matches the F1-best
    # operating point. Falls back to summary.json's production threshold.
    thr = args.threshold
    train_max = None
    gt_eval_path = pred_dir / 'gt_eval.json'
    summary_path = pred_dir / 'summary.json'
    if thr is None and gt_eval_path.exists():
        try:
            ge = json.loads(gt_eval_path.read_text(encoding='utf-8'))
            thr = ge['best']['threshold']
        except Exception:
            pass
    if summary_path.exists():
        d = json.loads(summary_path.read_text(encoding='utf-8'))
        if thr is None:
            thr = d.get('threshold_meta', {}).get('threshold')
        train_max = d.get('train_pixel_max')
    if thr is None:
        thr = 0.5
    if train_max is None:
        train_max = thr * 2

    # GT lookup with fuzzy suffix matching
    gt_lookup = {}
    if args.gt_dir:
        for p in Path(args.gt_dir).rglob('*_mask.png'):
            gt_lookup[p.stem.replace('_mask', '')] = p

    def fuzzy(stem):
        if stem in gt_lookup:
            return gt_lookup[stem]
        for suf in ('_Normal', '_normal', '_Defect', '_defect'):
            if stem.endswith(suf) and stem[:-len(suf)] in gt_lookup:
                return gt_lookup[stem[:-len(suf)]]
        return None

    print(f'pred_dir={pred_dir}  threshold={thr:.4f}  anchor={train_max:.4f}')
    for f in sorted((pred_dir / 'predictions').glob('*_scores.npy')):
        stem_full = f.stem.replace('_scores', '')
        stem = stem_full.split('_', 1)[-1] if '_' in stem_full else stem_full
        # original image
        orig_path = None
        for ext in ('.bmp', '.png', '.jpg', '.jpeg', '.tif'):
            cand = Path(args.orig_dir) / f'{stem}{ext}'
            if cand.exists():
                orig_path = cand; break
        if orig_path is None:
            print(f'  skip {stem}: no orig found'); continue
        orig = np.asarray(Image.open(orig_path).convert('RGB'))
        if args.roi_crop:
            x0, y0, x1, y1 = roi_bbox_from_image(orig, args.roi_threshold, args.roi_margin)
            orig = orig[y0:y1, x0:x1]
        H, W = orig.shape[:2]
        sm = np.load(f)
        if sm.shape != (H, W):
            sm = cv2.resize(sm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        mask = (sm >= thr).astype(np.uint8) * 255
        gt = None
        gt_path = fuzzy(stem)
        if gt_path is not None:
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt.shape != (H, W):
                # crop GT to ROI if shapes differ
                full_h, full_w = gt.shape
                if args.roi_crop and full_h > H and full_w > W:
                    gt = gt[y0:y1, x0:x1]
                if gt.shape != (H, W):
                    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
        panel = make_6panel(orig, mask, gt, sm, train_max, args.panel_size)
        out_path = out / f'{stem}_panel.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f'  wrote {out_path.name}')


if __name__ == '__main__':
    main()
