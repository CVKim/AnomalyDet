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
                   help='Folder with <stem>_mask.png. Optional. Used only '
                        'if --rect-labels-dir is not given.')
    p.add_argument('--rect-labels-dir', default=None,
                   help='Folder with LabelMe rectangle JSON files. When set, '
                        'GT boxes in the "box" column are pulled directly from '
                        'the JSON points instead of being re-derived from the '
                        'binary mask.')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--threshold', type=float, default=None,
                   help='Mask threshold (defaults to summary.json threshold).')
    p.add_argument('--roi-crop', action='store_true', default=False)
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--panel-size', type=int, default=320)
    return p.parse_args()


def _load_rect_boxes(json_path: Path, label: str = 'CRACK'):
    """Returns [(x0, y0, x1, y1)] for every rectangle shape with the given label."""
    import json as _json
    data = _json.loads(Path(json_path).read_text(encoding='utf-8'))
    boxes = []
    for s in data.get('shapes', []):
        if s.get('label') != label:
            continue
        if s.get('shape_type') == 'rectangle' and len(s['points']) == 2:
            (x0, y0), (x1, y1) = s['points']
            boxes.append((int(round(min(x0, x1))), int(round(min(y0, y1))),
                          int(round(max(x0, x1))), int(round(max(y0, y1)))))
        elif s.get('shape_type') == 'polygon' and len(s['points']) >= 3:
            pts = np.array(s['points'], dtype=np.float32)
            x0 = int(round(pts[:, 0].min())); x1 = int(round(pts[:, 0].max()))
            y0 = int(round(pts[:, 1].min())); y1 = int(round(pts[:, 1].max()))
            boxes.append((x0, y0, x1, y1))
    return boxes


def _label(panel, txt, panel_size):
    bar = np.zeros((30, panel_size, 3), dtype=np.uint8)
    cv2.putText(bar, txt, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, panel], axis=0)


def _connected_bboxes_with_scores(mask, score_map):
    """List of (x0, y0, x1, y1, max_score) per connected component in mask."""
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        region = score_map[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i]
        smax = float(region.max()) if region.size else 0.0
        out.append((x, y, x + w, y + h, smax))
    return out


def make_5panel(orig_rgb, mask, gt, score_map, anchor, panel_size,
                gt_boxes=None, pred_show_score=True):
    """5-panel: image | mask pred | pred conf fg | pred conf bg | box.

    box column draws:
      GT boxes  (from gt_boxes list of (x0,y0,x1,y1))   in GREEN
      PRED boxes (from connected components of `mask`)  in RED, with
      score label (raw score_map max within the region, normalised by
      `anchor`).
    """
    H, W = orig_rgb.shape[:2]

    # heatmap colourised (used by pred conf fg / bg, NOT shown as its own column)
    sm_norm = np.clip(score_map / max(anchor, 1e-6), 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((sm_norm * 255).astype(np.uint8),
                                            cv2.COLORMAP_JET),
                         cv2.COLOR_BGR2RGB)

    # mask pred: image with predicted defect overlaid in cyan
    pred_mask_img = orig_rgb.copy()
    pred_b = (mask > 0)
    pred_mask_img[pred_b] = (0, 255, 255)
    pred_mask_img = cv2.addWeighted(orig_rgb, 0.45, pred_mask_img, 0.55, 0)

    # pred conf fg: heatmap masked to predicted defect pixels (black elsewhere)
    fg = np.zeros_like(orig_rgb)
    fg[pred_b] = heat[pred_b]

    # pred conf bg: heatmap masked to predicted non-defect pixels
    bg = np.zeros_like(orig_rgb)
    bg[~pred_b] = heat[~pred_b]

    # box column: image + GT boxes (green) + PRED boxes (red) with score label
    box_img = orig_rgb.copy()
    if gt_boxes:
        for (x0, y0, x1, y1) in gt_boxes:
            cv2.rectangle(box_img, (x0, y0), (x1, y1), (0, 255, 0), 8)
    pred_boxes = _connected_bboxes_with_scores(mask, score_map)
    # Filter tiny predicted components so the box panel doesn't get spammed
    min_area = max(50, (H * W) // 200_000)
    pred_boxes = [b for b in pred_boxes
                  if (b[2] - b[0]) * (b[3] - b[1]) >= min_area]
    for (x0, y0, x1, y1, smax) in pred_boxes:
        cv2.rectangle(box_img, (x0, y0), (x1, y1), (255, 0, 0), 6)
        if pred_show_score:
            txt = f'{smax / max(anchor, 1e-6):.2f}'
            cv2.putText(box_img, txt, (x0, max(y0 - 6, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4, cv2.LINE_AA)

    def resize_pad(img):
        scale = panel_size / max(H, W)
        nw, nh = int(W * scale), int(H * scale)
        r = cv2.resize(img, (nw, nh))
        pad_h, pad_w = panel_size - nh, panel_size - nw
        return cv2.copyMakeBorder(r, pad_h // 2, pad_h - pad_h // 2,
                                   pad_w // 2, pad_w - pad_w // 2,
                                   cv2.BORDER_CONSTANT, value=0)

    tiles = [
        _label(resize_pad(orig_rgb),       'image',                  panel_size),
        _label(resize_pad(pred_mask_img),  'mask pred',              panel_size),
        _label(resize_pad(fg),              'pred conf fg',           panel_size),
        _label(resize_pad(bg),              'pred conf bg',           panel_size),
        _label(resize_pad(box_img),         'box (GT green / PRED red)', panel_size),
    ]
    return np.concatenate(tiles, axis=1)


# Aliases kept so older callers don't break.
make_6panel = make_5panel


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

    # Optional: rect-label JSON lookup for the "box" column
    rect_lookup = {}
    if args.rect_labels_dir:
        for p in Path(args.rect_labels_dir).rglob('*.json'):
            rect_lookup[p.stem] = p

    def fuzzy(stem, lookup):
        if stem in lookup:
            return lookup[stem]
        for suf in ('_Normal', '_normal', '_Defect', '_defect', '_NG', '_OK'):
            if stem.endswith(suf) and stem[:-len(suf)] in lookup:
                return lookup[stem[:-len(suf)]]
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

        # GT mask (used by panel viz internals if no rect labels given)
        gt = None
        gt_path = fuzzy(stem, gt_lookup)
        if gt_path is not None:
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt.shape != (H, W):
                full_h, full_w = gt.shape
                if args.roi_crop and full_h > H and full_w > W:
                    gt = gt[y0:y1, x0:x1]
                if gt.shape != (H, W):
                    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

        # GT boxes for the box panel: prefer rect-labels-dir
        gt_boxes_for_panel = None
        rect_path = fuzzy(stem, rect_lookup)
        if rect_path is not None:
            raw_boxes = _load_rect_boxes(rect_path)
            # Re-project to ROI-cropped coords if needed
            if args.roi_crop:
                gt_boxes_for_panel = []
                for (rx0, ry0, rx1, ry1) in raw_boxes:
                    nx0 = max(0, rx0 - x0); nx1 = min(W, rx1 - x0)
                    ny0 = max(0, ry0 - y0); ny1 = min(H, ry1 - y0)
                    if nx1 > nx0 and ny1 > ny0:
                        gt_boxes_for_panel.append((nx0, ny0, nx1, ny1))
            else:
                gt_boxes_for_panel = raw_boxes
        elif gt is not None and (gt > 0).any():
            # Fallback: derive bboxes from the mask via connected components
            gt_boxes_for_panel = [(b[0], b[1], b[2], b[3])
                                   for b in _connected_bboxes_with_scores(gt, sm)]

        panel = make_5panel(orig, mask, gt, sm, train_max, args.panel_size,
                             gt_boxes=gt_boxes_for_panel)
        out_path = out / f'{stem}_panel.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f'  wrote {out_path.name}')


if __name__ == '__main__':
    main()
