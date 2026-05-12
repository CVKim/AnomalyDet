"""3-panel viz from Level 3 _prob.npy outputs: image+GT (green), image+pred
(red), overlay (both)."""
import argparse
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
                   help='Folder of <stem>_prob.npy')
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--orig-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    gt_lookup = {p.stem.replace('_mask', ''): p
                 for p in Path(args.gt_dir).rglob('*_mask.png')}
    for f in sorted(Path(args.pred_dir).glob('*_prob.npy')):
        stem = f.stem.replace('_prob', '')
        gt_path = gt_lookup.get(stem)
        if gt_path is None:
            for suf in ('_Normal', '_normal', '_Defect', '_defect', '_NG', '_OK', '_ng', '_ok'):
                if stem.endswith(suf) and stem[:-len(suf)] in gt_lookup:
                    gt_path = gt_lookup[stem[:-len(suf)]]; break
        if gt_path is None:
            continue
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if not (gt > 0).any():
            continue
        orig_path = None
        for ext in ('.bmp', '.png', '.jpg', '.jpeg', '.tif'):
            cand = Path(args.orig_dir) / f'{stem}{ext}'
            if cand.exists():
                orig_path = cand; break
        if orig_path is None:
            continue
        orig = np.asarray(Image.open(orig_path).convert('RGB'))
        x0, y0, x1, y1 = roi_bbox_from_image(orig, args.roi_threshold,
                                              args.roi_margin)
        orig = orig[y0:y1, x0:x1]
        gt = gt[y0:y1, x0:x1]
        H, W = orig.shape[:2]
        prob = np.load(f)
        if prob.shape != (H, W):
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        pred = (prob >= args.threshold).astype(np.uint8) * 255

        p1 = orig.copy()
        cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(p1, (x, y), (x + w, y + h), (0, 255, 0), 8)

        red = np.zeros_like(orig); red[..., 0] = 255
        p2 = np.where(pred[..., None] > 0,
                      (orig * 0.5 + red * 0.5).astype(np.uint8), orig)
        p3 = p2.copy()
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(p3, (x, y), (x + w, y + h), (0, 255, 0), 8)

        h_target = 600
        scale = h_target / H
        nw = int(W * scale)
        tiles = []
        for img, lab in zip([p1, p2, p3],
                             ['image + GT bbox (green)',
                              'image + Level3 PRED (red)',
                              'overlap: green=GT, red=PRED']):
            t = cv2.resize(img, (nw, h_target))
            cv2.rectangle(t, (0, 0), (nw - 1, 30), (0, 0, 0), -1)
            cv2.putText(t, lab, (5, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        comp = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(out / f'{stem}_predVSgt.png'),
                    cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f'  wrote {stem}_predVSgt.png')


if __name__ == '__main__':
    main()
