"""Convert Foosung LabelMe-style JSON (rectangle CRACK shapes) into binary
mask PNGs that scripts/evaluate_against_gt.py can sweep against.

Output layout matches MVTec ground_truth/<defect>/<stem>_mask.png so the
existing evaluator works without changes:

    <out>/<defect>/<stem>_mask.png       white = CRACK, black = ok

For images with empty shapes (= no defect, normal), an all-black mask is
emitted under <out>/good/<stem>_mask.png so the evaluator still sees
those frames.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labels-dir', required=True,
                   help='Folder of LabelMe JSON files (one per image).')
    p.add_argument('--out-dir', required=True,
                   help='Where to write GT mask PNGs.')
    p.add_argument('--defect-label', default='CRACK')
    return p.parse_args()


def main():
    args = parse_args()
    labels = sorted(Path(args.labels_dir).glob('*.json'))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_defect = 0
    n_good = 0
    n_shape_total = 0
    n_rect = 0
    n_poly = 0
    for j in labels:
        data = json.loads(j.read_text(encoding='utf-8'))
        H = int(data['imageHeight'])
        W = int(data['imageWidth'])
        mask = np.zeros((H, W), dtype=np.uint8)
        shapes = [s for s in data.get('shapes', [])
                  if s.get('label') == args.defect_label]
        for s in shapes:
            st = s.get('shape_type')
            if st == 'rectangle' and len(s['points']) == 2:
                (x0, y0), (x1, y1) = s['points']
                x0, x1 = sorted([int(round(x0)), int(round(x1))])
                y0, y1 = sorted([int(round(y0)), int(round(y1))])
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(W, x1); y1 = min(H, y1)
                mask[y0:y1, x0:x1] = 255
                n_rect += 1
            elif st == 'polygon' and len(s['points']) >= 3:
                pts = np.array([[int(round(x)), int(round(y))]
                                 for x, y in s['points']], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                n_poly += 1
            else:
                print(f'  WARN {j.stem}: unsupported shape_type={st} '
                      f'(pts={len(s["points"])})')
                continue
        if shapes:
            n_defect += 1
            n_shape_total += len(shapes)
            subdir = out / args.defect_label.lower()
        else:
            n_good += 1
            subdir = out / 'good'
        subdir.mkdir(parents=True, exist_ok=True)
        stem = j.stem
        cv2.imwrite(str(subdir / f'{stem}_mask.png'), mask)
        # report mask pixel coverage (helpful sanity for polygon vs rect)
        mask_pct = 100.0 * (mask > 0).sum() / mask.size
        print(f'  {stem:10s}  shapes={len(shapes):>2d}  '
              f'mask_pct={mask_pct:>6.3f}%  -> {subdir.name}/')
    print()
    print(f'wrote {n_defect} defect masks, {n_good} good masks, '
          f'{n_shape_total} shapes ({n_rect} rect + {n_poly} polygon)')


if __name__ == '__main__':
    main()
