"""Generate synthetic crack-like defects on normal images to expand the
Level-3 training set. Addresses the core bottleneck: only 6 labelled
defective images.

Strategy ("CutPaste with crack patches"):
  For each normal training image, with prob p, sample a real crack
  region from a defective image's GT, paste it into the normal at a
  random foreground location. Produces (score_map, grayscale, mask)
  triples that look like new defective samples.

Differs from the earlier `--synth-aug` flag (which spatially shuffled
patches within the SAME image at training time) in two ways:
  - Pastes between samples (real crack patch from a defective image
    into a NORMAL image), creating genuinely new positive examples
  - Synthesises full samples up-front (not at every dataloader step),
    so the model sees the same synthetic positives repeatedly and can
    actually learn them
"""
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
                   help='PatchCore run dir with predictions/test_*_scores.npy')
    p.add_argument('--gt-dir', required=True,
                   help='Folder with <stem>_mask.png GT masks (poly recommended)')
    p.add_argument('--orig-dir', required=True,
                   help='Folder with the original test images')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-synth', type=int, default=24,
                   help='Number of synthetic defective samples to generate.')
    p.add_argument('--paste-per-image', type=int, default=2,
                   help='Number of crack patches to paste per synthetic image.')
    p.add_argument('--max-translate', type=int, default=200,
                   help='Max +/- pixel jitter when placing the patch.')
    p.add_argument('--roi-threshold', type=int, default=8)
    p.add_argument('--roi-margin', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    pred_dir = Path(args.pred_dir)
    out = Path(args.out_dir)
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    gt_lookup = {p.stem.replace('_mask', ''): p
                 for p in Path(args.gt_dir).rglob('*_mask.png')}

    # Load all samples first.
    samples = []
    for f in sorted((pred_dir / 'predictions').glob('*_scores.npy')):
        stem_full = f.stem.replace('_scores', '')
        stem = stem_full.split('_', 1)[-1] if '_' in stem_full else stem_full
        scores = np.load(f)
        orig_path = None
        for ext in ('.bmp', '.png', '.jpg'):
            cand = Path(args.orig_dir) / f'{stem}{ext}'
            if cand.exists(): orig_path = cand; break
        if orig_path is None:
            continue
        orig = np.asarray(Image.open(orig_path).convert('RGB'))
        x0, y0, x1, y1 = roi_bbox_from_image(orig, args.roi_threshold, args.roi_margin)
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
        H, W = gray.shape
        # gt: try fuzzy match
        gt = np.zeros((H, W), dtype=np.uint8)
        for s in (stem, stem.replace('_Normal', '').replace('_normal', '')):
            if s in gt_lookup:
                gt_full = cv2.imread(str(gt_lookup[s]), cv2.IMREAD_GRAYSCALE)
                gt = gt_full[y0:y1, x0:x1]
                if gt.shape != (H, W):
                    gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
                break
        defect = bool((gt > 0).any())
        samples.append(dict(stem=stem, scores=scores, gray=gray, gt=gt,
                            defect=defect))
        print(f'  loaded {stem:14s}  defect={defect}  shape={gray.shape}')

    defective = [s for s in samples if s['defect']]
    normal    = [s for s in samples if not s['defect']]
    print(f'\n{len(defective)} defective, {len(normal)} normal')

    # Extract crack patches (connected components in GT)
    patches = []
    for s in defective:
        n_lab, labels = cv2.connectedComponents((s['gt'] > 0).astype(np.uint8))
        for lbl in range(1, n_lab):
            yy, xx = np.where(labels == lbl)
            if xx.size < 20: continue
            x0, x1 = int(xx.min()), int(xx.max()) + 1
            y0, y1 = int(yy.min()), int(yy.max()) + 1
            if (x1 - x0) < 5 or (y1 - y0) < 5: continue
            patch_sm  = s['scores'][y0:y1, x0:x1].copy()
            patch_gr  = s['gray'][y0:y1, x0:x1].copy()
            patch_gt  = (labels == lbl)[y0:y1, x0:x1].astype(np.uint8) * 255
            patches.append(dict(sm=patch_sm, gr=patch_gr, gt=patch_gt,
                                src=s['stem']))
    print(f'extracted {len(patches)} real crack patches from {len(defective)} defective images')

    # Generate synthetic samples: pick a random normal, paste a patch.
    n_made = 0
    for i in range(args.n_synth):
        base = normal[rng.integers(len(normal))]
        sm = base['scores'].astype(np.float32).copy()
        gr = base['gray'].copy()
        gt = np.zeros_like(base['gt'])
        # mask of foreground (non-bg) area where pasting is OK
        fg_mask = gr > 20
        ys_fg, xs_fg = np.where(fg_mask)
        if xs_fg.size == 0: continue
        for _ in range(args.paste_per_image):
            patch = patches[rng.integers(len(patches))]
            ph, pw = patch['gt'].shape
            H, W = gr.shape
            if ph >= H or pw >= W: continue
            # pick a random foreground pixel as the patch center
            pick_idx = rng.integers(xs_fg.size)
            cx = int(xs_fg[pick_idx])
            cy = int(ys_fg[pick_idx])
            # small additional jitter
            jx = int(rng.integers(-args.max_translate, args.max_translate + 1))
            jy = int(rng.integers(-args.max_translate, args.max_translate + 1))
            cx = np.clip(cx + jx, pw // 2, W - pw // 2 - 1)
            cy = np.clip(cy + jy, ph // 2, H - ph // 2 - 1)
            x0, x1 = cx - pw // 2, cx - pw // 2 + pw
            y0, y1 = cy - ph // 2, cy - ph // 2 + ph
            patch_mask = (patch['gt'] > 0).astype(np.float32)
            # blend: max for score (high = defect), alpha blend for gray
            roi_sm = sm[y0:y1, x0:x1]
            roi_gr = gr[y0:y1, x0:x1].astype(np.float32)
            sm[y0:y1, x0:x1] = np.maximum(roi_sm, patch['sm'] * patch_mask)
            blended_gr = roi_gr * (1 - patch_mask) + patch['gr'].astype(np.float32) * patch_mask
            gr[y0:y1, x0:x1] = blended_gr.astype(np.uint8)
            gt[y0:y1, x0:x1] = np.maximum(gt[y0:y1, x0:x1],
                                          (patch['gt'] > 0).astype(np.uint8) * 255)
        # save: same naming as PatchCore runner so foosung_level3_refinement
        # can pick them up: test_<stem>_scores.npy + grayscale + GT mask
        synth_stem = f'synth{i:03d}'
        np.save(str(out / 'predictions' / f'test_{synth_stem}_scores.npy'),
                sm.astype(np.float32))
        cv2.imwrite(str(out / 'predictions' / f'test_{synth_stem}_orig.png'), gr)
        cv2.imwrite(str(out / 'predictions' / f'test_{synth_stem}_real_gt.png'), gt)
        n_made += 1

    # Copy real samples through so foosung_level3_refinement can find them too
    real_out = out / 'predictions'
    for s in samples:
        np.save(str(real_out / f"test_{s['stem']}_scores.npy"),
                s['scores'].astype(np.float32))
        cv2.imwrite(str(real_out / f"test_{s['stem']}_orig.png"), s['gray'])
        if s['defect']:
            cv2.imwrite(str(real_out / f"test_{s['stem']}_real_gt.png"), s['gt'])

    # Mirror summary.json so the loader picks up train_pixel_max
    import json, shutil
    shutil.copy(pred_dir / 'summary.json', out / 'summary.json')
    print(f'\nwrote {n_made} synthetic samples to {real_out}')


if __name__ == '__main__':
    main()
