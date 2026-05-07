import argparse
from pathlib import Path
import sys

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
from src.data.transforms import build_image_transform
from src.models.patchcore import PatchCore
from src.utils.postprocess import (adaptive_pixel_threshold, clean_mask,
                                   mask_to_labelme_json, save_outputs)
from src.utils.visualize import normalize_map, overlay_heatmap, overlay_mask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--memory-bank', type=str, required=True)
    p.add_argument('--data-root', type=str, default='data/mvtec')
    p.add_argument('--category', type=str, default='bottle')
    p.add_argument('--input-dir', type=str, default=None,
                   help='If set, run on a generic folder instead of MVTec test split.')
    p.add_argument('--output', type=str, default=None,
                   help='Default: outputs/<category>/predictions.')
    p.add_argument('--threshold', type=float, default=None,
                   help='Hard override; bypasses threshold-mode entirely.')
    p.add_argument('--threshold-mode', type=str, default=None,
                   choices=['adaptive', 'train_max', 'train_p999', 'train_p99',
                            'test_percentile'],
                   help='Override the threshold strategy from config.')
    p.add_argument('--save-overlays', action='store_true', default=True)
    return p.parse_args()


def _resolve_global_threshold(mode, cfg, model, results):
    """Resolve a global pixel threshold for non-adaptive modes."""
    train_attr_map = {
        'train_max': 'train_pixel_max',
        'train_p999': 'train_pixel_p999',
        'train_p99': 'train_pixel_p99',
    }
    if mode in train_attr_map:
        val = getattr(model, train_attr_map[mode], None)
        if val is None:
            print(f'warning: model has no calibration for {mode!r}, '
                  f'falling back to test_percentile')
            mode = 'test_percentile'
        else:
            return float(val), mode
    all_pix = np.concatenate([r['heatmap'].flatten() for r in results])
    pct = float(cfg.get('test_percentile', cfg.get('threshold_percentile', 99.0)))
    return float(np.percentile(all_pix, pct)), f'test_p{pct}'


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    transform = build_image_transform(cfg['input_size'])
    if args.input_dir:
        ds = FolderDataset.from_dir(args.input_dir, transform=transform)
    else:
        ds = MVTecDataset(args.data_root, args.category,
                          split='test', transform=transform)
    if len(ds) == 0:
        raise RuntimeError('No test images found.')
    print(f'test images: {len(ds)}')

    loader = DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=(device == 'cuda'),
    )

    model = PatchCore(
        backbone=cfg['backbone'],
        layers=tuple(cfg['layers']),
        input_size=cfg['input_size'],
        coreset_ratio=cfg['coreset_ratio'],
        coreset_projection_dim=cfg.get('coreset_projection_dim', 128),
        device=device,
    )
    model.load(args.memory_bank)

    out_root = Path(args.output) if args.output else Path('outputs') / args.category / 'predictions'
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        with torch.no_grad():
            heatmaps, image_scores = model.predict(images)
        heatmaps = heatmaps.cpu().numpy()
        image_scores = image_scores.cpu().numpy()
        for i in range(len(batch['image_path'])):
            results.append({
                'path': batch['image_path'][i],
                'heatmap': heatmaps[i],
                'image_score': float(image_scores[i]),
                'defect_type': batch.get('defect_type', ['unknown'] * len(images))[i],
            })

    if args.threshold is not None:
        global_threshold = float(args.threshold)
        threshold_mode = 'manual'
    else:
        threshold_mode = args.threshold_mode or cfg.get('threshold_mode', 'adaptive')
        if threshold_mode == 'adaptive':
            global_threshold = None
            if model.train_image_max is None:
                print('warning: model has no train_image_max calibration; '
                      'adaptive mode will fall back to global train_max')
                global_threshold = model.train_pixel_max
                threshold_mode = 'train_max'
        else:
            global_threshold, threshold_mode = _resolve_global_threshold(
                threshold_mode, cfg, model, results)
    image_gate_factor = float(cfg.get('image_gate_factor', 1.3))
    severity_fraction = float(cfg.get('severity_fraction', 0.5))
    pixel_floor_factor = float(cfg.get('pixel_floor_factor', 1.1))
    print(f'threshold mode: {threshold_mode}'
          + (f' value={global_threshold:.4f}' if global_threshold is not None else
             f' gate={image_gate_factor} severity={severity_fraction} '
             f'floor={pixel_floor_factor}'))

    for r in results:
        img_path = r['path']
        defect = r.get('defect_type', 'unknown') or 'unknown'
        unique_name = f'{defect}_{Path(img_path).stem}'
        orig = np.array(Image.open(img_path).convert('RGB'))
        H, W = orig.shape[:2]
        hm_full = cv2.resize(r['heatmap'], (W, H), interpolation=cv2.INTER_LINEAR)

        if threshold_mode == 'adaptive':
            t = adaptive_pixel_threshold(
                hm_full,
                image_score=r['image_score'],
                train_image_max=model.train_image_max,
                train_pixel_max=model.train_pixel_max,
                image_gate_factor=image_gate_factor,
                severity_fraction=severity_fraction,
                pixel_floor_factor=pixel_floor_factor,
            )
        else:
            t = global_threshold

        if not np.isfinite(t):
            mask = np.zeros((H, W), dtype=np.uint8)
        else:
            mask = (hm_full >= t).astype(np.uint8) * 255
            mask = clean_mask(
                mask,
                kernel_size=cfg.get('morph_kernel', 3),
                min_area=cfg.get('min_area', 30),
            )
        json_data = mask_to_labelme_json(
            mask, img_path, orig.shape[:2],
            image_score=r['image_score'],
            poly_eps_ratio=cfg.get('poly_eps_ratio', 0.005),
        )
        json_data['defectType'] = defect
        json_data['threshold'] = float(t) if np.isfinite(t) else None
        json_data['thresholdMode'] = threshold_mode

        hm_norm = normalize_map(hm_full)
        save_outputs(out_root, unique_name, hm_norm, mask, json_data)

        if args.save_overlays:
            ov_hm = overlay_heatmap(orig, hm_norm)
            ov_mask = overlay_mask(orig, mask)
            cv2.imwrite(str(out_root / f'{unique_name}_overlay_heatmap.png'),
                        cv2.cvtColor(ov_hm, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_root / f'{unique_name}_overlay_mask.png'),
                        cv2.cvtColor(ov_mask, cv2.COLOR_RGB2BGR))

    print(f'saved {len(results)} predictions to {out_root}')
    scores = [r['image_score'] for r in results]
    print(f'image scores: min={min(scores):.4f} '
          f'max={max(scores):.4f} '
          f'mean={float(np.mean(scores)):.4f}')


if __name__ == '__main__':
    main()
