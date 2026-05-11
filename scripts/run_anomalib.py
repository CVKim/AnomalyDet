"""Run the anomalib reference PatchCore on an MVTec category and dump
heatmap + mask PNGs and per-image image scores into a folder that lines
up with `outputs/<cat>/predictions/` from our own pipeline. The point
is direct side-by-side comparison: same data, same task, different
implementation.

Two presets:
    --preset wrn50    : official PatchCore (wide_resnet50_2, layers 2+3)
    --preset dinov2   : PatchCore + DINOv2 ViT-S/14 via timm

Usage:
    python scripts/run_anomalib.py --preset wrn50 `
        --data-root "E:\\dataset\\mvtec_anomaly_detection_" `
        --category hazelnut `
        --output outputs/anomalib_wrn50_hazelnut
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
import lightning.pytorch as pl
from anomalib.data import MVTec
from anomalib.models import Patchcore


PRESETS = {
    'wrn50': dict(
        backbone='wide_resnet50_2',
        layers=['layer2', 'layer3'],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    ),
    'dinov2': dict(
        # timm name for DINOv2 ViT-S/14 (Meta's release).
        backbone='vit_small_patch14_dinov2.lvd142m',
        # timm exposes transformer blocks as blocks.<i> on hook taps.
        layers=['blocks.5', 'blocks.11'],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', choices=list(PRESETS), default='wrn50')
    p.add_argument('--data-root', required=True)
    p.add_argument('--category', default='hazelnut')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--output', required=True)
    p.add_argument('--max-epochs', type=int, default=1,
                   help='PatchCore is single-pass; 1 epoch is enough.')
    return p.parse_args()


def overlay_heatmap_bgr(image_rgb, heat_norm, alpha=0.5):
    hm8 = (np.clip(heat_norm, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, rgb, alpha, 0)


def overlay_mask_bgr(image_rgb, mask, alpha=0.4):
    out = image_rgb.copy()
    out[mask > 0] = np.array([255, 0, 0], dtype=out.dtype)
    return cv2.addWeighted(image_rgb, 1 - alpha, out, alpha, 0)


def main():
    args = parse_args()
    preset = PRESETS[args.preset]

    print(f'preset={args.preset}, backbone={preset["backbone"]}, '
          f'layers={preset["layers"]}')

    datamodule = MVTec(
        root=args.data_root,
        category=args.category,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
        image_size=(args.image_size, args.image_size),
    )

    model = Patchcore(**preset)

    out_root = Path(args.output)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Snapshot exact configuration so the output dir is self-describing.
    with open(out_root / 'config_used.yaml', 'w', encoding='utf-8') as f:
        import yaml as _yaml
        _yaml.safe_dump(dict(
            preset=args.preset, backbone=preset['backbone'],
            layers=preset['layers'], coreset_sampling_ratio=preset['coreset_sampling_ratio'],
            num_neighbors=preset['num_neighbors'], pre_trained=preset['pre_trained'],
            image_size=args.image_size, category=args.category,
            data_root=args.data_root, max_epochs=args.max_epochs,
        ), f, sort_keys=False)
    with open(out_root / 'run_command.txt', 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write(f'run_at: {datetime.now().isoformat(timespec="seconds")}\n')

    # Use Lightning Trainer directly: anomalib's Engine wrapper creates
    # symlinks for versioned output dirs which fail on Windows without
    # admin / dev-mode. The underlying LightningModule works fine.
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        default_root_dir=str(out_root / 'lightning_logs'),
        logger=False,
        enable_checkpointing=False,
        # PatchCore collects embeddings during training_step and only
        # builds the memory bank in on_validation_start. The Lightning
        # sanity validation runs before any training_step has executed,
        # so embeddings is empty and torch.vstack blows up. Skip it.
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)
    test_results = trainer.test(model=model, datamodule=datamodule)
    print('test metrics:', test_results)

    predictions = trainer.predict(model=model, datamodule=datamodule)

    # Walk anomalib predictions and write outputs in the same layout as
    # our pipeline (defect_stem prefixed filenames).
    def _get(b, key):
        if isinstance(b, dict):
            return b.get(key)
        return getattr(b, key, None)

    def _to_np(v):
        if v is None:
            return None
        if hasattr(v, 'cpu'):
            return v.cpu().numpy()
        return np.asarray(v)

    # anomalib v1.2 emits per-batch dicts with keys:
    #   image_path, label, image, mask (GT), anomaly_maps, pred_scores
    # No thresholded pred_mask is produced at predict-time; we threshold
    # ourselves at the test-set max-good-pixel (recall-first) so the
    # comparison against our own pipeline isn't unfairly tight.

    # First pass: gather all heatmaps + GT-label info to derive threshold.
    all_max_normal_pix = []
    rows = []
    def _first_present(b, *keys):
        for k in keys:
            v = _get(b, k)
            if v is not None:
                return v
        return None

    for batch in predictions:
        anomaly_maps = _to_np(_first_present(batch, 'anomaly_maps', 'anomaly_map'))
        image_scores = _to_np(_first_present(batch, 'pred_scores', 'pred_score'))
        image_paths = _first_present(batch, 'image_path', 'image_paths') or []
        labels = _to_np(_get(batch, 'label'))
        for i, ip in enumerate(image_paths):
            rows.append(dict(
                path=str(ip),
                hm=anomaly_maps[i],
                score=float(image_scores[i]),
                label=int(labels[i]) if labels is not None else -1,
            ))
    normal_pix = [r['hm'].max() for r in rows if r['label'] == 0]
    threshold_value = float(np.percentile(np.concatenate([r['hm'].flatten() for r in rows if r['label'] == 0]), 99.9)) if normal_pix else float(np.percentile(np.concatenate([r['hm'].flatten() for r in rows]), 99.0))
    print(f'derived threshold (train_p99.9 of normal heatmaps): {threshold_value:.4f}')

    summary = []
    for r in rows:
        img_path = r['path']
        defect = Path(img_path).parent.name
        stem = f'{defect}_{Path(img_path).stem}'

        orig = cv2.imread(img_path)[..., ::-1].copy()
        H, W = orig.shape[:2]
        hm = r['hm']
        if hm.ndim == 3:
            hm = hm[0]
        hm_full = cv2.resize(hm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        hm_norm = hm_full - hm_full.min()
        denom = hm_norm.max() if hm_norm.max() > 1e-8 else 1.0
        hm_norm = hm_norm / denom

        mask = (hm_full >= threshold_value).astype(np.uint8) * 255

        cv2.imwrite(str(out_root / f'{stem}_heatmap.png'),
                    (hm_norm * 255).astype(np.uint8))
        # Raw float score map at full image resolution -- needed for
        # GT-tuned threshold sweeps after the fact.
        np.save(str(out_root / f'{stem}_scores.npy'), hm_full.astype(np.float32))
        cv2.imwrite(str(out_root / f'{stem}_mask.png'), mask)
        cv2.imwrite(str(out_root / f'{stem}_overlay_heatmap.png'),
                    cv2.cvtColor(overlay_heatmap_bgr(orig, hm_norm), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_root / f'{stem}_overlay_mask.png'),
                    cv2.cvtColor(overlay_mask_bgr(orig, mask), cv2.COLOR_RGB2BGR))
        summary.append(dict(
            image_path=img_path,
            defect_type=defect,
            image_score=r['score'],
            mask_pct=float(100.0 * (mask > 0).sum() / mask.size),
        ))

    # (legacy loop above replaced by two-pass logic)

    with open(out_root / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'preset': args.preset,
            'backbone': preset['backbone'],
            'layers': preset['layers'],
            'category': args.category,
            'image_size': args.image_size,
            'n_test': len(summary),
            'predictions': summary,
            'test_metrics': test_results,
        }, f, indent=2, default=str)

    # quick per-defect breakdown to stdout
    by_defect = {}
    for s in summary:
        by_defect.setdefault(s['defect_type'], []).append(s)
    print(f'\n=== anomalib PatchCore preset={args.preset} on {args.category} ===')
    for d in sorted(by_defect):
        rows = by_defect[d]
        masks = [r['mask_pct'] for r in rows]
        scores = [r['image_score'] for r in rows]
        print(f"  {d:14s} n={len(rows):2d}  mask% mean={np.mean(masks):5.2f}  max={np.max(masks):5.2f}  "
              f"img_score mean={np.mean(scores):6.3f}")


if __name__ == '__main__':
    main()
