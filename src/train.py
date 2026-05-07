import argparse
from pathlib import Path
import sys

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import MVTecDataset
from src.data.transforms import build_image_transform
from src.models.patchcore import PatchCore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--data-root', type=str, default='data/mvtec')
    p.add_argument('--category', type=str, default='bottle')
    p.add_argument('--output', type=str, default=None,
                   help='Where to save memory_bank.pt. Default: outputs/<category>')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    transform = build_image_transform(cfg['input_size'])
    train_ds = MVTecDataset(args.data_root, args.category,
                            split='train', transform=transform)
    if len(train_ds) == 0:
        raise RuntimeError(f'No training images found under '
                           f'{Path(args.data_root) / args.category / "train" / "good"}')
    print(f'training images: {len(train_ds)}')

    loader = DataLoader(
        train_ds,
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
    model.fit(loader)
    # Re-iterate the training set to record the pixel-score distribution
    # the model itself emits on known-good data. The recall-first threshold
    # used at inference is derived from this.
    model.calibrate(loader)

    out_root = Path(args.output) if args.output else Path('outputs') / args.category
    out_root.mkdir(parents=True, exist_ok=True)
    bank_path = out_root / 'memory_bank.pt'
    model.save(str(bank_path))
    print(f'saved memory bank: {bank_path}')


if __name__ == '__main__':
    main()
