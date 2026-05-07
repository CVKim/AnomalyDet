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
from src.data.transforms import build_image_transform, build_train_transform
from src.models.patchcore import PatchCore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/default.yaml')
    p.add_argument('--data-root', type=str, default='data/mvtec')
    p.add_argument('--category', type=str, default='bottle')
    p.add_argument('--output', type=str, default=None,
                   help='Where to save memory_bank.pt. Default: outputs/<category>')
    p.add_argument('--augment', action='store_true', default=None,
                   help='Override config: enable rotation/flip augmentation when '
                        'building the memory bank (for parts with pose variation).')
    p.add_argument('--no-augment', dest='augment', action='store_false',
                   help='Override config: disable augmentation.')
    p.add_argument('--repeat', type=int, default=None,
                   help='Override config: each train image is fetched N times '
                        'under different random transforms.')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    augment = (args.augment if args.augment is not None
               else bool(cfg.get('train_augment', False)))
    repeat = int(args.repeat if args.repeat is not None
                 else cfg.get('train_repeat', 1))

    # fit_loader builds the memory bank — uses augmentation when configured
    # so the bank covers natural pose variation.
    fit_transform = build_train_transform(cfg['input_size'], augment=augment)
    fit_ds = MVTecDataset(args.data_root, args.category,
                          split='train', transform=fit_transform, repeat=repeat)
    if len(fit_ds) == 0:
        raise RuntimeError(f'No training images found under '
                           f'{Path(args.data_root) / args.category / "train" / "good"}')
    print(f'fit images: {len(fit_ds)} (augment={augment}, repeat={repeat})')

    fit_loader = DataLoader(
        fit_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=(device == 'cuda'),
    )

    # cal_loader: original (non-augmented, repeat=1) training data so the
    # threshold floor reflects real-image baseline scores against the
    # augmentation-covered memory bank.
    cal_transform = build_image_transform(cfg['input_size'])
    cal_ds = MVTecDataset(args.data_root, args.category,
                          split='train', transform=cal_transform, repeat=1)
    cal_loader = DataLoader(
        cal_ds,
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
    model.fit(fit_loader)
    model.calibrate(cal_loader)

    out_root = Path(args.output) if args.output else Path('outputs') / args.category
    out_root.mkdir(parents=True, exist_ok=True)
    bank_path = out_root / 'memory_bank.pt'
    model.save(str(bank_path))
    print(f'saved memory bank: {bank_path}')


if __name__ == '__main__':
    main()
