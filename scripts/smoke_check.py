"""Lightweight end-to-end check on synthetic data (no MVTec download needed).

Run after env install to verify torch+CUDA+pipeline:
    python scripts/smoke_check.py
"""
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import MVTecDataset
from src.data.transforms import build_image_transform
from src.models.patchcore import PatchCore
from torch.utils.data import DataLoader


def make_synth(root: Path, size: int = 64) -> None:
    rng = np.random.default_rng(0)
    cat = root / 'fake'
    (cat / 'train' / 'good').mkdir(parents=True, exist_ok=True)
    (cat / 'test' / 'defect').mkdir(parents=True, exist_ok=True)
    for i in range(8):
        arr = rng.integers(80, 160, (size, size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cat / 'train' / 'good' / f'{i:03d}.png')
    for i in range(2):
        arr = rng.integers(80, 160, (size, size, 3), dtype=np.uint8)
        arr[10:30, 10:30] = 240
        Image.fromarray(arr).save(cat / 'test' / 'defect' / f'{i:03d}.png')


def main():
    print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, '
          f'devices: {torch.cuda.device_count()}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_synth(tmp)
        transform = build_image_transform(64)
        train_ds = MVTecDataset(tmp, 'fake', split='train', transform=transform)
        test_ds = MVTecDataset(tmp, 'fake', split='test', transform=transform)
        train_loader = DataLoader(train_ds, batch_size=4, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=2, num_workers=0)

        model = PatchCore(
            backbone='resnet18',
            layers=('layer2', 'layer3'),
            input_size=64,
            coreset_ratio=0.3,
            coreset_projection_dim=32,
            device=device,
        )
        model.fit(train_loader)

        for batch in test_loader:
            heatmaps, scores = model.predict(batch['image'].to(device))
            print(f'  shapes: heatmap {tuple(heatmaps.shape)}, scores {tuple(scores.shape)}')
            print(f'  defect score range: {scores.min().item():.4f} .. {scores.max().item():.4f}')

        print('OK')


if __name__ == '__main__':
    main()
