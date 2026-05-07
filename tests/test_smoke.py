"""Tiny smoke tests that exercise the full pipeline on synthetic images.
Run with: python -m pytest tests/ -q
"""
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import MVTecDataset, FolderDataset
from src.data.transforms import build_image_transform
from src.models.patchcore import PatchCore
from src.utils.postprocess import clean_mask, mask_to_labelme_json


def _make_synthetic_mvtec(root: Path, n_train: int = 4, n_test: int = 2,
                          size: int = 64) -> None:
    rng = np.random.default_rng(0)
    cat = root / 'fake'
    (cat / 'train' / 'good').mkdir(parents=True, exist_ok=True)
    (cat / 'test' / 'good').mkdir(parents=True, exist_ok=True)
    (cat / 'test' / 'defect').mkdir(parents=True, exist_ok=True)

    for i in range(n_train):
        arr = rng.integers(80, 160, (size, size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cat / 'train' / 'good' / f'{i:03d}.png')
    for i in range(n_test):
        arr = rng.integers(80, 160, (size, size, 3), dtype=np.uint8)
        Image.fromarray(arr).save(cat / 'test' / 'good' / f'{i:03d}.png')
    for i in range(n_test):
        arr = rng.integers(80, 160, (size, size, 3), dtype=np.uint8)
        arr[10:25, 10:25] = 230
        Image.fromarray(arr).save(cat / 'test' / 'defect' / f'{i:03d}.png')


def test_pipeline(tmp_path):
    _make_synthetic_mvtec(tmp_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = build_image_transform(64)

    train_ds = MVTecDataset(tmp_path, 'fake', split='train', transform=transform)
    assert len(train_ds) == 4
    test_ds = MVTecDataset(tmp_path, 'fake', split='test', transform=transform)
    assert len(test_ds) == 4

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    model = PatchCore(backbone='resnet18', layers=('layer2', 'layer3'),
                      input_size=64, coreset_ratio=0.2,
                      coreset_projection_dim=32, device=device)
    model.fit(loader)
    assert model.memory_bank is not None
    assert model.memory_bank.shape[0] >= 1

    for batch in test_loader:
        heatmaps, scores = model.predict(batch['image'].to(device))
        assert heatmaps.shape == (batch['image'].shape[0], 64, 64)
        assert scores.shape == (batch['image'].shape[0],)

    save_path = tmp_path / 'mb.pt'
    model.save(str(save_path))
    model2 = PatchCore(backbone='resnet18', layers=('layer2', 'layer3'),
                       input_size=64, coreset_ratio=0.2,
                       coreset_projection_dim=32, device=device)
    model2.load(str(save_path))
    assert model2.memory_bank.shape == model.memory_bank.shape


def test_postprocess():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 255
    mask[60:65, 60:65] = 255  # tiny region, should be filtered
    cleaned = clean_mask(mask, kernel_size=3, min_area=100)
    assert cleaned[30, 30] == 255
    assert cleaned[62, 62] == 0

    js = mask_to_labelme_json(cleaned, 'foo.png', (100, 100), image_score=0.9)
    assert js['imageWidth'] == 100
    assert len(js['shapes']) == 1
    assert js['shapes'][0]['shape_type'] == 'polygon'
