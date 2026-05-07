from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class MVTecDataset(Dataset):
    """MVTec AD style folder layout.

    root/<category>/train/good/*.png
    root/<category>/test/<defect_type>/*.png
    root/<category>/ground_truth/<defect_type>/*_mask.png
    """

    def __init__(self, root: str, category: str, split: str = 'train',
                 transform=None, mask_transform=None, repeat: int = 1):
        self.root = Path(root) / category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        # repeat>1 returns the same underlying image multiple times under
        # different random transforms (only meaningful when transform is
        # stochastic, e.g. rotation augmentation for memory-bank build).
        self.repeat = max(1, int(repeat))
        self.samples: List[dict] = []
        self._collect()

    def _collect(self):
        if self.split == 'train':
            train_dir = self.root / 'train' / 'good'
            for p in sorted(train_dir.glob('*')):
                if p.suffix.lower() in IMG_EXTS:
                    self.samples.append({
                        'image': p, 'mask': None, 'label': 0, 'defect_type': 'good',
                    })
            return

        test_root = self.root / 'test'
        gt_root = self.root / 'ground_truth'
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            for p in sorted(defect_dir.glob('*')):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                label = 0 if defect == 'good' else 1
                mask_path = None
                if label == 1:
                    candidate = gt_root / defect / f'{p.stem}_mask{p.suffix}'
                    if candidate.exists():
                        mask_path = candidate
                self.samples.append({
                    'image': p, 'mask': mask_path, 'label': label, 'defect_type': defect,
                })

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, idx):
        s = self.samples[idx % len(self.samples)]
        img = Image.open(s['image']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        item = {
            'image': img,
            'label': s['label'],
            'defect_type': s['defect_type'],
            'image_path': str(s['image']),
        }
        item['mask_path'] = str(s['mask']) if s['mask'] is not None else ''
        if s['mask'] is not None and self.mask_transform is not None:
            mask = Image.open(s['mask']).convert('L')
            item['mask'] = self.mask_transform(mask)
        return item


class FolderDataset(Dataset):
    """Generic dataset for inference on an arbitrary folder of images."""

    def __init__(self, paths: Sequence[str], transform=None):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    @classmethod
    def from_dir(cls, directory: str, transform=None, recursive: bool = True):
        d = Path(directory)
        glob = d.rglob('*') if recursive else d.glob('*')
        paths = sorted(p for p in glob if p.is_file() and p.suffix.lower() in IMG_EXTS)
        return cls([str(p) for p in paths], transform=transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'image_path': str(p)}
