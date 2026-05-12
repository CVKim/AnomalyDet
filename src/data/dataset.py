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
    """Generic dataset for an arbitrary folder of images.

    Returns the same dict shape as MVTecDataset so it can drop into
    the same training / inference loop. No GT masks; `mask_path` is
    always empty and `defect_type` defaults to `'good'` (for a train
    folder) or `'test'` (for a test folder).
    """

    def __init__(self, paths: Sequence[str], transform=None,
                 defect_type: str = 'test', label: int = 1, repeat: int = 1):
        self.paths = [Path(p) for p in paths]
        self.transform = transform
        self.defect_type = defect_type
        self.label = int(label)
        self.repeat = max(1, int(repeat))
        # Match MVTecDataset.samples for callers (e.g. train manifest).
        self.samples = [{'image': p, 'mask': None, 'label': self.label,
                         'defect_type': self.defect_type}
                        for p in self.paths]

    @classmethod
    def from_dir(cls, directory, transform=None, recursive: bool = False,
                 defect_type: str = 'test', label: int = 1, repeat: int = 1):
        """Default to non-recursive so a test-dir doesn't accidentally pull
        in a subdir of training images (e.g. an MVTec-style nested layout).

        `directory` may be a single path or a list of paths. When a list,
        all matching images across all directories are concatenated.
        Useful for pulling multiple normal-data folders into one train set.
        """
        if isinstance(directory, (str, Path)):
            dirs = [directory]
        else:
            dirs = list(directory)
        paths = []
        for d in dirs:
            d = Path(d)
            glob = d.rglob('*') if recursive else d.glob('*')
            for p in glob:
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    paths.append(p)
        paths = sorted(paths)
        return cls([str(p) for p in paths], transform=transform,
                   defect_type=defect_type, label=label, repeat=repeat)

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        p = self.paths[idx % len(self.paths)]
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {
            'image': img,
            'label': self.label,
            'defect_type': self.defect_type,
            'image_path': str(p),
            'mask_path': '',
        }
