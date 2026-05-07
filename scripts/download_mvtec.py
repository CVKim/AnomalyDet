"""Download a single MVTec AD category and extract it under data/mvtec/.

Usage:
    python scripts/download_mvtec.py --category bottle
"""
import argparse
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

import requests
from tqdm import tqdm


CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper',
]

BASE_URL = (
    'https://www.mydrive.ch/shares/38536/'
    '3830184030e49fe74747669442f0f282/download/'
    '420938113-1629952094/{category}.tar.xz'
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_tar_xz(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:xz') as tar:
        tar.extractall(out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--category', type=str, default='bottle',
                   choices=CATEGORIES)
    p.add_argument('--out-root', type=str, default='data/mvtec')
    p.add_argument('--keep-archive', action='store_true')
    args = p.parse_args()

    out_root = Path(args.out_root)
    target_dir = out_root / args.category
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f'{target_dir} already exists; skipping download.')
        return 0

    url = BASE_URL.format(category=args.category)
    archive = out_root / f'{args.category}.tar.xz'
    print(f'downloading {url}')
    try:
        download(url, archive)
    except Exception as e:
        print(f'\ndownload failed: {e}\n', file=sys.stderr)
        print(
            'manual download:\n'
            '  https://www.mvtec.com/company/research/datasets/mvtec-ad/\n'
            f'place {args.category}.tar.xz under {out_root}/ and re-run extract:\n'
            f'  python -c "import tarfile; '
            f"tarfile.open(r'{archive}').extractall(r'{out_root}')\"",
            file=sys.stderr,
        )
        return 1

    print(f'extracting {archive}')
    extract_tar_xz(archive, out_root)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
    print(f'done: {target_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
