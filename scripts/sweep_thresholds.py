"""Run inference under several threshold strategies and write a side-by-side
comparison so you can pick a tightness that matches inspection appetite.

Usage:
    python scripts/sweep_thresholds.py `
        --data-root "E:\\dataset\\mvtec_anomaly_detection_" `
        --category bottle `
        --memory-bank outputs/bottle/memory_bank.pt
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# Stricter than the current default (which the user said overshoots).
# Each entry: (label, list of extra CLI args appended to the inference call).
EXPERIMENTS = [
    ('adaptive_baseline',
     ['--threshold-mode', 'adaptive',
      '--image-gate-factor', '1.3',
      '--severity-fraction', '0.50']),
    ('adaptive_strict_gate',
     ['--threshold-mode', 'adaptive',
      '--image-gate-factor', '1.5',
      '--severity-fraction', '0.50']),
    ('adaptive_tight_severity',
     ['--threshold-mode', 'adaptive',
      '--image-gate-factor', '1.3',
      '--severity-fraction', '0.65']),
    ('adaptive_strict_both',
     ['--threshold-mode', 'adaptive',
      '--image-gate-factor', '1.5',
      '--severity-fraction', '0.70']),
    ('fixed_15', ['--threshold', '15.0']),
    ('fixed_18', ['--threshold', '18.0']),
    ('fixed_22', ['--threshold', '22.0']),
    ('train_p999', ['--threshold-mode', 'train_p999']),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    p.add_argument('--data-root', required=True)
    p.add_argument('--category', default='bottle')
    p.add_argument('--memory-bank', required=True)
    p.add_argument('--output-root', default=None,
                   help='Default: outputs/<category>/sweep')
    p.add_argument('--summary-csv', default=None,
                   help='Default: <output-root>/summary.csv')
    return p.parse_args()


def run_inference(args, exp_name, extra_args, out_dir):
    cmd = [
        sys.executable, '-m', 'src.inference',
        '--config', args.config,
        '--data-root', args.data_root,
        '--category', args.category,
        '--memory-bank', args.memory_bank,
        '--output', str(out_dir),
        *extra_args,
    ]
    print(f'\n>>> [{exp_name}] {" ".join(extra_args)}')
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError(f'inference failed for {exp_name}')
    print(res.stdout.strip().splitlines()[-3:])


def stats_for_dir(out_dir):
    """Walk an experiment output directory and bucket mask coverage by category."""
    buckets = {}
    for jp in sorted(out_dir.glob('*.json')):
        with open(jp) as f:
            j = json.load(f)
        d = j.get('defectType', 'unknown') or 'unknown'
        m = cv2.imread(str(jp).replace('.json', '_mask.png'), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        gated = j.get('threshold') is None
        pct = 100.0 * (m > 0).sum() / m.size
        b = buckets.setdefault(d, {'n': 0, 'gated': 0, 'mask_pct': []})
        b['n'] += 1
        if gated:
            b['gated'] += 1
        b['mask_pct'].append(pct)
    return buckets


def main():
    args = parse_args()
    out_root = Path(args.output_root or f'outputs/{args.category}/sweep')
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for exp_name, extra in EXPERIMENTS:
        out_dir = out_root / exp_name
        run_inference(args, exp_name, extra, out_dir)
        buckets = stats_for_dir(out_dir)
        for cat in sorted(buckets):
            b = buckets[cat]
            mean = float(np.mean(b['mask_pct'])) if b['mask_pct'] else 0.0
            mx = float(np.max(b['mask_pct'])) if b['mask_pct'] else 0.0
            summary_rows.append({
                'experiment': exp_name,
                'category': cat,
                'n': b['n'],
                'gated': b['gated'],
                'mask_pct_mean': round(mean, 2),
                'mask_pct_max': round(mx, 2),
            })

    csv_path = Path(args.summary_csv or out_root / 'summary.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f'\nsummary written to {csv_path}')

    print('\n=== sweep summary ===')
    print(f"{'experiment':25s} {'category':18s} {'n':>3s} {'gated':>5s} "
          f"{'mask% mean':>11s} {'mask% max':>10s}")
    for r in summary_rows:
        print(f"{r['experiment']:25s} {r['category']:18s} {r['n']:>3d} "
              f"{r['gated']:>5d} {r['mask_pct_mean']:>10.2f}% "
              f"{r['mask_pct_max']:>9.2f}%")


if __name__ == '__main__':
    main()
