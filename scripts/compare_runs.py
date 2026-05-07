"""Aggregate sweep summary.csv files from multiple (backbone, category)
runs into a single side-by-side report.

Usage:
    python scripts/compare_runs.py outputs/bottle/sweep/summary.csv:wrn50_bottle `
                                   outputs/bottle_dinov2/sweep/summary.csv:dinov2_bottle `
                                   outputs/hazelnut/sweep/summary.csv:wrn50_hazelnut `
                                   outputs/hazelnut_dinov2/sweep/summary.csv:dinov2_hazelnut `
                                   --out outputs/comparison.md
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+',
                   help='Each item: <path-to-summary.csv>:<run_label>')
    p.add_argument('--out', default='outputs/comparison.md')
    return p.parse_args()


def load(csv_path):
    with open(csv_path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()

    # rows[(experiment, run_label)][category] = {n, gated, mean, max}
    rows = defaultdict(dict)
    experiments = []
    runs = []
    for spec in args.runs:
        path, label = spec.split(':', 1)
        runs.append(label)
        for r in load(path):
            exp = r['experiment']
            cat = r['category']
            if exp not in experiments:
                experiments.append(exp)
            rows[(exp, label)][cat] = r

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('# Threshold sweep comparison\n')
    lines.append(f'Runs: {", ".join(runs)}\n')

    # For each experiment, emit a table comparing runs.
    for exp in experiments:
        lines.append(f'\n## {exp}\n')
        all_cats = set()
        for run in runs:
            all_cats |= set(rows[(exp, run)].keys())
        cats = sorted(all_cats)

        # header: category, then for each run: gated/n  mask% mean  mask% max
        header = ['category'] + [
            f'{r}: gated/n | mean | max' for r in runs
        ]
        lines.append('| ' + ' | '.join(header) + ' |')
        lines.append('|' + '|'.join(['---'] * len(header)) + '|')

        for cat in cats:
            row = [cat]
            for run in runs:
                d = rows[(exp, run)].get(cat)
                if d is None:
                    row.append('—')
                    continue
                cell = (f"{d['gated']}/{d['n']} | "
                        f"{float(d['mask_pct_mean']):.2f}% | "
                        f"{float(d['mask_pct_max']):.2f}%")
                row.append(cell)
            lines.append('| ' + ' | '.join(row) + ' |')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
