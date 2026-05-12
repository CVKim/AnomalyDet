"""Print one markdown table comparing every Foosung run we have on disk.

Reads:
    outputs/foosung_*/gt_eval.json   (bare anomaly model vs GT)
    outputs/foosung_*_level3*/level3_summary.json   (refined)

Picks rect-GT and poly-GT F1/IoU for each, prints a comparison.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runs = [
    ('PatchCore D (10 train, baseline)', 'foosung_D_thr30', None),
    ('PatchCore D v2 (12 train)',       'foosung_D_v2_train12_noaug', None),
    ('PatchCore+ (pos + reweight)',     'foosung_pcplus', None),
    ('PaDiM (proj=30, eps=1.0)',        'foosung_padim_stable', None),
    ('PatchCore D + Level 3 (10 train)', None, 'foosung_level3_e200'),
    ('PatchCore D + Level 3 (12 train)', None, 'foosung_level3_v2'),
    ('PatchCore+ + Level 3',             None, 'foosung_pcplus_level3'),
    ('PaDiM + Level 3',                  None, 'foosung_padim_level3'),
]

print('| Setup | rect F1 | rect IoU | poly F1 | poly IoU |')
print('|---|---|---|---|---|')
for label, bare_dir, l3_dir in runs:
    rect_f1 = rect_iou = poly_f1 = poly_iou = '—'
    if bare_dir is not None:
        p = ROOT / 'outputs' / bare_dir / 'gt_eval.json'
        if p.exists():
            d = json.loads(p.read_text())
            best = d['best']
            rect_f1 = f"{best['f1']:.3f}"
            rect_iou = f"{best['iou']:.3f}"
    if l3_dir is not None:
        for gt_kind, key_f1, key_iou in [('rect', '_f1_rect', '_iou_rect'),
                                          ('poly', '_f1_poly', '_iou_poly')]:
            pass  # not used below
        # Level 3 summary has its own best (vs whatever GT it trained against)
        p = ROOT / 'outputs' / l3_dir / 'level3_summary.json'
        if p.exists():
            d = json.loads(p.read_text())
            best = d['best_thr_meta']
            # we ran rect-trained models; report bare numbers from the
            # _eval_prob_vs_gt runs we executed manually
            rect_f1 = f"{best['f1']:.3f}"
            rect_iou = f"{best['iou']:.3f}"
    print(f'| {label:38s} | {rect_f1:>7s} | {rect_iou:>8s} | {poly_f1:>7s} | {poly_iou:>8s} |')
