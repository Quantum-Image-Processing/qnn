"""Is the dashboard's Zernike "confidence" percentage related to being correct?
Read-only: uses the saved model, split and calibration on the held-out test scenes.
Run:  python scripts/paper_pure/confidence_check.py .   (from the repo root)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / 'code'))
from cmfd_zernike_swaptest import ZernikeSwapQNN, image_states  # noqa: E402
from evaluate_zernike import load, SIZE  # noqa: E402

models = ROOT / 'models'
cal = json.load(open(models / 'zernike_9w_calibration.json'))
tau, m = cal['sim_threshold'], cal['decision_pairs']
ids = [f'{int(s):03d}' for s in json.load(open(models / 'split.json'))['test_ids']]
root = ROOT / 'data' / 'comofod'
data = root / 'CoMoFoD_small_v2' if (root / 'CoMoFoD_small_v2').exists() else root

qnn = ZernikeSwapQNN(n_layers=3, img_size=SIZE, seed=1)
qnn.load(models / 'zernike_9w.npz')

rows = []
for sid in ids:
    for kind in 'FO':
        _, states = image_states(load(data / f'{sid}_{kind}.png'))
        s = qnn.all_pair_scores(states)
        n = int((s[np.triu_indices(len(s), 1)] >= tau).sum())
        forged = n >= m
        raw = n / (2 * m)
        shown = min(raw, 1.0) if forged else 1 - min(raw, 0.5)   # same as app.py
        rows.append((sid, kind, n, forged, shown, forged == (kind == 'F')))

n = np.array([r[2] for r in rows])
kind = np.array([r[1] for r in rows])
shown = np.array([r[4] for r in rows])
right = np.array([r[5] for r in rows])

print(f'{len(rows)} images ({len(ids)} scenes), threshold {tau}, {m} pairs')
print(f'accuracy {right.mean() * 100:.1f}%')
print(f'pair count: forged median {np.median(n[kind == "F"]):.0f}, authentic median {np.median(n[kind == "O"]):.0f}')
paired = n[kind == 'F'] - n[kind == 'O']
print(f'forged minus its own authentic original: median {np.median(paired):.0f}, '
      f'forged higher in {(paired > 0).sum()}/{len(paired)}')
u = mannwhitneyu(n[kind == 'F'], n[kind == 'O']).statistic / (len(ids) ** 2)
print(f'AUC of the pair count, forged vs authentic: {u:.3f}')
print('\nshown confidence -> how often the verdict was right')
for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0001)):
    k = (shown >= lo) & (shown < hi)
    if k.any():
        print(f'  {lo * 100:3.0f}-{min(hi, 1) * 100:3.0f}%: {k.sum():3d} images, right {right[k].mean() * 100:5.1f}%')
print(f'Mean shown confidence {shown.mean() * 100:.1f}% vs accuracy {right.mean() * 100:.1f}%')
