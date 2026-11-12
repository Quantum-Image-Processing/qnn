"""
Choose the 9-wire Zernike model's decision settings on validation scenes.
=========================================================================
v3's sim_threshold 0.90 and 8 pairs do not carry over, because this circuit's
scores are on a different scale. This sweeps both settings on the validation
scenes in split.json, forged and authentic, and saves the pair with the best
balanced accuracy for the dashboard. Test scenes are only scored afterwards, at
the chosen settings, to report the result honestly.

Example:
    cd code
    python calibrate_zernike.py --dataset-dir ../data/comofod \
        --checkpoint ../models/zernike_9w.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cmfd_zernike_swaptest import ZernikeSwapQNN, image_states
from evaluate_zernike import load, SIZE, PATCH

TAUS = np.round(np.concatenate([np.arange(0.50, 0.99, 0.01), np.arange(0.990, 0.9999, 0.001)]), 4)
COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def pair_counts(qnn, image):
    """Sorted upper-triangle scores and the per-patch best score of one image."""
    coords, states = image_states(image)
    s = qnn.all_pair_scores(states)
    np.fill_diagonal(s, -1)
    return np.array(coords), s


def sweep(qnn, data_dir, ids):
    """counts[kind][scene, tau] for kind F (forged) and O (authentic)."""
    out = {}
    for kind in 'FO':
        rows = []
        for sid in ids:
            _, s = pair_counts(qnn, load(data_dir / f'{sid}_{kind}.png'))
            v = np.sort(s[np.triu_indices(len(s), 1)])
            rows.append(v.size - np.searchsorted(v, TAUS, side='left'))
        out[kind] = np.array(rows)
    return out


def rates(counts, t, m):
    tpr = float((counts['F'][:, t] >= m).mean())
    fpr = float((counts['O'][:, t] >= m).mean())
    return tpr, fpr, (tpr + 1 - fpr) / 2


def pixel_f1(qnn, data_dir, ids, tau):
    f1 = []
    for sid in ids:
        coords, s = pair_counts(qnn, load(data_dir / f'{sid}_F.png'))
        gt = np.asarray(Image.open(data_dir / f'{sid}_B.png').convert('L').resize((SIZE, SIZE), Image.NEAREST)) > 128
        pred = np.zeros((SIZE, SIZE), dtype=bool)
        for i in np.where(s.max(1) >= tau)[0]:
            y, x = coords[i]
            pred[y:y + PATCH, x:x + PATCH] = True
        f1.append(2 * (pred & gt).sum() / max(1, pred.sum() + gt.sum()))
    return float(np.mean(f1))


def main():
    ap = argparse.ArgumentParser(description="Calibrate the 9-wire Zernike model on validation scenes")
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n-layers', type=int, default=3)
    ap.add_argument('--out', default=None, help='Default: <checkpoint>_calibration.json')
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    root = Path(args.dataset_dir)
    data_dir = root / 'CoMoFoD_small_v2' if (root / 'CoMoFoD_small_v2').exists() else root
    split = json.load(open(ckpt.parent / 'split.json'))
    val_ids = [f'{int(s):03d}' for s in split['val_ids']]
    test_ids = [f'{int(s):03d}' for s in split['test_ids']]

    qnn = ZernikeSwapQNN(n_layers=args.n_layers, img_size=SIZE, seed=1)
    qnn.load(ckpt)

    val = sweep(qnn, data_dir, val_ids)
    best = None
    for t in range(len(TAUS)):           # ascending, so ties keep the higher threshold
        for m in COUNTS:
            r = rates(val, t, m)
            if best is None or r[2] >= best[0]:
                best = (r[2], t, m, r)
    _, t, m, r = best
    tau = float(TAUS[t])
    print(f"[VAL] {len(val_ids)} scenes: threshold {tau}, {m} pairs -> forged called forged {r[0] * 100:.1f}%, "
          f"authentic called forged {r[1] * 100:.1f}%, balanced accuracy {r[2] * 100:.1f}%")

    test = sweep(qnn, data_dir, test_ids)
    tr = rates(test, t, m)
    f1 = pixel_f1(qnn, data_dir, test_ids, tau)
    print(f"[TEST] {len(test_ids)} held-out scenes at those settings: forged called forged {tr[0] * 100:.1f}%, "
          f"authentic called forged {tr[1] * 100:.1f}%, accuracy {tr[2] * 100:.1f}%, pixel F1 {f1:.3f}")

    out = Path(args.out) if args.out else ckpt.with_name(ckpt.stem + '_calibration.json')
    json.dump(dict(sim_threshold=tau, decision_pairs=m,
                   validation=dict(tpr=r[0], fpr=r[1], balanced_accuracy=r[2]),
                   test=dict(tpr=tr[0], fpr=tr[1], accuracy=tr[2], pixel_f1=f1)),
              open(out, 'w'), indent=1)
    print(f"[DONE] saved -> {out}")


if __name__ == '__main__':
    main()
