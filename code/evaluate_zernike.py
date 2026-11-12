"""
Measure the 9-wire Zernike QNN against v3 for speed, rotation and detection.
============================================================================
Each verdict printed here comes from this run's measurements, so a detection
gain is only reported if the numbers show one. Higher F1 alone does not count:
a detector must also mark clearly more of a forged image than of an authentic
one, and rank forged images above authentic ones.

Example:
    cd code
    python evaluate_zernike.py --dataset-dir ../data/comofod \
        --checkpoint ../models/zernike_9w.npz

Without --checkpoint it compares v3 with the plain 9-wire swap test only. Scenes
come from split.json next to the checkpoint (held-out scenes), or --split.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pennylane as qml
from PIL import Image

from cmfd_swaptest_pipeline import (SwapTestSiameseQNN, extract_all_patches,
                                    generate_all_pairs, encode_offset)
from cmfd_zernike_swaptest import ZernikeSwapQNN, image_states, zernike_state

SIZE, PATCH = 128, 8
TAUS = np.unique(np.concatenate([[0.0], 1 - np.logspace(0, -6, 240)[1:]]))
# The bar for calling detection improved. A project choice, not a standard.
MIN_AUC = 0.70
MIN_AREA_GAP = 0.10


def load(path):
    gray = Image.open(path).convert('L').resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(gray, dtype=np.float32) / 255.0


def v3_scorer(checkpoint):
    """All v3 pair scores from one state per patch, equal to its 15-wire circuit."""
    q = SwapTestSiameseQNN(patch_size=PATCH, n_layers=3, n_offset_qubits=2, img_size=SIZE, seed=1)
    q.load(checkpoint)
    w, ow = np.asarray(q.weights, dtype=float), np.asarray(q.offset_weights, dtype=float)

    @qml.qnode(qml.device('default.qubit', wires=6), diff_method=None)
    def register(s):
        qml.AmplitudeEmbedding(s, wires=range(6), normalize=True, pad_with=0.0)
        q._reupload_block(w, list(range(6)), s)
        return qml.state()

    def score(image):
        coords, states = extract_all_patches(image, PATCH, PATCH)
        c = np.array(coords)
        p0 = np.sin((ow[0, 0] + np.pi * np.abs(c[:, None, 0] - c[None, :, 0]) / SIZE) / 2) ** 2
        p1 = np.sin((ow[0, 1] + np.pi * np.abs(c[:, None, 1] - c[None, :, 1]) / SIZE) / 2) ** 2
        gain = ((1 - p0) * (1 - p1) + p0 * (1 - p1) * np.cos(ow[1, 0])
                + (1 - p0) * p1 * np.cos(ow[1, 1]) + p0 * p1 * np.cos(ow[1, 0] + ow[1, 1]))
        psi = np.asarray(register(states))
        return c, np.clip(np.abs(psi.conj() @ psi.T) ** 2 * gain, 0.0, 1.0)
    return q, score


def plain_scorer(image):
    coords, states = image_states(image)
    return np.array(coords), np.abs(states @ states.T) ** 2


def trained_scorer(q):
    def score(image):
        coords, states = image_states(image)
        return np.array(coords), q.all_pair_scores(states)
    return score


def curves(coords, scores, gt=None):
    """F1, marked area and matched-pair count at every threshold in TAUS."""
    np.fill_diagonal(scores, -1)
    best = scores.max(1)
    pix = np.full((SIZE, SIZE), -1.0)
    for i in np.argsort(best):
        y, x = coords[i]
        pix[y:y + PATCH, x:x + PATCH] = best[i]
    srt = np.argsort(pix.ravel())
    vals = pix.ravel()[srt]
    idx = np.searchsorted(vals, TAUS, side='left')
    pred = vals.size - idx
    pairs = np.sort(scores[np.triu_indices(len(coords), 1)])
    count = pairs.size - np.searchsorted(pairs, TAUS, side='left')
    if gt is None:
        return None, pred / vals.size, count
    cum = np.concatenate([[0], np.cumsum(gt.ravel()[srt].astype(np.int64))])
    tp = cum[-1] - cum[idx]
    return 2 * tp / np.maximum(pred + gt.sum(), 1), pred / vals.size, count


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def evaluate(score, data_dir, ids):
    F, AF, AO, CF, CO = [], [], [], [], []
    for s in ids:
        mask = Image.open(data_dir / f'{s}_B.png').convert('L').resize((SIZE, SIZE), Image.NEAREST)
        f, af, cf = curves(*score(load(data_dir / f'{s}_F.png')), np.asarray(mask) > 128)
        _, ao, co = curves(*score(load(data_dir / f'{s}_O.png')))
        F.append(f); AF.append(af); AO.append(ao); CF.append(cf); CO.append(co)
    F, AF, AO, CF, CO = map(np.array, (F, AF, AO, CF, CO))
    # Fixed threshold for each scene chosen on the other scenes
    picks = ((F.sum(0)[None, :] - F) / max(1, len(F) - 1)).argmax(1)
    t = F.mean(0).argmax()
    return dict(best_f1=float(F.max(1).mean()), fixed_f1=float(F[np.arange(len(F)), picks].mean()),
                tau=float(TAUS[t]), area_forged=float(AF[:, t].mean()), area_auth=float(AO[:, t].mean()),
                auc=auc(CF[:, t], CO[:, t]))


def time_real_circuit(name, fn, n_pairs):
    fn()
    t = time.perf_counter()
    fn()
    ms = (time.perf_counter() - t) / 256 * 1e3
    print(f"  {name:32s} {ms:7.2f} ms/pair  {ms * n_pairs / 6e4:6.2f} min/image")
    return ms * n_pairs / 6e4


def main():
    ap = argparse.ArgumentParser(description="Zernike 9-wire QNN vs v3: speed, rotation, detection")
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--checkpoint', default=None, help='Trained Zernike model (.npz)')
    ap.add_argument('--v3-checkpoint', default='../checkpoints_pure/swaptest_qnn_epoch10.npz')
    ap.add_argument('--split', default=None, help='split.json (default: next to --checkpoint)')
    ap.add_argument('--n-layers', type=int, default=3)
    ap.add_argument('--skip-timing', action='store_true')
    ap.add_argument('--out', default=None, help='Also write the results to this JSON file')
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    data_dir = root / 'CoMoFoD_small_v2' if (root / 'CoMoFoD_small_v2').exists() else root
    split = args.split or (str(Path(args.checkpoint).parent / 'split.json') if args.checkpoint
                           else '../checkpoints_pure/split.json')
    ids = sorted(json.load(open(split))['test_ids'], key=int)
    ids = [f'{int(s):03d}' for s in ids]
    print(f"[INFO] {len(ids)} held-out scenes from {split}, forged and authentic")

    v3_model, v3_score = v3_scorer(args.v3_checkpoint)
    zq = None
    if args.checkpoint:
        zq = ZernikeSwapQNN(n_layers=args.n_layers, img_size=SIZE, seed=1)
        zq.load(args.checkpoint)
    results = {}

    # ---- speed: real circuits, one image ----
    v3_min = zer_min = None
    if not args.skip_timing:
        print("\nSPEED (real circuits, scaled from 256 pairs of one image)")
        image = load(data_dir / f'{ids[0]}_F.png')
        coords, st = extract_all_patches(image, PATCH, PATCH)
        pairs = generate_all_pairs(coords, min_spatial_dist=PATCH)
        a, b = [i for i, _ in pairs[:256]], [j for _, j in pairs[:256]]
        offs = np.array([encode_offset(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1], SIZE)
                         for i, j in pairs[:256]])
        v3_min = time_real_circuit('v3, 15 wires', lambda: v3_model.similarity(st[a], st[b], offs), len(pairs))
        _, zs = image_states(image)
        timing_q = zq or ZernikeSwapQNN(n_layers=args.n_layers, img_size=SIZE, seed=1)
        label = 'Zernike, 9 wires, trained' if zq else 'Zernike, 9 wires, untrained'
        zer_min = time_real_circuit(label, lambda: timing_q.score(zs[a], zs[b]), len(pairs))
        results['speed_min_per_image'] = dict(v3=v3_min, zernike=zer_min)

    # ---- rotation: is the true source the best match among all patches? ----
    # A mean score against the rotated copy would flatter raw pixels, whose
    # all-positive vectors score high against almost anything.
    print("\nROTATION (rotated copy finds its true source, within 2 px, among all patches on a 2-px grid)")
    rng = np.random.default_rng(0)
    rows = {}
    for angle in (15, 45, 90):
        raw_hits, zer_hits, total = 0, 0, 0
        for s in ids[:20]:
            arr = load(data_dir / f'{s}_O.png')
            win = np.lib.stride_tricks.sliding_window_view(arr, (PATCH, PATCH))[::2, ::2]
            h, w = win.shape[:2]
            flat = win.reshape(-1, PATCH, PATCH).astype(np.float64)
            grid = np.stack(np.mgrid[0:h, 0:w], -1).reshape(-1, 2) * 2
            raw_db = flat.reshape(-1, PATCH * PATCH)
            raw_db = raw_db / np.maximum(np.linalg.norm(raw_db, axis=1, keepdims=True), 1e-12)
            zer_db = zernike_state(flat)
            for _ in range(10):
                y, x = rng.integers(6, 54, 2) * 2
                crop = Image.fromarray(np.uint8(arr[y - 12:y + 20, x - 12:x + 20] * 255))
                rot = np.asarray(crop.rotate(angle, Image.BICUBIC), dtype=np.float64)[12:20, 12:20] / 255.0
                raw_q = rot.ravel() / max(np.linalg.norm(rot), 1e-12)
                for db, q, name in ((raw_db, raw_q, 'raw'), (zer_db, zernike_state(rot), 'zer')):
                    found = grid[np.argmax((db @ q) ** 2)]
                    hit = abs(found[0] - y) <= 2 and abs(found[1] - x) <= 2
                    raw_hits += hit if name == 'raw' else 0
                    zer_hits += hit if name == 'zer' else 0
                total += 1
        rows[angle] = dict(raw_pixels=100 * raw_hits / total, zernike=100 * zer_hits / total)
        print(f"  {angle:3d} deg   raw pixels {rows[angle]['raw_pixels']:5.1f}%   Zernike {rows[angle]['zernike']:5.1f}%")
    results['rotation_source_found_percent'] = rows

    # ---- detection ----
    print("\nDETECTION (held-out scenes)")
    methods = {'v3, 15 wires': v3_score, 'Zernike, 9 wires, plain': plain_scorer}
    if zq:
        methods['Zernike, 9 wires, trained'] = trained_scorer(zq)
    print(f"  {'method':28s} {'best F1':>8s} {'fixed F1':>9s} {'marked forged':>14s} {'marked auth':>12s} {'AUC':>6s}")
    for name, score in methods.items():
        r = evaluate(score, data_dir, ids)
        results[name] = r
        print(f"  {name:28s} {r['best_f1']:8.3f} {r['fixed_f1']:9.3f} {r['area_forged'] * 100:13.1f}% "
              f"{r['area_auth'] * 100:11.1f}% {r['auc']:6.3f}")

    # ---- verdicts, computed from the numbers above ----
    print("\nVERDICT")
    if v3_min:
        print(f"  Speed: {'faster' if zer_min < v3_min else 'NOT faster'} than v3 "
              f"({v3_min:.2f} -> {zer_min:.2f} min/image)")
    rotated = all(rows[a]['zernike'] > rows[a]['raw_pixels'] for a in rows)
    print(f"  Rotation: {'better' if rotated else 'NOT better'} than raw pixels at 15, 45 and 90 degrees")
    best = results['Zernike, 9 wires, trained' if zq else 'Zernike, 9 wires, plain']
    gap = best['area_forged'] - best['area_auth']
    if best['auc'] >= MIN_AUC and gap >= MIN_AREA_GAP:
        print(f"  Detection: IMPROVED (AUC {best['auc']:.3f}, marks {gap * 100:.1f} points more on forged images)")
    else:
        print(f"  Detection: NOT improved (AUC {best['auc']:.3f}, needs {MIN_AUC}; marks {gap * 100:.1f} points "
              f"more on forged images, needs {MIN_AREA_GAP * 100:.0f}). Any F1 change here is not better detection.")
    results['detection_improved'] = bool(best['auc'] >= MIN_AUC and gap >= MIN_AREA_GAP)
    if args.out:
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
