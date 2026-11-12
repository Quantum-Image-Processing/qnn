"""Pure-quantum shift search, then v3 scoring at that shift.

Step 1: amplitude-encode the whole image (14 qubits at 128px). A Hadamard test
of |I> against its cyclically shifted self (1 ancilla, 15 wires) gives
R(d) = <I|T_d|I>. Simulated exactly here with an FFT, which equals the ideal
expectation value (no shot noise).
Step 2: score only the patch pairs (x, x+d) with the existing v3 weights.
Classical steps: resize, mean removal, normalise, argmax of readouts, threshold.
"""
import sys
import numpy as np
import pennylane as qml
from PIL import Image
sys.path.insert(0, 'code')
from cmfd_swaptest_pipeline import SwapTestSiameseQNN, prepare_patch_state

D = 'data/comofod/CoMoFoD_small_v2/'
P, S = 8, 128
MIN_SHIFT = 16
TAU = np.concatenate([[0.0], 1 - np.logspace(0, -6, 240)[1:]])

q = SwapTestSiameseQNN(patch_size=P, n_layers=3, n_offset_qubits=0, img_size=S, seed=1)
q.load('checkpoints_pure/swaptest_qnn_epoch10.npz')
W = np.asarray(q.weights, dtype=float)
dev = qml.device('default.qubit', wires=6)


@qml.qnode(dev, diff_method=None)
def single(s):
    qml.AmplitudeEmbedding(s, wires=range(6), normalize=True, pad_with=0.0)
    q._reupload_block(W, list(range(6)), s)
    return qml.state()


def load(path):
    return np.asarray(Image.open(path).convert('L').resize((S, S), Image.LANCZOS), dtype=np.float64) / 255.


def shift_scores(img, top=1):
    """Peak R(d) over admissible shifts and its z-score against all other shifts."""
    a = img - img.mean()
    R = np.fft.ifft2(np.abs(np.fft.fft2(a)) ** 2).real / max((a ** 2).sum(), 1e-12)
    dy = np.minimum(np.arange(S), S - np.arange(S))[:, None]
    dx = np.minimum(np.arange(S), S - np.arange(S))[None, :]
    ok = dy ** 2 + dx ** 2 >= MIN_SHIFT ** 2
    v = R[ok]
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    Rm = np.where(ok, R, -np.inf)
    i = np.unravel_index(Rm.argmax(), R.shape)
    d = tuple(int(k if k < S / 2 else k - S) for k in i)
    return d, float(Rm[i]), float((Rm[i] - med) / mad)


def pixel_map(img, d, kind, min_std=0.02):
    """Score every patch pair (x, x+d); paint both patches with the score."""
    dy, dx = d
    A, B, ca, cb = [], [], [], []
    for y in range(max(0, -dy), min(S - P, S - P - dy) + 1):
        for x in range(max(0, -dx), min(S - P, S - P - dx) + 1):
            pa, pb = img[y:y + P, x:x + P], img[y + dy:y + dy + P, x + dx:x + dx + P]
            if pa.std() < min_std or pb.std() < min_std:
                continue
            A.append(prepare_patch_state(pa - pa.mean()))
            B.append(prepare_patch_state(pb - pb.mean()))
            ca.append((y, x)); cb.append((y + dy, x + dx))
    pix = np.zeros((S, S), dtype=np.float32)
    if not A:
        return pix, 0
    A, B = np.array(A), np.array(B)
    if kind == 'qnn':
        pa = np.concatenate([np.asarray(single(A[i:i + 2048])) for i in range(0, len(A), 2048)])
        pb = np.concatenate([np.asarray(single(B[i:i + 2048])) for i in range(0, len(B), 2048)])
        s = np.abs((pa.conj() * pb).sum(1)) ** 2
    else:
        s = (A * B).sum(1) ** 2
    for i in np.argsort(s):
        y, x = ca[i]; pix[y:y + P, x:x + P] = s[i]
        y, x = cb[i]; pix[y:y + P, x:x + P] = np.maximum(pix[y:y + P, x:x + P], s[i])
    return pix, len(A)


def f1_curve(pix, gt):
    return np.array([2 * ((pix >= t) & gt).sum() / max(1, (pix >= t).sum() + gt.sum()) for t in TAU])


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


# Image-level: all 200 scenes, raw forged vs authentic
ids = [f'{i:03d}' for i in range(1, 201)]
rows = {}
for s in ids:
    f, o = load(f'{D}{s}_F.png'), load(f'{D}{s}_O.png')
    rows[s] = (shift_scores(f), shift_scores(o))
groups = [('translation 001-040', 1, 40), ('rotation 041-080', 41, 80), ('scaling 081-120', 81, 120),
          ('distortion 121-160', 121, 160), ('combination 161-200', 161, 200), ('all 200', 1, 200)]
print('IMAGE LEVEL: can the peak separate forged from its authentic original? (0.5 = chance)')
for name, a, b in groups:
    sub = ids[a - 1:b]
    print(f'  {name:24s} AUC peak {auc([rows[s][0][1] for s in sub],[rows[s][1][1] for s in sub]):.3f} '
          f'| AUC z-score {auc([rows[s][0][2] for s in sub],[rows[s][1][2] for s in sub]):.3f}', flush=True)

# Localisation at the single best shift
for label, sub in [('scenes 001-008', ids[:8]), ('translation 001-040', ids[:40]), ('rotation 041-060', ids[40:60])]:
    for kind in ('qnn', 'cos'):
        F, ff, fa, n = [], [], [], []
        maps = []
        for s in sub:
            gt = np.asarray(Image.open(f'{D}{s}_B.png').convert('L').resize((S, S), Image.NEAREST)) > 128
            pf, c = pixel_map(load(f'{D}{s}_F.png'), rows[s][0][0], kind)
            po, _ = pixel_map(load(f'{D}{s}_O.png'), rows[s][1][0], kind)
            F.append(f1_curve(pf, gt)); maps.append((pf, po)); n.append(c)
        F = np.array(F)
        oracle = F.max(1).mean()
        idx = np.arange(len(F))
        loo = np.mean([F[i, F[idx != i].mean(0).argmax()] for i in idx])
        tau = TAU[F.mean(0).argmax()]
        ff = np.mean([(a >= tau).mean() for a, _ in maps]); fa = np.mean([(b >= tau).mean() for _, b in maps])
        print(f'{label:20s} {kind:3s} | oracle F1 {oracle:.3f} | fixed-thr(LOO) {loo:.3f} | tau {tau:.4f} '
              f'| flagged forged {ff*100:5.1f}% authentic {fa*100:5.1f}% | pairs/img {np.mean(n):.0f}', flush=True)
print('done')
