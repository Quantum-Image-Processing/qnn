"""
Pure-QNN benchmark on all 200 raw CoMoFoD scenes, forged and authentic.
Run from the repository root:  python scripts/paper_pure/benchmark.py [out.json] [qft]

Every score is the exact expectation value of a swap test, computed from one
single-register state per patch (v3's output factorizes; see the paper):
    v3 15 wires : |<psi_a|psi_b>|^2 * g(offset), with the trained weights
    plain swap  : |<a|b>|^2, amplitude encoding and swap test, nothing trained
    Zernike     : plain swap test on 12 Zernike magnitudes on 4 qubits (9 wires)
Reads the v3 model and checkpoint only; changes neither.
"""
import json
import math
import sys
import numpy as np
import pennylane as qml
from PIL import Image
sys.path.insert(0, 'code')
from cmfd_swaptest_pipeline import (SwapTestSiameseQNN, prepare_patch_state,
                                    encode_offset, extract_all_patches,
                                    generate_all_pairs)

D = 'data/comofod/CoMoFoD_small_v2/'
CKPT = 'checkpoints_pure/swaptest_qnn_epoch10.npz'
S, P = 128, 8
TAU_V3, M0 = 0.90, 8
TAU = np.unique(np.concatenate([[0.0, TAU_V3], 1 - np.logspace(0, -6, 240)[1:]]))
IDS = [f'{i:03d}' for i in range(1, 201)]
GROUPS = [('translation', 0, 40), ('rotation', 40, 80), ('scaling', 80, 120),
          ('distortion', 120, 160), ('combination', 160, 200), ('all', 0, 200)]

q = SwapTestSiameseQNN(patch_size=P, n_layers=3, n_offset_qubits=2, img_size=S, seed=1)
q.load(CKPT)
W = np.asarray(q.weights, dtype=float)
OW = np.asarray(q.offset_weights, dtype=float)
dev = qml.device('default.qubit', wires=6)


@qml.qnode(dev, diff_method=None)
def single(s):
    qml.AmplitudeEmbedding(s, wires=range(6), normalize=True, pad_with=0.0)
    q._reupload_block(W, list(range(6)), s)
    return qml.state()


def v3_states(st):
    return np.concatenate([np.asarray(single(st[i:i + 2048])) for i in range(0, len(st), 2048)]).astype(np.complex128)


def offset_gain(co):
    """g(o) for every coordinate pair: the offset register's exact factor."""
    dy = np.abs(co[:, None, 0] - co[None, :, 0]) / S
    dx = np.abs(co[:, None, 1] - co[None, :, 1]) / S
    b0, b1 = OW[0, 0] + np.pi * dy, OW[0, 1] + np.pi * dx
    p0, p1 = np.sin(b0 / 2) ** 2, np.sin(b1 / 2) ** 2
    f0, f1 = OW[1, 0], OW[1, 1]
    return ((1 - p0) * (1 - p1) + p0 * (1 - p1) * np.cos(f0)
            + (1 - p0) * p1 * np.cos(f1) + p0 * p1 * np.cos(f0 + f1))


def zernike_matrix():
    """Linear map from an 8x8 patch to the 12 complex moments with n <= 5."""
    orders = [(n, m) for n in range(6) for m in range(n % 2, n + 1, 2)]
    rho = (np.arange(13) + 0.5) / 13
    th = 2 * np.pi * np.arange(32) / 32
    M = np.zeros((len(orders), P * P), dtype=np.complex128)
    for k, (n, m) in enumerate(orders):
        for r in rho:
            R = sum((-1) ** s * math.factorial(n - s)
                    / (math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s))
                    * r ** (n - 2 * s) for s in range((n - m) // 2 + 1))
            for t in th:
                y, x = 3.5 + 4 * r * np.sin(t), 3.5 + 4 * r * np.cos(t)
                y0, x0 = int(np.clip(np.floor(y), 0, 6)), int(np.clip(np.floor(x), 0, 6))
                a, b = np.clip(y - y0, 0, 1), np.clip(x - x0, 0, 1)
                w = (n + 1) / np.pi * R * np.exp(-1j * m * t) * r / (len(rho) * len(th))
                M[k, y0 * P + x0] += w * (1 - a) * (1 - b)
                M[k, y0 * P + x0 + 1] += w * (1 - a) * b
                M[k, (y0 + 1) * P + x0] += w * a * (1 - b)
                M[k, (y0 + 1) * P + x0 + 1] += w * a * b
    return M


ZM = zernike_matrix()


def polar_matrix():
    """2 radii x 32 angles inside an 8x8 patch: 64 amplitudes on six qubits."""
    M = np.zeros((64, P * P))
    for i, r in enumerate((2.0, 4.0)):
        for j in range(32):
            t = 2 * np.pi * j / 32
            y, x = 3.5 + r * np.sin(t), 3.5 + r * np.cos(t)
            y0, x0 = int(np.clip(np.floor(y), 0, 6)), int(np.clip(np.floor(x), 0, 6))
            a, b = np.clip(y - y0, 0, 1), np.clip(x - x0, 0, 1)
            M[i * 32 + j, y0 * P + x0] += (1 - a) * (1 - b)
            M[i * 32 + j, y0 * P + x0 + 1] += (1 - a) * b
            M[i * 32 + j, (y0 + 1) * P + x0] += a * (1 - b)
            M[i * 32 + j, (y0 + 1) * P + x0 + 1] += a * b
    return M


PM = polar_matrix()


def qft_scores(pat):
    """Tr(rho_a rho_b) / sqrt(purities) for the QFT-plus-dephasing kernel."""
    s = (pat @ PM.T).reshape(-1, 2, 32)
    s = s - s.mean(-1, keepdims=True)
    f = normalise(np.fft.fft(s, axis=-1, norm='ortho').reshape(-1, 64)).reshape(-1, 2, 32).astype(np.complex64)
    g = np.zeros((len(f), len(f)), dtype=np.float32)
    for m in range(32):
        g += np.abs(f[:, :, m].conj() @ f[:, :, m].T) ** 2
    d = np.sqrt(np.maximum(np.diag(g), 1e-12))
    return (g / d[:, None] / d[None, :]).astype(np.float64)


def normalise(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1), 1 / np.sqrt(v.shape[1]))


def load(path):
    return np.asarray(Image.open(path).convert('L').resize((S, S), Image.LANCZOS), dtype=np.float32) / 255.


def pair_scores(img, kind, stride, mind, min_std):
    win = np.lib.stride_tricks.sliding_window_view(img, (P, P))[::stride, ::stride]
    h, w = win.shape[:2]
    co = np.array([(y * stride, x * stride) for y in range(h) for x in range(w)])
    pat = win.reshape(-1, P * P).astype(np.float64)
    keep = pat.std(1) >= min_std
    co, pat = co[keep], pat[keep]
    if kind == 'qft':
        psi = None
    elif kind == 'v3':
        psi = v3_states(normalise(pat))
    elif kind == 'raw':
        psi = normalise(pat)
    elif kind == 'zm':
        psi = normalise(pat - pat.mean(1, keepdims=True))
    elif kind == 'zer':
        psi = normalise(np.abs(pat @ ZM.T))
    elif kind == 'zer0':
        psi = normalise(np.abs(pat @ ZM.T)[:, 1:])
    s = qft_scores(pat) if kind == 'qft' else np.abs(psi.conj() @ psi.T) ** 2
    if kind == 'v3':
        s = np.clip(s * offset_gain(co), 0, 1)
    d2 = ((co[:, None, :] - co[None, :, :]) ** 2).sum(-1)
    s[d2 < mind ** 2] = -1
    np.fill_diagonal(s, -1)
    return co, s


def curves(img, gt, kind, stride, mind, min_std):
    co, s = pair_scores(img, kind, stride, mind, min_std)
    m = s.max(1) if len(co) else np.zeros(0)
    pix = np.full((S, S), -1.0, dtype=np.float64)
    for i in np.argsort(m):
        y, x = co[i]
        pix[y:y + P, x:x + P] = m[i]
    order = np.sort(s[np.triu_indices(len(co), 1)])
    count = len(order) - np.searchsorted(order, TAU, side='left')
    srt = np.argsort(pix.ravel())
    v = pix.ravel()[srt]
    pred = v.size - np.searchsorted(v, TAU, side='left')
    if gt is None:
        return count, pred / v.size, None
    g = gt.ravel()[srt].astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(g)])
    tp = cum[-1] - cum[np.searchsorted(v, TAU, side='left')]
    return count, pred / v.size, 2 * tp / np.maximum(pred + gt.sum(), 1)


def auc(p, n):
    p, n = np.asarray(p, dtype=float), np.asarray(n, dtype=float)
    return float(np.mean([(x > n).mean() + .5 * (x == n).mean() for x in p]))


def run(name, kind, stride, mind, min_std):
    F, CF, CO, AF, AO = [], [], [], [], []
    for s in IDS:
        gt = np.asarray(Image.open(f'{D}{s}_B.png').convert('L').resize((S, S), Image.NEAREST)) > 128
        cf, af, f = curves(load(f'{D}{s}_F.png'), gt, kind, stride, mind, min_std)
        co, ao, _ = curves(load(f'{D}{s}_O.png'), None, kind, stride, mind, min_std)
        F.append(f); CF.append(cf); CO.append(co); AF.append(af); AO.append(ao)
    F, CF, CO, AF, AO = map(np.array, (F, CF, CO, AF, AO))
    idx = np.arange(200)
    # Threshold chosen on the other 199 scenes, per scene and in one pass
    tot = F.sum(0)
    pick = ((tot[None, :] - F) / 199).argmax(1)
    fixed = F[idx, pick]
    t = F.mean(0).argmax()
    out = dict(name=name, tau=float(TAU[t]), flagged_forged=float(AF[:, t].mean()), flagged_auth=float(AO[:, t].mean()),
               groups={})
    for g, a, b in GROUPS:
        out['groups'][g] = dict(oracle=float(F[a:b].max(1).mean()), fixed=float(fixed[a:b].mean()),
                                image_auc=auc(CF[a:b, t], CO[a:b, t]))
    a = out['groups']['all']
    print(f'{name:40s} oracle {a["oracle"]:.3f} fixed {a["fixed"]:.3f} tau {TAU[t]:.4f} '
          f'flagged F {AF[:, t].mean()*100:5.1f}% O {AO[:, t].mean()*100:5.1f}% image AUC {a["image_auc"]:.3f}', flush=True)
    print('   ' + ' | '.join(f'{g[:5]} {out["groups"][g]["oracle"]:.3f}/{out["groups"][g]["fixed"]:.3f}/{out["groups"][g]["image_auc"]:.2f}'
                             for g, _, _ in GROUPS[:5]), flush=True)
    if kind == 'v3':
        k = np.argmin(np.abs(TAU - TAU_V3))
        tp = (CF[:, k] >= M0).mean(); fp = (CO[:, k] >= M0).mean()
        out['dashboard'] = dict(f1=float(F[:, k].mean()), tpr=float(tp), fpr=float(fp),
                                accuracy=float((tp + 1 - fp) / 2),
                                pairs_001=int(CF[0, k]), median_pairs_forged=float(np.median(CF[:, k])),
                                median_pairs_auth=float(np.median(CO[:, k])))
        print('   dashboard rule tau 0.90, 8 pairs:', out['dashboard'], flush=True)
    return out


# ---- checks against the real circuits ------------------------------------- #
img = load(D + '001_F.png')
co, st = extract_all_patches(img, P, P)
pairs = generate_all_pairs(co, min_spatial_dist=P)
rng = np.random.default_rng(0)
ch = [pairs[i] for i in rng.integers(0, len(pairs), 128)]
off = np.array([encode_offset(co[j][0] - co[i][0], co[j][1] - co[i][1], S) for i, j in ch])
real = q.similarity(st[[i for i, j in ch]], st[[j for i, j in ch]], off)
_, s = pair_scores(img, 'v3', P, P, 0.0)
print(f'check v3: max|factored - real 15-wire circuit| = {np.abs(real - np.array([s[i, j] for i, j in ch])).max():.2e}', flush=True)

zdev = qml.device('lightning.qubit', wires=9)


@qml.qnode(zdev, diff_method=None)
def swap9(a, b):
    qml.AmplitudeEmbedding(a, wires=range(4), normalize=True, pad_with=0.0)
    qml.AmplitudeEmbedding(b, wires=range(4, 8), normalize=True, pad_with=0.0)
    qml.Hadamard(8)
    for k in range(4):
        qml.CSWAP(wires=[8, k, k + 4])
    qml.Hadamard(8)
    return qml.expval(qml.PauliZ(8))


win = np.lib.stride_tricks.sliding_window_view(img, (P, P))[::P, ::P].reshape(-1, P * P).astype(np.float64)
zs = normalise(np.abs(win @ ZM.T))
r9 = [float(swap9(zs[i], zs[j])) for i, j in ch[:32]]
f9 = [(zs[i] @ zs[j]) ** 2 for i, j in ch[:32]]
print(f'check Zernike: max|exact - real 9-wire circuit| = {np.abs(np.array(r9) - np.array(f9)).max():.2e}', flush=True)

R = []
if len(sys.argv) > 2 and sys.argv[2] == 'qft':
    R.append(run('QFT kernel, zero-mean, skip flat', 'qft', 8, 8, 0.02))
    R.append(run('QFT kernel, zero-mean, skip flat, s2', 'qft', 2, 16, 0.02))
    json.dump(R, open(sys.argv[1], 'w'), indent=1)
    sys.exit(0)
R.append(run('v3, 15 wires, as deployed', 'v3', 8, 8, 0.0))
R.append(run('plain swap, raw, 13 wires', 'raw', 8, 8, 0.0))
R.append(run('plain swap, zero-mean, skip flat', 'zm', 8, 8, 0.02))
R.append(run('plain swap, zero-mean, skip flat, s2', 'zm', 2, 16, 0.02))
R.append(run('Zernike 12, 9 wires', 'zer', 8, 8, 0.0))
R.append(run('Zernike 11 no Z00, 9 wires, skip flat', 'zer0', 8, 8, 0.02))
R.append(run('Zernike 12, 9 wires, skip flat, s2', 'zer', 2, 16, 0.02))
R.append(run('Zernike 11 no Z00, skip flat, s2', 'zer0', 2, 16, 0.02))
if len(sys.argv) > 1:
    json.dump(R, open(sys.argv[1], 'w'), indent=1)
print('done', flush=True)
