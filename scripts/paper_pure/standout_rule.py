"""Does 'one match stands out' separate real copies from look-alikes?
Same v3 weights and scores; only the readout rule changes.
  z     : (best - mean of row) / std of row
  ratio : 1 - (1 - best) / (1 - second best), second best not overlapping the best
Thresholds are shared across images: oracle per image, and leave-one-scene-out."""
import sys
import numpy as np
import pennylane as qml
from PIL import Image
sys.path.insert(0, 'code')
from cmfd_swaptest_pipeline import SwapTestSiameseQNN, prepare_patch_state

D = 'data/comofod/CoMoFoD_small_v2/'
P = 8
q = SwapTestSiameseQNN(patch_size=P, n_layers=3, n_offset_qubits=0, img_size=128, seed=1)
q.load('checkpoints_pure/swaptest_qnn_epoch10.npz')
W = np.asarray(q.weights, dtype=float)
dev = qml.device('default.qubit', wires=6)


@qml.qnode(dev, diff_method=None)
def single(s):
    qml.AmplitudeEmbedding(s, wires=range(6), normalize=True, pad_with=0.0)
    q._reupload_block(W, list(range(6)), s)
    return qml.state()


def load(path, size):
    return np.asarray(Image.open(path).convert('L').resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.


def rule_scores(img, stride, kind, zero_mean, mind, min_std):
    """Returns coords and per-patch z and ratio scores."""
    H = img.shape[0]
    co, st = [], []
    for y in range(0, H - P + 1, stride):
        for x in range(0, H - P + 1, stride):
            p = img[y:y + P, x:x + P].astype(np.float64)
            if p.std() < min_std:
                continue
            co.append((y, x))
            st.append(prepare_patch_state(p - p.mean() if zero_mean else p))
    co, st = np.array(co), np.array(st)
    if kind == 'qnn':
        psi = np.concatenate([np.asarray(single(st[i:i + 2048])) for i in range(0, len(st), 2048)]).astype(np.complex64)
    else:
        psi = st.astype(np.complex64)
    z, ratio = np.zeros(len(co), np.float32), np.zeros(len(co), np.float32)
    for a in range(0, len(co), 512):
        c = co[a:a + 512]
        s = np.abs(psi[a:a + 512].conj() @ psi.T) ** 2
        d2 = ((c[:, None, :] - co[None, :, :]) ** 2).sum(-1)
        s[d2 < mind ** 2] = np.nan
        best = np.nanargmax(s, 1)
        r = np.arange(len(c))
        m1 = s[r, best]
        z[a:a + 512] = (m1 - np.nanmean(s, 1)) / (np.nanstd(s, 1) + 1e-9)
        near = ((co[best][:, None, :] - co[None, :, :]) ** 2).sum(-1) < P ** 2
        s2 = np.where(near, np.nan, s)
        m2 = np.nanmax(s2, 1)
        ratio[a:a + 512] = 1 - (1 - m1) / np.maximum(1 - m2, 1e-9)
    return co, z, ratio


def paint(co, v, H):
    pix = np.full((H, H), -np.inf, dtype=np.float32)
    for i in np.argsort(v):
        y, x = co[i]
        pix[y:y + P, x:x + P] = v[i]
    return pix


def f1s(pix, gt, taus):
    return np.array([2 * ((pix >= t) & gt).sum() / max(1, (pix >= t).sum() + gt.sum()) for t in taus])


def auc(p, n):
    p, n = np.asarray(p), np.asarray(n)
    return float(np.mean([(x > n).mean() + .5 * (x == n).mean() for x in p]))


def run(label, ids, size, stride, kind, zm, mind, min_std):
    res = {'z': [], 'ratio': []}
    for s in ids:
        gt = np.asarray(Image.open(f'{D}{s}_B.png').convert('L').resize((size, size), Image.NEAREST)) > 128
        cf, zf, rf = rule_scores(load(f'{D}{s}_F.png', size), stride, kind, zm, mind, min_std)
        co, zo, ro = rule_scores(load(f'{D}{s}_O.png', size), stride, kind, zm, mind, min_std)
        res['z'].append((paint(cf, zf, size), paint(co, zo, size), gt, np.percentile(zf, 99), np.percentile(zo, 99)))
        res['ratio'].append((paint(cf, rf, size), paint(co, ro, size), gt, np.percentile(rf, 99), np.percentile(ro, 99)))
    for rule, rows in res.items():
        pool = np.concatenate([r[0][np.isfinite(r[0])].ravel()[::7] for r in rows])
        taus = np.unique(np.quantile(pool, np.linspace(0, .9995, 200)))
        F = np.array([f1s(r[0], r[2], taus) for r in rows])
        idx = np.arange(len(F))
        loo = np.mean([F[i, F[idx != i].mean(0).argmax()] for i in idx])
        t = taus[F.mean(0).argmax()]
        ff = np.mean([(r[0] >= t).mean() for r in rows]); fa = np.mean([(r[1] >= t).mean() for r in rows])
        print(f'{label:26s} {kind:3s} {rule:5s} {size}px s{stride} n={len(ids):3d} | oracle {F.max(1).mean():.3f} | '
              f'fixed {loo:.3f} | flagged forged {ff*100:5.1f}% authentic {fa*100:5.1f}% | '
              f'image AUC {auc([r[3] for r in rows],[r[4] for r in rows]):.3f}', flush=True)


S8 = [f'{i:03d}' for i in range(1, 9)]
T40 = [f'{i:03d}' for i in range(1, 41)]
R20 = [f'{i:03d}' for i in range(41, 61)]
run('v3 raw, as is', S8, 128, 8, 'qnn', False, 8, 0.0)
run('raw', S8, 128, 4, 'qnn', False, 16, 0.0)
run('raw', S8, 128, 2, 'qnn', False, 16, 0.0)
run('zero-mean, skip flat', S8, 128, 8, 'qnn', True, 8, 0.02)
run('zero-mean, skip flat', S8, 128, 4, 'qnn', True, 16, 0.02)
run('zero-mean, skip flat', S8, 128, 2, 'qnn', True, 16, 0.02)
run('zero-mean, skip flat', S8, 128, 2, 'cos', True, 16, 0.02)
run('zero-mean, skip flat', T40, 128, 2, 'qnn', True, 16, 0.02)
run('zero-mean, skip flat', T40, 128, 2, 'cos', True, 16, 0.02)
run('zero-mean, skip flat', R20, 128, 2, 'qnn', True, 16, 0.02)
print('done')
