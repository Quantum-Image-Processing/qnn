"""
Patch-representation robustness on real CoMoFoD originals at 512x512.
Run from the repository root:  python scripts/paper_pure/descriptors.py [out.json]

A neighbourhood is copied and transformed; a query is correct when the true
source is the best match, within 2 px, among all positions on a 2-px grid.
20 random originals, 25 locations each, so 500 queries per condition.
"""
import io
import json
import math
import sys
import numpy as np
from PIL import Image

D = 'data/comofod/CoMoFoD_small_v2/'
R, W = 8, 17
rng = np.random.default_rng(0)
IDS = [f'{i:03d}' for i in sorted(rng.choice(np.arange(1, 201), 20, replace=False))]


def bilinear(rows, cols, y, x, w):
    y0, x0 = int(np.clip(np.floor(y), 0, W - 2)), int(np.clip(np.floor(x), 0, W - 2))
    a, b = np.clip(y - y0, 0, 1), np.clip(x - x0, 0, 1)
    rows[y0 * W + x0] += w * (1 - a) * (1 - b)
    rows[y0 * W + x0 + 1] += w * (1 - a) * b
    rows[(y0 + 1) * W + x0] += w * a * (1 - b)
    rows[(y0 + 1) * W + x0 + 1] += w * a * b


def zernike_matrix():
    orders = [(n, m) for n in range(6) for m in range(n % 2, n + 1, 2)]
    rho = (np.arange(26) + 0.5) / 26
    th = 2 * np.pi * np.arange(32) / 32
    M = np.zeros((len(orders), W * W), dtype=np.complex128)
    for k, (n, m) in enumerate(orders):
        for r in rho:
            Rn = sum((-1) ** s * math.factorial(n - s)
                     / (math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s))
                     * r ** (n - 2 * s) for s in range((n - m) // 2 + 1))
            for t in th:
                w = (n + 1) / np.pi * Rn * np.exp(-1j * m * t) * r / (len(rho) * len(th))
                bilinear(M[k], None, R + R * r * np.sin(t), R + R * r * np.cos(t), w)
    return M


def polar_matrix():
    """2 radii x 32 angles, so 64 amplitudes on six qubits."""
    S = np.zeros((64, W * W))
    for i, r in enumerate((R / 2, R)):
        for j in range(32):
            t = 2 * np.pi * j / 32
            bilinear(S[i * 32 + j], None, R + r * np.sin(t), R + r * np.cos(t), 1.0)
    return S


ZM, PM = zernike_matrix(), polar_matrix()


def norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


def raw_vec(p):
    c = R - 4
    return norm(p[..., c:c + 8, c:c + 8].reshape(*p.shape[:-2], 64))


def zer_vec(p):
    return norm(np.abs(p.reshape(*p.shape[:-2], W * W) @ ZM.T))


def qft_vec(p, zero_mean):
    s = p.reshape(*p.shape[:-2], W * W) @ PM.T
    if zero_mean:
        s = s - s.mean(-1, keepdims=True)
    s = s.reshape(*s.shape[:-1], 2, 32)
    f = norm(np.fft.fft(s, axis=-1, norm='ortho').reshape(*s.shape[:-2], 64)).reshape(*s.shape)
    return f


def qft_pur(f):
    return (np.abs((f.conj() * f).sum(-2)) ** 2).sum(-1)


def transform(c, cond):
    kind, v = cond
    u8 = np.uint8(np.clip(c, 0, 1) * 255 + .5)
    if kind == 'rot':
        return np.asarray(Image.fromarray(u8).rotate(v, Image.BICUBIC), dtype=np.float64) / 255.
    if kind == 'scale':
        h = c.shape[0]
        n = int(round(h * v))
        im = np.asarray(Image.fromarray(u8).resize((n, n), Image.BICUBIC), dtype=np.float64) / 255.
        o = (n - h) // 2
        return im[o:o + h, o:o + h]
    if kind == 'jpeg':
        b = io.BytesIO(); Image.fromarray(u8).save(b, 'JPEG', quality=v)
        return np.asarray(Image.open(b).convert('L'), dtype=np.float64) / 255.
    if kind == 'noise':
        return np.clip(c + rng.normal(0, v, c.shape), 0, 1)
    if kind == 'rotjpeg':
        return transform(transform(c, ('rot', v)), ('jpeg', 70))


CONDS = [('none', ('rot', 0)), ('rot5', ('rot', 5)), ('rot15', ('rot', 15)), ('rot30', ('rot', 30)), ('rot45', ('rot', 45)),
         ('rot90', ('rot', 90)), ('jpeg70', ('jpeg', 70)), ('noise.02', ('noise', .02)), ('scale1.07', ('scale', 1.07)),
         ('rot15+jpeg70', ('rotjpeg', 15))]
METHODS = ['raw', 'zernike', 'qft', 'qft-zm']
hits = {m: {c: 0 for c, _ in CONDS} for m in METHODS}
total = 0
for sid in IDS:
    im = np.asarray(Image.open(f'{D}{sid}_O.png').convert('L').resize((512, 512), Image.LANCZOS), dtype=np.float64) / 255.
    win = np.lib.stride_tricks.sliding_window_view(im, (W, W))[::2, ::2]
    h, w = win.shape[:2]
    flat = win.reshape(-1, W, W)
    yy, xx = np.mgrid[0:h, 0:w]
    grid = np.stack([yy.ravel() * 2, xx.ravel() * 2], 1)
    db = {'raw': raw_vec(flat), 'zernike': zer_vec(flat)}
    fq, fz = qft_vec(flat, False), qft_vec(flat, True)
    pq, pz = qft_pur(fq), qft_pur(fz)
    locs = np.stack([rng.integers(30, 220, 25) * 2, rng.integers(30, 220, 25) * 2], 1)
    for (y0, x0) in locs:
        total += 1
        C = 40
        cy, cx = y0 + R, x0 + R
        crop = im[cy - C:cy + C + 1, cx - C:cx + C + 1]
        for cname, cond in CONDS:
            t = transform(crop, cond)
            q = t[C - R:C + R + 1, C - R:C + R + 1]
            for m in METHODS:
                if m == 'raw':
                    s = np.abs(db['raw'] @ raw_vec(q)) ** 2
                elif m == 'zernike':
                    s = (db['zernike'] @ zer_vec(q)) ** 2
                else:
                    F, P = (fq, pq) if m == 'qft' else (fz, pz)
                    v = qft_vec(q, m == 'qft-zm')
                    s = (np.abs((v.conj()[None] * F).sum(1)) ** 2).sum(1) / np.sqrt(np.maximum(P * qft_pur(v), 1e-18))
                g = grid[np.nanargmax(s)]
                hits[m][cname] += int(abs(g[0] - y0) <= 2 and abs(g[1] - x0) <= 2)
    print(sid, flush=True)
res = {m: {c: 100 * hits[m][c] / total for c, _ in CONDS} for m in METHODS}
print('queries per condition', total)
print(f'{"":10s}' + ''.join(f'{c:>13s}' for c, _ in CONDS))
for m in METHODS:
    print(f'{m:10s}' + ''.join(f'{res[m][c]:13.1f}' for c, _ in CONDS))
if len(sys.argv) > 1:
    json.dump(dict(queries=total, percent=res), open(sys.argv[1], 'w'), indent=1)
