"""
Checks behind the paper's closed-form and QFT-kernel statements.
Run from the repository root:  python scripts/paper_pure/verify_claims.py
"""
import sys, time
import numpy as np
import pennylane as qml
from PIL import Image
sys.path.insert(0, 'code')
from cmfd_swaptest_pipeline import SwapTestSiameseQNN, extract_all_patches

D = 'data/comofod/CoMoFoD_small_v2/'
q = SwapTestSiameseQNN(patch_size=8, n_layers=3, n_offset_qubits=2, img_size=128, seed=1)
q.load('checkpoints_pure/swaptest_qnn_epoch10.npz')
W, OW = np.asarray(q.weights, float), np.asarray(q.offset_weights, float)
dev6 = qml.device('default.qubit', wires=6)


@qml.qnode(dev6, diff_method=None)
def single(s):
    qml.AmplitudeEmbedding(s, wires=range(6), normalize=True, pad_with=0.0)
    q._reupload_block(W, list(range(6)), s)
    return qml.state()


def gain(co, size):
    dy = np.abs(co[:, None, 0] - co[None, :, 0]) / size
    dx = np.abs(co[:, None, 1] - co[None, :, 1]) / size
    p0, p1 = np.sin((OW[0, 0] + np.pi * dy) / 2) ** 2, np.sin((OW[0, 1] + np.pi * dx) / 2) ** 2
    f0, f1 = OW[1, 0], OW[1, 1]
    return ((1 - p0) * (1 - p1) + p0 * (1 - p1) * np.cos(f0) + (1 - p0) * p1 * np.cos(f1) + p0 * p1 * np.cos(f0 + f1))


for size in (128, 256):
    img = np.asarray(Image.open(D + '001_F.png').convert('L').resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.
    co, st = extract_all_patches(img, 8, 8)
    co = np.array(co)
    single(st[:8])
    t = time.perf_counter()
    psi = np.concatenate([np.asarray(single(st[i:i + 2048])) for i in range(0, len(st), 2048)])
    g = gain(co, size)
    s = np.clip(np.abs(psi.conj() @ psi.T) ** 2 * g, 0, 1)
    print(f'{size}px factored scores for {len(co)*(len(co)-1)//2} pairs: {time.perf_counter()-t:.2f} s')
    if size == 128:
        iu = np.triu_indices(len(co), 1)
        print(f'g(o) range [{g[iu].min():.4f}, {g[iu].max():.4f}] mean {g[iu].mean():.4f} std {g[iu].std():.4f}')

# QFT kernel: one real circuit against the formula
rng = np.random.default_rng(0)
ang, n = 5, 6
wires = {'a': list(range(6)), 'ta': list(range(6, 11)), 'b': list(range(11, 17)), 'tb': list(range(17, 22)), 'anc': 22}
dev = qml.device('lightning.qubit', wires=23)


@qml.qnode(dev, diff_method=None)
def kernel(a, b):
    qml.AmplitudeEmbedding(a, wires=wires['a'], normalize=True)
    qml.AmplitudeEmbedding(b, wires=wires['b'], normalize=True)
    for reg, tr in (('a', 'ta'), ('b', 'tb')):
        qml.QFT(wires=wires[reg][1:])
        for c, d in zip(wires[reg][1:], wires[tr]):
            qml.CNOT(wires=[c, d])
    qml.Hadamard(wires['anc'])
    for x, y in zip(wires['a'], wires['b']):
        qml.CSWAP(wires=[wires['anc'], x, y])
    qml.Hadamard(wires['anc'])
    return qml.expval(qml.PauliZ(wires['anc']))


def formula(a, b):
    fa = np.fft.ifft(a.reshape(2, 32), axis=1, norm='ortho'); fb = np.fft.ifft(b.reshape(2, 32), axis=1, norm='ortho')
    return float((np.abs((fa.conj() * fb).sum(0)) ** 2).sum())


devs = []
for i in range(3):
    a, b = rng.normal(size=64), rng.normal(size=64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    t = time.perf_counter(); r = float(kernel(a, b)); e = time.perf_counter() - t
    devs.append(abs(r - formula(a, b)))
    print(f'QFT kernel pair {i}: circuit {r:.6f} formula {formula(a, b):.6f} ({e:.1f} s, 23 wires)', flush=True)
print(f'max deviation {max(devs):.2e}')
