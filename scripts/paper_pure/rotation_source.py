"""Which part gives rotation handling: the Zernike input, the trained circuit, or its entanglement?
Same test as evaluate_zernike.py: a rotated 8x8 copy must find its true source (within 2 px)
among all patches of the image on a 2-px grid. Read-only."""
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from PIL import Image

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / 'code'))
from cmfd_swaptest_pipeline import SwapTestSiameseQNN  # noqa: E402
from cmfd_zernike_swaptest import ZernikeSwapQNN, zernike_state  # noqa: E402
from evaluate_zernike import load, SIZE, PATCH  # noqa: E402


def raw_state(p):
    flat = np.asarray(p, dtype=np.float64).reshape(len(p), -1)
    n = np.linalg.norm(flat, axis=1, keepdims=True)
    return np.where(n > 1e-12, flat / np.maximum(n, 1e-12), 1 / np.sqrt(flat.shape[1]))


def register_fn(q, n_qubits, entangle=True):
    w = np.asarray(q.weights, dtype=float)
    wires = list(range(n_qubits))

    @qml.qnode(qml.device('default.qubit', wires=n_qubits), diff_method=None)
    def reg(s):
        qml.AmplitudeEmbedding(s, wires=wires, normalize=True, pad_with=0.0)
        idx = 0
        for layer in range(q.n_layers):
            for k, wire in enumerate(wires):
                for g, axis in ((qml.RX, 0), (qml.RY, 1), (qml.RZ, 2)):
                    g(w[layer, k, axis] + q._reupload_angle(s, idx), wires=wire); idx += 1
            if entangle:
                for k, wire in enumerate(wires):
                    qml.CNOT(wires=[wire, wires[(k + 1) % len(wires)]])
        return qml.state()
    return lambda s: np.asarray(reg(s))


def entanglement(psi, n_qubits):
    """Mean entropy (bits) between qubits 0-1 and the rest: 0 = no entanglement."""
    m = psi.reshape(len(psi), 4, -1)
    sv = np.linalg.svd(m, compute_uv=False) ** 2
    return float(np.mean(-(sv * np.log2(np.clip(sv, 1e-15, 1))).sum(1)))


v3 = SwapTestSiameseQNN(patch_size=PATCH, n_layers=3, n_offset_qubits=2, img_size=SIZE, seed=1)
v3.load(ROOT / 'checkpoints_pure' / 'swaptest_qnn_epoch10.npz')
zq = ZernikeSwapQNN(n_layers=3, img_size=SIZE, seed=1)
zq.load(ROOT / 'models' / 'zernike_9w.npz')

methods = {   # name -> (input prep, map from input to compared vector)
    'raw pixels, no circuit':                (raw_state, lambda s: s),
    'raw pixels, v3 trained circuit':        (raw_state, register_fn(v3, 6)),
    'Zernike, no circuit (plain swap test)': (zernike_state, lambda s: s),
    'Zernike, trained circuit':              (zernike_state, register_fn(zq, 4)),
    'Zernike, trained circuit, CNOTs removed': (zernike_state, register_fn(zq, 4, entangle=False)),
}

ids = [f'{int(s):03d}' for s in json.load(open(ROOT / 'models' / 'split.json'))['test_ids']][:20]
root = ROOT / 'data' / 'comofod'
data = root / 'CoMoFoD_small_v2' if (root / 'CoMoFoD_small_v2').exists() else root

ent = {}
for angle in (15, 45, 90):
    rng = np.random.default_rng(0)
    hits = dict.fromkeys(methods, 0); total = 0
    for sid in ids:
        arr = load(data / f'{sid}_O.png')
        win = np.lib.stride_tricks.sliding_window_view(arr, (PATCH, PATCH))[::2, ::2]
        h, w = win.shape[:2]
        flat = win.reshape(-1, PATCH, PATCH).astype(np.float64)
        grid = np.stack(np.mgrid[0:h, 0:w], -1).reshape(-1, 2) * 2
        dbs = {}
        for name, (prep, f) in methods.items():
            dbs[name] = f(prep(flat))
            if angle == 15 and 'trained circuit' in name and sid == ids[0]:
                ent[name] = entanglement(dbs[name], 6 if 'v3' in name else 4)
        for _ in range(10):
            y, x = rng.integers(6, 54, 2) * 2
            crop = Image.fromarray(np.uint8(arr[y - 12:y + 20, x - 12:x + 20] * 255))
            rot = np.asarray(crop.rotate(angle, Image.BICUBIC), dtype=np.float64)[12:20, 12:20] / 255.0
            for name, (prep, f) in methods.items():
                qv = f(prep(rot[None]))[0]
                found = grid[np.argmax(np.abs(dbs[name].conj() @ qv) ** 2)]
                hits[name] += abs(found[0] - y) <= 2 and abs(found[1] - x) <= 2
            total += 1
    print(f'\n{angle} deg rotation, {total} queries: true source found')
    for name in methods:
        print(f'  {name:42s} {100 * hits[name] / total:5.1f}%')
print('\nentanglement in the trained register states (mean, bits):', {k: round(v, 3) for k, v in ent.items()})
