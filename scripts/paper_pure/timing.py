"""Time per pair for each real circuit on one process, same 256 pairs of scene 001."""
import sys, time
import numpy as np
import pennylane as qml
from PIL import Image
sys.path.insert(0, 'code')
from cmfd_swaptest_pipeline import SwapTestSiameseQNN, extract_all_patches, generate_all_pairs, encode_offset
CK = 'checkpoints_pure/swaptest_qnn_epoch10.npz'
img = np.asarray(Image.open('data/comofod/CoMoFoD_small_v2/001_F.png').convert('L').resize((128, 128), Image.LANCZOS), dtype=np.float32) / 255.
co, st = extract_all_patches(img, 8, 8)
pairs = generate_all_pairs(co, min_spatial_dist=8)
ch = pairs[:256]
A, B = st[[i for i, j in ch]], st[[j for i, j in ch]]
off = np.array([encode_offset(co[j][0] - co[i][0], co[j][1] - co[i][1], 128) for i, j in ch])
N = len(pairs)


def report(name, wires, f):
    f(); t = time.perf_counter(); f(); d = (time.perf_counter() - t) / 256
    print(f'{name:34s} {wires:2d} wires {d*1e3:7.2f} ms/pair  {d*N/60:6.2f} min/image', flush=True)


for nq, name in ((2, 'v3 as deployed'), (0, 'v3 no offset register')):
    q = SwapTestSiameseQNN(patch_size=8, n_layers=3, n_offset_qubits=nq, img_size=128, seed=1)
    q.load(CK)
    report(name, q.n_wires, lambda: q.similarity(A, B, off))


def plain(nq, a, b):
    dev = qml.device('lightning.qubit', wires=2 * nq + 1)

    @qml.qnode(dev, diff_method=None)
    def c(x, y):
        qml.AmplitudeEmbedding(x, wires=range(nq), normalize=True, pad_with=0.0)
        qml.AmplitudeEmbedding(y, wires=range(nq, 2 * nq), normalize=True, pad_with=0.0)
        qml.Hadamard(2 * nq)
        for k in range(nq):
            qml.CSWAP(wires=[2 * nq, k, k + nq])
        qml.Hadamard(2 * nq)
        return qml.expval(qml.PauliZ(2 * nq))
    return lambda: c(a, b)


report('plain swap, no trained layers', 13, plain(6, A, B))
rng = np.random.default_rng(0)
za, zb = np.abs(rng.normal(size=(256, 12))), np.abs(rng.normal(size=(256, 12)))
report('plain swap on 12 Zernike values', 9, plain(4, za, zb))
