"""
Zernike swap-test QNN: 9 wires, 36 trainable rotation angles.
=============================================================
What this model does and does not deliver:
  - Speed and rotation handling: Zernike at 9 wires delivers these, as a
    classical pre-processing step.
  - Better detection: nothing tested so far delivers it, and this code alone
    does not change that. evaluate_zernike.py measures it on held-out scenes;
    do not claim a detection gain unless that script reports one.

Same Siamese swap-test circuit as v3 (cmfd_swaptest_pipeline.py), with two
changes:
  - Each 8x8 patch is described by 12 rotation-invariant Zernike magnitudes,
    amplitude-encoded on 4 qubits instead of 64 raw pixels on 6.
  - No offset register. Its trained gain is effectively constant, so it only
    added wires.

Wires: 4 (register A) + 4 (register B) + 1 (ancilla) = 9.
Trainable: tied weights of shape (n_layers, 4, 3), so 36 at n_layers=3. They
are all circuit rotation angles, with no classical trainable layer. The Zernike
moments are fixed classical data preparation and nothing in them is trained,
so describe this as a pure QNN with classical Zernike data preparation.

Train with train_zernike.py. v3 itself is unchanged by this file.
"""
import math

import numpy as np
import pennylane as qml

from cmfd_swaptest_pipeline import SwapTestSiameseQNN

MAX_ORDER = 5           # orders n <= 5 with m >= 0 give 12 moments
N_RADII = 13
N_ANGLES = 32
STATE_DIM = 16          # 12 moments zero-padded onto 4 qubits
# v3's sqrt(dim) = 4 made copies score only about 0.49 and training unstable.
# In trials 2 gave the lowest validation loss and a stable curve; 1 and 4 did not.
REUPLOAD_SCALE = 2.0


def zernike_matrix(patch_size: int = 8) -> np.ndarray:
    """Fixed linear map from a flattened patch to its complex Zernike moments,
    sampled on a polar grid by bilinear interpolation."""
    orders = [(n, m) for n in range(MAX_ORDER + 1) for m in range(n % 2, n + 1, 2)]
    centre = (patch_size - 1) / 2
    radius = patch_size / 2
    rhos = (np.arange(N_RADII) + 0.5) / N_RADII
    thetas = 2 * np.pi * np.arange(N_ANGLES) / N_ANGLES
    M = np.zeros((len(orders), patch_size * patch_size), dtype=np.complex128)
    for k, (n, m) in enumerate(orders):
        for rho in rhos:
            radial = sum(
                (-1) ** h * math.factorial(n - h)
                / (math.factorial(h) * math.factorial((n + m) // 2 - h) * math.factorial((n - m) // 2 - h))
                * rho ** (n - 2 * h) for h in range((n - m) // 2 + 1))
            for theta in thetas:
                y = centre + radius * rho * np.sin(theta)
                x = centre + radius * rho * np.cos(theta)
                y0 = int(np.clip(np.floor(y), 0, patch_size - 2))
                x0 = int(np.clip(np.floor(x), 0, patch_size - 2))
                a, b = np.clip(y - y0, 0, 1), np.clip(x - x0, 0, 1)
                w = (n + 1) / np.pi * radial * np.exp(-1j * m * theta) * rho / (N_RADII * N_ANGLES)
                M[k, y0 * patch_size + x0] += w * (1 - a) * (1 - b)
                M[k, y0 * patch_size + x0 + 1] += w * (1 - a) * b
                M[k, (y0 + 1) * patch_size + x0] += w * a * (1 - b)
                M[k, (y0 + 1) * patch_size + x0 + 1] += w * a * b
    return M


_MATRICES = {}


def zernike_state(patch: np.ndarray) -> np.ndarray:
    """Patch(es) of shape (..., p, p) -> amplitude vector(s) of length 16.
    A zero descriptor gets the uniform state, as v3 does for a zero patch."""
    patch = np.asarray(patch, dtype=np.float64)
    p = patch.shape[-1]
    if p not in _MATRICES:
        _MATRICES[p] = zernike_matrix(p)
    mags = np.abs(patch.reshape(*patch.shape[:-2], p * p) @ _MATRICES[p].T)
    pad = np.zeros((*mags.shape[:-1], STATE_DIM - mags.shape[-1]))
    state = np.concatenate([mags, pad], axis=-1)
    norm = np.linalg.norm(state, axis=-1, keepdims=True)
    return np.where(norm > 0, state / np.where(norm > 0, norm, 1), 1 / np.sqrt(STATE_DIM))


def image_states(image: np.ndarray, patch_size: int = 8, stride: int = 8):
    """Tile an image the way v3 does. Returns (coords, states)."""
    win = np.lib.stride_tricks.sliding_window_view(image, (patch_size, patch_size))[::stride, ::stride]
    h, w = win.shape[:2]
    coords = [(y * stride, x * stride) for y in range(h) for x in range(w)]
    return coords, zernike_state(win.reshape(-1, patch_size, patch_size))


class ZernikeSwapQNN(SwapTestSiameseQNN):
    """v3's circuit on 4-qubit registers with no offset register. patch_size=4
    here only sets the register size; image patches stay 8x8 in image_states."""

    def __init__(self, n_layers: int = 3, img_size: int = 128,
                 diff_method: str = 'auto', seed: int = None):
        super().__init__(patch_size=4, n_layers=n_layers, n_offset_qubits=0,
                         img_size=img_size, diff_method=diff_method, seed=seed)
        self.reupload_scale = REUPLOAD_SCALE

    def n_trainable(self) -> int:
        return int(np.prod(self.weights.shape))

    def score(self, states_a, states_b):
        """Swap-test score for each pair, one real 9-wire circuit per pair."""
        offsets = np.zeros((*np.shape(states_a)[:-1], 2))
        return self.similarity(states_a, states_b, offsets)

    def all_pair_scores(self, states) -> np.ndarray:
        """N x N scores from one 4-qubit state per patch. Exact and far faster
        than a circuit per pair, in simulation only."""
        wires = list(range(self.n_qubits))
        weights = np.asarray(self.weights, dtype=float)

        @qml.qnode(qml.device('default.qubit', wires=self.n_qubits), diff_method=None)
        def register(s):
            qml.AmplitudeEmbedding(s, wires=wires, normalize=True, pad_with=0.0)
            self._reupload_block(weights, wires, s)
            return qml.state()

        psi = np.asarray(register(states))
        return np.clip(np.abs(psi.conj() @ psi.T) ** 2, 0.0, 1.0)
