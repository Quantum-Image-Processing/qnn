"""
CMFD Pipeline v3 - PURE Siamese Swap-Test QNN for Copy-Move Detection
=======================================================================
v3 removes every hand-engineered classical stage from the v2 hybrid so
that the detector is quantum end-to-end: raw pixels go into the circuit
and a measured expectation value comes out. Nothing is learned or
decided classically.

What changed from v2 (the hybrid), and why
------------------------------------------
  - v2 fed the circuit a DCT + intensity-histogram descriptor. That is
    classical feature engineering and was the single biggest reason v2
    could not be called a quantum model: the circuit only ever saw
    24 numbers a classical algorithm had already decided were the
    interesting ones. v3 amplitude-encodes the RAW patch pixels.
  - v2 pre-filtered candidate pairs with Fridrich-style lexicographic
    block matching (a classical O(N log N) sort). v3 scores every
    admissible pair with the circuit. The quantum analogue of the
    prefilter is Grover/amplitude amplification over a pair-index
    register, which needs QRAM to load all patches in superposition -
    not simulable here, so it is documented as future work rather than
    faked. The honest cost of dropping it is O(N^2) circuit calls,
    which is why the default stride is coarser than v2's.
  - v2 decided which matches were real with a classical Hough-style
    (dy, dx) histogram vote. v3 encodes the pair's spatial offset into
    a third quantum register and lets trainable controlled rotations
    fold it into the ancilla readout (see the limitation below).
  - v2 cleaned its output mask with binary dilation/erosion. v3 does
    not: morphology is a classical CV operator.

Architecture
------------
1. Patch preparation: a raw patch is flattened and L2-normalized. That
   is state preparation, not feature extraction - no basis transform,
   no descriptor, no learned or hand-chosen projection.
2. Pair generation: all patch pairs at least `min_spatial_dist` apart.
3. Pure quantum scoring, one circuit per pair:
     - AmplitudeEmbedding loads all patch_size^2 pixels of patch A into
       register A and patch B into register B (needs exactly
       n_qubits = log2(patch_size^2) wires per register).
     - `n_layers` data re-uploading blocks apply RX/RY/RZ with angles
       `weight + pixel*pi` plus a CNOT ring, using the SAME (tied)
       weight tensor for both registers - the quantum analogue of a
       Siamese network's shared-weight distance metric.
     - The pair's (dy, dx) offset is angle-encoded into a small offset
       register with its own trainable rotations.
     - Swap test: H(ancilla) -> CSWAP(ancilla, a_k, b_k) for all k ->
       trainable CRY from each offset wire onto the ancilla -> H.
     - Readout is <Z> on the ancilla, clipped to [0, 1].
   IMPORTANT (why re-uploading is mandatory here): a plain
   AmplitudeEmbedding followed by a data-INDEPENDENT tied unitary U(w)
   is provably degenerate under a swap test -
   |<U(w)f_a|U(w)f_b>|^2 = |<f_a|f_b>|^2 for ANY U(w) - so the loss
   gradient w.r.t. w is exactly zero everywhere (confirmed empirically
   in v2 at n_qubits in {2,3,4}: grad norm ~1e-16, similarity
   bit-identical across independent random weight draws). Re-uploading
   the pixels INSIDE the rotation angles makes each register's unitary
   data-dependent, so there is no shared U(w) left to cancel. Keep this
   in mind before "simplifying" the ansatz - see SwapTestSiameseQNN.
4. Training: MSE on similarity labels over self-supervised synthetic
   copy-move pairs (an authentic patch duplicated to a new location,
   optionally rotated in {0,90,180,270} -> exact positive by
   construction; random non-matching locations -> negatives). CoMoFoD
   masks carry no patch CORRESPONDENCE, so supervised pair labels do
   not exist and this augmentation is what makes training possible.
   The ONLY trainable parameters are circuit rotation angles - there is
   no classical trainable layer anywhere in the model.
5. Inference: every pair is scored by the circuit, thresholded, and
   surviving pairs paint BOTH the source and target region into the
   mask. No vote, no morphology.

Known limitations (stated plainly, not glossed over)
----------------------------------------------------
  - The offset register is a WEAKER substitute for v2's classical
    vote, not an equivalent one. Offset consistency is a GLOBAL
    property of the pair ensemble ("many pairs share one offset"); a
    per-pair circuit cannot observe clustering it is not shown. A
    faithful quantum version needs quantum counting / amplitude
    estimation over all pairs at once, which again needs QRAM. Expect
    lower precision than v2 on homogeneous regions (sky, walls) - that
    is the real, measured price of purity, and the report should say
    so rather than hide it.
  - Offset handling is translation-only, as in v2. Rotated copy-move
    needs a dtheta component. (NOTE: in CoMoFoD, "CR" is Color
    Reduction post-processing, NOT rotation - rotated forgeries are
    identified by base scene id range, see get_geometry_type. BC/CA/
    CR/IB/JC/NA are post-processing categories applied independently
    of the geometry used to build the forgery.)
  - Dropping the classical prefilter makes inference O(N^2) in patch
    count. Use a coarser --stride; see the cost table in main().
  - No accuracy guarantees; this is architecture, not a tuned model.
    v3 is expected to score LOWER than the v2 hybrid. Purity is the
    goal here, not the leaderboard.
"""

import os
import sys
import json
import argparse
from io import BytesIO
import numpy as np
from pathlib import Path

import pennylane as qml
from pennylane import numpy as pnp

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ============================================================================
# DATASET LOADING (same contract as v1's CoMoFoDDataset)
# ============================================================================

class CoMoFoDDataset:
    """
    Load CoMoFoD dataset with ground truth masks.

    Supports two on-disk layouts:
      - "legacy": dataset_dir/{Forged,Masks,Original}/  (kept for back-compat,
        was never matched by any real CoMoFoD download)
      - "flat": the real CoMoFoD_small_v2 layout as distributed - a single
        flat directory of {id}_F.png, {id}_F_{CODE}{level}.{png,jpg},
        {id}_O.png, {id}_B.png (ground truth mask), etc. `dataset_dir` may
        point either directly at that flat folder or at its parent (a
        `CoMoFoD_small_v2/` subfolder is auto-detected).

    Ground truth: CoMoFoD_small_v2 ships two mask-shaped files per base
    image, `{id}_B.png` and `{id}_M.png`. Inspection shows `_B` is a clean
    near-binary mask ({0, 255} plus minor anti-alias, area consistent with
    readme.txt's documented forged-region sizes) while `_M` is a 200+ level
    grayscale image that is not a segmentation mask - so `_B` is used here.
    Post-processing (BC/CA/CR/IB/JC/NA) does not move the forged region, so
    one `_B` mask is shared across all post-processed variants of a base
    image.
    """

    # Real CoMoFoD post-processing codes (from the bundled readme.txt),
    # distinct from the geometric transformation used to build the forgery.
    TRANSFORM_MAP = {
        'BC': 'Brightness Change', 'CA': 'Contrast Adjustment',
        'CR': 'Color Reduction', 'IB': 'Image Blurring',
        'JC': 'JPEG Compression', 'NA': 'Noise Adding',
    }

    # Base scene id ranges (1-indexed, inclusive) documented in readme.txt
    # for the GEOMETRIC transformation used to construct the copy-move
    # forgery itself - independent of the post-processing code above.
    GEOMETRY_RANGES = [
        (1, 40, 'Translation'), (41, 80, 'Rotation'), (81, 120, 'Scaling'),
        (121, 159, 'Distortion'), (160, 200, 'Combination'),
    ]

    def __init__(self, dataset_dir: str, img_size: int = 256, grayscale: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.img_size = img_size
        self.grayscale = grayscale

        legacy_forged = self.dataset_dir / 'Forged'
        legacy_masks = self.dataset_dir / 'Masks'

        if legacy_forged.exists() and legacy_masks.exists():
            self.layout = 'legacy'
            self.flat_dir = None
            self.forged_dir = legacy_forged
            self.masks_dir = legacy_masks
            self.original_dir = self.dataset_dir / 'Original'
            if not self.original_dir.exists():
                self.original_dir = self.forged_dir

            self.image_files = sorted(list(self.forged_dir.glob('*.jpg')) +
                                       list(self.forged_dir.glob('*.png')))
            self.original_files = sorted(list(self.original_dir.glob('*.jpg')) +
                                          list(self.original_dir.glob('*.png')))
        else:
            self.layout = 'flat'
            self.flat_dir = self.dataset_dir
            nested = self.dataset_dir / 'CoMoFoD_small_v2'
            if nested.exists():
                self.flat_dir = nested

            if not self.flat_dir.exists():
                raise FileNotFoundError(
                    f"Could not find CoMoFoD dataset at {self.dataset_dir}\n"
                    "Expected either dataset_dir/{Forged,Masks}/ (legacy) or "
                    "a flat CoMoFoD_small_v2/ directory of {id}_F*.png files."
                )

            # Forged test images: base ({id}_F.ext) + post-processed variants
            # ({id}_F_{CODE}{level}.ext). Explicitly excludes {id}_O* (clean)
            # and {id}_B/{id}_M (mask-shaped files).
            self.image_files = sorted(
                p for p in list(self.flat_dir.glob('*_F.png')) +
                list(self.flat_dir.glob('*_F.jpg')) +
                list(self.flat_dir.glob('*_F_*.png')) +
                list(self.flat_dir.glob('*_F_*.jpg'))
            )
            # Clean base images only ({id}_O.png, no post-processing) - the
            # pool used for synthetic copy-move training pairs.
            self.original_files = sorted(self.flat_dir.glob('*_O.png'))
            self.masks_dir = self.flat_dir

            if not self.image_files:
                raise FileNotFoundError(
                    f"No '*_F*.png'/'*_F*.jpg' forged images found under {self.flat_dir}"
                )

        print(f"[INFO] Dataset layout: {self.layout}")
        print(f"[INFO] Found {len(self.image_files)} forged images")
        print(f"[INFO] Found {len(self.original_files)} images available for synthetic training")

    def __len__(self):
        return len(self.image_files)

    def _load_image(self, path):
        img = Image.open(path)
        if self.grayscale:
            img = img.convert('L')
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    def _mask_path_for(self, img_name: str):
        if self.layout == 'legacy':
            for ext in ('.png', '.jpg'):
                p = self.masks_dir / f"{img_name}{ext}"
                if p.exists():
                    return p
            return None

        # flat layout: mask is shared across all post-processed variants of
        # a base scene, e.g. "001_F_BC1" and "001_F" both use "001_B.png".
        base_id = img_name.split('_')[0]
        p = self.masks_dir / f"{base_id}_B.png"
        return p if p.exists() else None

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_name = img_path.stem
        img_array = self._load_image(img_path)

        mask_path = self._mask_path_for(img_name)
        if mask_path is None:
            return None, None, img_name

        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_array = (np.array(mask, dtype=np.uint8) > 128).astype(np.uint8)

        return img_array, mask_array, img_name

    def get_training_image(self, idx):
        """Authentic image for synthetic copy-move pair generation."""
        path = self.original_files[idx % len(self.original_files)]
        return self._load_image(path)

    @classmethod
    def get_geometry_type(cls, filename: str) -> str:
        """
        Geometric transformation used to CONSTRUCT the copy-move forgery
        (Translation/Rotation/Scaling/Distortion/Combination), looked up by
        base scene id per readme.txt. Independent of post-processing
        (BC/CA/CR/IB/JC/NA) - a "_CR1" post-processed image can be a
        Rotation-geometry forgery and vice versa; the two axes are not the
        same thing despite the confusable "CR" vs "rotation" naming.
        """
        base_id_str = filename.split('_')[0]
        try:
            base_id = int(base_id_str)
        except ValueError:
            return 'Unknown'
        for lo, hi, name in cls.GEOMETRY_RANGES:
            if lo <= base_id <= hi:
                return name
        return 'Unknown'

    @classmethod
    def get_transformation_type(cls, filename: str) -> str:
        """
        Post-processing category applied AFTER the copy-move forgery was
        constructed: Brightness/Contrast/ColorReduction/Blur/JPEG/Noise, per
        the real CoMoFoD_small_v2 naming ({id}_F_{CODE}{level}). "Raw" means
        no post-processing ({id}_F with no suffix). NOTE: this is NOT the
        geometric transformation (translation/rotation/scaling/...) used to
        build the forgery itself - see get_geometry_type() for that axis.
        """
        parts = filename.split('_')
        if len(parts) == 2 and parts[1] == 'F':
            return 'Raw'
        if len(parts) >= 3:
            code = parts[-1].rstrip('0123456789')
            return cls.TRANSFORM_MAP.get(code, 'Unknown')
        return 'Unknown'


# ============================================================================
# QUANTUM STATE PREPARATION (replaces v2's classical DCT + histogram
# descriptor - see module docstring)
# ============================================================================

def qubits_for_patch(patch_size: int) -> int:
    """
    Wires needed per register to amplitude-encode a patch_size^2 patch.
    Amplitude encoding packs 2^n real amplitudes into n qubits, so a
    16x16 patch (256 px) needs exactly 8 qubits - the reason patch sizes
    that are powers of two are strongly preferred here.
    """
    n_px = patch_size * patch_size
    n_q = int(np.ceil(np.log2(n_px)))
    return max(1, n_q)


def prepare_patch_state(patch: np.ndarray) -> np.ndarray:
    """
    Raw patch -> normalized amplitude vector, zero-padded to 2^n.

    This is STATE PREPARATION, not feature extraction: the pixels are
    flattened and L2-normalized and nothing else. No basis transform, no
    descriptor, no hand-chosen projection - that distinction is the whole
    point of v3 (v2 fed the circuit DCT coefficients + a histogram here).

    A flat/black patch has zero norm and cannot be normalized into a
    quantum state, so it falls back to the uniform superposition, which
    is the correct "no information" state for it.
    """
    flat = np.asarray(patch, dtype=np.float64).ravel()
    dim = 2 ** qubits_for_patch(int(np.sqrt(len(flat))))
    if len(flat) < dim:
        flat = np.pad(flat, (0, dim - len(flat)), mode='constant')

    norm = np.linalg.norm(flat)
    if norm < 1e-9:
        return np.full(dim, 1.0 / np.sqrt(dim))
    return flat / norm


def extract_all_patches(image: np.ndarray, patch_size: int, stride: int):
    """
    Sliding-window patch extraction. Returns (coords, states) where each
    state is the amplitude vector fed straight to AmplitudeEmbedding.
    Kept the same windowing as v2 so masks stay comparable.
    """
    H, W = image.shape
    coords, states = [], []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            coords.append((y, x))
            states.append(prepare_patch_state(patch))
    return coords, np.array(states, dtype=np.float64)


# ============================================================================
# PAIR GENERATION - exhaustive (v2's classical prefilter removed)
# ============================================================================

def generate_all_pairs(coords, min_spatial_dist: int = 24):
    """
    Every patch pair at least `min_spatial_dist` apart, scored by the
    circuit. v2 pre-filtered these with a classical lexicographic sort
    over DCT descriptors (Fridrich et al.); that is a classical matching
    algorithm doing the detector's job before the circuit ever runs, so
    v3 removes it.

    The quantum replacement would be Grover / amplitude amplification
    over a pair-index register, giving back the O(sqrt(N)) advantage -
    but that needs QRAM to hold all patches in superposition and is not
    simulable at this scale, so it stays documented as future work
    instead of being faked with a classical stand-in.

    The honest consequence: this is O(N^2) circuit evaluations. Patch
    count is set by --stride, so stride is now the main cost dial.

    min_spatial_dist still drops trivially-adjacent pairs - a patch
    always resembles its immediate neighbour and that is not a
    copy-move. It is a geometric admissibility rule on coordinates, not
    a similarity judgement, so it carries no classical matching signal.
    """
    n = len(coords)
    pairs = []
    for i in range(n):
        yi, xi = coords[i]
        for j in range(i + 1, n):
            yj, xj = coords[j]
            if np.hypot(yi - yj, xi - xj) < min_spatial_dist:
                continue
            pairs.append((i, j))
    return pairs


# ============================================================================
# SIAMESE SWAP-TEST QNN
# ============================================================================

def encode_offset(dy: float, dx: float, img_size: int) -> np.ndarray:
    """
    Pair spatial offset -> angles in [-1, 1] for the offset register.

    Sign is folded away because a copy-move offset is symmetric: pair
    (i, j) and (j, i) describe the same duplication event, so the circuit
    must not learn to prefer one direction.
    """
    return np.array([abs(dy) / img_size, abs(dx) / img_size], dtype=np.float64)


class SwapTestSiameseQNN:
    """
    Pure quantum pair scorer: two amplitude-encoded patch registers, a
    tied re-uploading ansatz, a small offset register, and a swap-test
    ancilla readout. Every trainable parameter in the model is a rotation
    angle in this circuit - there is no classical trainable layer.

    WHY THE ANSATZ LOOKS LIKE THIS (do not "simplify" it away)
    ----------------------------------------------------------
    AmplitudeEmbedding followed by a data-INDEPENDENT tied unitary U(w)
    is provably degenerate under a swap test:
        |<U(w)f_a|U(w)f_b>|^2 = |<f_a|U(w)^dag U(w)|f_b>|^2 = |<f_a|f_b>|^2
    for ANY unitary U(w). The output - and therefore the gradient w.r.t.
    w - is exactly constant, confirmed empirically in v2 (grad norm
    ~1e-16 at n_qubits in {2,3,4}, similarity bit-identical across
    independent random weight draws). The fix, kept here, is data
    re-uploading (Perez-Salinas et al., 2020): the pixels are folded
    back INTO the rotation angles at every layer, so each register's
    unitary is data-dependent and nothing cancels. Weights stay tied -
    one weight tensor drives both registers, which is exactly the
    shared-weight property that makes a Siamese network a Siamese
    network.

    v3 additionally amplitude-encodes the raw pixels first, so the
    circuit sees the whole patch (all patch_size^2 values) rather than
    the handful of pixels the re-upload layers have room for. The
    embedding supplies the information; the re-upload layers supply the
    trainability.
    """

    @staticmethod
    def _pick_device(n_wires, requested):
        """
        lightning.qubit (C++ statevector) is 3-7x faster than default.qubit
        here and it matters: pair scoring is exhaustive in v3, so the
        per-circuit cost is multiplied by tens of thousands of pairs.
        lightning has no 'backprop', so it pairs with 'adjoint' - which is
        also the more hardware-honest gradient of the two.
        """
        if requested == 'auto':
            try:
                return qml.device('lightning.qubit', wires=n_wires), 'adjoint'
            except Exception:
                return qml.device('default.qubit', wires=n_wires), 'backprop'
        if requested in ('adjoint', 'parameter-shift'):
            try:
                return qml.device('lightning.qubit', wires=n_wires), requested
            except Exception:
                return qml.device('default.qubit', wires=n_wires), requested
        return qml.device('default.qubit', wires=n_wires), requested

    def __init__(self, patch_size: int = 8, n_layers: int = 3,
                 n_offset_qubits: int = 2, img_size: int = 256,
                 diff_method: str = 'auto', seed: int = None):
        self.patch_size = patch_size
        self.n_qubits = qubits_for_patch(patch_size)
        self.n_layers = n_layers
        self.n_offset_qubits = n_offset_qubits
        self.img_size = img_size
        self.diff_method = diff_method
        self.state_dim = 2 ** self.n_qubits

        self.reg_a = list(range(0, self.n_qubits))
        self.reg_b = list(range(self.n_qubits, 2 * self.n_qubits))
        self.reg_off = list(range(2 * self.n_qubits,
                                  2 * self.n_qubits + n_offset_qubits))
        self.ancilla = 2 * self.n_qubits + n_offset_qubits
        self.n_wires = self.ancilla + 1

        # Amplitudes of a normalized 2^n vector are O(1/sqrt(2^n)), which
        # would make the re-upload angles vanishingly small and the data
        # dependence (the thing keeping gradients alive) numerically
        # negligible. Rescaling by sqrt(dim) puts a mid-grey pixel near
        # 1 rad. Angles wrap mod 2*pi, so overshoot on bright pixels is
        # harmless.
        self.reupload_scale = np.sqrt(self.state_dim)

        self.dev, self.diff_method = self._pick_device(self.n_wires, diff_method)

        rng = pnp.random.default_rng(seed) if seed is not None else pnp.random
        self.weights = pnp.array(
            rng.uniform(0, 2 * np.pi, (n_layers, self.n_qubits, 3)),
            requires_grad=True)
        # Row 0: offset encoding bias (random). Row 1: CRY readout coupling,
        # initialized NEAR ZERO on purpose. The CRY gates sit inside the
        # swap-test interferometer, so a random coupling partially decoheres
        # the ancilla and caps the output: at random init, two IDENTICAL
        # patches scored only ~0.32 instead of 1.0, meaning the model could
        # not reach its own positive label and training started above the
        # constant-predictor floor. Starting at CRY ~ identity makes the
        # circuit a clean swap test (identical patches -> 1.0) that then
        # LEARNS how much offset coupling to introduce.
        self.offset_weights = pnp.array(
            np.concatenate([
                rng.uniform(0, 2 * np.pi, (1, n_offset_qubits)),
                rng.uniform(-0.05, 0.05, (1, n_offset_qubits)),
            ]),
            requires_grad=True)

        self.qnode = qml.QNode(self._circuit, self.dev, diff_method=self.diff_method)

    def _reupload_angle(self, state, idx):
        """
        One re-upload angle from the encoded state, cycling with modulo so
        any patch_size / n_layers combination has enough values.

        Indexes the LAST axis so a batched state of shape (B, dim) works
        as well as a single (dim,) vector - that is what lets the QNode be
        parameter-broadcast over many pairs at once, which is the
        difference between usable and unusable inference speed here.
        """
        state = pnp.asarray(state)
        return state[..., idx % state.shape[-1]] * self.reupload_scale * np.pi

    def _reupload_block(self, weights, wires, state):
        """Tied weights + that register's own re-uploaded data."""
        idx = 0
        for layer in range(self.n_layers):
            for k, wire in enumerate(wires):
                a_x = weights[layer, k, 0] + self._reupload_angle(state, idx); idx += 1
                a_y = weights[layer, k, 1] + self._reupload_angle(state, idx); idx += 1
                a_z = weights[layer, k, 2] + self._reupload_angle(state, idx); idx += 1
                qml.RX(a_x, wires=wire)
                qml.RY(a_y, wires=wire)
                qml.RZ(a_z, wires=wire)
            for k, wire in enumerate(wires):
                qml.CNOT(wires=[wire, wires[(k + 1) % len(wires)]])

    def _circuit(self, state_a, state_b, offset, weights, offset_weights):
        # 1. Load the RAW patches (all patch_size^2 pixels, no descriptor).
        qml.AmplitudeEmbedding(state_a, wires=self.reg_a,
                               normalize=True, pad_with=0.0)
        qml.AmplitudeEmbedding(state_b, wires=self.reg_b,
                               normalize=True, pad_with=0.0)

        # 2. Tied, data-re-uploading variational blocks (see class docstring).
        self._reupload_block(weights, self.reg_a, state_a)
        self._reupload_block(weights, self.reg_b, state_b)

        # 3. Encode the pair's spatial offset - the in-circuit stand-in for
        #    v2's classical (dy, dx) Hough vote.
        offset = pnp.asarray(offset)
        for k, wire in enumerate(self.reg_off):
            qml.RY(offset_weights[0, k]
                   + offset[..., k % offset.shape[-1]] * np.pi,
                   wires=wire)

        # 4. Swap test with a trainable, offset-conditioned readout. The
        #    CRY gates sit INSIDE the interferometer, so the offset
        #    register modulates the ancilla fringe rather than being read
        #    out separately. This makes the output a learned function of
        #    (patch A, patch B, offset) instead of a textbook overlap -
        #    which is why it is no longer guaranteed to land in [0, 1] and
        #    similarity() clips.
        qml.Hadamard(wires=self.ancilla)
        for a, b in zip(self.reg_a, self.reg_b):
            qml.CSWAP(wires=[self.ancilla, a, b])
        for k, wire in enumerate(self.reg_off):
            qml.CRY(offset_weights[1, k], wires=[wire, self.ancilla])
        qml.Hadamard(wires=self.ancilla)

        return qml.expval(qml.PauliZ(self.ancilla))

    def similarity(self, state_a, state_b, offset, weights=None,
                   offset_weights=None):
        """
        Learned match score(s) in [0, 1]. 1 = confident copy-move pair.

        Pass single-pair 1-D state_a/state_b/offset for one score (returns
        a float), or stacked 2-D arrays (leading batch dim) to score many
        pairs in ONE QNode call via PennyLane parameter broadcasting
        (returns an array). Batching measured ~1.3x faster than the same
        number of single-pair calls on lightning.qubit, bit-identical
        output - a real but modest win, since lightning still evaluates
        the batch dimension internally rather than truly vectorizing it.
        """
        w = self.weights if weights is None else weights
        wo = self.offset_weights if offset_weights is None else offset_weights
        val = np.clip(np.asarray(self.qnode(state_a, state_b, offset, w, wo),
                                  dtype=float), 0.0, 1.0)
        return float(val) if val.ndim == 0 else val

    # ---- persistence -------------------------------------------------- #
    # v3 has two weight tensors plus architecture metadata, so checkpoints
    # are .npz. v2's single-array .npy checkpoints are NOT loadable - the
    # circuit shape changed and the old weights are meaningless here.
    def save(self, path):
        np.savez(path,
                 weights=np.array(self.weights),
                 offset_weights=np.array(self.offset_weights),
                 patch_size=self.patch_size,
                 n_qubits=self.n_qubits,
                 n_layers=self.n_layers,
                 n_offset_qubits=self.n_offset_qubits,
                 img_size=self.img_size)

    def load(self, path):
        path = str(path)
        if path.endswith('.npy'):
            raise ValueError(
                f"{path} is a v2 (hybrid) checkpoint. v3 is a different "
                "circuit - amplitude-encoded raw pixels, an offset register "
                "and a different weight shape - so v2 weights cannot be "
                "reused. Retrain with --mode train.")
        data = np.load(path, allow_pickle=False)
        self.weights = pnp.array(data['weights'], requires_grad=True)
        self.offset_weights = pnp.array(data['offset_weights'], requires_grad=True)

        ckpt_patch = int(data['patch_size'])
        ckpt_layers = int(data['n_layers'])
        if ckpt_patch != self.patch_size or ckpt_layers != self.n_layers:
            raise ValueError(
                f"Checkpoint architecture mismatch: checkpoint was trained "
                f"with patch_size={ckpt_patch}, n_layers={ckpt_layers} but "
                f"this model was built with patch_size={self.patch_size}, "
                f"n_layers={self.n_layers}.")


# ============================================================================
# SYNTHETIC TRAINING PAIR GENERATION (self-supervised - solves the
# "CoMoFoD masks have no patch correspondence" problem)
# ============================================================================

def _jpeg_roundtrip(patch: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode a [0,1] float patch through JPEG at `quality` and back -
    the artifact real CoMoFoD JC-category post-processing leaves."""
    u8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(u8, mode='L').save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert('L'), dtype=np.float64) / 255.0


def _degrade_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly apply brightness/contrast shift, sensor noise, and/or JPEG
    re-compression - the artifacts CoMoFoD's BC/CA/NA/JC post-processing
    categories leave on a real forged region. Without this, a pasted
    synthetic positive is pixel-identical to its source (see
    make_synthetic_pairs docstring history), which no real copy-move ever
    is - training on exact duplicates only teaches "spot an exact copy,"
    not "spot a duplicate that also got compressed/blended like a real
    forgery." Each effect is independently a coin flip so plenty of
    positives stay clean too (real CoMoFoD includes un-post-processed
    "Raw" forgeries alongside the post-processed categories).
    """
    out = patch.astype(np.float64).copy()
    if rng.random() < 0.6:
        contrast = rng.uniform(0.85, 1.15)
        brightness = rng.uniform(-0.06, 0.06)
        out = np.clip(out * contrast + brightness, 0.0, 1.0)
    if rng.random() < 0.5:
        out = np.clip(out + rng.normal(0.0, rng.uniform(0.01, 0.03), out.shape), 0.0, 1.0)
    if rng.random() < 0.5:
        out = _jpeg_roundtrip(out, quality=int(rng.integers(35, 90)))
    return out


def _feather_paste(patch: np.ndarray, background: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Alpha-blend `patch` onto `background` with a soft edge instead of a
    hard rectangular cut - real copy-move paste boundaries are usually
    feathered or blurred, not a perfect seam. 30% of positives keep a
    hard edge, since some real forgeries do too.
    """
    if rng.random() < 0.3:
        return patch
    h, w = patch.shape
    feather = max(1, int(round(min(h, w) * rng.uniform(0.08, 0.2))))
    yy, xx = np.mgrid[0:h, 0:w]
    dist_to_edge = np.minimum.reduce([yy, xx, h - 1 - yy, w - 1 - xx]).astype(np.float64)
    mask = np.clip(dist_to_edge / feather, 0.0, 1.0)
    return mask * patch + (1.0 - mask) * background


def make_synthetic_pairs(image: np.ndarray, patch_size: int, n_pos: int = 6, n_neg: int = 6,
                          rng: np.random.Generator = None, rotate: bool = False):
    """
    Paste a real patch (optionally rotated) into a NEW location of a copy
    of `image`, then extract features from that pasted region - a genuine
    relocated copy-move pair by construction, not just a patch compared to
    its own in-place rotation. Sample random non-matching patches as
    negatives. Returns (features_list_a, features_list_b, labels).

    Each pair also carries its spatial offset, which the offset register
    in the circuit consumes. Positives get the true (dy, dx) of the
    duplication; negatives get the offset of the two sampled locations.

    Positives are run through _degrade_patch() and _feather_paste() before
    being written into the forged copy, so f_b is NOT pixel-identical to
    f_a's source content the way a bare paste would be - see those two
    functions for why. This directly targets the "positives were too
    easy" gap: real CoMoFoD forgeries carry post-processing and blended
    boundaries that a hard, undegraded paste never reproduces.

    Negatives get the same care, for three reasons found once positives
    stopped being trivial:
      - Unlike positives, a naive random (y2, x2) has no separation
        guarantee from (y1, x1) and can land close enough to overlap -
        an accidental near-duplicate mislabeled 0, contradicting the
        positive signal. Negatives now re-roll under the same
        overlap-avoidance rule positives use.
      - Only degrading positives makes "looks a bit noisy/JPEG'd" a
        free giveaway for label 1 regardless of actual content match.
        Negatives now get _degrade_patch() at the same rate, so
        degradation carries no label information.
      - Two fully random patches (e.g. sky vs. brick) are usually
        trivially different in gross statistics, which risks the model
        learning "compare average brightness" instead of comparing
        content. Half of negatives are now "hard": the second location
        is chosen, among several candidates, to have brightness closest
        to the first patch's - a different patch that merely looks
        similar, not an easy outlier.
    """
    rng = rng or np.random.default_rng()
    H, W = image.shape
    pos_pairs, neg_pairs = [], []

    def rand_loc():
        y = rng.integers(0, H - patch_size)
        x = rng.integers(0, W - patch_size)
        return y, x

    def not_overlapping(y1, x1, y2, x2):
        return not (abs(y2 - y1) < patch_size and abs(x2 - x1) < patch_size)

    for _ in range(n_pos):
        y1, x1 = rand_loc()
        src_patch = image[y1:y1 + patch_size, x1:x1 + patch_size].copy()
        # Rotation is OFF by default, and that is a deliberate correction,
        # not a regression. A swap test compares amplitude vectors, so a
        # 90-degree rotation permutes the amplitudes into a nearly
        # orthogonal state: measured on the untrained circuit, an exact
        # rot90 duplicate scores ~0.01 while an exact copy scores ~1.0.
        # Labelling those pairs 1.0 asks the model to fit a target its
        # architecture provably cannot represent (offset handling here is
        # translation-only), which pins training loss ABOVE the 0.25
        # constant-predictor floor - exactly the non-convergence the v2
        # model was reported to have. Turn back on only together with a
        # dtheta component in the offset register.
        rot_k = rng.choice([0, 1, 2, 3]) if rotate else 0
        src_patch_rot = np.rot90(src_patch, k=rot_k)
        if src_patch_rot.shape != (patch_size, patch_size):
            src_patch_rot = src_patch  # guard for non-square edge cases

        y2, x2 = rand_loc()
        while not not_overlapping(y1, x1, y2, x2):
            y2, x2 = rand_loc()

        background = image[y2:y2 + patch_size, x2:x2 + patch_size]
        pasted = _degrade_patch(src_patch_rot, rng) if rng.random() < 0.8 else src_patch_rot
        pasted = _feather_paste(pasted, background, rng)

        forged = image.copy()
        forged[y2:y2 + patch_size, x2:x2 + patch_size] = pasted

        f_a = prepare_patch_state(src_patch)
        f_b = prepare_patch_state(forged[y2:y2 + patch_size, x2:x2 + patch_size])
        off = encode_offset(y2 - y1, x2 - x1, H)
        pos_pairs.append((f_a, f_b, off, 1.0))

    for _ in range(n_neg):
        y1, x1 = rand_loc()
        p1 = image[y1:y1 + patch_size, x1:x1 + patch_size]

        if rng.random() < 0.5:
            # Hard negative: pick whichever of several candidate locations
            # has brightness closest to p1's, so the pair looks similar at
            # a glance despite being genuinely different content.
            candidates = [rand_loc() for _ in range(8)]
            candidates = [(y, x) for (y, x) in candidates
                          if not_overlapping(y1, x1, y, x)] or [rand_loc()]
            p1_mean = p1.mean()
            y2, x2 = min(
                candidates,
                key=lambda yx: abs(
                    image[yx[0]:yx[0] + patch_size, yx[1]:yx[1] + patch_size].mean()
                    - p1_mean))
        else:
            y2, x2 = rand_loc()
            while not not_overlapping(y1, x1, y2, x2):
                y2, x2 = rand_loc()

        p2 = image[y2:y2 + patch_size, x2:x2 + patch_size]
        # Same degradation rate as positives (see docstring) so noise/JPEG
        # artifacts stop being a shortcut signal for the label.
        p2 = _degrade_patch(p2, rng) if rng.random() < 0.8 else p2

        f_a = prepare_patch_state(p1)
        f_b = prepare_patch_state(p2)
        off = encode_offset(y2 - y1, x2 - x1, H)
        neg_pairs.append((f_a, f_b, off, 0.0))

    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)
    return all_pairs


# ============================================================================
# TRAINING - image-level batching across epochs (same batching pattern as v1)
# ============================================================================

def train(qnn: SwapTestSiameseQNN, dataset: CoMoFoDDataset, epochs: int = 10,
          images_per_epoch: int = 20, pairs_per_image: int = 12, patch_size: int = 16,
          lr: float = 0.1, checkpoint_dir: str = './checkpoints', seed: int = 42,
          rotate: bool = False):
    """
    Both trainable tensors are circuit rotation angles - `weights` (the
    tied re-uploading ansatz) and `offset_weights` (the offset register
    and its readout coupling). There is no classical trainable parameter
    anywhere, which is what makes this a pure QNN rather than a hybrid.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    rng = np.random.default_rng(seed)

    history = []
    for epoch in range(epochs):
        epoch_losses = []
        img_indices = rng.integers(0, len(dataset.original_files), size=images_per_epoch)

        pbar = tqdm(img_indices, desc=f"Epoch {epoch+1}/{epochs}")
        for img_idx in pbar:
            image = dataset.get_training_image(int(img_idx))
            n_pos = pairs_per_image // 2
            n_neg = pairs_per_image - n_pos
            batch = make_synthetic_pairs(image, patch_size, n_pos=n_pos, n_neg=n_neg,
                                         rng=rng, rotate=rotate)

            def cost(weights, offset_weights, batch=batch):
                total = 0.0
                for f_a, f_b, off, label in batch:
                    pred = qnn.qnode(f_a, f_b, off, weights, offset_weights)
                    total = total + (pred - label) ** 2
                return total / len(batch)

            (qnn.weights, qnn.offset_weights), loss_val = opt.step_and_cost(
                cost, qnn.weights, qnn.offset_weights)
            epoch_losses.append(float(loss_val))
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=f"{float(loss_val):.4f}")

        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        print(f"[Epoch {epoch+1}/{epochs}] mean loss = {mean_loss:.4f}")

        ckpt_path = Path(checkpoint_dir) / f"swaptest_qnn_epoch{epoch+1}.npz"
        qnn.save(ckpt_path)
        print(f"[INFO] Checkpoint saved: {ckpt_path}")

    with open(Path(checkpoint_dir) / 'history.json', 'w') as f:
        json.dump(history, f)
    return history


# ============================================================================
# INFERENCE: pairwise scoring + spatial offset-consistency filtering
# ============================================================================

def detect_copy_move(image: np.ndarray, qnn: SwapTestSiameseQNN, patch_size: int = 16,
                      stride: int = 16, sim_threshold: float = 0.8,
                      verbose: bool = True, batch_size: int = 256):
    """
    Full pure-quantum pipeline for one test image -> binary forgery mask.

    Every stage that used to be classical is gone: no DCT descriptor, no
    lexicographic prefilter, no (dy, dx) Hough vote, no morphology. What
    remains is state preparation, circuit evaluation, and a threshold on
    the measured expectation value.

    Pairs are scored `batch_size` at a time in one QNode call each (see
    SwapTestSiameseQNN.similarity) instead of one call per pair -
    measured ~1.3x faster on lightning.qubit, bit-identical output.
    """
    H, W = image.shape
    coords, states = extract_all_patches(image, patch_size, stride)
    pairs = generate_all_pairs(coords, min_spatial_dist=patch_size)

    if verbose:
        print(f"  [INFO] {len(coords)} patches -> {len(pairs)} pairs "
              f"(exhaustive; v2's classical prefilter removed)")

    matched_pairs = []
    pbar = tqdm(total=len(pairs), desc="  Quantum pair scoring", disable=not verbose)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        offs = np.array([encode_offset(coords[j][0] - coords[i][0],
                                        coords[j][1] - coords[i][1], H)
                          for (i, j) in chunk])
        states_a = states[[i for (i, j) in chunk]]
        states_b = states[[j for (i, j) in chunk]]
        sims = np.atleast_1d(qnn.similarity(states_a, states_b, offs))
        for (i, j), sim in zip(chunk, sims):
            if sim >= sim_threshold:
                matched_pairs.append((i, j, float(sim)))
        pbar.update(len(chunk))
    pbar.close()

    if verbose:
        print(f"  [INFO] {len(matched_pairs)} pairs above threshold {sim_threshold} "
              f"(circuit output is the only decision signal)")

    mask = np.zeros((H, W), dtype=np.uint8)
    for (i, j, sim) in matched_pairs:
        for idx in (i, j):
            y, x = coords[idx]
            mask[y:y + patch_size, x:x + patch_size] = 1

    return mask.astype(np.uint8), matched_pairs, coords


# ============================================================================
# METRICS (unchanged from v1)
# ============================================================================

def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    tp = int(np.sum((pred_mask == 1) & (gt_mask == 1)))
    fp = int(np.sum((pred_mask == 1) & (gt_mask == 0)))
    fn = int(np.sum((pred_mask == 0) & (gt_mask == 1)))
    tn = int(np.sum((pred_mask == 0) & (gt_mask == 0)))
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
            'accuracy': accuracy, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CMFD v3 - PURE Siamese Swap-Test QNN (no classical stages)')
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='Path to a .npz weights file (required for --mode test). '
                              'v2 .npy checkpoints are not loadable - retrain.')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_pure')
    parser.add_argument('--patch-size', type=int, default=8,
                         help='Powers of two only; sets qubits per register to '
                              'log2(patch_size^2). 8->6q (15 wires, ~0.036 s/pair), '
                              '16->8q (19 wires, ~0.82 s/pair - 23x slower, since '
                              'simulation cost is exponential in total wires).')
    parser.add_argument('--img-size', type=int, default=128,
                         help='Working resolution. Lowered from 256 to 128 so that '
                              'FULL TILING (stride == patch-size) stays affordable: '
                              'exhaustive pair scoring is O(patches^2), and 256px '
                              'tiled is ~520k pairs (~5 h/image) vs ~33k (~20 min) '
                              'at 128px.')
    parser.add_argument('--stride', type=int, default=8,
                         help='Set this EQUAL to --patch-size for full coverage. '
                              'Larger strides leave gaps: at 256px/stride 32 only 3 of '
                              '64 patches landed inside a 2%%-area forged region and '
                              'forged vs authentic images scored identically. Cost is '
                              'O(patches^2): 128px tiled -> 256 patches / ~33k pairs / '
                              '~20 min per image.')
    parser.add_argument('--n-layers', type=int, default=3,
                         help='Data re-uploading blocks. More = more expressive '
                              'and proportionally slower.')
    parser.add_argument('--n-offset-qubits', type=int, default=2,
                         help='Wires for the in-circuit offset register.')
    parser.add_argument('--diff-method', type=str, default='auto',
                         choices=['auto', 'adjoint', 'backprop', 'parameter-shift'],
                         help="'auto' uses lightning.qubit + adjoint when available "
                              "(fastest), else default.qubit + backprop. 'backprop' "
                              "is simulator-only. 'parameter-shift' is the "
                              "hardware-realizable gradient but costs 2 circuit "
                              "evals PER PARAMETER - far slower.")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--images-per-epoch', type=int, default=20)
    parser.add_argument('--pairs-per-image', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--sim-threshold', type=float, default=0.8)
    parser.add_argument('--batch-size', type=int, default=256,
                         help='Pairs scored per QNode call (test mode only) via '
                              'PennyLane parameter broadcasting. Measured ~1.3x '
                              'faster than one call per pair on lightning.qubit, '
                              'bit-identical output. Higher = fewer Python-call '
                              'overheads but more memory per call.')
    parser.add_argument('--train-rotations', action='store_true',
                         help='Include 90/180/270-rotated positives in the '
                              'synthetic training pairs. OFF by default: the '
                              'circuit is translation-only, so rotated pairs are '
                              'unfittable targets that hold loss above the 0.25 '
                              'constant-predictor floor. See make_synthetic_pairs.')
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='./results_pure')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    dataset = CoMoFoDDataset(args.dataset_dir, img_size=args.img_size, grayscale=True)
    qnn = SwapTestSiameseQNN(patch_size=args.patch_size, n_layers=args.n_layers,
                             n_offset_qubits=args.n_offset_qubits,
                             img_size=args.img_size,
                             diff_method=args.diff_method)
    if args.stride > args.patch_size:
        print(f"[WARN] stride ({args.stride}) > patch-size ({args.patch_size}): patches "
              f"do NOT tile, so {100 * args.patch_size / args.stride:.0f}% of each axis is "
              f"never examined. Measured consequence at 256px/stride 32: only 3 of 64 "
              f"patches fell inside a 2%-area forged region, and forged vs authentic "
              f"images produced IDENTICAL output. Use --stride == --patch-size for "
              f"full coverage.")
    n_params = int(np.prod(qnn.weights.shape) + np.prod(qnn.offset_weights.shape))
    print(f"[INFO] PURE QNN: {qnn.n_wires} wires "
          f"({qnn.n_qubits}/register x2 + {args.n_offset_qubits} offset + 1 ancilla), "
          f"{n_params} trainable rotation angles (tied across both registers). "
          f"No classical trainable parameters.")
    print(f"[INFO] Device: {qnn.dev.name}, diff_method: {qnn.diff_method}")

    if args.mode == 'train':
        train(qnn, dataset, epochs=args.epochs, images_per_epoch=args.images_per_epoch,
              pairs_per_image=args.pairs_per_image, patch_size=args.patch_size,
              lr=args.lr, checkpoint_dir=args.checkpoint_dir, seed=args.seed,
              rotate=args.train_rotations)
        return

    # test mode
    if not args.checkpoint:
        print("[ERROR] --checkpoint is required for --mode test")
        sys.exit(1)
    qnn.load(args.checkpoint)
    print(f"[INFO] Loaded weights from {args.checkpoint}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    num_test = min(args.num_images, len(dataset))
    for idx in range(num_test):
        img, gt_mask, img_name = dataset[idx]
        if img is None:
            continue
        print(f"\n--- {img_name} ---")
        pred_mask, matched_pairs, coords = detect_copy_move(
            img, qnn, patch_size=args.patch_size, stride=args.stride,
            sim_threshold=args.sim_threshold, batch_size=args.batch_size)
        metrics = compute_metrics(pred_mask, gt_mask)
        transform_type = CoMoFoDDataset.get_transformation_type(img_name)
        print(f"  {transform_type}: F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f}")
        results.append({'image_name': img_name, 'transformation': transform_type, **metrics})

        np.save(Path(args.output_dir) / f"{img_name}_pred_mask.npy", pred_mask)

    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(args.output_dir) / 'results_pure.csv', index=False)
    print(f"\n[INFO] Results saved to {args.output_dir}/results_pure.csv")


if __name__ == '__main__':
    main()
