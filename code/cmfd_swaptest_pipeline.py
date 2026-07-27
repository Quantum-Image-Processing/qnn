"""
CMFD Pipeline v2 - Siamese Swap-Test QNN for TRUE Copy-Move Detection
=======================================================================
This replaces the single-patch artifact classifier with a pairwise
matching architecture, which is what copy-move forgery detection
actually requires.

Key differences from the v1 (single-patch) pipeline:
  - v1: score(patch) -> forged probability   [structurally can't localize
        copy-move pairs at all - a per-patch classifier has no notion of
        "this patch was duplicated to that patch", regardless of which
        post-processing category (BC/CA/CR/IB/JC/NA) is applied]
  - v2: similarity(patch_i, patch_j) -> match probability, then
        spatial offset-consistency filtering to keep only genuine
        copy-move pairs and reject coincidental look-alikes
        (e.g. sky, walls, other homogeneous regions).

Architecture
------------
1. Feature extraction: same DCT + histogram descriptor as v1 (kept
   identical on purpose, per your request), one vector per patch.
2. Candidate pair generation: O(N log N) lexicographic block matching
   (Fridrich-style: sort patches by feature vector, only compare
   patches within a sliding window of the sorted order) instead of
   brute-force O(N^2) all-pairs comparison.
3. Siamese swap-test QNN: both patches in a candidate pair are encoded
   into two separate quantum registers via a data re-uploading ansatz
   (weights and each register's OWN classical features are combined
   additively inside the same rotation gates at every layer, using the
   SAME/tied weight tensor for both registers), then a swap test
   against an ancilla qubit is performed. The ancilla's <Z> expectation
   value equals the squared overlap |<psi_a|psi_b>|^2 of the two
   encoded states -> a learned similarity score in [0, 1]. This is the
   natural quantum analogue of a Siamese network's shared-weight
   distance metric. (An earlier fixed-embedding-then-tied-unitary
   version of this - AmplitudeEmbedding(data) followed by a
   data-independent U(w) - was provably degenerate: for a swap test,
   |<U(w)f_a|U(w)f_b>|^2 = |<f_a|f_b>|^2 for ANY U(w), so gradients
   w.r.t. w were exactly zero everywhere. Confirmed empirically before
   being replaced with the re-uploading form below - see
   SwapTestSiameseQNN's docstring.)
4. Training: uses PennyLane's parameter-shift differentiation
   (unchanged from v1) with contrastive-style MSE loss on similarity
   labels, batched per-image, iterated over epochs (same
   image-level batching pattern as v1's per-image processing loop).
   Ground-truth patch CORRESPONDENCE (which patch copied to where)
   does not exist in CoMoFoD masks, so training pairs are built with
   self-supervised synthetic copy-move augmentation: an authentic
   patch is duplicated to a new location (optionally rotated in
   {0,90,180,270}) -> exact positive pair by construction. Negative
   pairs are random non-matching patch locations.
5. Inference / localization: candidate pairs from a real forged
   test image are scored, thresholded, then filtered by spatial
   offset-consistency (a histogram vote over (dx, dy) offsets, kept
   from classical block-matching CMFD literature) to suppress
   isolated false matches. Surviving pairs mark BOTH the source and
   the target region in the output mask.

Known limitations (carried over honestly, not glossed over):
  - Offset-consistency filtering as implemented handles pure
    translation only. Rotated copy-move forgeries need a rotation
    parameter added to the offset vote (dx, dy, dtheta) - noted as a
    TODO with a hook left in `find_dominant_offsets`. (NOTE: in the
    real CoMoFoD dataset, "CR" denotes Color Reduction post-processing,
    NOT rotation - rotated forgeries are identified by base scene id
    range instead, see CoMoFoDDataset.get_geometry_type. BC/CA/CR/IB/
    JC/NA are all post-processing categories applied independently of
    which geometric transformation built the underlying forgery.)
  - Blocking reduces but does not eliminate false candidate pairs on
    homogeneous regions (sky, walls) - the offset-consistency step is
    what actually suppresses those, not the QNN alone.
  - No accuracy guarantees; this is architecture, not a tuned model.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pennylane as qml
from pennylane import numpy as pnp

from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import binary_dilation, binary_erosion, label as cc_label

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
# FEATURE EXTRACTION (identical to v1 - unchanged on purpose)
# ============================================================================

def extract_patch_features(patch: np.ndarray, target_dim: int = 24) -> np.ndarray:
    dct2 = dct(dct(patch, axis=0), axis=1)
    dct_flat = dct2.flatten()

    if len(dct_flat) >= 12:
        idx_sorted = np.argsort(np.abs(dct_flat))[-12:]
        dct_feat = dct_flat[idx_sorted]
    else:
        dct_feat = np.pad(dct_flat, (0, 12 - len(dct_flat)), mode='constant')

    hist, _ = np.histogram(patch, bins=12, range=(0, 1))
    hist_feat = hist.astype(np.float32)

    features = np.concatenate([dct_feat, hist_feat])
    if len(features) < target_dim:
        features = np.pad(features, (0, target_dim - len(features)), mode='constant')
    elif len(features) > target_dim:
        features = features[:target_dim]

    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features = features / norm
    return features


def extract_all_patches(image: np.ndarray, patch_size: int, stride: int):
    """Sliding window patch extraction - same windowing as v1's compute_heatmap."""
    H, W = image.shape
    coords, feats = [], []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            coords.append((y, x))
            feats.append(extract_patch_features(patch))
    return coords, np.array(feats, dtype=np.float32)


# ============================================================================
# CANDIDATE PAIR GENERATION - O(N log N) lexicographic block matching
# ============================================================================

def generate_candidate_pairs(coords, feats, window: int = 12, min_spatial_dist: int = 24):
    """
    Classical CMFD trick (Fridrich et al.) adapted as a pre-filter for the
    QNN: sort patches by feature vector, only pair patches that land near
    each other in sorted order. This turns O(N^2) all-pairs comparison
    into O(N * window), which is what makes this tractable on 256x256
    images with dense strides.

    min_spatial_dist discards trivially-adjacent pairs (a patch always
    looks similar to its immediate neighbour; that's not a copy-move).
    """
    n = len(coords)
    if n == 0:
        return []

    # Lexicographic sort key: round to reduce noise sensitivity, then sort.
    sort_key = np.lexsort(feats.round(3).T[::-1])
    pairs = set()
    for rank in range(n):
        i = sort_key[rank]
        for off in range(1, window + 1):
            rank2 = rank + off
            if rank2 >= n:
                break
            j = sort_key[rank2]
            yi, xi = coords[i]
            yj, xj = coords[j]
            spatial_dist = np.hypot(yi - yj, xi - xj)
            if spatial_dist < min_spatial_dist:
                continue
            pair = (min(i, j), max(i, j))
            pairs.add(pair)
    return list(pairs)


# ============================================================================
# SIAMESE SWAP-TEST QNN
# ============================================================================

class SwapTestSiameseQNN:
    """
    Shared-weight (tied) variational circuit applied to two quantum
    registers, followed by a swap test. <Z> on the ancilla equals the
    squared state overlap |<psi_a|psi_b>|^2 - a trainable similarity score.

    Embedding note (data re-uploading): an earlier version of this class
    used a fixed AmplitudeEmbedding(data) followed by a data-INDEPENDENT
    tied unitary U(w). That is provably degenerate for a swap-test readout:
    |<U(w)f_a|U(w)f_b>|^2 = |<f_a|U(w)^dagger U(w)|f_b>|^2 = |<f_a|f_b>|^2
    for ANY unitary U(w), so the swap test output (and therefore the loss
    gradient w.r.t. w) is exactly constant in w - confirmed empirically
    (grad norm ~1e-16 at n_qubits in {2,3,4}, similarity bit-identical
    across independent random weight draws). Fixed via data re-uploading
    (Perez-Salinas et al., "Data re-uploading for a universal quantum
    classifier", 2020): weights and each register's OWN classical features
    are combined additively INSIDE the same rotation gates at every layer,
    so there is no data-independent U(w) shared between the two branches
    for the swap test to cancel against - the weights remain tied (the
    same weight tensor parametrizes both registers) but now have a
    genuine, non-cancelling effect on the measured overlap.
    """

    def __init__(self, n_qubits: int = 5, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reg_a = list(range(0, n_qubits))
        self.reg_b = list(range(n_qubits, 2 * n_qubits))
        self.ancilla = 2 * n_qubits
        self.n_wires = 2 * n_qubits + 1

        self.dev = qml.device('default.qubit', wires=self.n_wires)
        self.weights = pnp.random.uniform(
            0, 2 * np.pi, (n_layers, n_qubits, 3), requires_grad=True
        )
        self.qnode = qml.QNode(self._circuit, self.dev, diff_method='parameter-shift')

    @staticmethod
    def _feature_angle(features, idx):
        """Angle-encode one (already L2-normalized) feature component,
        cycling through the vector via modulo so any n_qubits/n_layers
        combination can draw as many re-upload angles as it needs."""
        return float(features[idx % len(features)]) * np.pi

    def _reupload_block(self, weights, wires, features):
        """
        Tied weights, re-uploaded data: this exact weight tensor is used
        for BOTH registers, but at every layer the rotation angles are
        `weight + (that register's own feature angle)`, so reg_a and reg_b
        apply genuinely different gates (see class docstring for why the
        additive/interleaved form - not a post-embedding-only U(w) - is
        required to avoid the swap test cancelling the weight dependence).
        """
        idx = 0
        for layer in range(self.n_layers):
            for k, wire in enumerate(wires):
                a_x = weights[layer, k, 0] + self._feature_angle(features, idx); idx += 1
                a_y = weights[layer, k, 1] + self._feature_angle(features, idx); idx += 1
                a_z = weights[layer, k, 2] + self._feature_angle(features, idx); idx += 1
                qml.RX(a_x, wires=wire)
                qml.RY(a_y, wires=wire)
                qml.RZ(a_z, wires=wire)
            for k, wire in enumerate(wires):
                qml.CNOT(wires=[wire, wires[(k + 1) % len(wires)]])

    def _circuit(self, features_a, features_b, weights):
        # Shared/tied weight tensor, re-uploaded with each register's own
        # data at every layer (replaces AmplitudeEmbedding + a
        # data-independent tied block - see class docstring).
        self._reupload_block(weights, self.reg_a, features_a)
        self._reupload_block(weights, self.reg_b, features_b)

        # Swap test
        qml.Hadamard(wires=self.ancilla)
        for a, b in zip(self.reg_a, self.reg_b):
            qml.CSWAP(wires=[self.ancilla, a, b])
        qml.Hadamard(wires=self.ancilla)

        return qml.expval(qml.PauliZ(self.ancilla))

    def similarity(self, features_a, features_b, weights=None):
        """Returns learned similarity in [0, 1]. 1 = identical states."""
        w = self.weights if weights is None else weights
        val = self.qnode(features_a, features_b, w)
        return float(val)

    def save(self, path):
        np.save(path, qml.numpy.array(self.weights))

    def load(self, path):
        loaded = np.load(path)
        self.weights = pnp.array(loaded, requires_grad=True)


# ============================================================================
# SYNTHETIC TRAINING PAIR GENERATION (self-supervised - solves the
# "CoMoFoD masks have no patch correspondence" problem)
# ============================================================================

def make_synthetic_pairs(image: np.ndarray, patch_size: int, n_pos: int = 6, n_neg: int = 6,
                          rng: np.random.Generator = None):
    """
    Paste a real patch (optionally rotated) into a NEW location of a copy
    of `image`, then extract features from that pasted region - a genuine
    relocated copy-move pair by construction, not just a patch compared to
    its own in-place rotation. Sample random non-matching patches as
    negatives. Returns (features_list_a, features_list_b, labels).

    Note: extract_patch_features() is purely local to a patch's own pixel
    content (DCT + histogram, no neighborhood context), so pasting-then-
    extracting is numerically identical to extracting from the rotated
    source patch directly - relocation alone doesn't add signal without
    also modeling paste-boundary/blending effects, which this does not do.
    This fix corrects the pair-construction logic to match its documented
    intent (an earlier version sampled a target location but never used
    it), it does not by itself model photometric/blending mismatches a
    real copy-move forgery would have.
    """
    rng = rng or np.random.default_rng()
    H, W = image.shape
    pos_pairs, neg_pairs = [], []

    def rand_loc():
        y = rng.integers(0, H - patch_size)
        x = rng.integers(0, W - patch_size)
        return y, x

    for _ in range(n_pos):
        y1, x1 = rand_loc()
        src_patch = image[y1:y1 + patch_size, x1:x1 + patch_size].copy()
        rot_k = rng.choice([0, 1, 2, 3])  # 0/90/180/270 - discrete rotation robustness
        src_patch_rot = np.rot90(src_patch, k=rot_k)
        if src_patch_rot.shape != (patch_size, patch_size):
            src_patch_rot = src_patch  # guard for non-square edge cases

        y2, x2 = rand_loc()
        while abs(y2 - y1) < patch_size and abs(x2 - x1) < patch_size:
            y2, x2 = rand_loc()

        forged = image.copy()
        forged[y2:y2 + patch_size, x2:x2 + patch_size] = src_patch_rot

        f_a = extract_patch_features(src_patch)
        f_b = extract_patch_features(forged[y2:y2 + patch_size, x2:x2 + patch_size])
        pos_pairs.append((f_a, f_b, 1.0))

    for _ in range(n_neg):
        y1, x1 = rand_loc()
        y2, x2 = rand_loc()
        p1 = image[y1:y1 + patch_size, x1:x1 + patch_size]
        p2 = image[y2:y2 + patch_size, x2:x2 + patch_size]
        f_a = extract_patch_features(p1)
        f_b = extract_patch_features(p2)
        neg_pairs.append((f_a, f_b, 0.0))

    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)
    return all_pairs


# ============================================================================
# TRAINING - image-level batching across epochs (same batching pattern as v1)
# ============================================================================

def train(qnn: SwapTestSiameseQNN, dataset: CoMoFoDDataset, epochs: int = 10,
          images_per_epoch: int = 20, pairs_per_image: int = 12, patch_size: int = 32,
          lr: float = 0.1, checkpoint_dir: str = './checkpoints', seed: int = 42):
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
            batch = make_synthetic_pairs(image, patch_size, n_pos=n_pos, n_neg=n_neg, rng=rng)

            def cost(weights, batch=batch):
                total = 0.0
                for f_a, f_b, label in batch:
                    pred = qnn.qnode(f_a, f_b, weights)
                    total = total + (pred - label) ** 2
                return total / len(batch)

            qnn.weights, loss_val = opt.step_and_cost(cost, qnn.weights)
            epoch_losses.append(float(loss_val))
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(loss=f"{float(loss_val):.4f}")

        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)
        print(f"[Epoch {epoch+1}/{epochs}] mean loss = {mean_loss:.4f}")

        ckpt_path = Path(checkpoint_dir) / f"swaptest_qnn_epoch{epoch+1}.npy"
        qnn.save(ckpt_path)
        print(f"[INFO] Checkpoint saved: {ckpt_path}")

    with open(Path(checkpoint_dir) / 'history.json', 'w') as f:
        json.dump(history, f)
    return history


# ============================================================================
# INFERENCE: pairwise scoring + spatial offset-consistency filtering
# ============================================================================

def find_dominant_offsets(matched_pairs, coords, offset_bin: int = 4, min_votes: int = 3):
    """
    Histogram vote over (dy, dx) offsets between matched patch centers.
    A genuine copy-move produces MANY pairs sharing (approximately) the
    same offset; coincidental look-alikes (sky, walls) do not cluster.
    This currently votes on translation only.

    TODO (rotation, CR category): extend the vote key to
    (dy_bin, dx_bin, dtheta_bin) using a dominant-orientation feature
    (e.g. patch gradient histogram peak) per patch so that rotated
    copy-move pairs also cluster instead of being spread across many
    offset bins.
    """
    votes = defaultdict(list)
    for (i, j, sim) in matched_pairs:
        yi, xi = coords[i]
        yj, xj = coords[j]
        dy, dx = yj - yi, xj - xi
        key = (round(dy / offset_bin), round(dx / offset_bin))
        votes[key].append((i, j, sim))
        # offsets are symmetric (i,j) vs (j,i) - vote both directions
        key_rev = (round(-dy / offset_bin), round(-dx / offset_bin))
        votes[key_rev].append((i, j, sim))

    dominant_keys = [k for k, v in votes.items() if len(v) >= min_votes]
    kept = set()
    for k in dominant_keys:
        for (i, j, sim) in votes[k]:
            kept.add((i, j, sim))
    return list(kept)


def detect_copy_move(image: np.ndarray, qnn: SwapTestSiameseQNN, patch_size: int = 32,
                      stride: int = 8, block_window: int = 12, sim_threshold: float = 0.8,
                      offset_bin: int = 4, min_offset_votes: int = 3, verbose: bool = True):
    """Full pairwise pipeline for one test image -> binary forgery mask."""
    H, W = image.shape
    coords, feats = extract_all_patches(image, patch_size, stride)

    candidate_pairs = generate_candidate_pairs(coords, feats, window=block_window,
                                                min_spatial_dist=patch_size)
    if verbose:
        print(f"  [INFO] {len(coords)} patches -> {len(candidate_pairs)} candidate pairs "
              f"(vs {len(coords) * (len(coords) - 1) // 2} for brute force)")

    scored_pairs = []
    it = tqdm(candidate_pairs, desc="  Swap-test scoring", disable=not verbose)
    for (i, j) in it:
        sim = qnn.similarity(feats[i], feats[j])
        if sim >= sim_threshold:
            scored_pairs.append((i, j, sim))

    if verbose:
        print(f"  [INFO] {len(scored_pairs)} pairs above similarity threshold {sim_threshold}")

    matched_pairs = find_dominant_offsets(scored_pairs, coords, offset_bin=offset_bin,
                                           min_votes=min_offset_votes)
    if verbose:
        print(f"  [INFO] {len(matched_pairs)} pairs kept after offset-consistency filtering")

    mask = np.zeros((H, W), dtype=np.uint8)
    for (i, j, sim) in matched_pairs:
        for idx in (i, j):
            y, x = coords[idx]
            mask[y:y + patch_size, x:x + patch_size] = 1

    kernel = np.ones((3, 3), dtype=bool)
    mask = binary_dilation(mask.astype(bool), structure=kernel, iterations=1)
    mask = binary_erosion(mask, structure=kernel, iterations=1)
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
    parser = argparse.ArgumentParser(description='CMFD v2 - Siamese Swap-Test QNN')
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None,
                         help='Path to a .npy weights file (required for --mode test)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--n-qubits', type=int, default=4,
                         help='Per register. Total wires = 2*n_qubits + 1 ancilla.')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--images-per-epoch', type=int, default=20)
    parser.add_argument('--pairs-per-image', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--block-window', type=int, default=12)
    parser.add_argument('--sim-threshold', type=float, default=0.8)
    parser.add_argument('--offset-bin', type=int, default=4)
    parser.add_argument('--min-offset-votes', type=int, default=3)
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='./results_v2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    dataset = CoMoFoDDataset(args.dataset_dir, img_size=256, grayscale=True)
    qnn = SwapTestSiameseQNN(n_qubits=args.n_qubits, n_layers=args.n_layers)
    print(f"[INFO] QNN wires: {qnn.n_wires} ({args.n_qubits} per register + 1 ancilla), "
          f"{np.prod(qnn.weights.shape)} trainable parameters (shared/tied across both registers)")

    if args.mode == 'train':
        train(qnn, dataset, epochs=args.epochs, images_per_epoch=args.images_per_epoch,
              pairs_per_image=args.pairs_per_image, patch_size=args.patch_size,
              lr=args.lr, checkpoint_dir=args.checkpoint_dir, seed=args.seed)
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
            block_window=args.block_window, sim_threshold=args.sim_threshold,
            offset_bin=args.offset_bin, min_offset_votes=args.min_offset_votes)
        metrics = compute_metrics(pred_mask, gt_mask)
        transform_type = CoMoFoDDataset.get_transformation_type(img_name)
        print(f"  {transform_type}: F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f}")
        results.append({'image_name': img_name, 'transformation': transform_type, **metrics})

        np.save(Path(args.output_dir) / f"{img_name}_pred_mask.npy", pred_mask)

    import pandas as pd
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(args.output_dir) / 'results_v2.csv', index=False)
    print(f"\n[INFO] Results saved to {args.output_dir}/results_v2.csv")


if __name__ == '__main__':
    main()
