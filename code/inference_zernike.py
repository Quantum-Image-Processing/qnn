"""
Single-call detector for the 9-wire Zernike QNN, for the dashboard.
===================================================================
Same result object and display overlays as inference.ForgeryDetector. See the
top of cmfd_zernike_swaptest.py for what this model does and does not deliver:
it is faster and handles rotation, but it does not detect forgeries better.

Scores come from one 4-qubit state per patch, which equals running the 9-wire
swap test on every pair (checked to under 1e-15) and takes well under a second.
Pass use_real_circuits=True to run all 32,640 circuits instead, about 3-4 minutes.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label   # display-only, as in inference.py

from cmfd_zernike_swaptest import ZernikeSwapQNN, image_states
from inference import AnalysisResult, ForgeryDetector


class ZernikeForgeryDetector(ForgeryDetector):
    def __init__(self, checkpoint: str, n_layers: int = 3, img_size: int = 128,
                 patch_size: int = 8, stride: int = 8, batch_size: int = 256,
                 use_real_circuits: bool = False):
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.use_real_circuits = use_real_circuits
        self.qnn = ZernikeSwapQNN(n_layers=n_layers, img_size=img_size)
        self.qnn.load(checkpoint)
        self.checkpoint = checkpoint

    def _all_scores(self, states, progress=None):
        if not self.use_real_circuits:
            return self.qnn.all_pair_scores(states)
        n = len(states)
        i, j = np.triu_indices(n, 1)
        s = np.zeros((n, n))
        for start in range(0, len(i), self.batch_size):
            a, b = i[start:start + self.batch_size], j[start:start + self.batch_size]
            s[a, b] = s[b, a] = np.atleast_1d(self.qnn.score(states[a], states[b]))
            if progress is not None:
                progress(min(start + len(a), len(i)) / len(i))
        return s

    def analyze(self, pil_image: Image.Image, sim_threshold: float, decision_pairs: int,
                min_box_area: int = 24, progress=None) -> AnalysisResult:
        original = pil_image.convert('RGB')
        ow, oh = original.size
        gray = pil_image.convert('L').resize((self.img_size, self.img_size), Image.LANCZOS)
        img = np.asarray(gray, dtype=np.float32) / 255.0

        coords, states = image_states(img, self.patch_size, self.stride)
        scores = self._all_scores(states, progress)
        if progress is not None:
            progress(1.0)

        i, j = np.triu_indices(len(coords), 1)
        hit = scores[i, j] >= sim_threshold
        matched = list(zip(i[hit], j[hit], scores[i, j][hit]))
        sims = [float(s) for (_, _, s) in matched]

        # Same image-level rule as v3: how many pairs the circuit flagged.
        is_forged = len(matched) >= decision_pairs
        confidence = len(matched) / max(1, decision_pairs * 2)
        confidence = float(np.clip(confidence, 0, 1 if is_forged else 0.5))

        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        heat = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for (a, b, sim) in matched:
            for idx in (a, b):
                y, x = coords[idx]
                mask[y:y + self.patch_size, x:x + self.patch_size] = 1
                heat[y:y + self.patch_size, x:x + self.patch_size] += sim

        lbl, ncc = cc_label(mask)
        sx, sy = ow / self.img_size, oh / self.img_size
        boxes = []
        for c in range(1, ncc + 1):
            ys, xs = np.where(lbl == c)
            if ys.size >= min_box_area:
                boxes.append((int(xs.min() * sx), int(ys.min() * sy), int(xs.max() * sx), int(ys.max() * sy)))

        res = AnalysisResult(is_forged=is_forged, confidence=confidence, n_matched_pairs=len(matched),
                             mean_similarity=float(np.mean(sims)) if sims else 0.0,
                             max_similarity=float(np.max(sims)) if sims else 0.0,
                             boxes=boxes, original=original, working_size=self.img_size)
        res.heatmap_overlay = self._heatmap_overlay(original, heat)
        res.box_overlay = self._box_overlay(original, boxes, is_forged)
        res.mask_overlay = self._mask_overlay(original, mask)
        return res
