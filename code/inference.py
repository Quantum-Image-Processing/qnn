"""
Inference wrapper for the Siamese swap-test CMFD model.
=======================================================
Turns the trained QNN into a single-call detector for an arbitrary uploaded
image, producing exactly what the demo needs:

    result = detector.analyze(pil_image)
    result.is_forged      # bool  -> "FORGED" / "AUTHENTIC"
    result.confidence     # float in [0,1]
    result.boxes          # [(x0,y0,x1,y1), ...] in ORIGINAL image coords
    result.heatmap_overlay# PIL.Image (original size) - forgery evidence heatmap
    result.box_overlay    # PIL.Image (original size) - boxes drawn on the image
    result.mask_overlay   # PIL.Image (original size) - binary region shaded

Design notes
------------
* Works on any .jpg/.jpeg/.png, colour or grayscale, any size. The image is
  converted to grayscale and resized to a square working resolution (256,
  matching how the model was trained) for detection; all outputs are mapped
  BACK to the original resolution for display.
* PURE-QNN path (v3). Every detection decision here comes from the circuit's
  measured expectation value. Specifically:
    - patches are amplitude-encoded raw pixels, not DCT/histogram features
    - all admissible pairs are scored (no classical lexicographic prefilter)
    - the FORGED/AUTHENTIC decision is the COUNT of pairs the circuit scored
      above sim_threshold; v2's classical (dy,dx) Hough vote is gone, its job
      having moved into the circuit's offset register
    - no morphological cleanup of the mask
* The ONLY classical operation left is connected-component labelling, used
  purely to draw bounding boxes for the UI. It runs AFTER the decision and
  never feeds back into is_forged, the mask, or any reported metric - it is
  display formatting, not detection.
* Honest caveat: expect this to over-detect MORE than the v2 hybrid did.
  Removing the offset vote removed the stage that actually suppressed
  false matches on homogeneous regions (sky, walls). Thresholds are exposed
  and tunable; see the Phase-2 report.
"""

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
from scipy.ndimage import label as cc_label   # display-only, see module docstring

_INFERNO = matplotlib.colormaps['inferno']

from cmfd_swaptest_pipeline import (
    SwapTestSiameseQNN, extract_all_patches, generate_all_pairs, encode_offset,
)


@dataclass
class AnalysisResult:
    """
    Shared by both engines, so it carries the union of their signals.

    peak_votes / dominant_offset are SIFT-only: they come from that
    engine's RANSAC cluster. The pure-QNN path leaves them at their
    defaults because v3 has no classical offset vote to read a peak
    from - it reports mean/max_similarity instead. Check which engine
    produced a result before interpreting these.
    """
    is_forged: bool
    confidence: float
    n_matched_pairs: int = 0
    mean_similarity: float = 0.0       # QNN path
    max_similarity: float = 0.0        # QNN path
    peak_votes: int = 0                # SIFT path
    dominant_offset: tuple = (0, 0)    # SIFT path
    boxes: list = field(default_factory=list)          # original-image coords
    original: Image.Image = None
    heatmap_overlay: Image.Image = None
    box_overlay: Image.Image = None
    mask_overlay: Image.Image = None
    working_size: int = 256


class ForgeryDetector:
    def __init__(self, checkpoint: str, n_layers: int = 3,
                 img_size: int = 128, patch_size: int = 8, stride: int = 8,
                 n_offset_qubits: int = 2, diff_method: str = 'auto',
                 batch_size: int = 256):
        """
        stride defaults to patch_size so patches TILE the image. This is a
        correctness requirement, not a speed knob: with stride 16 at 256px
        only 25% of each image was examined, and a forged image and its
        authentic original produced byte-identical output (15 matched pairs,
        mean 0.940) because the 2%-area forged region was never sampled.
        img_size dropped 256 -> 128 to keep tiled scoring affordable, since
        exhaustive pair scoring is O(patches^2).

        batch_size: pairs scored per QNode call via PennyLane parameter
        broadcasting, instead of one call per pair. Measured ~1.3x faster
        on lightning.qubit with bit-identical output - a real but modest
        win (lightning still evaluates the batch dimension internally
        rather than truly vectorizing it), not the main speed lever
        img_size is; see SwapTestSiameseQNN.similarity for the benchmark.
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        if stride > patch_size:
            print(f"[WARN] stride ({stride}) > patch_size ({patch_size}): patches do "
                  f"not tile; part of every image will never be examined.")
        self.qnn = SwapTestSiameseQNN(patch_size=patch_size, n_layers=n_layers,
                                      n_offset_qubits=n_offset_qubits,
                                      img_size=img_size, diff_method=diff_method)
        self.qnn.load(checkpoint)
        self.checkpoint = checkpoint

    # --------------------------------------------------------------------- #
    def analyze(self, pil_image: Image.Image,
                sim_threshold: float = 0.90,
                decision_pairs: int = 8,
                min_box_area: int = 24,
                progress=None) -> AnalysisResult:
        """
        sim_threshold  : circuit output above which a pair counts as a match
        decision_pairs : matched-pair count at/above which -> FORGED
        min_box_area   : discard connected components smaller than this
                         (px, working res) - affects DRAWN BOXES ONLY
        progress       : optional callback(fraction) for a UI progress bar

        Note there is no block_window / offset_bin / min_offset_votes any
        more: those parameterized classical stages that v3 removed.
        """
        original = pil_image.convert('RGB')
        ow, oh = original.size

        gray = pil_image.convert('L').resize((self.img_size, self.img_size), Image.LANCZOS)
        img = np.asarray(gray, dtype=np.float32) / 255.0

        coords, states = extract_all_patches(img, self.patch_size, self.stride)
        pairs = generate_all_pairs(coords, min_spatial_dist=self.patch_size)

        # Score every admissible pair with the circuit, batch_size pairs per
        # QNode call (see ForgeryDetector.__init__ and SwapTestSiameseQNN.
        # similarity). This is the whole detector - what survives the
        # threshold IS the result.
        matched = []
        n = len(pairs)
        for start in range(0, n, self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            offs = np.array([encode_offset(coords[j][0] - coords[i][0],
                                            coords[j][1] - coords[i][1], self.img_size)
                              for (i, j) in chunk])
            states_a = states[[i for (i, j) in chunk]]
            states_b = states[[j for (i, j) in chunk]]
            sims = np.atleast_1d(self.qnn.similarity(states_a, states_b, offs))
            for (i, j), sim in zip(chunk, sims):
                if sim >= sim_threshold:
                    matched.append((i, j, float(sim)))
            if progress is not None and n:
                progress(min(start + len(chunk), n) / n)
        if progress is not None:
            progress(1.0)

        sims = [s for (_, _, s) in matched]
        mean_similarity = float(np.mean(sims)) if sims else 0.0
        max_similarity = float(np.max(sims)) if sims else 0.0

        # Image-level decision: how many pairs the circuit flagged. This is
        # an aggregation of measurement outcomes, not a classical matching
        # rule - it makes no use of patch content or geometry.
        is_forged = len(matched) >= decision_pairs
        confidence = float(np.clip(len(matched) / max(1, decision_pairs * 2), 0, 1)) \
            if is_forged else float(np.clip(len(matched) / max(1, decision_pairs * 2), 0, 0.5))

        # ---- build working-resolution mask + heatmap (no morphology) ----
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        heat = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for (i, j, sim) in matched:
            for idx in (i, j):
                y, x = coords[idx]
                mask[y:y + self.patch_size, x:x + self.patch_size] = 1
                heat[y:y + self.patch_size, x:x + self.patch_size] += sim

        # ---- bounding boxes from connected components (working res) ----
        # DISPLAY ONLY - runs after the decision, feeds nothing back into it.
        boxes_ws = []
        lbl, ncc = cc_label(mask)
        for c in range(1, ncc + 1):
            ys, xs = np.where(lbl == c)
            if ys.size < min_box_area:
                continue
            boxes_ws.append((xs.min(), ys.min(), xs.max(), ys.max()))

        # map boxes to original coords
        sx, sy = ow / self.img_size, oh / self.img_size
        boxes = [(int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
                 for (x0, y0, x1, y1) in boxes_ws]

        res = AnalysisResult(
            is_forged=is_forged, confidence=confidence,
            n_matched_pairs=len(matched), mean_similarity=mean_similarity,
            max_similarity=max_similarity,
            boxes=boxes, original=original, working_size=self.img_size)
        res.heatmap_overlay = self._heatmap_overlay(original, heat)
        res.box_overlay = self._box_overlay(original, boxes, is_forged)
        res.mask_overlay = self._mask_overlay(original, mask)
        return res

    # --------------------------------------------------------------------- #
    @staticmethod
    def _heatmap_overlay(original: Image.Image, heat: np.ndarray, alpha=0.55) -> Image.Image:
        ow, oh = original.size
        h = heat.copy()
        if h.max() > 0:
            h = h / h.max()
        h_img = Image.fromarray((h * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR)
        h_arr = np.asarray(h_img, dtype=np.float32) / 255.0
        rgba = (_INFERNO(h_arr) * 255).astype(np.uint8)   # (oh,ow,4)
        color = Image.fromarray(rgba[..., :3], 'RGB')
        # blend only where there is evidence, so cool background stays the photo
        a = (h_arr * alpha * 255).astype(np.uint8)
        base = original.copy()
        base.paste(color, (0, 0), Image.fromarray(a, 'L'))
        return base

    @staticmethod
    def _box_overlay(original: Image.Image, boxes, is_forged: bool) -> Image.Image:
        im = original.copy()
        d = ImageDraw.Draw(im)
        col = (220, 40, 40) if is_forged else (40, 170, 60)
        w = max(2, int(round(min(im.size) * 0.006)))
        for (x0, y0, x1, y1) in boxes:
            d.rectangle([x0, y0, x1, y1], outline=col, width=w)
        return im

    @staticmethod
    def _mask_overlay(original: Image.Image, mask: np.ndarray, alpha=0.45) -> Image.Image:
        ow, oh = original.size
        m = Image.fromarray((mask * 255).astype(np.uint8)).resize((ow, oh), Image.NEAREST)
        red = Image.new('RGB', (ow, oh), (220, 40, 40))
        base = original.copy()
        a = (np.asarray(m, dtype=np.float32) / 255.0 * alpha * 255).astype(np.uint8)
        base.paste(red, (0, 0), Image.fromarray(a, 'L'))
        return base
