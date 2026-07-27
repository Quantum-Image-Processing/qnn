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
* The image-level FORGED/AUTHENTIC decision is derived from the strength of
  the dominant translation-offset cluster (a genuine copy-move produces many
  matched patch pairs sharing ONE (dy,dx) offset; authentic images do not
  cluster). This is the signal the pipeline's offset-consistency filter
  already relies on - here we also read off its PEAK to make a decision.
* Honest caveat: the underlying model over-detects (see the Phase-2 report),
  so thresholds are exposed and tunable. This module gives the *output
  format* the advisor asked for; tuning quality is separate work.
"""

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
from scipy.ndimage import binary_dilation, binary_erosion, label as cc_label

_INFERNO = matplotlib.colormaps['inferno']

from cmfd_swaptest_pipeline import (
    SwapTestSiameseQNN, extract_all_patches, generate_candidate_pairs,
)


@dataclass
class AnalysisResult:
    is_forged: bool
    confidence: float
    peak_votes: int
    n_matched_pairs: int
    dominant_offset: tuple
    boxes: list = field(default_factory=list)          # original-image coords
    original: Image.Image = None
    heatmap_overlay: Image.Image = None
    box_overlay: Image.Image = None
    mask_overlay: Image.Image = None
    working_size: int = 256


class ForgeryDetector:
    def __init__(self, checkpoint: str, n_qubits: int = 4, n_layers: int = 2,
                 img_size: int = 256, patch_size: int = 32, stride: int = 8):
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.qnn = SwapTestSiameseQNN(n_qubits=n_qubits, n_layers=n_layers)
        self.qnn.load(checkpoint)
        self.checkpoint = checkpoint

    # --------------------------------------------------------------------- #
    def analyze(self, pil_image: Image.Image,
                sim_threshold: float = 0.90,
                block_window: int = 10,
                offset_bin: int = 4,
                min_offset_votes: int = 6,
                decision_votes: int = 8,
                min_box_area: int = 24,
                progress=None) -> AnalysisResult:
        """
        sim_threshold    : swap-test similarity above which a pair is a match
        block_window     : lexicographic block-match window (bigger = more pairs, slower)
        offset_bin       : (dy,dx) histogram bin size in px
        min_offset_votes : pairs needed in an offset bin to survive filtering
        decision_votes   : peak offset-cluster size at/above which -> FORGED
        min_box_area     : discard connected components smaller than this (px, working res)
        progress         : optional callback(fraction) for a UI progress bar
        """
        original = pil_image.convert('RGB')
        ow, oh = original.size

        gray = pil_image.convert('L').resize((self.img_size, self.img_size), Image.LANCZOS)
        img = np.asarray(gray, dtype=np.float32) / 255.0

        coords, feats = extract_all_patches(img, self.patch_size, self.stride)
        pairs = generate_candidate_pairs(coords, feats, window=block_window,
                                         min_spatial_dist=self.patch_size)

        # score candidate pairs with the QNN
        scored = []
        n = len(pairs)
        for k, (i, j) in enumerate(pairs):
            sim = self.qnn.similarity(feats[i], feats[j])
            if sim >= sim_threshold:
                scored.append((i, j, sim))
            if progress is not None and n and (k % 64 == 0):
                progress(k / n)
        if progress is not None:
            progress(1.0)

        # offset-consistency voting; read off the dominant cluster
        votes = defaultdict(list)
        for (i, j, sim) in scored:
            yi, xi = coords[i]; yj, xj = coords[j]
            dy, dx = yj - yi, xj - xi
            votes[(round(dy / offset_bin), round(dx / offset_bin))].append((i, j, sim))
            votes[(round(-dy / offset_bin), round(-dx / offset_bin))].append((i, j, sim))

        peak_votes, dominant_key = 0, (0, 0)
        for key, v in votes.items():
            if len(v) > peak_votes:
                peak_votes, dominant_key = len(v), key
        dominant_offset = (dominant_key[0] * offset_bin, dominant_key[1] * offset_bin)

        kept = set()
        for key, v in votes.items():
            if len(v) >= min_offset_votes:
                kept.update(v)
        kept = list(kept)

        is_forged = peak_votes >= decision_votes
        # confidence: how far the peak clears the decision threshold, capped
        confidence = float(np.clip(peak_votes / max(1, decision_votes * 2), 0, 1)) \
            if is_forged else float(np.clip(peak_votes / max(1, decision_votes * 2), 0, 0.5))

        # ---- build working-resolution mask + heatmap ----
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        heat = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        for (i, j, sim) in kept:
            for idx in (i, j):
                y, x = coords[idx]
                mask[y:y + self.patch_size, x:x + self.patch_size] = 1
                heat[y:y + self.patch_size, x:x + self.patch_size] += sim
        k3 = np.ones((3, 3), dtype=bool)
        mask = binary_erosion(binary_dilation(mask.astype(bool), k3), k3).astype(np.uint8)

        # ---- bounding boxes from connected components (working res) ----
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
            is_forged=is_forged, confidence=confidence, peak_votes=peak_votes,
            n_matched_pairs=len(kept), dominant_offset=dominant_offset,
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
