"""
SIFT + RANSAC copy-move forgery detector (classical, robust).
==============================================================
This is the working detection engine for the demo. Unlike the swap-test QNN
(which over-detects because it can't separate a copy-move from natural
self-similarity), keypoint matching + geometric verification distinguishes a
genuine duplication from coincidental look-alikes: a real copy-move produces
many matched keypoints related by a SINGLE geometric transform (translation/
rotation/scale), which RANSAC recovers; natural repetition does not.

Method (Amerini et al. 2011, "A SIFT-based forensic method for copy-move
attack detection", + RANSAC geometric verification):
  1. SIFT keypoints + descriptors on the grayscale image.
  2. Self-matching with the g2NN test (generalized nearest-neighbour ratio),
     which finds one-to-many duplications, excluding spatially-close pairs.
  3. RANSAC affine estimation over the matched point set -> inliers are the
     matches consistent with one transform. Enough inliers => copy-move.
  4. Localize: inlier source keypoints and their matched targets form two
     clusters -> bounding boxes (source + pasted copy) + a match-density
     heatmap.

Returns the same AnalysisResult shape as inference.ForgeryDetector, so the
Streamlit app can use either engine interchangeably.
"""
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

from inference import AnalysisResult

_INFERNO = matplotlib.colormaps['inferno']


class SiftForgeryDetector:
    engine = "SIFT + RANSAC"

    def __init__(self, work_long_side: int = 1000, n_features: int = 0):
        # 0 = keep all SIFT features; downscale huge images for speed only.
        self.work_long_side = work_long_side
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    # --------------------------------------------------------------------- #
    def analyze(self, pil_image: Image.Image,
                ratio: float = 0.5,           # g2NN ratio threshold
                min_kp_dist: int = 10,        # ignore matches closer than this (px, work res)
                ransac_thresh: float = 6.0,   # RANSAC reprojection tolerance (px)
                min_cluster: int = 12,        # compact-cluster size to call it FORGED
                progress=None) -> AnalysisResult:
        original = pil_image.convert('RGB')
        ow, oh = original.size

        # work-resolution grayscale (downscale only if very large)
        scale = min(1.0, self.work_long_side / float(max(ow, oh)))
        ww, wh = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
        gray = np.asarray(pil_image.convert('L').resize((ww, wh), Image.LANCZOS), np.uint8)

        if progress: progress(0.15)
        kps, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kps) < 4:
            return self._empty(original, ww, wh, scale, ow, oh)

        pts = np.float32([k.pt for k in kps])            # (N,2) x,y in work res

        # ---- g2NN self-matching ----
        if progress: progress(0.4)
        matches = self._g2nn_matches(des, pts, ratio, min_kp_dist)
        if len(matches) < 4:
            return self._empty(original, ww, wh, scale, ow, oh)

        src = np.float32([pts[i] for i, j in matches])
        dst = np.float32([pts[j] for i, j in matches])

        # ---- RANSAC geometric verification ----
        if progress: progress(0.7)
        inlier_mask = None
        try:
            M, inlier_mask = cv2.estimateAffine2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
                maxIters=5000, confidence=0.99)
        except cv2.error:
            M = None
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        keep = (inlier_mask.ravel() == 1) if inlier_mask is not None else np.zeros(len(matches), bool)
        src_in = src[keep]; dst_in = dst[keep]

        # ---- compactness gate: real copy-move forms COMPACT blobs; natural
        # facade repetition spreads matches diffusely. Score = largest compact
        # DBSCAN cluster among inlier keypoints (separates forged/authentic far
        # better than raw inlier count on repetitive scenes). ----
        diag = float(np.hypot(wh, ww))
        cluster_score, cluster_pts = self._compact_clusters(
            np.vstack([src_in, dst_in]) if len(src_in) else np.empty((0, 2)),
            eps=diag * 0.04, min_samples=4)
        is_forged = cluster_score >= min_cluster
        confidence = float(np.clip(cluster_score / max(1, min_cluster * 2.5), 0, 1)) if is_forged \
            else float(np.clip(cluster_score / max(1, min_cluster * 2.5), 0, 0.49))

        if progress: progress(0.9)

        # scale factor work-res -> original; boxes around the compact blobs
        inv = 1.0 / scale
        boxes = []
        if is_forged:
            for cp in cluster_pts[:4]:                       # up to 4 duplicated blobs
                boxes.append(self._bbox(cp, inv, ow, oh))

        heat = self._match_heat(np.vstack([src_in, dst_in]) if len(src_in) else np.empty((0, 2)),
                                (wh, ww))
        if progress: progress(1.0)

        res = AnalysisResult(
            is_forged=is_forged, confidence=confidence, peak_votes=cluster_score,
            n_matched_pairs=len(matches),
            dominant_offset=tuple(np.round(M[:, 2]).astype(int)) if M is not None else (0, 0),
            boxes=boxes, original=original, working_size=max(ww, wh))
        res.heatmap_overlay = self._heatmap_overlay(original, heat)
        res.box_overlay = self._match_overlay(original, src_in, dst_in, inv, boxes, is_forged)
        res.mask_overlay = res.box_overlay
        return res

    @staticmethod
    def _compact_clusters(pts, eps, min_samples):
        """DBSCAN the inlier keypoints; return (largest cluster size, list of
        cluster point-arrays sorted largest-first). Diffuse matches (natural
        repetition) yield small/no clusters; a copy-move yields compact ones."""
        if len(pts) < min_samples:
            return 0, []
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
        groups = [pts[labels == c] for c in set(labels) if c != -1]
        groups.sort(key=len, reverse=True)
        largest = len(groups[0]) if groups else 0
        return largest, groups

    # --------------------------------------------------------------------- #
    def _g2nn_matches(self, des, pts, ratio, min_dist):
        """Generalized 2NN self-matching (handles multiple copies)."""
        des = des.astype(np.float32)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        k = min(10, len(des))
        knn = bf.knnMatch(des, des, k=k)
        seen = set()
        out = []
        for i, neigh in enumerate(knn):
            # neigh[0] is self (distance ~0); evaluate from index 1
            neigh = [m for m in neigh if m.trainIdx != i]
            if len(neigh) < 2:
                continue
            d = [m.distance for m in neigh]
            # find how many leading neighbours qualify via the ratio chain
            t = 1
            while t < len(d) - 1 and d[t] / (d[t + 1] + 1e-9) < ratio:
                t += 1
            for m in neigh[:t]:
                j = m.trainIdx
                if np.linalg.norm(pts[i] - pts[j]) < min_dist:
                    continue
                key = (min(i, j), max(i, j))
                if key in seen:
                    continue
                seen.add(key)
                out.append((i, j))
        return out

    @staticmethod
    def _bbox(pts, inv, ow, oh, pad=8):
        xs = pts[:, 0] * inv; ys = pts[:, 1] * inv
        x0 = max(0, int(xs.min()) - pad); y0 = max(0, int(ys.min()) - pad)
        x1 = min(ow, int(xs.max()) + pad); y1 = min(oh, int(ys.max()) + pad)
        return (x0, y0, x1, y1)

    @staticmethod
    def _match_heat(pts, shape):
        h = np.zeros(shape, np.float32)
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < shape[0] and 0 <= xi < shape[1]:
                h[yi, xi] += 1.0
        if h.max() > 0:
            h = gaussian_filter(h, sigma=max(shape) * 0.02)
        return h

    def _empty(self, original, ww, wh, scale, ow, oh):
        res = AnalysisResult(is_forged=False, confidence=0.0, peak_votes=0,
                             n_matched_pairs=0, dominant_offset=(0, 0),
                             boxes=[], original=original, working_size=max(ww, wh))
        res.heatmap_overlay = original.copy()
        res.box_overlay = original.copy()
        res.mask_overlay = original.copy()
        return res

    # ---- overlays ----
    @staticmethod
    def _heatmap_overlay(original, heat, alpha=0.55):
        ow, oh = original.size
        h = heat / heat.max() if heat.max() > 0 else heat
        h_img = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize((ow, oh), Image.BILINEAR),
                           np.float32) / 255.0
        rgba = (_INFERNO(h_img) * 255).astype(np.uint8)
        color = Image.fromarray(rgba[..., :3], 'RGB')
        a = (h_img * alpha * 255).astype(np.uint8)
        base = original.copy()
        base.paste(color, (0, 0), Image.fromarray(a, 'L'))
        return base

    @staticmethod
    def _match_overlay(original, src_in, dst_in, inv, boxes, is_forged):
        im = original.copy()
        d = ImageDraw.Draw(im)
        w = max(2, int(round(min(im.size) * 0.004)))
        # match lines source -> target
        for (sx, sy), (tx, ty) in zip(src_in, dst_in):
            d.line([sx * inv, sy * inv, tx * inv, ty * inv], fill=(255, 215, 0), width=1)
            d.ellipse([sx*inv-3, sy*inv-3, sx*inv+3, sy*inv+3], fill=(40, 120, 240))
            d.ellipse([tx*inv-3, ty*inv-3, tx*inv+3, ty*inv+3], fill=(235, 100, 50))
        col = (220, 40, 40) if is_forged else (40, 170, 60)
        for (x0, y0, x1, y1) in boxes:
            d.rectangle([x0, y0, x1, y1], outline=col, width=w)
        return im
