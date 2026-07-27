"""
GRIP Copy-Move Forgery Dataset loader
=====================================
Drop-in test-time dataset for the Siamese swap-test CMFD pipeline
(cmfd_swaptest_pipeline.py). Exposes the SAME contract as CoMoFoDDataset:

    len(ds)                  -> number of forged test images
    ds[i]                    -> (img_array, mask_array, name)   # both resized, aligned
    ds.get_training_image(i) -> authentic image (for synthetic-pair training, optional)
    ds.original_files        -> list of pristine image paths (may be empty)

Why a separate loader (not reuse CoMoFoDDataset):
  - GRIP has NO fixed, publicly-documented filename convention, and the
    archive layout has drifted across redistributions. So instead of
    hardcoding names this loader AUTO-DISCOVERS forged/mask pairs:
      * a file is treated as a MASK if its pixels are near-binary
        (GRIP ground-truth masks are strictly {0,255}); everything else
        is a photographic image;
      * files are grouped by the first integer in their filename (the
        scene id), and within a group the near-binary file is the mask.
    Manual overrides are provided (see __init__ args) for the case where
    auto-detection guesses wrong.
  - GRIP images are 1024x768 (NON-square). The CoMoFoD loader force-resizes
    to a square, which anisotropically distorts geometry and misaligns the
    mask. This loader instead does an ASPECT-RATIO-PRESERVING resize (long
    side -> `target_long_side`, each side snapped to a multiple of
    `align` so the sliding window tiles cleanly). See the Q&A discussion:
    for copy-move a *uniform* rescale is harmless (source and target scale
    identically, patches stay matched); an *anisotropic* one is not.

  - GRIP masks, as distributed, often include BOTH the source region and
    the target region (documented behaviour). This loader does NOT strip
    the source region - it keeps whatever is in the mask file - because the
    swap-test pipeline marks BOTH source and target in its predicted mask
    (detect_copy_move fills mask at both i and j), so a both-regions GT is
    the correct comparison. If you obtain target-only masks, they still
    load fine.

GRIP ground truth about the mask resolution question you raised: by
default __getitem__ returns the mask resized to the working resolution
(to match the predicted mask). For native-resolution evaluation - where
the predicted mask is upsampled to the mask's ORIGINAL size instead of
shrinking the GT - use load_native_mask(i) together with native_size[i]
(see test_grip.py for the wiring).
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = ('.png', '.tif', '.tiff', '.bmp', '.jpg', '.jpeg')

# Filename tokens that hint at each role, checked case-insensitively on the
# stem. Auto-detection does not RELY on these (mask detection is by pixel
# content), but they disambiguate forged-vs-original when a scene has both.
_FORGED_HINTS = ('forg', 'copy', 'tamp', 'fake', 'cm', 'splic', '_f')
_ORIG_HINTS = ('orig', 'prist', 'auth', 'real', '_o', 'gt_orig')
_MASK_HINTS = ('mask', 'gt', 'ground', '_m', 'binary')


def _first_int(stem: str):
    m = re.search(r'\d+', stem)
    return m.group(0) if m else None


def _is_binary_mask(arr: np.ndarray) -> bool:
    """A GRIP GT mask is (near-)binary. Robust test: almost every pixel is
    close to 0 or 255, and there are very few distinct levels."""
    u = np.unique(arr)
    if len(u) <= 4:
        return True
    frac_extreme = np.mean((arr <= 12) | (arr >= 243))
    return bool(frac_extreme >= 0.95)


class GRIPDataset:
    def __init__(self, dataset_dir: str, target_long_side: int = 384, align: int = 8,
                 grayscale: bool = True, id_regex: str = r'\d+',
                 forged_glob: str = None, mask_glob: str = None,
                 mask_union: bool = False, verbose: bool = True):
        """
        dataset_dir      : folder containing the extracted GRIP files (searched
                           recursively).
        target_long_side : longest output side in px (default 384 -> 384x288 for
                           a 4:3 GRIP image). Must be >= patch_size used at test.
        align            : snap each output side to a multiple of this (set to
                           the sliding-window `stride`, or `patch_size`, so
                           extract_all_patches tiles evenly). Default 8.
        id_regex         : how to extract the scene id used to pair forged<->mask.
                           Default: first run of digits in the filename.
        forged_glob      : OPTIONAL explicit override, e.g. '*_forged.png'. When
                           given, only files matching it are treated as forged
                           images (still paired to masks by id).
        mask_glob        : OPTIONAL explicit override for mask files, e.g.
                           '*_gt.png'. When given, pixel-content auto-detection
                           is skipped for classification.
        mask_union       : if a scene has SEVERAL mask files (e.g. COVERAGE's
                           per-image copy/paste/forged masks), OR them into one
                           GT mask instead of using only the first. Use this for
                           COVERAGE - the pipeline marks both source and target,
                           so the union is the correct comparison.
        """
        self.dataset_dir = Path(dataset_dir)
        self.target_long_side = int(target_long_side)
        self.align = max(1, int(align))
        self.grayscale = grayscale
        self.id_regex = re.compile(id_regex)
        self.mask_union = mask_union
        self.verbose = verbose

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"GRIP dir not found: {self.dataset_dir}")

        all_files = [p for p in self.dataset_dir.rglob('*')
                     if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not all_files:
            raise FileNotFoundError(
                f"No image files ({IMG_EXTS}) found under {self.dataset_dir}. "
                "Point --dataset-dir at the extracted GRIP folder.")

        forged_matches = set()
        mask_matches = set()
        if forged_glob:
            forged_matches = {p.resolve() for p in self.dataset_dir.rglob(forged_glob)}
        if mask_glob:
            mask_matches = {p.resolve() for p in self.dataset_dir.rglob(mask_glob)}

        # Classify every file into mask / forged / original, grouped by scene id.
        groups = {}  # id -> {'masks': [], 'forged': [], 'orig': [], 'other': []}
        for p in sorted(all_files):
            stem = p.stem
            low = stem.lower()
            m = self.id_regex.search(stem)
            key = m.group(0) if m else stem
            g = groups.setdefault(key, {'masks': [], 'forged': [], 'orig': [], 'other': []})

            # --- decide if this file is a MASK ---
            if mask_glob:
                is_mask = p.resolve() in mask_matches
            elif forged_glob and p.resolve() in forged_matches:
                is_mask = False
            else:
                is_mask = self._file_is_binary(p) or any(h in low for h in _MASK_HINTS)

            if is_mask:
                g['masks'].append(p)
                continue

            # --- non-mask: forged vs original ---
            if forged_glob:
                (g['forged'] if p.resolve() in forged_matches else g['other']).append(p)
            elif any(h in low for h in _FORGED_HINTS):
                g['forged'].append(p)
            elif any(h in low for h in _ORIG_HINTS):
                g['orig'].append(p)
            else:
                g['other'].append(p)

        self.samples = []      # (forged_path, [mask_paths], name)
        self.original_files = []
        ambiguous = []
        for key, g in sorted(groups.items()):
            if not g['masks']:
                continue  # can't evaluate a scene with no GT mask
            # keep ALL masks for the scene; loading uses the first one, or the
            # union of all of them when mask_union=True (COVERAGE-style GT).
            mask_paths = sorted(g['masks'])

            forged_candidates = g['forged'] or g['other']
            if not forged_candidates:
                continue
            if len(forged_candidates) > 1 and not g['forged']:
                ambiguous.append(key)
            forged_path = forged_candidates[0]

            name = f"GRIP_{key}_{forged_path.stem}"
            self.samples.append((forged_path, mask_paths, name))

            # Anything left over (declared original, or the 2nd 'other') is a
            # pristine image usable for optional synthetic-pair training.
            self.original_files.extend(g['orig'])
            self.original_files.extend(forged_candidates[1:])
            if g['forged']:
                # forged was identified explicitly (hint or glob), so any
                # remaining 'other' photos are pristine (COVERAGE's X.tif).
                self.original_files.extend(g['other'])

        if not self.samples:
            raise RuntimeError(
                f"Discovered {len(all_files)} files but built 0 (forged, mask) "
                "pairs. The auto-detector likely mis-grouped the files. Inspect "
                "the folder and pass --forged-glob / --mask-glob to override.")

        if self.verbose:
            print(f"[INFO] GRIP: {len(self.samples)} forged/mask pairs discovered "
                  f"under {self.dataset_dir}")
            print(f"[INFO] GRIP: {len(self.original_files)} pristine images available "
                  f"for optional training")
            for fp, mps, nm in self.samples[:3]:
                mask_desc = ' + '.join(m.name for m in mps) if self.mask_union \
                    else mps[0].name
                print(f"       e.g. {nm}: forged={fp.name}  mask={mask_desc}")
            if ambiguous:
                print(f"[WARN] GRIP: {len(ambiguous)} scene(s) had multiple photo "
                      f"files with no forged/original hint in the name; picked the "
                      f"first alphabetically. If metrics look wrong, pass "
                      f"--forged-glob to disambiguate. ids: {ambiguous[:10]}")

    # ---- pixel-content mask test with a tiny cache keyed by resolved path ----
    def _file_is_binary(self, path: Path) -> bool:
        try:
            arr = np.asarray(Image.open(path).convert('L'))
        except Exception:
            return False
        return _is_binary_mask(arr)

    def __len__(self):
        return len(self.samples)

    def _target_size(self, w: int, h: int):
        """Aspect-ratio-preserving target (W, H), each snapped to `align`."""
        scale = self.target_long_side / float(max(w, h))
        nw = max(self.align, int(round(w * scale / self.align)) * self.align)
        nh = max(self.align, int(round(h * scale / self.align)) * self.align)
        return nw, nh

    def _load_image_array(self, path: Path):
        img = Image.open(path)
        if self.grayscale:
            img = img.convert('L')
        w, h = img.size
        nw, nh = self._target_size(w, h)
        img = img.resize((nw, nh), Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    def _native_mask(self, mask_paths):
        """Binary GT at native resolution: first mask, or OR of all masks for
        the scene when mask_union=True (COVERAGE's copy/paste/forged files)."""
        use = mask_paths if self.mask_union else mask_paths[:1]
        out = None
        for mp in use:
            arr = (np.asarray(Image.open(mp).convert('L'), dtype=np.uint8) > 128)
            if out is None:
                out = arr
            elif arr.shape == out.shape:
                out |= arr
            # differently-sized extra masks (rare/corrupt) are skipped
        return out.astype(np.uint8)

    def __getitem__(self, idx):
        forged_path, mask_paths, name = self.samples[idx]
        img_array = self._load_image_array(forged_path)   # (H, W), working res
        nh, nw = img_array.shape

        native = self._native_mask(mask_paths)
        mask = Image.fromarray(native * 255).resize((nw, nh), Image.NEAREST)
        mask_array = (np.array(mask, dtype=np.uint8) > 128).astype(np.uint8)
        return img_array, mask_array, name

    def load_native_mask(self, idx):
        """Full-resolution binary GT mask (no downscale) + its (H, W).
        For native-resolution evaluation: upsample the PREDICTED mask to this
        size instead of shrinking the GT down to the working resolution."""
        _, mask_paths, _ = self.samples[idx]
        arr = self._native_mask(mask_paths)
        return arr, arr.shape

    def get_training_image(self, idx):
        if not self.original_files:
            raise RuntimeError("GRIP: no pristine images discovered for training.")
        path = self.original_files[idx % len(self.original_files)]
        return self._load_image_array(Path(path))
