"""
Test the trained Siamese swap-test QNN on the GRIP copy-move dataset.
=====================================================================
Reuses the model + detector + metrics from cmfd_swaptest_pipeline.py
unchanged; only the dataset loader differs (GRIP is non-square and has
no fixed naming convention - see grip_dataset.py).

Example
-------
    python test_grip.py \
        --dataset-dir ../data/grip \
        --checkpoint ../checkpoints_full/swaptest_qnn_epoch10.npy \
        --n-qubits 4 --n-layers 2 \
        --num-images 20 --output-dir ../results_grip

Add --native-eval to score at the mask's ORIGINAL resolution (predicted
mask is upsampled to native size) instead of shrinking the GT to the
working resolution - this removes the mask-downscaling penalty on recall
for GRIP's small copied regions (~4000-50000 px).

COVERAGE
--------
The loader is dataset-agnostic, so this same script tests COVERAGE
(image/Xt.tif = forged, image/X.tif = original, mask/Xcopy|paste|forged.tif
= per-region GT masks). Two overrides are REQUIRED for it:

    python test_grip.py \
        --dataset-dir ../data/coverage \
        --checkpoint ../checkpoints_full/swaptest_qnn_epoch10.npy \
        --forged-glob '*t.tif' --mask-union \
        --num-images 100 --native-eval --output-dir ../results_coverage

--forged-glob '*t.tif' : COVERAGE's forged images end in 't'; without this
                         the auto-detector may pick the pristine X.tif.
--mask-union           : OR all of a scene's masks (copy + paste + forged)
                         into one GT, since detect_copy_move marks BOTH the
                         source and the target region.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# Import the pipeline that lives alongside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmfd_swaptest_pipeline import (        # noqa: E402
    SwapTestSiameseQNN, detect_copy_move, compute_metrics,
)
from grip_dataset import GRIPDataset          # noqa: E402


def upsample_mask(mask: np.ndarray, size_hw) -> np.ndarray:
    """NEAREST upsample a binary mask to (H, W) - keeps it binary."""
    h, w = size_hw
    im = Image.fromarray((mask.astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)
    return (np.array(im, dtype=np.uint8) > 128).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description='Test swap-test QNN on GRIP')
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--checkpoint', required=True,
                    help='.npy weights file, e.g. ../checkpoints_full/swaptest_qnn_epoch10.npy')
    ap.add_argument('--n-qubits', type=int, default=4)
    ap.add_argument('--n-layers', type=int, default=2)
    # working-resolution / windowing (must match how the model was trained)
    ap.add_argument('--target-long-side', type=int, default=384,
                    help='GRIP long side after aspect-preserving resize')
    ap.add_argument('--patch-size', type=int, default=32)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--block-window', type=int, default=12)
    ap.add_argument('--sim-threshold', type=float, default=0.8)
    ap.add_argument('--offset-bin', type=int, default=4)
    ap.add_argument('--min-offset-votes', type=int, default=3)
    # eval options
    ap.add_argument('--native-eval', action='store_true',
                    help='score at the mask original resolution (upsample pred)')
    ap.add_argument('--num-images', type=int, default=20)
    ap.add_argument('--output-dir', default='../results_grip')
    ap.add_argument('--seed', type=int, default=42)
    # optional loader overrides for when auto-detection mis-groups files
    ap.add_argument('--forged-glob', default=None)
    ap.add_argument('--mask-glob', default=None)
    ap.add_argument('--mask-union', action='store_true',
                    help='OR all masks of a scene into one GT (use for COVERAGE)')
    args = ap.parse_args()

    np.random.seed(args.seed)

    ds = GRIPDataset(args.dataset_dir, target_long_side=args.target_long_side,
                     align=args.stride, grayscale=True,
                     forged_glob=args.forged_glob, mask_glob=args.mask_glob,
                     mask_union=args.mask_union)

    qnn = SwapTestSiameseQNN(n_qubits=args.n_qubits, n_layers=args.n_layers)
    qnn.load(args.checkpoint)
    print(f"[INFO] Loaded weights from {args.checkpoint} "
          f"({qnn.n_wires} wires, {np.prod(qnn.weights.shape)} params)")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    n = min(args.num_images, len(ds))
    for idx in range(n):
        img, gt_mask, name = ds[idx]
        print(f"\n--- {name}  (working res {img.shape[1]}x{img.shape[0]}) ---")

        pred_mask, matched_pairs, coords = detect_copy_move(
            img, qnn, patch_size=args.patch_size, stride=args.stride,
            block_window=args.block_window, sim_threshold=args.sim_threshold,
            offset_bin=args.offset_bin, min_offset_votes=args.min_offset_votes)

        if args.native_eval:
            native_gt, native_hw = ds.load_native_mask(idx)
            pred_eval = upsample_mask(pred_mask, native_hw)
            gt_eval = native_gt
            res_tag = f"native {native_hw[1]}x{native_hw[0]}"
        else:
            pred_eval, gt_eval = pred_mask, gt_mask
            res_tag = f"working {img.shape[1]}x{img.shape[0]}"

        metrics = compute_metrics(pred_eval, gt_eval)
        print(f"  [{res_tag}] F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}")
        results.append({'image_name': name, 'eval_res': res_tag, **metrics})

        np.save(out / f"{name}_pred_mask.npy", pred_mask)

    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(out / 'results_grip.csv', index=False)
        print(f"\n[INFO] {len(results)} images. "
              f"mean F1={df['f1'].mean():.4f}  mean IoU={df['iou'].mean():.4f}")
        print(f"[INFO] Results -> {out/'results_grip.csv'}")
    else:
        print("[WARN] No images evaluated.")


if __name__ == '__main__':
    main()
