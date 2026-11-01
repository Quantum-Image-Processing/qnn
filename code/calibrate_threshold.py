"""
Calibrate sim_threshold against real ground truth instead of guessing.
=======================================================================
results_coverage/results_grip.csv and results_coverage_tuned/results_grip.csv
both show recall ~1.0 with precision 0.05-0.34: the shipped --sim-threshold
values (0.8 / 0.9) were never actually swept against a PR curve, they were
guesses. That is very likely costing more F1 than anything wrong with the
trained circuit itself.

This script calls score_all_pairs() ONCE per image (the expensive O(N^2)
circuit pass) and then sweeps a whole grid of thresholds against the
already-computed similarities via mask_from_scores() + compute_metrics(),
which are cheap numpy operations. So calibrating against a threshold grid
costs the same runtime as a SINGLE detect_copy_move() call per image, not
one run per threshold tried.

Usage (GRIP/COVERAGE, same loader/flags as test_grip.py):
    python calibrate_threshold.py \
        --dataset-dir ../data/coverage \
        --checkpoint ../checkpoints_pure/swaptest_qnn_epoch10.npz \
        --forged-glob '*t.tif' --mask-union --native-eval \
        --num-images 20

Prints a per-threshold mean-F1 table and the best threshold, and writes the
full per-image x per-threshold sweep to --output-csv for further analysis
(e.g. plotting a PR curve).
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmfd_swaptest_pipeline import (        # noqa: E402
    SwapTestSiameseQNN, score_all_pairs, mask_from_scores, compute_metrics,
)
from grip_dataset import GRIPDataset          # noqa: E402


def upsample_mask(mask: np.ndarray, size_hw) -> np.ndarray:
    h, w = size_hw
    im = Image.fromarray((mask.astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)
    return (np.array(im, dtype=np.uint8) > 128).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description='Calibrate sim_threshold against real GT')
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n-layers', type=int, default=3)
    ap.add_argument('--n-offset-qubits', type=int, default=2)
    ap.add_argument('--target-long-side', type=int, default=384)
    ap.add_argument('--patch-size', type=int, default=8)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--native-eval', action='store_true')
    ap.add_argument('--num-images', type=int, default=20)
    ap.add_argument('--forged-glob', default=None)
    ap.add_argument('--mask-glob', default=None)
    ap.add_argument('--mask-union', action='store_true')
    ap.add_argument('--threshold-min', type=float, default=0.5)
    ap.add_argument('--threshold-max', type=float, default=0.99)
    ap.add_argument('--threshold-step', type=float, default=0.02)
    ap.add_argument('--output-csv', default='../results_threshold_sweep.csv')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    ds = GRIPDataset(args.dataset_dir, target_long_side=args.target_long_side,
                     align=args.stride, grayscale=True,
                     forged_glob=args.forged_glob, mask_glob=args.mask_glob,
                     mask_union=args.mask_union)

    qnn = SwapTestSiameseQNN(patch_size=args.patch_size, n_layers=args.n_layers,
                             n_offset_qubits=args.n_offset_qubits,
                             img_size=args.target_long_side)
    qnn.load(args.checkpoint)
    print(f"[INFO] Loaded weights from {args.checkpoint}")

    thresholds = np.round(
        np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step), 4)

    n = min(args.num_images, len(ds))
    rows = []
    for idx in range(n):
        img, gt_mask, name = ds[idx]
        H, W = img.shape
        print(f"\n--- {name} (working res {W}x{H}) --- scoring all pairs once ---")
        coords, pairs, sims = score_all_pairs(img, qnn, patch_size=args.patch_size,
                                              stride=args.stride)
        print(f"  sim range: [{sims.min():.3f}, {sims.max():.3f}]  mean={sims.mean():.3f}")

        if args.native_eval:
            native_gt, native_hw = ds.load_native_mask(idx)
        for t in thresholds:
            pred_mask, matched = mask_from_scores(coords, pairs, sims, args.patch_size,
                                                  H, W, float(t))
            if args.native_eval:
                pred_eval = upsample_mask(pred_mask, native_hw)
                gt_eval = native_gt
            else:
                pred_eval, gt_eval = pred_mask, gt_mask
            m = compute_metrics(pred_eval, gt_eval)
            rows.append({'image_name': name, 'threshold': float(t),
                         'n_matched_pairs': len(matched), **m})

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = df.groupby('threshold')[['precision', 'recall', 'f1', 'iou']].mean()
    print("\n=== Mean metrics per threshold (across all images) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    best_t = summary['f1'].idxmax()
    print(f"\n[RESULT] Best threshold by mean F1: {best_t:.4f} "
          f"(F1={summary.loc[best_t, 'f1']:.4f}, "
          f"P={summary.loc[best_t, 'precision']:.4f}, "
          f"R={summary.loc[best_t, 'recall']:.4f})")
    print(f"[INFO] Full sweep written to {out}")
    print("[NOTE] This calibrates --sim-threshold only. detect_copy_move()'s mask is "
          "per-pair thresholded; ForgeryDetector.analyze() in inference.py also has a "
          "separate 'decision_pairs' image-level FORGED/AUTHENTIC cutoff that this "
          "script does not sweep - recalibrate that against the app's own use case "
          "(single-image forged/authentic decision) if you rely on it.")


if __name__ == '__main__':
    main()
