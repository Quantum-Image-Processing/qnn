"""
Step 2 driver: test the trained swap-test QNN on a STRATIFIED sample of 10
real CoMoFoD forged images (spanning all post-processing categories and, as
a bonus, spread across different geometric-transformation base-scene
ranges) and report per-category F1/IoU - not just the first 10 alphabetical
files, which would all come from a single base scene (001) and cover only
2-3 of the 6 real post-processing categories.

Does not modify cmfd_swaptest_pipeline.py; only imports and drives it.
"""
import sys
sys.path.insert(0, 'code')
import numpy as np
import pandas as pd
from pathlib import Path
from cmfd_swaptest_pipeline import (
    CoMoFoDDataset, SwapTestSiameseQNN, detect_copy_move, compute_metrics
)

CHECKPOINT = 'checkpoints_pure/swaptest_qnn_epoch10.npz'
OUTPUT_DIR = Path('results_v2_step2')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ds = CoMoFoDDataset('./data/comofod', img_size=256, grayscale=True)

# Stratified selection: one image per post-processing category, each drawn
# from a DIFFERENT geometry range where possible, for both category and
# geometry diversity within just 10 images.
targets = [
    ('001_F', 'Raw / no post-processing'),          # scene 1   -> Translation
    ('050_F_BC1', 'Brightness Change'),               # scene 50  -> Rotation
    ('090_F_CA1', 'Contrast Adjustment'),             # scene 90  -> Scaling
    ('130_F_CR1', 'Color Reduction'),                 # scene 130 -> Distortion
    ('170_F_IB1', 'Image Blurring'),                  # scene 170 -> Combination
    ('020_F_JC5', 'JPEG Compression'),                # scene 20  -> Translation
    ('060_F_NA1', 'Noise Adding'),                    # scene 60  -> Rotation
    ('100_F_BC2', 'Brightness Change (2nd sample)'),  # scene 100 -> Scaling
    ('045_F', 'Raw (2nd sample)'),                    # scene 45  -> Rotation
    ('190_F_CR2', 'Color Reduction (2nd sample)'),    # scene 190 -> Combination
]

name_to_path = {p.stem: p for p in ds.image_files}

qnn = SwapTestSiameseQNN(patch_size=8, n_layers=3)
qnn.load(CHECKPOINT)
print(f"[INFO] Loaded {CHECKPOINT}")

results = []
for name, label in targets:
    if name not in name_to_path:
        print(f"[WARN] {name} not found in dataset, skipping")
        continue
    idx = ds.image_files.index(name_to_path[name])
    img, gt_mask, img_name = ds[idx]
    if img is None or gt_mask is None:
        print(f"[WARN] {name}: no ground truth mask found, skipping")
        continue

    print(f"\n--- {img_name} ({label}) ---")
    # v3: stride 16 rather than 8 - pair scoring is exhaustive now, and
    # stride 8 would mean ~520k circuit evaluations per image.
    pred_mask, matched_pairs, coords = detect_copy_move(
        img, qnn, patch_size=8, stride=16,
        sim_threshold=0.8, verbose=False,
    )
    m = compute_metrics(pred_mask, gt_mask)
    geom = CoMoFoDDataset.get_geometry_type(img_name)
    post = CoMoFoDDataset.get_transformation_type(img_name)
    print(f"  post={post} geom={geom} F1={m['f1']:.4f} IoU={m['iou']:.4f} "
          f"prec={m['precision']:.4f} rec={m['recall']:.4f}")
    results.append({'image_name': img_name, 'post_processing': post,
                     'geometry': geom, **m})
    np.save(OUTPUT_DIR / f"{img_name}_pred_mask.npy", pred_mask)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / 'step2_results.csv', index=False)
print(f"\n[INFO] Saved {OUTPUT_DIR / 'step2_results.csv'}")

print("\n=== Per post-processing category ===")
print(df.groupby('post_processing')[['f1', 'iou', 'precision', 'recall']].mean().round(4))

print("\n=== Per geometry type ===")
print(df.groupby('geometry')[['f1', 'iou', 'precision', 'recall']].mean().round(4))

print(f"\n=== Overall mean F1={df['f1'].mean():.4f} IoU={df['iou'].mean():.4f} ===")
