"""
Train the swap-test QNN with a proper train/test split, and save a single
deployable model for the demo app.
====================================================================

The model is trained self-supervised on synthetic copy-move pairs built from
the *training-split* authentic images only; the test-split scene ids are held
out and written to models/split.json so evaluation never touches them.

Example (train on the full training split):
    cd code
    python train_split.py --dataset-dir ../data/comofod \
        --epochs 10 --images-per-epoch 80 --out ../models/cmfd_model.npz

Then point the Streamlit app's checkpoint at ../models/cmfd_model.npy.

Note: QNN training uses parameter-shift gradients and is SLOW. Increasing
--images-per-epoch and --epochs raises quality but cost scales roughly
linearly. Start small to validate the loop, then scale up.
"""
import re
import json
import argparse
from pathlib import Path

import numpy as np

from cmfd_swaptest_pipeline import CoMoFoDDataset, SwapTestSiameseQNN, train


def scene_id(path) -> str:
    m = re.search(r'\d+', Path(path).stem)
    return m.group(0) if m else Path(path).stem


def main():
    ap = argparse.ArgumentParser(description="Train/test split trainer for CMFD QNN")
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--test-frac', type=float, default=0.2)
    # v3 (pure QNN): qubits per register are derived from --patch-size, so
    # there is no --n-qubits any more.
    ap.add_argument('--n-layers', type=int, default=3)
    ap.add_argument('--n-offset-qubits', type=int, default=2)
    ap.add_argument('--train-rotations', action='store_true',
                    help='Include rotated positives. Off by default - the '
                         'circuit is translation-only and rotated targets '
                         'hold loss above the constant-predictor floor.')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--images-per-epoch', type=int, default=80)
    ap.add_argument('--pairs-per-image', type=int, default=12)
    ap.add_argument('--patch-size', type=int, default=8,
                    help='Powers of two. Sets qubits per register to '
                         'log2(patch_size^2): 8->6q (15 wires), 16->8q (19 wires, '
                         '~23x slower per pair).')
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='../models/cmfd_model.npz')
    ap.add_argument('--checkpoint-dir', default='../models/checkpoints')
    args = ap.parse_args()

    ds = CoMoFoDDataset(args.dataset_dir, img_size=256, grayscale=True)

    # ---- split by scene id so a scene is never in both train and test ----
    ids = sorted({scene_id(p) for p in ds.original_files}, key=lambda s: int(s))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * args.test_frac)))
    test_ids = set(ids[:n_test])
    train_ids = set(ids[n_test:])

    train_originals = [p for p in ds.original_files if scene_id(p) in train_ids]
    print(f"[SPLIT] {len(ids)} scenes -> {len(train_ids)} train / {len(test_ids)} test")
    print(f"[SPLIT] {len(train_originals)} training originals")

    # restrict the training pool to the train split (train() samples from this)
    ds.original_files = train_originals

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / 'split.json', 'w') as f:
        json.dump({'train_ids': sorted(train_ids, key=int),
                   'test_ids': sorted(test_ids, key=int),
                   'seed': args.seed, 'test_frac': args.test_frac}, f, indent=2)
    print(f"[SPLIT] wrote {out.parent/'split.json'}")

    qnn = SwapTestSiameseQNN(patch_size=args.patch_size, n_layers=args.n_layers,
                             n_offset_qubits=args.n_offset_qubits)
    n_params = int(np.prod(qnn.weights.shape) + np.prod(qnn.offset_weights.shape))
    print(f"[INFO] training {n_params} params, "
          f"{qnn.n_wires} wires, {args.epochs} epochs")

    train(qnn, ds, epochs=args.epochs, images_per_epoch=args.images_per_epoch,
          pairs_per_image=args.pairs_per_image, patch_size=args.patch_size,
          lr=args.lr, checkpoint_dir=args.checkpoint_dir, seed=args.seed,
          rotate=args.train_rotations)

    qnn.save(out)
    print(f"[DONE] saved deployable model -> {out}")
    print(f"       run the demo:  streamlit run app.py   "
          f"(set checkpoint = {out})")


if __name__ == '__main__':
    main()
