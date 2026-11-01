"""
Kaggle driver for training the pure Swap-Test Siamese QNN (v3).
================================================================
This script is pushed to Kaggle as a "script" kernel via `kaggle kernels push`
(see kernel-metadata.json). It does NOT duplicate the model/training code -
that lives in code/cmfd_swaptest_pipeline.py and code/train_split.py exactly
as in this repo, shipped to Kaggle as a small PRIVATE Kaggle Dataset (see
dataset-metadata.json) so local edits stay the single source of truth: bump
the dataset with `kaggle datasets version` and re-push the kernel to pick
up changes, instead of hand-copying code into this file.

Two dataset_sources are attached (see kernel-metadata.json):
  - "mahnoor23/comofod"            : the CoMoFoD_small_v2 image data (the
    original setup.sh slug, debadityachakraborty/..., is dead/403 as of
    2026-09-28 - this is a verified-working public mirror with the same
    content/size; swap the slug in kernel-metadata.json if this one also
    goes away).
  - "<KAGGLE_USERNAME>/cmfd-qnn-code" : this repo's code/ directory.

CONFIG below starts deliberately small (a "smoke test"): 2 epochs, 10
images/epoch. The point of the first push is to validate the whole
plumbing (dataset auto-discovery, pip installs, import path, checkpoint
saving) cheaply before spending real GPU-quota hours on it. Bump
epochs/images_per_epoch for the real run once a smoke-test push succeeds
end to end.
"""
import os
import re
import sys
import json
import subprocess
from pathlib import Path

# --------------------------------------------------------------------- #
# CONFIG - edit this block between pushes, nothing else needs to change.
# --------------------------------------------------------------------- #
CONFIG = dict(
    epochs=2,
    images_per_epoch=10,
    pairs_per_image=12,
    patch_size=8,
    n_layers=3,
    n_offset_qubits=2,
    lr=0.1,
    optimizer='adam',          # 'adam' or 'gd'
    val_images=4,
    test_frac=0.2,
    seed=42,
    use_gpu_lightning=False,   # only meaningful if kernel-metadata.json has enable_gpu: true
)

WORKDIR = Path('/kaggle/working')
CODE_DATASET_HINT = 'cmfd-qnn-code'   # substring to find the code dataset under /kaggle/input


# --------------------------------------------------------------------- #
# 1. Make sure the pinned QML stack is present (Kaggle's base image does
#    not ship pennylane). Version-pinned to match what was validated
#    locally (see requirements.txt / `pip list` on the dev machine).
# --------------------------------------------------------------------- #
def pip_install(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *pkgs], check=True)


print("[SETUP] installing pennylane stack...")
pip_install('pennylane==0.45.1', 'pennylane-lightning==0.45.0')

if CONFIG['use_gpu_lightning']:
    try:
        pip_install('pennylane-lightning-gpu', 'custatevec-cu12')
        print("[SETUP] pennylane-lightning-gpu installed - lightning.gpu may be picked up "
              "automatically if the SwapTestSiameseQNN device-selection logic is extended "
              "to try it (currently it tries lightning.qubit then default.qubit only).")
    except subprocess.CalledProcessError as e:
        print(f"[SETUP][WARN] pennylane-lightning-gpu install failed ({e}); "
              f"continuing on CPU lightning.qubit.")


# --------------------------------------------------------------------- #
# 2. Locate the two attached datasets under /kaggle/input without
#    hardcoding exact nesting depth - different uploaders zip things at
#    different folder depths, so search for content instead of guessing.
# --------------------------------------------------------------------- #
def find_code_dir() -> Path:
    root = Path('/kaggle/input')
    for p in root.rglob('cmfd_swaptest_pipeline.py'):
        return p.parent
    raise FileNotFoundError(
        f"Could not find cmfd_swaptest_pipeline.py under {root}. Check that the "
        f"'{CODE_DATASET_HINT}' dataset is attached in kernel-metadata.json "
        f"dataset_sources and was pushed with the code/ files at its root.")


def find_comofod_root() -> Path:
    root = Path('/kaggle/input')
    # A flat CoMoFoD_small_v2 dir has files like "001_F.png"; find one and
    # use its parent, regardless of how many levels the mirror nested it at.
    for pattern in ('*_F.png', '*_F.jpg'):
        for p in root.rglob(pattern):
            return p.parent
    raise FileNotFoundError(
        f"Could not find any *_F.png/*_F.jpg under {root}. Check the CoMoFoD "
        f"dataset_sources slug in kernel-metadata.json is still valid.")


code_dir = find_code_dir()
comofod_dir = find_comofod_root()
print(f"[SETUP] code dir      -> {code_dir}")
print(f"[SETUP] CoMoFoD root  -> {comofod_dir}")
sys.path.insert(0, str(code_dir))

from cmfd_swaptest_pipeline import CoMoFoDDataset, SwapTestSiameseQNN, train  # noqa: E402
from train_split import scene_id  # noqa: E402


# --------------------------------------------------------------------- #
# 3. Same scene-id train/test split logic as train_split.py, so a
#    checkpoint produced here is evaluated the same honest way (test-split
#    scene ids never touched during training).
# --------------------------------------------------------------------- #
ds = CoMoFoDDataset(str(comofod_dir), img_size=256, grayscale=True)

ids = sorted({scene_id(p) for p in ds.original_files}, key=lambda s: int(s))
import numpy as np
rng = np.random.default_rng(CONFIG['seed'])
rng.shuffle(ids)
n_test = max(1, int(round(len(ids) * CONFIG['test_frac'])))
test_ids = set(ids[:n_test])
train_ids = set(ids[n_test:])
train_originals = [p for p in ds.original_files if scene_id(p) in train_ids]
print(f"[SPLIT] {len(ids)} scenes -> {len(train_ids)} train / {len(test_ids)} test")
ds.original_files = train_originals

WORKDIR.mkdir(parents=True, exist_ok=True)
with open(WORKDIR / 'split.json', 'w') as f:
    json.dump({'train_ids': sorted(train_ids, key=int),
              'test_ids': sorted(test_ids, key=int),
              'seed': CONFIG['seed'], 'test_frac': CONFIG['test_frac']}, f, indent=2)

# --------------------------------------------------------------------- #
# 4. Train and save straight into /kaggle/working, which Kaggle persists
#    as the kernel's downloadable output (`kaggle kernels output`).
# --------------------------------------------------------------------- #
qnn = SwapTestSiameseQNN(patch_size=CONFIG['patch_size'], n_layers=CONFIG['n_layers'],
                         n_offset_qubits=CONFIG['n_offset_qubits'])
n_params = int(np.prod(qnn.weights.shape) + np.prod(qnn.offset_weights.shape))
print(f"[INFO] {qnn.n_wires} wires, {n_params} trainable params, "
      f"device={qnn.dev.name}, diff_method={qnn.diff_method}")
print(f"[INFO] CONFIG = {CONFIG}")

train(qnn, ds, epochs=CONFIG['epochs'], images_per_epoch=CONFIG['images_per_epoch'],
     pairs_per_image=CONFIG['pairs_per_image'], patch_size=CONFIG['patch_size'],
     lr=CONFIG['lr'], checkpoint_dir=str(WORKDIR / 'checkpoints'), seed=CONFIG['seed'],
     optimizer=CONFIG['optimizer'], val_images=CONFIG['val_images'])

final_out = WORKDIR / 'cmfd_model.npz'
qnn.save(final_out)
print(f"[DONE] saved deployable model -> {final_out}")
print("[DONE] pull results with: kaggle kernels output <owner>/cmfd-pure-qnn-train -p <local_dir>")
