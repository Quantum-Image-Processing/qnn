# Usage Guide - Parameters & Examples

This covers the two ways to run the CMFD pure-QNN pipeline: the interactive
Streamlit demo (`app.py`) and the command-line trainer/evaluator
(`cmfd_swaptest_pipeline.py`). Both share the same `SwapTestSiameseQNN` model
from `code/cmfd_swaptest_pipeline.py` — see that file's module docstring for
the full architecture rationale.

---

## 1. Streamlit Demo (`app.py`)

```bash
cd code
streamlit run app.py
```

The sidebar offers exactly one choice: **Pure classical (SIFT+RANSAC)** or
**Pure quantum (v3 swap-test QNN)**. Every other threshold is pinned at the
values the Phase-2 evaluation used (`SIFT_CFG` / `QNN_CFG` at the top of
`app.py`), so the two engines are compared as whole methods, not as piles of
sliders. The quantum engine loads `checkpoints_pure/swaptest_qnn_epoch10.npz`
by default — no training required to try the demo.

---

## 2. Command Line (`cmfd_swaptest_pipeline.py`)

### Basic
```bash
python code/cmfd_swaptest_pipeline.py --mode test \
    --dataset-dir ./data/comofod \
    --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz
```

### Required
```bash
--mode {train,test}
--dataset-dir PATH
```
`--dataset-dir` must contain either the legacy `Forged/`+`Masks/` layout, or a
flat `CoMoFoD_small_v2/` directory (auto-detected either way).

`--checkpoint PATH` is additionally required for `--mode test` — a `.npz` file
produced by a prior `--mode train` run. v2's old `.npy` checkpoints are a
different circuit shape and are deliberately rejected.

### Optional

#### `--patch-size N` (default: 8)
Powers of two only. Sets qubits per register to `log2(patch_size²)`:
```bash
--patch-size 8      # 6 qubits/register, 15 wires total, ~0.036 s/pair
--patch-size 16     # 8 qubits/register, 19 wires total, ~0.82 s/pair (~23x slower)
```
Simulation cost is exponential in total wires — don't raise this casually.

#### `--img-size N` (default: 128)
Working resolution. Kept at 128 (down from 256) so that full tiling stays
affordable: at 256px tiled, exhaustive pair scoring is ~520k pairs
(~5 h/image); at 128px it's ~33k pairs (~20 min/image).

#### `--stride N` (default: 8, i.e. == `--patch-size`)
**Must equal `--patch-size` for full coverage.** This is a correctness
requirement, not a speed knob — a larger stride leaves gaps, and measured
runs show a forged image and its authentic original can return *identical*
output when the stride skips over the forged region entirely. The pipeline
prints a `[WARN]` if `--stride > --patch-size`.

#### `--n-layers N` (default: 3)
Data re-uploading blocks. Pixels are re-uploaded into the rotation angles at
every layer — this is what keeps gradients alive under the swap test (see the
`SwapTestSiameseQNN` docstring for why a plain tied unitary is provably
degenerate here). More layers = more expressive and proportionally slower.

#### `--n-offset-qubits N` (default: 2)
Wires for the in-circuit spatial-offset register (the quantum stand-in for a
classical (dy, dx) vote).

#### `--diff-method {auto,adjoint,backprop,parameter-shift}` (default: auto)
`auto` uses `lightning.qubit` + `adjoint` when available (fastest simulator
path). `parameter-shift` is the hardware-realizable gradient but costs 2
circuit evals *per parameter* — much slower, use only if you need it.

#### `--epochs N` / `--images-per-epoch N` / `--pairs-per-image N` / `--lr F`
Training loop knobs (`--mode train` only). Training is self-supervised on
synthetic copy-move pairs (a real patch pasted into a new location of the
same image) — CoMoFoD's masks carry no patch-level correspondence, so this
augmentation is what makes training possible at all.

#### `--train-rotations`
Off by default. Includes 90/180/270-rotated positives in synthetic training
pairs. Leave this off unless you've also added rotation handling to the
offset register — the circuit is translation-only, so rotated positives are
targets it provably cannot represent, and including them pins training loss
above the constant-predictor floor instead of converging.

#### `--sim-threshold F` (default: 0.8, test mode only)
Circuit output above which a pair counts as a match.

#### `--num-images N`, `--output-dir PATH`, `--seed N`, `--checkpoint-dir PATH`
Same meaning as any CLI tool: how many images to evaluate, where to write
results/checkpoints, and the RNG seed.

---

## Configuration Examples

### Quick evaluation (5 images)
```bash
python code/cmfd_swaptest_pipeline.py --mode test \
    --dataset-dir ./data/comofod \
    --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz \
    --num-images 5 --output-dir ./results_pure
```

### Train from scratch
```bash
python code/cmfd_swaptest_pipeline.py --mode train \
    --dataset-dir ./data/comofod \
    --epochs 10 --images-per-epoch 20 --pairs-per-image 12 --lr 0.05 \
    --checkpoint-dir ./checkpoints_pure
```

### Train/test split (held-out scenes, deployable checkpoint for the demo)
```bash
python code/train_split.py --dataset-dir ./data/comofod \
    --test-frac 0.2 --epochs 10 --images-per-epoch 80 \
    --patch-size 8 --n-layers 3 \
    --out ./models/cmfd_model.npz
```

### Evaluate on GRIP or COVERAGE instead of CoMoFoD
```bash
python code/test_grip.py \
    --dataset-dir ./data/grip \
    --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz \
    --patch-size 8 --n-layers 3 \
    --num-images 20 --output-dir ./results_grip
```

---

## Analyzing Results

```bash
# View metrics
cat results_pure/results_pure.csv

# Statistical summary
python -c "
import pandas as pd
df = pd.read_csv('results_pure/results_pure.csv')
print(df[['f1', 'iou', 'accuracy', 'precision', 'recall']].describe())
print(df.groupby('transformation')[['f1', 'iou']].mean())
"
```

---

## Common Issues & Solutions

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "v2 (hybrid) checkpoint" error on `--mode test`
You pointed `--checkpoint` at an old `.npy` file. v3's circuit shape
(amplitude-encoded raw pixels + offset register) is different — retrain with
`--mode train` to get a `.npz` checkpoint.

### "Takes 20+ minutes per image"
Expected at default settings — this is the honest cost of exhaustive
circuit-scored pair matching with no classical prefilter. Lower `--img-size`
(e.g. 96) if you need faster iteration, at the cost of localization precision.

### "Out of memory"
```bash
--num-images 3
```

### "Different results each run"
```bash
--seed 42
```

### "Want to use GPU"
```bash
pip install pennylane-lightning-gpu
```

---

## Output Files Explained

### results_pure.csv
- `image_name`, `transformation` — which image and post-processing type
- `precision`, `recall`, `f1`, `iou`, `accuracy` — standard segmentation metrics
- `tp`, `fp`, `fn`, `tn` — confusion matrix counts

### `*_pred_mask.npy`
Binary predicted forgery mask at the working resolution, one per test image.

---

## Getting Help

See:
- **SETUP_GUIDE.md** — Installation troubleshooting
- **COMOFOD_INFO.md** — Dataset structure details
- **README.md** — Project overview and the full "v3: PURE QNN" architecture writeup
- Module docstrings in `cmfd_swaptest_pipeline.py`, `inference.py`, `sift_cmfd.py`
