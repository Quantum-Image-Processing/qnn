# Copy-Move Forgery Detection with a Siamese Swap-Test QNN

FYP Phase 2 (Amrita). Pure quantum neural network for copy-move forgery
detection on CoMoFoD, simulated in PennyLane.

The current model is the **9-wire Zernike QNN**, which runs in four phases:

1. **Ingestion and pre-processing** (classical). 128×128 grayscale image,
   cut into 256 patches of 8×8, each turned into 12 Zernike magnitudes on 4 qubits.
2. **QNN circuit**. Two 4-qubit registers with 3 data re-uploading layers
   that share 36 trained rotation angles.
3. **Swap test and Siamese QNN**. A fixed swap test on one ancilla gives the
   pair score ⟨Z⟩ = |⟨φi|φj⟩|².
4. **Batch extraction and output**. All 32,640 pair scores per image, then a
   threshold of 0.6 and a verdict at 8,192 matched pairs.

What it does and does not deliver (measured, see the paper):

- **Speed and rotation handling**: yes, as a classical pre-processing step.
  15 → 9 wires, about 15–20 min → under 1 s per image in the dashboard.
  A 45°-rotated copy finds its source 93–94.5% of the time, against 3% with raw pixels.
- **Better detection**: no. Held-out image accuracy is 53.8% (AUC 0.50).

The earlier 15-wire raw-pixel model (v3, `code/cmfd_swaptest_pipeline.py`) is
kept unchanged. The Zernike model is built on it, and it is the comparison baseline.

## Layout

```
code/
  cmfd_swaptest_pipeline.py   v3 15-wire ST-QNN (base class, synthetic pairs, CoMoFoD loader)
  cmfd_zernike_swaptest.py    9-wire Zernike QNN: zernike_matrix, zernike_state, ZernikeSwapQNN
  train_zernike.py            training (Adam, cosine learning rate, fixed validation pairs)
  calibrate_zernike.py        picks the threshold and pair count on validation scenes
  evaluate_zernike.py         speed, rotation and detection against v3
  inference.py                v3 detector used by the dashboard
  inference_zernike.py        Zernike detector used by the dashboard
  sift_cmfd.py                classical SIFT + RANSAC engine for the dashboard
  train_split.py              scene-level train/test split helper
  app.py                      Streamlit dashboard (SIFT, v3, Zernike)
checkpoints_pure/             trained v3 weights, history, split
models/                       trained Zernike weights, calibration, split, training history
scripts/setup.sh              installs requirements and downloads CoMoFoD
scripts/paper_pure/           scripts behind the paper's measured numbers
docs/figures/gen_*.py         generate the paper's five figures
paper/paper/main.tex          the paper (compiled on Overleaf)
paper/paper/Paper-img/        the five figures the paper uses
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The dataset is not in git. Put CoMoFoD_small_v2 at `data/comofod/CoMoFoD_small_v2/`,
or run `bash scripts/setup.sh`, which needs a Kaggle API token.

## Run

```bash
cd code

# dashboard: choose "Zernike QNN · 9-wire swap test"
streamlit run app.py

# retrain the 9-wire model (about 2 minutes on one CPU thread),
# then calibrate and evaluate it
python train_zernike.py --dataset-dir ../data/comofod
python calibrate_zernike.py --dataset-dir ../data/comofod --checkpoint ../models/zernike_9w.npz
python evaluate_zernike.py --dataset-dir ../data/comofod --checkpoint ../models/zernike_9w.npz
```

Regenerate the figures from the repo root:

```bash
python docs/figures/gen_zernike_modules.py      # phase 1, 2 and 4 figures
python docs/figures/gen_siamese_swaptest.py     # phase 3 figure (needs the model and data)
python docs/figures/gen_overall_pipeline.py     # overall-system figure (needs the data)
```

They write SVGs to `docs/figures/`. The paper uses JPEG copies in `paper/paper/Paper-img/`.

## Not in git

Ignored in `.gitignore`: the dataset (`data/`), the virtual environment
(`.venv/`), per-epoch Zernike checkpoints (regenerate with `train_zernike.py`),
generated figure SVGs and PNGs in `docs/figures/`, and LaTeX build output.
