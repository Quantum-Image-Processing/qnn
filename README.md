# CMFD Testing Pipeline - CoMoFoD Dataset
## PennyLane QNN + Sliding Window + Real Transformations

**Complete ready-to-run project for Copy-Move Forgery Detection using:**
- CoMoFoD dataset (10,800 real images with 7 transformation types)
- PennyLane QNN (amplitude encoding, no PyTorch)
- Sliding window + heatmap aggregation
- Per-transformation analysis (rotation, scaling, JPEG, noise, blur)

---

## ⚛️ v3: PURE QNN (`code/cmfd_swaptest_pipeline.py`)

The swap-test pipeline is now **quantum end to end**. Raw pixels go into the
circuit, a measured expectation value comes out, and no classical algorithm
makes any part of the detection decision.

### What was removed to get there

| Stage | v2 (hybrid) | v3 (pure) |
|---|---|---|
| Patch representation | DCT coefficients + intensity histogram | `AmplitudeEmbedding` of the **raw pixels** |
| Candidate pairs | Fridrich lexicographic block matching (classical sort) | **Exhaustive** — every admissible pair scored by the circuit |
| Match filtering | Classical (dy, dx) Hough vote | **In-circuit offset register** + trainable CRY readout |
| Mask cleanup | `binary_dilation` / `binary_erosion` | none |
| Trainable params | circuit angles only | circuit angles only (unchanged — v2 was already pure *here*) |

The only classical operation left anywhere is connected-component labelling in
`inference.py`, used solely to draw bounding boxes for the UI. It runs after
the decision and feeds nothing back into it.

### Architecture

```
raw patch A ──► AmplitudeEmbedding ──► re-upload layers (tied weights) ─┐
raw patch B ──► AmplitudeEmbedding ──► re-upload layers (tied weights) ─┼─► swap test ──► <Z> ──► score
(dy, dx)    ──► offset register     ──► trainable CRY ──────────────────┘
```

Wires = `2 * log2(patch_size²) + n_offset_qubits + 1`. At the default
`--patch-size 8` that is **15 wires** (6 per register + 2 offset + 1 ancilla)
and 58 trainable rotation angles.

**Do not replace the re-uploading ansatz with a plain embedding + tied
unitary.** Under a swap test, `|<U(w)f_a|U(w)f_b>|² = |<f_a|f_b>|²` for *any*
`U(w)`, so the gradient w.r.t. `w` is exactly zero. This was measured, not
assumed. Re-uploading the pixels into the rotation angles is what keeps the
weights alive (verified gradient norm ~1.2e-2, vs ~1e-16 for the degenerate
form).

### Running it

```bash
# train (checkpoints are .npz — v2 .npy weights are a different circuit and are rejected)
python code/cmfd_swaptest_pipeline.py --mode train \
    --dataset-dir ./data/comofod \
    --epochs 10 --images-per-epoch 20 --lr 0.05 \
    --checkpoint-dir ./checkpoints_pure

# test  (defaults are already the tiled config: --img-size 128 --patch-size 8 --stride 8)
python code/cmfd_swaptest_pipeline.py --mode test \
    --dataset-dir ./data/comofod \
    --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz \
    --num-images 5
```

`--n-qubits` is **gone**: qubits per register are derived from `--patch-size`
(8 → 6 qubits, 16 → 8 qubits). Simulation cost is exponential in total wires,
so `--patch-size 16` is ~23× slower per pair than `--patch-size 8`.

### ⚠️ `--stride` must equal `--patch-size`

This is a **correctness requirement, not a speed knob.** A larger stride leaves
gaps between patches, so part of every image is never examined at all.

Measured on `001_F` (forged region = 2.0% of the image, 256px working size):

| stride | patches | pixel coverage | patches inside forged region |
|---|---|---|---|
| 32 | 64 | 6.2% | **3** |
| 16 | 256 | 25.0% | **4** |
| 8 (tiled) | 1024 | 100% | 13 |

At stride 32 the detector returned **byte-identical output for a forged image
and its authentic original** (15 matched pairs, mean score 0.940) — it never
sampled the forgery, and every "match" was natural self-similarity elsewhere.
That, not the threshold, is why the first stride-32 run scored F1 = 0.0000.
The pipeline now prints a `[WARN]` if `stride > patch-size`.

### Cost (measured, lightning.qubit, patch-size 8, tiled)

| working size | patches | pairs | time/image |
|---|---|---|---|
| 96 | 144 | ~10k | ~6 min |
| **128 (default)** | 256 | ~33k | **~20 min** |
| 256 | 1024 | ~520k | ~5 h |

Dropping the classical prefilter made pair scoring **O(N²)**, so `--img-size`
is now the cost dial (tiling fixes `--stride`). The quantum replacement for
that prefilter is Grover/amplitude amplification over a pair-index register,
which needs QRAM to hold all patches in superposition — not simulable here, so
it is documented as future work rather than faked with a classical stand-in.
Working size dropped from 256 to 128 purely to keep tiled scoring affordable;
that trades spatial precision for the ability to examine the whole image.

### Honest limitations

- **The offset register is weaker than the vote it replaced.** Offset
  consistency is a *global* property of the pair ensemble ("many pairs share
  one offset"); a per-pair circuit cannot see clustering it is never shown.
  Expect more false positives on homogeneous regions (sky, walls) than v2. A
  faithful quantum version needs quantum counting over all pairs at once.
- **Translation only.** A rot90 duplicate permutes the amplitude vector into a
  nearly orthogonal state and scores ~0.01 where an exact copy scores ~1.0.
  Rotated positives are therefore excluded from synthetic training by default
  (`--train-rotations` re-enables them); including them pins training loss
  above the 0.25 constant-predictor floor, because they are targets the
  architecture provably cannot represent. **This was the actual cause of the
  v2 model's reported non-convergence** — with rotations off, loss drops to
  ~0.003–0.10.
- **Small forgeries are near the resolution floor.** At 128px tiled, a
  2%-area forged region contains only ~4 patches, and copy-move needs a matched
  *pair*, so the achievable F1 ceiling on small forgeries is low regardless of
  the model. Raise `--img-size` to improve this, at O(N²) cost.
- **Training positives may be too easy.** Synthetic positives are pixel-identical
  copies, so training loss reached ~0.047 in the first epoch and barely moved
  after. Real CoMoFoD forgeries carry JPEG and paste-boundary artifacts that
  this signal does not model, so the trained similarity may not transfer.
  This is a data-realism gap, separate from the classical-vs-quantum question.
- Purity costs accuracy here; the report should say so rather than hide it.

---

## 🚀 Quick Start (5 Minutes)

### 1. Open in VS Code
```bash
cd cmfd_project
code .
```

### 2. Run Setup (Downloads Dataset + Installs Dependencies)
```bash
bash scripts/setup.sh
```

### 3. Run the Demo
```bash
cd code
streamlit run app.py
```
Upload an image, pick an engine in the sidebar (**Pure classical · SIFT+RANSAC**
or **Pure quantum · swap-test QNN**), and click **Run analysis**. A trained
QNN checkpoint already ships at `checkpoints_pure/swaptest_qnn_epoch10.npz`,
so the quantum engine works out of the box — no training required first.

### 4. Or Evaluate on CoMoFoD from the Command Line
```bash
python code/cmfd_swaptest_pipeline.py --mode test \
    --dataset-dir ./data/comofod \
    --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz \
    --num-images 5 --output-dir ./results_pure
cat results_pure/results_pure.csv
```

---

## 📁 Project Structure

```
cmfd_project/
├── README.md                              ← You are here
├── QUICKSTART.txt                         ← One-page reference
├── requirements.txt                       ← Python dependencies
│
├── code/
│   ├── app.py                             ← Streamlit demo (SIFT + pure-QNN engines)
│   ├── inference.py                       ← Pure-QNN inference wrapper used by app.py
│   ├── sift_cmfd.py                       ← Classical SIFT+RANSAC baseline engine
│   ├── cmfd_swaptest_pipeline.py          ← v3 PURE QNN: model, training, CoMoFoD eval
│   ├── train_split.py                     ← Train/test-split trainer, saves a deployable checkpoint
│   ├── grip_dataset.py                    ← GRIP/COVERAGE dataset loader (alt. eval sets)
│   └── test_grip.py                       ← Evaluate the trained QNN on GRIP/COVERAGE
│
├── checkpoints_pure/                      ← Trained pure-QNN weights (.npz, per epoch)
│
├── scripts/
│   └── setup.sh                           ← Auto setup + download
│
├── docs/
│   ├── SETUP_GUIDE.md                     ← Detailed guide
│   ├── USAGE_GUIDE.md                     ← Parameters explained
│   └── COMOFOD_INFO.md                    ← Dataset info
│
├── data/
│   └── comofod/                           ← Dataset (after download)
│
└── results_pure/                          ← Output (created after a --mode test run)
    ├── *_pred_mask.npy
    └── results_pure.csv
```

---

## ⚙️ Installation

### Automatic (Recommended)
```bash
bash scripts/setup.sh
```

### Manual
```bash
# Install dependencies
pip install -r requirements.txt

# Download CoMoFoD dataset
pip install kaggle
kaggle datasets download -d debadityachakraborty/comofod-copy-move-forgery-detection
unzip comofod-copy-move-forgery-detection.zip -d data/
mv data/comofod-copy-move-forgery-detection data/comofod
```

---

## 🏃 Running It

### Demo (both engines, interactive)
```bash
cd code
streamlit run app.py
```

### Train the pure QNN from scratch
```bash
# self-supervised on synthetic copy-move pairs, train/test split by scene id
python code/train_split.py --dataset-dir ./data/comofod \
    --epochs 10 --images-per-epoch 80 --out ./models/cmfd_model.npz
```
Or drive `cmfd_swaptest_pipeline.py` directly — this is what produced the
checkpoints already in `checkpoints_pure/`:
```bash
python code/cmfd_swaptest_pipeline.py --mode train \
    --dataset-dir ./data/comofod \
    --epochs 10 --images-per-epoch 20 --lr 0.05 \
    --checkpoint-dir ./checkpoints_pure
```

### Evaluate the pure QNN on CoMoFoD
```bash
python code/cmfd_swaptest_pipeline.py --mode test \
    --dataset-dir ./data/comofod \
    --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz \
    --num-images 5 --output-dir ./results_pure
```

### Evaluate on GRIP / COVERAGE instead of CoMoFoD
```bash
python code/test_grip.py \
    --dataset-dir ./data/grip \
    --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz \
    --patch-size 8 --n-layers 3 \
    --num-images 20 --output-dir ./results_grip
```

See the **v3: PURE QNN** section above for what each flag controls, expected
runtime per image, and why `--stride` must equal `--patch-size`.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| "ModuleNotFoundError: pennylane" / "streamlit" / "cv2" | `pip install -r requirements.txt` |
| "Could not find CoMoFoD" | Check `data/comofod/` has either `Forged/`+`Masks/` or a flat `CoMoFoD_small_v2/` folder |
| Kaggle auth fails | See docs/SETUP_GUIDE.md for API token setup |
| QNN mode very slow (~20 min/image) | Expected — see the cost table in the v3 section above; lower `--img-size` or use the SIFT engine in the demo |
| "v2 (hybrid) checkpoint" load error | You pointed `--checkpoint` at an old `.npy` file; v3 needs a `.npz` checkpoint, retrain with `--mode train` |
| Out of memory | Use `--num-images 3` and a smaller `--img-size` |

---

## 📖 Documentation

- **QUICKSTART.txt** — One-page command reference
- **docs/SETUP_GUIDE.md** — Detailed installation
- **docs/USAGE_GUIDE.md** — Parameter explanations
- **docs/COMOFOD_INFO.md** — Dataset structure

---

## ❓ Questions?

Check the `docs/` folder for detailed guides, or see the module docstrings in
`code/cmfd_swaptest_pipeline.py`, `code/inference.py`, and `code/sift_cmfd.py`
— each explains its own design decisions in depth.

Good luck! 🚀
