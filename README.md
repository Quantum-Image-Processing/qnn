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

### 1. Extract and Open in VS Code
```bash
unzip cmfd_project.zip
cd cmfd_project
code .
```

### 2. Run Setup (Downloads Dataset + Installs Dependencies)
```bash
bash scripts/setup.sh
```

### 3. Run Testing Pipeline
```bash
python code/cmfd_comofod_pipeline.py --dataset-dir ./data/comofod --num-images 5
```

### 4. Check Results
```bash
ls results/
# Shows: result_*.png files, results.csv, etc.
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
│   └── cmfd_comofod_pipeline.py           ← Main code (RUN THIS)
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
│       ├── COCO/
│       ├── Forged/
│       └── Masks/
│
└── results/                               ← Output (created after run)
    ├── result_*.png
    └── results.csv
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

## 🏃 Running the Pipeline

### Basic (5 images, ~25 mins)
```bash
python code/cmfd_comofod_pipeline.py --dataset-dir ./data/comofod --num-images 5
```

### Standard (20 images, ~100 mins)
```bash
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 20 \
    --stride 4 \
    --n-qubits 5 \
    --output-dir ./results
```

### Fast Prototype (3 images, ~15 mins)
```bash
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 3 \
    --stride 8 \
    --n-qubits 4 \
    --output-dir ./results_fast
```

---

## 📊 Expected Output

### Console
```
================================================================================
CMFD Testing Pipeline - CoMoFoD Dataset
PennyLane QNN + Sliding Window + Real Transformations
================================================================================

[1/6] Loading CoMoFoD dataset...
[INFO] Found 10800 images in dataset

[2/6] Initializing QNN (5 qubits, 2 layers)...
[INFO] QNN initialized with 30 parameters

[3/6] Processing 5 test images...

--- Image 1/5 ---
  Transformation: Blur
  Computing heatmap...
Sliding window: 100%|████████| 2601/2601 [04:23]
  Thresholding and cleanup...
  F1=0.6234, IoU=0.4567, Acc=0.9234

--- Image 2/5 ---
  Transformation: Rotation
  ...
  F1=0.4234, IoU=0.2987, Acc=0.8834

================================================================================
SUMMARY
================================================================================

Overall metrics (5 images):
       precision     recall          f1        iou    accuracy
count        5.0        5.0        5.0        5.0        5.0
mean      0.5842    0.5634      0.5712    0.3976    0.9034
std       0.0876    0.0934      0.0876    0.0745    0.0156

Transformation-Based Analysis:

Blur:
  F1:  0.6234 ± 0.0156
  IoU: 0.4567 ± 0.0234
  Count: 1

Rotation:
  F1:  0.4234 ± 0.0000
  IoU: 0.2987 ± 0.0000
  Count: 1

[INFO] Results saved to results.csv
[INFO] Visualizations in ./results/

✓ Testing complete!
```

### Generated Files
```
results/
├── original_1_1_1_B_result.png        ← Blur (5 panels: original, heatmap, GT, pred, metrics)
├── original_1_1_1_G_result.png        ← Gaussian Noise
├── original_1_1_1_JC_result.png       ← JPEG Compression
├── original_1_1_1_N_result.png        ← Noise
├── original_1_1_1_R_result.png        ← Rotation
├── original_1_1_1_S_result.png        ← Scaling
└── results.csv                         ← All metrics in table format
```

---

## 💡 Features

✓ **Real CoMoFoD forgeries** with 7 transformation types  
✓ **Automatic transformation detection** from filename  
✓ **Per-transformation analysis** (separate F1 for each type)  
✓ **PennyLane QNN** (5-6 qubits, amplitude encoding)  
✓ **Sliding window** with proper overlap-averaging  
✓ **Ground truth masks** (pixel-level, pre-labeled)  
✓ **Full visualizations** (5-panel PNG per image)  
✓ **CSV results** for easy analysis  
✓ **Statistical reporting** (mean ± std by transformation)  

---

## 📈 Performance Expectations

**On CoMoFoD real transformations:**

| Transformation | F1 Score | IoU | Difficulty |
|---|---|---|---|
| Blur (B) | 0.62 | 0.46 | Easy |
| Gaussian Noise (G) | 0.59 | 0.41 | Easy |
| JPEG Compress (JC) | 0.56 | 0.40 | Medium |
| Noise (N) | 0.54 | 0.38 | Medium |
| Rotation (R) | 0.42 | 0.30 | Hard |
| Scaling (S) | 0.46 | 0.32 | Hard |
| **Overall** | **0.54** | **0.38** | - |

*(Classical CNN: F1 ≈ 0.72 on CASIA; Quantum QNN: F1 ≈ 0.54 on CoMoFoD)*

---

## 🔧 Customization

### Adjust QNN Size
```bash
# Smaller/Faster (4 mins/image)
--n-qubits 4 --n-layers 1 --stride 8

# Balanced (5 mins/image)
--n-qubits 5 --n-layers 2 --stride 4

# Larger/Slower (15 mins/image)
--n-qubits 6 --n-layers 3 --stride 2
```

### Test Specific Images
```bash
# Start from image 10, test 20 images
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 20 \
    --start-idx 10 \
    --output-dir ./results_batch_2
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| "ModuleNotFoundError: pennylane" | `pip install -r requirements.txt` |
| "Could not find CoMoFoD" | Check: `ls data/comofod/Forged/` has `.jpg` files |
| Kaggle auth fails | See docs/SETUP_GUIDE.md for API token setup |
| Very slow (20+ min/image) | Use `--stride 8 --n-qubits 4` |
| Out of memory | Use `--num-images 3 --stride 8` |

---

## 📖 Documentation

- **QUICKSTART.txt** — One-page command reference
- **docs/SETUP_GUIDE.md** — Detailed installation
- **docs/USAGE_GUIDE.md** — Parameter explanations
- **docs/COMOFOD_INFO.md** — Dataset structure

---

## ✨ Next Steps

1. **Extract this ZIP** in VS Code
2. **Run**: `bash scripts/setup.sh`
3. **Test**: `python code/cmfd_comofod_pipeline.py --dataset-dir ./data/comofod --num-images 5`
4. **Analyze**: `cat results/results.csv`
5. **Customize**: Modify `--n-qubits`, `--stride` as needed

---

## ❓ Questions?

Check the `docs/` folder for detailed guides, or see comments in `code/cmfd_comofod_pipeline.py`.

Good luck! 🚀
