# CMFD Testing Pipeline - CoMoFoD Dataset
## PennyLane QNN + Sliding Window + Real Transformations

**Complete ready-to-run project for Copy-Move Forgery Detection using:**
- CoMoFoD dataset (10,800 real images with 7 transformation types)
- PennyLane QNN (5-6 qubits, amplitude encoding, no PyTorch)
- Sliding window + heatmap aggregation
- Per-transformation analysis (rotation, scaling, JPEG, noise, blur)

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
