# Usage Guide - Parameters & Examples

## Basic Command

```bash
python code/cmfd_comofod_pipeline.py --dataset-dir ./data/comofod
```

---

## All Parameters

### Required
```bash
--dataset-dir PATH
```
Path to CoMoFoD dataset (must have `Forged/` and `Masks/` subdirectories)

### Optional

#### `--num-images N` (default: 10)
Number of images to test
```bash
--num-images 5      # Test 5 images
--num-images 100    # Test 100 images
```

#### `--patch-size N` (default: 32)
Size of analysis patches (32×32 recommended)
```bash
--patch-size 16     # Smaller patches (faster but less accurate)
--patch-size 32     # Balanced (recommended)
--patch-size 64     # Larger patches (slower but more context)
```

#### `--stride N` (default: 4)
Step size between patches (controls overlap)
```bash
--stride 1          # Maximum overlap (slow, most accurate)
--stride 2          # High overlap (~15 min/image)
--stride 4          # Balanced (~5 min/image, recommended)
--stride 8          # Low overlap (~1 min/image)
--stride 16         # Minimum overlap (fast but coarse)
```

**Stride Impact:**
- Smaller stride → denser patches → finer localization → SLOWER
- Larger stride → sparse patches → coarser localization → FASTER

#### `--n-qubits N` (default: 5)
Number of qubits in quantum circuit (4-6 practical)
```bash
--n-qubits 4        # 16 amplitudes (faster)
--n-qubits 5        # 32 amplitudes (balanced, recommended)
--n-qubits 6        # 64 amplitudes (more expressive, slower)
```

#### `--n-layers N` (default: 2)
Number of variational circuit layers (1-3 recommended)
```bash
--n-layers 1        # Shallow circuit (faster)
--n-layers 2        # Balanced (recommended)
--n-layers 3        # Deep circuit (more expressive, slower)
```

#### `--output-dir PATH` (default: ./results)
Where to save visualizations and CSV results
```bash
--output-dir ./results_test1
--output-dir /tmp/cmfd_results
```

#### `--start-idx N` (default: 0)
Skip first N images (useful for resuming after interruption)
```bash
--start-idx 0       # Start from first image
--start-idx 10      # Skip first 10, start from image 11
--start-idx 50      # Resume from image 50
```

#### `--seed N` (default: 42)
Random seed for reproducibility
```bash
--seed 42           # Same results each time
--seed 123          # Different random initialization
```

---

## Configuration Examples

### Quick Prototype (3 images, 15 mins)
```bash
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 3 \
    --stride 8 \
    --n-qubits 4 \
    --n-layers 1 \
    --output-dir ./results_prototype
```

### Standard Testing (10 images, 50 mins)
```bash
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 10 \
    --stride 4 \
    --n-qubits 5 \
    --n-layers 2 \
    --output-dir ./results_standard
```

### Publication-Quality (50 images, 5 hours)
```bash
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 50 \
    --stride 2 \
    --n-qubits 6 \
    --n-layers 3 \
    --output-dir ./results_publication
```

### Robustness Test (Test all transformations)
```bash
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 100 \
    --stride 4 \
    --n-qubits 5 \
    --output-dir ./results_robustness
```

### Resume After Interrupt
```bash
# If interrupted at image 25, resume from 25
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 50 \
    --start-idx 25 \
    --output-dir ./results_continued
```

### Batch Processing (Multiple runs)
```bash
# Batch 1: Images 0-9
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 10 \
    --start-idx 0 \
    --output-dir ./results_batch_1 &

# Batch 2: Images 10-19 (parallel)
python code/cmfd_comofod_pipeline.py \
    --dataset-dir ./data/comofod \
    --num-images 10 \
    --start-idx 10 \
    --output-dir ./results_batch_2 &

# Wait for both to finish
wait

# Combine results
cat results_batch_1/results.csv > all_results.csv
tail -n +2 results_batch_2/results.csv >> all_results.csv
```

---

## Speed vs Accuracy Trade-offs

### Fastest (1 min/image)
```bash
--stride 8 --n-qubits 4 --n-layers 1
```
Use when: Prototyping, testing code changes, quick evaluation

### Balanced (5 min/image) ← RECOMMENDED
```bash
--stride 4 --n-qubits 5 --n-layers 2
```
Use when: Standard testing, good quality results needed

### Most Accurate (15 min/image)
```bash
--stride 2 --n-qubits 6 --n-layers 3
```
Use when: Publication quality, final evaluation needed

---

## Analyzing Results

### View CSV Results
```bash
# Linux/Mac
cat results/results.csv

# Windows
type results\results.csv
```

### Statistical Summary
```bash
python -c "
import pandas as pd
df = pd.read_csv('results/results.csv')
print('Overall Statistics:')
print(df[['f1', 'iou', 'accuracy', 'precision', 'recall']].describe())
print('\nBy Transformation:')
print(df.groupby('transformation')[['f1', 'iou']].mean())
"
```

### List Visualizations
```bash
# Count PNGs
ls results/*.png | wc -l

# View with image viewer (Linux/Mac)
open results/          # macOS
xdg-open results/      # Linux

# View with image viewer (Windows)
start results\
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Issue: "Takes 30+ mins per image"
Your QNN config is too large. Try:
```bash
--stride 8 --n-qubits 4 --n-layers 1
```

### Issue: "Out of memory"
Reduce number of images:
```bash
--num-images 3 --stride 8
```

### Issue: "Different results each run"
Use fixed seed:
```bash
--seed 42
```

### Issue: "Want to use GPU"
Install GPU backend:
```bash
pip install pennylane-lightning-gpu
```

---

## Output Files Explained

### results.csv
CSV with columns:
- `image_name` - Filename
- `transformation` - Type (Blur, Rotation, etc.)
- `precision` - TP / (TP + FP)
- `recall` - TP / (TP + FN)
- `f1` - Harmonic mean of precision/recall
- `iou` - Intersection over Union
- `accuracy` - (TP + TN) / total
- `tp`, `fp`, `fn`, `tn` - Confusion matrix

### result_*.png
5-panel visualization:
1. Original forged image
2. QNN heatmap (red = high confidence)
3. Ground truth mask
4. Predicted mask
5. Metrics box

---

## Getting Help

See:
- **SETUP_GUIDE.md** - Installation troubleshooting
- **COMOFOD_INFO.md** - Dataset structure details
- **README.md** - Project overview
- Code comments in `cmfd_comofod_pipeline.py`

Good luck! 🚀
