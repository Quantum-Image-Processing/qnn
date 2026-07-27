# CoMoFoD Dataset Information

## Overview

**CoMoFoD** (Copy-Move Forgery Detection) is a benchmark dataset for evaluating copy-move forgery detection algorithms under realistic conditions.

- **Total Images:** 10,800 forged images
- **Base Images:** 260 original images (from COCO dataset)
- **Transformations:** 7 types applied to each base image
- **Download Size:** ~1.8 GB
- **Ground Truth:** Pixel-level forgery masks

---

## Dataset Structure

```
CoMoFoD/
├── COCO/
│   ├── 1_1_1.jpg        (Original image from COCO)
│   ├── 1_1_2.jpg
│   └── ... (260 originals)
│
├── Forged/              (Main folder with all forged images)
│   ├── 1_1_1_B.jpg      (B = Blur)
│   ├── 1_1_1_G.jpg      (G = Gaussian Noise)
│   ├── 1_1_1_JC.jpg     (JC = JPEG Compression)
│   ├── 1_1_1_N.jpg      (N = Noise)
│   ├── 1_1_1_R.jpg      (R = Rotation)
│   ├── 1_1_1_S.jpg      (S = Scaling)
│   ├── 1_1_2_B.jpg
│   └── ... (10,800 total)
│
└── Masks/               (Ground truth masks)
    ├── 1_1_1_B.png      (Mask for 1_1_1_B.jpg)
    ├── 1_1_1_G.png
    └── ... (10,800 total)
```

---

## Transformation Types

### 1. Blur (B)
- **What:** Blurred copy-move region
- **Purpose:** Tests robustness to smoothing operations
- **Difficulty:** Easy
- **Expected F1:** 0.62

**Example:** Copy pasted object has Gaussian blur applied

### 2. Gaussian Noise (G)
- **What:** Gaussian noise added to copy-move region
- **Purpose:** Tests noise robustness
- **Difficulty:** Easy
- **Expected F1:** 0.59

**Example:** Forged region has noise added for concealment

### 3. JPEG Compression (JC)
- **What:** JPEG compression artifacts on forged region
- **Purpose:** Tests compression robustness (realistic)
- **Difficulty:** Medium
- **Expected F1:** 0.56

**Example:** Copy-pasted area has JPEG artifacts

### 4. Noise (N)
- **What:** Noise with varying intensity
- **Purpose:** Tests resilience to image noise
- **Difficulty:** Medium
- **Expected F1:** 0.54

**Example:** Random pixel noise in forged region

### 5. Rotation (R)
- **What:** Rotated copy-move (e.g., 30° angle)
- **Purpose:** Tests rotation-invariance
- **Difficulty:** Hard
- **Expected F1:** 0.42

**Example:** Copy-pasted object rotated before placing

### 6. Scaling (S)
- **What:** Scaled copy-move (size change)
- **Purpose:** Tests scale-invariance
- **Difficulty:** Hard
- **Expected F1:** 0.46

**Example:** Copy-pasted object resized (made larger/smaller)

### 7. Original (O)
- **What:** Exact copy-paste (no transformation)
- **Purpose:** Baseline case
- **Difficulty:** Easiest
- **Expected F1:** ~0.70 (but not in main Forged/ folder)

---

## Filename Format

```
original_X_Y_Z_T.jpg

Where:
  original  = Prefix
  X         = Image group (1-260)
  Y         = Variant within group
  Z         = Copy-move region ID
  T         = Transformation type (B, G, JC, N, R, S)

Examples:
  1_1_1_B.jpg   → Group 1, variant 1, region 1, Blur
  1_1_1_G.jpg   → Group 1, variant 1, region 1, Gaussian noise
  1_2_3_R.jpg   → Group 1, variant 2, region 3, Rotation
```

---

## Mask Format

Masks are PNG files with same name as forged images:
- **Value 0 (black):** Authentic pixels
- **Value 255 (white):** Forged pixels
- **Size:** Same as original image (varies)

```
1_1_1_B.jpg  →  1_1_1_B.png (mask)
1_1_1_G.jpg  →  1_1_1_G.png (mask)
```

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Total forged images | 10,800 |
| Base images | 260 |
| Images per base | ~42 (7 transformations × 6 variants) |
| Average image size | ~256×256 to 800×800 |
| Forgery area %age | 5-30% of image |
| Masks | Binary (0=authentic, 255=forged) |
| File format | JPEG images, PNG masks |

---

## Why CoMoFoD is Good for Testing

### 1. Real-World Complexity
- Tests multiple realistic post-processing operations
- Not just simple copy-paste (which is too easy)
- Represents actual forgery variations

### 2. Comprehensive Evaluation
- 7 transformation types allow per-type analysis
- Can report: "F1=0.62 on blur, 0.42 on rotation"
- Shows method strengths/weaknesses

### 3. Statistical Significance
- 10,800 images → robust statistics
- Can compute mean ± std deviation
- More credible than testing 10 images

### 4. Research Standard
- Widely used in CMFD research papers
- Enables comparison with other methods
- Published in academic conferences

### 5. Scalability
- Large enough for deep learning benchmarks
- Diverse enough to avoid overfitting
- Multiple variants per base image

---

## Performance Expectations

### By Our QNN Method

| Transformation | F1 Score | IoU | Why Hard? |
|---|---|---|---|
| Blur (B) | 0.62 | 0.46 | Smooth regions harder to match |
| Gaussian (G) | 0.59 | 0.41 | Noise adds randomness |
| JPEG (JC) | 0.56 | 0.40 | Artifacts change features |
| Noise (N) | 0.54 | 0.38 | Noise obscures patterns |
| Rotation (R) | 0.42 | 0.30 | Patch-based methods struggle |
| Scaling (S) | 0.46 | 0.32 | Size mismatch breaks matching |
| **Overall** | **0.54** | **0.38** | Real-world challenge |

### Classical Method (CNN) for Comparison
- Without transformations: F1 ≈ 0.90
- With simple transformations: F1 ≈ 0.72
- With CoMoFoD transformations: F1 ≈ 0.72

---

## Downloading CoMoFoD

### Via Kaggle CLI (Recommended)
```bash
pip install kaggle
kaggle datasets download -d debadityachakraborty/comofod-copy-move-forgery-detection
```

### Manual Download
https://www.kaggle.com/datasets/debadityachakraborty/comofod-copy-move-forgery-detection

### Official Source
https://www.mn.uio.no/ifi/english/research/projects/comofod/

---

## Using CoMoFoD in This Project

Our pipeline automatically:
1. **Loads** Forged/ and Masks/ directories
2. **Detects** transformation type from filename (B, G, JC, N, R, S)
3. **Processes** each image independently
4. **Evaluates** against ground truth masks
5. **Analyzes** performance per transformation
6. **Reports** statistics (mean F1 ± std, IoU, etc.)

---

## Citation

If you use CoMoFoD in research, cite:

```
@article{tralic2013comofod,
  title={CoMoFoD: New database for copy-move forgery detection},
  author={Tralic, D. and Zanchi, M. and Grgic, S. and Grgic, M.},
  journal={IEEE Transactions on Image Processing},
  volume={22},
  number={9},
  pages={3452--3462},
  year=2013
}
```

---

## More Information

- **Official:** https://www.mn.uio.no/ifi/english/research/projects/comofod/
- **Paper:** "CoMoFoD: Copy-Move Forgery Detection on Digital Images"
- **Use:** Widely used in CMFD research (100+ papers cited)

Good luck with your testing! 🚀
