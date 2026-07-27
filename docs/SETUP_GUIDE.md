# Complete Setup Guide

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- ~2 GB disk space (for CoMoFoD dataset)
- Internet connection (to download dataset)
- Kaggle account (for dataset download)

---

## Step 1: Kaggle API Setup (One-time)

### 1.1 Create Kaggle Account
1. Go to https://www.kaggle.com
2. Click "Sign Up"
3. Create account

### 1.2 Generate API Token
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. This downloads `kaggle.json`

### 1.3 Place API Token
**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```bash
mkdir C:\Users\<YourUsername>\.kaggle
move C:\Users\<YourUsername>\Downloads\kaggle.json C:\Users\<YourUsername>\.kaggle\
```

---

## Step 2: Automatic Setup (Recommended)

```bash
# Navigate to project directory
cd cmfd_project

# Run setup script
bash scripts/setup.sh

# Follow prompts:
# - Installs dependencies
# - Asks about downloading dataset
# - Downloads CoMoFoD (~1.8 GB)
# - Verifies structure
```

**Expected time: 10-15 minutes** (depends on internet speed)

---

## Step 3: Manual Setup (Alternative)

### 3.1 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Install Kaggle CLI
```bash
pip install kaggle
```

### 3.3 Download Dataset
```bash
# Create data directory
mkdir -p data

# Download from Kaggle
kaggle datasets download -d debadityachakraborty/comofod-copy-move-forgery-detection -p data/

# Extract
unzip data/comofod-copy-move-forgery-detection.zip -d data/

# Rename folder
mv data/comofod-copy-move-forgery-detection data/comofod
```

### 3.4 Verify Structure
```bash
# Check images exist
ls data/comofod/Forged/ | head -5
# Should show: original_1_1_1_B.jpg, original_1_1_1_G.jpg, ...

# Check masks exist
ls data/comofod/Masks/ | head -5
# Should show: original_1_1_1_B.png, original_1_1_1_G.png, ...
```

---

## Troubleshooting

### Issue: "No module named 'pennylane'"
```bash
pip install -r requirements.txt
```

### Issue: "Kaggle auth failed"
```bash
# Verify kaggle.json exists
ls ~/.kaggle/kaggle.json

# Check permissions
chmod 600 ~/.kaggle/kaggle.json

# Test authentication
kaggle datasets list | head
```

### Issue: "Connection timeout downloading dataset"
- Try again later (server may be overloaded)
- Or download manually from: https://www.kaggle.com/datasets/debadityachakraborty/comofod-copy-move-forgery-detection

### Issue: "Unzip command not found" (Windows)
```bash
# Use Python instead
python -m zipfile -e data/comofod-copy-move-forgery-detection.zip data/
```

### Issue: "Not enough disk space"
- CoMoFoD is ~1.8 GB
- Free up space and try again
- Or delete other datasets first

---

## Verify Installation

```bash
# Test Python
python -c "print('Python OK')"

# Test dependencies
python -c "import pennylane; print(f'PennyLane {pennylane.__version__} OK')"
python -c "import numpy; print(f'NumPy {numpy.__version__} OK')"

# Test dataset
python -c "from pathlib import Path; print('Dataset exists' if (Path('data/comofod/Forged').exists()) else 'Dataset NOT found')"

# Test code
python code/cmfd_comofod_pipeline.py --help
```

All should print "OK" or show help message.

---

## VS Code Setup

1. **Open project in VS Code:**
   ```bash
   cd cmfd_project
   code .
   ```

2. **Select Python interpreter:**
   - Ctrl+Shift+P → "Python: Select Interpreter"
   - Choose the one you installed dependencies into

3. **Create run configuration (.vscode/launch.json):**
   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Run CMFD Pipeline",
               "type": "python",
               "request": "launch",
               "program": "${workspaceFolder}/code/cmfd_comofod_pipeline.py",
               "console": "integratedTerminal",
               "args": [
                   "--dataset-dir", "./data/comofod",
                   "--num-images", "5",
                   "--output-dir", "./results"
               ]
           }
       ]
   }
   ```

4. **Use Claude Code (Cursor/VS Code):**
   - Select code snippets and ask questions
   - Ask to modify parameters, add features, etc.

---

## Next Steps

Once setup is complete:

1. **Run quick test:**
   ```bash
   python code/cmfd_comofod_pipeline.py --dataset-dir ./data/comofod --num-images 5
   ```

2. **Check results:**
   ```bash
   cat results/results.csv
   ls results/*.png
   ```

3. **Adjust parameters:**
   - Modify `--stride`, `--n-qubits` for speed/accuracy trade-off
   - See USAGE_GUIDE.md for all options

---

## Getting Help

- See **USAGE_GUIDE.md** for parameter explanations
- See **COMOFOD_INFO.md** for dataset details
- Check code comments in `cmfd_comofod_pipeline.py`
- View error messages carefully (they often show the fix)

Good luck! 🚀
