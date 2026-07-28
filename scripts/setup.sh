#!/bin/bash

# CMFD CoMoFoD Setup Script
# Installs dependencies and downloads the dataset automatically

echo "=========================================="
echo "CMFD Setup - CoMoFoD Dataset"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Install Python dependencies
echo -e "\n${YELLOW}[1/3] Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Step 2: Install Kaggle CLI
echo -e "\n${YELLOW}[2/3] Installing Kaggle CLI...${NC}"
pip install kaggle

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Kaggle CLI installed${NC}"
else
    echo -e "${RED}✗ Failed to install Kaggle${NC}"
fi

# Step 3: Download CoMoFoD dataset
echo -e "\n${YELLOW}[3/3] Downloading CoMoFoD dataset...${NC}"
echo -e "${YELLOW}Make sure you have:${NC}"
echo "  1. Created a Kaggle account (kaggle.com)"
echo "  2. Generated API token (Settings → Account → Create New API Token)"
echo "  3. Placed kaggle.json in ~/.kaggle/"
echo ""
echo "Continue? (y/n)"
read -r response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    mkdir -p data
    
    echo -e "${YELLOW}Downloading from Kaggle...${NC}"
    kaggle datasets download -d debadityachakraborty/comofod-copy-move-forgery-detection -p data/
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dataset downloaded${NC}"
        
        echo -e "${YELLOW}Extracting dataset...${NC}"
        unzip -q data/comofod-copy-move-forgery-detection.zip -d data/
        
        if [ -d "data/comofod-copy-move-forgery-detection" ]; then
            mv data/comofod-copy-move-forgery-detection data/comofod
        fi
        
        echo -e "${GREEN}✓ Dataset extracted${NC}"
        
        # Verify structure
        echo -e "${YELLOW}Verifying dataset structure...${NC}"
        if [ -d "data/comofod/Forged" ] && [ -d "data/comofod/Masks" ]; then
            echo -e "${GREEN}✓ Dataset structure verified${NC}"
            echo -e "${GREEN}  - Found: data/comofod/Forged/${NC}"
            echo -e "${GREEN}  - Found: data/comofod/Masks/${NC}"
        else
            echo -e "${RED}✗ Dataset structure incorrect${NC}"
            echo "Please check data/comofod/ directory"
            exit 1
        fi
    else
        echo -e "${RED}✗ Failed to download dataset${NC}"
        echo "Check your Kaggle API credentials"
        exit 1
    fi
else
    echo -e "${YELLOW}Skipping dataset download${NC}"
    echo "You can download manually from:"
    echo "https://www.kaggle.com/datasets/debadityachakraborty/comofod-copy-move-forgery-detection"
fi

echo -e "\n${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Try the demo:  cd code && streamlit run app.py"
echo "  2. Or evaluate from the CLI:"
echo "     python code/cmfd_swaptest_pipeline.py --mode test \\"
echo "         --dataset-dir ./data/comofod \\"
echo "         --checkpoint ./checkpoints_pure/swaptest_qnn_epoch10.npz \\"
echo "         --num-images 5 --output-dir ./results_pure"
echo ""
