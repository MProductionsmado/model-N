#!/bin/bash

#chmod +x setup_and_train.sh
#./setup_and_train.sh


# =============================================================================
# Minecraft 3D Asset Generator - Full Setup & Training Script
# =============================================================================
# This script will:
# 1. Create a Python virtual environment
# 2. Install all dependencies from requirements.txt
# 3. Download the text encoder model
# 4. Preprocess the schematic dataset
# 5. Start training the discrete diffusion model
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Minecraft 3D Asset Generator Setup${NC}"
echo -e "${BLUE}=============================================${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Working directory: $SCRIPT_DIR${NC}"

# =============================================================================
# Step 1: Create Virtual Environment
# =============================================================================
echo -e "\n${GREEN}[1/5] Creating Python virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}  Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}  ✓ Virtual environment created${NC}"
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}  ✓ Virtual environment activated${NC}"

# Upgrade pip
pip install --upgrade pip --quiet

# =============================================================================
# Step 2: Install Requirements
# =============================================================================
echo -e "\n${GREEN}[2/5] Installing dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}  ✓ Dependencies installed${NC}"
else
    echo -e "${RED}  ✗ requirements.txt not found!${NC}"
    exit 1
fi

# =============================================================================
# Step 3: Preprocess Dataset
# =============================================================================
echo -e "\n${GREEN}[4/5] Preprocessing schematic dataset...${NC}"

# Check if input data exists
if [ -d "out" ] || [ -d "data/processed" ]; then
    # Check if already preprocessed
    if [ -d "data/processed" ] && [ "$(ls -A data/processed 2>/dev/null)" ]; then
        echo -e "${YELLOW}  Preprocessed data already exists.${NC}"
        read -p "  Re-run preprocessing? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python scripts/preprocess_data.py
            echo -e "${GREEN}  ✓ Preprocessing complete${NC}"
        else
            echo -e "${YELLOW}  Skipping preprocessing${NC}"
        fi
    else
        python scripts/preprocess_data.py
        echo -e "${GREEN}  ✓ Preprocessing complete${NC}"
    fi
else
    echo -e "${RED}  ✗ No input data found in 'out/' directory!${NC}"
    echo -e "${RED}    Please place your .schem files in the 'out/' folder.${NC}"
    exit 1
fi

# =============================================================================
# Step 4: Start Training
# =============================================================================
echo -e "\n${GREEN}[5/5] Starting training...${NC}"
echo -e "${BLUE}=============================================${NC}"
echo -e "${YELLOW}Training will now begin. Press Ctrl+C to stop.${NC}"
echo -e "${BLUE}=============================================${NC}\n"

# Check for existing checkpoint to resume
LAST_CHECKPOINT=$(ls -t models/*.ckpt 2>/dev/null | head -n1)

if [ -n "$LAST_CHECKPOINT" ]; then
    echo -e "${YELLOW}Found existing checkpoint: $LAST_CHECKPOINT${NC}"
    read -p "Resume from this checkpoint? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        python scripts/train_discrete_diffusion.py --resume "$LAST_CHECKPOINT"
    else
        python scripts/train_discrete_diffusion.py
    fi
else
    python scripts/train_discrete_diffusion.py
fi

echo -e "\n${GREEN}=============================================${NC}"
echo -e "${GREEN}  Training complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
