#!/bin/bash

# =============================================================================
# Minecraft 3D Asset Generator - Full Setup & Training Script
# =============================================================================
# Usage:
#   ./setup.sh              # Interactive mode - choose which model to train
#   ./setup.sh normal       # Train only NORMAL model (16x16x16)
#   ./setup.sh big          # Train only BIG model (16x32x16)
#   ./setup.sh huge         # Train only HUGE model (24x64x24)
#   ./setup.sh all          # Train all three models sequentially
#   ./setup.sh preprocess   # Only run preprocessing (no training)
# git clone --branch v2-rtx4090-optimized --single-branch https://github.com/MProductionsmado/model-N
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}=============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================${NC}"
}

print_model_info() {
    case $1 in
        normal)
            echo -e "${CYAN}  Size: 16x16x16 (4,096 voxels)${NC}"
            echo -e "${CYAN}  Use case: Small structures, bushes, rocks${NC}"
            echo -e "${CYAN}  VRAM: ~6 GB | Batch: 64${NC}"
            ;;
        big)
            echo -e "${CYAN}  Size: 16x32x16 (8,192 voxels)${NC}"
            echo -e "${CYAN}  Use case: Standard trees, small houses${NC}"
            echo -e "${CYAN}  VRAM: ~12 GB | Batch: 24${NC}"
            ;;
        huge)
            echo -e "${CYAN}  Size: 24x64x24 (36,864 voxels)${NC}"
            echo -e "${CYAN}  Use case: Giant trees, complex builds${NC}"
            echo -e "${CYAN}  VRAM: ~22 GB | Batch: 4${NC}"
            ;;
    esac
}
train_model() {
    local SIZE=$1
    print_header "Training $SIZE model"
    print_model_info $SIZE
    
    # Check for existing checkpoint
    CHECKPOINT_DIR="models/checkpoints/$SIZE"
    LAST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/*.ckpt 2>/dev/null | head -n1 || true)
    
    if [ -n "$LAST_CHECKPOINT" ]; then
        echo -e "\n${YELLOW}Found existing checkpoint: $LAST_CHECKPOINT${NC}"
        read -p "  Resume from this checkpoint? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            python scripts/train_discrete_diffusion.py --size $SIZE --resume "$LAST_CHECKPOINT"
        else
            python scripts/train_discrete_diffusion.py --size $SIZE
        fi
    else
        python scripts/train_discrete_diffusion.py --size $SIZE
    fi
    
    echo -e "${GREEN}  ✓ $SIZE model training complete${NC}"
}

# =============================================================================
# Main Script
# =============================================================================

print_header "Minecraft 3D Asset Generator Setup"
echo -e "${YELLOW}Working directory: $SCRIPT_DIR${NC}"

# =============================================================================
# Step 1: Create Virtual Environment
# =============================================================================
echo -e "\n${GREEN}[1/4] Setting up Python environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${YELLOW}  Virtual environment already exists.${NC}"
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
echo -e "\n${GREEN}[2/4] Installing dependencies...${NC}"

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
echo -e "\n${GREEN}[3/4] Preprocessing schematic dataset...${NC}"

if [ -d "out" ] || [ -d "data/processed" ]; then
    if [ -d "data/processed" ] && [ "$(ls -A data/processed 2>/dev/null)" ]; then
        echo -e "${YELLOW}  Preprocessed data already exists.${NC}"
        if [ "$1" != "preprocess" ]; then
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
        python scripts/preprocess_data.py
        echo -e "${GREEN}  ✓ Preprocessing complete${NC}"
    fi
else
    echo -e "${RED}  ✗ No input data found in 'out/' directory!${NC}"
    echo -e "${RED}    Please place your .schem files in the 'out/' folder.${NC}"
    exit 1
fi

# Exit if only preprocessing was requested
if [ "$1" == "preprocess" ]; then
    echo -e "\n${GREEN}Preprocessing complete. Exiting.${NC}"
    exit 0
fi

# =============================================================================
# Step 4: Training
# =============================================================================
echo -e "\n${GREEN}[4/4] Model Training${NC}"

# Create checkpoint directories
mkdir -p models/checkpoints/normal
mkdir -p models/checkpoints/big
mkdir -p models/checkpoints/huge

# Determine which model(s) to train
case "${1:-interactive}" in
    normal)
        train_model normal
        ;;
    big)
        train_model big
        ;;
    huge)
        train_model huge
        ;;
    all)
        echo -e "${YELLOW}Training all three models sequentially...${NC}"
        train_model normal
        train_model big
        train_model huge
        ;;
    interactive|*)
        print_header "Select Model to Train"
        echo -e "${CYAN}  1) normal  - 16x16x16 (fastest, smallest structures)${NC}"
        echo -e "${CYAN}  2) big     - 16x32x16 (trees, small buildings)${NC}"
        echo -e "${CYAN}  3) huge    - 24x64x24 (giant trees, complex builds)${NC}"
        echo -e "${CYAN}  4) all     - Train all three sequentially${NC}"
        echo -e "${CYAN}  5) exit    - Exit without training${NC}"
        echo
        read -p "Enter choice [1-5]: " -n 1 -r
        echo
        
        case $REPLY in
            1) train_model normal ;;
            2) train_model big ;;
            3) train_model huge ;;
            4)
                train_model normal
                train_model big
                train_model huge
                ;;
            5)
                echo -e "${YELLOW}Exiting without training.${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
        ;;
esac

print_header "Training Complete!"
echo -e "${GREEN}Checkpoints saved in: models/checkpoints/{normal,big,huge}/${NC}"
