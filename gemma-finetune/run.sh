#!/bin/bash
# ============================================================================
# RUN.SH — One-Click Launcher for Gemma Fine-Tuning Pipeline
# ============================================================================
#
# PURPOSE:
#   Runs the entire pipeline: env check → train → evaluate
#
# USAGE:
#   chmod +x run.sh
#   bash run.sh
#
# PREREQUISITES:
#   1. Python >= 3.10
#   2. NVIDIA GPU with >= 8 GB VRAM
#   3. Dependencies installed: pip install -r requirements.txt
#   4. Hugging Face token: export HF_TOKEN=your_token_here
#
# ============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          GEMMA FINE-TUNING PIPELINE LAUNCHER           ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Step 0: Check HF_TOKEN
# ============================================================================
echo -e "${YELLOW}[Step 0/4]${NC} Checking Hugging Face token..."
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}ERROR: HF_TOKEN is not set!${NC}"
    echo ""
    echo "  Gemma requires authentication. Set your token:"
    echo "    export HF_TOKEN=hf_your_token_here"
    echo ""
    echo "  Get your token: https://huggingface.co/settings/tokens"
    echo "  Accept Gemma license: https://huggingface.co/google/gemma-2b"
    echo ""
    exit 1
fi
echo -e "${GREEN}  ✅ HF_TOKEN is set${NC}"

# ============================================================================
# Step 1: Environment Check
# ============================================================================
echo ""
echo -e "${YELLOW}[Step 1/4]${NC} Running environment checks..."
echo ""
python setup_environment.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}Environment check found issues. Fix them before continuing.${NC}"
    echo -e "${YELLOW}You can try running training anyway with: python train.py${NC}"
    read -p "Continue anyway? (y/N): " choice
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        exit 1
    fi
fi

# ============================================================================
# Step 2: Training
# ============================================================================
echo ""
echo -e "${YELLOW}[Step 2/4]${NC} Starting fine-tuning..."
echo ""
echo "  This will take 30-60 minutes on a single GPU with default settings."
echo "  You'll see training loss values — they should decrease over time."
echo ""

python train.py

# Find the most recent output directory
LATEST_DIR=$(ls -td ./outputs/run_* 2>/dev/null | head -1)
MODEL_DIR="${LATEST_DIR}/final_model"

if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}ERROR: Model directory not found at ${MODEL_DIR}${NC}"
    echo "  Training may have failed. Check the logs above."
    exit 1
fi

echo -e "${GREEN}  ✅ Training complete! Model saved to: ${MODEL_DIR}${NC}"

# ============================================================================
# Step 3: Evaluation
# ============================================================================
echo ""
echo -e "${YELLOW}[Step 3/4]${NC} Running evaluation..."
echo ""

python evaluate.py --model_dir "$MODEL_DIR" --max_samples 30

echo -e "${GREEN}  ✅ Evaluation complete!${NC}"

# ============================================================================
# Step 4: Demo Inference
# ============================================================================
echo ""
echo -e "${YELLOW}[Step 4/4]${NC} Running demo inference..."
echo ""

python inference.py \
    --model_dir "$MODEL_DIR" \
    --prompt "This wireless charger is fantastic! Charges my phone quickly and the LED indicator is really helpful. Build quality feels premium. Only downside is it gets a bit warm during extended use." \
    --rating 4.5 \
    --title "Wireless Charger Pro" \
    --category "Electronics"

# ============================================================================
# Done!
# ============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              PIPELINE COMPLETE! 🎉                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Model saved at:    ${MODEL_DIR}"
echo "  Eval results at:   ${MODEL_DIR}/evaluation_results.json"
echo ""
echo "  To run interactive inference:"
echo "    python inference.py --model_dir ${MODEL_DIR} --interactive"
echo ""
echo "  To run on your own reviews:"
echo "    python inference.py --model_dir ${MODEL_DIR} --prompt 'Your review here' --rating 4"
echo ""
