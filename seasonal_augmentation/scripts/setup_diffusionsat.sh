#!/bin/bash
# Setup DiffusionSat in a SEPARATE conda environment.
#
# DiffusionSat pins old dependencies (torch 2.2.2, diffusers fork 0.17, transformers 4.31)
# that conflict with modern envs, so it needs its own env.
#
# Usage:
#   bash setup_diffusionsat.sh
#
# This creates:
#   - conda env: diffusionsat
#   - repo: /tmp/DiffusionSat
#   - checkpoint: /tmp/diffusionsat_ckpt/

set -e

INSTALL_DIR="/tmp/DiffusionSat"
CKPT_DIR="/tmp/diffusionsat_ckpt"
ENV_NAME="diffusion_sat"

echo "=== DiffusionSat Setup ==="
echo "  Conda env:   $ENV_NAME (Python 3.10)"
echo "  Repo:        $INSTALL_DIR"
echo "  Checkpoint:  $CKPT_DIR"
echo ""

# Step 1: Create dedicated conda env
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/4] Conda env '$ENV_NAME' already exists"
else
    echo "[1/4] Creating conda env '$ENV_NAME' (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -c conda-forge -y
fi

# Step 2: Clone repo
if [ -d "$INSTALL_DIR" ]; then
    echo "[2/4] DiffusionSat repo already exists at $INSTALL_DIR"
else
    echo "[2/4] Cloning DiffusionSat..."
    git clone https://github.com/samar-khanna/DiffusionSat.git "$INSTALL_DIR"
fi

# Step 3: Install dependencies in the dedicated env
echo "[3/4] Installing DiffusionSat dependencies..."
conda run -n "$ENV_NAME" --no-banner pip install \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

cd "$INSTALL_DIR"
conda run -n "$ENV_NAME" --no-banner pip install -e ".[torch]"
conda run -n "$ENV_NAME" --no-banner pip install -r requirements_remaining.txt
conda run -n "$ENV_NAME" --no-banner pip install matplotlib

# Step 4: Download checkpoint (512x512 single-image model, ~5GB)
CKPT_FOLDER="$CKPT_DIR/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"

if [ -d "$CKPT_FOLDER" ]; then
    echo "[4/4] Checkpoint already exists at $CKPT_FOLDER"
else
    echo "[4/4] Downloading DiffusionSat checkpoint (512x512)..."
    echo "      This is ~5GB, may take a while..."
    mkdir -p "$CKPT_DIR"
    CKPT_ZIP="$CKPT_DIR/diffusionsat_512.zip"
    wget -q --show-progress -O "$CKPT_ZIP" \
        "https://zenodo.org/records/13751498/files/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64.zip"
    echo "      Extracting..."
    unzip -q "$CKPT_ZIP" -d "$CKPT_DIR"
    rm -f "$CKPT_ZIP"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run the test:"
echo "  conda run -n $ENV_NAME python seasonal_augmentation/scripts/test_diffusionsat.py \\"
echo "    --ckpt $CKPT_FOLDER \\"
echo "    --repo $INSTALL_DIR"
echo ""
echo "With a real UAV image for comparison:"
echo "  conda run -n $ENV_NAME python seasonal_augmentation/scripts/test_diffusionsat.py \\"
echo "    --ckpt $CKPT_FOLDER \\"
echo "    --repo $INSTALL_DIR \\"
echo "    --image data/DenseUAV/drone/some_image.jpg"
