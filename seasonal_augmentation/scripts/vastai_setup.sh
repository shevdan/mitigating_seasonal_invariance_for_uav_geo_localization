#!/bin/bash
# Setup a Vast.ai instance for seasonal augmentation generation.
#
# Upload this script + the data tarball, then run:
#   bash vastai_setup.sh
#
# Prerequisites on Vast.ai:
#   - Minimum 10GB GPU VRAM (RTX 3060+), 50GB disk
#   - SSH access enabled

set -e

echo "=== Vast.ai Setup for Seasonal Augmentation ==="

# Step 1: Create a clean venv with compatible PyTorch
echo "[1/4] Creating virtual environment..."
python -m venv /workspace/myenv
source /workspace/myenv/bin/activate

echo "[2/4] Installing PyTorch (CUDA 12.1 compatible)..."
pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo "[3/4] Installing dependencies..."
pip install -q \
    diffusers==0.36.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    safetensors>=0.4.0 \
    controlnet-aux>=0.0.7 \
    timm>=0.9.0 \
    pyyaml \
    scipy \
    opencv-python-headless \
    tqdm \
    matplotlib \
    Pillow \
    scikit-image

echo "[4/4] Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "IMPORTANT: Always activate the venv before running:"
echo "  source /workspace/myenv/bin/activate"
echo ""
echo "Example generation commands:"
echo "  cd /workspace"
echo "  python seasonal_augmentation/scripts/generate_for_dataset.py \\"
echo "    --dataset uavvisloc \\"
echo "    --dataset-config seasonal_augmentation/configs/datasets_vastai.yaml \\"
echo "    --transformation summer_to_autumn \\"
echo "    --method multicontrolnet \\"
echo "    --skip-depth \\"
echo "    --quality-gate --qg-dino-cls 0.75 --qg-dino-patch 0.72 --qg-edge-f1 0.74"
