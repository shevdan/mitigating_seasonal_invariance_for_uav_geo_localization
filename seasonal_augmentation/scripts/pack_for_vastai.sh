#!/bin/bash
# Pack everything needed for Vast.ai into a single tarball.
#
# Creates: ~/denseuav_bundle.tar.gz (~9GB)
#
# On Vast.ai, extracts to:
#   /workspace/seasonal_augmentation/scripts/
#   /workspace/seasonal_augmentation/configs/
#   /workspace/seasonal_augmentation/outputs/denseuav/depth_maps/
#   /workspace/data/DenseUAV/train/drone/
#   /workspace/data/DenseUAV/test/query_drone/
#
# Usage:
#   cd ~/Documents/msc/diploma
#   bash seasonal_augmentation/scripts/pack_for_vastai.sh

set -e

DIPLOMA_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIPLOMA_DIR"

OUTPUT="$HOME/denseuav_bundle.tar.gz"

echo "=== Packing data for Vast.ai ==="
echo "Source: $DIPLOMA_DIR"
echo "Output: $OUTPUT"
echo ""

echo "Creating tarball (this takes a few minutes)..."
tar czf "$OUTPUT" \
    seasonal_augmentation/scripts/generate_multicontrolnet.py \
    seasonal_augmentation/scripts/generate_controlnet.py \
    seasonal_augmentation/scripts/generate_instructpix2pix.py \
    seasonal_augmentation/scripts/generate_for_dataset.py \
    seasonal_augmentation/scripts/estimate_depth.py \
    seasonal_augmentation/scripts/evaluate_consistency.py \
    seasonal_augmentation/scripts/consistency.py \
    seasonal_augmentation/scripts/vastai_setup.sh \
    seasonal_augmentation/configs/ \
    seasonal_augmentation/outputs/denseuav/depth_maps/ \
    data/DenseUAV/train/drone/ \
    data/DenseUAV/test/query_drone/

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo "=== Done ==="
echo "Bundle: $OUTPUT ($SIZE)"
echo ""
echo "Upload to Vast.ai instance:"
echo "  scp -P <PORT> $OUTPUT root@<VAST_IP>:/workspace/"
echo ""
echo "Then on the instance:"
echo "  cd /workspace && tar xzf denseuav_bundle.tar.gz"
echo "  bash seasonal_augmentation/scripts/vastai_setup.sh"
