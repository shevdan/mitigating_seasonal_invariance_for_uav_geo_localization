#!/usr/bin/env python3
"""
Depth estimation for UAV images using MiDaS/DPT models.

This creates depth maps that will be used as conditioning for ControlNet
to preserve scene structure during seasonal transformation.
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_midas_model(model_type="DPT_Large", device="cuda"):
    """Load MiDaS depth estimation model."""
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform


def estimate_depth(model, transform, image_path, device="cuda"):
    """Estimate depth for a single image."""
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform
    input_batch = transform(img).to(device)

    # Predict
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    # Normalize to 0-255 for saving
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_normalized = depth_normalized.astype(np.uint8)

    return depth_normalized


def process_directory(
    input_dir: Path,
    output_dir: Path,
    model_type: str = "DPT_Large",
    device: str = "cuda",
    extensions: tuple = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
):
    """Process all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.rglob(f"*{ext}"))

    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images")

    # Load model
    print(f"Loading MiDaS model ({model_type})...")
    model, transform = load_midas_model(model_type, device)

    # Process images
    for img_path in tqdm(image_paths, desc="Estimating depth"):
        # Maintain directory structure
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.parent / f"{img_path.stem}_depth.png"

        # Skip if already exists
        if out_path.exists():
            continue

        # Create output directory
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Estimate depth
        try:
            depth = estimate_depth(model, transform, img_path, device)
            cv2.imwrite(str(out_path), depth)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print(f"Depth maps saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Estimate depth maps for UAV images")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for depth maps",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DPT_Large",
        choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
        help="MiDaS model type",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        model_type=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
