#!/usr/bin/env python3
"""
Evaluate geometric consistency of seasonal augmentation outputs.

Metrics (from consistency.py):
  1. DINOv2 cosine similarity (CLS + patch-level) — structural preservation,
     invariant to seasonal color changes.
  2. Edge F1 (Canny) — direct geometric consistency of boundaries/edges.
  3. SSIM on edge maps — structural similarity without color confound.

Usage:
    # Compare a single pair
    python evaluate_consistency.py --original img_a.jpg --generated img_b.jpg

    # Evaluate a full directory (original -> generated mapping by filename)
    python evaluate_consistency.py \
        --original-dir data/DenseUAV/train/drone \
        --generated-dir outputs/denseuav/generated_summer_to_winter/train/drone

    # Evaluate with spatial heatmaps saved
    python evaluate_consistency.py \
        --original-dir ... --generated-dir ... --save-heatmaps

    # Compare multiple methods
    python evaluate_consistency.py \
        --original-dir data/DenseUAV/train/drone \
        --generated-dirs \
            outputs/denseuav/controlnet/train/drone \
            outputs/denseuav/multicontrolnet/train/drone \
            outputs/denseuav/instructpix2pix/train/drone \
        --method-names controlnet multicontrolnet instructpix2pix
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from consistency import (
    DINOv2Similarity,
    compute_edge_f1,
    compute_ssim_edges,
)


# ─── Evaluation Runner ────────────────────────────────────────────────────────


def evaluate_pair(
    original: Image.Image,
    generated: Image.Image,
    dino: DINOv2Similarity = None,
    canny_low: int = 50,
    canny_high: int = 150,
    edge_tolerance: int = 2,
) -> dict:
    """Evaluate all consistency metrics for a single image pair."""
    results = {}

    # DINOv2
    if dino is not None:
        dino_results = dino.compute(original, generated)
        patch_map = dino_results.pop("_patch_sim_map")
        results.update(dino_results)
        results["_patch_sim_map"] = patch_map

    # Edge F1
    results.update(compute_edge_f1(
        original, generated,
        tolerance_px=edge_tolerance,
        canny_low=canny_low,
        canny_high=canny_high,
    ))

    # SSIM on edges
    results.update(compute_ssim_edges(
        original, generated,
        canny_low=canny_low,
        canny_high=canny_high,
    ))

    return results


def find_image_pairs(
    original_dir: Path,
    generated_dir: Path,
    transformation: str = None,
) -> list:
    """Find matching (original, generated) image pairs by relative path.

    Uses the relative path (including subdirectories) as the key, so
    datasets like DenseUAV where every location has H80.JPG/H90.JPG/H100.JPG
    are matched correctly (e.g., 000806/H100 matches 000806/H100_summer_to_winter).

    Supports two naming conventions:
      1. Same relative path: subdir/img.jpg <-> subdir/img.jpg
      2. Transformation suffix: subdir/img.jpg <-> subdir/img_summer_to_winter.jpg
    """
    extensions = {".jpg", ".jpeg", ".png"}
    original_dir = Path(original_dir)
    generated_dir = Path(generated_dir)

    # Index originals by relative path WITHOUT extension: "000806/H100" -> full path
    originals = {}
    for f in original_dir.rglob("*"):
        if f.suffix.lower() in extensions:
            rel = f.relative_to(original_dir)
            key = str(rel.parent / rel.stem)  # e.g. "000806/H100"
            originals[key] = f

    pairs = []
    for f in generated_dir.rglob("*"):
        if f.suffix.lower() not in extensions:
            continue

        rel = f.relative_to(generated_dir)
        rel_key = str(rel.parent / rel.stem)  # e.g. "000806/H100_summer_to_winter"

        # Try exact relative path match
        if rel_key in originals:
            pairs.append((originals[rel_key], f))
            continue

        # Try removing transformation suffix
        suffixes_to_try = []
        if transformation:
            suffixes_to_try.append(f"_{transformation}")
        suffixes_to_try.extend([
            "_summer_to_autumn", "_summer_to_winter",
            "_autumn_to_summer", "_autumn_to_winter",
            "_winter_to_summer", "_winter_to_autumn",
        ])

        for suffix in suffixes_to_try:
            if rel_key.endswith(suffix):
                orig_key = rel_key[: -len(suffix)]
                if orig_key in originals:
                    pairs.append((originals[orig_key], f))
                    break

    return sorted(pairs, key=lambda p: str(p[0]))


def save_heatmap(
    patch_sim_map: np.ndarray,
    original: Image.Image,
    generated: Image.Image,
    output_path: Path,
):
    """Save a DINOv2 patch similarity heatmap overlaid on the images."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(generated)
    axes[1].set_title("Generated")
    axes[1].axis("off")

    im = axes[2].imshow(
        patch_sim_map, cmap="RdYlGn", vmin=0.5, vmax=1.0,
        interpolation="nearest",
    )
    axes[2].set_title(f"DINOv2 patch sim (mean={patch_sim_map.mean():.3f})")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def print_summary(all_results: list, method_name: str = None):
    """Print aggregate statistics."""
    header = f"=== Results: {method_name} ===" if method_name else "=== Results ==="
    print(f"\n{header}")
    print(f"  Images evaluated: {len(all_results)}")

    metric_keys = [
        k for k in all_results[0]
        if not k.startswith("_") and isinstance(all_results[0][k], (int, float))
    ]

    for key in metric_keys:
        values = [r[key] for r in all_results]
        arr = np.array(values)
        print(f"  {key}:")
        print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")


def evaluate_directory(
    original_dir: Path,
    generated_dir: Path,
    output_dir: Path = None,
    transformation: str = None,
    save_heatmaps: bool = False,
    max_images: int = 0,
    device: str = "cuda",
    canny_low: int = 50,
    canny_high: int = 150,
    edge_tolerance: int = 2,
) -> list:
    """Evaluate consistency metrics across a directory of image pairs."""
    pairs = find_image_pairs(original_dir, generated_dir, transformation)
    if not pairs:
        print(f"No matching pairs found between\n  {original_dir}\n  {generated_dir}")
        return []

    if max_images > 0:
        pairs = pairs[:max_images]

    print(f"\nFound {len(pairs)} image pairs")

    dino = DINOv2Similarity(device=device)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_heatmaps:
            (output_dir / "heatmaps").mkdir(exist_ok=True)

    all_results = []
    for orig_path, gen_path in tqdm(pairs, desc="Evaluating"):
        original = Image.open(orig_path).convert("RGB")
        generated = Image.open(gen_path).convert("RGB")

        result = evaluate_pair(
            original, generated,
            dino=dino,
            canny_low=canny_low,
            canny_high=canny_high,
            edge_tolerance=edge_tolerance,
        )

        if save_heatmaps and output_dir and "_patch_sim_map" in result:
            heatmap_path = output_dir / "heatmaps" / f"{orig_path.stem}_heatmap.png"
            save_heatmap(result["_patch_sim_map"], original, generated, heatmap_path)

        result["original"] = str(orig_path)
        result["generated"] = str(gen_path)
        result.pop("_patch_sim_map", None)
        all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate geometric consistency of seasonal augmentation"
    )

    # Single pair mode
    parser.add_argument("--original", type=str, help="Path to original image")
    parser.add_argument("--generated", type=str, help="Path to generated image")

    # Directory mode
    parser.add_argument("--original-dir", type=str, help="Directory of original images")
    parser.add_argument("--generated-dir", type=str, help="Directory of generated images")

    # Multi-method comparison mode
    parser.add_argument(
        "--generated-dirs", type=str, nargs="+",
        help="Multiple generated directories for comparison",
    )
    parser.add_argument(
        "--method-names", type=str, nargs="+",
        help="Names for each method (same order as --generated-dirs)",
    )

    # Options
    parser.add_argument("--output", type=str, default="output/consistency_eval",
                        help="Output directory for results")
    parser.add_argument("--transformation", type=str, default=None,
                        help="Transformation name for filename matching")
    parser.add_argument("--save-heatmaps", action="store_true",
                        help="Save DINOv2 patch similarity heatmaps")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to evaluate (0 = all)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--canny-low", type=int, default=50,
                        help="Canny low threshold")
    parser.add_argument("--canny-high", type=int, default=150,
                        help="Canny high threshold")
    parser.add_argument("--edge-tolerance", type=int, default=2,
                        help="Edge F1 tolerance in pixels")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Single pair mode ──
    if args.original and args.generated:
        print(f"Evaluating single pair:")
        print(f"  Original:  {args.original}")
        print(f"  Generated: {args.generated}")

        dino = DINOv2Similarity(device=args.device)
        original = Image.open(args.original).convert("RGB")
        generated = Image.open(args.generated).convert("RGB")

        result = evaluate_pair(
            original, generated, dino=dino,
            canny_low=args.canny_low, canny_high=args.canny_high,
            edge_tolerance=args.edge_tolerance,
        )

        if args.save_heatmaps and "_patch_sim_map" in result:
            save_heatmap(
                result["_patch_sim_map"], original, generated,
                output_dir / "heatmap.png",
            )

        result.pop("_patch_sim_map", None)
        print_summary([result])

        with open(output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {output_dir / 'result.json'}")
        return

    # ── Multi-method comparison mode ──
    if args.generated_dirs:
        if not args.original_dir:
            parser.error("--original-dir required with --generated-dirs")

        method_names = args.method_names or [
            Path(d).name for d in args.generated_dirs
        ]

        all_method_results = {}
        for gen_dir, name in zip(args.generated_dirs, method_names):
            print(f"\n{'='*60}")
            print(f"Method: {name}")
            print(f"{'='*60}")

            results = evaluate_directory(
                original_dir=args.original_dir,
                generated_dir=gen_dir,
                output_dir=output_dir / name,
                transformation=args.transformation,
                save_heatmaps=args.save_heatmaps,
                max_images=args.max_images,
                device=args.device,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                edge_tolerance=args.edge_tolerance,
            )
            if results:
                print_summary(results, method_name=name)
                all_method_results[name] = results

        # Save comparison
        summary = {}
        for name, results in all_method_results.items():
            metric_keys = [
                k for k in results[0]
                if not k.startswith("_")
                and k not in ("original", "generated")
                and isinstance(results[0][k], (int, float))
            ]
            summary[name] = {
                key: {
                    "mean": float(np.mean([r[key] for r in results])),
                    "std": float(np.std([r[key] for r in results])),
                }
                for key in metric_keys
            }

        with open(output_dir / "comparison.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nComparison saved to {output_dir / 'comparison.json'}")

        # Print comparison table
        print(f"\n{'='*60}")
        print("Method Comparison")
        print(f"{'='*60}")
        key_metrics = ["dino_cls_sim", "dino_patch_sim_mean", "edge_f1", "ssim_edge"]
        header = f"{'Method':<25}" + "".join(f"{m:<20}" for m in key_metrics)
        print(header)
        print("-" * len(header))
        for name in all_method_results:
            row = f"{name:<25}"
            for m in key_metrics:
                if m in summary[name]:
                    val = summary[name][m]["mean"]
                    std = summary[name][m]["std"]
                    row += f"{val:.4f}\u00b1{std:.4f}     "
                else:
                    row += f"{'N/A':<20}"
            print(row)
        return

    # ── Single directory mode ──
    if args.original_dir and args.generated_dir:
        results = evaluate_directory(
            original_dir=args.original_dir,
            generated_dir=args.generated_dir,
            output_dir=output_dir,
            transformation=args.transformation,
            save_heatmaps=args.save_heatmaps,
            max_images=args.max_images,
            device=args.device,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            edge_tolerance=args.edge_tolerance,
        )

        if results:
            print_summary(results)

            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nPer-image results saved to {output_dir / 'results.json'}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
