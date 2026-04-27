#!/usr/bin/env python3
"""
Pilot comparison: InstructPix2Pix vs Multi-ControlNet on a small DenseUAV
sample. Generates seasonal variants with both methods, computes DINOv2 +
Edge F1 metrics on every output, and reports means and pass-rates at the
thresholds calibrated for Multi-ControlNet in the thesis.

Backs the related-works claim that InstructPix2Pix outputs do not meet
the structural-similarity bar that Multi-ControlNet outputs do.

Usage (from seasonal_augmentation/):
    conda run -n diploma_controlnet python scripts/compare_ip2p_vs_multicontrolnet.py \\
        --input-dir ../data/DenseUAV/train/drone \\
        --output-dir outputs/compare_ip2p_pilot \\
        --num-samples 30 \\
        --transformations summer_to_winter summer_to_autumn summer_to_spring

Per-image metrics are written to results.json under --output-dir.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from consistency import QualityGate
from estimate_depth import load_midas_model
from generate_instructpix2pix import (
    SEASONAL_INSTRUCTIONS,
    load_pipeline as load_ip2p_pipeline,
    generate_seasonal_image as generate_ip2p,
)
from generate_multicontrolnet import (
    load_pipeline as load_mcn_pipeline,
    generate_seasonal_image as generate_mcn,
)


# Calibrated thresholds (from CLAUDE.md / thesis Section sec:exp_quality_gate).
THESIS_THRESHOLDS = {
    "img2img": {  # autumn, spring
        "dino_cls": 0.75,
        "dino_patch": 0.72,
        "edge_f1": 0.74,
    },
    "txt2img": {  # winter
        "dino_cls": 0.35,
        "dino_patch": 0.43,
        "edge_f1": 0.35,
    },
}


CONFIG_DIR = SCRIPT_DIR.parent / "configs"


def load_season_config(transformation: str) -> dict:
    """Load the YAML config for one transformation."""
    path = CONFIG_DIR / f"{transformation}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("use_img2img", False)
    cfg.setdefault("img2img_strength", 0.4)
    cfg.setdefault("controlnet_scale", 0.8)
    cfg.setdefault("guidance_scale", 7.5)
    cfg.setdefault("num_inference_steps", 30)
    return cfg


def transformation_mode(cfg: dict) -> str:
    return "img2img" if cfg.get("use_img2img") else "txt2img"


def collect_pilot_images(input_dir: Path, n: int, seed: int) -> List[Path]:
    """Pick n drone images at random (deterministic given seed)."""
    extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    candidates = []
    for ext in extensions:
        candidates.extend(input_dir.rglob(f"*{ext}"))
    candidates = sorted(set(candidates))
    if len(candidates) < n:
        raise RuntimeError(f"Only {len(candidates)} images under {input_dir}; requested {n}")
    rng = random.Random(seed)
    return rng.sample(candidates, n)


def estimate_depth_pil(midas_model, midas_transform, image: Image.Image, device: str) -> Image.Image:
    """Run MiDaS on a PIL image, return a depth map as a PIL grayscale image."""
    arr = np.array(image.convert("RGB"))
    input_batch = midas_transform(arr).to(device)
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=arr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = prediction.cpu().numpy()
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_img = (depth_norm * 255).astype(np.uint8)
    return Image.fromarray(depth_img).convert("RGB")


def passes_thresholds(metrics: Dict[str, float], mode: str) -> bool:
    t = THESIS_THRESHOLDS[mode]
    return (
        metrics["dino_cls_sim"] >= t["dino_cls"]
        and metrics["dino_patch_sim_mean"] >= t["dino_patch"]
        and metrics["edge_f1"] >= t["edge_f1"]
    )


def aggregate(per_image: List[Dict]) -> Dict[str, float]:
    if not per_image:
        return {"n": 0}
    n = len(per_image)
    keys = ("dino_cls_sim", "dino_patch_sim_mean", "edge_f1")
    means = {k: sum(m[k] for m in per_image) / n for k in keys}
    pass_rate = sum(1 for m in per_image if m["passes_thesis_gate"]) / n
    return {
        "n": n,
        "dino_cls": means["dino_cls_sim"],
        "dino_patch": means["dino_patch_sim_mean"],
        "edge_f1": means["edge_f1"],
        "pass_rate": pass_rate,
    }


def print_table(label_to_results: Dict[str, Dict]):
    print()
    print("| Method | Transformation | n | DINOv2 CLS | DINOv2 patch | Edge F1 | Pass rate |")
    print("|---|---|---|---|---|---|---|")
    for label in label_to_results:
        for transformation, per_image in label_to_results[label].items():
            agg = aggregate(per_image)
            if agg["n"] == 0:
                print(f"| {label} | {transformation} | 0 | - | - | - | - |")
                continue
            print(
                f"| {label} | {transformation} | {agg['n']} | "
                f"{agg['dino_cls']:.3f} | {agg['dino_patch']:.3f} | "
                f"{agg['edge_f1']:.3f} | {agg['pass_rate']*100:.0f}% |"
            )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Directory of source drone images, e.g. ../data/DenseUAV/train/drone")
    parser.add_argument("--output-dir", default=Path("outputs/compare_ip2p_pilot"), type=Path)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--transformations", nargs="+",
                        default=["summer_to_winter", "summer_to_autumn", "summer_to_spring"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-ip2p", action="store_true",
                        help="Skip IP2P generation; recompute metrics from saved IP2P outputs if present.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pilot_paths = collect_pilot_images(args.input_dir, args.num_samples, args.seed)
    print(f"Pilot set: {len(pilot_paths)} images from {args.input_dir}")
    print(f"Transformations: {args.transformations}")

    season_configs = {t: load_season_config(t) for t in args.transformations}

    # Quality gate computes the same DINOv2 + Edge F1 metrics our thesis pipeline uses.
    qg = QualityGate(
        dino_cls_threshold=0.0,  # we do not gate during generation; we just measure
        dino_patch_threshold=0.0,
        edge_f1_threshold=0.0,
        device=args.device,
    )

    # MiDaS depth (used for Multi-ControlNet only).
    print("\nLoading MiDaS for depth estimation...")
    midas_model, midas_transform = load_midas_model("DPT_Large", device=args.device)

    # Precompute depth maps once per pilot image (Multi-ControlNet uses them).
    print("Precomputing depth maps for the pilot set...")
    depth_cache: Dict[Path, Image.Image] = {}
    for img_path in tqdm(pilot_paths):
        image = Image.open(img_path).convert("RGB")
        depth_cache[img_path] = estimate_depth_pil(midas_model, midas_transform, image, args.device)
    del midas_model, midas_transform
    torch.cuda.empty_cache()

    # ---- InstructPix2Pix ----
    ip2p_results: Dict[str, List[Dict]] = {t: [] for t in args.transformations}
    if args.skip_ip2p:
        print("\n[1/2] InstructPix2Pix (skipped; recomputing metrics from saved outputs)")
        for transformation in args.transformations:
            mode = transformation_mode(season_configs[transformation])
            save_dir = args.output_dir / "instructpix2pix" / transformation
            for img_path in tqdm(pilot_paths, desc=f"ip2p (saved) / {transformation}"):
                gen_path = save_dir / f"{img_path.stem}.jpg"
                if not gen_path.exists():
                    tqdm.write(f"  missing IP2P output: {gen_path}")
                    continue
                image = Image.open(img_path).convert("RGB")
                generated = Image.open(gen_path).convert("RGB")
                _, metrics = qg.check(image, generated)
                metrics["passes_thesis_gate"] = passes_thresholds(metrics, mode)
                metrics["image"] = img_path.name
                ip2p_results[transformation].append(metrics)
            qg.total_checked = qg.total_passed = qg.total_rejected = 0
    else:
        print("\n[1/2] InstructPix2Pix")
        ip2p_pipe = load_ip2p_pipeline(device=args.device)
        for transformation in args.transformations:
            instruction = SEASONAL_INSTRUCTIONS[transformation]
            mode = transformation_mode(season_configs[transformation])
            for img_path in tqdm(pilot_paths, desc=f"ip2p / {transformation}"):
                image = Image.open(img_path).convert("RGB")
                try:
                    generated = generate_ip2p(
                        pipe=ip2p_pipe,
                        image=image,
                        instruction=instruction,
                        seed=args.seed,
                    )
                except Exception as e:
                    tqdm.write(f"  ERR ip2p {img_path.name}: {e}")
                    continue
                save_dir = args.output_dir / "instructpix2pix" / transformation
                save_dir.mkdir(parents=True, exist_ok=True)
                generated.save(save_dir / f"{img_path.stem}.jpg", quality=95)

                _, metrics = qg.check(image, generated)
                metrics["passes_thesis_gate"] = passes_thresholds(metrics, mode)
                metrics["image"] = img_path.name
                ip2p_results[transformation].append(metrics)
            qg.total_checked = qg.total_passed = qg.total_rejected = 0
        del ip2p_pipe
        torch.cuda.empty_cache()

    # ---- Multi-ControlNet (one txt2img pipeline + one img2img pipeline) ----
    print("\n[2/2] Multi-ControlNet (depth + canny)")
    needs_txt2img = any(transformation_mode(season_configs[t]) == "txt2img" for t in args.transformations)
    needs_img2img = any(transformation_mode(season_configs[t]) == "img2img" for t in args.transformations)

    pipe_txt2img = load_mcn_pipeline(device=args.device, use_img2img=False) if needs_txt2img else None
    pipe_img2img = load_mcn_pipeline(device=args.device, use_img2img=True) if needs_img2img else None

    mcn_results: Dict[str, List[Dict]] = {t: [] for t in args.transformations}
    for transformation in args.transformations:
        cfg = season_configs[transformation]
        mode = transformation_mode(cfg)
        pipe = pipe_img2img if cfg.get("use_img2img") else pipe_txt2img
        for img_path in tqdm(pilot_paths, desc=f"mcn / {transformation}"):
            image = Image.open(img_path).convert("RGB")
            depth = depth_cache[img_path]
            try:
                generated, _orig_size = generate_mcn(
                    pipe=pipe,
                    original_image=image,
                    depth_image=depth,
                    positive_prompt=cfg["positive_prompt"],
                    negative_prompt=cfg["negative_prompt"],
                    num_inference_steps=cfg["num_inference_steps"],
                    depth_conditioning_scale=cfg["controlnet_scale"],
                    canny_conditioning_scale=cfg.get("canny_scale", 0.5),
                    guidance_scale=cfg["guidance_scale"],
                    seed=args.seed,
                    use_img2img=bool(cfg.get("use_img2img", False)),
                    img2img_strength=cfg.get("img2img_strength", 0.4),
                )
            except Exception as e:
                tqdm.write(f"  ERR mcn {img_path.name}: {e}")
                continue
            save_dir = args.output_dir / "multicontrolnet" / transformation
            save_dir.mkdir(parents=True, exist_ok=True)
            generated.save(save_dir / f"{img_path.stem}.jpg", quality=95)

            _, metrics = qg.check(image, generated)
            metrics["passes_thesis_gate"] = passes_thresholds(metrics, mode)
            metrics["image"] = img_path.name
            mcn_results[transformation].append(metrics)
        qg.total_checked = qg.total_passed = qg.total_rejected = 0

    # ---- Save and report ----
    label_to_results = {
        "instructpix2pix": ip2p_results,
        "multicontrolnet": mcn_results,
    }
    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(label_to_results, indent=2, default=float))
    print(f"\nWrote per-image metrics to {json_path}")
    print_table(label_to_results)

    print("Numbers for the thesis paragraph (mean over the pilot set):")
    for label in label_to_results:
        for transformation, per_image in label_to_results[label].items():
            agg = aggregate(per_image)
            if agg["n"] == 0:
                continue
            print(
                f"  {label}/{transformation}: "
                f"DINOv2 CLS={agg['dino_cls']:.2f}, patch={agg['dino_patch']:.2f}, "
                f"edge_f1={agg['edge_f1']:.2f}, "
                f"pass={agg['pass_rate']*100:.0f}%"
            )


if __name__ == "__main__":
    main()