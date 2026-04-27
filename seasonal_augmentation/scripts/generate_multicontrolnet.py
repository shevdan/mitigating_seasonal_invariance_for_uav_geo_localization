#!/usr/bin/env python3
"""
Generate seasonal variations using Multi-ControlNet (depth + canny edges).

Using both depth AND canny edge conditioning provides much stronger structure
preservation than depth alone, while still allowing meaningful color/texture changes.

- Depth: Preserves 3D structure and relative positions
- Canny: Preserves fine edges, boundaries, and object outlines
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from consistency import add_quality_gate_args, create_quality_gate_from_args


# Default prompts for seasonal transformations
SEASONAL_PROMPTS = {
    "summer_to_autumn": {
        "positive": (
            "aerial photograph with warm autumn color grading, "
            "desaturated warm tones, golden hour warm lighting, "
            "yellowing edges on foliage, dried brown grass, "
            "same composition, drone view, high resolution, photorealistic"
        ),
        "negative": (
            "bright red maple leaves, intense orange foliage, peak fall colors, "
            "carpet of fallen leaves, bare trees, winter, snow, "
            "vivid green, lush vegetation, saturated colors, "
            "new objects, changed layout, blurry, low quality, distorted, unrealistic"
        ),
    },
    "summer_to_winter": {
        "positive": (
            "winter aerial photograph, snow covered landscape, frost on surfaces, "
            "white snow on ground, cold winter tones, drone view, high resolution, "
            "realistic, same composition, same objects"
        ),
        "negative": (
            "summer, green leaves, autumn, fall colors, blurry, "
            "low quality, distorted, unrealistic, different objects, new trees"
        ),
    },
    "summer_to_spring": {
        "positive": (
            "spring aerial photograph, fresh light green foliage, cherry blossoms, "
            "blooming flowers, new growth vegetation, soft spring colors, "
            "drone view, high resolution, realistic, same composition"
        ),
        "negative": (
            "summer, dense dark green, autumn, fall colors, winter, snow, "
            "blurry, low quality, distorted, unrealistic"
        ),
    },
}


def get_canny_image(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Extract canny edges from an image."""
    image_np = np.array(image)

    # Convert to grayscale if needed
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Convert to RGB (3 channels)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(edges_rgb)


def load_pipeline(
    depth_model: str = "lllyasviel/control_v11f1p_sd15_depth",
    canny_model: str = "lllyasviel/control_v11p_sd15_canny",
    sd_model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_img2img: bool = False,
):
    """Load Multi-ControlNet pipeline with depth + canny."""
    print(f"Loading ControlNet (depth): {depth_model}")
    controlnet_depth = ControlNetModel.from_pretrained(
        depth_model,
        torch_dtype=dtype,
    )

    print(f"Loading ControlNet (canny): {canny_model}")
    controlnet_canny = ControlNetModel.from_pretrained(
        canny_model,
        torch_dtype=dtype,
    )

    PipeClass = StableDiffusionControlNetImg2ImgPipeline if use_img2img else StableDiffusionControlNetPipeline
    print(f"Loading Stable Diffusion: {sd_model} ({'img2img' if use_img2img else 'txt2img'})")
    pipe = PipeClass.from_pretrained(
        sd_model,
        controlnet=[controlnet_depth, controlnet_canny],
        torch_dtype=dtype,
        safety_checker=None,
    )

    # Use faster scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Memory optimization
    pipe.enable_model_cpu_offload()

    return pipe


def generate_seasonal_image(
    pipe,
    original_image: Image.Image,
    depth_image: Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 30,
    depth_conditioning_scale: float = 0.8,
    canny_conditioning_scale: float = 0.5,
    guidance_scale: float = 7.5,
    canny_low_threshold: int = 100,
    canny_high_threshold: int = 200,
    seed: int = None,
    use_img2img: bool = False,
    img2img_strength: float = 0.4,
):
    """Generate a seasonal variation using Multi-ControlNet.

    Args:
        depth_conditioning_scale: Weight for depth ControlNet (0.0-1.0)
        canny_conditioning_scale: Weight for canny ControlNet (0.0-1.0)
        use_img2img: If True, use original image as starting point (better for subtle changes)
        img2img_strength: How much to change (0.0 = no change, 1.0 = full regeneration)
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Downscale large images to fit in VRAM (SD 1.5 native res is 512x512)
    # Keep aspect ratio, max dimension 1024, round to 8 for diffusion compatibility
    original_size = original_image.size  # (w, h)
    max_dim = 1024
    w, h = original_size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale) // 8 * 8
        new_h = int(h * scale) // 8 * 8
        original_image = original_image.resize((new_w, new_h), Image.LANCZOS)
        depth_image = depth_image.resize((new_w, new_h), Image.BILINEAR)
    else:
        # Resize depth to match original
        depth_image = depth_image.resize(original_image.size, Image.BILINEAR)

    # Generate canny edges from original image
    canny_image = get_canny_image(
        original_image,
        low_threshold=canny_low_threshold,
        high_threshold=canny_high_threshold
    )

    if use_img2img:
        # img2img: start from original image, depth+canny guide the structure
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=original_image,
            control_image=[depth_image, canny_image],
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=[depth_conditioning_scale, canny_conditioning_scale],
            guidance_scale=guidance_scale,
            strength=img2img_strength,
            generator=generator,
        ).images[0]
    else:
        # txt2img: generate from noise, depth+canny condition only
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=[depth_image, canny_image],
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=[depth_conditioning_scale, canny_conditioning_scale],
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    return result, original_size


def process_directory(
    input_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    transformation: str = "summer_to_winter",
    config: dict = None,
    device: str = "cuda",
    quality_gate=None,
    max_retries: int = 3,
    _pipe=None,
):
    """Process all images in a directory."""
    input_dir = Path(input_dir)
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)

    # Get prompts
    if transformation in SEASONAL_PROMPTS:
        prompts = SEASONAL_PROMPTS[transformation].copy()
    else:
        raise ValueError(f"Unknown transformation: {transformation}")

    # Override with config if provided
    if config:
        prompts["positive"] = config.get("positive_prompt", prompts["positive"])
        prompts["negative"] = config.get("negative_prompt", prompts["negative"])

    # img2img parameters
    use_img2img = config.get("use_img2img", False) if config else False
    img2img_strength = config.get("img2img_strength", 0.4) if config else 0.4

    # Generation parameters
    params = {
        "num_inference_steps": config.get("num_inference_steps", 30) if config else 30,
        "depth_conditioning_scale": config.get("depth_scale", 0.8) if config else 0.8,
        "canny_conditioning_scale": config.get("canny_scale", 0.5) if config else 0.5,
        "guidance_scale": config.get("guidance_scale", 7.5) if config else 7.5,
        "canny_low_threshold": config.get("canny_low", 100) if config else 100,
        "canny_high_threshold": config.get("canny_high", 200) if config else 200,
        "seed": config.get("seed", None) if config else None,
        "use_img2img": use_img2img,
        "img2img_strength": img2img_strength,
    }

    # Find all original images
    extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.rglob(f"*{ext}"))

    # Filter to only images that have depth maps
    valid_pairs = []
    for img_path in image_paths:
        rel_path = img_path.relative_to(input_dir)
        depth_path = depth_dir / rel_path.parent / f"{img_path.stem}_depth.png"
        if depth_path.exists():
            valid_pairs.append((img_path, depth_path))
        else:
            print(f"Warning: No depth map for {img_path}")

    if not valid_pairs:
        print("No valid image-depth pairs found!")
        return

    # Resume support
    to_generate = []
    already_done = 0
    for img_path, depth_path in valid_pairs:
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.parent / f"{img_path.stem}_{transformation}.jpg"
        if out_path.exists():
            already_done += 1
        else:
            to_generate.append((img_path, depth_path, out_path))

    print(f"Found {len(valid_pairs)} image-depth pairs")
    print(f"Already generated: {already_done} (will be skipped)")
    print(f"To generate: {len(to_generate)}")
    print(f"Transformation: {transformation}")
    print(f"Depth scale: {params['depth_conditioning_scale']}, Canny scale: {params['canny_conditioning_scale']}")
    print(f"Positive prompt: {prompts['positive'][:80]}...")

    if len(to_generate) == 0:
        print("\nAll images already generated! Nothing to do.")
        return

    # Load pipeline (or reuse if passed in)
    pipe = _pipe if _pipe is not None else load_pipeline(device=device, use_img2img=use_img2img)

    # Process images
    generated_count = 0
    error_count = 0
    rejected_count = 0

    for img_path, depth_path, out_path in tqdm(to_generate, desc="Generating (Multi-ControlNet)"):
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load images
            original = Image.open(img_path).convert("RGB")
            depth = Image.open(depth_path).convert("RGB")

            attempts = max_retries + 1 if quality_gate else 1
            accepted = False

            for attempt in range(attempts):
                # Vary seed on retries
                attempt_params = params.copy()
                if attempt > 0 and "seed" in attempt_params and attempt_params["seed"] is not None:
                    attempt_params["seed"] = attempt_params["seed"] + attempt

                result, original_size = generate_seasonal_image(
                    pipe=pipe,
                    original_image=original,
                    depth_image=depth,
                    positive_prompt=prompts["positive"],
                    negative_prompt=prompts["negative"],
                    **attempt_params,
                )

                if quality_gate:
                    # Compare at generation resolution (result may be smaller than original)
                    qg_original = original.resize(result.size, Image.LANCZOS) if original.size != result.size else original
                    passed, metrics = quality_gate.check(qg_original, result)
                    if passed:
                        accepted = True
                        break
                    else:
                        retry_str = f" (retry {attempt+1}/{max_retries})" if attempt < max_retries else " (REJECTED)"
                        tqdm.write(
                            f"  QG fail{retry_str}: {img_path.name} — "
                            f"dino_cls={metrics['dino_cls_sim']:.3f} "
                            f"dino_patch={metrics['dino_patch_sim_mean']:.3f} "
                            f"edge_f1={metrics['edge_f1']:.3f}"
                        )
                else:
                    accepted = True

            if accepted:
                # Resize back to original dimensions if we downscaled for generation
                if result.size != original_size:
                    result = result.resize(original_size, Image.LANCZOS)
                result.save(out_path, quality=95)
                generated_count += 1
            else:
                rejected_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            error_count += 1
            continue

    print(f"\nGeneration complete!")
    print(f"  Generated: {generated_count}")
    print(f"  Errors: {error_count}")
    if quality_gate:
        print(f"  Rejected (hallucinated): {rejected_count}")
        print(f"  {quality_gate.summary()}")
    print(f"  Total (including previously done): {already_done + generated_count}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate seasonal variations using Multi-ControlNet (depth + canny)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing original images",
    )
    parser.add_argument(
        "--depth",
        type=str,
        required=True,
        help="Directory containing depth maps",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default="summer_to_winter",
        choices=list(SEASONAL_PROMPTS.keys()),
        help="Seasonal transformation to apply",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.8,
        help="Depth ControlNet conditioning scale (0.0-1.0)",
    )
    parser.add_argument(
        "--canny-scale",
        type=float,
        default=0.5,
        help="Canny ControlNet conditioning scale (0.0-1.0)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=100,
        help="Canny edge detection low threshold",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=200,
        help="Canny edge detection high threshold",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None = random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--use-img2img",
        action="store_true",
        help="Use img2img mode — start from original image, only modify colors/textures. "
             "Much better for subtle seasonal changes (e.g., autumn, spring).",
    )
    parser.add_argument(
        "--img2img-strength",
        type=float,
        default=0.4,
        help="img2img strength (0.0=no change, 1.0=full regeneration). "
             "Recommended: 0.3-0.5 for seasonal changes.",
    )

    add_quality_gate_args(parser)

    args = parser.parse_args()

    # Build config from args
    config = {
        "num_inference_steps": args.steps,
        "depth_scale": args.depth_scale,
        "canny_scale": args.canny_scale,
        "guidance_scale": args.guidance_scale,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "seed": args.seed,
        "use_img2img": args.use_img2img,
        "img2img_strength": args.img2img_strength,
    }

    quality_gate, max_retries = create_quality_gate_from_args(args)

    process_directory(
        input_dir=args.input,
        depth_dir=args.depth,
        output_dir=args.output,
        transformation=args.transformation,
        config=config,
        device=args.device,
        quality_gate=quality_gate,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    main()
