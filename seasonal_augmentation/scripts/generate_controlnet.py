#!/usr/bin/env python3
"""
Generate seasonal variations using ControlNet with depth conditioning.

This preserves the scene structure while changing seasonal appearance.
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

from consistency import add_quality_gate_args, create_quality_gate_from_args


# Default prompts for different seasonal transformations
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
            "winter aerial photograph, snow covered landscape, bare trees, "
            "frost, cold atmosphere, drone view, high resolution, realistic, "
            "overcast sky, white snow on ground"
        ),
        "negative": (
            "summer, green leaves, autumn, fall colors, blurry, "
            "low quality, distorted, unrealistic"
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


def load_pipeline(
    controlnet_model: str = "lllyasviel/control_v11f1p_sd15_depth",
    sd_model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_img2img: bool = False,
):
    """Load ControlNet pipeline.

    Args:
        use_img2img: If True, use img2img pipeline for better structure preservation.
                     The original image is used as starting point, only changing colors/textures.
    """
    print(f"Loading ControlNet: {controlnet_model}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=dtype,
    )

    print(f"Loading Stable Diffusion: {sd_model}")

    if use_img2img:
        print("Using img2img mode (better structure preservation)")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            sd_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )

    # Use faster scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()  # Uncomment if xformers installed

    return pipe


def generate_seasonal_image(
    pipe,
    original_image: Image.Image,
    depth_image: Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    num_inference_steps: int = 30,
    controlnet_conditioning_scale: float = 0.8,
    guidance_scale: float = 7.5,
    seed: int = None,
    use_img2img: bool = False,
    img2img_strength: float = 0.5,
):
    """Generate a seasonal variation of an image.

    Args:
        use_img2img: If True, use original image as starting point (better structure preservation)
        img2img_strength: How much to change from original (0.0 = no change, 1.0 = complete change)
                         Lower values = more faithful to original structure
                         Recommended: 0.3-0.5 for seasonal changes
    """
    # Set seed for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # Resize depth to match original
    depth_image = depth_image.resize(original_image.size, Image.BILINEAR)

    if use_img2img:
        # img2img mode: use original image as starting point
        # This preserves structure much better
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=original_image,  # Original image as base
            control_image=depth_image,  # Depth for structure guidance
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            strength=img2img_strength,  # How much to deviate from original
            generator=generator,
        ).images[0]
    else:
        # txt2img mode with ControlNet: generate from depth only
        result = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=depth_image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    return result


def process_directory(
    input_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    transformation: str = "summer_to_autumn",
    config: dict = None,
    device: str = "cuda",
    use_img2img: bool = False,
    quality_gate=None,
    max_retries: int = 3,
    _pipe=None,
):
    """Process all images in a directory.

    Args:
        use_img2img: If True, use img2img mode for better structure preservation.
                     This uses the original image as a starting point and only changes
                     colors/textures while keeping the exact same objects and layout.
    """
    input_dir = Path(input_dir)
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)

    # Get prompts
    if transformation in SEASONAL_PROMPTS:
        prompts = SEASONAL_PROMPTS[transformation]
    else:
        raise ValueError(f"Unknown transformation: {transformation}")

    # Override with config if provided
    if config:
        prompts["positive"] = config.get("positive_prompt", prompts["positive"])
        prompts["negative"] = config.get("negative_prompt", prompts["negative"])

    # Generation parameters
    params = {
        "num_inference_steps": config.get("num_inference_steps", 30) if config else 30,
        "controlnet_conditioning_scale": config.get("controlnet_scale", 0.8) if config else 0.8,
        "guidance_scale": config.get("guidance_scale", 7.5) if config else 7.5,
        "seed": config.get("seed", None) if config else None,
        "use_img2img": use_img2img,
        "img2img_strength": config.get("img2img_strength", 0.4) if config else 0.4,
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

    # Count already generated (for resume support)
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
    print(f"Positive prompt: {prompts['positive'][:100]}...")

    if len(to_generate) == 0:
        print("\nAll images already generated! Nothing to do.")
        return

    # Load pipeline (or reuse if passed in)
    pipe = _pipe if _pipe is not None else load_pipeline(device=device, use_img2img=use_img2img)

    # Process images
    generated_count = 0
    error_count = 0
    rejected_count = 0

    for img_path, depth_path, out_path in tqdm(to_generate, desc="Generating seasonal images"):
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load images
            original = Image.open(img_path).convert("RGB")
            depth = Image.open(depth_path).convert("RGB")

            attempts = max_retries + 1 if quality_gate else 1
            accepted = False

            for attempt in range(attempts):
                attempt_params = params.copy()
                if attempt > 0 and "seed" in attempt_params and attempt_params["seed"] is not None:
                    attempt_params["seed"] = attempt_params["seed"] + attempt

                result = generate_seasonal_image(
                    pipe=pipe,
                    original_image=original,
                    depth_image=depth,
                    positive_prompt=prompts["positive"],
                    negative_prompt=prompts["negative"],
                    **attempt_params,
                )

                if quality_gate:
                    passed, metrics = quality_gate.check(original, result)
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
        description="Generate seasonal variations using ControlNet"
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
        default="summer_to_autumn",
        choices=list(SEASONAL_PROMPTS.keys()),
        help="Seasonal transformation to apply",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
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
        help="Use img2img mode for better structure preservation. "
             "This uses the original image as starting point, only changing colors/textures.",
    )
    parser.add_argument(
        "--img2img-strength",
        type=float,
        default=0.4,
        help="Strength for img2img mode (0.0=no change, 1.0=complete change). "
             "Lower values preserve more structure. Recommended: 0.3-0.5",
    )

    add_quality_gate_args(parser)

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # Override img2img_strength in config if provided via CLI
    if config and args.img2img_strength != 0.4:
        config["img2img_strength"] = args.img2img_strength

    quality_gate, max_retries = create_quality_gate_from_args(args)

    process_directory(
        input_dir=args.input,
        depth_dir=args.depth,
        output_dir=args.output,
        transformation=args.transformation,
        config=config,
        device=args.device,
        use_img2img=args.use_img2img,
        quality_gate=quality_gate,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    main()
