#!/usr/bin/env python3
"""
Generate seasonal variations using InstructPix2Pix.

InstructPix2Pix is specifically trained for instruction-based image editing.
It preserves structure much better than standard img2img while making targeted changes.

Key advantage: Understands "make it winter" type instructions naturally.
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from consistency import add_quality_gate_args, create_quality_gate_from_args


# Instructions for seasonal transformations
SEASONAL_INSTRUCTIONS = {
    "summer_to_autumn": "Apply warm autumn color grading to this scene with desaturated warm tones, yellowing foliage edges, and dried brown grass",
    "summer_to_winter": "Transform this summer scene into winter with snow covering the ground. Trees are bare and covered in snow. Cars are covered in snow.",
    "summer_to_spring": "Transform this summer scene into spring with fresh light green foliage and blooming flowers",
}


def load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Load InstructPix2Pix pipeline."""
    print("Loading InstructPix2Pix model...")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=dtype,
        safety_checker=None,
    )

    # Use faster scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # Memory optimization
    pipe.enable_model_cpu_offload()

    return pipe


def generate_seasonal_image(
    pipe,
    image: Image.Image,
    instruction: str,
    num_inference_steps: int = 30,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    seed: int = None,
):
    """Generate a seasonal variation using InstructPix2Pix.

    Args:
        image: Input image
        instruction: Edit instruction (e.g., "make it winter")
        num_inference_steps: Number of denoising steps
        image_guidance_scale: How much to follow input image (higher = more faithful)
                             Recommended: 1.2-2.0 for seasonal changes
        guidance_scale: How strongly to follow the instruction
                       Recommended: 7-12 for seasonal changes
        seed: Random seed for reproducibility
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=instruction,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return result


def process_directory(
    input_dir: Path,
    output_dir: Path,
    transformation: str,
    config: dict = None,
    device: str = "cuda",
    quality_gate=None,
    max_retries: int = 3,
    _pipe=None,
):
    """Process all images in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Get instruction
    if transformation in SEASONAL_INSTRUCTIONS:
        instruction = SEASONAL_INSTRUCTIONS[transformation]
    else:
        raise ValueError(f"Unknown transformation: {transformation}. Available: {list(SEASONAL_INSTRUCTIONS.keys())}")

    # Override with config if provided
    if config and "instruction" in config:
        instruction = config["instruction"]

    # Generation parameters
    params = {
        "num_inference_steps": config.get("num_inference_steps", 30) if config else 30,
        "image_guidance_scale": config.get("image_guidance_scale", 1.5) if config else 1.5,
        "guidance_scale": config.get("guidance_scale", 7.5) if config else 7.5,
        "seed": config.get("seed", None) if config else None,
    }

    # Find all images
    extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_dir.rglob(f"*{ext}"))

    # Check what's already done (resume support)
    to_generate = []
    already_done = 0

    for img_path in image_paths:
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path.parent / f"{img_path.stem}_{transformation}.jpg"

        if out_path.exists():
            already_done += 1
        else:
            to_generate.append((img_path, out_path))

    print(f"Found {len(image_paths)} images")
    print(f"Already generated: {already_done} (will be skipped)")
    print(f"To generate: {len(to_generate)}")
    print(f"Transformation: {transformation}")
    print(f"Instruction: {instruction}")
    print(f"Image guidance scale: {params['image_guidance_scale']} (higher = more faithful to original)")
    print(f"Guidance scale: {params['guidance_scale']} (higher = stronger seasonal effect)")

    if len(to_generate) == 0:
        print("\nAll images already generated! Nothing to do.")
        return

    # Load pipeline (or reuse if passed in)
    pipe = _pipe if _pipe is not None else load_pipeline(device=device)

    # Process images
    generated_count = 0
    error_count = 0
    rejected_count = 0

    for img_path, out_path in tqdm(to_generate, desc="Generating seasonal images"):
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")

            attempts = max_retries + 1 if quality_gate else 1
            accepted = False

            for attempt in range(attempts):
                attempt_params = params.copy()
                if attempt > 0 and "seed" in attempt_params and attempt_params["seed"] is not None:
                    attempt_params["seed"] = attempt_params["seed"] + attempt

                result = generate_seasonal_image(
                    pipe=pipe,
                    image=image,
                    instruction=instruction,
                    **attempt_params,
                )

                if quality_gate:
                    passed, metrics = quality_gate.check(image, result)
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
        description="Generate seasonal variations using InstructPix2Pix"
    )
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
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--transformation",
        type=str,
        default="summer_to_winter",
        choices=list(SEASONAL_INSTRUCTIONS.keys()),
        help="Seasonal transformation to apply",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Custom instruction (overrides transformation preset)",
    )
    parser.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="How much to follow input image (1.0-2.0, higher = more faithful to original)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="How strongly to follow instruction (7-12 recommended)",
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

    add_quality_gate_args(parser)

    args = parser.parse_args()

    # Build config from args
    config = {
        "num_inference_steps": args.steps,
        "image_guidance_scale": args.image_guidance_scale,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }

    if args.instruction:
        config["instruction"] = args.instruction

    quality_gate, max_retries = create_quality_gate_from_args(args)

    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        transformation=args.transformation,
        config=config,
        device=args.device,
        quality_gate=quality_gate,
        max_retries=max_retries,
    )


if __name__ == "__main__":
    main()
