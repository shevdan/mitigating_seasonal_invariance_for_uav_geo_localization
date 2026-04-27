#!/usr/bin/env python3
"""
Prepare a curated subset of source + generated seasonal drone views for
Google Drive upload. Each location ends up in its own folder containing
four files: original_summer.jpg, generated_autumn.jpg,
generated_winter.jpg, generated_spring.jpg.

Default: 50 DenseUAV locations + 25 UAV-VisLoc images = ~300 files,
roughly 150-250 MB depending on source resolutions.

Usage (from seasonal_augmentation/):
    python scripts/prepare_drive_subset.py \\
        --denseuav-n 50 \\
        --uavvisloc-n 25 \\
        --seed 7 \\
        --out-dir outputs/drive_subset
"""
import argparse
import random
import shutil
import textwrap
from pathlib import Path
from typing import List, Tuple


SEASONS = ["autumn", "winter", "spring"]


def copy_or_skip(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return True


def collect_denseuav(rng: random.Random, n: int, out_dir: Path) -> List[Tuple[str, Path]]:
    """Pick n DenseUAV locations whose H90 source + all 3 seasons exist; copy them."""
    base = Path("../data/DenseUAV/train/drone")
    gen = Path("outputs/denseuav")
    height = "H90"

    candidates = sorted([d.name for d in base.iterdir() if d.is_dir()])
    rng.shuffle(candidates)

    picked: List[Tuple[str, Path]] = []
    for loc in candidates:
        src = base / loc / f"{height}.JPG"
        gens = {
            s: gen / f"generated_summer_to_{s}" / "train" / "drone" / loc / f"{height}_summer_to_{s}.jpg"
            for s in SEASONS
        }
        if not src.exists() or not all(g.exists() for g in gens.values()):
            continue

        loc_dir = out_dir / "denseuav" / f"location_{loc}_{height}"
        copy_or_skip(src, loc_dir / "original_summer.jpg")
        for s in SEASONS:
            copy_or_skip(gens[s], loc_dir / f"generated_{s}.jpg")
        picked.append((f"DenseUAV {loc} {height}", loc_dir))
        if len(picked) == n:
            break
    return picked


def collect_uavvisloc(rng: random.Random, n: int, out_dir: Path) -> List[Tuple[str, Path]]:
    """Pick n UAV-VisLoc images whose source + all 3 seasons exist; copy them."""
    base = Path("../data/UAV_VisLoc_dataset")
    gen = Path("outputs/uavvisloc")

    candidates = []
    for flight_dir in sorted(base.iterdir()):
        if not flight_dir.is_dir():
            continue
        flight = flight_dir.name
        drone_dir = flight_dir / "drone"
        if not drone_dir.exists():
            continue
        for img in sorted(drone_dir.glob("*.JPG")):
            candidates.append((flight, img))
    rng.shuffle(candidates)

    picked: List[Tuple[str, Path]] = []
    for flight, src in candidates:
        stem = src.stem  # e.g., 01_0042
        gens = {
            s: gen / f"generated_summer_to_{s}" / flight / "drone" / f"{stem}_summer_to_{s}.jpg"
            for s in SEASONS
        }
        if not all(g.exists() for g in gens.values()):
            continue

        loc_dir = out_dir / "uavvisloc" / f"flight_{flight}_image_{stem}"
        copy_or_skip(src, loc_dir / "original_summer.jpg")
        for s in SEASONS:
            copy_or_skip(gens[s], loc_dir / f"generated_{s}.jpg")
        picked.append((f"UAV-VisLoc flight {flight} image {stem}", loc_dir))
        if len(picked) == n:
            break
    return picked


def write_readme(out_dir: Path, denseuav_n: int, uavvisloc_n: int, seed: int):
    readme = textwrap.dedent(f"""\
        Curated subset of Multi-ControlNet seasonal drone-view generations
        ====================================================================

        This folder contains a small curated subset of generations produced
        by the Multi-ControlNet pipeline described in the master's thesis
        "Mitigating Seasonal Variance in GPS-Denied UAV Geo-Localization"
        (Bohdan Shevchuk, UCU APPS).

        Each location has its own folder with four files:
            original_summer.jpg   - real summer drone capture from the
                                    original benchmark dataset
            generated_autumn.jpg  - Multi-ControlNet output, autumn
            generated_winter.jpg  - Multi-ControlNet output, winter
            generated_spring.jpg  - Multi-ControlNet output, spring

        Subfolders
        ----------
        - denseuav/   : {denseuav_n} locations from DenseUAV
                        (Dai et al., 2024).
        - uavvisloc/  : {uavvisloc_n} images from UAV-VisLoc
                        (Xu et al., 2024).

        Notes
        -----
        - The sampling seed is {seed}. Re-running with the same seed
          reproduces the same selection.
        - Original drone images are redistributed under the licences of the
          source benchmarks. Generated views are derivative works of those
          captures.
        - The full extended dataset (~47k generated drone views) is
          available via the Google Form referenced in the thesis
          contributions list.
        """)
    (out_dir / "README.txt").write_text(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--denseuav-n", type=int, default=50)
    parser.add_argument("--uavvisloc-n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/drive_subset"))
    args = parser.parse_args()

    if args.out_dir.exists():
        print(f"Note: {args.out_dir} already exists; existing files will be kept.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    print(f"Picking {args.denseuav_n} DenseUAV + {args.uavvisloc_n} UAV-VisLoc locations (seed={args.seed})")

    denseuav = collect_denseuav(rng, args.denseuav_n, args.out_dir)
    uavvisloc = collect_uavvisloc(rng, args.uavvisloc_n, args.out_dir)

    if len(denseuav) < args.denseuav_n:
        print(f"WARN: only found {len(denseuav)} eligible DenseUAV locations (asked {args.denseuav_n})")
    if len(uavvisloc) < args.uavvisloc_n:
        print(f"WARN: only found {len(uavvisloc)} eligible UAV-VisLoc images (asked {args.uavvisloc_n})")

    write_readme(args.out_dir, len(denseuav), len(uavvisloc), args.seed)

    total_files = sum(1 for _ in args.out_dir.rglob("*.jpg")) + \
                  sum(1 for _ in args.out_dir.rglob("*.JPG"))
    print(f"\nWrote {total_files} files to {args.out_dir}")
    print(f"  {len(denseuav)} DenseUAV locations")
    print(f"  {len(uavvisloc)} UAV-VisLoc images")
    print(f"  README at {args.out_dir / 'README.txt'}")
    print(f"\nNext: zip the folder, upload to Google Drive, set link sharing to")
    print(f"'Anyone with the link can view', and paste the URL into the appendix.")


if __name__ == "__main__":
    main()