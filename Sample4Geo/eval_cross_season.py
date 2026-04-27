#!/usr/bin/env python3
"""
Cross-season evaluation for UAV-VisLoc.

This script evaluates how well a model trained on one season
generalizes to another season.

Usage:
    python eval_cross_season.py --checkpoint ./model/weights.pth
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from sample4geo.dataset.uavvisloc import (
    UAVVisLocDatasetEval,
    get_transforms,
    SUMMER_LOCATIONS,
    AUTUMN_LOCATIONS,
    ALL_LOCATIONS
)
from sample4geo.model import TimmModel
from sample4geo.evaluate.university import evaluate


@dataclass
class EvalConfig:
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int = 384
    batch_size_eval: int = 32
    normalize_features: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4


def evaluate_season_pair(model, config, data_folder, query_locs, gallery_locs, transforms):
    """Evaluate retrieval from query_locs to gallery_locs."""

    query_dataset = UAVVisLocDatasetEval(
        data_root=data_folder,
        locations=query_locs,
        transforms=transforms,
        mode='query'
    )

    gallery_dataset = UAVVisLocDatasetEval(
        data_root=data_folder,
        locations=gallery_locs,
        transforms=transforms,
        mode='gallery'
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    r1 = evaluate(
        config=config,
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        ranks=[1, 5, 10],
        step_size=1000,
        cleanup=True
    )

    return r1


def main():
    parser = argparse.ArgumentParser(description="Cross-season evaluation")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_folder', type=str,
                        default='./data/UAV_VisLoc_processed',
                        help='Path to processed UAV-VisLoc data')
    parser.add_argument('--model', type=str,
                        default='convnext_base.fb_in22k_ft_in1k_384',
                        help='Model architecture')
    parser.add_argument('--img_size', type=int, default=384,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()

    config = EvalConfig()
    config.model = args.model
    config.img_size = args.img_size
    config.batch_size_eval = args.batch_size

    # Load model
    print(f"Loading model: {args.model}")
    model = TimmModel(
        config.model,
        pretrained=False,
        img_size=config.img_size
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(config.device)
    model.eval()

    # Get transforms
    data_config = model.get_config()
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    val_transforms, _, _ = get_transforms(img_size, mean=mean, std=std)

    print("\n" + "="*60)
    print("Cross-Season Evaluation Results")
    print("="*60)

    results = {}

    # Same season evaluation
    print("\n--- Same Season Evaluation ---")

    print("\nSummer → Summer:")
    r1_ss = evaluate_season_pair(
        model, config, args.data_folder,
        SUMMER_LOCATIONS, SUMMER_LOCATIONS, val_transforms
    )
    results['summer_to_summer'] = r1_ss

    print("\nAutumn → Autumn:")
    r1_aa = evaluate_season_pair(
        model, config, args.data_folder,
        AUTUMN_LOCATIONS, AUTUMN_LOCATIONS, val_transforms
    )
    results['autumn_to_autumn'] = r1_aa

    # Cross season evaluation
    print("\n--- Cross Season Evaluation ---")

    print("\nSummer → Autumn:")
    r1_sa = evaluate_season_pair(
        model, config, args.data_folder,
        SUMMER_LOCATIONS, AUTUMN_LOCATIONS, val_transforms
    )
    results['summer_to_autumn'] = r1_sa

    print("\nAutumn → Summer:")
    r1_as = evaluate_season_pair(
        model, config, args.data_folder,
        AUTUMN_LOCATIONS, SUMMER_LOCATIONS, val_transforms
    )
    results['autumn_to_summer'] = r1_as

    # All locations
    print("\n--- All Locations ---")

    print("\nAll → All:")
    r1_all = evaluate_season_pair(
        model, config, args.data_folder,
        ALL_LOCATIONS, ALL_LOCATIONS, val_transforms
    )
    results['all_to_all'] = r1_all

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Summer → Summer R@1: {results['summer_to_summer']:.4f}")
    print(f"Autumn → Autumn R@1: {results['autumn_to_autumn']:.4f}")
    print(f"Summer → Autumn R@1: {results['summer_to_autumn']:.4f}")
    print(f"Autumn → Summer R@1: {results['autumn_to_summer']:.4f}")
    print(f"All → All R@1:       {results['all_to_all']:.4f}")

    # Calculate seasonal gap
    same_season_avg = (results['summer_to_summer'] + results['autumn_to_autumn']) / 2
    cross_season_avg = (results['summer_to_autumn'] + results['autumn_to_summer']) / 2
    seasonal_gap = same_season_avg - cross_season_avg

    print(f"\nSame-season average R@1:  {same_season_avg:.4f}")
    print(f"Cross-season average R@1: {cross_season_avg:.4f}")
    print(f"Seasonal gap:             {seasonal_gap:.4f} ({seasonal_gap/same_season_avg*100:.1f}% drop)")

    return results


if __name__ == "__main__":
    main()
