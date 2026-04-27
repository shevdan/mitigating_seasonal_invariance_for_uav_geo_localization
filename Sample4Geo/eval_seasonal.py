"""Standalone evaluation of a saved seasonal checkpoint.

Loads weights from --checkpoint, builds query/gallery loaders for the
given --query-seasons and --gallery-seasons, and runs the same
evaluator the training script uses.

Usage examples are in the docstring of `main`.
"""
import argparse
import os
import sys
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample4geo.evaluate.university import evaluate
from sample4geo.model import TimmModel


@dataclass
class EvalConfig:
    device: str = "cuda"
    verbose: bool = True
    normalize_features: bool = True
    batch_size_eval: int = 32
    img_size: int = 384
    model: str = "convnext_base.fb_in22k_ft_in1k_384"


def build_denseuav_loaders(data_folder, query_seasons, gallery_seasons,
                           transforms_val, batch_size_eval, height_filter=None):
    from sample4geo.dataset.denseuav_multiseasonal import DenseUAVMultiseasonalEval

    query_folder = f"{data_folder}/test/query_drone"
    gallery_folder = f"{data_folder}/test/gallery_satellite"

    query_ds = DenseUAVMultiseasonalEval(
        data_folder=query_folder, mode="query", seasons=query_seasons,
        transforms=transforms_val, height_filter=height_filter,
    )
    gallery_ds = DenseUAVMultiseasonalEval(
        data_folder=gallery_folder, mode="gallery", seasons=gallery_seasons,
        transforms=transforms_val, height_filter=height_filter,
        loc_id_to_int=query_ds.get_loc_id_mapping(),
    )

    q_loader = DataLoader(query_ds, batch_size=batch_size_eval,
                          num_workers=4, shuffle=False, pin_memory=True)
    g_loader = DataLoader(gallery_ds, batch_size=batch_size_eval,
                          num_workers=4, shuffle=False, pin_memory=True)
    return q_loader, g_loader


def build_uavvisloc_loaders(data_folder, query_seasons, gallery_seasons,
                            transforms_val, batch_size_eval):
    from sample4geo.dataset.uavvisloc_multiseasonal import UAVVisLocMultiseasonalEval

    query_ds = UAVVisLocMultiseasonalEval(
        data_root=data_folder, mode="query",
        seasons=query_seasons, transforms=transforms_val,
    )
    gallery_ds = UAVVisLocMultiseasonalEval(
        data_root=data_folder, mode="gallery",
        seasons=gallery_seasons, transforms=transforms_val,
        pos_id_to_int=query_ds.get_pos_id_mapping(),
    )

    q_loader = DataLoader(query_ds, batch_size=batch_size_eval,
                          num_workers=4, shuffle=False, pin_memory=True)
    g_loader = DataLoader(gallery_ds, batch_size=batch_size_eval,
                          num_workers=4, shuffle=False, pin_memory=True)
    return q_loader, g_loader


def main():
    """
    Examples:
        python eval_seasonal.py \\
            --dataset denseuav \\
            --data-folder ./data/DenseUAV_multiseasonal \\
            --checkpoint ./weights/denseuav_seasonal/M1_du_multi_sumsat/best.pth \\
            --query-seasons autumn \\
            --gallery-seasons summer

        python eval_seasonal.py \\
            --dataset uavvisloc \\
            --data-folder ./data/UAV_VisLoc_multiseasonal \\
            --checkpoint ./weights/uavvisloc_seasonal_15k/M1_vl_multi_wintest/best.pth \\
            --query-seasons spring \\
            --gallery-seasons summer
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["denseuav", "uavvisloc"], required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-seasons", nargs="+", required=True)
    parser.add_argument("--gallery-seasons", nargs="+", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size-eval", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--height-filter", type=str, default=None,
                        help="DenseUAV only")
    args = parser.parse_args()

    config = EvalConfig(
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        batch_size_eval=args.batch_size_eval,
        img_size=args.img_size,
    )

    if args.dataset == "denseuav":
        from sample4geo.dataset.denseuav import get_transforms
    else:
        from sample4geo.dataset.uavvisloc import get_transforms

    val_transforms, _, _ = get_transforms(img_size=(config.img_size, config.img_size))

    if args.dataset == "denseuav":
        q_loader, g_loader = build_denseuav_loaders(
            args.data_folder, args.query_seasons, args.gallery_seasons,
            val_transforms, config.batch_size_eval, args.height_filter,
        )
    else:
        q_loader, g_loader = build_uavvisloc_loaders(
            args.data_folder, args.query_seasons, args.gallery_seasons,
            val_transforms, config.batch_size_eval,
        )

    model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARN missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"WARN unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(config.device)
    model.eval()

    print(f"\nDataset:        {args.dataset}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Query seasons:  {args.query_seasons}")
    print(f"Gallery seasons:{args.gallery_seasons}")
    print()

    results = evaluate(config, model, q_loader, g_loader)
    print(f"\nR@1: {results['R@1']:.2f}  R@5: {results['R@5']:.2f}  "
          f"R@10: {results['R@10']:.2f}  AP: {results['AP']:.2f}")


if __name__ == "__main__":
    main()