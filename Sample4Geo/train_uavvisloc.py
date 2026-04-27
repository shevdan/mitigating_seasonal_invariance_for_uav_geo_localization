#!/usr/bin/env python3
"""
Training script for Sample4Geo on UAV-VisLoc dataset.

Supports:
- Training on all locations
- Training on specific seasons (summer/autumn)
- Cross-season evaluation

Usage:
    python train_uavvisloc.py
"""

import os
import time
import shutil
import sys
import torch
from dataclasses import dataclass, field
from typing import List
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from sample4geo.dataset.uavvisloc import (
    UAVVisLocDatasetTrain,
    UAVVisLocDatasetEval,
    get_transforms,
    SUMMER_LOCATIONS,
    AUTUMN_LOCATIONS,
    ALL_LOCATIONS
)
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate  # Can reuse university eval
from sample4geo.loss import InfoNCE
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int = 384

    # Training
    mixed_precision: bool = True
    custom_sampling: bool = True
    seed: int = 1
    epochs: int = 20
    batch_size: int = 8  # Smaller batch for single GPU
    verbose: bool = True
    gpu_ids: tuple = (0,)

    # Eval
    batch_size_eval: int = 32
    eval_every_n_epoch: int = 1
    normalize_features: bool = True

    # Optimizer
    clip_grad: float = 100.0
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.001
    scheduler: str = "cosine"
    warmup_epochs: float = 0.1
    lr_end: float = 0.0001

    # Dataset
    data_folder: str = "./data/UAV_VisLoc_processed"

    # Seasonal split
    train_locations: List[str] = field(default_factory=lambda: SUMMER_LOCATIONS)
    test_locations: List[str] = field(default_factory=lambda: AUTUMN_LOCATIONS)

    # Mode
    mode: str = 'drone2sat'  # 'drone2sat' or 'sat2drone'

    # Augmentation
    prob_flip: float = 0.5

    # Savepath
    model_path: str = "./uavvisloc"
    experiment_name: str = "summer_only"  # e.g., "all_seasons", "summer_only", "autumn_only"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint
    checkpoint_start: str = None

    # Workers
    num_workers: int = 0 if os.name == 'nt' else 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


def main():
    config = Configuration()

    # Create experiment directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if config.experiment_name:
        exp_dir = f"{config.experiment_name}_{timestamp}"
    else:
        exp_dir = timestamp
    model_path = "{}/{}/{}".format(
        config.model_path,
        config.model,
        exp_dir
    )

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

    # Setup logging
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    setup_system(
        seed=config.seed,
        cudnn_benchmark=config.cudnn_benchmark,
        cudnn_deterministic=config.cudnn_deterministic
    )

    # Print configuration
    print("\n" + "="*60)
    print("UAV-VisLoc Training Configuration")
    print("="*60)
    print(f"Model: {config.model}")
    print(f"Image size: {config.img_size}")
    print(f"Train locations: {config.train_locations}")
    print(f"Test locations: {config.test_locations}")
    print(f"Mode: {config.mode}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print("="*60 + "\n")

    # Create model
    model = TimmModel(
        config.model,
        pretrained=True,
        img_size=config.img_size
    )

    data_config = model.get_config()
    print(f"Data config: {data_config}")
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)

    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)

    if config.checkpoint_start is not None:
        print(f"Loading checkpoint: {config.checkpoint_start}")
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel if multiple GPUs
    print(f"GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)

    # Get transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(
        img_size, mean=mean, std=std
    )

    # Create datasets
    train_dataset = UAVVisLocDatasetTrain(
        data_root=config.data_folder,
        locations=config.train_locations,
        transforms_query=train_drone_transforms if config.mode == 'drone2sat' else train_sat_transforms,
        transforms_gallery=train_sat_transforms if config.mode == 'drone2sat' else train_drone_transforms,
        prob_flip=config.prob_flip,
        shuffle_batch_size=config.batch_size,
        mode=config.mode
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=not config.custom_sampling,
        pin_memory=True
    )

    # Test datasets
    query_dataset_test = UAVVisLocDatasetEval(
        data_root=config.data_folder,
        locations=config.test_locations,
        transforms=val_transforms,
        mode='query'
    )

    gallery_dataset_test = UAVVisLocDatasetEval(
        data_root=config.data_folder,
        locations=config.test_locations,
        transforms=val_transforms,
        mode='gallery'
    )

    query_dataloader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    gallery_dataloader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Query images (test): {len(query_dataset_test)}")
    print(f"Gallery images (test): {len(gallery_dataset_test)}")

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn, device=config.device)

    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    # Optimizer
    if config.decay_exclue_bias:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # Scheduler
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs

    if config.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=config.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps
        )
    elif config.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps
        )
    else:
        scheduler = None

    print(f"\nScheduler: {config.scheduler}")
    print(f"Warmup epochs: {config.warmup_epochs} - Warmup steps: {warmup_steps}")
    print(f"Train epochs: {config.epochs} - Train steps: {train_steps}")

    # Zero shot evaluation
    if config.zero_shot:
        print("\n" + "-"*30 + "[Zero Shot]" + "-"*30)
        evaluate(
            config=config,
            model=model,
            query_loader=query_dataloader_test,
            gallery_loader=gallery_dataloader_test,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True
        )

    # Custom sampling shuffle
    if config.custom_sampling:
        train_dataloader.dataset.shuffle()

    # Training loop
    best_score = 0

    for epoch in range(1, config.epochs + 1):
        print("\n" + "-"*30 + f"[Epoch: {epoch}]" + "-"*30)

        train_loss = train(
            config,
            model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )

        print(f"Epoch: {epoch}, Train Loss = {train_loss:.3f}, Lr = {optimizer.param_groups[0]['lr']:.6f}")

        # Evaluate
        if (epoch % config.eval_every_n_epoch == 0) or epoch == config.epochs:
            print("\n" + "-"*30 + "[Evaluate]" + "-"*30)

            results = evaluate(
                config=config,
                model=model,
                query_loader=query_dataloader_test,
                gallery_loader=gallery_dataloader_test,
                ranks=[1, 5, 10],
                step_size=1000,
                cleanup=True
            )
            r1_test = results['R@1']

            if r1_test > best_score:
                best_score = r1_test
                save_path = f'{model_path}/weights_e{epoch}_{r1_test:.4f}.pth'
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"New best! Saved to {save_path}")

        if config.custom_sampling:
            train_dataloader.dataset.shuffle()

    # Save final model
    final_path = f'{model_path}/weights_final.pth'
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), final_path)
    else:
        torch.save(model.state_dict(), final_path)

    # Per-season evaluation to measure seasonal gap
    print("\n" + "="*60)
    print("Per-Season Evaluation (Seasonal Gap Analysis)")
    print("="*60)

    season_results = {}

    for season_name, season_locs in [("Summer", SUMMER_LOCATIONS), ("Autumn", AUTUMN_LOCATIONS), ("All", ALL_LOCATIONS)]:
        # Check if any data exists for this season
        query_dataset_season = UAVVisLocDatasetEval(
            data_root=config.data_folder,
            locations=season_locs,
            transforms=val_transforms,
            mode='query'
        )

        if len(query_dataset_season) == 0:
            print(f"\n{season_name}: No data available")
            continue

        gallery_dataset_season = UAVVisLocDatasetEval(
            data_root=config.data_folder,
            locations=season_locs,
            transforms=val_transforms,
            mode='gallery'
        )

        query_loader_season = DataLoader(
            query_dataset_season,
            batch_size=config.batch_size_eval,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True
        )

        gallery_loader_season = DataLoader(
            gallery_dataset_season,
            batch_size=config.batch_size_eval,
            num_workers=config.num_workers,
            shuffle=False,
            pin_memory=True
        )

        print(f"\n--- {season_name} ({len(query_dataset_season)} pairs) ---")
        results_season = evaluate(
            config=config,
            model=model,
            query_loader=query_loader_season,
            gallery_loader=gallery_loader_season,
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True
        )
        season_results[season_name] = results_season

    # Print seasonal gap summary
    print("\n" + "="*60)
    print("SEASONAL GAP SUMMARY")
    print("="*60)
    print(f"Training locations: {config.train_locations}")
    print(f"\n{'Season':<10} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@top1%':>8} {'AP':>8}")
    print("-" * 52)
    for season, metrics in season_results.items():
        print(f"{season:<10} {metrics['R@1']:>8.2f} {metrics['R@5']:>8.2f} {metrics['R@10']:>8.2f} {metrics['R@top1']:>8.2f} {metrics['AP']:>8.2f}")

    if "Summer" in season_results and "Autumn" in season_results:
        print("\n" + "-" * 52)
        print("Seasonal Gap (|Summer - Autumn|):")
        for metric in ['R@1', 'R@5', 'R@10', 'AP']:
            gap = abs(season_results["Summer"][metric] - season_results["Autumn"][metric])
            print(f"  {metric}: {gap:.2f}")

    print("="*60)

    print(f"\nTraining complete!")
    print(f"Best R@1 (combined): {best_score:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
