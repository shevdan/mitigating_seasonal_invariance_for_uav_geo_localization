#!/usr/bin/env python3
"""
Training script for Sample4Geo on SUES-200 dataset.

SUES-200 contains:
- 200 locations with drone images at 4 altitudes (150m, 200m, 250m, 300m)
- 50 frames per altitude per location
- 1 satellite image per location

Default train/test split: 150 train / 50 test locations.

Usage:
    python train_sues200.py
    python train_sues200.py --altitude_filter 150  # Only 150m altitude
"""

import os
import time
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from sample4geo.dataset.sues200 import (
    SUES200DatasetTrain,
    SUES200DatasetEval,
    get_transforms,
    DEFAULT_TRAIN_LOCATIONS,
    DEFAULT_TEST_LOCATIONS
)
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
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
    batch_size: int = 8
    verbose: bool = True
    gpu_ids: tuple = (0,)

    # Eval
    batch_size_eval: int = 32
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all

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
    data_folder: str = "./data/SUES-200"

    # Altitude filter: None for all altitudes, or '150', '200', '250', '300' for specific altitude
    altitude_filter: str = None

    # Number of frames to sample per location/altitude (None for all 50)
    frames_per_location: int = None

    # Augmentation
    prob_flip: float = 0.5

    # Savepath
    model_path: str = "./sues200"
    experiment_name: str = "baseline"

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

    altitude_suffix = f"_{config.altitude_filter}m" if config.altitude_filter else "_all_altitudes"
    model_path = "{}/{}/{}{}".format(
        config.model_path,
        config.model,
        exp_dir,
        altitude_suffix
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
    print("SUES-200 Training Configuration")
    print("="*60)
    print(f"Model: {config.model}")
    print(f"Image size: {config.img_size}")
    print(f"Altitude filter: {config.altitude_filter or 'All altitudes (150, 200, 250, 300)'}")
    print(f"Frames per location: {config.frames_per_location or 'All (50)'}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Train locations: {len(DEFAULT_TRAIN_LOCATIONS)} (0001-0150)")
    print(f"Test locations: {len(DEFAULT_TEST_LOCATIONS)} (0151-0200)")
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

    # Data parallel
    print(f"GPUs available: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)

    model = model.to(config.device)

    # Get transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(
        img_size, mean=mean, std=std
    )

    # Dataset paths
    drone_folder = f"{config.data_folder}/drone_view_512"
    satellite_folder = f"{config.data_folder}/satellite-view"

    # Create datasets
    train_dataset = SUES200DatasetTrain(
        drone_folder=drone_folder,
        satellite_folder=satellite_folder,
        transforms_query=train_drone_transforms,
        transforms_gallery=train_sat_transforms,
        prob_flip=config.prob_flip,
        shuffle_batch_size=config.batch_size,
        locations=DEFAULT_TRAIN_LOCATIONS,
        altitude_filter=config.altitude_filter,
        frames_per_location=config.frames_per_location
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=not config.custom_sampling,
        pin_memory=True
    )

    # Test datasets
    query_dataset_test = SUES200DatasetEval(
        data_folder=drone_folder,
        mode='query',
        transforms=val_transforms,
        locations=DEFAULT_TEST_LOCATIONS,
        altitude_filter=config.altitude_filter,
        frames_per_location=config.frames_per_location
    )

    gallery_dataset_test = SUES200DatasetEval(
        data_folder=satellite_folder,
        mode='gallery',
        transforms=val_transforms,
        locations=DEFAULT_TEST_LOCATIONS,
        gallery_n=config.eval_gallery_n,
        loc_id_to_int=query_dataset_test.get_loc_id_mapping()  # Share mapping with query
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
        results = evaluate(
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

    print(f"\nTraining complete!")
    print(f"Best R@1: {best_score:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
