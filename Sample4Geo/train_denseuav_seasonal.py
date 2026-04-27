#!/usr/bin/env python3
"""
Seasonal experiments for DenseUAV with Sample4Geo.

Supports all experiment configurations via command-line args:
  - Season selection for train drone / train satellite / test drone / test satellite
  - Multipositive learning (all satellite seasons at same location are positives)
  - Standard and multipositive InfoNCE loss

Usage:
    # B1: Baseline (summer-summer)
    python train_denseuav_seasonal.py --name B1_baseline

    # B3: Cross-season degradation (train summer, test winter drone)
    python train_denseuav_seasonal.py --name B3_winter_drop \
        --test-query-seasons winter

    # T2: Train with all satellite seasons, test winter→summer
    python train_denseuav_seasonal.py --name T2_allsat_winter \
        --train-sat-seasons summer autumn winter spring \
        --test-query-seasons winter

    # M2: Full multipositive pipeline
    python train_denseuav_seasonal.py --name M2_full \
        --train-drone-seasons summer autumn winter spring \
        --train-sat-seasons summer autumn winter spring \
        --multipositive \
        --test-query-seasons winter \
        --test-gallery-seasons winter
"""

import argparse
import os
import sys
import time

import torch
from dataclasses import dataclass, field
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from sample4geo.dataset.denseuav_multiseasonal import (
    DenseUAVMultiseasonalTrain,
    DenseUAVMultiseasonalEval,
)
from sample4geo.dataset.denseuav import get_transforms
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
from sample4geo.evaluate.sim_sample import calc_sim
from sample4geo.loss import InfoNCE
from sample4geo.loss_multipositive import MultipositiveInfoNCE
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
    eval_gallery_n: int = -1

    # Optimizer
    clip_grad: float = 100.0
    decay_exclue_bias: bool = False
    lr: float = 0.001
    weight_decay: float = 0.03
    warmup_epochs: int = 1
    scheduler: str = "cosine"

    # Loss
    label_smoothing: float = 0.1

    # Training budget
    max_iterations: int = 20000  # total gradient updates (0 = use epochs)

    # Hard negative sampling
    gps_sample: bool = False
    sim_sample: bool = False
    gps_dict_path: str = ""
    neighbour_select: int = 64
    neighbour_range: int = 128

    # Paths
    data_folder: str = "./data/DenseUAV_multiseasonal"

    # Seasonal config (set via CLI)
    train_sat_seasons: list = field(default_factory=lambda: ["summer"])
    test_query_seasons: list = field(default_factory=lambda: ["summer"])
    test_gallery_seasons: list = field(default_factory=lambda: ["summer"])
    multipositive: bool = False
    height_filter: str = None

    # Experiment
    name: str = "baseline"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def multipositive_collate_fn(batch):
    """Custom collate for multipositive batches.

    Each item is (query, gallery, loc_idx, gen_views_tensor).
    gen_views_tensor is [K, C, H, W] with variable K per item.

    Returns:
        queries: [N, C, H, W]
        galleries: [N, C, H, W]
        loc_ids: [N]
        gen_views: [G, C, H, W]  (all generated views concatenated)
        gen_loc_ids: [G]  (location index for each generated view)
    """
    queries, galleries, loc_ids, gen_list = [], [], [], []
    gen_views_all = []
    gen_loc_ids = []

    for item in batch:
        query, gallery, loc_idx, gen_tensor = item
        queries.append(query)
        galleries.append(gallery)
        loc_ids.append(loc_idx)

        if gen_tensor.numel() > 0:
            gen_views_all.append(gen_tensor)
            gen_loc_ids.extend([loc_idx] * len(gen_tensor))

    queries = torch.stack(queries)
    galleries = torch.stack(galleries)
    loc_ids = torch.tensor(loc_ids, dtype=torch.long)

    if gen_views_all:
        gen_views = torch.cat(gen_views_all, dim=0)
        gen_loc_ids = torch.tensor(gen_loc_ids, dtype=torch.long)
    else:
        gen_views = torch.empty(0)
        gen_loc_ids = torch.empty(0, dtype=torch.long)

    return queries, galleries, loc_ids, gen_views, gen_loc_ids


def train_multipositive(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    """Training loop with generated drone views as additional positives.

    Each batch contains:
    - N original drone images (queries)
    - N satellite images (references)
    - G generated drone images (extra positive anchors)

    The generated views go through the encoder but only contribute
    to the loss as additional positive rows in the similarity matrix.
    """
    from torch.cuda.amp import autocast
    from sample4geo.utils import AverageMeter

    model.train()
    losses = AverageMeter()
    time.sleep(0.1)
    optimizer.zero_grad(set_to_none=True)

    bar = dataloader if not config.verbose else __import__('tqdm').tqdm(dataloader, total=len(dataloader))

    for batch_data in bar:
        query, reference, loc_ids, gen_views, gen_loc_ids = batch_data

        if scaler:
            with autocast():
                query = query.to(config.device)
                reference = reference.to(config.device)
                loc_ids = loc_ids.to(config.device)

                # Forward pass: original drone + satellite
                features_query, features_ref = model(query, reference)
                scale = model.module.logit_scale.exp() if hasattr(model, 'module') else model.logit_scale.exp()

                # Forward pass: generated drone views
                features_gen = None
                gen_loc_ids_dev = None
                if gen_views.numel() > 0:
                    gen_views = gen_views.to(config.device)
                    gen_loc_ids_dev = gen_loc_ids.to(config.device)
                    features_gen = model(gen_views)  # single-image forward

                loss = loss_function(
                    features_query, features_ref, scale,
                    loc_ids=loc_ids,
                    features_gen=features_gen,
                    gen_loc_ids=gen_loc_ids_dev,
                )
                losses.update(loss.item())

            scaler.scale(loss).backward()
            if config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            query = query.to(config.device)
            reference = reference.to(config.device)
            loc_ids = loc_ids.to(config.device)

            features_query, features_ref = model(query, reference)
            scale = model.module.logit_scale.exp() if hasattr(model, 'module') else model.logit_scale.exp()

            features_gen = None
            gen_loc_ids_dev = None
            if gen_views.numel() > 0:
                gen_views = gen_views.to(config.device)
                gen_loc_ids_dev = gen_loc_ids.to(config.device)
                features_gen = model(gen_views)

            loss = loss_function(
                features_query, features_ref, scale,
                loc_ids=loc_ids,
                features_gen=features_gen,
                gen_loc_ids=gen_loc_ids_dev,
            )
            losses.update(loss.item())

            loss.backward()
            if config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

        if scheduler and config.scheduler in ("polynomial", "cosine", "constant"):
            scheduler.step()

        if config.verbose and hasattr(bar, 'set_postfix'):
            bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg=f"{losses.avg:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.6f}")

    if config.verbose and hasattr(bar, 'close'):
        bar.close()
    return losses.avg


def main():
    parser = argparse.ArgumentParser(description="DenseUAV Seasonal Experiments")
    parser.add_argument("--name", type=str, default="baseline")
    parser.add_argument("--data-folder", type=str, default="./data/DenseUAV_multiseasonal")
    parser.add_argument("--train-sat-seasons", nargs="+", default=["summer"])
    parser.add_argument("--test-query-seasons", nargs="+", default=["summer"])
    parser.add_argument("--test-gallery-seasons", nargs="+", default=["summer"])
    parser.add_argument("--multipositive", action="store_true")
    parser.add_argument("--height-filter", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=5000, help="Max epochs (use --max-iter instead)")
    parser.add_argument("--max-iter", type=int, default=20000, help="Stop after this many gradient updates (0=use epochs)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default='convnext_base.fb_in22k_ft_in1k_384')
    parser.add_argument("--gps-sample", action="store_true", help="GPS-based hard negative sampling")
    parser.add_argument("--sim-sample", action="store_true", help="Similarity-based hard negative mining")
    parser.add_argument("--gps-dict", type=str, default="", help="Path to pre-computed GPS dict")
    parser.add_argument("--neighbour-select", type=int, default=64)
    parser.add_argument("--neighbour-range", type=int, default=128)
    args = parser.parse_args()

    config = Configuration(
        name=args.name,
        data_folder=args.data_folder,
        train_sat_seasons=args.train_sat_seasons,
        test_query_seasons=args.test_query_seasons,
        test_gallery_seasons=args.test_gallery_seasons,
        multipositive=args.multipositive,
        height_filter=args.height_filter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gpu_ids=(args.gpu,),
        model=args.model,
        max_iterations=args.max_iter,
        gps_sample=args.gps_sample,
        sim_sample=args.sim_sample,
        gps_dict_path=args.gps_dict,
        neighbour_select=args.neighbour_select,
        neighbour_range=args.neighbour_range,
    )

    model_path = f"./weights/denseuav_seasonal/{config.name}"
    os.makedirs(model_path, exist_ok=True)

    # Redirect output to log
    sys.stdout = Logger(os.path.join(model_path, "log.txt"))

    setup_system(seed=config.seed, cudnn_benchmark=True, cudnn_deterministic=False)

    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"{'='*60}")
    print(f"Train sat seasons:     {config.train_sat_seasons}")
    print(f"Test query seasons:    {config.test_query_seasons}")
    print(f"Test gallery seasons:  {config.test_gallery_seasons}")
    print(f"Multipositive:         {config.multipositive}")
    print(f"GPS sampling:          {config.gps_sample}")
    print(f"Sim sampling:          {config.sim_sample}")
    print(f"Height filter:         {config.height_filter}")
    print(f"Model:                 {config.model}")
    print(f"{'='*60}\n")

    # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(
        img_size=(config.img_size, config.img_size)
    )

    # Folders
    query_folder_train = f"{config.data_folder}/train/drone"
    gallery_folder_train = f"{config.data_folder}/train/satellite"
    query_folder_test = f"{config.data_folder}/test/query_drone"
    gallery_folder_test = f"{config.data_folder}/test/gallery_satellite"

    # Train dataset
    # Note: query is ALWAYS original (summer) drone only.
    # gallery_seasons controls which satellite seasons are used.
    # multipositive loads generated drone views as extra positive anchors.
    train_dataset = DenseUAVMultiseasonalTrain(
        query_folder=query_folder_train,
        gallery_folder=gallery_folder_train,
        gallery_seasons=config.train_sat_seasons,
        multipositive=config.multipositive,
        transforms_query=train_drone_transforms,
        transforms_gallery=train_sat_transforms,
        prob_flip=0.5,
        shuffle_batch_size=config.batch_size,
        height_filter=config.height_filter,
    )

    # Eval datasets
    query_dataset_test = DenseUAVMultiseasonalEval(
        data_folder=query_folder_test,
        mode="query",
        seasons=config.test_query_seasons,
        transforms=val_transforms,
        height_filter=config.height_filter,
    )

    gallery_dataset_test = DenseUAVMultiseasonalEval(
        data_folder=gallery_folder_test,
        mode="gallery",
        seasons=config.test_gallery_seasons,
        transforms=val_transforms,
        height_filter=config.height_filter,
        loc_id_to_int=query_dataset_test.get_loc_id_mapping(),
    )

    # Dataloaders
    collate = multipositive_collate_fn if config.multipositive else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        shuffle=not config.custom_sampling,
        pin_memory=True,
        collate_fn=collate,
    )

    query_loader_test = DataLoader(
        query_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    gallery_loader_test = DataLoader(
        gallery_dataset_test,
        batch_size=config.batch_size_eval,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    # Model
    model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
    model = model.to(config.device)

    print(f"Model: {config.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    if config.multipositive:
        loss_function = MultipositiveInfoNCE(
            label_smoothing=config.label_smoothing,
            device=config.device,
        )
        train_fn = train_multipositive
    else:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        loss_function = InfoNCE(loss_function=loss_fn, device=config.device)
        train_fn = train

    # Optimizer
    if config.decay_exclue_bias:
        param_groups = [
            {"params": [p for n, p in model.named_parameters() if "bias" not in n], "weight_decay": config.weight_decay},
            {"params": [p for n, p in model.named_parameters() if "bias" in n], "weight_decay": 0.0},
        ]
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * config.warmup_epochs,
        num_training_steps=len(train_loader) * config.epochs,
    )

    scaler = GradScaler() if config.mixed_precision else None

    # GPS/Similarity sampling setup
    import pickle
    sim_dict = None
    if config.gps_sample and config.gps_dict_path:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
        print(f"Loaded GPS dict: {len(sim_dict)} locations")

    # For sim_sample, we need train eval dataloaders (summer only, for feature extraction)
    if config.sim_sample:
        query_dataset_train_eval = DenseUAVMultiseasonalEval(
            data_folder=f"{config.data_folder}/train/drone",
            mode="query_train",
            seasons=["summer"],
            transforms=val_transforms,
            height_filter=config.height_filter,
        )
        gallery_dataset_train_eval = DenseUAVMultiseasonalEval(
            data_folder=f"{config.data_folder}/train/satellite",
            mode="gallery_train",
            seasons=["summer"],
            transforms=val_transforms,
            height_filter=config.height_filter,
            loc_id_to_int=query_dataset_train_eval.get_loc_id_mapping(),
        )
        query_loader_train_eval = DataLoader(query_dataset_train_eval, batch_size=config.batch_size_eval,
                                             num_workers=4, shuffle=False, pin_memory=True)
        gallery_loader_train_eval = DataLoader(gallery_dataset_train_eval, batch_size=config.batch_size_eval,
                                               num_workers=4, shuffle=False, pin_memory=True)

    # Training loop
    import math
    total_iterations = 0
    best_r1 = 0.0

    max_epochs = config.epochs
    print(f"Max iterations: {config.max_iterations}")
    print(f"Max epochs (upper bound): {max_epochs}")

    for epoch in range(1, max_epochs + 1):
        print(f"\n--- Epoch {epoch}/{max_epochs} (iter {total_iterations}/{config.max_iterations}) ---")

        if config.custom_sampling:
            train_dataset.shuffle(
                sim_dict=sim_dict,
                neighbour_select=config.neighbour_select,
                neighbour_range=config.neighbour_range,
            )
            iters_per_epoch = math.ceil(len(train_dataset.samples) / config.batch_size)
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                num_workers=4,
                shuffle=False,
                pin_memory=True,
                collate_fn=collate,
            )

        train_loss = train_fn(
            config, model, train_loader, loss_function,
            optimizer, scheduler, scaler,
        )
        total_iterations += iters_per_epoch
        images_seen = total_iterations * config.batch_size

        print(f"Epoch {epoch}: loss={train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, "
              f"iter={total_iterations}, images={images_seen}")

        if epoch % config.eval_every_n_epoch == 0 or epoch == max_epochs or (config.max_iterations > 0 and total_iterations >= config.max_iterations):
            print(f"\nEval @ iter={total_iterations} images={images_seen} "
                  f"(query={config.test_query_seasons}, gallery={config.test_gallery_seasons}):")
            results = evaluate(
                config, model, query_loader_test, gallery_loader_test,
            )

            r1 = results['R@1']
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), f"{model_path}/best.pth")
                print(f"  New best R@1: {r1:.2f}%")

            # Update sim_dict from current model embeddings
            if config.sim_sample:
                sim_dict = calc_sim(
                    config, model,
                    gallery_loader_train_eval, query_loader_train_eval,
                    neighbour_range=config.neighbour_range,
                )

        if config.max_iterations > 0 and total_iterations >= config.max_iterations:
            print(f"\nReached {config.max_iterations} iterations, stopping.")
            break

    print(f"\n{'='*60}")
    print(f"Experiment {config.name} done. Best R@1: {best_r1:.2f}%")
    print(f"Total: {total_iterations} iterations, {total_iterations * config.batch_size} images, {epoch} epochs")
    print(f"Weights: {model_path}/best.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
