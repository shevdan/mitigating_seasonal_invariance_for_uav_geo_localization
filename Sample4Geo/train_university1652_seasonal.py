#!/usr/bin/env python3
"""
Seasonal experiments for University-1652 with Sample4Geo.

University-1652 has no generated drone views — only seasonal satellite from ArcGIS.
Training pairs: satellite (query) → drone (gallery).

Usage:
    # Baseline
    python train_university1652_seasonal.py --name U1_baseline

    # Train with all satellite seasons
    python train_university1652_seasonal.py --name U2_allsat \
        --train-sat-seasons summer autumn winter spring

    # Multipositive: all satellite seasons as positives
    python train_university1652_seasonal.py --name U3_multi \
        --train-sat-seasons summer autumn winter spring --multipositive \
        --test-gallery-seasons summer autumn winter spring
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

from sample4geo.dataset.university_multiseasonal import (
    U1652MultiseasonalTrain,
    U1652MultiseasonalEval,
)
from sample4geo.dataset.university import get_transforms
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.university import evaluate
from sample4geo.evaluate.sim_sample import calc_sim
from sample4geo.loss import InfoNCE
from sample4geo.loss_multipositive import MultipositiveInfoNCE
from sample4geo.model import TimmModel


@dataclass
class Configuration:
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int = 384
    mixed_precision: bool = True
    custom_sampling: bool = True
    seed: int = 1
    epochs: int = 20
    batch_size: int = 8
    verbose: bool = True
    gpu_ids: tuple = (0,)
    batch_size_eval: int = 32
    eval_every_n_epoch: int = 1
    normalize_features: bool = True
    eval_gallery_n: int = -1
    clip_grad: float = 100.0
    decay_exclue_bias: bool = False
    lr: float = 0.001
    weight_decay: float = 0.03
    warmup_epochs: int = 1
    scheduler: str = "cosine"
    label_smoothing: float = 0.1
    max_iterations: int = 20000
    gps_sample: bool = False
    sim_sample: bool = False
    gps_dict_path: str = ""
    neighbour_select: int = 64
    neighbour_range: int = 128

    data_folder: str = "./data/University1652_multiseasonal"
    train_sat_seasons: list = field(default_factory=lambda: ["summer"])
    test_query_seasons: list = field(default_factory=lambda: ["summer"])
    test_gallery_seasons: list = field(default_factory=lambda: ["summer"])
    multipositive: bool = False

    name: str = "baseline"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_multipositive(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    """Training loop with multipositive InfoNCE."""
    from torch.cuda.amp import autocast
    from sample4geo.utils import AverageMeter

    model.train()
    losses = AverageMeter()
    time.sleep(0.1)
    optimizer.zero_grad(set_to_none=True)

    bar = dataloader if not config.verbose else __import__('tqdm').tqdm(dataloader, total=len(dataloader))

    for query, reference, loc_ids in bar:
        if scaler:
            with autocast():
                query = query.to(config.device)
                reference = reference.to(config.device)
                loc_ids = loc_ids.to(config.device)
                features1, features2 = model(query, reference)
                scale = model.module.logit_scale.exp() if hasattr(model, 'module') else model.logit_scale.exp()
                loss = loss_function(features1, features2, scale, loc_ids=loc_ids)
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
            features1, features2 = model(query, reference)
            scale = model.module.logit_scale.exp() if hasattr(model, 'module') else model.logit_scale.exp()
            loss = loss_function(features1, features2, scale, loc_ids=loc_ids)
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
    parser = argparse.ArgumentParser(description="University-1652 Seasonal Experiments")
    parser.add_argument("--name", type=str, default="baseline")
    parser.add_argument("--data-folder", type=str, default="./data/University1652_multiseasonal")
    parser.add_argument("--train-sat-seasons", nargs="+", default=["summer"])
    parser.add_argument("--test-query-seasons", nargs="+", default=["summer"])
    parser.add_argument("--test-gallery-seasons", nargs="+", default=["summer"])
    parser.add_argument("--multipositive", action="store_true")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default='convnext_base.fb_in22k_ft_in1k_384')
    parser.add_argument("--gps-sample", action="store_true")
    parser.add_argument("--sim-sample", action="store_true")
    parser.add_argument("--gps-dict", type=str, default="")
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

    model_path = f"./weights/university1652_seasonal/{config.name}"
    os.makedirs(model_path, exist_ok=True)
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
    print(f"{'='*60}\n")

    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(
        img_size=(config.img_size, config.img_size)
    )

    # University-1652 training: satellite (query) → drone (gallery)
    train_dataset = U1652MultiseasonalTrain(
        query_folder=f"{config.data_folder}/train/satellite",
        gallery_folder=f"{config.data_folder}/train/drone",
        query_seasons=config.train_sat_seasons,
        multipositive=config.multipositive,
        transforms_query=train_sat_transforms,
        transforms_gallery=train_drone_transforms,
        shuffle_batch_size=config.batch_size,
    )

    # Eval: query = drone, gallery = satellite
    eval_query = U1652MultiseasonalEval(
        data_folder=f"{config.data_folder}/test/query_drone",
        mode="query",
        seasons=config.test_query_seasons,
        transforms=val_transforms,
    )
    eval_gallery = U1652MultiseasonalEval(
        data_folder=f"{config.data_folder}/test/gallery_satellite",
        mode="gallery",
        seasons=config.test_gallery_seasons,
        transforms=val_transforms,
        sample_ids=eval_query.get_sample_ids(),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4,
                              shuffle=not config.custom_sampling, pin_memory=True)
    query_loader = DataLoader(eval_query, batch_size=config.batch_size_eval, num_workers=4,
                              shuffle=False, pin_memory=True)
    gallery_loader = DataLoader(eval_gallery, batch_size=config.batch_size_eval, num_workers=4,
                                shuffle=False, pin_memory=True)

    model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
    model = model.to(config.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.multipositive:
        loss_function = MultipositiveInfoNCE(label_smoothing=config.label_smoothing, device=config.device)
        train_fn = train_multipositive
    else:
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        loss_function = InfoNCE(loss_function=loss_fn, device=config.device)
        train_fn = train

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * config.warmup_epochs,
        num_training_steps=len(train_loader) * config.epochs,
    )
    scaler = GradScaler() if config.mixed_precision else None

    import pickle
    sim_dict = None
    if config.gps_sample and config.gps_dict_path:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
        print(f"Loaded GPS dict: {len(sim_dict)} locations")

    if config.sim_sample:
        query_loader_train_eval = DataLoader(
            U1652MultiseasonalEval(f"{config.data_folder}/train/satellite", mode="query_train",
                                   seasons=["summer"], transforms=val_transforms),
            batch_size=config.batch_size_eval, num_workers=4, shuffle=False, pin_memory=True)
        gallery_loader_train_eval = DataLoader(
            U1652MultiseasonalEval(f"{config.data_folder}/train/drone", mode="gallery_train",
                                   seasons=["summer"], transforms=val_transforms,
                                   sample_ids=query_loader_train_eval.dataset.get_sample_ids()),
            batch_size=config.batch_size_eval, num_workers=4, shuffle=False, pin_memory=True)

    import math
    total_iterations = 0
    best_r1 = 0.0

    max_epochs = config.epochs
    print(f"Max iterations: {config.max_iterations}")
    print(f"Max epochs (upper bound): {max_epochs}")

    for epoch in range(1, max_epochs + 1):
        print(f"\n--- Epoch {epoch}/{max_epochs} (iter {total_iterations}/{config.max_iterations}) ---")
        if config.custom_sampling:
            train_dataset.shuffle(sim_dict=sim_dict, neighbour_select=config.neighbour_select,
                                  neighbour_range=config.neighbour_range)
            iters_per_epoch = math.ceil(len(train_dataset.samples) / config.batch_size)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4,
                                      shuffle=False, pin_memory=True)

        train_loss = train_fn(config, model, train_loader, loss_function, optimizer, scheduler, scaler)
        total_iterations += iters_per_epoch
        images_seen = total_iterations * config.batch_size

        print(f"Epoch {epoch}: loss={train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, "
              f"iter={total_iterations}, images={images_seen}")

        if epoch % config.eval_every_n_epoch == 0 or epoch == max_epochs or (config.max_iterations > 0 and total_iterations >= config.max_iterations):
            print(f"\nEval @ iter={total_iterations} images={images_seen}:")
            results = evaluate(config, model, query_loader, gallery_loader)
            r1 = results['R@1']
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), f"{model_path}/best.pth")
                print(f"  New best R@1: {r1:.2f}%")

            if config.sim_sample:
                sim_dict = calc_sim(config, model, gallery_loader_train_eval, query_loader_train_eval,
                                    neighbour_range=config.neighbour_range)

        if config.max_iterations > 0 and total_iterations >= config.max_iterations:
            print(f"\nReached {config.max_iterations} iterations, stopping.")
            break

    print(f"\n{'='*60}")
    print(f"Experiment {config.name} done. Best R@1: {best_r1:.2f}%")
    print(f"Total: {total_iterations} iterations, {total_iterations * config.batch_size} images, {epoch} epochs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
