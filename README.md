# Mitigating Seasonal Variance in GPS-Denied UAV Geo-Localization

Code release for the master's thesis *"Mitigating Seasonal Variance in GPS-Denied UAV Geo-Localization"* (Bohdan Shevchuk, UCU Faculty of Applied Sciences, 2026).

## Problem

Cross-view UAV geo-localization estimates a drone's position by matching its image against a geo-referenced satellite database, without GNSS. Recent contrastive-learning methods reach above 90% Recall@1 on benchmarks, but only when the query and gallery are from the same season. In practice, satellite reference maps are captured once and rarely updated, so the gallery is frozen in summer. A UAV flying in winter sees snow, leafless trees, and a different colour distribution, and same-season performance collapses: winter Recall@1 drops by 36 points on DenseUAV and by 76 points on UAV-VisLoc.

## Approach

This repository implements the dataset extension and training extension proposed in the thesis on top of the [Sample4Geo](https://github.com/Skyy93/Sample4Geo) framework.

* A Multi-ControlNet generation pipeline (Stable Diffusion 1.5 conditioned on depth and Canny edges) produces $47{,}619$ synthetic seasonal drone views from real summer captures across DenseUAV and UAV-VisLoc, with a quality gate (DINOv2 + Edge F1) that rejects generations that drift from the source.
* An ArcGIS Wayback retrieval pipeline collects multi-season satellite tiles for every location in DenseUAV, UAV-VisLoc, and University-1652.
* A multipositive InfoNCE loss treats the generated seasonal drone views as additional positive anchors for the corresponding satellite tile.

The multipositive loss recovers most of the cross-season degradation. Winter Recall@1 rises from $59.55$ to $93.44$ on DenseUAV ($+33.89$) and from $5.45$ to $72.79$ on UAV-VisLoc ($+67.34$).

## Repository structure

```
.
├── seasonalgeo/                ArcGIS Wayback / Sentinel-2 / Planet retrieval pipeline
│   ├── parsers/                Per-dataset GPS parsers
│   ├── providers/              Provider implementations (arcgis, gee_sentinel2, planet)
│   ├── scripts/                CLI: s01 parse, s02 retrieve, s03 format
│   └── configs/                Per-provider YAML configs
│
├── seasonal_augmentation/      Multi-ControlNet seasonal drone-view generation
│   ├── scripts/
│   │   ├── generate_multicontrolnet.py       Main generation method
│   │   ├── generate_for_dataset.py           Dataset-level orchestrator
│   │   ├── generate_instructpix2pix.py       IP2P baseline (related-works comparison)
│   │   ├── consistency.py                    Quality gate (DINOv2 + Edge F1)
│   │   ├── estimate_depth.py                 MiDaS DPT-Large depth maps
│   │   ├── prepare_multiseasonal.py          Builds multi-seasonal dataset structure
│   │   ├── prepare_for_sample4geo.py         Converts outputs to Sample4Geo format
│   │   ├── compare_ip2p_vs_multicontrolnet.py  Backs the IP2P pilot in the thesis
│   └── configs/                Per-season generation hyperparameters (YAML)
│
├── Sample4Geo/                 Training framework (forked from Sample4Geo)
│   ├── sample4geo/             Models, losses, datasets, evaluators
│   ├── train_*_seasonal.py     Multipositive training scripts (DenseUAV, UAV-VisLoc, U-1652)
│   ├── train_{denseuav,uavvisloc,university}.py   Single-positive baselines
│   ├── eval_seasonal.py        Standalone evaluator for saved checkpoints
│
├── pyproject.toml              seasonalgeo package configuration
├── requirements.txt            seasonalgeo dependencies
├── LICENSE
└── README.md
```

## Installation

The three subsystems use different conda environments. Install only the ones you need.

### seasonalgeo (satellite retrieval)

```bash
conda create -n seasonalgeo python=3.10 -y
conda activate seasonalgeo
pip install -r requirements.txt
pip install -e .
```

This registers two CLI commands: `seasonalgeo-parse` and `seasonalgeo-retrieve`.

### seasonal_augmentation (drone-view generation)

```bash
conda create -n diploma_controlnet python=3.12 -y
conda activate diploma_controlnet
pip install -r seasonal_augmentation/requirements.txt
```

Diffusers $\geq 0.36$ is recommended for stability with the Multi-ControlNet pipeline.

### Sample4Geo (training and evaluation)

```bash
conda create -n sample4geo python=3.10 -y
conda activate sample4geo
pip install -r Sample4Geo/requirements.txt
```

PyTorch should be installed first with the CUDA build that matches the local driver, e.g.\ `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121`.

## Usage

The end-to-end pipeline runs in three stages. Each subsystem has its own README with detailed usage; the high-level flow is below.

### 1. Parse benchmark GPS coordinates and retrieve satellite tiles

```bash
# Parse the three benchmarks (writes data/georecords/*.csv)
seasonalgeo-parse --data-dir data/

# Retrieve ArcGIS Wayback satellite tiles for every parsed location
seasonalgeo-retrieve --provider arcgis --data-dir data/ --num-locations 3000
```

ArcGIS imagery is not redistributed in this repository because of Esri's Terms of Use; running the retrieval pipeline locally produces the same tiles used in the thesis. Sentinel-2 (via Google Earth Engine) and Planet Labs PlanetScope are alternative providers, but they provide poor quality for the tasks of this work.

### 2. Generate seasonal drone views

```bash
# Estimate depth maps for the source summer drone images
python seasonal_augmentation/scripts/estimate_depth.py \
    --input-dir data/DenseUAV/train/drone

# Run the Multi-ControlNet generation pipeline with the quality gate
python seasonal_augmentation/scripts/generate_for_dataset.py \
    --dataset denseuav --transformation summer_to_winter \
    --method multicontrolnet \
    --quality-gate
```

Per-season hyperparameters are in `seasonal_augmentation/configs/summer_to_{autumn,winter,spring}.yaml`.

### 3. Train the multipositive retrieval model

```bash
cd Sample4Geo

# Multipositive training on DenseUAV, all four seasons, all-season gallery
python train_denseuav_seasonal.py \
    --train-sat-seasons summer autumn winter spring \
    --test-query-seasons winter \
    --test-gallery-seasons summer autumn winter spring \
    --multipositive

# Standalone evaluation of a saved checkpoint
python eval_seasonal.py \
    --dataset denseuav \
    --data-folder ../data/DenseUAV_multiseasonal \
    --checkpoint ./weights/denseuav_seasonal/M4_du_multi_win_allgal/best.pth \
    --query-seasons spring --gallery-seasons summer autumn winter spring
```

## Datasets and weights

The benchmark datasets are not redistributed by this repository. Obtain them from the upstream sources:

* DenseUAV — https://github.com/Dmmm1997/DenseUAV
* UAV-VisLoc — https://github.com/IntelliSensing/UAV-VisLoc
* University-1652 — https://github.com/layumi/University1652-Baseline

The extended datasets used in the thesis distributed under the underlying datasets' licences. ArcGIS satellite tiles are not redistributed. They can be retrieved locally with `seasonalgeo-retrieve`.

Access to the extended datasets and checkpoints is provided via a [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdxZV7zjdiuROneyu43A3_XK3bRxfHBqBIDo-gmotMst2eAUg/viewform?usp=dialog).

## Citation

```bibtex
@mastersthesis{shevchuk2026seasonal,
  author = {Bohdan Shevchuk},
  title  = {Mitigating Seasonal Variance in {GPS}-Denied {UAV} Geo-Localization},
  school = {Ukrainian Catholic University, Faculty of Applied Sciences},
  year   = {2026}
}
```

If you use the multipositive loss or the generated seasonal drone views, please also cite the Sample4Geo framework, the Multi-ControlNet generation pipeline (Stable Diffusion + ControlNet), and the underlying benchmark datasets you train on.

## License

Code in this repository is released under the MIT License (see `LICENSE`). The benchmark datasets retain their original licences from the upstream releases. Generated drone views are derivatives of the upstream drone captures and inherit their licences. ArcGIS Wayback satellite tiles retrieved through `seasonalgeo` are subject to Esri's Terms of Use.

## Use of AI tools

Anthropic's Claude and Claude Code were used as coding and writing assistants during the development of this thesis, for code generation, refactoring, and grammatical and stylistic review. All experimental design, model selection, dataset construction decisions, hyperparameter choices, analyses, and conclusions are the author's own. Every script was inspected and every reported number verified before inclusion.