"""
DenseUAV Multiseasonal Dataset for Sample4Geo.

Supports two training modes:
1. Standard: original drone + satellite (any season). No generated views.
2. Multipositive: original drone + satellite (any season) as primary pair,
   PLUS generated drone views as additional positive anchors. Generated views
   are NOT used as queries — they only provide extra positive signal in the
   contrastive loss.

Batch structure (multipositive, N locations):
    query:      N original drone images         → encoder → [N, D]
    reference:  N satellite images (any season)  → encoder → [N, D]
    extra:      up to 3N generated drone images  → encoder → [3N, D]

    Loss matrix: [N+3N, N] where ALL drone embeddings at location A
    (original + generated) should match satellite at location A.

Structure:
    DenseUAV_multiseasonal/
        train/drone/000000/H80.JPG, H80_autumn.JPG, H80_winter.JPG, H80_spring.JPG
        train/satellite/000000/H80.tif, H80_autumn.tif, H80_winter.tif, H80_spring.tif
"""

import copy
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

SEASONS = ["summer", "autumn", "winter", "spring"]
GENERATED_SEASONS = ["autumn", "winter", "spring"]
SEASON_SUFFIXES = {"summer": "", "autumn": "_autumn", "winter": "_winter", "spring": "_spring"}


def _parse_season(filename: str) -> str:
    """Extract season from filename. No suffix = summer (original)."""
    stem = Path(filename).stem
    for season in GENERATED_SEASONS:
        if stem.endswith(f"_{season}"):
            return season
    return "summer"


def _get_height(filename: str) -> str:
    """Extract height prefix (H80/H90/H100) from filename."""
    stem = Path(filename).stem
    for h in ["H100", "H90", "H80"]:
        if stem.startswith(h):
            return h
    return stem


def get_data_seasonal(
    path: str,
    seasons: list[str] | None = None,
    height_filter: str | None = None,
):
    """Get images grouped by location, filtered by season and height."""
    extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.tif', '.TIF')
    path = Path(path)

    if seasons is None:
        seasons = SEASONS

    data = {}
    for location_dir in sorted(path.iterdir()):
        if not location_dir.is_dir():
            continue
        loc_id = location_dir.name

        files_by_season = {s: [] for s in seasons}
        all_files = []

        for f in sorted(location_dir.iterdir()):
            if f.suffix not in extensions or '_old' in f.name:
                continue

            season = _parse_season(f.name)
            if season not in seasons:
                continue

            if height_filter and not f.stem.startswith(height_filter):
                continue

            files_by_season[season].append(f.name)
            all_files.append(f.name)

        if all_files:
            data[loc_id] = {
                "path": str(location_dir),
                "files": all_files,
                "files_by_season": files_by_season,
            }

    return data


def _load_image(path: str):
    """Load image, return None on failure."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = cv2.imread(path)
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class DenseUAVMultiseasonalTrain(Dataset):
    """Training dataset with multipositive generated-view support.

    Two modes:
    - multipositive=False: standard pairs (original drone, satellite any season)
    - multipositive=True: returns (original drone, satellite, loc_idx, gen_views)
      where gen_views is a list of transformed generated drone images at the
      same location. The training loop uses these as extra positive anchors.

    Args:
        query_folder: Path to drone images.
        gallery_folder: Path to satellite images.
        gallery_seasons: Which satellite seasons to use.
        multipositive: Load generated drone views as extra positives.
        gen_seasons: Which generated seasons to load (default: autumn,winter,spring).
        height_filter: Filter to specific height (e.g., 'H80').
    """

    def __init__(
        self,
        query_folder,
        gallery_folder,
        gallery_seasons=None,
        multipositive=False,
        gen_seasons=None,
        transforms_query=None,
        transforms_gallery=None,
        prob_flip=0.5,
        shuffle_batch_size=128,
        height_filter=None,
    ):
        super().__init__()

        self.gallery_seasons = gallery_seasons or ["summer"]
        self.multipositive = multipositive
        self.gen_seasons = gen_seasons or GENERATED_SEASONS
        self.query_folder = query_folder
        self.height_filter = height_filter

        # Query: ONLY original (summer) drone images
        self.query_dict = get_data_seasonal(
            query_folder, seasons=["summer"], height_filter=height_filter,
        )
        # Gallery: satellite images at specified seasons
        self.gallery_dict = get_data_seasonal(
            gallery_folder, seasons=self.gallery_seasons, height_filter=height_filter,
        )

        # Common locations
        self.ids = sorted(
            set(self.query_dict.keys()) & set(self.gallery_dict.keys())
        )
        self.id_to_idx = {loc_id: i for i, loc_id in enumerate(self.ids)}

        # Build pairs: (loc_id, original_drone_path, satellite_path)
        # Drone is ALWAYS original (summer). Satellite can be any season.
        self.pairs = []
        for loc_id in self.ids:
            q_data = self.query_dict[loc_id]
            g_data = self.gallery_dict[loc_id]

            for q_file in q_data["files"]:
                q_path = f"{q_data['path']}/{q_file}"
                q_height = _get_height(q_file)

                for g_file in g_data["files"]:
                    if _get_height(g_file) != q_height:
                        continue
                    g_path = f"{g_data['path']}/{g_file}"
                    self.pairs.append((loc_id, q_path, g_path))

        # For multipositive: index generated drone views per location
        if self.multipositive:
            self._build_generated_index()

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

        n_q = sum(len(d["files"]) for d in self.query_dict.values())
        n_g = sum(len(d["files"]) for d in self.gallery_dict.values())
        n_gen = sum(len(v) for v in self.gen_paths_by_loc.values()) if self.multipositive else 0
        print(
            f"DenseUAV Multiseasonal Train: {len(self.ids)} locations, "
            f"{n_q} original drone, {n_g} satellite ({self.gallery_seasons}), "
            f"{len(self.pairs)} pairs, multipositive={self.multipositive}"
            f"{f', {n_gen} generated views' if self.multipositive else ''}"
        )

    def _build_generated_index(self):
        """Build mapping: loc_id → {height: [gen_drone_paths]}."""
        self.gen_paths_by_loc = {}
        gen_dict = get_data_seasonal(
            self.query_folder, seasons=self.gen_seasons,
            height_filter=self.height_filter,
        )
        for loc_id in self.ids:
            if loc_id not in gen_dict:
                self.gen_paths_by_loc[loc_id] = []
                continue
            g = gen_dict[loc_id]
            # All generated files for this location
            self.gen_paths_by_loc[loc_id] = [
                f"{g['path']}/{f}" for f in g["files"]
            ]

    def _load_one_generated_view(self, loc_id: str, height: str, do_flip: bool):
        """Load one random generated drone view for a location.

        Picks a random season each call. Across epochs the model sees
        all generated seasons for each location.
        Returns transformed image tensor or None.
        """
        if loc_id not in self.gen_paths_by_loc:
            return None

        candidates = [
            p for p in self.gen_paths_by_loc[loc_id]
            if _get_height(Path(p).name) == height
        ]
        if not candidates:
            return None

        path = random.choice(candidates)
        img = _load_image(path)
        if img is None:
            return None
        if do_flip:
            img = cv2.flip(img, 1)
        if self.transforms_query is not None:
            img = self.transforms_query(image=img)['image']
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img

    def __getitem__(self, index):
        loc_id, query_path, gallery_path = self.samples[index]

        query_img = _load_image(query_path)
        if query_img is None:
            print(f"WARNING: cannot read {query_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        gallery_img = _load_image(gallery_path)
        if gallery_img is None:
            print(f"WARNING: cannot read {gallery_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        do_flip = np.random.random() < self.prob_flip
        if do_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        loc_idx = self.id_to_idx[loc_id]

        if self.multipositive:
            height = _get_height(Path(query_path).name)
            gen_view = self._load_one_generated_view(loc_id, height, do_flip)
            # Return single generated view or empty tensor
            if gen_view is not None:
                gen_tensor = gen_view.unsqueeze(0)  # [1, C, H, W]
            else:
                gen_tensor = torch.empty(0)
            return query_img, gallery_img, loc_idx, gen_tensor

        return query_img, gallery_img, loc_idx

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """Custom shuffle with optional hard negative mining."""
        print("\nShuffle Dataset:")

        pairs_by_loc = {}
        for pair in self.pairs:
            loc_idx = self.id_to_idx[pair[0]]
            pairs_by_loc.setdefault(loc_idx, []).append(pair)

        idx_pool = list(pairs_by_loc.keys())
        random.shuffle(idx_pool)

        neighbour_split = neighbour_select // 2
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        idx_epoch = set()
        idx_batch = set()
        batches = []
        current_batch = []
        break_counter = 0

        pbar = tqdm()
        while True:
            pbar.update()
            if not idx_pool:
                break

            idx = idx_pool.pop(0)

            if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:
                idx_batch.add(idx)
                current_batch.append(random.choice(pairs_by_loc[idx]))
                idx_epoch.add(idx)
                break_counter = 0

                if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                    if idx in similarity_pool:
                        near_sim = similarity_pool[idx][:neighbour_range]
                        near_neighbours = copy.deepcopy(near_sim[:neighbour_split])
                        far_neighbours = copy.deepcopy(near_sim[neighbour_split:])
                        random.shuffle(far_neighbours)
                        far_neighbours = far_neighbours[:neighbour_split]
                        candidates = near_neighbours + far_neighbours

                        for idx_near in candidates:
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                            if idx_near not in idx_batch and idx_near not in idx_epoch:
                                if idx_near in pairs_by_loc:
                                    idx_batch.add(idx_near)
                                    current_batch.append(random.choice(pairs_by_loc[idx_near]))
                                    idx_epoch.add(idx_near)
                                    break_counter = 0
            else:
                if idx not in idx_batch and idx not in idx_epoch:
                    idx_pool.append(idx)
                break_counter += 1

            if break_counter >= 512:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()
        time.sleep(0.3)

        if current_batch:
            batches.extend(current_batch)

        self.samples = batches
        print(f"Shuffle: {len(self.pairs)} -> {len(self.samples)}")


class DenseUAVMultiseasonalEval(Dataset):
    """Evaluation dataset with season-aware loading."""

    def __init__(
        self,
        data_folder,
        mode="query",
        seasons=None,
        transforms=None,
        height_filter=None,
        gallery_n=-1,
        loc_id_to_int=None,
    ):
        super().__init__()

        self.seasons = seasons or ["summer"]
        self.data_dict = get_data_seasonal(
            data_folder, seasons=self.seasons, height_filter=height_filter,
        )

        self.images = []
        self.labels = []
        self.sample_ids_set = set()

        if loc_id_to_int is not None:
            self.loc_id_to_int = loc_id_to_int
        else:
            self.loc_id_to_int = {
                loc_id: i for i, loc_id in enumerate(sorted(self.data_dict.keys()))
            }

        for loc_id, loc_data in sorted(self.data_dict.items()):
            if loc_id not in self.loc_id_to_int:
                continue
            for f in loc_data["files"]:
                self.images.append(f"{loc_data['path']}/{f}")
                self.labels.append(self.loc_id_to_int[loc_id])
                self.sample_ids_set.add(loc_id)

        if gallery_n > 0 and len(self.images) > gallery_n:
            indices = random.sample(range(len(self.images)), gallery_n)
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        self.transforms = transforms
        print(
            f"DenseUAV Multiseasonal {mode}: {len(self.images)} images "
            f"from {len(self.sample_ids_set)} locations ({self.seasons})"
        )

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        img = _load_image(img_path)
        if img is None:
            print(f"WARNING: cannot read {img_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.images) - 1))

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return self.sample_ids_set

    def get_loc_id_mapping(self):
        return self.loc_id_to_int
