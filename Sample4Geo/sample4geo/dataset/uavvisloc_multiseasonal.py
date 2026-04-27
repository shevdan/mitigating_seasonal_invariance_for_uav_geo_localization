"""
UAV-VisLoc Multiseasonal Dataset for Sample4Geo.

Training modes:
1. Standard: original drone + satellite (any season).
2. Multipositive: original drone + satellite as primary pair,
   plus ONE random generated drone view as extra positive anchor.

Structure:
    UAV_VisLoc_multiseasonal/
        {flight_id}/
            drone/01_0001.JPG, 01_0001_autumn.JPG, 01_0001_winter.JPG, ...
            satellite/01_0001_sat.jpg, 01_0001_sat_autumn.jpg, ...
"""

import copy
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

SUMMER_LOCATIONS = ['01', '02', '05', '07', '08', '09', '10']
AUTUMN_LOCATIONS = ['03', '04', '06', '11']
ALL_LOCATIONS = sorted(SUMMER_LOCATIONS + AUTUMN_LOCATIONS)


def _parse_season_drone(filename: str) -> str:
    stem = Path(filename).stem
    for season in GENERATED_SEASONS:
        if stem.endswith(f"_{season}"):
            return season
    return "summer"


def _parse_season_sat(filename: str) -> str:
    stem = Path(filename).stem
    for season in GENERATED_SEASONS:
        if stem.endswith(f"_{season}"):
            return season
    return "summer"


def _get_position_id_drone(filename: str) -> str:
    stem = Path(filename).stem
    for season in ["_autumn", "_winter", "_spring"]:
        if stem.endswith(season):
            return stem[:-len(season)]
    return stem


def _get_position_id_sat(filename: str) -> str:
    stem = Path(filename).stem
    for season in ["_autumn", "_winter", "_spring"]:
        if stem.endswith(season):
            stem = stem[:-len(season)]
            break
    if stem.endswith("_sat"):
        stem = stem[:-4]
    return stem


def _load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class UAVVisLocMultiseasonalTrain(Dataset):
    """Training dataset for UAV-VisLoc with multipositive support.

    Query is ALWAYS original (summer) drone. Satellite can be any season.
    When multipositive=True, returns one random generated drone view as
    an additional positive anchor per batch item.
    """

    def __init__(
        self,
        data_root,
        flight_ids=None,
        sat_seasons=None,
        multipositive=False,
        transforms_query=None,
        transforms_gallery=None,
        prob_flip=0.5,
        shuffle_batch_size=128,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.multipositive = multipositive
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        sat_seasons = sat_seasons or ["summer"]
        if flight_ids is None:
            flight_ids = ALL_LOCATIONS

        # Build pairs: original drone → satellite (any season)
        self.pairs = []
        self.position_ids = set()

        for fid in flight_ids:
            drone_dir = self.data_root / fid / "drone"
            sat_dir = self.data_root / fid / "satellite"
            if not drone_dir.exists() or not sat_dir.exists():
                continue

            # Index satellite by (pos_id, season)
            sat_index = {}
            for f in sat_dir.iterdir():
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                season = _parse_season_sat(f.name)
                if season not in sat_seasons:
                    continue
                pos_id = _get_position_id_sat(f.name)
                sat_index.setdefault(pos_id, []).append(str(f))

            # Only original drone files
            for f in sorted(drone_dir.iterdir()):
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                if _parse_season_drone(f.name) != "summer":
                    continue
                pos_id = _get_position_id_drone(f.name)
                if pos_id not in sat_index:
                    continue
                for sat_path in sat_index[pos_id]:
                    self.pairs.append((pos_id, str(f), sat_path))
                    self.position_ids.add(pos_id)

        self.position_ids = sorted(self.position_ids)
        self.pos_id_to_idx = {pid: i for i, pid in enumerate(self.position_ids)}

        # Index generated drone views per position
        if self.multipositive:
            self._build_generated_index(flight_ids)

        self.samples = copy.deepcopy(self.pairs)

        n_gen = sum(len(v) for v in self.gen_paths_by_pos.values()) if self.multipositive else 0
        print(
            f"UAV-VisLoc Multiseasonal Train: {len(self.position_ids)} positions, "
            f"{len(self.pairs)} pairs, sat={sat_seasons}, "
            f"multipositive={self.multipositive}"
            f"{f', {n_gen} generated views' if self.multipositive else ''}"
        )

    def _build_generated_index(self, flight_ids):
        """Index generated drone views: pos_id → [path, ...]."""
        self.gen_paths_by_pos = {}
        for fid in flight_ids:
            drone_dir = self.data_root / fid / "drone"
            if not drone_dir.exists():
                continue
            for f in drone_dir.iterdir():
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                season = _parse_season_drone(f.name)
                if season == "summer":
                    continue  # skip original
                pos_id = _get_position_id_drone(f.name)
                self.gen_paths_by_pos.setdefault(pos_id, []).append(str(f))

    def _load_one_generated_view(self, pos_id: str, do_flip: bool):
        """Load one random generated drone view for a position."""
        if pos_id not in self.gen_paths_by_pos:
            return None
        candidates = self.gen_paths_by_pos[pos_id]
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
        pos_id, drone_path, sat_path = self.samples[index]

        drone_img = _load_image(drone_path)
        if drone_img is None:
            print(f"WARNING: cannot read {drone_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        sat_img = _load_image(sat_path)
        if sat_img is None:
            print(f"WARNING: cannot read {sat_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        do_flip = np.random.random() < self.prob_flip
        if do_flip:
            drone_img = cv2.flip(drone_img, 1)
            sat_img = cv2.flip(sat_img, 1)

        if self.transforms_query is not None:
            drone_img = self.transforms_query(image=drone_img)['image']
        if self.transforms_gallery is not None:
            sat_img = self.transforms_gallery(image=sat_img)['image']

        loc_idx = self.pos_id_to_idx[pos_id]

        if self.multipositive:
            gen_view = self._load_one_generated_view(pos_id, do_flip)
            if gen_view is not None:
                gen_tensor = gen_view.unsqueeze(0)  # [1, C, H, W]
            else:
                gen_tensor = torch.empty(0)
            return drone_img, sat_img, loc_idx, gen_tensor

        return drone_img, sat_img, loc_idx

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """Custom shuffle with optional hard negative mining."""
        print("\nShuffle Dataset:")

        pairs_by_idx = {}
        for pair in self.pairs:
            idx = self.pos_id_to_idx[pair[0]]
            pairs_by_idx.setdefault(idx, []).append(pair)

        idx_pool = list(pairs_by_idx.keys())
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
                current_batch.append(random.choice(pairs_by_idx[idx]))
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
                                if idx_near in pairs_by_idx:
                                    idx_batch.add(idx_near)
                                    current_batch.append(random.choice(pairs_by_idx[idx_near]))
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


class UAVVisLocMultiseasonalEval(Dataset):
    """Evaluation dataset with season-aware loading."""

    def __init__(
        self,
        data_root,
        mode="query",
        seasons=None,
        flight_ids=None,
        transforms=None,
        pos_id_to_int=None,
    ):
        super().__init__()

        data_root = Path(data_root)
        seasons = seasons or ["summer"]
        if flight_ids is None:
            flight_ids = ALL_LOCATIONS

        self.images = []
        self.labels = []
        self.position_ids_set = set()

        all_positions = set()
        entries = []

        for fid in flight_ids:
            if mode == "query" or mode == "query_train":
                src_dir = data_root / fid / "drone"
            else:
                src_dir = data_root / fid / "satellite"

            if not src_dir.exists():
                continue

            for f in sorted(src_dir.iterdir()):
                if f.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue

                if mode in ("query", "query_train"):
                    season = _parse_season_drone(f.name)
                    pos_id = _get_position_id_drone(f.name)
                else:
                    season = _parse_season_sat(f.name)
                    pos_id = _get_position_id_sat(f.name)

                if season not in seasons:
                    continue

                all_positions.add(pos_id)
                entries.append((pos_id, str(f)))

        if pos_id_to_int is not None:
            self.pos_id_to_int = pos_id_to_int
        else:
            self.pos_id_to_int = {
                pid: i for i, pid in enumerate(sorted(all_positions))
            }

        for pos_id, path in entries:
            if pos_id not in self.pos_id_to_int:
                continue
            self.images.append(path)
            self.labels.append(self.pos_id_to_int[pos_id])
            self.position_ids_set.add(pos_id)

        self.transforms = transforms

        print(
            f"UAV-VisLoc Multiseasonal {mode}: {len(self.images)} images "
            f"from {len(self.position_ids_set)} positions ({seasons})"
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
        return self.position_ids_set

    def get_pos_id_mapping(self):
        return self.pos_id_to_int
