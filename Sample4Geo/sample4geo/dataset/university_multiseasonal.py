"""
University-1652 Multiseasonal Dataset for Sample4Geo.

Extends University-1652 with season-aware satellite loading.
Drone views are original only (no generated views for this dataset).

Structure:
    University1652_multiseasonal/
        train/
            satellite/0001/0001.jpg, 0001_autumn.jpg, 0001_winter.jpg, ...
            drone/0001/image-01.jpeg, image-02.jpeg, ...
        test/
            query_drone/0001/...
            gallery_satellite/0001/0001.jpg, 0001_autumn.jpg, ...
"""

import copy
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

SEASONS = ["summer", "autumn", "winter", "spring"]


def _parse_season(filename: str) -> str:
    """Extract season from filename. No suffix = summer (original)."""
    stem = Path(filename).stem
    for season in ["autumn", "winter", "spring"]:
        if stem.endswith(f"_{season}"):
            return season
    return "summer"


def get_data_seasonal(path: str, seasons: list[str] | None = None):
    """Get images grouped by location folder, filtered by season.

    For satellite: season is determined by filename suffix.
    For drone/other: all files are treated as 'summer' (original).
    """
    if seasons is None:
        seasons = SEASONS

    data = {}
    path = Path(path)

    for location_dir in sorted(path.iterdir()):
        if not location_dir.is_dir():
            continue
        loc_id = location_dir.name

        files = []
        for f in sorted(location_dir.iterdir()):
            if f.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.tif'):
                continue
            season = _parse_season(f.name)
            if season in seasons:
                files.append(f.name)

        if files:
            data[loc_id] = {
                "path": str(location_dir),
                "files": files,
            }

    return data


class U1652MultiseasonalTrain(Dataset):
    """Training dataset with season-aware satellite loading.

    For University-1652, the query is satellite (single view per building)
    and gallery is drone (multiple views). The original code pairs:
    query=satellite[0], gallery=drone[*].

    With seasonal satellite, we have multiple satellite views per location.
    Multipositive mode treats all satellite seasons as positives.

    Args:
        query_folder: Path to satellite images (with seasonal variants).
        gallery_folder: Path to drone images (original only).
        query_seasons: Which satellite seasons to use.
        gallery_seasons: Ignored for drone (always loads all).
        multipositive: All satellite seasons at same location are positives.
    """

    def __init__(
        self,
        query_folder,
        gallery_folder,
        query_seasons=None,
        gallery_seasons=None,
        multipositive=False,
        transforms_query=None,
        transforms_gallery=None,
        prob_flip=0.5,
        shuffle_batch_size=128,
    ):
        super().__init__()

        self.multipositive = multipositive
        self.query_seasons = query_seasons or ["summer"]

        self.query_dict = get_data_seasonal(query_folder, seasons=self.query_seasons)
        # Drone views don't have seasonal variants — load all
        self.gallery_dict = get_data_seasonal(gallery_folder, seasons=SEASONS)

        self.ids = sorted(
            set(self.query_dict.keys()) & set(self.gallery_dict.keys())
        )
        self.id_to_idx = {loc_id: i for i, loc_id in enumerate(self.ids)}

        # Build pairs: each satellite image paired with each drone image
        self.pairs = []
        for loc_id in self.ids:
            q_data = self.query_dict[loc_id]
            g_data = self.gallery_dict[loc_id]

            for q_file in q_data["files"]:
                q_path = f"{q_data['path']}/{q_file}"
                for g_file in g_data["files"]:
                    g_path = f"{g_data['path']}/{g_file}"
                    self.pairs.append((loc_id, q_path, g_path))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

        n_q = sum(len(d["files"]) for d in self.query_dict.values())
        n_g = sum(len(d["files"]) for d in self.gallery_dict.values())
        print(
            f"U1652 Multiseasonal Train: {len(self.ids)} locations, "
            f"{n_q} satellite imgs ({self.query_seasons}), "
            f"{n_g} drone imgs, {len(self.pairs)} pairs, "
            f"multipositive={self.multipositive}"
        )

    def __getitem__(self, index):
        loc_id, query_path, gallery_path = self.samples[index]

        query_img = cv2.imread(query_path)
        if query_img is None:
            print(f"WARNING: cannot read {query_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = cv2.imread(gallery_path)
        if gallery_img is None:
            print(f"WARNING: cannot read {gallery_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        loc_idx = self.id_to_idx[loc_id]
        return query_img, gallery_img, loc_idx

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """Custom shuffle with optional hard negative mining."""
        print("\nShuffle Dataset:")

        pairs_by_idx = {}
        for pair in self.pairs:
            loc_id = pair[0]
            idx = self.id_to_idx[loc_id]
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


class U1652MultiseasonalEval(Dataset):
    """Evaluation dataset with season-aware loading.

    Args:
        data_folder: Path to images.
        mode: "query" or "gallery".
        seasons: Which seasons to include (relevant for satellite folders).
        sample_ids: Set of valid sample IDs (for filtering gallery to match query).
    """

    def __init__(
        self,
        data_folder,
        mode="query",
        seasons=None,
        transforms=None,
        sample_ids=None,
        gallery_n=-1,
    ):
        super().__init__()

        self.transforms = transforms
        self.mode = mode
        self.given_sample_ids = sample_ids

        self.data_dict = get_data_seasonal(data_folder, seasons=seasons or SEASONS)

        self.images = []
        self.sample_ids_list = []

        for sample_id in sorted(self.data_dict.keys()):
            for f in self.data_dict[sample_id]["files"]:
                self.images.append(f"{self.data_dict[sample_id]['path']}/{f}")
                self.sample_ids_list.append(sample_id)

        if gallery_n > 0 and len(self.images) > gallery_n:
            indices = random.sample(range(len(self.images)), gallery_n)
            self.images = [self.images[i] for i in indices]
            self.sample_ids_list = [self.sample_ids_list[i] for i in indices]

        print(
            f"U1652 Multiseasonal {mode}: {len(self.images)} images "
            f"from {len(set(self.sample_ids_list))} locations ({seasons})"
        )

    def __getitem__(self, index):
        img_path = self.images[index]
        sample_id = self.sample_ids_list[index]

        img = cv2.imread(img_path)
        if img is None:
            print(f"WARNING: cannot read {img_path}, substituting random sample")
            return self.__getitem__(random.randint(0, len(self.images) - 1))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids_list)
