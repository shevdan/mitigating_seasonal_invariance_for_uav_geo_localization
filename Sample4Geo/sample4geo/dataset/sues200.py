"""
SUES-200 Dataset for Sample4Geo.

SUES-200 contains:
- 200 locations (0001-0200)
- 4 altitudes per location: 150m, 200m, 250m, 300m
- 50 drone frames per altitude per location (40,000 total drone images)
- 1 satellite image per location (200 total)

Structure:
    drone_view_512/
        0001/
            150/0.jpg, 1.jpg, ..., 49.jpg
            200/0.jpg, ..., 49.jpg
            250/0.jpg, ..., 49.jpg
            300/0.jpg, ..., 49.jpg
        0002/
            ...
        0200/
    satellite-view/
        0001/0.png
        0002/0.png
        ...
        0200/0.png

No predefined train/test split - we use 150 train / 50 test locations by default.
"""

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
from pathlib import Path


# Default train/test split: 150 train, 50 test
DEFAULT_TRAIN_LOCATIONS = [f"{i:04d}" for i in range(1, 151)]  # 0001-0150
DEFAULT_TEST_LOCATIONS = [f"{i:04d}" for i in range(151, 201)]  # 0151-0200

ALTITUDES = ['150', '200', '250', '300']


def get_drone_data(path, locations=None, altitude_filter=None):
    """Get all drone images from directory structure.

    Args:
        path: Path to drone_view_512 folder
        locations: List of location IDs to include (e.g., ['0001', '0002'])
        altitude_filter: None for all altitudes, or '150', '200', '250', '300'

    Returns:
        dict: {location_id: {altitude: [frame_paths]}}
    """
    data = {}
    path = Path(path)

    altitudes = [altitude_filter] if altitude_filter else ALTITUDES

    for location_dir in sorted(path.iterdir()):
        if not location_dir.is_dir():
            continue

        loc_id = location_dir.name

        # Filter by locations if provided
        if locations is not None and loc_id not in locations:
            continue

        data[loc_id] = {}

        for alt in altitudes:
            alt_dir = location_dir / alt
            if not alt_dir.exists():
                continue

            frames = []
            for f in sorted(alt_dir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    frames.append(str(f))

            if frames:
                data[loc_id][alt] = frames

    return data


def get_satellite_data(path, locations=None):
    """Get all satellite images from directory structure.

    Args:
        path: Path to satellite-view folder
        locations: List of location IDs to include

    Returns:
        dict: {location_id: satellite_path}
    """
    data = {}
    path = Path(path)

    for location_dir in sorted(path.iterdir()):
        if not location_dir.is_dir():
            continue

        loc_id = location_dir.name

        # Filter by locations if provided
        if locations is not None and loc_id not in locations:
            continue

        # Find satellite image (usually 0.png)
        for f in location_dir.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                data[loc_id] = str(f)
                break

    return data


class SUES200DatasetTrain(Dataset):
    """Training dataset for SUES-200."""

    def __init__(self,
                 drone_folder,
                 satellite_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 locations=None,
                 altitude_filter=None,
                 frames_per_location=None):
        """
        Args:
            drone_folder: Path to drone_view_512 folder
            satellite_folder: Path to satellite-view folder
            transforms_query: Transforms for drone images
            transforms_gallery: Transforms for satellite images
            prob_flip: Probability of horizontal flip
            shuffle_batch_size: Batch size for custom shuffle
            locations: List of location IDs to use (default: 0001-0150)
            altitude_filter: None for all altitudes, or '150', '200', '250', '300'
            frames_per_location: Number of frames to sample per location/altitude (None for all)
        """
        super().__init__()

        if locations is None:
            locations = DEFAULT_TRAIN_LOCATIONS

        self.drone_dict = get_drone_data(drone_folder, locations, altitude_filter)
        self.satellite_dict = get_satellite_data(satellite_folder, locations)

        # Use only locations that exist in both
        self.ids = list(set(self.drone_dict.keys()).intersection(self.satellite_dict.keys()))
        self.ids.sort()

        self.pairs = []

        for loc_id in self.ids:
            sat_path = self.satellite_dict[loc_id]

            for altitude, frames in self.drone_dict[loc_id].items():
                # Sample frames if specified
                if frames_per_location is not None and len(frames) > frames_per_location:
                    frames = random.sample(frames, frames_per_location)

                for drone_path in frames:
                    # Store: (location_id, altitude, drone_path, satellite_path)
                    self.pairs.append((loc_id, altitude, drone_path, sat_path))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

        n_altitudes = len([altitude_filter] if altitude_filter else ALTITUDES)
        print(f"SUES-200 Train: {len(self.ids)} locations, {n_altitudes} altitudes, {len(self.pairs)} pairs")

    def __getitem__(self, index):
        loc_id, altitude, drone_path, sat_path = self.samples[index]

        # Read drone image
        drone_img = cv2.imread(drone_path)
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

        # Read satellite image
        sat_img = cv2.imread(sat_path)
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        # Random flip (same for both)
        if np.random.random() < self.prob_flip:
            drone_img = cv2.flip(drone_img, 1)
            sat_img = cv2.flip(sat_img, 1)

        # Apply transforms
        if self.transforms_query is not None:
            drone_img = self.transforms_query(image=drone_img)['image']

        if self.transforms_gallery is not None:
            sat_img = self.transforms_gallery(image=sat_img)['image']

        return drone_img, sat_img, loc_id

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        """Custom shuffle for unique class_id sampling in batch."""
        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)

        # Group by location ID
        idx_batch = set()
        pairs_epoch = set()

        batches = []
        current_batch = []
        break_counter = 0

        pbar = tqdm()

        while True:
            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                loc_id = pair[0]

                if loc_id not in idx_batch and loc_id not in pairs_epoch:
                    idx_batch.add(loc_id)
                    current_batch.append(pair)
                    pairs_epoch.add(loc_id)
                    break_counter = 0
                else:
                    if loc_id not in pairs_epoch:
                        pair_pool.append(pair)
                    break_counter += 1

                if break_counter >= 512:
                    break
            else:
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


class SUES200DatasetEval(Dataset):
    """Evaluation dataset for SUES-200."""

    def __init__(self,
                 data_folder,
                 mode="query",
                 transforms=None,
                 locations=None,
                 altitude_filter=None,
                 gallery_n=-1,
                 loc_id_to_int=None,
                 frames_per_location=None):
        """
        Args:
            data_folder: Path to drone_view_512 (query) or satellite-view (gallery)
            mode: 'query' for drone, 'gallery' for satellite
            transforms: Image transforms
            locations: List of location IDs to use (default: 0151-0200 for test)
            altitude_filter: None for all altitudes, or specific altitude
            gallery_n: Limit gallery size (-1 for all)
            loc_id_to_int: Mapping from location ID to integer (for consistent labels)
            frames_per_location: Number of frames to sample per location (None for all)
        """
        super().__init__()

        self.mode = mode
        self.transforms = transforms

        if locations is None:
            locations = DEFAULT_TEST_LOCATIONS

        self.images = []
        self.labels = []
        self.sample_ids_set = set()

        if mode == "query":
            # Load drone images
            data_dict = get_drone_data(data_folder, locations, altitude_filter)

            # Create or use provided mapping
            if loc_id_to_int is not None:
                self.loc_id_to_int = loc_id_to_int
            else:
                all_loc_ids = sorted(data_dict.keys())
                self.loc_id_to_int = {loc_id: i for i, loc_id in enumerate(all_loc_ids)}

            for loc_id in sorted(data_dict.keys()):
                for altitude, frames in data_dict[loc_id].items():
                    # Sample frames if specified
                    if frames_per_location is not None and len(frames) > frames_per_location:
                        frames = random.sample(frames, frames_per_location)

                    for frame_path in frames:
                        self.images.append(frame_path)
                        self.labels.append(self.loc_id_to_int[loc_id])
                        self.sample_ids_set.add(loc_id)

        else:
            # Load satellite images
            data_dict = get_satellite_data(data_folder, locations)

            # Create or use provided mapping
            if loc_id_to_int is not None:
                self.loc_id_to_int = loc_id_to_int
            else:
                all_loc_ids = sorted(data_dict.keys())
                self.loc_id_to_int = {loc_id: i for i, loc_id in enumerate(all_loc_ids)}

            for loc_id in sorted(data_dict.keys()):
                self.images.append(data_dict[loc_id])
                self.labels.append(self.loc_id_to_int[loc_id])
                self.sample_ids_set.add(loc_id)

        # Limit gallery size if specified
        if gallery_n > 0 and len(self.images) > gallery_n:
            indices = list(range(len(self.images)))
            random.shuffle(indices)
            indices = indices[:gallery_n]
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        print(f"SUES-200 {mode}: {len(self.images)} images from {len(self.sample_ids_set)} locations")

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return self.sample_ids_set

    def get_loc_id_mapping(self):
        """Return the location ID to integer mapping for use by gallery dataset."""
        return self.loc_id_to_int


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    """Get transforms for SUES-200 dataset."""

    val_transforms = A.Compose([
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_sat_transforms = A.Compose([
        A.ImageCompression(quality_range=(90, 100), p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(
                num_holes_range=(10, 25),
                hole_height_range=(int(0.1*img_size[0]), int(0.2*img_size[0])),
                hole_width_range=(int(0.1*img_size[0]), int(0.2*img_size[0])),
                p=1.0
            ),
        ], p=0.3),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    train_drone_transforms = A.Compose([
        A.ImageCompression(quality_range=(90, 100), p=0.5),
        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
        A.OneOf([
            A.AdvancedBlur(p=1.0),
            A.Sharpen(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GridDropout(ratio=0.4, p=1.0),
            A.CoarseDropout(
                num_holes_range=(10, 25),
                hole_height_range=(int(0.1*img_size[0]), int(0.2*img_size[0])),
                hole_width_range=(int(0.1*img_size[0]), int(0.2*img_size[0])),
                p=1.0
            ),
        ], p=0.3),
        A.Normalize(mean, std),
        ToTensorV2(),
    ])

    return val_transforms, train_sat_transforms, train_drone_transforms
