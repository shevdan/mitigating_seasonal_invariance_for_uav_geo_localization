"""
DenseUAV Dataset for Sample4Geo.

DenseUAV contains:
- 2256 training locations
- 777 test query locations
- 3033 test gallery locations (train + test)
- Multiple heights per location: H80, H90, H100 (80m, 90m, 100m)
- Satellite images in TIF format

Structure:
    train/
        drone/000000/H80.JPG, H90.JPG, H100.JPG
        satellite/000000/H80.tif, H90.tif, H100.tif, H80_old.tif, ...
    test/
        query_drone/002256/H80.JPG, ...
        gallery_satellite/000000/H80.tif, ...
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


def get_data(path, extensions=('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.tif', '.TIF')):
    """Get all images from directory structure."""
    data = {}
    path = Path(path)

    for location_dir in sorted(path.iterdir()):
        if location_dir.is_dir():
            loc_id = location_dir.name
            files = [f.name for f in location_dir.iterdir()
                    if f.suffix in extensions and '_old' not in f.name]  # Skip _old variants
            if files:
                data[loc_id] = {
                    "path": str(location_dir),
                    "files": sorted(files)
                }

    return data


class DenseUAVDatasetTrain(Dataset):
    """Training dataset for DenseUAV."""

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 height_filter=None):  # e.g., 'H80' to use only 80m height
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.gallery_dict = get_data(gallery_folder)

        # Use only locations that exist in both
        self.ids = list(set(self.query_dict.keys()).intersection(self.gallery_dict.keys()))
        self.ids.sort()

        self.pairs = []

        for idx in self.ids:
            query_path = self.query_dict[idx]["path"]
            query_files = self.query_dict[idx]["files"]

            gallery_path = self.gallery_dict[idx]["path"]
            gallery_files = self.gallery_dict[idx]["files"]

            # Filter by height if specified
            if height_filter:
                query_files = [f for f in query_files if height_filter in f]
                gallery_files = [f for f in gallery_files if height_filter in f]

            # Create pairs for each query-gallery combination
            for q_file in query_files:
                q_path = f"{query_path}/{q_file}"

                # Match with corresponding gallery (same height)
                q_height = q_file.split('.')[0]  # e.g., 'H80'
                matching_gallery = [f for f in gallery_files if q_height in f]

                for g_file in matching_gallery:
                    g_path = f"{gallery_path}/{g_file}"
                    self.pairs.append((idx, q_path, g_path))

        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size

        self.samples = copy.deepcopy(self.pairs)

        print(f"DenseUAV Train: {len(self.ids)} locations, {len(self.pairs)} pairs")

    def __getitem__(self, index):
        idx, query_img_path, gallery_img_path = self.samples[index]

        # Read query (drone) image
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # Read gallery (satellite) image - handle TIF format
        gallery_img = cv2.imread(gallery_img_path, cv2.IMREAD_UNCHANGED)
        if gallery_img is None:
            # Try with different flags for TIF
            gallery_img = cv2.imread(gallery_img_path)
        if len(gallery_img.shape) == 2:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_GRAY2RGB)
        elif gallery_img.shape[2] == 4:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGRA2RGB)
        else:
            gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        # Random flip
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

        # Apply transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, idx

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
                idx = pair[0]

                if idx not in idx_batch and idx not in pairs_epoch:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(idx)
                    break_counter = 0
                else:
                    if idx not in pairs_epoch:
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


class DenseUAVDatasetEval(Dataset):
    """Evaluation dataset for DenseUAV."""

    def __init__(self,
                 data_folder,
                 mode="query",
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1,
                 height_filter=None,
                 loc_id_to_int=None):
        super().__init__()

        self.data_folder = Path(data_folder)
        self.mode = mode
        self.transforms = transforms
        self.height_filter = height_filter

        # Get all data
        self.data_dict = get_data(data_folder)

        # Filter by sample_ids if provided (for gallery matching query)
        if sample_ids is not None:
            self.data_dict = {k: v for k, v in self.data_dict.items() if k in sample_ids}

        # Build image list
        self.images = []
        self.labels = []
        self.sample_ids_set = set()

        # Create or use provided mapping from string location ID to integer
        if loc_id_to_int is not None:
            self.loc_id_to_int = loc_id_to_int
        else:
            sorted_loc_ids = sorted(self.data_dict.keys())
            self.loc_id_to_int = {loc_id: i for i, loc_id in enumerate(sorted_loc_ids)}

        for loc_id, loc_data in sorted(self.data_dict.items()):
            files = loc_data["files"]

            if height_filter:
                files = [f for f in files if height_filter in f]

            for f in files:
                self.images.append(f"{loc_data['path']}/{f}")
                # Use integer label for evaluation
                self.labels.append(self.loc_id_to_int[loc_id])
                self.sample_ids_set.add(loc_id)

        # Limit gallery size if specified
        if gallery_n > 0 and len(self.images) > gallery_n:
            indices = list(range(len(self.images)))
            random.shuffle(indices)
            indices = indices[:gallery_n]
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

        print(f"DenseUAV {mode}: {len(self.images)} images from {len(self.sample_ids_set)} locations")

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = cv2.imread(img_path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
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
    """Get transforms for DenseUAV dataset."""

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
