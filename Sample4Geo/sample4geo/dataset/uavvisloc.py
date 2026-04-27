"""
UAV-VisLoc Dataset for Sample4Geo.

This dataset contains:
- 11 locations across China
- Drone images with GPS metadata
- Large satellite TIF maps

Seasonal distribution:
- Summer (June): Locations 05, 08, 09
- Autumn (Sept-Oct): Locations 01, 02, 03, 04, 06, 10, 11
"""

import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import time
import random
from pathlib import Path

try:
    import rasterio
    from rasterio.windows import Window
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not installed. Install with: pip install rasterio")


# Seasonal groupings
SUMMER_LOCATIONS = ['01', '02', '05', '07', '08', '09', '10']
AUTUMN_LOCATIONS = ['03', '04', '06', '11']
ALL_LOCATIONS = SUMMER_LOCATIONS + AUTUMN_LOCATIONS


def get_satellite_bounds(csv_path):
    """Read satellite coordinate bounds from CSV."""
    df = pd.read_csv(csv_path)
    bounds = {}
    for _, row in df.iterrows():
        loc_id = row['mapname'].replace('satellite', '').replace('.tif', '')
        bounds[loc_id] = {
            'lt_lat': row['LT_lat_map'],  # Top-left latitude
            'lt_lon': row['LT_lon_map'],  # Top-left longitude
            'rb_lat': row['RB_lat_map'],  # Bottom-right latitude
            'rb_lon': row['RB_lon_map'],  # Bottom-right longitude
            'region': row['region']
        }
    return bounds


def gps_to_pixel(lat, lon, bounds, img_height, img_width):
    """Convert GPS coordinates to pixel coordinates in satellite image."""
    # Latitude decreases from top to bottom
    lat_range = bounds['lt_lat'] - bounds['rb_lat']
    # Longitude increases from left to right
    lon_range = bounds['rb_lon'] - bounds['lt_lon']

    # Normalized position (0-1)
    y_norm = (bounds['lt_lat'] - lat) / lat_range
    x_norm = (lon - bounds['lt_lon']) / lon_range

    # Pixel coordinates
    px_y = int(y_norm * img_height)
    px_x = int(x_norm * img_width)

    return px_x, px_y


def extract_satellite_patch(tif_path, center_x, center_y, patch_size=512):
    """Extract a patch from satellite TIF at given pixel coordinates."""
    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio required for TIF processing")

    half_size = patch_size // 2

    with rasterio.open(tif_path) as src:
        # Calculate window bounds
        col_off = max(0, center_x - half_size)
        row_off = max(0, center_y - half_size)

        # Adjust for boundaries
        width = min(patch_size, src.width - col_off)
        height = min(patch_size, src.height - row_off)

        window = Window(col_off, row_off, width, height)

        # Read RGB bands
        patch = src.read([1, 2, 3], window=window)
        patch = np.transpose(patch, (1, 2, 0))  # CHW -> HWC

        # Pad if necessary
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
            padded[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded

    return patch


class UAVVisLocPreprocessor:
    """Preprocess UAV-VisLoc dataset to extract satellite patches."""

    def __init__(self, data_root, output_dir, patch_size=512):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size

        # Read satellite bounds
        bounds_file = self.data_root / "satellite_ coordinates_range.csv"
        self.sat_bounds = get_satellite_bounds(bounds_file)

    def process_location(self, loc_id):
        """Process a single location to extract satellite patches."""
        loc_dir = self.data_root / loc_id
        csv_path = loc_dir / f"{loc_id}.csv"
        tif_path = loc_dir / f"satellite{loc_id}.tif"

        if not tif_path.exists():
            print(f"Satellite TIF not found for location {loc_id}")
            return []

        # Read drone metadata
        df = pd.read_csv(csv_path)
        bounds = self.sat_bounds[loc_id]

        # Get satellite image dimensions
        with rasterio.open(tif_path) as src:
            img_height = src.height
            img_width = src.width

        # Create output directories
        drone_out = self.output_dir / "drone" / loc_id
        sat_out = self.output_dir / "satellite" / loc_id
        drone_out.mkdir(parents=True, exist_ok=True)
        sat_out.mkdir(parents=True, exist_ok=True)

        pairs = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Location {loc_id}"):
            filename = row['filename']
            lat = row['lat']
            lon = row['lon']

            # Convert GPS to pixel coordinates
            px_x, px_y = gps_to_pixel(lat, lon, bounds, img_height, img_width)

            # Extract satellite patch
            try:
                sat_patch = extract_satellite_patch(
                    str(tif_path), px_x, px_y, self.patch_size
                )
            except Exception as e:
                print(f"Error extracting patch for {filename}: {e}")
                continue

            # Get drone image path
            drone_path = loc_dir / "drone" / filename
            if not drone_path.exists():
                continue

            # Save satellite patch
            sat_filename = filename.replace('.JPG', '_sat.jpg').replace('.jpg', '_sat.jpg')
            sat_save_path = sat_out / sat_filename
            cv2.imwrite(str(sat_save_path), cv2.cvtColor(sat_patch, cv2.COLOR_RGB2BGR))

            # Copy/link drone image
            drone_save_path = drone_out / filename
            if not drone_save_path.exists():
                # Copy drone image
                drone_img = cv2.imread(str(drone_path))
                cv2.imwrite(str(drone_save_path), drone_img)

            pairs.append({
                'loc_id': loc_id,
                'drone_path': str(drone_save_path),
                'sat_path': str(sat_save_path),
                'lat': lat,
                'lon': lon,
                'date': row['date'],
                'height': row['height']
            })

        return pairs

    def process_all(self, locations=None):
        """Process all or specified locations."""
        if locations is None:
            locations = ALL_LOCATIONS

        all_pairs = []
        for loc_id in locations:
            pairs = self.process_location(loc_id)
            all_pairs.extend(pairs)

        # Save pairs metadata
        df = pd.DataFrame(all_pairs)
        df.to_csv(self.output_dir / "pairs.csv", index=False)

        return all_pairs


class UAVVisLocDatasetTrain(Dataset):
    """Training dataset for UAV-VisLoc."""

    def __init__(self,
                 data_root,
                 locations=None,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128,
                 mode='drone2sat'):
        """
        Args:
            data_root: Path to preprocessed UAV-VisLoc data
            locations: List of location IDs to use (for seasonal split)
            transforms_query: Transforms for query images
            transforms_gallery: Transforms for gallery images
            prob_flip: Probability of flipping both images
            shuffle_batch_size: Batch size for custom sampling
            mode: 'drone2sat' or 'sat2drone'
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.mode = mode

        # Load pairs metadata
        pairs_csv = self.data_root / "pairs.csv"
        if pairs_csv.exists():
            df = pd.read_csv(pairs_csv)
        else:
            raise FileNotFoundError(
                f"Pairs CSV not found at {pairs_csv}. "
                "Run UAVVisLocPreprocessor first."
            )

        # Filter by locations if specified
        if locations is not None:
            df = df[df['loc_id'].astype(str).str.zfill(2).isin(locations)]

        # Reset index to get unique pair IDs
        df = df.reset_index(drop=True)

        # Create pairs list - use row index as unique ID (each GPS position is unique)
        self.pairs = []
        for idx, row in df.iterrows():
            # Use unique pair index as ID for contrastive learning
            # Each drone image has a unique GPS position
            self.pairs.append((
                idx,  # Unique pair ID
                row['drone_path'],
                row['sat_path']
            ))

        self.samples = copy.deepcopy(self.pairs)
        print(f"Loaded {len(self.samples)} training pairs from {len(df['loc_id'].unique())} locations")

    def __getitem__(self, index):
        pair_id, drone_path, sat_path = self.samples[index]

        # Read images
        drone_img = cv2.imread(drone_path)
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)

        sat_img = cv2.imread(sat_path)
        sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)

        # Random flip
        if np.random.random() < self.prob_flip:
            drone_img = cv2.flip(drone_img, 1)
            sat_img = cv2.flip(sat_img, 1)

        # Apply transforms
        if self.mode == 'drone2sat':
            query_img = drone_img
            gallery_img = sat_img
            query_transforms = self.transforms_query
            gallery_transforms = self.transforms_gallery
        else:
            query_img = sat_img
            gallery_img = drone_img
            query_transforms = self.transforms_gallery
            gallery_transforms = self.transforms_query

        if query_transforms is not None:
            query_img = query_transforms(image=query_img)['image']
        if gallery_transforms is not None:
            gallery_img = gallery_transforms(image=gallery_img)['image']

        return query_img, gallery_img, pair_id

    def __len__(self):
        return len(self.samples)

    def shuffle(self):
        """Custom shuffle for unique ID sampling in batch.

        For UAV-VisLoc, each pair has a unique ID (GPS position),
        so we ensure no duplicate IDs in a batch.
        """
        print("\nShuffle Dataset:")

        pair_pool = copy.deepcopy(self.pairs)
        random.shuffle(pair_pool)

        pairs_epoch = set()
        idx_batch = set()

        batches = []
        current_batch = []
        break_counter = 0

        pbar = tqdm()

        while True:
            pbar.update()

            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                idx = pair[0]  # Unique pair ID

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

        # Add remaining samples from current batch
        if current_batch:
            batches.extend(current_batch)

        self.samples = batches
        print(f"Original: {len(self.pairs)} - After shuffle: {len(self.samples)}")


class UAVVisLocDatasetEval(Dataset):
    """Evaluation dataset for UAV-VisLoc."""

    def __init__(self,
                 data_root,
                 locations=None,
                 transforms=None,
                 mode='query'):
        """
        Args:
            data_root: Path to preprocessed UAV-VisLoc data
            locations: List of location IDs to use
            transforms: Image transforms
            mode: 'query' (drone) or 'gallery' (satellite)
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.transforms = transforms
        self.mode = mode

        # Load pairs metadata
        pairs_csv = self.data_root / "pairs.csv"
        df = pd.read_csv(pairs_csv)

        if locations is not None:
            df = df[df['loc_id'].astype(str).str.zfill(2).isin(locations)]

        # Reset index for consistent labeling between query and gallery
        df = df.reset_index(drop=True)

        self.images = []
        self.labels = []

        for idx, row in df.iterrows():
            if mode == 'query':
                self.images.append(row['drone_path'])
            else:
                self.images.append(row['sat_path'])

            # Use row index as label - each pair has unique GPS position
            # Query index i should match Gallery index i
            self.labels.append(idx)

        print(f"Loaded {len(self.images)} {mode} images")

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.labels)


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    """Get transforms for UAV-VisLoc dataset."""

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


if __name__ == "__main__":
    # Test preprocessing
    data_root = "/home/shevdan/Documents/msc/diploma/data/UAV_VisLoc_dataset"
    output_dir = "/home/shevdan/Documents/msc/diploma/data/UAV_VisLoc_processed"

    preprocessor = UAVVisLocPreprocessor(data_root, output_dir, patch_size=512)

    # Process just one location for testing
    pairs = preprocessor.process_location('01')
    print(f"Processed {len(pairs)} pairs for location 01")
