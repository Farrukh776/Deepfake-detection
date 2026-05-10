"""
dataset.py — Kaggle pre-split structure
Label: 0 = real, 1 = fake
"""

import cv2, yaml, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_train_transforms(cfg):
    aug  = cfg["augmentation"]
    size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=aug["horizontal_flip"]),
        A.Rotate(limit=aug["rotation_limit"], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_contrast"],
            contrast_limit=aug["brightness_contrast"], p=0.5),
        A.GaussianBlur(blur_limit=aug["blur_limit"], p=0.3),
        A.ImageCompression(
            quality_lower=aug["jpeg_quality"][0],
            quality_upper=aug["jpeg_quality"][1], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ToTensorV2(),
    ])


def get_val_transforms(cfg):
    aug  = cfg["augmentation"]
    size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ToTensorV2(),
    ])


class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)


def collect_image_paths(directory, extensions=(".jpg", ".jpeg", ".png", ".webp")):
    paths = []
    for ext in extensions:
        paths.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(paths)


def load_split(real_dir, fake_dir, max_per_class=None, label_name=""):
    real_paths = collect_image_paths(real_dir)
    fake_paths = collect_image_paths(fake_dir)

    # Cap samples per class if specified
    if max_per_class:
        random.seed(42)
        if len(real_paths) > max_per_class:
            real_paths = random.sample(real_paths, max_per_class)
        if len(fake_paths) > max_per_class:
            fake_paths = random.sample(fake_paths, max_per_class)

    paths  = real_paths + fake_paths
    labels = [0] * len(real_paths) + [1] * len(fake_paths)

    if label_name:
        print(f"  {label_name:<6} → Real: {len(real_paths):>6,}  "
              f"Fake: {len(fake_paths):>6,}  Total: {len(paths):>7,}")
    return paths, labels


def build_dataloaders(cfg, verbose=True):
    d   = cfg["data"]
    cap = d.get("max_samples_per_class", None)  # None = use all

    if verbose:
        print("\n[Dataset] Loading pre-split Kaggle dataset...")
        if cap:
            print(f"  Capping training at {cap:,} images per class")

    val_cap  = d.get("max_val_samples_per_class", 2000)
    train_paths, train_labels = load_split(d["train_real"], d["train_fake"], cap,     "Train")
    val_paths,   val_labels   = load_split(d["val_real"],   d["val_fake"],   val_cap, "Val")
    test_paths,  test_labels  = load_split(d["test_real"],  d["test_fake"],  None,    "Test")

    if not train_paths:
        raise ValueError("No training images found! Check data/train/real and data/train/fake.")

    train_tf = get_train_transforms(cfg)
    val_tf   = get_val_transforms(cfg)

    train_ds = DeepfakeDataset(train_paths, train_labels, transform=train_tf)
    val_ds   = DeepfakeDataset(val_paths,   val_labels,   transform=val_tf)
    test_ds  = DeepfakeDataset(test_paths,  test_labels,  transform=val_tf)

    # Weighted sampler
    n_real = train_labels.count(0)
    n_fake = train_labels.count(1)
    total  = n_real + n_fake
    class_weights  = torch.tensor([total/(2*n_real), total/(2*n_fake)], dtype=torch.float32)
    sample_weights = [class_weights[int(l)].item() for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,   num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,   num_workers=nw, pin_memory=True)

    if verbose:
        print(f"  Class weights → Real: {class_weights[0]:.3f} | Fake: {class_weights[1]:.3f}")

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    cfg = load_config()
    train_loader, val_loader, test_loader, cw = build_dataloaders(cfg)
    imgs, labels = next(iter(train_loader))
    print(f"\nBatch shape : {imgs.shape}")
    print(f"Labels      : {labels[:8]}")
