"""
dataset.py
──────────
Handles all data loading, face detection, preprocessing, and augmentation
for the Deepfake Detection System.

Supported datasets:
  - FaceForensics++ (FF++)
  - Celeb-DF v2
  - DFDC (DeepFake Detection Challenge)
  - Any folder with real/ and fake/ subfolders
"""

import os
import cv2
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ── Load config ──────────────────────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Albumentations transforms ────────────────────────────────────────────────

def get_train_transforms(cfg):
    aug = cfg["augmentation"]
    size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=aug["horizontal_flip"]),
        A.Rotate(limit=aug["rotation_limit"], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=aug["brightness_contrast"],
            contrast_limit=aug["brightness_contrast"],
            p=0.5
        ),
        A.GaussianBlur(blur_limit=aug["blur_limit"], p=0.3),
        A.ImageCompression(
            quality_lower=aug["jpeg_quality"][0],
            quality_upper=aug["jpeg_quality"][1],
            p=0.3
        ),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
        A.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ToTensorV2(),
    ])


def get_val_transforms(cfg):
    aug = cfg["augmentation"]
    size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ToTensorV2(),
    ])


# ── Dataset class ────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Binary classification dataset.
    Label: 0 = real, 1 = fake
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback: return black image if file is corrupt
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, torch.tensor(label, dtype=torch.float32)


# ── Data builder ─────────────────────────────────────────────────────────────

def collect_image_paths(directory, extensions=(".jpg", ".jpeg", ".png", ".webp")):
    """Recursively collect all image paths from a directory."""
    paths = []
    for ext in extensions:
        paths.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(paths)


def build_datasets(cfg, verbose=True):
    """
    Scans real/ and fake/ folders, splits into train/val/test,
    returns three DeepfakeDataset objects and a class weight tensor.
    """
    real_dir = cfg["data"]["real_dir"]
    fake_dir = cfg["data"]["fake_dir"]
    seed = cfg["data"]["random_seed"]

    real_paths = collect_image_paths(real_dir)
    fake_paths = collect_image_paths(fake_dir)

    if verbose:
        print(f"  Real images : {len(real_paths):,}")
        print(f"  Fake images : {len(fake_paths):,}")
        print(f"  Total       : {len(real_paths) + len(fake_paths):,}")

    if len(real_paths) == 0 or len(fake_paths) == 0:
        raise ValueError(
            "No images found. Make sure data/real/ and data/fake/ "
            "contain images. See README for dataset setup."
        )

    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths)

    # Stratified split: train / val / test
    val_size = cfg["data"]["val_split"] / (cfg["data"]["val_split"] + cfg["data"]["test_split"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels,
        test_size=(cfg["data"]["val_split"] + cfg["data"]["test_split"]),
        stratify=all_labels,
        random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_size),
        stratify=y_temp,
        random_state=seed
    )

    if verbose:
        print(f"\n  Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}")

    train_tf = get_train_transforms(cfg)
    val_tf   = get_val_transforms(cfg)

    train_ds = DeepfakeDataset(X_train, y_train, transform=train_tf)
    val_ds   = DeepfakeDataset(X_val,   y_val,   transform=val_tf)
    test_ds  = DeepfakeDataset(X_test,  y_test,  transform=val_tf)

    # Class weights for imbalanced datasets
    n_real = y_train.count(0)
    n_fake = y_train.count(1)
    total  = n_real + n_fake
    class_weights = torch.tensor([total / (2 * n_real), total / (2 * n_fake)], dtype=torch.float32)

    return train_ds, val_ds, test_ds, class_weights


def build_dataloaders(cfg, verbose=True):
    """Returns train, val, test DataLoaders."""
    if verbose:
        print("\n[Dataset] Scanning image folders...")

    train_ds, val_ds, test_ds, class_weights = build_datasets(cfg, verbose)

    # Weighted sampler to handle class imbalance during training
    sample_weights = [
        class_weights[int(label)].item()
        for label in train_ds.labels
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    bs = cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,    num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,    num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights


# ── Face extractor (optional preprocessing step) ─────────────────────────────

def extract_faces_from_video(video_path, output_dir, max_frames=30, detector=None):
    """
    Extract face crops from a video file.
    Used during dataset preparation (run once, save crops).
    Requires facenet-pytorch: pip install facenet-pytorch
    """
    try:
        from facenet_pytorch import MTCNN
        if detector is None:
            detector = MTCNN(margin=20, keep_all=False, device="cpu")
    except ImportError:
        print("facenet-pytorch not installed. Skipping face detection.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

    saved = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        try:
            face = detector(pil_img)
            if face is not None:
                # face is a tensor (C, H, W) in [0,1] from MTCNN
                face_np = (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(out_path, face_bgr)
                saved.append(out_path)
        except Exception:
            continue

    cap.release()
    return saved


def prepare_ff_dataset(ff_root, output_root, max_frames_per_video=30):
    """
    Converts FaceForensics++ folder structure into flat real/ and fake/ folders.

    FF++ structure:
        ff_root/
          original_sequences/actors/c23/videos/
          manipulated_sequences/Deepfakes/c23/videos/
          manipulated_sequences/Face2Face/c23/videos/
          manipulated_sequences/FaceSwap/c23/videos/
          manipulated_sequences/NeuralTextures/c23/videos/
    """
    from facenet_pytorch import MTCNN
    detector = MTCNN(margin=20, keep_all=False, device="cpu")

    real_out = os.path.join(output_root, "real")
    fake_out = os.path.join(output_root, "fake")
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    # --- Real videos ---
    real_video_dir = os.path.join(ff_root, "original_sequences", "actors", "c23", "videos")
    if os.path.exists(real_video_dir):
        videos = list(Path(real_video_dir).glob("*.mp4"))
        print(f"Processing {len(videos)} real videos...")
        for v in tqdm(videos):
            out_dir = os.path.join(real_out, v.stem)
            extract_faces_from_video(v, out_dir, max_frames_per_video, detector)

    # --- Fake videos (4 manipulation types) ---
    fake_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    for fake_type in fake_types:
        fake_video_dir = os.path.join(ff_root, "manipulated_sequences", fake_type, "c23", "videos")
        if os.path.exists(fake_video_dir):
            videos = list(Path(fake_video_dir).glob("*.mp4"))
            print(f"Processing {len(videos)} {fake_type} videos...")
            for v in tqdm(videos):
                out_dir = os.path.join(fake_out, f"{fake_type}_{v.stem}")
                extract_faces_from_video(v, out_dir, max_frames_per_video, detector)

    print(f"\nDone. Faces saved to {output_root}")


if __name__ == "__main__":
    cfg = load_config()
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)
    print(f"\nClass weights → Real: {class_weights[0]:.3f} | Fake: {class_weights[1]:.3f}")
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape} | Labels: {labels[:8]}")
