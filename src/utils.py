"""
utils.py
────────
Shared utility functions used across the project.
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def set_seed(seed=42):
    """Reproducibility — fix all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  CPU mode (training will be slow — use GPU for best results)")
    return device


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_model_summary(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"\n  Model Summary:")
    print(f"  {'Total params':<20}: {total:>12,}")
    print(f"  {'Trainable params':<20}: {trainable:>12,}")
    print(f"  {'Frozen params':<20}: {frozen:>12,}")
    print(f"  {'Model size':<20}: {total * 4 / 1e6:>11.1f} MB (fp32)")


def count_dataset_stats(real_dir, fake_dir):
    """Print statistics about the dataset."""
    from pathlib import Path
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    real_count = sum(1 for p in Path(real_dir).rglob("*") if p.suffix.lower() in exts)
    fake_count = sum(1 for p in Path(fake_dir).rglob("*") if p.suffix.lower() in exts)
    total      = real_count + fake_count
    ratio      = fake_count / real_count if real_count > 0 else 0

    print(f"\n  Dataset Statistics:")
    print(f"  {'Real images':<20}: {real_count:>8,}")
    print(f"  {'Fake images':<20}: {fake_count:>8,}")
    print(f"  {'Total':<20}: {total:>8,}")
    print(f"  {'Fake/Real ratio':<20}: {ratio:>8.2f}x")
    print(f"  {'Balance':<20}: {'Balanced' if 0.5 < ratio < 2.0 else 'Imbalanced'}")


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"
