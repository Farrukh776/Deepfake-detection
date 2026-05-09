"""
gradcam.py
──────────
Grad-CAM visualization for model explainability.

Generates heatmap overlays showing WHICH regions of a face
the model focuses on when predicting "real" or "fake".

Strong Grad-CAM activations on face boundaries, eyes, or mouth
are common indicators of manipulation artifacts.
"""

import os
import sys
import yaml
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Target layer helper ───────────────────────────────────────────────────────

def get_target_layer(model):
    """
    Returns the last convolutional block of the EfficientNet backbone.
    This is where spatial feature maps are richest for Grad-CAM.
    """
    # For EfficientNet-B4 (timm), the last block is model.backbone.blocks[-1]
    # We target the last depthwise conv in the last block
    try:
        return [model.backbone.blocks[-1]]
    except (AttributeError, IndexError):
        # Fallback: get any Conv2d layer
        for module in reversed(list(model.backbone.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return [module]
    raise ValueError("Could not find target layer for Grad-CAM")


# ── Denormalize image for display ─────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    """Convert normalized tensor → numpy uint8 image (H, W, 3) in [0, 255]."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return img.astype(np.float32)


# ── Single image Grad-CAM ─────────────────────────────────────────────────────

def generate_gradcam(model, image_tensor, target_layers, use_cuda=False):
    """
    Generate Grad-CAM heatmap for one image.

    Args:
        model:         DeepfakeDetector model (eval mode)
        image_tensor:  (1, C, H, W) normalized tensor
        target_layers: list of target layer modules
        use_cuda:      whether input is on GPU

    Returns:
        cam:          (H, W) heatmap in [0, 1]
        overlay:      (H, W, 3) RGB image with heatmap overlaid
        raw_img:      (H, W, 3) original denormalized RGB image
        score:        model's fake probability (float)
    """
    model.eval()

    # Get model prediction first
    with torch.no_grad():
        score = model(image_tensor).item()

    # GradCAM++ gives sharper, more localized heatmaps than vanilla GradCAM
    cam_algo = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # Target = "make fake score high" (class 1)
    targets = [BinaryClassifierOutputTarget(1)]

    cam = cam_algo(input_tensor=image_tensor, targets=targets)
    cam = cam[0]  # (H, W) — squeeze batch dim

    raw_img = denormalize(image_tensor[0])
    overlay = show_cam_on_image(raw_img, cam, use_rgb=True)

    return cam, overlay, raw_img, score


# ── Batch Grad-CAM grid ───────────────────────────────────────────────────────

def generate_gradcam_grid(model, test_loader, cfg, device, n_samples=16):
    """
    Generate a grid of Grad-CAM visualizations for the test set.
    Saves a 4×4 panel (or n_samples panel) of:
      Original | Heatmap overlay | Prediction
    """
    target_layers = get_target_layer(model)

    samples_real, samples_fake = [], []
    target_each = n_samples // 2

    model.eval()
    for images, labels in test_loader:
        for i in range(len(images)):
            label = int(labels[i].item())
            if label == 0 and len(samples_real) < target_each:
                samples_real.append((images[i], label))
            elif label == 1 and len(samples_fake) < target_each:
                samples_fake.append((images[i], label))
        if len(samples_real) >= target_each and len(samples_fake) >= target_each:
            break

    samples = samples_real + samples_fake
    np.random.shuffle(samples)
    samples = samples[:n_samples]

    ncols = 4
    nrows = (len(samples) + ncols - 1) // ncols

    fig = plt.figure(figsize=(ncols * 5, nrows * 4))
    fig.suptitle(
        "Grad-CAM++ Visualizations — Model Attention Maps\n"
        "(Warm regions = high model focus)",
        fontsize=14, fontweight="bold", y=1.01
    )

    gradcam_dir = Path(cfg["paths"]["gradcam_dir"])
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_tensor, true_label) in enumerate(tqdm(samples, desc="  Generating Grad-CAM")):
        inp = img_tensor.unsqueeze(0).to(device)
        cam, overlay, raw_img, score = generate_gradcam(model, inp, target_layers, device.type == "cuda")

        pred_label = "Fake" if score >= cfg["evaluation"]["threshold"] else "Real"
        true_name  = "Fake" if true_label == 1 else "Real"
        correct    = pred_label == true_name

        ax = fig.add_subplot(nrows, ncols, idx + 1)

        # Stack: raw image on left, overlay on right
        combined = np.concatenate([
            (raw_img * 255).astype(np.uint8),
            overlay
        ], axis=1)
        ax.imshow(combined)

        color = "#57A773" if correct else "#E84855"
        ax.set_title(
            f"GT: {true_name}  |  Pred: {pred_label}\nScore: {score:.3f}",
            fontsize=9, color=color, fontweight="bold"
        )
        ax.axis("off")

        # Also save individual heatmap
        individual_path = gradcam_dir / f"gradcam_{idx:03d}_gt{true_name}_pred{pred_label}.jpg"
        cv2.imwrite(str(individual_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    plt.tight_layout()
    grid_path = gradcam_dir / "gradcam_grid.png"
    plt.savefig(grid_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"\n  Grad-CAM grid saved → {grid_path}")
    print(f"  Individual maps    → {gradcam_dir}/")
    return grid_path


# ── Predict single image (inference) ─────────────────────────────────────────

def predict_single_image(model, image_path, cfg, device, save_gradcam=True):
    """
    Run inference on a single image file.
    Returns prediction, score, and optionally saves Grad-CAM.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    aug = cfg["augmentation"]
    size = cfg["data"]["image_size"]
    transform = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ToTensorV2(),
    ])

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

    target_layers = get_target_layer(model)
    cam, overlay, raw_img, score = generate_gradcam(model, img_tensor, target_layers)

    threshold = cfg["evaluation"]["threshold"]
    prediction = "FAKE" if score >= threshold else "REAL"

    print(f"\n  Image    : {image_path}")
    print(f"  Score    : {score:.4f}")
    print(f"  Decision : {prediction}  (threshold={threshold})")

    if save_gradcam:
        out_dir = Path(cfg["paths"]["gradcam_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Prediction: {prediction} (score={score:.4f})", fontsize=13, fontweight="bold")

        axes[0].imshow((raw_img * 255).astype(np.uint8))
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis("off")

        im = axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Grad-CAM heatmap", fontsize=11)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].axis("off")

        plt.tight_layout()
        stem = Path(image_path).stem
        out_path = out_dir / f"predict_{stem}.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  Grad-CAM → {out_path}")

    return prediction, score, overlay


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True, help="Path to an image to analyze")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--ckpt",    default=None, help="Checkpoint path (uses best_model from config if None)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.model import build_model, load_checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg, device)
    ckpt   = args.ckpt or cfg["paths"]["best_model"]
    if Path(ckpt).exists():
        load_checkpoint(model, ckpt, device=device)

    predict_single_image(model, args.image, cfg, device)
