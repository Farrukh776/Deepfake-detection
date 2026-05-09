"""
inference.py
────────────
Run inference on new images using a trained DeepfakeDetector.

Usage:
  # Single image
  python src/inference.py --image path/to/face.jpg

  # Folder of images
  python src/inference.py --folder path/to/images/

  # With Grad-CAM
  python src/inference.py --image face.jpg --gradcam

  # Custom threshold
  python src/inference.py --image face.jpg --threshold 0.6
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model, load_checkpoint
from src.gradcam import predict_single_image


# ── Transform ─────────────────────────────────────────────────────────────────

def build_inference_transform(cfg):
    aug  = cfg["augmentation"]
    size = cfg["data"]["image_size"]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=aug["normalize_mean"], std=aug["normalize_std"]),
        ToTensorV2(),
    ])


# ── Predict one image ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_image(model, image_path, transform, device, threshold=0.5):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, "ERROR"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor  = transform(image=img_rgb)["image"].unsqueeze(0).to(device)

    score      = model(tensor).item()
    prediction = "FAKE" if score >= threshold else "REAL"
    confidence = score if score >= threshold else 1 - score

    return score, confidence, prediction


# ── Batch inference ───────────────────────────────────────────────────────────

def run_batch_inference(model, folder_path, transform, device, threshold=0.5, extensions=(".jpg", ".jpeg", ".png")):
    folder = Path(folder_path)
    images = []
    for ext in extensions:
        images.extend(folder.rglob(f"*{ext}"))
    images = sorted(images)

    if not images:
        print(f"No images found in {folder_path}")
        return []

    results = []
    for img_path in tqdm(images, desc="Running inference", ncols=80):
        score, conf, pred = predict_image(model, img_path, transform, device, threshold)
        results.append({
            "image":      str(img_path),
            "score":      round(score, 4) if score is not None else None,
            "confidence": round(conf,  4) if conf  is not None else None,
            "prediction": pred,
        })
        status = "✓" if pred != "ERROR" else "✗"
        tqdm.write(f"  {status} {img_path.name:<40} → {pred}  (score={score:.3f})" if score else
                   f"  ✗ {img_path.name} → ERROR")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--image",     type=str,   default=None,          help="Single image path")
    parser.add_argument("--folder",    type=str,   default=None,          help="Folder of images")
    parser.add_argument("--config",    type=str,   default="config.yaml", help="Config file")
    parser.add_argument("--ckpt",      type=str,   default=None,          help="Checkpoint path")
    parser.add_argument("--threshold", type=float, default=None,          help="Decision threshold")
    parser.add_argument("--gradcam",   action="store_true",               help="Generate Grad-CAM for single image")
    parser.add_argument("--output",    type=str,   default="results/inference_results.csv", help="CSV output path")
    args = parser.parse_args()

    if args.image is None and args.folder is None:
        parser.error("Provide --image or --folder")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = args.threshold or cfg["evaluation"]["threshold"]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(cfg, device)
    ckpt  = args.ckpt or cfg["paths"]["best_model"]
    if Path(ckpt).exists():
        load_checkpoint(model, ckpt, device=device)
    else:
        print(f"WARNING: checkpoint not found at {ckpt}")

    model.eval()
    transform = build_inference_transform(cfg)

    # ── Single image ──────────────────────────────────────────────────────────
    if args.image:
        if args.gradcam:
            from src.gradcam import predict_single_image
            prediction, score, _ = predict_single_image(model, args.image, cfg, device, save_gradcam=True)
        else:
            score, conf, prediction = predict_image(model, args.image, transform, device, threshold)
            print(f"\n  Image      : {args.image}")
            print(f"  Score      : {score:.4f}")
            print(f"  Confidence : {conf:.4f}")
            print(f"  Decision   : {prediction}  (threshold={threshold})")

    # ── Batch folder ──────────────────────────────────────────────────────────
    elif args.folder:
        print(f"\nRunning batch inference on: {args.folder}")
        results = run_batch_inference(model, args.folder, transform, device, threshold)

        if results:
            df = pd.DataFrame(results)
            real_count = (df["prediction"] == "REAL").sum()
            fake_count = (df["prediction"] == "FAKE").sum()

            print(f"\n{'='*50}")
            print(f"  Total images : {len(results)}")
            print(f"  REAL         : {real_count} ({100*real_count/len(results):.1f}%)")
            print(f"  FAKE         : {fake_count} ({100*fake_count/len(results):.1f}%)")
            print(f"{'='*50}")

            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
