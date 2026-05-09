"""
evaluate.py
───────────
Comprehensive evaluation module for the Deepfake Detection System.

Computes:
  - Accuracy, Precision, Recall, F1-score
  - AUC-ROC (Area Under the ROC Curve)
  - EER (Equal Error Rate) — key biometric metric
  - Confusion matrix
  - Classification report
  - ROC curve plot
  - Precision-Recall curve plot
  - Training history plots
  - Benchmark comparison table
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "real":   "#2E86AB",
    "fake":   "#E84855",
    "teal":   "#3BB5AC",
    "amber":  "#F6AE2D",
    "purple": "#7B5EA7",
    "gray":   "#6C757D",
    "green":  "#57A773",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# ── Core metric computation ───────────────────────────────────────────────────

def compute_eer(labels, scores):
    """
    Equal Error Rate — the threshold at which FAR == FRR.
    Lower EER = better detector.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # Interpolate to find exact crossing point
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0., 1.)
    return float(eer)


def compute_metrics(labels, scores, threshold=0.5):
    """
    Compute all evaluation metrics from raw scores.

    Args:
        labels:    ground-truth binary labels (0=real, 1=fake)
        scores:    model output probabilities in [0, 1]
        threshold: decision boundary

    Returns:
        dict with all metrics
    """
    labels  = np.array(labels)
    scores  = np.array(scores)
    preds   = (scores >= threshold).astype(int)

    metrics = {
        "accuracy":   float(accuracy_score(labels, preds)),
        "precision":  float(precision_score(labels, preds, zero_division=0)),
        "recall":     float(recall_score(labels, preds, zero_division=0)),
        "f1":         float(f1_score(labels, preds, zero_division=0)),
        "auc":        float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "threshold":  threshold,
    }
    # EER
    try:
        metrics["eer"] = compute_eer(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    except Exception:
        metrics["eer"] = 0.0

    return metrics


# ── Plot: ROC curve ───────────────────────────────────────────────────────────

def plot_roc_curve(labels, scores, auc, eer, save_path):
    fpr, tpr, _ = roc_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["real"], lw=2.5,
            label=f"EfficientNet-B4  (AUC = {auc:.4f}, EER = {eer:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random classifier")

    # Mark EER point
    eer_fpr = eer
    eer_tpr = 1 - eer
    ax.scatter([eer_fpr], [eer_tpr], color=PALETTE["fake"], zorder=5, s=80,
               label=f"EER point ({eer:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — Deepfake Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ROC curve saved → {save_path}")


# ── Plot: Precision-Recall curve ──────────────────────────────────────────────

def plot_pr_curve(labels, scores, save_path):
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=PALETTE["purple"], lw=2.5,
            label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.15, color=PALETTE["purple"])
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  PR curve saved  → {save_path}")


# ── Plot: Confusion matrix ────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Confusion Matrix (counts)"),
        (axes[1], cm_norm, ".2f",  "Confusion Matrix (normalized)"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, ax=ax,
            cmap="Blues", linewidths=0.5,
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            annot_kws={"size": 14}
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual",    fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Confusion matrix → {save_path}")


# ── Plot: Training curves ─────────────────────────────────────────────────────

def plot_training_curves(history, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold", y=1.01)

    metrics_map = [
        ("loss", "Loss",     "train_loss", "val_loss"),
        ("acc",  "Accuracy", "train_acc",  "val_acc"),
        ("auc",  "AUC-ROC",  "train_auc",  "val_auc"),
        ("f1",   "F1-Score", "train_f1",   "val_f1"),
    ]

    for ax, (key, label, train_key, val_key) in zip(axes.flat, metrics_map):
        ax.plot(epochs, history[train_key], color=PALETTE["real"],   lw=2, label="Train", marker="o", markersize=3)
        ax.plot(epochs, history[val_key],   color=PALETTE["amber"],  lw=2, label="Val",   marker="s", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.set_xlim([1, max(epochs)])

    plt.tight_layout()
    path = save_dir / "training_curves.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Training curves → {path}")


# ── Plot: Benchmark comparison bar chart ──────────────────────────────────────

BENCHMARKS = {
    "Face X-ray\n(2020)":           {"auc": 0.980, "accuracy": 0.954, "eer": 0.052},
    "Multi-Attentional\n(2021)":    {"auc": 0.993, "accuracy": 0.976, "eer": 0.033},
    "LipForensics\n(2021)":         {"auc": 0.997, "accuracy": None,  "eer": 0.025},
    "RECCE\n(2022)":                {"auc": 0.991, "accuracy": 0.979, "eer": 0.030},
    "UniFace\n(2023)":              {"auc": 0.994, "accuracy": 0.981, "eer": 0.022},
}


def plot_benchmark_comparison(our_metrics, save_dir):
    save_dir = Path(save_dir)

    # Add our result
    all_methods = dict(BENCHMARKS)
    all_methods["Our Model\n(EfficientNet-B4)"] = {
        "auc":      our_metrics["auc"],
        "accuracy": our_metrics["accuracy"],
        "eer":      our_metrics["eer"],
    }

    methods = list(all_methods.keys())
    colors  = [PALETTE["gray"]] * (len(methods) - 1) + [PALETTE["teal"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Benchmark Comparison on FaceForensics++", fontsize=15, fontweight="bold")

    metric_config = [
        ("auc",      "AUC-ROC",  True,  0.95),
        ("accuracy", "Accuracy", True,  0.90),
        ("eer",      "EER",      False, 0.0),
    ]

    for ax, (key, label, higher_better, y_min) in zip(axes, metric_config):
        vals = [all_methods[m].get(key) for m in methods]
        valid_vals = [v for v in vals if v is not None]
        bar_vals   = [v if v is not None else 0 for v in vals]
        bars = ax.bar(range(len(methods)), bar_vals, color=colors, edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{'Higher' if higher_better else 'Lower'} is better", fontsize=10, color=PALETTE["gray"])
        if valid_vals:
            spread = max(valid_vals) - min(valid_vals)
            ax.set_ylim([max(0, min(valid_vals) - spread * 0.5),
                         min(1.05, max(valid_vals) + spread * 0.5)])

    plt.tight_layout()
    path = save_dir / "benchmark_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Benchmark plot  → {path}")


# ── Score distribution plot ───────────────────────────────────────────────────

def plot_score_distribution(labels, scores, save_path):
    labels  = np.array(labels)
    scores  = np.array(scores)
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(real_scores, bins=50, alpha=0.7, color=PALETTE["real"],  label="Real", density=True)
    ax.hist(fake_scores, bins=50, alpha=0.7, color=PALETTE["fake"], label="Fake", density=True)
    ax.axvline(x=0.5, color="black", ls="--", lw=1.5, label="Threshold (0.5)")
    ax.set_xlabel("Predicted Probability (Fake)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Score Distribution — Real vs Fake", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Score dist.     → {save_path}")


# ── Full evaluation pipeline ──────────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(cfg, test_loader, device, checkpoint_path=None):
    from src.model import build_model, load_checkpoint

    model = build_model(cfg, device)
    ckpt_path = checkpoint_path or cfg["paths"]["best_model"]

    if Path(ckpt_path).exists():
        load_checkpoint(model, ckpt_path, device=device)
    else:
        print(f"  WARNING: No checkpoint found at {ckpt_path}. Using untrained model.")

    model.eval()
    all_scores, all_labels = [], []

    for images, labels in tqdm(test_loader, desc="  Evaluating test set", ncols=90):
        images = images.to(device, non_blocking=True)
        with autocast():
            scores = model(images)
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(labels.numpy())

    threshold = cfg["evaluation"]["threshold"]
    metrics   = compute_metrics(all_labels, all_scores, threshold)
    preds     = (np.array(all_scores) >= threshold).astype(int)

    plots_dir   = Path(cfg["paths"]["plots_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  AUC-ROC   : {metrics['auc']:.4f}")
    print(f"  EER       : {metrics['eer']:.4f}  ({metrics['eer']*100:.2f}%)")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"{'='*50}")

    print("\n  Classification Report:")
    print(classification_report(all_labels, preds, target_names=["Real", "Fake"]))

    # Save all plots
    print("\n  Saving evaluation plots...")
    plot_roc_curve(all_labels, all_scores, metrics["auc"], metrics["eer"],
                   plots_dir / "roc_curve.png")
    plot_pr_curve(all_labels, all_scores,
                  plots_dir / "pr_curve.png")
    plot_confusion_matrix(all_labels, preds,
                          plots_dir / "confusion_matrix.png")
    plot_score_distribution(all_labels, all_scores,
                            plots_dir / "score_distribution.png")
    plot_benchmark_comparison(metrics, plots_dir)

    # Save metrics JSON
    metrics_path = metrics_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved   → {metrics_path}")

    # Save benchmark CSV
    rows = []
    for method, vals in BENCHMARKS.items():
        rows.append({"Method": method.replace("\n", " "), **vals})
    rows.append({"Method": "Our Model (EfficientNet-B4)", **{
        "auc": metrics["auc"], "accuracy": metrics["accuracy"], "eer": metrics["eer"]
    }})
    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "benchmark_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Benchmark CSV   → {csv_path}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from src.dataset import build_dataloaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _ = build_dataloaders(cfg, verbose=False)
    full_evaluation(cfg, test_loader, device, checkpoint_path=args.checkpoint)
