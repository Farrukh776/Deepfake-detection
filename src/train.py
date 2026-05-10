"""
train.py
────────
Full training pipeline for the Deepfake Detection System.

Features:
  - Mixed precision training (AMP) for faster GPU training
  - Backbone freeze/unfreeze schedule
  - Early stopping
  - Best model checkpointing (by val AUC)
  - TensorBoard logging
  - Loss curve + metric plots saved automatically
"""

import os
import sys
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import build_dataloaders, load_config
from src.model import build_model, save_checkpoint
from src.evaluate import compute_metrics, plot_training_curves


# ── Loss function ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss using BCEWithLogitsLoss — AMP safe."""
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ── One epoch ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Training", leave=False, ncols=90)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        total_loss += loss.item()
        all_preds.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader)
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Validating", leave=False, ncols=90):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        total_loss += loss.item()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=5, mode="max", delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True  # is_best

        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


# ── Main training loop ───────────────────────────────────────────────────────

def train(config_path="config.yaml"):
    cfg = load_config(config_path)

    # ── Setup ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Deepfake Detection — Training")
    print(f"{'='*60}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # Create output directories
    for key in ["checkpoint_dir", "plots_dir", "metrics_dir", "logs_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)

    # ── Model ──────────────────────────────────────────────
    model = build_model(cfg, device)

    # Freeze backbone initially
    freeze_epochs = cfg["model"]["freeze_epochs"]
    if freeze_epochs > 0:
        model.freeze_backbone()

    # ── Loss, optimizer, scheduler ─────────────────────────
    criterion = FocalLoss(alpha=0.8, gamma=2.0)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    total_epochs = cfg["training"]["epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - freeze_epochs, eta_min=1e-6
    )

    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"] and device.type == "cuda")
    early_stopping = EarlyStopping(patience=cfg["training"]["early_stopping_patience"], mode="max")

    # ── TensorBoard ────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg["paths"]["logs_dir"])

    # ── History ────────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
        "train_auc": [],  "val_auc": [],
        "train_f1": [],   "val_f1": [],
    }

    print(f"\n  Starting training for {total_epochs} epochs...")
    print(f"  Freeze schedule: backbone frozen for first {freeze_epochs} epochs\n")

    best_val_auc = 0.0
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        print(f"Epoch [{epoch:3d}/{total_epochs}]")

        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            # Reinitialize optimizer with lower LR for backbone
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": cfg["training"]["learning_rate"] * 0.1},
                {"params": model.classifier.parameters(), "lr": cfg["training"]["learning_rate"]}
            ], weight_decay=cfg["training"]["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - epoch, eta_min=1e-6
            )

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            cfg["training"]["grad_clip"]
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Log metrics
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Train → Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"AUC: {train_metrics['auc']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   → Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | LR: {lr:.2e}")

        # TensorBoard
        writer.add_scalars("Loss",     {"train": train_metrics["loss"],     "val": val_metrics["loss"]},     epoch)
        writer.add_scalars("Accuracy", {"train": train_metrics["accuracy"], "val": val_metrics["accuracy"]}, epoch)
        writer.add_scalars("AUC",      {"train": train_metrics["auc"],      "val": val_metrics["auc"]},      epoch)
        writer.add_scalars("F1",       {"train": train_metrics["f1"],       "val": val_metrics["f1"]},       epoch)
        writer.add_scalar("LR", lr, epoch)

        # Update history
        for key, m in [("train", train_metrics), ("val", val_metrics)]:
            history[f"{key}_loss"].append(m["loss"])
            history[f"{key}_acc"].append(m["accuracy"])
            history[f"{key}_auc"].append(m["auc"])
            history[f"{key}_f1"].append(m["f1"])

        # Early stopping & checkpointing
        val_auc = val_metrics["auc"]
        is_best = early_stopping(val_auc)
        save_checkpoint(model, optimizer, scheduler, epoch, val_auc, cfg, is_best)

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        if early_stopping.stop:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

        print()

    elapsed = time.time() - start_time
    writer.close()

    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Best Val AUC : {best_val_auc:.4f}")
    print(f"  Best model   : {cfg['paths']['best_model']}")
    print(f"{'='*60}")

    # Save training curves
    plot_training_curves(history, save_dir=cfg["paths"]["plots_dir"])
    print(f"\n  Training curves saved to {cfg['paths']['plots_dir']}/")

    # Import and run full evaluation on test set
    from src.evaluate import full_evaluation
    print("\n[Evaluation] Running on test set...")
    full_evaluation(cfg, test_loader, device)

    return history, best_val_auc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    train(args.config)
