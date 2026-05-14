"""
model.py
────────
Defines the DeepfakeDetector model using EfficientNet-B4 backbone
with a custom binary classification head.

Architecture:
  EfficientNet-B0 (pretrained on ImageNet)
    → Global Average Pooling
    → Dropout(0.4)
    → Linear(1792 → 512)
    → ReLU + BatchNorm
    → Dropout(0.3)
    → Linear(512 → 1)
    → Sigmoid output
"""

import timm
import torch
import torch.nn as nn
import yaml
from pathlib import Path


# ── Model definition ─────────────────────────────────────────────────────────

class DeepfakeDetector(nn.Module):
    """
    Transfer-learning based deepfake image detector.

    Args:
        backbone_name: timm model identifier (default: 'efficientnet_b4')
        pretrained: load ImageNet weights
        dropout: dropout rate before final classifier
    """

    def __init__(self, backbone_name="efficientnet_b4", pretrained=True, dropout=0.4):
        super().__init__()

        # Load pretrained backbone (without its classifier head)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # remove the original classification head
            global_pool="avg"       # global average pooling
        )

        # Get the number of features coming out of the backbone
        n_features = self.backbone.num_features  # 1792 for EfficientNet-B0

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1),
        )

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)     # (B, n_features)
        out = self.classifier(features) # (B, 1)
        return out.squeeze(1)           # (B,)

    def freeze_backbone(self):
        """Freeze backbone parameters — only train the classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone frozen. Training classifier head only.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone unfrozen. Full fine-tuning enabled.")

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# ── Factory function ─────────────────────────────────────────────────────────

def build_model(cfg, device="cpu"):
    """Build model from config dict and move to device."""
    model_cfg = cfg["model"]

    model = DeepfakeDetector(
        backbone_name=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"]
    )

    model = model.to(device)

    total  = model.get_total_params()
    print(f"\n[Model] {model_cfg['backbone']} loaded")
    print(f"  Total params      : {total:,}")
    print(f"  Trainable params  : {model.get_trainable_params():,}")

    return model


# ── Save / Load helpers ──────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, val_auc, cfg, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_auc": val_auc,
        "config": cfg,
    }
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    path = ckpt_dir / f"checkpoint_epoch{epoch:03d}.pth"
    torch.save(checkpoint, path)

    if is_best:
        best_path = Path(cfg["paths"]["best_model"])
        torch.save(checkpoint, best_path)
        print(f"  ✓ Best model saved → {best_path}")

    return path


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val AUC: {checkpoint['val_auc']:.4f})")
    return checkpoint["epoch"], checkpoint["val_auc"]


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, device)

    # Forward pass test
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out = model(dummy)
    print(f"\n  Input shape  : {dummy.shape}")
    print(f"  Output shape : {out.shape}")
    print(f"  Output range : [{out.min():.3f}, {out.max():.3f}]")
    print(f"\n  Model OK ✓")
