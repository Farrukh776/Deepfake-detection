"""
api.py — FastAPI backend for Deepfake Detection
Place this file in the ROOT of your deepfake-detection/ project folder.
Run with: uvicorn api:app --reload
"""

import io
import base64
import traceback
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── pytorch-grad-cam (same library your gradcam.py uses) ─────────────────────
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

# ── Your exact model from src ─────────────────────────────────────────────────
from src.model import DeepfakeDetector

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API",
    description="EfficientNet-B0 deepfake detector with Grad-CAM++ explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[API] Using device: {DEVICE}")

# ── Transform — exactly matches your training pipeline ────────────────────────
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
])

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    """Normalized tensor → float32 numpy (H,W,3) in [0,1] for show_cam_on_image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1).astype(np.float32)

# ── Load model ────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = Path("checkpoints/best_model.pth")

def load_model():
    model = DeepfakeDetector(backbone_name="efficientnet_b0", pretrained=False)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'. "
            "Make sure best_model.pth is inside checkpoints/"
        )

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"[API] Model loaded from {CHECKPOINT_PATH}")
    return model

model = load_model()

# ── Target layer — same as src/gradcam.py get_target_layer() ─────────────────
TARGET_LAYERS = [model.backbone.blocks[-1]]

# ── Response schema ───────────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    label: str          # "FAKE" or "REAL"
    confidence: float   # fake probability 0-1
    gradcam_b64: str    # base64 JPEG of overlay


# ── Sync inference function (runs in thread executor) ─────────────────────────
def run_inference(img_np: np.ndarray, orig_w: int, orig_h: int) -> PredictResponse:
    augmented  = transform(image=img_np)
    img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logit = model(img_tensor)
        prob  = torch.sigmoid(logit).item()

    label = "FAKE" if prob >= 0.5 else "REAL"

    # Grad-CAM++ — runs in sync thread, gradients work correctly
    cam_algo = GradCAMPlusPlus(model=model, target_layers=TARGET_LAYERS)
    targets  = [BinaryClassifierOutputTarget(1)]
    cam      = cam_algo(input_tensor=img_tensor, targets=targets)[0]  # (H,W)

    raw_img  = denormalize(img_tensor[0])
    overlay  = show_cam_on_image(raw_img, cam, use_rgb=True)

    overlay_resized = cv2.resize(overlay, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR)

    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(overlay_resized, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )
    gradcam_b64 = base64.b64encode(buf).decode("utf-8")

    return PredictResponse(
        label=label,
        confidence=round(prob, 4),
        gradcam_b64=gradcam_b64,
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Deepfake Detection API is running.",
        "device": str(DEVICE),
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):

    # ── Validate ──────────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG."
        )

    try:
        raw     = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    try:
        img_np   = np.array(pil_img)
        orig_w, orig_h = pil_img.size

        # Run everything in a thread so PyTorch gradients work correctly
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_inference, img_np, orig_w, orig_h)
        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")