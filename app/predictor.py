"""
app/predictor.py
================
Inference engine for image and video deepfake detection.
Handles TTA, safe thresholding, face extraction, and prediction logging.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.architecture import load_model, DeepfakeDetector
from utils.face_utils    import preprocess, tta_tensors, get_primary_face, load_image_bgr
from utils.video_utils   import analyze_video

log = logging.getLogger("predictor")

# ── Thresholds (these can be changed at runtime from Flask) ──
FAKE_THRESH = 0.70   # Default
REAL_THRESH = 0.40

# ── Paths ──
LOG_FILE = ROOT / "logs" / "predictions.jsonl"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Device ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Singleton model ──
_model: Optional[DeepfakeDetector] = None

def get_model() -> Optional[DeepfakeDetector]:
    global _model
    if _model is None:
        try:
            _model = load_model(device=DEVICE)
        except Exception as e:
            log.error(f"Model load failed: {e}")
    return _model

# ── Core inference ──
@torch.no_grad()
def _predict_tensor(tensor: torch.Tensor) -> float:
    model = get_model()
    if model is None:
        return 0.5
    tensor = tensor.to(DEVICE)
    return float(model(tensor).item())

@torch.no_grad()
def _predict_with_tta(rgb: np.ndarray, n_tta: int = 5) -> float:
    model = get_model()
    if model is None:
        return 0.5
    tensors = tta_tensors(rgb, n=n_tta)
    probs = []
    for t in tensors:
        t = t.to(DEVICE)
        probs.append(float(model(t).item()))
    return float(np.mean(probs))

# ── Label helpers (now using dynamic thresholds) ──
def _label(prob: float) -> str:
    if prob >= FAKE_THRESH:
        return "FAKE"
    if prob <= REAL_THRESH:
        return "REAL"
    return "UNCERTAIN"

def _confidence(prob: float, label: str) -> float:
    if label == "FAKE":
        return round(prob, 4)
    if label == "REAL":
        return round(1.0 - prob, 4)
    # UNCERTAIN: distance from 0.5 (higher = more certain within uncertain zone)
    return round(1.0 - abs(prob - 0.5) * 2, 4)

def _explanation(label: str, conf: float, has_face: bool) -> str:
    if not has_face:
        return (
            "⚠️ No face was detected. Analysis was performed on the full image "
            "and may be less reliable than a face-cropped prediction."
        )
    pct = f"{conf*100:.1f}%"
    if label == "FAKE":
        return (
            f"High likelihood of AI manipulation detected ({pct} confidence). "
            "The model identified artefacts consistent with face-swapping, GAN synthesis, "
            "or diffusion-based generation."
        )
    if label == "REAL":
        return (
            f"No significant manipulation artefacts detected ({pct} confidence). "
            "The facial texture, spectral signature, and spatial features are consistent with authentic photography."
        )
    return (
        f"The model is uncertain (probability between {REAL_THRESH*100:.0f}%–{FAKE_THRESH*100:.0f}%). "
        "This can occur with low-quality images, heavy compression, unusual lighting, "
        "or manipulation methods not well-represented in training data. "
        "Consider re-capturing with better quality."
    )

# ── Prediction log ──
def _log(record: dict):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass

# ── Public API ──
def predict_image(
    bgr: np.ndarray,
    use_tta: bool = True,
    source: str = "unknown",
) -> Dict:
    if bgr is None or bgr.size == 0:
        return _error_result("Empty or invalid image.")

    tensor, face_rgb, has_face = preprocess(bgr)
    # Brightness normalization (optional)
    if face_rgb is not None:
        face_rgb = cv2.convertScaleAbs(face_rgb, alpha=1.2, beta=10)

    if use_tta:
        prob = _predict_with_tta(face_rgb, n_tta=5)
    else:
        prob = _predict_tensor(tensor)

    prob = float(prob)

    # Stabilisation (slight shift away from 0.5)
    if prob > 0.5:
        prob = min(prob + 0.05, 0.95)
    else:
        prob = max(prob - 0.05, 0.05)

    label = _label(prob)
    conf  = _confidence(prob, label)
    expl  = _explanation(label, conf, has_face)

    result = {
        "label":            label,
        "confidence":       conf,
        "fake_probability": round(prob, 4),
        "explanation":      expl,
        "has_face":         has_face,
        "face_rgb":         face_rgb,   # will be removed in Flask JSON response
    }

    _log({
        "ts": datetime.utcnow().isoformat(), "source": source,
        "label": label, "conf": conf, "prob": round(prob, 4), "face": has_face,
    })
    return result

def predict_image_path(path: str, use_tta: bool = True) -> Dict:
    bgr = load_image_bgr(path)
    if bgr is None:
        return _error_result(f"Cannot read image: {path}")
    return predict_image(bgr, use_tta=use_tta, source=Path(path).name)

def predict_webcam_frame(bgr: np.ndarray) -> Dict:
    # Multi‑frame stabilization for webcam
    probs = []
    for k in range(5):
        h, w = bgr.shape[:2]
        dx = int((k-2) * 0.01 * w)
        dy = int((k-2) * 0.01 * h)
        x1, y1 = max(0, dx), max(0, dy)
        x2, y2 = min(w, w+dx), min(h, h+dy)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            crop = bgr
        res = predict_image(crop, use_tta=False, source="webcam")
        probs.append(res["fake_probability"])
    prob = sum(probs) / len(probs)
    if prob > 0.5:
        prob = min(prob + 0.05, 0.95)
    else:
        prob = max(prob - 0.05, 0.05)
    label = _label(prob)
    conf  = _confidence(prob, label)
    expl  = _explanation(label, conf, True)
    return {
        "label": label,
        "confidence": conf,
        "fake_probability": round(prob, 4),
        "explanation": expl,
        "has_face": True,
        "face_rgb": None,
    }

def _frame_pred(bgr: np.ndarray, use_tta: bool = True) -> float:
    _, face_rgb, _ = preprocess(bgr)
    if face_rgb is None:
        return 0.5
    probs = []
    for _ in range(3):
        if use_tta:
            p = _predict_with_tta(face_rgb, n_tta=3)
        else:
            tensor = torch.from_numpy(face_rgb).permute(2,0,1).float().div(255)
            mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
            std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            tensor = (tensor - mean) / std
            tensor = tensor.unsqueeze(0)
            p = _predict_tensor(tensor)
        probs.append(p)
    prob = sum(probs) / len(probs)
    if prob > 0.5:
        prob = min(prob + 0.05, 0.95)
    else:
        prob = max(prob - 0.05, 0.05)
    return prob

def predict_batch(paths: List[str], use_tta: bool = False) -> List[Dict]:
    return [predict_image_path(p, use_tta=use_tta) for p in paths]

def _error_result(msg: str) -> Dict:
    return {
        "label": "ERROR", "confidence": 0.0,
        "fake_probability": 0.5, "explanation": msg,
        "has_face": False, "face_rgb": None,
    }