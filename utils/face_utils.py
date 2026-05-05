"""
utils/face_utils.py
===================
Face detection (Haar Cascade with MTCNN upgrade path),
preprocessing, and test-time augmentation.
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple, List

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE     = 224
MEAN         = (0.485, 0.456, 0.406)
STD          = (0.229, 0.224, 0.225)
MIN_FACE_PX  = 40
MARGIN = 0.35  # 35% margin around detected face box

# ── Haar Cascade (always available) ───────────────────────────────────────
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_HAAR      = cv2.CascadeClassifier(_HAAR_PATH)
assert not _HAAR.empty(), "Haar Cascade XML not found — OpenCV installation may be broken."

# ── MTCNN (optional upgrade) ───────────────────────────────────────────────
_MTCNN: Optional[object] = None
try:
    from mtcnn import MTCNN  # type: ignore
    _MTCNN = MTCNN()
except (ImportError, ModuleNotFoundError, Exception):
    pass


# ── Detectors ──────────────────────────────────────────────────────────────

def detect_faces_haar(bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """OpenCV Haar Cascade detector. Fast, always works."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    dets = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                   minSize=(MIN_FACE_PX, MIN_FACE_PX))
    if len(dets) == 0:
        return []
    return sorted([tuple(int(v) for v in d) for d in dets],
                  key=lambda b: b[2]*b[3], reverse=True)


def detect_faces_mtcnn(bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """MTCNN detector — more accurate than Haar, especially at angles."""
    if _MTCNN is None:
        return detect_faces_haar(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = _MTCNN.detect_faces(rgb)
    boxes = [tuple(r["box"]) for r in results if r["confidence"] > 0.85]
    return sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)


def detect_faces(bgr: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """Detect faces — MTCNN preferred, Haar fallback."""
    if _MTCNN is not None:
        boxes = detect_faces_mtcnn(bgr)
        if boxes:
            return boxes
    return detect_faces_haar(bgr)


# ── Crop & resize ─────────────────────────────────────────────────────────

def crop_face(bgr: np.ndarray, box: Tuple[int,int,int,int],
              margin: float = MARGIN, size: int = IMG_SIZE) -> Optional[np.ndarray]:
    """
    Crop a face region with margin from BGR image.
    Returns RGB uint8 array of shape (size, size, 3), or None on failure.
    """
    x, y, w, h = box
    if w < MIN_FACE_PX or h < MIN_FACE_PX:
        return None
    H, W = bgr.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx);  y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)


def get_primary_face(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
    """
    Detect and return the largest face crop (RGB uint8).
    Returns (crop, True) if face found, else (full_image_resized, False).
    """
    if bgr is None or bgr.size == 0:
        blank = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return blank, False

    boxes = detect_faces(bgr)
    for box in boxes:
        crop = crop_face(bgr, box)
        if crop is not None:
            return crop, True

    # Fallback: resize full image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return resized, False


def get_all_faces(bgr: np.ndarray) -> List[np.ndarray]:
    """Return all detected face crops (RGB uint8) from an image."""
    boxes = detect_faces(bgr)
    crops = []
    for box in boxes:
        c = crop_face(bgr, box)
        if c is not None:
            crops.append(c)
    return crops


# ── Tensor conversion ──────────────────────────────────────────────────────

def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 RGB HWC → float32 CHW, ImageNet normalised."""
    t    = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1)
    return (t - mean) / std


def preprocess(bgr: np.ndarray) -> Tuple[torch.Tensor, Optional[np.ndarray], bool]:
    """
    Full image → model-ready tensor.

    Returns
    -------
    tensor     : (1, 3, 224, 224)
    face_rgb   : uint8 RGB face crop shown in UI
    face_found : bool
    """
    face_rgb, found = get_primary_face(bgr)
    tensor = to_tensor(face_rgb).unsqueeze(0)   # add batch dim
    return tensor, face_rgb, found


# ── Test-Time Augmentation (TTA) ───────────────────────────────────────────

def tta_tensors(rgb: np.ndarray, n: int = 5) -> List[torch.Tensor]:
    """
    Generate n augmented versions for TTA.
    Averaging their logits reduces prediction variance.
    """
    H, W = rgb.shape[:2]
    variants = [rgb, cv2.flip(rgb, 1)]   # original + h-flip

    # Slight crops from different quadrants
    offsets = [(int(W*0.04), 0), (0, int(H*0.04)), (int(W*0.03), int(H*0.03))]
    for ox, oy in offsets[:max(0, n - 2)]:
        crop = rgb[oy:oy + int(H*0.96), ox:ox + int(W*0.96)]
        variants.append(cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4))

    return [to_tensor(v).unsqueeze(0) for v in variants[:n]]


# ── Video helpers ──────────────────────────────────────────────────────────

def load_image_bgr(path: str) -> Optional[np.ndarray]:
    """Load image from disk as BGR. Returns None on failure."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"[face_utils] Cannot read image: {path}")
    return img


def draw_bbox(bgr: np.ndarray, boxes: list, color=(0, 255, 0)) -> np.ndarray:
    """Draw face bounding boxes on a BGR image copy."""
    out = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
    return out
