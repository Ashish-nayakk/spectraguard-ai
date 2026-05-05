"""
utils/video_utils.py
====================
Video frame extraction, per-frame prediction, and aggregation logic.
Handles MP4, AVI, MOV, MKV, WebM.
"""

import cv2
import numpy as np
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

log = logging.getLogger("video_utils")

# ── Thresholds ────────────────────────────────────────────────────────────
FAKE_THRESH = 0.65
REAL_THRESH = 0.35


# ── Frame extraction ──────────────────────────────────────────────────────

def get_metadata(path: str) -> Dict:
    """Return basic video metadata."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": f"Cannot open {path}"}
    meta = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    meta["duration_sec"] = meta["total_frames"] / meta["fps"]
    cap.release()
    return meta


def extract_frames(
    path: str,
    interval: int = 10,
    max_frames: int = 100,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, bgr_frame) every `interval` frames.
    Stops after `max_frames` have been yielded.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.error(f"Cannot open video: {path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    log.info(f"Video: {Path(path).name} | {total} frames @ {fps:.1f}fps | "
             f"interval={interval} max_frames={max_frames}")

    idx = 0; yielded = 0
    while yielded < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            yield idx, frame
            yielded += 1
        idx += 1
    cap.release()
    log.info(f"Extracted {yielded} frames.")


# ── Aggregation ───────────────────────────────────────────────────────────

class VideoAggregator:
    """Aggregate per-frame FAKE probabilities into a single verdict."""

    MIN_FRAMES = 3

    def __init__(self, method: str = "average"):
        assert method in {"average", "majority"}
        self.method = method

    def aggregate(self, probs: List[float]) -> Dict:
        n = len(probs)
        if n == 0:
            return self._empty()

        arr         = np.array(probs, dtype=np.float32)
        fake_mask   = arr >= FAKE_THRESH
        real_mask   = arr <= REAL_THRESH
        unc_mask    = ~fake_mask & ~real_mask

        if self.method == "average":
            score = float(arr.mean())
        else:
            score = float(fake_mask.sum()) / n

        # Label
        if n < self.MIN_FRAMES:
            label = "UNCERTAIN"
        elif score >= FAKE_THRESH:
            label = "FAKE"
        elif score <= REAL_THRESH:
            label = "REAL"
        else:
            label = "UNCERTAIN"

        # Confidence: how far from the decision boundary
        if label == "FAKE":
            conf = float(score)
        elif label == "REAL":
            conf = float(1 - score)
        else:
            conf = float(1 - abs(score - 0.5) * 2)

        return {
            "label":            label,
            "confidence":       round(conf, 4),
            "fake_probability": round(score, 4),
            "method":           self.method,
            "num_frames":       n,
            "fake_frames":      int(fake_mask.sum()),
            "real_frames":      int(real_mask.sum()),
            "uncertain_frames": int(unc_mask.sum()),
            "frame_probs":      arr.tolist(),
            "prob_mean":        round(float(arr.mean()),  4),
            "prob_std":         round(float(arr.std()),   4),
            "prob_max":         round(float(arr.max()),   4),
            "prob_min":         round(float(arr.min()),   4),
        }

    @staticmethod
    def smooth(probs: List[float], window: int = 5) -> List[float]:
        """Moving-average smoothing to reduce frame-level noise."""
        if len(probs) < window:
            return probs
        arr = np.array(probs, dtype=np.float32)
        k = np.ones(window) / window
        s = np.convolve(arr, k, mode="same")
        h = window // 2
        s[:h] = arr[:h]; s[-h:] = arr[-h:]
        return s.tolist()

    def _empty(self) -> Dict:
        return {
            "label": "UNCERTAIN", "confidence": 0.0,
            "fake_probability": 0.5, "method": self.method,
            "num_frames": 0, "fake_frames": 0, "real_frames": 0,
            "uncertain_frames": 0, "frame_probs": [],
            "prob_mean": 0.5, "prob_std": 0.0, "prob_max": 0.5, "prob_min": 0.5,
        }


# ── Full pipeline ──────────────────────────────────────────────────────────

def analyze_video(
    path:        str,
    predict_fn:  Callable[[np.ndarray], float],
    interval:    int = 10,
    max_frames:  int = 80,
    method:      str = "average",
    smooth:      bool = True,
) -> Dict:
    """
    Full video analysis pipeline.

    Parameters
    ----------
    predict_fn : receives a BGR frame (np.ndarray), returns float P(fake).
    """
    probs   = []
    indices = []

    for idx, frame in extract_frames(path, interval, max_frames):
        try:
            p = predict_fn(frame)
            probs.append(float(p))
            indices.append(idx)
        except Exception as e:
            log.warning(f"Frame {idx} prediction failed: {e}")

    agg = VideoAggregator(method=method)
    if smooth and len(probs) >= 5:
        probs = VideoAggregator.smooth(probs)

    result = agg.aggregate(probs)
    result["frame_indices"] = indices
    result["video_metadata"] = get_metadata(path)
    return result
