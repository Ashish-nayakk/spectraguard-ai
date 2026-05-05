"""
model/train.py
==============
Two-phase transfer learning training pipeline.

Phase 1 — Frozen backbone (EfficientNet-B3): train head + freq stream
Phase 2 — Partial unfreeze (blocks 5-8): end-to-end fine-tuning

Usage
-----
    python model/train.py --data_dir data/dataset --epochs_p1 10 --epochs_p2 20
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.architecture import DeepfakeDetector, load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE     = 224
MEAN         = (0.485, 0.456, 0.406)
STD          = (0.229, 0.224, 0.225)
WEIGHTS_DIR  = ROOT / "model" / "weights"
LOGS_DIR     = ROOT / "logs"


# ── Dataset ────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    """
    Expects directory layout:
        root/
            real/   *.jpg | *.png | *.jpeg
            fake/   *.jpg | *.png | *.jpeg
    """

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def __init__(self, root: str, augment: bool = False):
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []
        root = Path(root)
        for label, idx in [("real", 0), ("fake", 1)]:
            d = root / label
            if not d.exists():
                log.warning(f"Label dir not found: {d}")
                continue
            for p in d.iterdir():
                if p.suffix.lower() in self.EXTS:
                    self.samples.append((p, idx))

        if not self.samples:
            raise ValueError(f"No images found under {root}. "
                             f"Create subfolders 'real/' and 'fake/'.")

        real_n = sum(1 for _, l in self.samples if l == 0)
        fake_n = sum(1 for _, l in self.samples if l == 1)
        log.info(f"Dataset {root.name}: {real_n} real | {fake_n} fake")

    # ------------------------------------------------------------------

    def _load_face(self, path: Path) -> np.ndarray:
        """Load image → detect face → return RGB crop (224×224)."""
        img = cv2.imread(str(path))
        if img is None:
            return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

        if len(dets) > 0:
            x, y, w, h = sorted(dets, key=lambda d: d[2]*d[3], reverse=True)[0]
            mx = int(w * 0.20); my = int(h * 0.20)
            H, W = img.shape[:2]
            x1, y1 = max(0, x-mx), max(0, y-my)
            x2, y2 = min(W, x+w+mx), min(H, y+h+my)
            img = img[y1:y2, x1:x2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Conservative augmentations that don't destroy manipulation artefacts."""
        if np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
        # Brightness / contrast
        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.80, 1.20)
            beta  = np.random.uniform(-15, 15)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # JPEG compression simulation
        if np.random.rand() < 0.30:
            q = int(np.random.uniform(60, 95))
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, q])
            img = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
        # Slight Gaussian blur
        if np.random.rand() < 0.20:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std  = torch.tensor(STD).view(3, 1, 1)
        return (t - mean) / std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = self._load_face(path)
        if self.augment:
            img = self._augment(img)
        return self._to_tensor(img), torch.tensor(label, dtype=torch.float32)

    def class_weights(self) -> torch.Tensor:
        from sklearn.utils.class_weight import compute_class_weight
        labels = [l for _, l in self.samples]
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
        return torch.tensor(w, dtype=torch.float32)


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, threshold: float = 0.5):
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        X = X.to(device)
        probs = model(X).squeeze(1).cpu()
        all_probs.extend(probs.tolist())
        all_labels.extend(y.tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    f1  = f1_score(labels, preds, zero_division=0)
    return acc, auc, f1, probs, labels


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_history(history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, key in zip(axes, ["loss", "auc", "acc"]):
        tr = history.get(f"train_{key}", [])
        va = history.get(f"val_{key}", [])
        ax.plot(tr, label="train", lw=2)
        ax.plot(va, label="val",   lw=2, linestyle="--")
        ax.set_title(key.upper()); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Training History", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(labels, preds, out_dir: Path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


# ── Training loop ──────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_probs, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            if train:
                optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(X)
            preds = (out.detach() >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total_n += len(X)
            all_probs.extend(out.detach().cpu().squeeze(1).tolist())
            all_labels.extend(y.cpu().squeeze(1).tolist())

    avg_loss = total_loss / max(total_n, 1)
    avg_acc  = total_correct / max(total_n, 1)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    return avg_loss, avg_acc, auc


def train(
    data_dir:     str,
    epochs_p1:    int = 10,
    epochs_p2:    int = 20,
    batch_size:   int = 32,
    lr_p1:        float = 1e-3,
    lr_p2:        float = 5e-6,
    device:       str = "cpu",
    num_workers:  int = 0,
):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)
    log.info("=" * 60)
    log.info("  DEEPFAKE DETECTION — TRAINING")
    log.info("=" * 60)

    # ── Datasets ──
    train_ds = FaceDataset(str(data_dir / "train"), augment=True)
    val_ds   = FaceDataset(str(data_dir / "val"),   augment=False)
    test_ds  = FaceDataset(str(data_dir / "test"),  augment=False)

    cw = train_ds.class_weights().to(device)
    log.info(f"Class weights: real={cw[0]:.3f}  fake={cw[1]:.3f}")

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # ── Model ──
    model = DeepfakeDetector().to(device)
    log.info(f"Total params: {model.total_params():,}")

    criterion = nn.BCELoss()
    history   = {k: [] for k in ["train_loss","val_loss","train_auc","val_auc","train_acc","val_acc"]}

    best_auc    = 0.0
    best_path   = WEIGHTS_DIR / "best_model.pt"
    final_path  = WEIGHTS_DIR / "deepfake_detector.pt"

    # ──────────────────────────────────────────────────────────────────
    # PHASE 1 — frozen backbone
    # ──────────────────────────────────────────────────────────────────
    log.info(f"\n── PHASE 1: Frozen backbone | LR={lr_p1} | Epochs={epochs_p1} ──")
    model.freeze_backbone()
    log.info(f"  Trainable: {model.trainable_params():,}")

    opt1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_p1, weight_decay=1e-4
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs_p1, eta_min=1e-5)

    for ep in range(1, epochs_p1 + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, opt1, criterion, device, True)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader,   opt1, criterion, device, False)
        sched1.step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_auc"].append(tr_auc);   history["val_auc"].append(va_auc)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), str(best_path))

        log.info(
            f"  P1 Ep {ep:02d}/{epochs_p1} | "
            f"loss {tr_loss:.4f}/{va_loss:.4f} | "
            f"auc {tr_auc:.4f}/{va_auc:.4f} | "
            f"acc {tr_acc:.3f}/{va_acc:.3f} | "
            f"{time.time()-t0:.1f}s"
        )

    # ──────────────────────────────────────────────────────────────────
    # PHASE 2 — partial unfreeze
    # ──────────────────────────────────────────────────────────────────
    log.info(f"\n── PHASE 2: Unfreeze blocks 5-8 | LR={lr_p2} | Epochs={epochs_p2} ──")
    model.unfreeze_backbone(from_block=5)
    log.info(f"  Trainable: {model.trainable_params():,}")

    opt2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_p2, weight_decay=1e-5
    )
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_p2, eta_min=1e-7)
    patience = 7; no_improve = 0

    for ep in range(1, epochs_p2 + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, opt2, criterion, device, True)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader,   opt2, criterion, device, False)
        sched2.step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_auc"].append(tr_auc);   history["val_auc"].append(va_auc)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)

        if va_auc > best_auc:
            best_auc = va_auc; no_improve = 0
            torch.save(model.state_dict(), str(best_path))
            log.info(f"  ✓ New best AUC: {best_auc:.4f} — checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {ep}.")
                break

        log.info(
            f"  P2 Ep {ep:02d}/{epochs_p2} | "
            f"loss {tr_loss:.4f}/{va_loss:.4f} | "
            f"auc {tr_auc:.4f}/{va_auc:.4f} | "
            f"acc {tr_acc:.3f}/{va_acc:.3f} | "
            f"{time.time()-t0:.1f}s"
        )

    # ── Load best and save final ──
    model.load_state_dict(torch.load(str(best_path), map_location=device, weights_only=True))
    torch.save(model.state_dict(), str(final_path))
    log.info(f"\nFinal model saved → {final_path}")

    # ── Test evaluation ──
    log.info("\n── TEST SET EVALUATION ──")
    _, _, _, probs, labels = evaluate(model, test_loader, device)
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    log.info(f"\n{classification_report(labels, preds, target_names=['REAL','FAKE'], digits=4)}")
    log.info(f"ROC-AUC: {auc:.4f}")

    # ── Save plots + metrics ──
    plot_history(history, LOGS_DIR)
    plot_confusion(labels, preds, LOGS_DIR)

    metrics = {
        "best_val_auc": float(best_auc),
        "test_auc":     float(auc),
        "test_acc":     float(accuracy_score(labels, preds)),
        "test_f1":      float(f1_score(labels, preds, zero_division=0)),
    }
    with open(LOGS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"\nMetrics: {metrics}")
    log.info("Training complete.")
    return model


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/dataset")
    p.add_argument("--epochs_p1",  type=int,   default=10)
    p.add_argument("--epochs_p2",  type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr_p1",      type=float, default=1e-3)
    p.add_argument("--lr_p2",      type=float, default=5e-6)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers",    type=int,   default=4)
    args = p.parse_args()
    train(
        data_dir=args.data_dir, epochs_p1=args.epochs_p1,
        epochs_p2=args.epochs_p2, batch_size=args.batch_size,
        lr_p1=args.lr_p1, lr_p2=args.lr_p2,
        device=args.device, num_workers=args.workers,
    )
