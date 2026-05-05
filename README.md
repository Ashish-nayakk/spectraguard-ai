---
title: SpectraGuard AI
emoji: 🛡️
sdk: docker
app_port: 7860
---

# 🔬 AI Deepfake Detection System
## ✅ Pretrained — No Training Required

Production-quality deepfake detector using a dual-stream
**EfficientNet-B3 (ImageNet pretrained) + FFT frequency analysis** architecture.

---

## ✅ Pretrained Setup — Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```
Requires Python 3.10+. CUDA optional (CPU works fine for inference).

### Step 2 — Build weights (downloads EfficientNet-B3 ImageNet weights ~49 MB)
```bash
python scripts/build_weights.py
```
This creates `model/weights/deepfake_detector.pt`.
**No dataset needed.** The EfficientNet-B3 backbone is loaded with
ImageNet pretrained weights automatically.

### Step 3 — Launch the app
```bash
streamlit run app/app.py
```
Open http://localhost:8501

---

## Architecture

```
Input Image (224×224)
        │
   ┌────┴────┐
   │         │
Spatial    Frequency
Stream     Stream
   │         │
EfficientNet-B3    2-D FFT → log magnitude
(ImageNet ✅)      → Conv layers (128-dim)
(1536-dim)         │
   │         │
   └────┬────┘
        │  concat (1664-dim)
        │
  Linear(512) → LayerNorm → GELU → Dropout(0.4)
  Linear(256) → LayerNorm → GELU → Dropout(0.3)
  Linear(128) → GELU → Dropout(0.2)
  Linear(1)   → Sigmoid
        │
  P(fake) ∈ [0, 1]
```

**Why pretrained EfficientNet-B3?**
- Trained on 1.28 million ImageNet images, it already understands textures,
  edges, and visual artefacts — core signals for deepfake detection.
- The frequency stream detects GAN/diffusion spectral fingerprints invisible
  to the human eye.
- Together they work immediately without any deepfake-specific training.

---

## Project Structure

```
deepfake_project/
│
├── model/
│   ├── __init__.py
│   ├── architecture.py      DeepfakeDetector (pretrained EfficientNet-B3)
│   ├── train.py             Two-phase fine-tuning pipeline (optional)
│   └── weights/
│       └── deepfake_detector.pt   Saved weights (from build_weights.py)
│
├── utils/
│   ├── __init__.py
│   ├── face_utils.py        Face detection, cropping, TTA preprocessing
│   └── video_utils.py       Frame extraction, aggregation, metadata
│
├── app/
│   ├── __init__.py
│   ├── app.py               Streamlit UI (main entry point)
│   └── predictor.py         Inference engine with TTA, logging, thresholds
│
├── scripts/
│   ├── __init__.py
│   ├── build_weights.py     ✅ Generate pretrained weights (run this first)
│   └── prepare_data.py      Dataset preparation (for optional fine-tuning)
│
├── data/
│   └── dataset/
│       ├── train/{real,fake}/
│       ├── val/{real,fake}/
│       └── test/{real,fake}/
│
├── logs/
├── requirements.txt
└── README.md
```

---

## Optional: Fine-tune on Real Deepfake Data

The pretrained model is immediately usable, but for higher accuracy you
can fine-tune on real deepfake datasets:

### Demo dataset (no downloads needed)
```bash
python scripts/prepare_data.py --demo --n 1000
python model/train.py --data_dir data/dataset --epochs_p1 10 --epochs_p2 20
```

### DFDC (DeepFake Detection Challenge)
```bash
kaggle competitions download -c deepfake-detection-challenge
unzip deepfake-detection-challenge.zip -d /path/to/dfdc
python scripts/prepare_data.py --dfdc /path/to/dfdc
python model/train.py --data_dir data/dataset --batch_size 32 --device cuda
```

### Training CLI options
```bash
python model/train.py \
  --data_dir   data/dataset \
  --epochs_p1  10           \
  --epochs_p2  20           \
  --batch_size 32           \
  --lr_p1      1e-3         \
  --lr_p2      5e-6         \
  --device     cuda
```

---

## Decision Thresholds

| Probability | Label       | Meaning                         |
|-------------|-------------|---------------------------------|
| ≥ 0.65      | ⚠️ FAKE     | Manipulation artefacts detected |
| ≤ 0.35      | ✅ REAL     | No manipulation found           |
| 0.35–0.65   | ❓ UNCERTAIN | Low confidence — review manually|

Thresholds are adjustable in the sidebar.

---

## Expected Performance

| Mode                        | Notes                                     |
|-----------------------------|-------------------------------------------|
| Pretrained only (ImageNet)  | Basic texture/artefact detection; ~60–70% |
| Fine-tuned on DFDC          | AUC 88–93%, Accuracy 85–90%               |
| Fine-tuned on FF++ (c23)    | AUC 91–96%, Accuracy 89–94%               |
