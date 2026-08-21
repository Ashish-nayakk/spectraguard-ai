<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=SpectraGuard+AI" alt="SpectraGuard AI Title" />

# 🛡️ SpectraGuard AI

### *Dual-Stream Deepfake Detection System for Financial Security*

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20Now-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=white)](https://ashish-kumar-nayak-spectraguard-ai.hf.space/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/Ashish-nayakk/spectraguard-ai)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)

## 🚀 Try It Live Now!

### [👉 Live Demo 👈](https://ashish-kumar-nayak-spectraguard-ai.hf.space/)

</div>

---

## 📋 Overview

**SpectraGuard AI** is a production-oriented deepfake detection system designed to help prevent **AI-enabled KYC fraud** in financial onboarding.

It uses a **dual-stream deep learning architecture** that analyzes both spatial and frequency-domain information:

* 🖼️ **Spatial Stream** → EfficientNet-B3 pretrained on ImageNet to detect visual artifacts, textures, and facial inconsistencies.
* 📡 **Frequency Stream** → 2D Fast Fourier Transform (FFT) analysis to identify spectral fingerprints that may not be visible to the human eye.
* 🛡️ **Risk Decision Layer** → Converts model probability into **REAL, UNCERTAIN, or FAKE** risk classifications.
* 📊 **Explainability & Auditability** → Confidence scores, threshold-based decisions, and logging support human review.

> ⚡ **Core idea:** SpectraGuard AI acts as a defense-oriented AI Risk Manager that can flag potentially synthetic or manipulated identities before they progress through financial onboarding.

---

## 🎯 Use Case: AI Risk Manager for KYC Fraud Prevention

| Challenge                                                | SpectraGuard AI Solution                      |
| :------------------------------------------------------- | :-------------------------------------------- |
| Fraudsters use AI-generated faces to bypass KYC          | Detects potential synthetic/manipulated faces |
| Deepfake artifacts may be difficult to identify visually | Combines spatial + frequency-domain analysis  |
| Binary AI decisions can be risky                         | Uses an **UNCERTAIN** zone for manual review  |
| Unsupported files can crash applications                 | Graceful validation and exception handling    |
| Financial institutions need auditability                 | Confidence scores and application logging     |
| Human reviewers need actionable signals                  | Clear REAL / UNCERTAIN / FAKE classifications |

---

## 🏗️ Architecture

```text
                         Input Image
                          224 × 224
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Spatial Stream       Frequency Stream
                    │                   │
             EfficientNet-B3          2D FFT
             ImageNet Weights      Log Magnitude
                    │                   │
                 1536-dim             128-dim
                    │                   │
                    └─────────┬─────────┘
                              │
                       Concatenation
                           1664-dim
                              │
                         Linear 512
                              │
                           LayerNorm
                              │
                            GELU
                              │
                         Dropout 0.4
                              │
                         Linear 256
                              │
                           LayerNorm
                              │
                            GELU
                              │
                         Dropout 0.3
                              │
                         Linear 128
                              │
                            GELU
                              │
                         Dropout 0.2
                              │
                          Linear 1
                              │
                           Sigmoid
                              │
                         P(fake) ∈ [0,1]
```

### 🔬 Why Dual-Stream?

Traditional image classifiers primarily operate in the spatial domain. Deepfake generation, however, can introduce subtle artifacts in both:

**Spatial Domain**

* Texture inconsistencies
* Facial artifacts
* Blending boundaries
* Abnormal skin details
* Local image distortions

**Frequency Domain**

* Spectral inconsistencies
* High-frequency artifacts
* Generator fingerprints
* Compression-related anomalies
* Frequency patterns invisible to human inspection

Combining both representations provides a broader signal for deepfake-risk assessment.

---

## 📊 Decision Thresholds

<div align="center">

|   Probability   |  Classification | Action                             |
| :-------------: | :-------------: | :--------------------------------- |
|    **≥ 0.65**   |   ⚠️ **FAKE**   | Flag for rejection / investigation |
|    **≤ 0.35**   |    ✅ **REAL**   | Approve / continue onboarding      |
| **0.35 – 0.65** | ❓ **UNCERTAIN** | Escalate to manual review          |

</div>

> **Note:** Thresholds are configurable through the Streamlit sidebar and should be calibrated using validation data before production deployment.

---

## 🚀 Quick Start

### Prerequisites

* Python **3.10+**
* Git
* Optional NVIDIA GPU with CUDA
* CPU inference is supported

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ashish-nayakk/spectraguard-ai.git
cd spectraguard-ai
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Build Model Weights

```bash
python scripts/build_weights.py
```

This downloads the required EfficientNet-B3 ImageNet weights automatically.

> **No training dataset is required for basic inference setup.**

### Step 4 — Launch the Application

```bash
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

---

## 📁 Project Structure

```text
spectraguard-ai/
│
├── app/
│   ├── app.py                  # Streamlit application
│   └── predictor.py            # Inference engine, TTA & logging
│
├── model/
│   ├── architecture.py         # Dual-stream DeepfakeDetector
│   ├── train.py                # Optional fine-tuning pipeline
│   └── weights/
│       └── deepfake_detector.pt
│
├── utils/
│   ├── face_utils.py           # Face detection & preprocessing
│   └── video_utils.py          # Frame extraction & aggregation
│
├── scripts/
│   ├── build_weights.py        # Generate/download model weights
│   └── prepare_data.py         # Dataset preparation
│
├── logs/                       # Audit trails & error logs
├── requirements.txt
└── README.md
```

---

## 🎓 Optional: Fine-Tune for Higher Accuracy

SpectraGuard AI can be fine-tuned on deepfake datasets for improved domain-specific performance.

### Demo Dataset

```bash
python scripts/prepare_data.py --demo --n 1000

python model/train.py \
  --data_dir data/dataset \
  --epochs_p1 10 \
  --epochs_p2 20
```

### DFDC Dataset

Download the **DeepFake Detection Challenge (DFDC)** dataset through Kaggle and prepare it using:

```bash
kaggle competitions download -c deepfake-detection-challenge

unzip deepfake-detection-challenge.zip -d /path/to/dfdc
```

Then:

```bash
python scripts/prepare_data.py --dfdc /path/to/dfdc
```

Train using:

```bash
python model/train.py \
  --data_dir data/dataset \
  --batch_size 32 \
  --device cuda
```

### Training CLI

```bash
python model/train.py \
  --data_dir data/dataset \
  --epochs_p1 10 \
  --epochs_p2 20 \
  --batch_size 32 \
  --lr_p1 1e-3 \
  --lr_p2 5e-6 \
  --device cuda
```

---

## 📈 Expected Performance

> **Important:** The figures below are development expectations, not independently verified benchmark results. Actual performance depends heavily on the dataset, preprocessing pipeline, threshold calibration, and evaluation protocol.

| Mode                     | Expected Accuracy | Notes                               |
| :----------------------- | :---------------: | :---------------------------------- |
| Pretrained ImageNet only |      ~60–70%      | General visual feature extraction   |
| Fine-tuned on DFDC       |      ~85–90%      | Dataset-specific deepfake detection |
| Fine-tuned on FF++ (C23) |      ~89–94%      | Strong benchmark-oriented setup     |

For production use, evaluation should include **precision, recall, F1-score, ROC-AUC, false-positive rate, false-negative rate, and calibration** rather than accuracy alone.

---

## 🧠 Model Components

### EfficientNet-B3

The spatial branch uses **EfficientNet-B3** to extract high-level visual representations from the input face image.

### 2D FFT

The frequency branch converts image information into the frequency domain using a **2D Fast Fourier Transform**.

The resulting spectrum is transformed using logarithmic magnitude scaling before being passed through convolutional layers.

### Feature Fusion

The two feature representations are concatenated:

```text
EfficientNet-B3: 1536 dimensions
Frequency branch: 128 dimensions
----------------------------------
Combined:         1664 dimensions
```

The fused representation is processed by a multilayer classification head.

---

## 🛡️ Risk Management Workflow

```text
User / KYC Image
       │
       ▼
Image Validation
       │
       ▼
Face Detection & Preprocessing
       │
       ▼
Dual-Stream Analysis
 ┌─────┴─────┐
 ▼           ▼
Spatial     Frequency
Analysis    Analysis
 └─────┬─────┘
       ▼
Feature Fusion
       │
       ▼
Deepfake Probability
       │
 ┌─────┼─────────────┐
 ▼     ▼             ▼
REAL  UNCERTAIN     FAKE
 │       │            │
 ▼       ▼            ▼
Proceed Manual      Flag /
       Review       Investigate
```

---

## 🔍 Key Features

* ✅ Dual-stream deepfake detection
* ✅ EfficientNet-B3 spatial feature extraction
* ✅ Frequency-domain FFT analysis
* ✅ Configurable risk thresholds
* ✅ REAL / UNCERTAIN / FAKE classification
* ✅ Manual-review escalation
* ✅ Face preprocessing
* ✅ Video frame analysis utilities
* ✅ Test-Time Augmentation (TTA)
* ✅ Error handling and validation
* ✅ Audit logging
* ✅ Streamlit interactive interface
* ✅ Hugging Face Spaces deployment
* ✅ CPU inference support
* ✅ Optional GPU acceleration
* ✅ Optional dataset fine-tuning

---

## 🛠️ Tech Stack

| Category               | Technologies                          |
| :--------------------- | :------------------------------------ |
| **Language**           | Python 3.10+                          |
| **Deep Learning**      | PyTorch                               |
| **Model Architecture** | EfficientNet-B3, Dual-Stream CNN      |
| **Computer Vision**    | OpenCV                                |
| **Signal Processing**  | 2D FFT, Log Magnitude Spectrum        |
| **ML Utilities**       | NumPy, torchvision                    |
| **Frontend**           | Streamlit                             |
| **Deployment**         | Hugging Face Spaces, Docker           |
| **Monitoring**         | Custom Logging, Confidence Thresholds |
| **Version Control**    | Git & GitHub                          |

---

## 🏆 Razorpay AI Buildathon 2026

SpectraGuard AI was developed for the **Razorpay AI Buildathon 2026** under the following tracks:

<div align="center">

![Track 02](https://img.shields.io/badge/Track%2002-AI%20Risk%20Manager-6366F1?style=for-the-badge)

![Track 05](https://img.shields.io/badge/Track%2005-Open%20Track-10B981?style=for-the-badge)

</div>

### Why This Fits the AI Risk Manager Track

Financial platforms increasingly face risks from AI-generated identities, synthetic media, and automated fraud.

SpectraGuard AI addresses this challenge by providing a **defense-oriented ML risk layer** that can:

1. Analyze KYC imagery.
2. Detect potential synthetic/manipulated faces.
3. Quantify model confidence.
4. Avoid overconfident binary decisions.
5. Escalate uncertain cases to human reviewers.
6. Maintain logs for audit and investigation.

> **Buildathon positioning:** An AI-powered visual risk assessment layer for financial onboarding and KYC fraud prevention.

---

## 🔐 Responsible AI & Limitations

SpectraGuard AI is intended as a **defensive research and educational system**.

It should **not** be used as the sole basis for rejecting customers, denying financial services, or making high-impact decisions.

Real-world deployment should include:

* Human-in-the-loop review
* Dataset diversity testing
* Bias and fairness evaluation
* Threshold calibration
* False-positive monitoring
* False-negative monitoring
* Model drift detection
* Security testing
* Privacy-preserving data handling
* Regulatory and compliance review

> A deepfake detector provides a **risk signal**, not definitive proof that an identity is fraudulent.

---

## 🧪 Evaluation Recommendations

For a production-oriented deployment, evaluate the model using:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
False Positive Rate
False Negative Rate
Confusion Matrix
Calibration Error
Inference Latency
```

Testing should include both **real-world KYC images** and diverse synthetic/manipulated media generated using multiple deepfake techniques.

---

## 🌐 Live Deployment

<div align="center">

### 🤗 Hugging Face Demo

**[Launch SpectraGuard AI →](https://ashish-kumar-nayak-spectraguard-ai.hf.space/)**

### 💻 Source Code

**[View GitHub Repository →](https://github.com/Ashish-nayakk/spectraguard-ai)**

</div>

---

## 📌 Future Improvements

* [ ] Train and benchmark on larger deepfake datasets
* [ ] Add face-landmark based artifact analysis
* [ ] Add video-level temporal modeling
* [ ] Add explainable heatmaps / Grad-CAM
* [ ] Add model calibration dashboard
* [ ] Add REST API for KYC integration
* [ ] Add database-backed audit trails
* [ ] Add drift monitoring
* [ ] Add ensemble models
* [ ] Add real-time video verification
* [ ] Improve low-resolution face detection
* [ ] Add comprehensive fairness evaluation

---

## 📝 License

This project is intended for **educational and research purposes**.

Use responsibly and ensure compliance with applicable privacy, security, financial, and AI regulations before deploying similar systems in production.

---

## 📬 Connect

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-Ashish--nayakk-181717?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/Ashish-nayakk)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ashish%20Kumar%20Nayak-0A66C2?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/)

[![Portfolio](https://img.shields.io/badge/Portfolio-ashishnayakworks.vercel.app-000000?style=for-the-badge\&logo=vercel\&logoColor=white)](https://ashishnayakworks.vercel.app/)

</div>

---

<div align="center">

⭐ **If you find SpectraGuard AI useful, please give the repository a star!** ⭐

### 🛡️ Built for Responsible AI Security

**SpectraGuard AI — Detect. Assess. Protect.**

☕ Built during the night shift.

</div>
