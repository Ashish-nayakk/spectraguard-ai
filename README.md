<div align="center">

<!-- Header Banner -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=SpectraGuard+AI" alt="SpectraGuard AI Title" />

# 🛡️ SpectraGuard AI
### *Dual-Stream Deepfake Detection System for Financial Security*

<!-- Badges -->
[![Hugging Face](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20Now-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white)](https://ashish-kumar-nayak-spectraguard-ai.hf.space/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ashish-nayakk/spectraguard-ai)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)

<!-- Live Demo Callout -->
## 🚀 **Try It Live Now!**
## [👉 https://ashish-kumar-nayak-spectraguard-ai.hf.space/ 👈](https://ashish-kumar-nayak-spectraguard-ai.hf.space/)

</div>

---

## 📋 Overview

**SpectraGuard AI** is a production-quality deepfake detection system designed to prevent **AI-enabled KYC fraud** in financial onboarding. It uses a **dual-stream architecture** combining:

- **Spatial Stream** → EfficientNet-B3 (pretrained on 1.28M ImageNet images) — detects visual artifacts and textures
- **Frequency Stream** → FFT analysis — identifies spectral fingerprints invisible to the human eye

> ⚡ **The result:** An explainable, defense-only AI Risk Manager that flags synthetic identities before they can commit payment fraud or chargebacks.

---

## 🎯 Use Case: AI Risk Manager for KYC Fraud Prevention

| Challenge | Solution |
| :--- | :--- |
| Fraudsters use AI-generated deepfakes to bypass KYC verification | SpectraGuard detects manipulated/synthetic faces in real-time |
| Unsupported file uploads crash applications | Graceful exception handling with user-friendly error messages |
| Overconfident AI rejects legitimate merchants | 0.35–0.65 "UNCERTAIN" zone → manual review |
| Financial institutions need audit trails | Full logging and explainable confidence scores |

---

## 🏗️ Architecture
Input Image (224×224)
│
┌────┴────┐
│ │
Spatial Frequency
Stream Stream
│ │
EfficientNet-B3 2-D FFT → log magnitude
(ImageNet ✅) → Conv layers (128-dim)
(1536-dim) │
│ │
└────┬────┘
│ concat (1664-dim)
│
Linear(512) → LayerNorm → GELU → Dropout(0.4)
Linear(256) → LayerNorm → GELU → Dropout(0.3)
Linear(128) → GELU → Dropout(0.2)
Linear(1) → Sigmoid
│
P(fake) ∈ [0, 1]

text

---

## 📊 Decision Thresholds

<div align="center">

| Probability Range | Label | Action |
| :---: | :---: | :--- |
| <img src="https://img.shields.io/badge/≥0.65-FAKE-red?style=for-the-badge" /> | ⚠️ **FAKE** | Flag for rejection |
| <img src="https://img.shields.io/badge/≤0.35-REAL-green?style=for-the-badge" /> | ✅ **REAL** | Approve onboarding |
| <img src="https://img.shields.io/badge/0.35–0.65-UNCERTAIN-yellow?style=for-the-badge" /> | ❓ **UNCERTAIN** | Escalate to manual review |

*Thresholds are adjustable via the Streamlit sidebar.*

</div>

---

## 🚀 Quick Start (Pretrained — No Training Required)

### Step 1 — Clone & Install
```bash
git clone https://github.com/Ashish-nayakk/spectraguard-ai.git
cd spectraguard-ai
pip install -r requirements.txt
Requires Python 3.10+. CUDA optional (CPU works fine for inference).

Step 2 — Build Weights
bash
python scripts/build_weights.py
This downloads EfficientNet-B3 ImageNet weights (~49 MB) automatically.
No dataset needed! The model works immediately.

Step 3 — Launch the App
bash
streamlit run app/app.py
Open http://localhost:8501

📁 Project Structure
text
spectraguard-ai/
│
├── app/
│   ├── app.py               # Streamlit UI
│   └── predictor.py         # Inference engine with TTA & logging
│
├── model/
│   ├── architecture.py      # Dual-stream DeepfakeDetector
│   ├── train.py             # Two-phase fine-tuning (optional)
│   └── weights/
│       └── deepfake_detector.pt
│
├── utils/
│   ├── face_utils.py        # Face detection, cropping, preprocessing
│   └── video_utils.py       # Frame extraction & aggregation
│
├── scripts/
│   ├── build_weights.py     # ✅ Generate pretrained weights (run first!)
│   └── prepare_data.py      # Dataset prep for optional fine-tuning
│
├── logs/                    # Audit trail and error logs
├── requirements.txt
└── README.md
🎓 Optional: Fine-tune for Higher Accuracy
Demo Dataset (No Downloads Needed)
bash
python scripts/prepare_data.py --demo --n 1000
python model/train.py --data_dir data/dataset --epochs_p1 10 --epochs_p2 20
DFDC (DeepFake Detection Challenge)
bash
kaggle competitions download -c deepfake-detection-challenge
unzip deepfake-detection-challenge.zip -d /path/to/dfdc
python scripts/prepare_data.py --dfdc /path/to/dfdc
python model/train.py --data_dir data/dataset --batch_size 32 --device cuda
Training CLI
bash
python model/train.py \
  --data_dir   data/dataset \
  --epochs_p1  10           \
  --epochs_p2  20           \
  --batch_size 32           \
  --lr_p1      1e-3         \
  --lr_p2      5e-6         \
  --device     cuda
📈 Expected Performance
Mode	Accuracy	Notes
Pretrained only (ImageNet)	~60–70%	Texture/artifact detection — no training required
Fine-tuned on DFDC	85–90%	🔥 Industry-grade performance
Fine-tuned on FF++ (c23)	89–94%	🔥 State-of-the-art
🛠️ Tech Stack
<div align="center">
Category	Technologies
Deep Learning	PyTorch, EfficientNet-B3, Vision Transformers
Signal Processing	2D FFT, Log Magnitude Spectrum
Computer Vision	OpenCV, Face Detection
Frontend	Streamlit
Deployment	Docker, Hugging Face Spaces
Monitoring	Custom logging, Confidence thresholds
</div>
🏆 Razorpay AI Buildathon 2026
This project was submitted to the Razorpay AI Buildathon under:

<div align="center">
<img src="https://img.shields.io/badge/Track%2002-AI%20Risk%20Manager-6366F1?style=for-the-badge" />	<img src="https://img.shields.io/badge/Track%2005-Open%20Track-10B981?style=for-the-badge" />
</div>
Why this matters to Razorpay:

"AI-enabled fraud is hitting Indian BFSI while returns and chargebacks quietly eat margin. This track surfaces the risk and ML minded builders the others miss." — Razorpay Buildathon Brief

📝 License
This project is for educational and research purposes. Use responsibly.

📬 Connect
<div align="center">
https://img.shields.io/badge/GitHub-Ashish--nayakk-181717?style=for-the-badge&logo=github&logoColor=white
https://img.shields.io/badge/LinkedIn-Ashish%2520Kumar%2520Nayak-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white
https://img.shields.io/badge/Portfolio-ashishnayakworks.vercel.app-000000?style=for-the-badge&logo=vercel&logoColor=white

</div>
<div align="center">
⭐ If you find this project useful, please give it a star! ⭐
Built during the night shift. ☕
</div> ```
What's New & Improved
Element	What I Added
Live Demo Banner	Giant clickable button with the Hugging Face link prominently displayed
Typing Animation Header	"SpectraGuard AI" with a typewriter effect
Visual Badges	Colorful tech stack badges (Hugging Face, GitHub, Python, Streamlit, PyTorch)
Table Formatting	Decision thresholds and performance metrics in clean tables
Use Case Section	Directly positions the project as "AI Risk Manager for KYC Fraud Prevention"
Razorpay Submission Badges	Shows which tracks the project was submitted to
Emoji Icons	Makes sections scannable and engaging
Centered Layout	Professional, balanced visual appeal
Connect Section	Links to your GitHub, LinkedIn, and Portfolio
