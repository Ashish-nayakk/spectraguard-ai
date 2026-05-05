"""
app/app.py
==========
AI Deepfake Detection System — Streamlit UI

Run:  streamlit run app/app.py
"""

import os
import sys
import time
import tempfile
import logging
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="AI Deepfake Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --surface2:  #21262d;
  --border:    #30363d;
  --accent:    #58a6ff;
  --fake:      #f85149;
  --real:      #3fb950;
  --uncertain: #d29922;
  --text:      #e6edf3;
  --muted:     #7d8590;
  --radius:    10px;
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important; color:var(--text)!important;
  font-family:'Inter',sans-serif!important;
}
[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
}
[data-testid="stSidebar"] *{ color:var(--text)!important; }
h1,h2,h3{ font-family:'Inter',sans-serif!important; font-weight:800; }

/* Buttons */
.stButton>button{
  background:transparent!important; border:1px solid var(--accent)!important;
  color:var(--accent)!important; border-radius:var(--radius)!important;
  font-family:'JetBrains Mono',monospace!important; font-size:0.82rem!important;
  padding:.45rem 1.1rem!important; transition:all .2s!important;
  letter-spacing:.04em;
}
.stButton>button:hover{
  background:var(--accent)!important; color:var(--bg)!important;
  box-shadow:0 0 18px rgba(88,166,255,0.35)!important;
}

/* Verdict cards */
.card{
  background:var(--surface2); border:1px solid var(--border);
  border-radius:var(--radius); padding:1.6rem 1.8rem;
  margin:.6rem 0; position:relative; overflow:hidden;
}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.card.fake::before{background:var(--fake);}
.card.real::before{background:var(--real);}
.card.uncertain::before{background:var(--uncertain);}
.card.error::before{background:var(--muted);}

.verdict{font-size:2.6rem;font-weight:800;letter-spacing:-.02em;margin:0;}
.verdict.fake{color:var(--fake);}
.verdict.real{color:var(--real);}
.verdict.uncertain{color:var(--uncertain);}

.conf-text{font-size:.95rem;color:var(--muted);margin:.25rem 0;}
.expl{
  background:var(--surface); border-left:3px solid var(--accent);
  border-radius:0 var(--radius) var(--radius) 0;
  padding:.85rem 1rem; font-size:.83rem; line-height:1.65;
  color:var(--text); margin-top:.75rem;
}

/* Section headers */
.sh{
  font-size:.7rem; text-transform:uppercase; letter-spacing:.14em;
  color:var(--muted); border-bottom:1px solid var(--border);
  padding-bottom:.35rem; margin-bottom:.5rem;
}

/* Stat chips */
.chip{
  display:inline-block; background:var(--surface2);
  border:1px solid var(--border); border-radius:5px;
  padding:.2rem .6rem; font-size:.74rem; color:var(--muted); margin:.15rem;
  font-family:'JetBrains Mono',monospace;
}

/* Info / warn banners */
.info-banner{
  background:rgba(88,166,255,.07); border:1px solid rgba(88,166,255,.25);
  border-radius:var(--radius); padding:.85rem 1rem;
  font-size:.83rem; color:#79c0ff; margin:.4rem 0;
}
.warn-banner{
  background:rgba(248,81,73,.07); border:1px solid rgba(248,81,73,.25);
  border-radius:var(--radius); padding:.85rem 1rem;
  font-size:.83rem; color:#ff7b72; margin:.4rem 0;
}

/* Metric tiles */
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin:.8rem 0;}
.metric-tile{
  background:var(--surface2); border:1px solid var(--border);
  border-radius:var(--radius); padding:.9rem; text-align:center;
}
.metric-val{font-size:1.7rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.metric-lbl{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);}

/* File uploader */
[data-testid="stFileUploader"]{
  background:var(--surface2)!important; border:1px dashed var(--border)!important;
  border-radius:var(--radius)!important;
}
/* Scrollbar */
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    from app.predictor import get_model
    m = get_model()
    return m is not None


# ── Header ─────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div style="padding:1.2rem 0 1rem;border-bottom:1px solid #30363d;margin-bottom:1.4rem;">
      <div style="display:flex;align-items:center;gap:1rem;">
        <div style="font-size:2.2rem;">🔬</div>
        <div>
          <h1 style="margin:0;font-size:1.85rem;color:#e6edf3;">
            AI Deepfake Detection System
          </h1>
          <p style="margin:0;color:#7d8590;font-size:.75rem;letter-spacing:.06em;">
            EFFICIENTNET-B3 + FFT DUAL-STREAM · TRANSFER LEARNING · TTA INFERENCE
          </p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<p class="sh">Detection Mode</p>', unsafe_allow_html=True)
        mode = st.selectbox("", [
            "📷  Image Upload",
            "🎥  Video Upload",
            "📸  Webcam Capture",
        ], label_visibility="collapsed")

        st.markdown('<p class="sh" style="margin-top:1.2rem;">Inference Settings</p>',
                    unsafe_allow_html=True)
        use_tta = st.toggle("Test-Time Augmentation (TTA)", value=True,
                            help="Average 5 augmented predictions. More accurate but slower.")
        fake_th = st.slider("FAKE threshold", 0.50, 0.90, 0.70, 0.05)
        real_th = st.slider("REAL threshold", 0.10, 0.49, 0.40, 0.05)

        st.markdown('<p class="sh" style="margin-top:1.2rem;">Video Settings</p>',
                    unsafe_allow_html=True)
        v_interval  = st.slider("Frame interval",  5, 60, 10)
        v_maxframes = st.slider("Max frames",      20, 150, 80)
        v_method    = st.selectbox("Aggregation", ["average", "majority"])

        st.markdown('<p class="sh" style="margin-top:1.2rem;">Model</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.76rem;color:#7d8590;line-height:1.7;">
        <b style="color:#e6edf3;">Architecture</b><br>
        EfficientNet-B3 + FFT stream<br>
        11.8M parameters | 45 MB<br><br>
        <b style="color:#e6edf3;">Thresholds</b><br>
        ≥ 70% → ⚠️ FAKE<br>
        ≤ 40% → ✅ REAL<br>
        40–70% → ❓ UNCERTAIN<br>
        <b style="color:#e6edf3;">⚠️ Limitations</b><br>
        Domain shift may affect accuracy on deepfake methods
        not in training data. Results are probabilistic.
        </div>
        """, unsafe_allow_html=True)

        return mode, use_tta, fake_th, real_th, v_interval, v_maxframes, v_method


# ── Verdict card ───────────────────────────────────────────────────────────
ICONS = {"FAKE": "⚠️", "REAL": "✅", "UNCERTAIN": "❓", "ERROR": "🚫"}

def render_verdict(result: dict, fake_th: float, real_th: float):
    label = result.get("label", "UNCERTAIN")
    conf  = result.get("confidence", 0.0)
    prob  = result.get("fake_probability", 0.5)
    expl  = result.get("explanation", "")
    icon  = ICONS.get(label, "❓")
    cls   = label.lower()

    st.markdown(f"""
    <div class="card {cls}">
      <p class="verdict {cls}">{icon} {label}</p>
      <p class="conf-text">
        Confidence&nbsp;<strong style="color:#e6edf3">{conf*100:.1f}%</strong>
        &nbsp;·&nbsp;
        Fake probability&nbsp;<strong style="color:#e6edf3">{prob*100:.1f}%</strong>
      </p>
      <div class="expl">{expl}</div>
    </div>
    """, unsafe_allow_html=True)

    # Gauge
    gcolor = "#f85149" if prob >= fake_th else ("#3fb950" if prob <= real_th else "#d29922")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Fake Probability %", "font": {"color": "#7d8590", "size": 12}},
        number={"suffix": "%", "font": {"color": gcolor, "size": 26}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#30363d",
                     "tickfont": {"color": "#7d8590", "size": 9}},
            "bar":  {"color": gcolor, "thickness": 0.22},
            "bgcolor": "#21262d", "bordercolor": "#30363d",
            "steps": [
                {"range": [0, real_th*100],   "color": "rgba(63,185,80,.08)"},
                {"range": [real_th*100, fake_th*100], "color": "rgba(210,153,34,.08)"},
                {"range": [fake_th*100, 100], "color": "rgba(248,81,73,.08)"},
            ],
            "threshold": {"line": {"color": "#e6edf3", "width": 2},
                          "thickness": 0.75, "value": prob * 100},
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=28, b=8),
                      paper_bgcolor="#0d1117", font={"color": "#e6edf3"})
    st.plotly_chart(fig, use_container_width=True)


# ── Video results ──────────────────────────────────────────────────────────
def handle_image(use_tta, fake_th, real_th):
    from app.predictor import predict_image
    from app import predictor  # 🔥 added

    uploaded = st.file_uploader(
        "Drop image here",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.markdown('<div class="info-banner">👆 Upload an image to begin analysis.</div>',
                    unsafe_allow_html=True)
        return

    raw = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        st.markdown('<div class="warn-banner">Cannot decode image.</div>', unsafe_allow_html=True)
        return

    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<p class="sh">Input image</p>', unsafe_allow_html=True)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        st.image(rgb, use_container_width=True)

    with st.spinner("Detecting face · Running inference…"):
        t0 = time.time()

        # 🔥 THIS IS THE IMPORTANT FIX
        predictor.FAKE_THRESH = fake_th
        predictor.REAL_THRESH = real_th

        result = predict_image(bgr, use_tta=use_tta, source=uploaded.name)

        elapsed = time.time() - t0

    with col_img:
        face = result.get("face_rgb")
        if face is not None:
            st.markdown('<p class="sh" style="margin-top:.8rem;">Detected face (model input)</p>',
                        unsafe_allow_html=True)
            st.image(face, width=180)
        else:
            st.markdown(
                '<div class="warn-banner">⚠️ No face detected — analysing full image.</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<span class="chip">⏱ {elapsed*1000:.0f} ms</span>'
                    f'<span class="chip">TTA: {"on" if use_tta else "off"}</span>',
                    unsafe_allow_html=True)

    with col_res:
        st.markdown('<p class="sh">Analysis result</p>', unsafe_allow_html=True)
        render_verdict(result, fake_th, real_th)

def handle_video(use_tta, fake_th, real_th, v_interval, v_maxframes, v_method):
    from app.predictor import predict_video

    uploaded = st.file_uploader(
        "Drop video here",
        type=["mp4","avi","mov","mkv","webm"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.markdown('<div class="info-banner">👆 Upload a video to begin analysis.</div>',
                    unsafe_allow_html=True)
        return

    st.markdown(
        f'<span class="chip">📁 {uploaded.name}</span>'
        f'<span class="chip">{uploaded.size//1024} KB</span>',
        unsafe_allow_html=True,
    )

    # Write to temp file
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    prog = st.progress(0, text="Extracting frames…")

    with st.spinner("Analysing video frames — this may take a moment…"):
        t0     = time.time()
        result = predict_video(
            path=tmp_path,
            interval=v_interval,
            max_frames=v_maxframes,
            method=v_method,
            use_tta=False,   # TTA per-frame would be too slow for video
        )
        elapsed = time.time() - t0

    prog.progress(1.0, text="Done.")
    try: os.unlink(tmp_path)
    except: pass

    col_v, col_s = st.columns([1, 1], gap="large")
    with col_v:
        st.markdown('<p class="sh">Aggregated verdict</p>', unsafe_allow_html=True)
        render_verdict(result, fake_th, real_th)
        st.markdown(f'<span class="chip">⏱ {elapsed:.1f}s total</span>'
                    f'<span class="chip">method: {v_method}</span>',
                    unsafe_allow_html=True)
    with col_s:
        st.markdown('<p class="sh">Frame-level statistics</p>', unsafe_allow_html=True)
        render_video_stats(result, fake_th, real_th)


def render_video_stats(result: dict, fake_th: float, real_th: float):
    """Render frame-level statistics for video results."""
    frames = result.get("frames", [])
    if not frames:
        st.markdown("No frame data available.", unsafe_allow_html=True)
        return

    fake_probs = [f.get("fake_probability", 0.5) for f in frames]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Frames", len(fake_probs))
    with col2:
        avg_fake = np.mean(fake_probs)
        st.metric("Avg Fake Prob", f"{avg_fake*100:.1f}%")
    with col3:
        fake_count = sum(1 for p in fake_probs if p >= fake_th)
        st.metric("Detected as Fake", f"{fake_count}/{len(fake_probs)}")


def handle_webcam(use_tta, fake_th, real_th):
    from app.predictor import predict_webcam_frame

    st.markdown('<div class="info-banner">📸 Capture a photo with your webcam below.</div>',
                unsafe_allow_html=True)

    cam = st.camera_input("Take a photo", label_visibility="collapsed")
    if cam is None:
        return

    raw = np.frombuffer(cam.read(), np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    with st.spinner("Analysing…"):
        result = predict_webcam_frame(bgr)

    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown('<p class="sh">Captured frame</p>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        face = result.get("face_rgb")
        if face is not None:
            st.markdown('<p class="sh" style="margin-top:.8rem;">Face crop</p>',
                        unsafe_allow_html=True)
            st.image(face, width=170)
    with col_b:
        st.markdown('<p class="sh">Result</p>', unsafe_allow_html=True)
        render_verdict(result, fake_th, real_th)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    render_header()

    # Sidebar
    mode, use_tta, fake_th, real_th, v_int, v_max, v_meth = render_sidebar()

    # Model status
    with st.spinner("Loading model…"):
        model_ok = load_predictor()

    if model_ok:
        st.markdown(
            '<div class="info-banner">✅ Model ready · EfficientNet-B3 + FFT · 11.8M params</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="warn-banner">⚠️ Weights not found. '
            'Run <code>python scripts/build_weights.py</code> then '
            '<code>python model/train.py</code>. '
            'Using random-init model — predictions will not be meaningful.</div>',
            unsafe_allow_html=True,
        )

    # Route to handler
    tag = mode.split("  ", 1)[-1].strip()
    if   "Image"  in tag: handle_image(use_tta, fake_th, real_th)
    elif "Video"  in tag: handle_video(use_tta, fake_th, real_th, v_int, v_max, v_meth)
    elif "Webcam" in tag: handle_webcam(use_tta, fake_th, real_th)

    # Footer
    st.markdown("""
    <div style="border-top:1px solid #30363d;margin-top:3rem;padding-top:1rem;
                text-align:center;color:#7d8590;font-size:.73rem;">
      AI Deepfake Detection System &nbsp;·&nbsp; EfficientNet-B3 + FFT Dual-Stream
      &nbsp;·&nbsp; Transfer Learning &nbsp;·&nbsp; DFDC / FaceForensics++ compatible
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
