import os
import cv2
import base64
import tempfile
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from huggingface_hub import hf_hub_download   # <-- ADD THIS IMPORT

# Import your custom predictor
import sys
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from app.predictor import predict_image, predict_webcam_frame, get_model
from utils.face_utils import get_primary_face

# Import Hugging Face model
from hf_model import HFDeepfakeDetector

# ------------------------------------------------------------------
# Function to download custom model weights from Hugging Face Dataset
# ------------------------------------------------------------------
def download_custom_weights():
    """Download custom model weights from Hugging Face Dataset if not present locally."""
    local_dir = "model/weights"
    os.makedirs(local_dir, exist_ok=True)
    weight_files = ["deepfake_detector.pt", "best_model.pt"]
    for file in weight_files:
        local_path = os.path.join(local_dir, file)
        if not os.path.exists(local_path):
            print(f"Downloading {file} from Hugging Face Dataset...")
            try:
                hf_hub_download(
                    repo_id="ashish-kumar-nayak/spectraguard-weights",
                    filename=file,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                print(f"Downloaded {file}")
            except Exception as e:
                print(f"Could not download {file}: {e}")
        else:
            print(f"{file} already exists locally.")

# ------------------------------------------------------------------
# Download weights before trying to load the custom model
# ------------------------------------------------------------------
download_custom_weights()

# ------------------------------------------------------------------
# Load custom model (only if weights are present)
# ------------------------------------------------------------------
custom_model_available = False
try:
    model = get_model()
    if model is not None:
        custom_model_available = True
        print("Custom model loaded successfully.")
    else:
        print("Custom model could not be loaded (weights missing).")
except Exception as e:
    print(f"Custom model loading error: {e}")

# ------------------------------------------------------------------
# Load Hugging Face model (downloads ~300 MB first time)
# ------------------------------------------------------------------
hf_detector = HFDeepfakeDetector()

# ------------------------------------------------------------------
# Flask app setup
# ------------------------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

def face_to_data_url(face_rgb):
    if face_rgb is None or face_rgb.size == 0:
        return None
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def frame_predict(bgr, use_tta, fake_th, real_th):
    import app.predictor as predictor
    original_fake = predictor.FAKE_THRESH
    original_real = predictor.REAL_THRESH
    predictor.FAKE_THRESH = fake_th
    predictor.REAL_THRESH = real_th
    try:
        result = predict_image(bgr, use_tta=use_tta, source="video_frame")
        return result['fake_probability']
    finally:
        predictor.FAKE_THRESH = original_fake
        predictor.REAL_THRESH = original_real

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# -------- Custom Model Endpoints --------
@app.route('/predict_image', methods=['POST'])
def api_predict_image():
    if not custom_model_available:
        return jsonify({'error': 'Custom model not available (weights missing). Please use the Hugging Face model.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    use_tta = request.form.get('use_tta', 'true').lower() == 'true'
    fake_th = float(request.form.get('fake_threshold', 0.70))
    real_th = float(request.form.get('real_threshold', 0.40))

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({'error': 'Invalid image'}), 400

    import app.predictor as predictor
    predictor.FAKE_THRESH = fake_th
    predictor.REAL_THRESH = real_th

    result = predict_image(bgr, use_tta=use_tta, source=file.filename)
    face = result.pop('face_rgb', None)
    result['face_data_url'] = face_to_data_url(face)
    result['fake_threshold'] = fake_th
    result['real_threshold'] = real_th
    return jsonify(result)

@app.route('/predict_webcam', methods=['POST'])
def api_predict_webcam():
    if not custom_model_available:
        return jsonify({'error': 'Custom model not available (weights missing). Please use the Hugging Face model.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    file = request.files['image']
    fake_th = float(request.form.get('fake_threshold', 0.70))
    real_th = float(request.form.get('real_threshold', 0.40))

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({'error': 'Invalid image'}), 400

    import app.predictor as predictor
    predictor.FAKE_THRESH = fake_th
    predictor.REAL_THRESH = real_th

    result = predict_webcam_frame(bgr)
    face = result.pop('face_rgb', None)
    result['face_data_url'] = face_to_data_url(face)
    return jsonify(result)

@app.route('/predict_video', methods=['POST'])
def api_predict_video():
    if not custom_model_available:
        return jsonify({'error': 'Custom model not available (weights missing). Please use the Hugging Face model.'}), 503
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    interval = int(request.form.get('interval', 10))
    max_frames = int(request.form.get('max_frames', 80))
    method = request.form.get('method', 'average')
    use_tta = request.form.get('use_tta', 'false').lower() == 'true'
    fake_th = float(request.form.get('fake_threshold', 0.70))
    real_th = float(request.form.get('real_threshold', 0.40))

    suffix = Path(file.filename).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        frame_probs = []
        frame_indices = []
        frame_idx = 0
        frames_extracted = 0

        while frames_extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                prob = frame_predict(frame, use_tta, fake_th, real_th)
                frame_probs.append(prob)
                frame_indices.append(frame_idx)
                frames_extracted += 1
            frame_idx += 1
        cap.release()

        if not frame_probs:
            return jsonify({'error': 'No frames extracted'}), 400

        probs_arr = np.array(frame_probs)
        avg_prob = float(probs_arr.mean())
        fake_frames = int((probs_arr >= fake_th).sum())
        real_frames = int((probs_arr <= real_th).sum())
        uncertain_frames = len(probs_arr) - fake_frames - real_frames

        if avg_prob >= fake_th:
            label = "FAKE"
            confidence = avg_prob
        elif avg_prob <= real_th:
            label = "REAL"
            confidence = 1 - avg_prob
        else:
            label = "UNCERTAIN"
            confidence = 1 - abs(avg_prob - 0.5) * 2

        result = {
            'label': label,
            'confidence': round(confidence, 4),
            'fake_probability': round(avg_prob, 4),
            'explanation': f"Video analysis over {len(frame_probs)} frames. Average fake probability: {avg_prob*100:.1f}%. Fake frames: {fake_frames}, Real frames: {real_frames}, Uncertain: {uncertain_frames}.",
            'frame_probs': frame_probs,
            'frame_indices': frame_indices,
            'num_frames': len(frame_probs),
            'fake_frames': fake_frames,
            'real_frames': real_frames,
            'uncertain_frames': uncertain_frames,
            'total_video_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0
        }
        return jsonify(result)
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# -------- Hugging Face Model Endpoint --------
@app.route('/predict_hf', methods=['POST'])
def api_predict_hf():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400

    fake_th = float(request.form.get('fake_threshold', 0.70))
    real_th = float(request.form.get('real_threshold', 0.40))

    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({'error': 'Invalid image'}), 400

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prob = hf_detector.predict(rgb)

    if prob >= fake_th:
        label = "FAKE"
        confidence = prob
    elif prob <= real_th:
        label = "REAL"
        confidence = 1 - prob
    else:
        label = "UNCERTAIN"
        confidence = 1 - abs(prob - 0.5) * 2

    _, buffer = cv2.imencode('.jpg', bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    face_data_url = f"data:image/jpeg;base64,{img_base64}"

    result = {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(prob, 4),
        "explanation": f"Hugging Face model analysis indicates a {label} image with {confidence*100:.1f}% confidence.",
        "face_data_url": face_data_url,
        "has_face": True,
    }
    return jsonify(result)

@app.route('/health')
def health_check():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Flask app on host=0.0.0.0 port={port}")
    app.run(debug=False, host='0.0.0.0', port=port)