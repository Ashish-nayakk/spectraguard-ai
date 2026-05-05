import * as THREE from 'three';

// Mobile detection for performance
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

// Three.js background setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 30;
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);
document.getElementById('three-canvas-container').appendChild(renderer.domElement);

const particlesCount = isMobile ? 600 : 1800;
const positions = new Float32Array(particlesCount * 3);
for (let i = 0; i < particlesCount; i++) {
    positions[i*3] = (Math.random() - 0.5) * 80;
    positions[i*3+1] = (Math.random() - 0.5) * 50;
    positions[i*3+2] = (Math.random() - 0.5) * 40 - 20;
}
const particlesGeo = new THREE.BufferGeometry();
particlesGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const particlesMat = new THREE.PointsMaterial({ color: 0x3b82f6, size: isMobile ? 0.08 : 0.12, transparent: true, blending: THREE.AdditiveBlending });
const particles = new THREE.Points(particlesGeo, particlesMat);
scene.add(particles);

const knotGeo = new THREE.TorusKnotGeometry(2, 0.5, 128, 16, 3, 4);
const knotMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, emissive: 0x1e3a8a, roughness: 0.3, metalness: 0.8 });
const knot = new THREE.Mesh(knotGeo, knotMat);
scene.add(knot);

const ambient = new THREE.AmbientLight(0x404060);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);

let time = 0;
function animate() {
    requestAnimationFrame(animate);
    time += 0.005;
    particles.rotation.y = time * 0.1;
    particles.rotation.x = Math.sin(time * 0.2) * 0.1;
    knot.rotation.x = time * 0.3;
    knot.rotation.y = time * 0.5;
    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// UI Helper Functions
function getSettings() {
    const modelSelect = document.getElementById('model-select');
    return {
        use_tta: document.getElementById('tta-toggle').checked,
        fake_th: parseFloat(document.getElementById('fake-thresh').value),
        real_th: parseFloat(document.getElementById('real-thresh').value),
        model: modelSelect ? modelSelect.value : 'custom'
    };
}

document.getElementById('fake-thresh').addEventListener('input', (e) => {
    document.getElementById('fake-val').innerText = (e.target.value * 100).toFixed(0) + '%';
});
document.getElementById('real-thresh').addEventListener('input', (e) => {
    document.getElementById('real-val').innerText = (e.target.value * 100).toFixed(0) + '%';
});

function drawGauge(canvasId, prob, label) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    const angle = (prob * Math.PI * 2) - Math.PI/2;
    const cx = w/2, cy = h/2, r = 60;
    ctx.beginPath();
    ctx.arc(cx,cy,r, -Math.PI/2, Math.PI*1.5);
    ctx.strokeStyle = '#2a3448';
    ctx.lineWidth = 10;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx,cy,r, -Math.PI/2, angle);
    let color = '#3b82f6';
    if (label === 'FAKE') color = '#ff3366';
    else if (label === 'REAL') color = '#00e676';
    else color = '#ffb74d';
    ctx.strokeStyle = color;
    ctx.lineWidth = 10;
    ctx.stroke();
    ctx.font = 'bold 20px "JetBrains Mono"';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.fillText(`${Math.round(prob*100)}%`, cx, cy+6);
}

function updateVerdict(containerId, result) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const label = result.label;
    let cls = label.toLowerCase();
    el.innerHTML = `<div class="verdict-label ${cls}">${label === 'FAKE' ? '⚠️ ' : (label === 'REAL' ? '✅ ' : '❓ ')}${label}</div>
                    <div style="font-size:0.85rem">Confidence: ${(result.confidence*100).toFixed(1)}%</div>`;
}

function showLoader(id, show) {
    const el = document.getElementById(id);
    if (show) el.classList.add('active');
    else el.classList.remove('active');
}

// -------- IMAGE FORENSICS (with model selection) --------
const imgDrop = document.getElementById('img-dropzone');
const imgInput = document.getElementById('img-input');
imgDrop.addEventListener('click', () => imgInput.click());
imgInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    showLoader('img-loader', true);
    const settings = getSettings();
    // Choose endpoint based on model
    let apiUrl = '/predict_image';
    if (settings.model === 'hf') {
        apiUrl = '/predict_hf';
    }
    const fd = new FormData();
    fd.append('image', file);
    fd.append('use_tta', settings.use_tta);
    fd.append('fake_threshold', settings.fake_th);
    fd.append('real_threshold', settings.real_th);
    if (settings.model === 'hf') {
        fd.append('model', 'hf');
    }
    try {
        const resp = await fetch(apiUrl, { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('img-result').style.display = 'block';
        document.getElementById('img-preview').src = URL.createObjectURL(file);
        document.getElementById('img-face').src = data.face_data_url || '';
        document.getElementById('img-conf').innerText = (data.confidence*100).toFixed(1)+'%';
        document.getElementById('img-prob').innerText = (data.fake_probability*100).toFixed(1)+'%';
        document.getElementById('img-explanation').innerText = data.explanation;
        updateVerdict('img-verdict', data);
        drawGauge('img-gauge', data.fake_probability, data.label);
    } catch(err) { alert('Error: '+err.message); }
    finally { showLoader('img-loader', false); }
});

// -------- WEBCAM (only custom model – can be extended later) --------
let stream = null;
const video = document.getElementById('webcam-video');
const startBtn = document.getElementById('webcam-start');
const captureBtn = document.getElementById('webcam-capture');
const scanRing = document.querySelector('.scan-ring');

startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        captureBtn.disabled = false;
        document.getElementById('webcam-status').innerHTML = 'LIVE';
        document.getElementById('webcam-status').classList.remove('led-blue');
        document.getElementById('webcam-status').classList.add('led-green');
    } catch(e) { alert('Camera access denied'); }
});

captureBtn.addEventListener('click', async () => {
    if (!video.srcObject) return;
    scanRing.classList.add('active');
    setTimeout(() => scanRing.classList.remove('active'), 400);
    const canvas = document.getElementById('webcam-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
        showLoader('webcam-loader', true);
        const fd = new FormData();
        fd.append('image', blob, 'webcam.jpg');
        const s = getSettings();
        fd.append('fake_threshold', s.fake_th);
        fd.append('real_threshold', s.real_th);
        // Webcam always uses custom model (can add HF later)
        const resp = await fetch('/predict_webcam', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('webcam-result').style.display = 'block';
        document.getElementById('webcam-preview').src = URL.createObjectURL(blob);
        if (data.face_data_url) document.getElementById('webcam-face').src = data.face_data_url;
        else document.getElementById('webcam-face').src = '';
        document.getElementById('webcam-conf').innerText = (data.confidence*100).toFixed(1)+'%';
        document.getElementById('webcam-prob').innerText = (data.fake_probability*100).toFixed(1)+'%';
        document.getElementById('webcam-explanation').innerText = data.explanation;
        updateVerdict('webcam-verdict', data);
        drawGauge('webcam-gauge', data.fake_probability, data.label);
        showLoader('webcam-loader', false);
    }, 'image/jpeg');
});

// -------- VIDEO DEEP SCAN (custom model only) --------
const videoDrop = document.getElementById('video-dropzone');
const videoInput = document.getElementById('video-input');
videoDrop.addEventListener('click', () => videoInput.click());
videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    showLoader('video-loader', true);
    const fd = new FormData();
    fd.append('video', file);
    const s = getSettings();
    fd.append('use_tta', s.use_tta);
    fd.append('fake_threshold', s.fake_th);
    fd.append('real_threshold', s.real_th);
    fd.append('interval', '10');
    fd.append('max_frames', '80');
    try {
        const resp = await fetch('/predict_video', { method: 'POST', body: fd });
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        document.getElementById('video-controls').style.display = 'block';
        document.getElementById('vid-frames').innerText = data.num_frames;
        document.getElementById('vid-fake').innerText = data.fake_frames;
        document.getElementById('vid-real').innerText = data.real_frames;
        document.getElementById('vid-unc').innerText = data.uncertain_frames;
        document.getElementById('video-explanation').innerText = data.explanation;
        updateVerdict('video-verdict', data);
        const ctx = document.getElementById('video-chart').getContext('2d');
        if (window.videoChart) window.videoChart.destroy();
        window.videoChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.frame_indices.map(i => `#${i}`),
                datasets: [{
                    label: 'Fake Probability',
                    data: data.frame_probs,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59,130,246,0.2)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                scales: { y: { min: 0, max: 1, title: { display: true, text: 'Fake Probability', color: '#94a3b8' } } }
            }
        });
    } catch(err) { alert('Video error: '+err.message); }
    finally { showLoader('video-loader', false); }
});