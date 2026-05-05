# download_hf_model.py
from transformers import AutoImageProcessor, AutoModelForImageClassification
print("Downloading Hugging Face model...")
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
print("Model downloaded and cached.")