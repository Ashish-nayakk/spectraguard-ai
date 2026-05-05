"""
Hugging Face Deepfake Detector
Model: dima806/deepfake_vs_real_image_detection
"""
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

class HFDeepfakeDetector:
    def __init__(self):
        print("Loading Hugging Face deepfake model...")
        self.processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
        self.model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
        self.model.eval()
        # Determine label mapping
        self.id2label = self.model.config.id2label
        print(f"Model loaded. Labels: {self.id2label}")

    def predict(self, rgb_image):
        """
        rgb_image: numpy array (H, W, 3) in RGB order
        returns fake_probability (float between 0 and 1)
        """
        pil_img = Image.fromarray(rgb_image)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]  # (2,)
        # Check which index corresponds to "fake"
        if self.id2label[0].lower() == "fake":
            fake_prob = probs[0].item()
        elif self.id2label[1].lower() == "fake":
            fake_prob = probs[1].item()
        else:
            # fallback: assume index 1 is fake
            fake_prob = probs[1].item()
        return fake_prob

# For quick testing
if __name__ == "__main__":
    detector = HFDeepfakeDetector()
    # Test with a dummy image
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    prob = detector.predict(dummy)
    print(f"Test probability: {prob:.4f}")