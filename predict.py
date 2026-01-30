import cv2
import numpy as np
import base64
import sys
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# -----------------------
# FACE DETECTOR
# -----------------------
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# -----------------------
# ANTI-SPOOFING MODEL
# -----------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'face_antispoofing_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pt')

# Class index mapping: 0/1 (unknown training order). Default assumes index 1 = REAL.
REAL_CLASS_INDEX = int(os.environ.get('REAL_CLASS_INDEX', '1'))

_model = None
_device = 'cpu'


def _ensure_model_archive():
    """If model.pt doesn't exist but extracted torch-save files exist, repackage them."""
    if os.path.exists(MODEL_PATH):
        return

    # extracted structure from torch.save: data.pkl, version, byteorder, .data/, data/
    needed = ['data.pkl', 'version', 'byteorder']
    if not all(os.path.exists(os.path.join(MODEL_DIR, n)) for n in needed):
        return

    import zipfile

    root_prefix = 'model'  # torch.save uses file-stem as top folder
    with zipfile.ZipFile(MODEL_PATH, 'w', compression=zipfile.ZIP_STORED) as z:
        for root, _, files in os.walk(MODEL_DIR):
            for fn in files:
                full = os.path.join(root, fn)
                if os.path.abspath(full) == os.path.abspath(MODEL_PATH):
                    continue
                rel = os.path.relpath(full, MODEL_DIR).replace('\\\\', '/')
                z.write(full, f'{root_prefix}/{rel}')


def _load_model():
    global _model
    if _model is not None:
        return _model

    _ensure_model_archive()
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Anti-spoofing model not found at {MODEL_PATH}")

    # This file is a state_dict (OrderedDict). We'll load into ResNet18.
    state = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(state, strict=True)
    m.eval()
    _model = m
    return _model


def _preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    # Resize to standard ResNet size and normalize (ImageNet)
    img = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    t = torch.from_numpy(img).unsqueeze(0)
    return t


def analyze_liveness(image_data):
    """Input: base64 JPEG from browser webcam
    Output: REAL HUMAN / SPOOF with confidence + ACCESS decision
    """
    try:
        if face_cascade.empty():
            return {"error": "Haar cascade not loaded"}

        # 1) Decode image
        if not isinstance(image_data, str) or not image_data.startswith('data:image'):
            return {"error": "Invalid image payload"}

        _, encoded = image_data.split(",", 1)
        data = base64.b64decode(encoded)
        nparr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "Decode failed"}

        img_h, img_w = bgr.shape[:2]

        # 2) Face detection (fast + reliable)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        target_w = 360
        scale = target_w / float(img_w) if img_w > 0 else 1.0
        small_h = max(1, int(img_h * scale))
        small_gray = cv2.resize(gray, (target_w, small_h))

        faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30),
        )

        face_count = int(len(faces))
        if face_count == 0:
            return {
                "prediction": "ACCESS BLOCKED",
                "reason": "No Face Detected",
                "confidence": "0.00%",
                "border": "red",
                "steps": {
                    "identity": "Explained",
                    "spoof": "Skipped",
                    "anomaly": "Explained",
                },
                "ml": {
                    "label": "NO FACE",
                    "confidence": 0.0,
                },
            }

        if face_count > 1:
            return {
                "prediction": "ACCESS BLOCKED",
                "reason": "Multiple Faces Detected",
                "confidence": "99.00%",
                "border": "red",
                "steps": {
                    "identity": "Explained",
                    "spoof": "Skipped",
                    "anomaly": "Explained",
                },
                "ml": {
                    "label": "MULTI FACE",
                    "confidence": 0.99,
                },
            }

        # 3) Crop face ROI and run Anti-Spoofing ML
        fx, fy, fw, fh = faces[0]
        inv = 1.0 / scale if scale != 0 else 1.0
        x1 = int(fx * inv)
        y1 = int(fy * inv)
        x2 = int((fx + fw) * inv)
        y2 = int((fy + fh) * inv)

        # clamp
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 <= x1 or y2 <= y1:
            return {"error": "Face crop failed"}

        face_bgr = bgr[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        model = _load_model()
        x = _preprocess_rgb(face_rgb)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0)
            prob_real = float(probs[REAL_CLASS_INDEX].item())
            prob_spoof = float((1.0 - prob_real))

        is_real = prob_real >= 0.9

        if is_real:
            ml_label = 'REAL HUMAN'
            ml_conf = prob_real
            prediction = 'ACCESS GRANTED'
            border = 'green'
            spoof_step = '✔'
            reason = 'REAL HUMAN (Anti-Spoofing ML)'
        else:
            ml_label = 'SPOOF'
            ml_conf = prob_spoof
            prediction = 'ACCESS BLOCKED'
            border = 'red'
            spoof_step = '❌'
            reason = 'SPOOF DETECTED (Anti-Spoofing ML)'

        return {
            "prediction": prediction,
            "reason": reason,
            "confidence": f"{ml_conf * 100.0:.2f}%",
            "border": border,
            "steps": {
                "identity": "Explained",
                "spoof": spoof_step,
                "anomaly": "Explained",
            },
            "ml": {
                "label": ml_label,
                "confidence": ml_conf,
                "prob_real": prob_real,
                "prob_spoof": prob_spoof,
                "real_class_index": REAL_CLASS_INDEX,
            },
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        
        data = line.strip()
        if data:
            try:
                result = analyze_liveness(data)
                print(json.dumps(result))
                sys.stdout.flush()
            except Exception as e:
                print(json.dumps({"error": str(e)}))
                sys.stdout.flush()
