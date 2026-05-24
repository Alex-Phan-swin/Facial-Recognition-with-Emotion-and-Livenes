import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Root folder = same folder where emotion_predictor.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")


def build_model(num_classes=len(EMOTIONS), dropout=0.3):
    model = models.mobilenet_v3_small(weights=None)

    model.classifier[3] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(1024, num_classes)
    )

    return model.to(DEVICE)


emotion_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


emotion_model = build_model()
emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
emotion_model.eval()

print("Emotion model loaded successfully.")


def predict_emotion(face_img):
    if face_img is None or face_img.size == 0:
        return "-", 0.0

    # OpenCV uses BGR, PyTorch model needs RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Convert NumPy image to PIL
    pil_img = Image.fromarray(face_rgb)

    # Transform image
    img_tensor = emotion_transform(pil_img)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = emotion_model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    confidence, class_idx = torch.max(probs, dim=0)

    emotion = EMOTIONS[class_idx.item()]
    confidence = confidence.item()

    return emotion, confidence