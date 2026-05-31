import cv2
import os
import time
import numpy as np
import json
import tensorflow as tf
import torch
from PIL import Image
import torch.nn as nn
from tensorflow import keras
from tensorflow.keras.models import load_model
import keras
from torchvision import models, transforms

from .config import DB_PATH, EXIT_DELAY, DISPLAY_DELAY, MAX_IMAGES, BASE_DIR, PROJECT_ROOT
from .logger import init_log, log_event
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from collections import deque
from laptop_detection import LaptopDetector
from anti_spoofing import LivenessChecker
from anti_spoofing.inference import LivenessResult

# -------------------------------
# SETUP
# -------------------------------
os.makedirs(DB_PATH, exist_ok=True)
init_log()

# -------------------------------
# REGISTER SAME CUSTOM OBJECTS
# -------------------------------
@keras.saving.register_keras_serializable(package="FaceModel")
class L2NormLayer(keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)

    def get_config(self):
        return super().get_config()

# -------------------------------
# PATHS & CONFIG
# -------------------------------
MODEL_PATH       = os.path.join(BASE_DIR, "face_model.keras")
TRIPLET_MODEL_PATH = os.path.join(PROJECT_ROOT, "face_triplet_model.keras")
CLASS_INDEX_PATH = os.path.join(BASE_DIR, "face_classes.json")
TRAIN_DIR        = os.path.join(PROJECT_ROOT, 'dataset', 'classification_data', 'train_data')

CONFIDENCE_THRESHOLD = 0.70

# -------------------------------
# LOAD CLASS NAMES
# -------------------------------
if os.path.exists(CLASS_INDEX_PATH):
    with open(CLASS_INDEX_PATH, "r") as f:
        class_indices = json.load(f)
    class_names = [None] * len(class_indices)
    for name, index in class_indices.items():
        class_names[index] = name
else:
    class_names = sorted(os.listdir(TRAIN_DIR))

# -------------------------------
# LOAD MODEL
# -------------------------------
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

embedding_model = keras.Model(
    inputs=model.input,
    outputs=model.get_layer("face_embedding").output
)

triplet_embedding_model = load_model(TRIPLET_MODEL_PATH, compile=False)
print("Triplet model loaded successfully.")

EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Build the model again (imports weights from the emotion_train file)
def build_emotion_model():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1024, len(EMOTIONS))
    )
    return model


emotion_model = build_emotion_model()
emotion_model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
emotion_model.to(DEVICE)
emotion_model.eval()

emotion_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # mean and std of the ImageNet ds that the original model was trained on
])


# Predicts the (NOW TWO) emotions from the face detected.
def predict_emotion(face_bgr):
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        tensor = emotion_transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(emotion_model(tensor), dim=1)[0]

        # top 2 emotions
        top2 = probs.topk(2)
        primary = EMOTIONS[top2.indices[0]], float(top2.values[0])
        secondary = EMOTIONS[top2.indices[1]], float(top2.values[1])

        return primary, secondary
    except Exception as e:
        print("Emotion error:", e)
        return ("unknown", 0.0), ("unknown", 0.0)


# ONLY LOAD FOLDERS
class_names = sorted(
    [name for name in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, name))]
)

print("Loaded classes:", class_names)

print("Press 'q' to quit | Press 'r' to register")

# -------------------------------
# REGISTER STATE
# -------------------------------
REGISTER_MODE = False
SAVE_COUNT = 0
new_person_name = ""

# -------------------------------
# STATE TRACKING
# -------------------------------
face_database = {}
triplet_database = {}
last_seen = {}
inside = set()
visible = set()
last_visible = {}
frame_count = 0
last_predictions = []

# -------------------------------
# HELPER FUNCTION
# -------------------------------
def get_embedding(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    emb = embedding_model.predict(face_img, verbose=0)[0]

    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb

    emb = emb / norm
    return emb

def get_triplet_embedding(face_img):
    face_img = cv2.resize(face_img, (80, 80))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    emb = triplet_embedding_model.predict(face_img, verbose=0)[0]
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

def verify_face(face_img, threshold=0.60):
    if len(triplet_database) == 0:
        return "Unknown", 0.0
    query_emb = get_triplet_embedding(face_img)
    best_name = "Unknown"
    best_score = -1.0
    for name, saved_emb in triplet_database.items():
        similarity = np.dot(query_emb, saved_emb)
        if similarity > best_score:
            best_score = similarity
            best_name = name
    if best_score < threshold:
        return "Unknown", best_score
    return best_name, best_score

def build_face_database():
    global face_database
    face_database = {}

    for person_name in os.listdir(DB_PATH):
        person_folder = os.path.join(DB_PATH, person_name)

        if not os.path.isdir(person_folder):
            continue

        embeddings = []

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            emb = get_embedding(img)
            embeddings.append(emb)

        if len(embeddings) > 0:
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            face_database[person_name] = avg_emb

    print("Loaded people from faces_db:", list(face_database.keys()))

def build_triplet_database():
    global triplet_database
    triplet_database = {}
    for person_name in os.listdir(DB_PATH):
        person_folder = os.path.join(DB_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue
        embeddings = []
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            emb = get_triplet_embedding(img)
            embeddings.append(emb)
        if len(embeddings) > 0:
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            triplet_database[person_name] = avg_emb
    print("Triplet database loaded:", list(triplet_database.keys()))

def predict_face(face_img):
    if len(face_database) == 0:
        return "Unknown", 0.0

    query_emb = get_embedding(face_img)

    best_name = "Unknown"
    best_score = -1.0

    for name, saved_emb in face_database.items():
        similarity = np.dot(query_emb, saved_emb)

        if similarity > best_score:
            best_score = similarity
            best_name = name

    if best_score < CONFIDENCE_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def run():
    global frame_count, REGISTER_MODE, SAVE_COUNT, new_person_name, last_predictions

    build_face_database()
    build_triplet_database()

    # -------------------------------
    # CAMERA
    # -------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # -------------------------------
    # FACE DETECTOR
    # -------------------------------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # ONLY LOAD FOLDERS
    class_names = sorted([
        name for name in os.listdir(DB_PATH)
        if os.path.isdir(os.path.join(DB_PATH, name))
    ])

    print("Loaded classes:", class_names)
    print("Press 'q' to quit | Press 'r' to register")

    # -------------------------------
    # LAPTOP DETECTOR
    # -------------------------------
    laptop_detector = LaptopDetector()
    laptop_result = None
    frame_count = 0

    #--------------------------------
    # ANTI-SPOOFING
    #--------------------------------

    liveness_checker = LivenessChecker()
    liveness_result = None
    liveness_scores = deque(maxlen=10)

    # -------------------------------
    # MAIN LOOP
    # -------------------------------
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Run laptop check every 15 frames
        if frame_count % 15 == 0:
            laptop_result = laptop_detector.check(frame)

        detected_people = set()

        # -------------------------------
        # FACE DETECTION
        # -------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        face_present = len(faces) > 0

        # -------------------------------
        # KEY INPUT
        # -------------------------------
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            for person in list(inside):
                log_event(person, "EXIT")
                print(person, "EXIT")
            break

        # -------------------------------
        # REGISTER MODE
        # -------------------------------
        if key == ord('r') and not REGISTER_MODE:
            new_person_name = input("Enter employee name: ")
            person_path = os.path.join(DB_PATH, new_person_name)
            os.makedirs(person_path, exist_ok=True)
            REGISTER_MODE = True
            SAVE_COUNT = 0
            print(f"Registering {new_person_name}...")

        # -------------------------------
        # SAVE FACES
        # -------------------------------
        if REGISTER_MODE and face_present:
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                face = cv2.resize(face, (224, 224))
                save_path = os.path.join(
                    DB_PATH, new_person_name, f"{new_person_name}_{SAVE_COUNT}.jpg"
                )
                cv2.imwrite(save_path, face)
                SAVE_COUNT += 1
                print(f"Saved {save_path}")
                time.sleep(0.3)
                if SAVE_COUNT >= MAX_IMAGES:
                    print(f"Finished registering {new_person_name}")
                    REGISTER_MODE = False
                    build_face_database()
                    build_triplet_database()
                    break

            if SAVE_COUNT >= MAX_IMAGES:
                REGISTER_MODE = False
                class_names = sorted([
                    name for name in os.listdir(DB_PATH)
                    if os.path.isdir(os.path.join(DB_PATH, name))
                ])
                print("Updated classes:", class_names)

        # -------------------------------
        # FACE RECOGNITION
        # -------------------------------
        for x, y, w, h in faces:
            face = frame[y:y + h, x:x + w]

            try:
                final_label, confidence = predict_face(face)
                verified, verify_score = verify_face(face)
                (primary_emo, primary_conf), (secondary_emo, secondary_conf) = predict_emotion(face)
                if final_label == verified and final_label != "Unknown":
                    final_label = final_label
                else:
                    final_label = "Unknown"
                liveness = liveness_checker.check(frame)
                liveness_scores.append(liveness.confidence)
                avg_conf = sum(liveness_scores) / len(liveness_scores)
                liveness_result = LivenessResult(
                    is_live=avg_conf >= liveness_checker.threshold,
                    confidence=avg_conf,
                )

                if final_label != "Unknown":
                    detected_people.add(final_label)
                    visible.add(final_label)
                    last_visible[final_label] = time.time()

                # -------------------------------
                # DRAW BOX
                # -------------------------------
                color = (0, 0, 255) if final_label == "Unknown" else (0, 255, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.putText(
                    frame,
                    f"{final_label} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"{primary_emo} {primary_conf:.0%} / {secondary_emo} {secondary_conf:.0%}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2,
                )
                liveness_color = (0, 255, 0) if liveness_result.is_live else (0, 0, 255)

                cv2.putText(
                    frame,
                    liveness_result.label,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    liveness_color,
                    2,
                )

            except Exception as e:
                print("Prediction error:", e)

        # -------------------------------
        # DISPLAY
        # -------------------------------
        for i, name in enumerate(visible):
            cv2.putText(frame, name, (50, 50 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # -------------------------------
        # ENTRY LOGIC
        # -------------------------------
        for person in detected_people:
            last_seen[person] = time.time()
            if person not in inside:
                inside.add(person)
                log_event(person, "ENTER")
                print(person, "ENTER")

        # -------------------------------
        # EXIT LOGIC
        # -------------------------------
        for person in list(inside):
            if person not in last_seen:
                continue
            if time.time() - last_seen[person] > EXIT_DELAY:
                inside.remove(person)
                log_event(person, "EXIT")
                print(person, "EXIT")

        # -------------------------------
        # CLEAN MEMORY
        # -------------------------------
        for person in list(last_seen.keys()):
            if time.time() - last_seen[person] > EXIT_DELAY * 2:
                del last_seen[person]

        # -------------------------------
        # DISPLAY CLEANUP
        # -------------------------------
        for person in list(visible):
            if person not in last_visible:
                continue
            if time.time() - last_visible[person] > DISPLAY_DELAY:
                visible.remove(person)
                del last_visible[person]

        # -------------------------------
        # LAPTOP DETECTION BANNER
        # -------------------------------
        if laptop_result and laptop_result.detected:
            cv2.putText(
                frame,
                f"LAPTOP DETECTED ({laptop_result.confidence:.2f})",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # -------------------------------
        # SHOW FRAME
        # -------------------------------
        cv2.imshow("Face Recognition", frame)

    # -------------------------------
    # CLEANUP
    # -------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
